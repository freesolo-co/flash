"""Use *measured* cost from real training runs as ground truth.

The honest test of an estimator is a real GPU run, not another equation. This module
maps a control-plane run-status payload (`slm status <id>`) to the `RunConfig` that run
priced and to its measured cost/wall-time. It is the status->measured-cost bridge the
break-even calibration is derived and refreshed from: launch real runs (RunPod/Vast),
parse their status here, and compare measured dollars against the analytical quote.

Pure parsing -- no network. A collector fetches the status dicts (via the `slm` CLI)
and feeds them here, so this stays unit-testable offline.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flash.engine.recipe import RECIPE

from .types import CostEstimate, RunConfig

# A function returning the ground-truth cost for a run (``measured_ground_truth_fn``
# builds one). Inlined here so the calculation/calibration path carries no dependency on
# the (pruned) LLM prompt-convergence experiment that previously defined it.
GroundTruthFn = Callable[[RunConfig], CostEstimate]


def _steps_from_train(algorithm: str, train: dict[str, Any]) -> int:
    """Effective optimizer steps for a run spec.

    GRPO carries ``steps`` directly (and runs them all -- ``max_steps`` is an SFT-only
    optimizer-step cap and never applies to GRPO). SFT runs by epochs over a (capped)
    dataset, so steps = ceil(max_examples / effective_batch) x epochs when those are
    known. For the SFT paths ``max_steps`` bounds the reconstructed count, so an estimate
    can't exceed what actually ran (which would inflate analytical cost).
    """
    steps = train.get("steps")
    if steps:
        return int(steps)  # GRPO carries its step count directly; max_steps is SFT-only

    # A GRPO run that omits `steps` runs RECIPE.rl.num_steps (the worker falls back to it
    # when RL_STEPS is unset) -- not the SFT 100/epoch default below, which would price a
    # default-GRPO run as a shorter run than actually executed.
    if algorithm == "grpo":
        return int(RECIPE.rl.num_steps)

    # max_steps is the worker's SFT optimizer-step cap -- only build the cap for SFT.
    max_steps = train.get("max_steps")

    def _cap(n: int) -> int:
        return min(n, int(max_steps)) if max_steps else n

    epochs = int(train.get("epochs") or 1)
    # An omitted SFT batch_size reconstructs against the recipe EFFECTIVE batch (the worker
    # sizes grad-accum to hit RECIPE.sft.effective_batch), not a bare per-device batch.
    batch = int(train.get("batch_size") or RECIPE.sft.effective_batch)
    max_examples = train.get("max_examples")
    if max_examples:
        return _cap(max(1, math.ceil(int(max_examples) / max(1, batch)) * epochs))
    return _cap(100)  # last-resort default when the dataset size isn't pinned in the spec


def runconfig_from_status(
    status: dict[str, Any],
    *,
    label: str | None = None,
    reward_seconds_per_completion: float | None = None,
) -> RunConfig:
    """Reconstruct the `RunConfig` a real run priced, from its status payload."""
    spec = status["spec"]
    train = spec.get("train", {}) or {}
    method = spec["algorithm"]
    is_grpo = method == "grpo"
    # A multi-seed run executes the full step count once per seed and accumulates
    # `cost_usd` across all of them (runner.py), so the measured bill covers N seeds.
    # Each seed is its own job that reprovisions and re-pays the cold start, so scale
    # BOTH the reconstructed step count AND the setup repeats by the seed count -- the
    # analytical estimate then prices the SAME total work (N x train + N x setup) the
    # measured dollars paid for.
    n_seeds = max(1, len(train.get("seeds") or (0,)))
    gpu_spec = spec.get("gpu") or {}
    # The spec's per-run wall cap (``gpu.max_wall_seconds``) and unvalidated-GPU policy
    # (``gpu.allow_unvalidated``) drive what the runner actually billed: the cap clamps each
    # seed's wall clock and the policy widens the allocator's GPU pool. Carry both so the
    # reconstructed estimate prices the SAME run the measured dollars covered (otherwise a
    # run with a short explicit cap is priced against the 24h default and overstated).
    max_wall = gpu_spec.get("max_wall_seconds")
    # ``gpu.allow_unvalidated`` is TRI-STATE in the spec/allocator: a MISSING key is not
    # "validated-only" -- it's "unspecified", which submit-time resolves via the managed
    # default (providers.base.unvalidated_allowed). Coercing a missing value to ``False`` here
    # would diverge from how the allocator resolves an absent flag. So keep ``None`` when absent
    # and only coerce to bool when a value is actually present; selection resolves ``None`` the
    # same way the runner does.
    allow_unvalidated = gpu_spec.get("allow_unvalidated")
    if allow_unvalidated is not None:
        allow_unvalidated = bool(allow_unvalidated)
    # Pin the card the run ACTUALLY ran on. A policy GPU (``gpu.type`` = "auto"/"cheapest")
    # is re-allocated to a concrete class at submit time, recorded in ``remote.allocated_gpu``
    # (with ``remote.provider``); falling back to ``gpu.type`` would leave the reconstructed
    # config pinned to a policy sentinel and re-derive the GPU from the offline heuristic at
    # grading time -- pricing the measured bill against a DIFFERENT card. Mirror
    # ``measured_from_status``/``calibration._config_of``: prefer the allocated GPU/provider
    # and only fall back to the requested type/default when the run never recorded one.
    remote = status.get("remote") or {}
    allocated_gpu = remote.get("allocated_gpu") or gpu_spec.get("type")
    allocated_provider = remote.get("provider")
    return RunConfig(
        model_id=spec["model"],
        method=method,
        steps=_steps_from_train(method, train) * n_seeds,
        setup_repeats=n_seeds,
        seq_len=train.get("max_length"),
        completion_len=train.get("max_tokens") if is_grpo else None,
        batch_size=train.get("batch_size"),
        group_size=train.get("group_size") if is_grpo else None,
        lora_rank=train.get("lora_rank"),
        thinking=bool(spec.get("thinking", False)),
        gpu=allocated_gpu,
        provider=allocated_provider if allocated_provider is not None else "auto",
        allow_unvalidated=allow_unvalidated,
        max_wall_seconds=int(max_wall) if max_wall is not None else None,
        environment=(spec.get("environment") or {}).get("id"),
        reward_seconds_per_completion=reward_seconds_per_completion,
        label=label or status.get("run_id"),
    )


@dataclass(frozen=True)
class MeasuredRun:
    run_id: str
    state: str
    config: RunConfig
    cost_usd: float
    wall_seconds: float
    gpu: str
    provider: str
    hourly_usd: float | None

    @property
    def ok(self) -> bool:
        return self.state == "done" and self.cost_usd > 0

    @property
    def effective_hourly_usd(self) -> float:
        """The billed rate, falling back to cost/wall when the provider omits ``hourly_usd``.

        Some providers don't report ``remote.hourly_usd``, but the measured ``cost_usd`` and
        ``wall_seconds`` are the ground truth -- so the implied rate (cost / hours) is a
        meaningful per-run rate for reports/notes instead of a misleading ``$0.00/hr``.
        """
        if self.hourly_usd is not None:
            return self.hourly_usd
        if self.wall_seconds > 0:
            return self.cost_usd / (self.wall_seconds / 3600.0)
        return 0.0


def measured_from_status(
    status: dict[str, Any],
    *,
    label: str | None = None,
    reward_seconds_per_completion: float | None = None,
) -> MeasuredRun:
    remote = status.get("remote") or {}
    started = remote.get("started_ts")
    if started is None:
        started = status.get("created_at", 0.0)
    wall = max(0.0, float(status.get("updated_at", 0.0)) - float(started))
    return MeasuredRun(
        run_id=status.get("run_id", "?"),
        state=status.get("state", "?"),
        config=runconfig_from_status(
            status, label=label, reward_seconds_per_completion=reward_seconds_per_completion
        ),
        cost_usd=float(status.get("cost_usd") or 0.0),
        wall_seconds=wall,
        gpu=remote.get("allocated_gpu") or (status["spec"].get("gpu") or {}).get("type", "?"),
        provider=remote.get("provider", "?"),
        hourly_usd=remote.get("hourly_usd"),
    )


def measured_as_estimate(m: MeasuredRun) -> CostEstimate:
    """Wrap a measured run as a `CostEstimate` (total_usd + gpu are the ground truth)."""
    rate = m.effective_hourly_usd  # billed rate, or cost/wall when the provider omits it
    return CostEstimate(
        model_id=m.config.model_id,
        method=m.config.method,
        steps=m.config.steps,
        gpu=m.gpu,
        provider=m.provider,
        gpu_vram_gb=0,
        required_vram_gb=0,
        gpu_hourly_usd=rate,
        setup_seconds=0.0,
        seconds_per_step=(m.wall_seconds / m.config.steps) if m.config.steps else 0.0,
        train_seconds=m.wall_seconds,
        wall_clock_seconds=m.wall_seconds,
        wall_capped=False,
        total_usd=m.cost_usd,
        notes=(
            f"MEASURED on {m.gpu}@{m.provider} (${rate:.2f}/hr, {m.wall_seconds:.0f}s)",
        ),
    )


def measured_ground_truth_fn(runs: list[MeasuredRun]) -> GroundTruthFn:
    """A ``GroundTruthFn`` that returns the measured cost for a given config.

    Keyed by ``config.display()``. Two runs that share a label can't both be the ground
    truth for the same grid cell, so reject duplicates loudly instead of silently keeping
    only the last and grading every earlier config against the wrong measured dollars.
    """
    by_label: dict[str, CostEstimate] = {}
    for r in runs:
        key = r.config.display()
        if key in by_label:
            raise ValueError(
                f"duplicate measured-run label {key!r}: every config must map to a unique "
                "measured cost (give the runs distinct labels or key by run_id)"
            )
        by_label[key] = measured_as_estimate(r)

    def fn(config: RunConfig) -> CostEstimate:
        return by_label[config.display()]

    return fn

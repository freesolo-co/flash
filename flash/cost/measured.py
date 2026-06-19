"""Use *measured* cost from real training runs as ground truth.

The honest test of an estimator is a real GPU run, not another equation. This module
maps a control-plane run-status payload (`slm status <id>`) to the `RunConfig` that run
priced and to its measured cost/wall-time, and exposes a `ground_truth_fn` so the
prompt-convergence harness can grade the LLM against *measured* dollars.

Pure parsing -- no network. The analysis script fetches the status dicts (via the
`slm` CLI) and feeds them here, so this stays unit-testable offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from flash.engine.recipe import RECIPE

from .config import RunConfig
from .estimate import CostEstimate
from .experiment import GroundTruthFn


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
    # analytical estimate and the LLM prompt then price the SAME total work (N x train
    # + N x setup) the measured dollars paid for.
    n_seeds = max(1, len(train.get("seeds") or (0,)))
    gpu_spec = spec.get("gpu") or {}
    # The spec's per-run wall cap (``gpu.max_wall_seconds``) and unvalidated-GPU policy
    # (``gpu.allow_unvalidated``) drive what the runner actually billed: the cap clamps each
    # seed's wall clock and the policy widens the allocator's GPU pool. Carry both so the
    # reconstructed estimate prices the SAME run the measured dollars covered (otherwise a
    # run with a short explicit cap is priced against the 24h default and overstated).
    max_wall = gpu_spec.get("max_wall_seconds")
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
        gpu=gpu_spec.get("type"),
        allow_unvalidated=bool(gpu_spec.get("allow_unvalidated", False)),
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


def measured_from_status(
    status: dict[str, Any],
    *,
    label: str | None = None,
    reward_seconds_per_completion: float | None = None,
) -> MeasuredRun:
    remote = status.get("remote") or {}
    started = remote.get("started_ts") or status.get("created_at", 0.0)
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
    return CostEstimate(
        model_id=m.config.model_id,
        method=m.config.method,
        steps=m.config.steps,
        gpu=m.gpu,
        provider=m.provider,
        gpu_vram_gb=0,
        required_vram_gb=0,
        gpu_hourly_usd=m.hourly_usd or 0.0,
        setup_seconds=0.0,
        seconds_per_step=(m.wall_seconds / m.config.steps) if m.config.steps else 0.0,
        train_seconds=m.wall_seconds,
        wall_clock_seconds=m.wall_seconds,
        wall_capped=False,
        total_usd=m.cost_usd,
        notes=(
            f"MEASURED on {m.gpu}@{m.provider} (${m.hourly_usd or 0:.2f}/hr, {m.wall_seconds:.0f}s)",
        ),
    )


def measured_ground_truth_fn(runs: list[MeasuredRun]) -> GroundTruthFn:
    """A `ground_truth_fn` for `run_experiment` that returns measured cost per config.

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

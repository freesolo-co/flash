"""Map a parsed training ``JobSpec`` to a cost ``RunConfig`` / ``CostEstimate``.

Shared by the CLI (``slm train --estimate``) and the control plane (which charges the
estimate to the user's org at submit time), so both price the SAME work the run will bill
for. Kept network-free: ``estimate_cost`` sizes VRAM offline, and the spec passed here is
already parsed, so nothing in this module touches the Hugging Face API.
"""

from __future__ import annotations

from flash.cost.analytical import estimate_cost
from flash.cost.types import CostEstimate, RunConfig

# Assumed SFT dataset size (examples) when the config pins no ``train.max_examples``: the
# worker trains the FULL environment dataset, whose size isn't readable here (the estimate is
# fully local — no Hub/env load). This stand-in makes the omitted-cap estimate a documented
# FLOOR rather than a magic optimizer-step constant; the CLI surfaces the floor caveat. A real
# env's train split is typically far larger, so a pinned ``max_examples`` always estimates tighter.
SFT_ASSUMED_EXAMPLES_WHEN_UNPINNED = 1000


def sft_realized_batch(batch_size: int) -> int:
    """The SFT global batch the WORKER realizes for a requested ``batch_size``.

    The worker (engine.worker) never trains at the raw requested batch: it fixes the per-device
    micro-batch at ``_sft_per_device_bs()`` (4) and reaches the target via CEIL grad-accum, so the
    realized global batch is ``per_device x ceil(target / per_device)`` -- which can EXCEED the
    request when it isn't a multiple of the micro-batch (e.g. 16/6 -> per_device 4, accum 2 -> 8).
    Mirror that here so the optimizer-step count matches what actually runs.
    """
    from flash.engine.vram import _sft_per_device_bs

    target = max(1, int(batch_size))
    per_device = max(1, min(_sft_per_device_bs(), target))
    grad_accum = max(1, -(-target // per_device))  # ceil
    return per_device * grad_accum


def spec_steps(spec) -> int:
    """Per-seed optimizer steps implied by a train spec (mirrors the worker).

    GRPO carries ``train.steps`` (default recipe ``num_steps``) -- ``train.steps`` is a GRPO
    concept and is NOT consulted for SFT. SFT runs by epochs over a (capped) dataset, so steps =
    ``epochs x ceil(num_examples / realized_batch)``, capped by ``max_steps``, where ``epochs``
    defaults to ``RECIPE.sft.num_epochs`` (2), ``realized_batch`` is the worker's grad-accum global
    batch (``sft_realized_batch``), and ``num_examples`` is ``max_examples`` if pinned else the
    documented unpinned floor (``SFT_ASSUMED_EXAMPLES_WHEN_UNPINNED``).
    """
    from flash.engine.recipe import RECIPE

    t = spec.train
    if spec.algorithm == "grpo":
        if t.steps is not None:
            return max(1, int(t.steps))
        return RECIPE.rl.num_steps
    # --- SFT ---
    cap = int(t.max_steps) if t.max_steps else 0  # SFT-only optimizer-step cap (0 = uncapped)
    epochs = int(t.epochs) if t.epochs is not None else RECIPE.sft.num_epochs
    requested_batch = int(t.batch_size) if t.batch_size is not None else RECIPE.sft.effective_batch
    batch = sft_realized_batch(requested_batch)  # worker's grad-accum-realized global batch
    examples = (
        int(t.max_examples) if t.max_examples is not None else SFT_ASSUMED_EXAMPLES_WHEN_UNPINNED
    )
    n = max(1, -(-examples // batch) * epochs)  # epochs x ceil(examples / realized_batch)
    return min(n, cap) if cap > 0 else n


def runconfig_from_spec(spec) -> RunConfig:
    """Map a parsed training ``JobSpec`` to a cost ``RunConfig``.

    Each seed is its own job that re-pays the cold start (runner.py), so scale both the step
    count and the setup repeats by the seed count -- the estimate then prices the same total
    work the run would bill for.
    """
    t, g = spec.train, spec.gpu
    is_grpo = spec.algorithm == "grpo"
    seeds = max(1, len(t.seeds or (0,)))
    return RunConfig(
        model_id=spec.model,
        method=spec.algorithm,
        steps=spec_steps(spec) * seeds,
        setup_repeats=seeds,
        seq_len=t.max_length,
        completion_len=t.max_tokens if is_grpo else None,
        batch_size=t.batch_size,
        group_size=t.group_size if is_grpo else None,
        lora_rank=t.lora_rank,
        thinking=spec.thinking,
        gpu=g.type,
        provider=g.provider,
        allow_unvalidated=g.allow_unvalidated,
        max_wall_seconds=g.max_wall_seconds,
        environment=spec.environment.id or None,
    )


def estimate_for_spec(spec) -> CostEstimate:
    """The pre-flight ``CostEstimate`` for a parsed training ``JobSpec``."""
    return estimate_cost(runconfig_from_spec(spec))

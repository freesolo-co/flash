"""Map a parsed training ``JobSpec`` to a cost ``RunConfig`` / step count / estimate.

Shared by ``flash train --cost`` and the control plane's submit-time charge, so both price the
same work on the same catalog-only, cheapest-fit basis."""

from __future__ import annotations

from flash.cost.analytical import estimate_cost
from flash.cost.types import CostEstimate, RunConfig


def sft_realized_batch(batch_size: int) -> int:
    """The SFT global batch the worker realizes for ``batch_size``: per-device micro-batch (4) x
    ceil grad-accum -- which can EXCEED the request when not a multiple of 4. Mirror it so the
    step count matches the run."""
    from flash.engine.vram import _sft_per_device_bs

    target = max(1, int(batch_size))
    per_device = max(1, min(_sft_per_device_bs(), target))
    grad_accum = max(1, -(-target // per_device))  # ceil
    return per_device * grad_accum


def count_env_examples(env_id: str, params: dict | None = None) -> int | None:
    """Training rows in ``env_id``'s dataset (the worker's train split), or ``None`` if it can't
    be loaded. Best-effort -- prices an uncapped SFT run on the real dataset size, not a guess."""
    if not env_id:
        return None
    try:
        from flash.envs import load_environment

        rows = load_environment(env_id, params or {}).dataset("train")
    except Exception:
        return None
    return len(rows) if rows is not None else None


def spec_steps(spec) -> int:
    """Per-seed optimizer steps implied by a train spec (mirrors the worker). GRPO: ``train.steps``
    (else recipe default). SFT: ``epochs x ceil(num_examples / realized_batch)`` capped by
    ``max_steps``, where ``num_examples`` is ``max_examples`` if pinned else the real env size."""
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
    batch = sft_realized_batch(requested_batch)
    # max_examples is a CAP; 0 (like None) means "no cap" (worker trains the full dataset), so
    # don't let max_examples=0 price a single step.
    pinned_examples = int(t.max_examples) if t.max_examples else 0
    if pinned_examples > 0:
        examples = pinned_examples
    else:
        # No cap: the worker trains the FULL env dataset, so price its real size.
        examples = count_env_examples(spec.environment.id, spec.environment.params)
        if examples is None:
            raise ValueError(
                f"could not load environment {spec.environment.id!r} to count its training "
                f"examples for the cost; install it (`slm env install {spec.environment.id}`) "
                "or pin [train].max_examples"
            )
    n = max(1, -(-examples // batch) * epochs)  # epochs x ceil(examples / realized_batch)
    return min(n, cap) if cap > 0 else n


def runconfig_from_spec(spec) -> RunConfig:
    """Map a parsed ``JobSpec`` to a cost ``RunConfig``. Each seed is its own job that re-pays the
    cold start, so steps and setup repeats scale by the seed count. The estimate doesn't pin a
    GPU -- it does its own cheapest-fit (provider="auto")."""
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
        provider="auto",
        max_wall_seconds=g.max_wall_seconds,
        environment=spec.environment.id or None,
    )


def estimate_for_spec(spec) -> CostEstimate:
    """The pre-flight ``CostEstimate`` for a parsed training ``JobSpec``."""
    return estimate_cost(runconfig_from_spec(spec))

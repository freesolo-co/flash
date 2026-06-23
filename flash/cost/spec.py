"""Map a parsed training ``JobSpec`` to a cost ``RunConfig`` / step count / estimate.

Shared by ``flash train --cost`` and the control plane's submit-time charge, so both price the
same work on the same catalog-only, cheapest-fit basis."""

from __future__ import annotations

from flash.cost.analytical import estimate_cost
from flash.cost.types import CostEstimate, RunConfig

# Fallback SFT dataset size when an uncapped run's env can't be counted locally. Most Freesolo
# training datasets land in the
# low-thousands of rows; this is a representative middle estimate so the quote is in the right
# ballpark rather than hard-failing.
DEFAULT_UNCOUNTED_SFT_EXAMPLES = 1000


def count_env_examples(env_id: str, params: dict | None = None) -> int | None:
    """Training rows in ``env_id``'s dataset (the worker's train split), or ``None`` if it can't
    be loaded. Best-effort -- prices an uncapped SFT run on the real dataset size, not a guess.

    Loading may need network access for managed Freesolo environments. If the environment
    cannot be loaded in this interpreter, this returns ``None`` and the caller falls back to a
    default count instead of hard-failing."""
    if not env_id:
        return None
    try:
        from flash.envs import load_environment

        rows = load_environment(env_id, params or {}).dataset()
    except Exception:
        return None
    return len(rows) if rows is not None else None


def spec_steps(spec) -> int:
    """Per-seed optimizer steps implied by a train spec (mirrors the worker). GRPO: ``train.steps``
    (else recipe default). SFT: ``epochs x ceil(num_examples / realized_batch)`` capped by
    ``max_steps``, where ``num_examples`` is ``max_examples`` if pinned else the real env size."""
    from flash.catalog import vocab_size_for
    from flash.engine.recipe import RECIPE
    from flash.engine.vram import resolve_params_b, sft_logits_fused, sft_realized_batch

    t = spec.train
    if spec.algorithm == "grpo":
        if t.steps is not None:
            return max(1, int(t.steps))
        return RECIPE.rl.num_steps
    # --- SFT ---
    cap = int(t.max_steps) if t.max_steps else 0  # SFT-only optimizer-step cap (0 = uncapped)
    epochs = int(t.epochs) if t.epochs is not None else RECIPE.sft.num_epochs
    requested_batch = int(t.batch_size) if t.batch_size is not None else RECIPE.sft.effective_batch
    # Mirror the worker's per-device micro-batch EXACTLY, incl. the big-vocab logits cap: when the
    # fused CE is OFF the worker vocab-sizes the micro-batch (engine.worker), which (with CEIL'd
    # grad-accum) can change the realized global batch and thus the step count. Feed the same
    # seq/vocab/fused so the priced step count matches what actually runs.
    sft_seq = (
        int(t.max_length)
        if t.max_length is not None
        else (RECIPE.sft.max_seq_len_thinking if spec.thinking else RECIPE.sft.max_seq_len)
    )
    # Resolve params_b via the shared helper (catalog stat else HF safetensors for an open model) —
    # the SAME resolution the worker's run_sft uses. The fused-CE decision (and thus the big-vocab
    # micro-batch cap) hinges on the >=3B threshold, so an uncataloged >=3B model must not be priced
    # as <3B (which would flip fused off, change the realized batch via the cap, and misprice the
    # step count). Best-effort: no network -> None -> the prior <3B (cap-on) behavior.
    sft_fused = sft_logits_fused(resolve_params_b(spec.model), sft_seq)
    batch = sft_realized_batch(
        requested_batch, seq_len=sft_seq, vocab=vocab_size_for(spec.model), fused=sft_fused
    )
    # max_examples is a CAP; 0 (like None) means "no cap" (worker trains the full dataset), so
    # don't let max_examples=0 price a single step.
    pinned_examples = int(t.max_examples) if t.max_examples else 0
    if pinned_examples > 0:
        examples = pinned_examples
    else:
        # No cap: the worker trains the FULL env dataset, so price its real size when we can
        # count it. A managed Freesolo environment may not be reachable in this interpreter, so
        # counting can return None. Fall back to a representative default instead of hard-failing.
        examples = count_env_examples(spec.environment.id, spec.environment.params)
        if examples is None:
            examples = DEFAULT_UNCOUNTED_SFT_EXAMPLES
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

"""Map a parsed training ``JobSpec`` to a cost ``RunConfig`` / step count / estimate."""

from __future__ import annotations

from flash.cost.analytical import estimate_cost
from flash.cost.types import CostEstimate, RunConfig

# Fallback when the env dataset can't be counted locally; keeps the quote in the right ballpark.
DEFAULT_UNCOUNTED_SFT_EXAMPLES = 1000


def count_env_examples(env_id: str, params: dict | None = None) -> int | None:
    """Training rows in ``env_id``'s dataset, or ``None`` if it can't be loaded."""
    if not env_id:
        return None
    try:
        from flash.envs import load_environment

        rows = load_environment(env_id, params or {}).dataset()
    except Exception:
        return None
    return len(rows) if rows is not None else None


def spec_steps(spec) -> int:
    """Per-seed optimizer steps implied by a train spec (mirrors the worker)."""
    from flash.catalog import vocab_size_for
    from flash.engine.recipe import RECIPE
    from flash.engine.vram import resolve_params_b, sft_logits_fused, sft_realized_batch

    t = spec.train
    if spec.algorithm == "grpo":
        if t.steps is not None:
            return max(1, int(t.steps))
        return RECIPE.rl.num_steps
    cap = int(t.max_steps) if t.max_steps else 0  # 0 = uncapped
    epochs = int(t.epochs) if t.epochs is not None else RECIPE.sft.num_epochs
    requested_batch = int(t.batch_size) if t.batch_size is not None else RECIPE.sft.effective_batch
    # Mirror the worker's per-device micro-batch exactly so the priced step count matches.
    sft_seq = (
        int(t.max_length)
        if t.max_length is not None
        else (RECIPE.sft.max_seq_len_thinking if spec.thinking else RECIPE.sft.max_seq_len)
    )
    sft_fused = sft_logits_fused(resolve_params_b(spec.model), sft_seq)
    batch = sft_realized_batch(
        requested_batch, seq_len=sft_seq, vocab=vocab_size_for(spec.model), fused=sft_fused
    )
    # max_examples=0 means "no cap" — don't treat it as zero examples.
    pinned_examples = int(t.max_examples) if t.max_examples else 0
    if pinned_examples > 0:
        examples = pinned_examples
    else:
        examples = count_env_examples(spec.environment.id, spec.environment.params)
        if examples is None:
            examples = DEFAULT_UNCOUNTED_SFT_EXAMPLES
    n = max(1, -(-examples // batch) * epochs)  # epochs x ceil(examples / realized_batch)
    return min(n, cap) if cap > 0 else n


def runconfig_from_spec(spec) -> RunConfig:
    """Map a parsed ``JobSpec`` to a cost ``RunConfig``. A run trains exactly one adapter, so the
    estimate covers a single job. The estimate doesn't pin a GPU -- it does its own cheapest-fit
    (provider="auto")."""
    t, g = spec.train, spec.gpu
    is_grpo = spec.algorithm == "grpo"
    return RunConfig(
        model_id=spec.model,
        method=spec.algorithm,
        steps=spec_steps(spec),
        setup_repeats=1,
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

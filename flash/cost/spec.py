"""Map a parsed training ``JobSpec`` to a cost ``RunConfig`` / step count / estimate.

Used by ``flash train --cost`` and submit-time quote snapshots. The control plane bills completed
runs from their persisted quote (``RunStatus.cost_usd``), not measured provider wall cost."""

from __future__ import annotations

import math

from flash.cost.analytical import estimate_cost
from flash.cost.types import CostEstimate, RunConfig


def _sft_epochs(spec) -> int:
    from flash.engine.recipe import RECIPE

    t = spec.train
    return int(t.epochs) if t.epochs is not None else RECIPE.sft.num_epochs


def _sft_seq_len(spec) -> int:
    from flash.engine.recipe import RECIPE

    t = spec.train
    return (
        int(t.max_length)
        if t.max_length is not None
        else (RECIPE.sft.max_seq_len_thinking if spec.thinking else RECIPE.sft.max_seq_len)
    )


def _sft_example_count(spec) -> int:
    t = spec.train
    pinned_examples = int(t.max_examples) if t.max_examples else 0
    if pinned_examples > 0:
        return pinned_examples
    raise ValueError(
        "cannot estimate SFT cost without [train].max_examples; set it to the number "
        "of rows to price (use the full dataset row count for an uncapped run)"
    )


def _sft_realized_batch(spec) -> int:
    from flash.catalog import vocab_size_for
    from flash.engine.recipe import RECIPE
    from flash.engine.vram import resolve_params_b, sft_logits_fused, sft_realized_batch

    t = spec.train
    requested_batch = int(t.batch_size) if t.batch_size is not None else RECIPE.sft.effective_batch
    sft_seq = _sft_seq_len(spec)
    # Resolve params_b via the shared helper (catalog stat else HF safetensors for an open model) —
    # the SAME resolution the worker's run_sft uses. The fused-CE decision (and thus the big-vocab
    # micro-batch cap) hinges on the >=3B threshold, so an uncataloged >=3B model must not be priced
    # as <3B (which would flip fused off, change the realized batch via the cap, and misprice the
    # step count). Best-effort: no network -> None -> the prior <3B (cap-on) behavior.
    sft_fused = sft_logits_fused(resolve_params_b(spec.model), sft_seq)
    return sft_realized_batch(
        requested_batch, seq_len=sft_seq, vocab=vocab_size_for(spec.model), fused=sft_fused
    )


def _sft_steps_from_examples(spec, examples: int, *, apply_cap: bool) -> int:
    t = spec.train
    cap = int(t.max_steps) if t.max_steps else 0  # SFT-only optimizer-step cap (0 = uncapped)
    n = max(1, math.ceil(examples / _sft_realized_batch(spec)) * _sft_epochs(spec))
    return min(n, cap) if apply_cap and cap > 0 else n


def spec_steps(spec) -> int:
    """Per-seed optimizer steps implied by a train spec (mirrors the worker). GRPO: ``train.steps``
    (else recipe default). SFT: ``epochs x ceil(num_examples / realized_batch)`` capped by
    ``max_steps``, where ``num_examples`` must be pinned by ``max_examples``."""
    from flash.engine.recipe import RECIPE

    t = spec.train
    if spec.algorithm == "grpo":
        if t.steps is not None:
            return max(1, int(t.steps))
        return RECIPE.rl.num_steps
    # max_examples is a CAP; 0 (like None) means "no cap" (worker trains the full dataset), so
    # don't let max_examples=0 price a single step.
    return _sft_steps_from_examples(spec, _sft_example_count(spec), apply_cap=True)


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

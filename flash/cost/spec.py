"""Map a parsed training ``JobSpec`` to a cost ``RunConfig`` / step count / estimate.

Used by ``flash train --cost`` and submit-time quote snapshots. The control plane bills completed
runs from their persisted quote (``RunStatus.cost_usd``), not measured provider wall cost."""

from __future__ import annotations

from flash.catalog import samples_on_policy
from flash.cost.analytical import estimate_cost
from flash.cost.types import CostEstimate, RunConfig
from flash.engine.steps import on_policy_steps, resolve_update_horizon, sft_update_steps


def _sft_epochs(spec) -> int:
    from flash.engine.recipe import RECIPE

    t = spec.train
    return int(t.epochs) if t.epochs is not None else RECIPE.sft.num_epochs


def _on_policy_epochs(spec) -> int:
    from flash.engine.recipe import RECIPE

    t = spec.train
    default = RECIPE.rl.num_epochs if spec.algorithm == "grpo" else RECIPE.opd.num_epochs
    return int(t.epochs) if t.epochs is not None else default


def _sft_seq_len(spec) -> int:
    from flash.engine.recipe import RECIPE

    t = spec.train
    return (
        int(t.max_context_tokens)
        if t.max_context_tokens is not None
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


def _on_policy_example_count(spec) -> int:
    t = spec.train
    pinned_examples = int(t.max_examples) if t.max_examples else 0
    if pinned_examples > 0:
        return pinned_examples
    env_examples = _env_max_examples(spec)
    if env_examples > 0:
        return env_examples
    return _on_policy_requested_prompts_per_step(spec)


def _env_max_examples(spec) -> int:
    params = getattr(getattr(spec, "environment", None), "params", {}) or {}
    if not isinstance(params, dict):
        return 0
    try:
        value = int(params.get("max_examples") or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, value)


def _on_policy_requested_prompts_per_step(spec) -> int:
    from flash.engine.recipe import RECIPE

    t = spec.train
    default = (
        RECIPE.rl.prompts_per_step if spec.algorithm == "grpo" else RECIPE.opd.prompts_per_step
    )
    return max(1, int(t.batch_size) if t.batch_size is not None else default)


def _on_policy_prompts_per_step(spec, examples: int) -> int:
    requested = _on_policy_requested_prompts_per_step(spec)
    return min(requested, max(1, int(examples)))


def _sft_realized_batch(spec) -> int:
    from flash.catalog import resolve_vocab_size
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
    sft_fused = sft_logits_fused(
        resolve_params_b(spec.model, revision=spec.model_revision), sft_seq
    )
    return sft_realized_batch(
        requested_batch,
        seq_len=sft_seq,
        vocab=resolve_vocab_size(spec.model, spec.model_revision),
        fused=sft_fused,
    )


def spec_steps(spec) -> int:
    """Per-seed optimizer steps implied by a train spec (mirrors the worker).

    SFT derives from examples and realized batch. GRPO/OPD derive passes over retained prompts.
    For every algorithm, positive ``max_steps`` replaces the derived optimizer-update count.
    """
    if spec.algorithm in ("grpo", "opd"):
        examples = _on_policy_example_count(spec)
        derived = on_policy_steps(
            epochs=_on_policy_epochs(spec),
            prompt_count=examples,
            prompts_per_step=_on_policy_prompts_per_step(spec, examples),
        )
        return resolve_update_horizon(derived, spec.train.max_steps)
    # max_examples is a CAP; 0 (like None) means "no cap" (worker trains the full dataset), so
    # don't let max_examples=0 price a single step.
    examples = _sft_example_count(spec)
    derived = sft_update_steps(
        epochs=_sft_epochs(spec),
        example_count=examples,
        examples_per_update=_sft_realized_batch(spec),
    )
    return resolve_update_horizon(derived, spec.train.max_steps)


def runconfig_from_spec(spec) -> RunConfig:
    """Map a parsed ``JobSpec`` to a cost ``RunConfig`` for one adapter-training job.

    unconstrained runs retain cheapest-fit pricing; authored provider/exact-type constraints are
    preserved so the quote matches the allocatable hardware contract.
    """
    t, g = spec.train, spec.gpu
    # Both grpo and opd sample on-policy student completions, so both carry the rollout
    # dimensions (completion length + group size) into the cost model.
    has_rollout = samples_on_policy(spec.algorithm)
    # Price the actual OPD teacher the run will use. [train].teacher_model is already the resolved
    # Fireworks model id (parse canonicalizes it), so a higher/lower-priced teacher moves the itemized
    # teacher-API estimate; "" => the default GLM 5.2, and "" for sft/grpo (no teacher).
    teacher_model = ""
    if spec.algorithm == "opd":
        from flash.engine.recipe import RECIPE

        teacher_model = t.teacher_model or RECIPE.opd.teacher_model
    return RunConfig(
        model_id=spec.model,
        method=spec.algorithm,
        steps=spec_steps(spec),
        seq_len=t.max_context_tokens,
        completion_len=t.max_completion_tokens if has_rollout else None,
        batch_size=t.batch_size,
        group_size=t.group_size if has_rollout else None,
        lora_rank=t.lora_rank,
        thinking=spec.thinking,
        teacher_model=teacher_model,
        provider=g.provider or "auto",
        exact_type=g.exact_type,
        model_revision=spec.model_revision,
        disk_gb=float(getattr(g, "disk_gb", 0.0) or 0.0),
        max_wall_seconds=g.max_wall_seconds,
        environment=spec.environment.id or None,
        save_at_steps=t.save_at_steps,
    )


def estimate_for_spec(spec) -> CostEstimate:
    """The pre-flight ``CostEstimate`` for a parsed training ``JobSpec``."""
    return estimate_cost(runconfig_from_spec(spec))

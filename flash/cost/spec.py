"""Map a parsed training ``JobSpec`` to a cost ``RunConfig`` / step count / estimate.

Used by ``flash train --cost`` and submit-time quote snapshots. The control plane bills completed
runs from their persisted quote (``RunStatus.cost_usd``), not measured provider wall cost."""

from __future__ import annotations

import time

from flash.core.catalog import samples_on_policy
from flash.cost.analytical import estimate_cost, estimate_profile_cost
from flash.cost.types import CostEstimate, RunConfig
from flash.engine.plan.steps import on_policy_steps, resolve_update_horizon
from flash.engine.profiling.workload_profile import (
    WorkloadProfileMismatch,
    require_matching_rollout_profile,
    require_matching_sft_profile,
    rollout_profile_input_digest,
    sft_profile_input_digest,
)


def _on_policy_epochs(spec) -> int:
    from flash.engine.plan.recipe import RECIPE

    t = spec.train
    default = RECIPE.rl.num_epochs if spec.algorithm == "grpo" else RECIPE.opd.num_epochs
    return int(t.epochs) if t.epochs is not None else default


def _sft_profile(spec):
    # the version that keyed the digest travels on the spec. re-deriving it from
    # `flash.__version__` would make this quote depend on which process is doing the arithmetic:
    # a worker has no flash distribution installed and resolves the "0+unknown" fallback.
    producer_version = spec.workload_profile_producer_version
    input_digest = sft_profile_input_digest(
        spec,
        tokenizer_revision=spec.model_revision,
        producer_version=producer_version,
    )
    if input_digest != spec.workload_profile_input_digest:
        raise WorkloadProfileMismatch(
            "sft workload profile input digest does not match the cost spec"
        )
    return require_matching_sft_profile(
        spec.workload_profile,
        input_digest=input_digest,
        producer_version=producer_version,
        tokenizer_revision=spec.model_revision,
    )


def _rollout_profile(spec):
    """The attached rollout profile when it describes THIS spec and is trustworthy, else None.

    Fails OPEN, which is the opposite of ``_sft_profile`` and deliberate. An sft profile is a
    census: it measures the exact rows training will consume, so a mismatch means the quote would
    describe different work and refusing is correct. A rollout profile is a SAMPLE of a stochastic
    process, and one cannot be taken for every model -- 3 of the 6 catalog models are too small for
    any provider to host. Raising here would make those models unquotable to buy accuracy on the
    others.

    So a missing, stale, mismatched or thin profile silently returns the caller to the declared
    cap, which is exactly today's pricing. The measured path is an improvement when available and
    never a new way for a quote to fail.
    """
    raw = getattr(spec, "workload_profile", None)
    if not raw:
        return None
    from flash import __version__

    try:
        input_digest = rollout_profile_input_digest(
            spec,
            tokenizer_revision=spec.model_revision,
            producer_version=__version__,
        )
        return require_matching_rollout_profile(
            raw,
            input_digest=input_digest,
            producer_version=__version__,
            tokenizer_revision=spec.model_revision,
            now=time.time(),
        )
    except WorkloadProfileMismatch:
        # the sft profile carried on the same field will not match a rollout digest, and neither
        # will a profile taken at a different cap, temperature or environment revision. all of
        # those mean "no measurement for this run", not "this run is unpriceable".
        return None


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
    from flash.engine.plan.recipe import RECIPE

    t = spec.train
    default = (
        RECIPE.rl.prompts_per_step if spec.algorithm == "grpo" else RECIPE.opd.prompts_per_step
    )
    return max(1, int(t.batch_size) if t.batch_size is not None else default)


def _on_policy_prompts_per_step(spec, examples: int) -> int:
    requested = _on_policy_requested_prompts_per_step(spec)
    return min(requested, max(1, int(examples)))


def spec_steps(spec) -> int:
    """Per-seed optimizer steps implied by a train spec (mirrors the worker).

    sft reads the horizon its workload profile measured: the profile already resolved epochs,
    retained rows, realized batch, and ``max_steps`` against the exact tokenized dataset, so
    re-deriving it here from the config would reintroduce the guess the profile exists to replace.
    grpo/opd still derive passes over retained prompts, and positive ``max_steps`` replaces that
    derived count.
    """
    if spec.algorithm in ("grpo", "opd"):
        examples = _on_policy_example_count(spec)
        derived = on_policy_steps(
            epochs=_on_policy_epochs(spec),
            prompt_count=examples,
            prompts_per_step=_on_policy_prompts_per_step(spec, examples),
        )
        return resolve_update_horizon(derived, spec.train.max_steps)
    return _sft_profile(spec).authoritative_steps


def profile_runconfig_from_spec(spec) -> RunConfig:
    """Map a workload-profile ``JobSpec`` to the ``RunConfig`` its bounded-wall quote reads.

    Deliberately not the training shape: a profile job runs no optimizer steps and loads no weights,
    so the only fields that survive are the ones a wall-cap charge needs (rate constraints and the
    cap itself). ``steps=1`` satisfies ``RunConfig``'s positive-step invariant and is never priced.
    """
    if not spec.workload_profile_kind:
        raise ValueError("profile_runconfig_from_spec requires a workload-profile spec")
    g = spec.gpu
    return RunConfig(
        model_id=spec.model,
        method=spec.algorithm,
        steps=1,
        thinking=spec.thinking,
        provider=g.provider or "auto",
        gpu_type=g.type,
        model_revision=spec.model_revision,
        disk_gb=float(getattr(g, "disk_gb", 0.0) or 0.0),
        gpu_count=g.count,
        max_wall_seconds=g.max_wall_seconds,
        environment=spec.environment.id or None,
    )


def runconfig_from_spec(spec) -> RunConfig:
    """Map a parsed ``JobSpec`` to a cost ``RunConfig`` for one adapter-training job.

    unconstrained runs retain cheapest-fit pricing; authored provider/exact-type constraints are
    preserved so the quote matches the allocatable hardware contract.
    """
    if spec.workload_profile_kind:
        raise ValueError(
            "a workload-profile job cannot be priced as training; use estimate_for_spec, which "
            "routes profile specs to their bounded-wall charge"
        )
    t, g = spec.train, spec.gpu
    # Both grpo and opd sample on-policy student completions, so both carry the rollout
    # dimensions (completion length + group size) into the cost model.
    has_rollout = samples_on_policy(spec.algorithm)
    # price the actual opd teacher the run will use. [train].teacher_model is the canonical friendly
    # alias, so a higher/lower-priced teacher moves the itemized
    # teacher-api estimate; "" means the default glm 5.2, and "" for sft/grpo (no teacher).
    teacher_model = ""
    opd_multi_turn = False
    opd_max_turns = None
    if spec.algorithm == "opd":
        from flash.engine.plan.recipe import RECIPE
        from flash.teacher.limits import configured_opd_turn_limit

        teacher_model = t.teacher_model or RECIPE.opd.teacher_model
        opd_multi_turn, opd_max_turns = configured_opd_turn_limit(spec.environment)
    profile = _sft_profile(spec) if spec.algorithm == "sft" else None
    rollout = _rollout_profile(spec) if has_rollout else None
    return RunConfig(
        model_id=spec.model,
        method=spec.algorithm,
        steps=spec_steps(spec),
        seq_len=profile.max_length if profile is not None else t.max_context_tokens,
        completion_len=t.max_completion_tokens if has_rollout else None,
        batch_size=profile.examples_per_update if profile is not None else t.batch_size,
        group_size=t.group_size if has_rollout else None,
        lora_rank=t.lora_rank,
        thinking=spec.thinking,
        teacher_model=teacher_model,
        opd_multi_turn=opd_multi_turn,
        opd_max_turns=opd_max_turns,
        provider=g.provider or "auto",
        gpu_type=g.type,
        model_revision=spec.model_revision,
        disk_gb=float(getattr(g, "disk_gb", 0.0) or 0.0),
        gpu_count=g.count,
        max_wall_seconds=g.max_wall_seconds,
        environment=spec.environment.id or None,
        save_at_steps=t.save_at_steps,
        train_tokens=profile.authoritative_compute_tokens if profile is not None else None,
        supervised_train_tokens=(
            profile.authoritative_supervised_tokens if profile is not None else None
        ),
        sft_packing_mode=profile.packing_mode if profile is not None else "",
        sft_packed_blocks=profile.packed_blocks if profile is not None else None,
        measured_completion_tokens=(
            rollout.completion_tokens_mean if rollout is not None else None
        ),
        measured_prompt_tokens=(rollout.prompt_tokens_mean if rollout is not None else None),
        reward_seconds_per_completion=(
            rollout.reward_seconds_per_completion
            if rollout is not None and rollout.reward_samples > 0
            else None
        ),
    )


def estimate_for_spec(spec, *, allocation=None) -> CostEstimate:
    """Cost estimate for a parsed spec, optionally pinned to the selected live candidate.

    A workload-profile job is priced from its bounded wall cap rather than the workload it exists to
    measure: routing it through the training estimator would require the very profile it produces.
    """
    if spec.workload_profile_kind:
        return estimate_profile_cost(profile_runconfig_from_spec(spec), allocation=allocation)
    return estimate_cost(runconfig_from_spec(spec), allocation=allocation)

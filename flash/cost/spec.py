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


class UnknownPromptPoolSize(ValueError):
    """Raised when a grpo/opd quote has no stated prompt-pool size to derive a horizon from."""


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
    """Retained prompts the run will iterate, from the only two places that state a row count.

    Deliberately NOT ``min()`` of the two caps. ``[train] max_examples`` is a validated ``TrainSpec``
    field the worker enforces unconditionally as ``train[:max_examples]``, so it is a real ceiling on
    the retained pool. ``[environment.params] max_examples`` is not a flash contract at all: params
    are opaque kwargs forwarded to the user's own environment factory, and neither flash nor the
    freesolo sdk applies the name to a dataset. An environment free to ignore it can return every
    row while the key says 2, and since a completed run is billed from this persisted quote, taking
    the smaller value would underquote real training. The train cap can overquote when the
    environment returns fewer rows, which is the safe direction.

    Falling back to one step's worth of prompts is what this raises instead of. The worker sizes
    the horizon from ``len(prompts)`` -- every row the environment yields -- so a pool the config
    never bounded derived exactly one step and quoted a full run at one step's price. The error
    below is the same refusal sft already makes: a horizon nothing measured is not a cheap quote,
    it is an absent one.
    """
    pinned_examples = int(spec.train.max_examples) if spec.train.max_examples else 0
    if pinned_examples > 0:
        return pinned_examples
    env_examples = _env_max_examples(spec)
    if env_examples > 0:
        return env_examples
    # only [train] max_examples is named here. [environment.params] max_examples is still READ
    # above, because an environment that honours it really does bound the pool -- but it reaches
    # the environment as an opaque `load_environment(**params)` kwarg, and the starter templates
    # swallow it into **kwargs and ignore it. Only [train] max_examples is applied by the worker
    # itself (rl/inputs.py:195, opd_train_runner.py:166), so it is the one this can promise.
    raise UnknownPromptPoolSize(
        f"cannot price {spec.algorithm} without a prompt-pool size: set [train] max_examples to "
        "the row count the run will train on, or [train] max_steps to state the horizon directly"
    )


def _env_max_examples(spec) -> int:
    params = getattr(getattr(spec, "environment", None), "params", {}) or {}
    if not isinstance(params, dict):
        return 0
    try:
        value = int(params.get("max_examples") or 0)
    except (TypeError, ValueError, OverflowError):
        # OverflowError is the one that is easy to miss: `int(nan)` raises ValueError but
        # `int(inf)` raises OverflowError, and `max_examples = inf` is valid toml an environment
        # may well use to mean "uncapped". params are opaque kwargs flash does not define, so
        # anything unreadable here means "no cap I can price", never a reason to abort. this
        # feeds an advisory warning that runs before `create_run`, so raising would turn a
        # courtesy line into a submit-blocking traceback.
        return 0
    return max(0, value)


def _on_policy_requested_prompts_per_step(spec) -> int:
    from flash.engine.plan.recipe import RECIPE

    t = spec.train
    default = (
        RECIPE.rl.prompts_per_step if spec.algorithm == "grpo" else RECIPE.opd.prompts_per_step
    )
    return max(1, int(t.prompts_per_step) if t.prompts_per_step is not None else default)


def _on_policy_prompts_per_step(spec, examples: int) -> int:
    return min(_on_policy_requested_prompts_per_step(spec), max(1, int(examples)))


def _rollout_batch_for_quote(spec) -> int:
    """Prompts one priced rollout step trains on: the requested batch, capped by the retained pool.

    The workers retain at most ``max_examples`` rows and then clamp the batch to what is left, so
    `prompts_per_step = 128` against `max_examples = 2` trains on 2. Pricing the raw 128 charges a
    completed or cancelled run for the work it did not do, and ``spec_steps`` already counts steps
    against the capped batch -- so without this one quote mixes a capped step COUNT with an uncapped
    per-step PRICE.

    A pool size is not always knowable, and that is not a pricing failure here. ``max_steps`` states
    the horizon outright, so ``spec_steps`` returns before ever asking for a row count; asking for
    one anyway would reject a fully specified run. There is nothing to cap against in that case, so
    the requested batch stands -- the same number this priced before the cap existed.
    """
    try:
        return _on_policy_prompts_per_step(spec, _on_policy_example_count(spec))
    except UnknownPromptPoolSize:
        return _on_policy_requested_prompts_per_step(spec)


def spec_steps(spec) -> int:
    """Per-seed optimizer steps implied by a train spec (mirrors the worker).

    sft reads the horizon its workload profile measured: the profile already resolved epochs,
    retained rows, realized batch, and ``max_steps`` against the exact tokenized dataset, so
    re-deriving it here from the config would reintroduce the guess the profile exists to replace.
    grpo/opd still derive passes over retained prompts, and positive ``max_steps`` replaces that
    derived count -- so it is read first: a stated horizon needs no pool size to derive one from,
    and asking for a row count the answer does not depend on would reject a fully specified run.
    """
    if spec.algorithm in ("grpo", "opd"):
        pinned_horizon = int(spec.train.max_steps or 0)
        if pinned_horizon > 0:
            return pinned_horizon
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
        # the workload profile never loads model weights and always rents one cheapest card.
        gpu_count=1,
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
        # RunConfig.batch_size is the cost model's own name for "examples per optimizer update", and
        # each algorithm reaches it by a different key: sft through the measured profile, grpo/opd
        # through the retained-prompt cap. reading t.batch_size for rl would always find None now and
        # silently price the recipe default, ignoring an authored batch.
        #
        # the cap matters because the workers apply it: they retain at most `max_examples` rows and
        # then clamp the batch to what is left, so `prompts_per_step = 128` against `max_examples = 2`
        # trains on 2. pricing the raw 128 charges a completed or cancelled run for ~64x the work it
        # performed, and can reject an affordable run on the pre-submit affordability check.
        # `spec_steps` above already counts steps against the capped batch, so taking it here also
        # stops one quote from mixing a capped step COUNT with an uncapped per-step PRICE.
        batch_size=(
            profile.examples_per_update
            if profile is not None
            else (_rollout_batch_for_quote(spec) if has_rollout else t.batch_size)
        ),
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
        # deliberately NOT spec.authored_gpu_count: this asks whether a shape has been RESOLVED
        # yet, not what the author wrote. the marker is provenance and survives allocation, so an
        # allocated run still reports no authored ceiling -- quoting that as auto would re-size a
        # run whose geometry is already rented and bill a different one. auto-size only while the
        # shape is still unresolved (no class chosen and the count still the placeholder).
        gpu_count=(None if spec.gpu_count_auto and g.count == 1 and not g.type else g.count),
        max_wall_seconds=g.max_wall_seconds,
        environment=spec.environment.id or None,
        save_at_steps=t.save_at_steps,
        train_tokens=profile.authoritative_compute_tokens if profile is not None else None,
        supervised_train_tokens=(
            profile.authoritative_supervised_tokens if profile is not None else None
        ),
        sft_packing_mode=profile.packing_mode if profile is not None else "",
        sft_packed_blocks=profile.packed_blocks if profile is not None else None,
        sft_retained_examples=profile.retained_examples if profile is not None else None,
        measured_completion_tokens=(
            rollout.completion_tokens_mean if rollout is not None else None
        ),
        measured_prompt_tokens=(rollout.prompt_tokens_mean if rollout is not None else None),
        # a probe that ran and FAILED is not a measurement. reward_failures == reward_samples is a
        # permitted profile state and trustworthy() has no reward-success check, so comparing
        # against 0 would let an all-failed probe contribute its ~0s latency as though it were
        # evidence. requiring a successful sample sends that case to the default instead.
        reward_seconds_per_completion=(
            rollout.reward_seconds_per_completion
            if rollout is not None and rollout.reward_samples > rollout.reward_failures
            else None
        ),
    )


def sft_ranking_overrides(spec) -> dict:
    """Profile-derived knobs hardware ranking must price SFT on, or ``{}`` when unavailable.

    Ranking runs BEFORE the quote (``_allocate_attempt`` precedes ``_estimate_selected_quote``) and
    must never fail a submission, so this fails OPEN where ``runconfig_from_spec`` fails closed.
    That is the only difference between them: both read the same digest-validated profile, and the
    keys here mirror the profile-derived fields there so the two cannot describe different work.

    ``batch_size`` is the executed batch, not the authored one -- the profile reduces it to
    ``examples_per_update``, which every exact-unpacked run pins to 1. It and
    ``sft_retained_examples`` both bound the width ``sft_data_parallel_cards`` credits, so ranking
    without them picks a wider, costlier shape than the run can use. ``seq_len`` follows for the
    same reason: the authored context length is not the one measured.
    """
    if getattr(spec, "algorithm", "") != "sft":
        return {}
    try:
        profile = _sft_profile(spec)
    except Exception:
        # unreadable or mismatched profile: rank exactly as unconstrained as before. the quote path
        # validates the digest and fails closed, so a bad profile still cannot reach a paid launch.
        return {}
    overrides = {}
    if int(profile.examples_per_update) >= 1:
        overrides["batch_size"] = int(profile.examples_per_update)
    if int(profile.retained_examples) > 0:
        overrides["sft_retained_examples"] = int(profile.retained_examples)
    if int(profile.max_length) >= 1:
        overrides["seq_len"] = int(profile.max_length)
    return overrides


def estimate_for_spec(spec, *, allocation=None) -> CostEstimate:
    """Cost estimate for a parsed spec, optionally pinned to the selected live candidate.

    A workload-profile job is priced from its bounded wall cap rather than the workload it exists to
    measure: routing it through the training estimator would require the very profile it produces.
    """
    if spec.workload_profile_kind:
        return estimate_profile_cost(profile_runconfig_from_spec(spec), allocation=allocation)
    return estimate_cost(runconfig_from_spec(spec), allocation=allocation)

"""Map a parsed training ``JobSpec`` to a cost ``RunConfig`` / step count / estimate.

Used by ``flash train --cost`` and submit-time quote snapshots. The control plane bills completed
runs from their persisted quote (``RunStatus.cost_usd``), not measured provider wall cost."""

from __future__ import annotations

from flash.core.catalog import samples_on_policy
from flash.cost.analytical import estimate_cost
from flash.cost.types import CostEstimate, RunConfig
from flash.engine.plan.steps import on_policy_steps, resolve_update_horizon
from flash.engine.profiling.workload_profile import (
    WorkloadProfileMismatch,
    require_matching_sft_profile,
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


def _on_policy_example_count(spec) -> int:
    """Retained prompts the run will iterate, from the one field that states a row count.

    Only ``[train] max_examples`` counts. It is a validated ``TrainSpec`` field the worker enforces
    unconditionally as ``train[:max_examples]`` (rl/inputs.py:195, opd_train_runner.py:166), so it
    is a real ceiling on the retained pool.

    ``[environment.params] max_examples`` is deliberately NOT read, in either direction. It is not a
    flash contract: params are opaque kwargs forwarded to the user's own environment factory, and
    neither flash nor the freesolo sdk applies the name to a dataset -- the starter templates
    swallow it into ``**kwargs`` and ignore it. Whether the run honours it is therefore unknowable
    here, and the value is wrong BOTH ways for a spec that states nothing else. Against
    ``prompts_per_step = 128`` and an env-params cap of 2: an environment that honours it trains 2
    prompts, so pricing an uncapped 128-prompt step overcharges 64x; one that ignores it yields
    every row, so deriving a 1-step horizon from the 2 underquotes a 1153-row pool 10x. A number
    nothing enforces cannot bound a price in either direction, so it is not evidence of a horizon.

    Falling back to one step's worth of prompts is what this raises instead of. The worker sizes
    the horizon from ``len(prompts)`` -- every row the environment yields -- so a pool the config
    never bounded derived exactly one step and quoted a full run at one step's price. The error
    below is the same refusal sft already makes: a horizon nothing measured is not a cheap quote,
    it is an absent one.
    """
    pinned_examples = int(spec.train.max_examples) if spec.train.max_examples else 0
    if pinned_examples > 0:
        return pinned_examples
    raise UnknownPromptPoolSize(
        f"cannot price {spec.algorithm} without a prompt-pool size: set [train] max_examples to "
        "the row count the run will train on, or [train] max_steps to state the horizon directly"
    )


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

    The cap is the pool ``_on_policy_example_count`` reports, which is exactly the enforced
    ``[train] max_examples`` -- the only row count anything applies. It deliberately does not read
    ``[environment.params] max_examples`` (see there), so capping against it here cannot price a
    batch by a number nothing enforces.

    A pool size is not always knowable, and that is not a pricing failure here. ``max_steps`` states
    the horizon outright, so ``spec_steps`` returns before ever asking for a row count; asking for
    one anyway would reject a fully specified run. There is nothing to cap against in that case, so
    the requested batch stands -- the same number this priced before the cap existed.

    That stated horizon is also where the unenforced key stops mattering. ``max_steps`` fixes the
    step COUNT but not how wide a step is, so a config that names the env-params cap and no enforced
    one leaves the batch unknowable in the same both-ways sense: an environment that honours it
    trains ``len(prompts)`` per step, so the requested batch overcharges 64x, and one that ignores
    it trains the full batch, so the env value undercharges 64x. Refusing looks right and is not:
    ``charge_usd_for_spec`` prices a CANCELLED run through here by pinning ``max_steps`` onto the
    spec, and its outer ``except Exception`` turns any refusal into the $0 fallback -- billing
    nothing for a run that really executed. A quote that is too high can at least be seen and
    disputed; a silent zero cannot. So the requested batch stands, and the guard against pricing an
    unenforced number lives where it costs nothing: ``_on_policy_example_count`` refuses to derive a
    HORIZON from it, which is the path a submit-time quote takes.
    """
    try:
        return _on_policy_prompts_per_step(spec, _on_policy_example_count(spec))
    except UnknownPromptPoolSize:
        return _on_policy_requested_prompts_per_step(spec)


def spec_steps(spec) -> int:
    """Per-run optimizer steps implied by a train spec (mirrors the worker).

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


def runconfig_from_spec(spec) -> RunConfig:
    """Map a parsed ``JobSpec`` to a cost ``RunConfig`` for one adapter-training job.

    unconstrained runs retain cheapest-fit pricing; authored provider/exact-type constraints are
    preserved so the quote matches the allocatable hardware contract.
    """
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
        # `g` is always a GpuSpec: `spec.gpu` is a typed field whose default_factory constructs one,
        # and `g.provider` / `g.type` / `g.count` / `g.max_wall_seconds` are read plainly right here,
        # so anything else would already have raised AttributeError before reaching this line. every
        # field below also carries a dataclass default, which is what makes a getattr default doubly
        # unreachable rather than merely unused.
        providers=g.providers or (),
        gpu_type=g.type,
        # the rest of an ordered `[gpu] type` list. allocation cost-ranks every acceptable class, so
        # dropping these here quotes the head alone and the affordability precheck can refuse a run
        # whose cheaper authored fallback it would really have rented.
        gpu_type_fallbacks=tuple(g.type_fallbacks or ()),
        model_revision=spec.model_revision,
        disk_gb=float(g.disk_gb or 0.0),
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
    )


def sft_ranking_overrides(spec) -> dict:
    """Profile-derived knobs hardware ranking must price SFT on, or ``{}`` when unavailable.

    Ranking runs after the customer quote is frozen and must never change that accepted amount.
    This therefore fails open where ``runconfig_from_spec`` fails closed: ranking may degrade to a
    conservative shape, while the quote path still requires the digest-validated profile. The keys
    here mirror the profile-derived fields so ranking describes the same work without repricing it.

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
    """Cost estimate for a parsed spec, optionally pinned to the selected live candidate."""
    return estimate_cost(runconfig_from_spec(spec), allocation=allocation)

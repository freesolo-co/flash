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
    """Prompts this quote prices, preferring the cap flash actually enforces.

    Deliberately NOT ``min()`` of the two caps. ``[train] max_examples`` is a validated ``TrainSpec``
    field the worker enforces unconditionally as ``train[:max_examples]``, so it is a real ceiling on
    the retained pool. ``[environment.params] max_examples`` is not a flash contract at all: params
    are opaque kwargs forwarded to the user's own environment factory, and neither flash nor the
    freesolo sdk applies the name to a dataset. An environment free to ignore it can return every
    row while the key says 2, and since a completed run is billed from this persisted quote, taking
    the smaller value would underquote real training. The train cap can overquote when the
    environment returns fewer rows, which is the safe direction.

    Reading the env value when no train cap is set is pre-existing behavior (#465) and is left
    alone; narrowing it needs a real enforced cap, which is pricing work outside this change.
    ``thin_rl_batch_warning`` still inspects both, because warning on a thin authored value costs
    nobody money.
    """
    pinned_examples = int(spec.train.max_examples) if spec.train.max_examples else 0
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


# the prompts-per-step below which an on-policy step stops being a batch. at 1 the update is a
# single prompt's group, so nothing averages the prompt-to-prompt spread out of the gradient; the
# few above it are still thin enough that one unlucky prompt dominates an update. not a hard
# minimum -- a deliberately tiny batch is a legitimate way to buy optimizer steps on a derived
# horizon (see TRAINING.md), which is exactly why this warns instead of rejecting.
RL_THIN_PROMPTS_PER_STEP = 4


def thin_rl_batch_warning(spec) -> str | None:
    """One user-facing line when an rl run's optimizer batch is too thin to be a batch.

    ``batch_size`` reaches the optimizer batch by two different routes. Under sft a measured
    workload profile sits in between: it resolves the authored value against the real tokenized
    dataset into ``examples_per_update`` (packed) or pins the batch to 1 (unpacked), and the
    per-device micro-batch is derived from that result. Under grpo/opd there is no profile: the key
    IS ``prompts_per_step``, then verl's ``data.train_batch_size`` and ``ppo_mini_batch_size``. So
    the standard sft memory workaround, ``batch_size = 1``, silently turns an rl run into one
    prompt per update, and nothing errors.

    Reads the configured prompts-per-step rather than the authored field. That is an UPPER BOUND on
    the runtime batch, not the resolved one: the worker filters prompts over the token budget before
    clamping (``_build_grpo_prompts`` then ``resolve_grpo_prompts_per_step``), so an uncapped config
    can still resolve thin and go unwarned. Closing that gap would mean importing and running the
    user's ``environment.py``, which an offline quote deliberately does not do. The bound is
    one-sided on purpose: everything it warns about is genuinely thin, and what it misses is a false
    negative rather than a false alarm on a healthy run.

    Within that bound it still beats the authored field, because a ``max_examples`` cap (from
    either ``[train]`` or ``[environment.params]``) clamps the batch to the retained prompt pool
    (``_on_policy_prompts_per_step``), so a scaffolded ``max_examples = 2`` run trains on 2 prompts
    per update whether or not a ``batch_size`` was ever written. The caps do not override each
    other -- the environment one bounds what ``env.dataset()`` returns and the train one slices
    that afterwards -- so the pool is the SMALLEST of them. Any of these can be thin at once, and
    raising one while another is still thin leaves prompts-per-step thin, so the message names
    every one of them. Note it does NOT claim raising one alone moves nothing: that only holds when
    the other is the strict minimum, and at ``batch_size = 2`` under ``max_examples = 3`` lifting
    the batch alone does move the result, to 3.

    What the thin batch costs differs by algorithm, so the message does too. grpo keeps a working
    per-prompt baseline at any batch size (verl centres each response against its own prompt's
    group; dr-grpo, ``norm_adv_by_std_in_grpo=False``) and only loses the averaging ACROSS prompts.
    opd has no advantages at all -- its objective is groupwise reverse KL against the teacher with
    ``use_policy_gradient=false`` -- so there is no baseline to reassure anyone about, and unlike
    grpo its ``batch_size`` genuinely IS a memory lever (it scales rollout concurrency and the loss
    microbatch in ``estimate_vram_gb``). Telling an opd user to raise it would be advice to OOM.

    The price of widening also flips with the horizon and with WHICH knob is raised, so the message
    never says "for the same money". On a DERIVED horizon a wider batch means no more updates and
    usually fewer, so it is cheaper (measured: grpo 9B over 800 prompts quotes $21.88 at 1 and
    $7.37 at 16). Under a positive ``max_steps`` the update count is pinned, so widening buys
    averaging at strictly more generation: the same run quotes $0.27 at 1 and $0.51 at 4 for an
    identical 10 steps. Raising a POOL cap only adds updates while the batch stays pinned (batch 2,
    cap 2 -> 8: 1 step at $0.035 to 4 steps at $0.141) -- and this message never asks for that in
    isolation, because a thin batch is named alongside the cap and the two then move together,
    holding the horizon flat (``ceil(2/2) == ceil(4/4) == 1``). With no authored batch at all,
    prompts-per-step rises with the cap for the same reason, and the quote is flat at $0.47 across
    caps 2 through 64 because it prices the recipe's full rollout width rather than the retained
    pool. So every widening branch except the pinned-``max_steps`` one names the mechanism and
    sends the user to ``--cost`` instead of promising a direction.
    """
    if spec.algorithm not in ("grpo", "opd"):
        return None
    # the SMALLEST configured cap, which is deliberately not what `_on_policy_example_count` prices.
    # that function must not read `[environment.params] max_examples` down over an enforced `[train]`
    # cap, because params are opaque kwargs to the user's own environment and a run billed off the
    # smaller number would underquote if the environment ignores the key. warning on it is the
    # opposite trade: if the environment DOES honor it, the pool really is 2 and the user wants to
    # know, and if it does not, they have read one accurate sentence about a knob they wrote.
    examples = min(
        [c for c in (int(spec.train.max_examples or 0), _env_max_examples(spec)) if c > 0],
        default=_on_policy_requested_prompts_per_step(spec),
    )
    prompts_per_step = _on_policy_prompts_per_step(spec, examples)
    if prompts_per_step >= RL_THIN_PROMPTS_PER_STEP:
        return None
    prompts = "1 prompt" if prompts_per_step == 1 else f"{prompts_per_step} prompts"
    authored = spec.train.batch_size
    # a cap only exists if one was actually written. with neither table setting it,
    # `_on_policy_example_count` falls back to the requested batch, which makes `examples` equal
    # `prompts_per_step` for reasons that have nothing to do with a pool -- reading a bind off that
    # equality would invent a cap the config does not contain and send the user to a phantom key.
    #
    # name every cap that is BELOW the healthy threshold, not merely the one at the current
    # minimum. staggered caps are the trap: with [train] 2 behind [environment.params] 3, naming
    # only the train cap sends the user to raise it and land on 3, still thin and still warned.
    # the remedy has to resolve the problem in one edit, so every configured constraint that would
    # still bind after the others are lifted gets named.
    cap_keys = [
        key
        for key, value in (
            ("[train] max_examples", int(spec.train.max_examples or 0)),
            ("[environment.params] max_examples", _env_max_examples(spec)),
        )
        if 0 < value < RL_THIN_PROMPTS_PER_STEP
    ]
    cap_key = " and ".join(f"`{k}`" for k in cap_keys) if cap_keys else None
    # same rule for the authored batch: it binds if it is itself thin, even when a smaller pool is
    # what resolves today. raising only the cap would leave `batch_size = 3` capping the batch at 3.
    batch_binds = authored is not None and authored < RL_THIN_PROMPTS_PER_STEP
    pool_binds = cap_key is not None
    if batch_binds:
        # the sft/rl name collision is only worth explaining when a batch_size was actually written
        lead = (
            f"`[train] batch_size = {authored}` is the OPTIMIZER batch for {spec.algorithm}: it "
            f"sets prompts-per-step, so each update trains on {prompts}. It does NOT mean here what "
            "it means under sft, where it is the batch a measured workload profile turns into the "
            "optimizer batch and its step horizon, so an sft memory workaround does not transfer."
        )
        if pool_binds:
            # deliberately does NOT say the cap holds the pool "below" the batch, or that raising
            # one alone moves nothing: neither survives a cap ABOVE the batch. at batch 2 behind a
            # cap of 3 the cap is higher, and lifting the batch alone does move the result, to 3.
            # what is true for every arrangement is the weaker claim, because prompts-per-step is
            # min(batch, pool) and both named knobs sit under the threshold: whichever one is
            # raised, the other is still thin and still the minimum.
            lead += (
                f" {cap_key} holds the prompt pool under a real batch too, so raising either one "
                "alone still leaves prompts-per-step thin."
            )
    else:
        caps_verb = "hold" if len(cap_keys) > 1 else "holds"
        lead = (
            f"This {spec.algorithm} run's OPTIMIZER batch is {prompts} per update: {cap_key} "
            f"{caps_verb} the prompt pool below a real batch, and prompts-per-step cannot exceed "
            "the pool."
        )
    targets = [
        k for k, binds in (("`batch_size`", batch_binds), (cap_key or "", pool_binds)) if binds
    ]
    raise_target = " and ".join(targets)
    if spec.algorithm == "grpo":
        consequence = (
            "Each prompt's completions are still centred against their own group, so the advantage "
            "signal survives; what a thin batch costs is the averaging ACROSS prompts, so every "
            "update follows only those few and the gradient is far noisier. Under grpo "
            "`batch_size` is not a memory lever, so a wider batch costs no extra vram."
        )
    else:
        # opd's batch_size is also a real vram lever, so "raise it" is not safe blanket advice.
        consequence = (
            "opd distills against the teacher rather than scoring completions against each other, "
            "so a thin batch breaks no baseline: it just makes every update follow very few "
            "prompts, which is noisier. Unlike grpo, opd's `batch_size` is ALSO a memory lever (it "
            "sizes rollout concurrency and the loss microbatch), so keep any widening within what "
            "the gpu allows."
        )
    remedy = _thin_rl_batch_remedy(
        spec, raise_target=raise_target, batch_binds=batch_binds, pool_binds=pool_binds
    )
    return f"{lead} {consequence} {remedy}"


def _thin_rl_batch_remedy(spec, *, raise_target: str, batch_binds: bool, pool_binds: bool) -> str:
    """The closing sentence: what raising the named knobs actually buys, and at what price.

    Split from ``thin_rl_batch_warning`` to keep it under the function-size limit. It is the one
    cohesive phase in that message: every branch here answers "what happens if I do what you just
    told me", and each direction it claims is measured (see the branch comments).
    """
    # what widening costs flips with the horizon, so the remedy has to name the right trade.
    if spec.train.max_steps and int(spec.train.max_steps) > 0 and batch_binds:
        return (
            f"Your `max_steps` pins the update count, so raising {raise_target} buys that averaging "
            "at strictly more generated work and a higher bill for the same number of updates."
        )
    if spec.train.max_steps and int(spec.train.max_steps) > 0:
        # cap-only under a pinned horizon: `runconfig_from_spec` leaves `batch_size=None`, so the
        # quote normalizes every widened cap to the same recipe batch over the same pinned steps
        # and caps 2, 4 and 8 all price identically. the generated work does grow, but this quote
        # will not show it, so promise nothing and say where the number really comes from.
        return (
            f"Your `max_steps` pins the update count, so raising {raise_target} buys that averaging "
            "without adding updates. Re-run `--cost` to see what it does to this quote."
        )
    if pool_binds and batch_binds:
        # both knobs are named here, so they move TOGETHER and the horizon does not grow:
        # ceil(2/2) == ceil(4/4) == ceil(8/8) == 1. "adds passes" describes raising the cap while
        # the batch stays pinned (batch 2, cap 2 -> 8: 1 step to 4), which is not what this remedy
        # asks for, so state the same-horizon, wider-update effect and price it with `--cost`.
        return (
            f"Raise {raise_target} together: both hold prompts-per-step down, so lifting them in "
            "step widens each update rather than adding updates. Re-run `--cost` to see what it "
            "does to this quote."
        )
    if pool_binds:
        # the pool is the only thin knob, so prompts-per-step rises WITH the cap -- but only up to
        # the requested batch, which is the recipe default when none was authored and the authored
        # value otherwise. this branch is reachable with a HEALTHY batch_size (8 is not thin, so it
        # is not named), and there the widening stops at that ceiling: batch 8 over caps 2, 4, 8,
        # 16, 800 gives 1, 1, 1, 2, 100 updates. so promise the widening only "until it reaches
        # your batch_size" rather than claiming the horizon never moves, and let `--cost` price it.
        return (
            f"Raise {raise_target}: prompts-per-step follows the pool until it reaches your "
            "`batch_size`, so this widens each update rather than adding updates. Re-run `--cost` "
            "to see what it does to this quote."
        )
    # the "buying steps" workflow is about lowering batch_size against a fixed pool. a thin POOL
    # does not buy steps, it just shortens the run, so that caveat would excuse the wrong config --
    # offer it only when the authored batch is what is holding the batch down.
    #
    # deliberately claims no direction for the bill, and no strict drop in updates either. the
    # derived horizon is ``ceil(examples / batch)``, which is non-increasing in batch but plateaus:
    # at ``max_examples = 5`` both batch 3 and batch 4 resolve to 2 updates, so the extra
    # generation is pure added cost and takes the quote from $0.0867 to $0.1027. "no more updates"
    # is the guarantee ceil() actually gives; "fewer" is only the common case, and `--cost` prices
    # the real answer in the line above this one.
    return (
        f"Raise {raise_target} unless you are deliberately buying optimizer steps on a derived "
        "horizon (see TRAINING.md): against a fixed prompt pool a wider batch spreads the same "
        "prompts over no more updates than it does now. Re-run `--cost` to see what it does to "
        "this quote."
    )


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

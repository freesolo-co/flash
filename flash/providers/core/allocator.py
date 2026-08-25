"""GPU allocation: cheapest fitting (provider, GPU class) across active providers."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import NoReturn

from flash._internal.logging import get_logger
from flash.providers.core.base import (
    GPU_INFO,
    Allocation,
    AllocationConstraints,
    Candidate,
    CapacityLookupError,
    CapacityUnavailableError,
    UnsupportedGpuError,
    _run_cost_key,
    authored_gpu_ceiling,
    canonical_gpu,
    largest_rentable_count,
    providers_for,
    rentable_gpu_counts,
    run_config_for_ranking,
    smallest_fitting_gpu_count,
    wider_shape_remedy,
)
from flash.providers.core.fit_errors import (
    batch_bound_width_note,
    catalog_check_hint,
    drop_pin_hint,
    rents_arbitrary_card_counts,
    vram_fit_error_message,
    widenable_gpu_names,
)
from flash.providers.core.registry import (
    PROVIDER_NAMES,
    available_providers,
    get_provider,
    validated_provider_preferences,
)
from flash.providers.core.sharding import (
    MAX_COMBINATION_CARDS,
    SHARD_VRAM_EFFICIENCY,
    combined_vram_gb,
)

logger = get_logger(__name__)


def vram_headroom() -> float:
    """Sizing headroom multiplier; shared by submit-time allocator and parse-time provisional_gpu."""
    return 1.1


def required_vram_gb(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
    model_revision: str = "",
) -> int:
    """VRAM required for the full run, sized to actual knobs via model_required_vram_gb."""
    from flash.engine.plan.vram import model_required_vram_gb

    return model_required_vram_gb(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        headroom=vram_headroom(),
        model_revision=model_revision,
    )


# RunConfig field -> the `train` knob holding the same quantity. The two vocabularies differ, so
# VRAM sizing (which reads `train`) cannot consume `overrides` without this mapping. `lora_rank`
# and the rest are absent because the profile does not measure them -- only these three move.
_TRAIN_KNOB_FOR_OVERRIDE = {
    "batch_size": "batch_size",
    "seq_len": "max_context_tokens",
}


def _overridden_train(train, overrides):
    """``train`` with profile-measured knobs substituted in, for VRAM sizing.

    Ranking and sizing must agree on the shape of the work: ranking takes the profile through
    ``overrides``, so sizing has to see the same numbers or submit reserves for a batch the run
    never executes. Returns ``train`` untouched when there is nothing to substitute, which keeps
    every non-SFT path byte-identical.
    """
    knobs = {
        _TRAIN_KNOB_FOR_OVERRIDE[key]: value
        for key, value in (overrides or {}).items()
        if key in _TRAIN_KNOB_FOR_OVERRIDE
    }
    if not knobs or train is None:
        return train
    if isinstance(train, dict):
        return {**train, **knobs}
    from dataclasses import is_dataclass, replace

    if is_dataclass(train):
        return replace(train, **knobs)
    return train


def _step_cost_ranker(model_id, algorithm, train, thinking, model_revision="", overrides=None):
    """``candidate -> dollars for one optimizer step``, or None when the run cannot be priced.

    Wraps the shared per-step cost key with the multi-card speedup, which is the one thing the
    single-card paths (parse-time provisional, cost estimate) have no notion of: a combination
    bills every card but only the gpu-bound half of a step is divided among them.
    """
    cost_key = _run_cost_key(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        model_revision=model_revision,
        overrides=overrides,
    )
    if cost_key is None:
        logger.debug("total-cost ranking unavailable; ranking on $/hr")
        return None

    from flash.cost.analytical import sharded_step_seconds

    # the same one-step config the cost key was built from, so the single- and multi-card branches
    # below cannot price a run off different knobs -- including the profile `overrides`.
    config = run_config_for_ranking(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        model_revision=model_revision,
        overrides=overrides,
    )

    def cost_per_step(candidate: Candidate) -> float:
        if candidate.gpu_count <= 1:
            # identical to the single-card key the preview and estimate use, so the three paths
            # agree exactly whenever one card is enough.
            return cost_key(candidate.gpu, candidate.hourly_usd)
        # scaling is per-class (an nvlink pair keeps ~0.88 of linear per card, a pcie pair ~0.71),
        # and for grpo/opd the step floor is only ~80% shardable -- sharded_step_seconds owns both,
        # so ranking cannot drift from the quote the run is actually billed against. the candidate's
        # provider is part of that: it is what decides whether the combination gets nvlink credit,
        # and `config` here is the ranking config, which carries no provider at all.
        seconds = sharded_step_seconds(
            config, candidate.gpu, candidate.gpu_count, candidate.provider
        )
        return candidate.total_hourly_usd * seconds / 3600.0

    return cost_per_step


def _fits(candidate: Candidate, need: int, executed_gpu_count: int = 0) -> bool:
    """Whether this rentable shape can actually hold the run on the ranks it will launch.

    ``combined_vram_gb`` is the shared fit model, so a shape accepted here is a shape parse-time
    sizing also accepted -- the two must not be able to disagree. See ``_executed_width`` for why
    the credited count is the launched one rather than the billed one.
    """
    fit_count = executed_gpu_count or candidate.gpu_count
    return combined_vram_gb(candidate.vram_gb, fit_count) >= need


def _executed_gpu_count(algorithm: str, train, overrides, gpu_count: int) -> int:
    """Ranks a run of this shape would actually launch, which is the card count unless a batch bounds it.

    Reads the executed batch from the SAME substituted knobs sizing does (``_overridden_train``) so
    the fit gate and the VRAM requirement describe one run. The retained row count comes from
    ``overrides`` instead: it is a profile measurement rather than a ``TrainSpec`` sizing knob, so
    ``_TRAIN_KNOB_FOR_OVERRIDE`` never carries it onto ``train`` and reading it there would miss
    every rows-bound narrowing.

    Both branches are the same rule -- verl divides the batch across dp ranks, so a rank with no
    share cannot launch -- over each algorithm's own unit of work: sft counts rows, grpo/opd count
    the sequences one step holds after the group repeat.
    """
    algo = (algorithm or "").strip().lower()
    if algo in ("grpo", "rl", "opd"):
        return _executed_rl_gpu_count(algo, train, gpu_count)
    if algo != "sft":
        return gpu_count
    batch = _sizing_int(train, "batch_size", 0)
    if batch <= 0:
        # an UNKNOWN batch is not a batch of 1. defaulting it would assert the most restrictive
        # width on every caller that ranks without knobs and reject multi-card sft outright, which
        # is a worse failure than the one this clamp prevents: the narrowing must only bite where
        # the executed width is actually known to be smaller.
        return gpu_count
    from flash.engine.plan.steps import sft_data_parallel_cards

    # 0 rows means UNMEASURED, and `sft_data_parallel_cards` reads it that way (it does not narrow),
    # so an absent profile keeps the batch-bound width rather than inventing a row limit.
    return sft_data_parallel_cards(
        gpu_count, batch, _sizing_int(overrides, "sft_retained_examples", 0)
    )


def _executed_rl_gpu_count(algorithm: str, train, gpu_count: int) -> int:
    """Ranks grpo/opd would launch: bounded by the SEQUENCES one step holds, not the prompt count.

    Both algorithms hand verl ``data.train_batch_size = prompts_per_step`` (grpo
    ``train/rl/verl_config.py``, opd ``opd_train_runner`` via ``train/opd/overrides.py``) and then
    ``batch.repeat(rollout.n)``, so a step carries ``prompts_per_step * group_size`` sequences. That
    product is the quantity verl chunks across dp ranks -- verl derives its own agent-loop worker
    count from the same product -- and it must divide the width exactly or the run aborts at step 0.

    An unknown prompt count does not narrow, matching the sft branch: defaulting it would assert the
    most restrictive width on every caller that ranks without knobs and reject multi-card rl
    outright.

    An unknown ``group_size`` takes the RECIPE default for this algorithm, which is the value the
    worker resolves it to (``train/rl/inputs.py``: ``gcfg.get("group_size") or rl.group_size``) and
    the one the cost path fills in (``RunConfig.normalized``). Hardcoding 1 here would under-credit a
    grpo step eightfold -- `RECIPE.rl.group_size` is 8 -- so one prompt on eight cards would be
    sized as a single rank and a model that fits across the fsdp group would be REJECTED before it
    could launch. The two defaults differ by algorithm (grpo groups, opd distills one completion per
    prompt), which is exactly why this reads the recipe rather than a literal.
    """
    prompts = _sizing_int(train, "prompts_per_step", 0)
    if prompts <= 0:
        return gpu_count
    from flash.engine.plan.recipe import RECIPE
    from flash.engine.plan.steps import rl_data_parallel_cards

    recipe = RECIPE.opd if algorithm == "opd" else RECIPE.rl
    group = _sizing_int(train, "group_size", recipe.group_size)
    return rl_data_parallel_cards(gpu_count, prompts * group)


def _executed_width(algorithm: str, train, overrides):
    """THE rule for how many of a rented card count actually join the run.

    Multi-card fit comes from sharding, so a card that never enters the fsdp group contributes no
    memory. sft shards by data and bounds its width by the batch and the row count, so an unpacked
    profile pins the batch to 1 and launches ONE rank however many cards are rented. Crediting the
    billed count admits a run on memory it will not have: a 4B at 32k needs 28 GB and is correctly
    rejected on one 24 GB card, but renting two credits 35 GB, passes, then OOMs on the single rank
    that starts. Renting more cards is not a way to pass the fit gate.

    Bound ONCE per allocation and handed to every site that asks a question about a card count: the
    filter that removes shapes, the classification that says why none were left, the pinned-class
    precheck, and each ``--gpus N`` remedy. One rule for all of them is the invariant -- each time a
    site was left crediting the RENTED count instead, it contradicted the filter: a width the filter
    deterministically rejects was reported as "structurally offered" (a terminal miss raised as
    retryable, re-polling a market that cannot widen a batch), and advice named a card count that
    buys memory the run never joins. ``method_card_speedup`` already clamps throughput to this same
    width; crediting VRAM is the other half of that.
    """
    return lambda gpu_count: _executed_gpu_count(algorithm, train, overrides, gpu_count)


def _fitting_candidates(candidates, need: int, executed_width) -> list:
    """The shapes that hold the run on the ranks they will actually launch.

    Providers report the shapes they can genuinely rent (RunPod takes a count, Lambda names it in the
    instance type, Vast bakes it into the offer); the allocator owns only whether a shape fits. The
    executed width is per-candidate because it is bounded by that candidate's own card count. Stamp it
    on every survivor so recovery values the failed and retry shapes with the exact rule used here.
    """
    fitting = []
    for candidate in candidates:
        executed_gpu_count = executed_width(candidate.gpu_count)
        if _fits(candidate, need, executed_gpu_count):
            fitting.append(replace(candidate, executed_gpu_count=executed_gpu_count))
    return fitting


def _sizing_int(train, name: str, default: int) -> int:
    """A positive int knob off a dict-or-object ``train``, or ``default`` when absent/unusable.

    ``OverflowError`` joins the type errors because ``int(float("inf"))`` raises it, and sizing runs
    BEFORE the knob validator that rejects a non-finite number: escaping here turns the clean 400
    that validator produces into a 500. Unusable is unusable however the conversion fails -- the run
    is still rejected, this only has to let the rejection reach the validator that words it.
    """
    if train is None:
        return default
    value = train.get(name) if isinstance(train, dict) else getattr(train, name, None)
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if number > 0 else default


def _structurally_fits(available, need: int, cap: int, executed_width, acceptable=()) -> bool:
    """Whether an acceptable provider class could hold the run, ignoring current stock.

    Reads advertised classes so capacity outages cannot affect the answer, and applies the executed
    width because a shape the run will not launch on is terminal rather than sold out.
    """
    allowed = set(acceptable)
    for name in available:
        try:
            classes = get_provider(name).gpu_classes()
        except Exception:  # a provider that cannot even list classes proves nothing either way
            continue
        for gpu_class in classes:
            if allowed and gpu_class.name not in allowed:
                continue
            for count in rentable_gpu_counts(cap):
                if combined_vram_gb(gpu_class.vram_gb, executed_width(count)) >= need:
                    return True
    return False


# Widest count safe to rent when a model's true head geometry could not be certified. Named because
# the guard that skips certification (`cap > _UNCERTIFIED_CAP`) is only correct while it matches the
# ceiling certification would otherwise apply: raise one without the other and a run either takes a
# hub round trip that cannot widen it, or skips the round trip that would have. ALLOC-004 tracks
# validating arbitrary off-catalog head geometry at every width.
_UNCERTIFIED_CAP = 4


def geometry_safe_gpu_cap(
    model_id: str, max_gpu_count: int, *, model_revision: str = "", certify: bool = False
) -> int:
    """Rentable ceiling whose head divisibility is known before paid allocation.

    The width becomes vLLM's ``tensor_model_parallel_size`` for the rollout engine (grpo
    ``train/rl/verl_config.py``, opd ``train/opd/overrides.py``), and vLLM requires
    ``num_attention_heads % tp_size == 0`` -- it raises at engine init otherwise -- so a catalog row
    is only safe at the counts that divide its OWN head count. Curated membership is not uniform
    geometry: catalog head counts are 8, 8, 16, 16, 24, and 16, so trusting membership alone
    accepted an 8-card width for the 27B (24 heads) that the engine rejects at init, after the box
    was already rented.

    This cap OUTLIVED ulysses. It was written when the width was also
    ``ulysses_sequence_parallel_size``; sequence parallelism is now pinned off on all three
    algorithms (it corrupts GatedDeltaNet state), but rollout tensor parallelism still consumes the
    same width, so the same divisibility gate is still what stands between a rented box and a
    post-payment engine failure.

    Scope, stated precisely: this certifies QUERY-head divisibility only. vLLM also constrains kv
    heads and the GDN linear dimensions under tensor parallelism, and Flash records both
    (``num_key_value_heads``, ``linear_num_value_heads``) without gating on them. Every current row
    divides 1/2/4/8 on all three axes, so nothing is mis-admitted today. Widening the check to those
    two axes is a separate invariant and deliberately not in this change.

    The head count is READ from the row (``num_attention_heads``), never derived: ``hidden_size //
    head_dim`` is a different number on four of the six rows -- see ``_query_attention_heads``.

    A revision whose geometry cannot be certified keeps the pre-existing four-card ceiling rather
    than renting 8 cards verl may reject at startup, but that ceiling only NARROWS the divisor
    search; it is not a substitute for it. A ceiling is a bound, not a divisibility proof -- 4
    divides 24 but not 20 -- so the heads are checked either way.

    A pin is not by itself unknown geometry. SFT reaches allocation with a revision ALWAYS resolved
    to a sha (``runner.submit.prepare_job`` -> ``_resolve_model_revision`` with ``required=True``),
    so treating "pinned" as "uncertifiable" capped every SFT run in the catalog at four cards and
    made ``--gpus 8`` unreachable for the algorithm that always pins -- including for a run that
    only fits at eight. The pinned commit's own ``config.json`` is what settles it: read the head
    count from that commit (validated against the catalog row, fail-closed, so a drifted pin is
    rejected rather than widened) and cap on the real number. An unreadable pin certifies nothing:
    it keeps the four-card ceiling AND falls back to the row's own head count for the divisor
    search, so it can only ever be narrower than the same run unpinned, never wider.

    ``certify`` is what permits the hub round trip, and ONLY the submission path passes it. Reading a
    pinned commit's config is network i/o, so a transient hub failure returns the uncertified
    four-card ceiling. On the submit path that is a safe conservative answer the allocator can still
    act on. On an OFFLINE path it is not: `spec_from_dict` feeds this cap to `provisional_gpu`, whose
    job is to REJECT an unplaceable run, so a blip would narrow a 35B that genuinely needs eight
    cards down to four and reject it as unplaceable during config parsing that is otherwise entirely
    offline -- turning a transient network error into a terminal, and wrong, user-facing rejection.
    The cost quote has the same shape (`_offline_gpu_shape` is documented as structural and must not
    consume live failures). Both keep the default and stay offline; certification belongs where a
    healthy retry and a real allocation decision live.
    """
    from flash.core.catalog import MODELS

    cap = largest_rentable_count(max_gpu_count)
    info = MODELS.get(model_id)
    heads = _query_attention_heads(info) if info is not None else 0
    if info is None:
        # nothing to certify a width against, and nothing to cross-check a pin's own config with.
        cap = min(cap, _UNCERTIFIED_CAP)
    elif certify and model_revision and cap > _UNCERTIFIED_CAP:
        # the weights the worker really loads are the pinned commit's, so its config -- not the
        # row's default-revision geometry -- is what may widen this run. only worth a hub round trip
        # when there is something to widen TO: at or below the uncertified cap, certification
        # cannot raise the ceiling.
        from flash.engine.plan.vram import certified_attention_heads

        certified = certified_attention_heads(model_id, model_revision)
        if certified > 0:
            heads = certified
        else:
            # uncertified: fall back to the ceiling, but keep checking the ROW's heads below. the
            # ceiling narrows the divisor search, it does not replace it, so a row whose heads do
            # not divide it is still narrowed further instead of rented at a width verl rejects.
            cap = min(cap, _UNCERTIFIED_CAP)
    if heads <= 0:
        # geometry we cannot read is geometry we cannot certify, so a catalog row that records no
        # head count is treated exactly like an uncertifiable revision rather than trusted for 8.
        return min(cap, _UNCERTIFIED_CAP)
    for count in rentable_gpu_counts(cap):
        if heads % count == 0:
            return count
    return 1


def _query_attention_heads(info) -> int:
    """Query-attention head count for a catalog row, or 0 when the row does not record one.

    Read, never derived. ``hidden_size // head_dim`` looks like the head count and is not: these
    checkpoints decouple ``head_dim`` from that ratio, so the quotient is wrong for four of the six
    catalog rows (3.5-4B is 16 heads, not 2560/256 = 10; 0.8B is 8, not 4; 3.6-27B is 24, not 20;
    35B-A3B is 16, not 8). A cap computed from the quotient divides the wrong number -- it happened
    to stay conservative on today's catalog, but nothing makes that hold for the next row added.
    """
    return int(getattr(info, "num_attention_heads", 0) or 0)


def _resolve_exact_gpu(
    gpu_type: str,
    *,
    need: float,
    cap: int,
    max_gpu_count: int,
    provider: str,
    available: tuple[str, ...],
    widest_cap: int = 1,
    unpinned: tuple[str, ...] | None = None,
    executed_width=None,
    algorithm: str = "",
) -> tuple[str, tuple[str, ...]]:
    """Validate an explicitly pinned GPU class and narrow ``available`` to providers that offer it.

    ``unpinned`` is the configured fleet before a provider pin narrowed ``available``, or ``None``
    when nothing was pinned. A pin can hide the only provider that rents this class at a wider
    count, and suppressing the remedy entirely would hide a fix the user can actually apply.

    ``executed_width`` is ``_executed_width``'s rule. Both the combination precheck below and every
    ``--gpus N`` remedy have to apply it, or a pinned sft run whose batch caps it at one rank is told
    to buy cards that cannot hold it. ``algorithm`` only words the reason for that narrowing, so it
    defaults to sft's phrasing rather than being required of every caller.
    """
    launched = executed_width or (lambda count: count)
    exact = canonical_gpu(gpu_type)
    exact_info = GPU_INFO.get(exact)
    if exact_info is None or not exact_info.validated:
        raise UnsupportedGpuError(f"exact GPU {exact!r} is not an active validated GPU class")
    # provider compatibility decides FIRST: a class the requested provider does not carry cannot be
    # rented at any width, so reporting a fit failure (and a `--gpus N` remedy) for it would send
    # the user to widen a shape that will never exist. the real defect is the pin/provider pair.
    exact_providers = providers_for(exact)
    if provider and provider not in exact_providers:
        raise UnsupportedGpuError(f"provider {provider!r} cannot provision exact GPU {exact!r}")
    reachable = tuple(name for name in available if name in exact_providers)
    # a card ceiling is the user's own `[gpu] count`, so a pin that fits at a wider rentable shape
    # is one flag from working; `above` is the width already tried, so the remedy only ever names
    # a wider one. bounded by the model's geometry cap so the suggestion is a width verl accepts
    # rather than one it rejects after the box is rented. only offered when a provider still in
    # play rents counts freely -- a Lambda/Vast-only pin has no offline proof the wider SKU exists.
    widths = (exact_info.vram_gb,) if rents_arbitrary_card_counts(reachable) else ()
    # the pin may be the only reason no width is offerable: this class can be carried by a provider
    # the pin excluded that DOES rent counts freely. saying so beats a bare shortfall the user
    # cannot act on. computed from the pre-pin fleet, so it stays silent when there is no pin.
    unpinned_reachable = (
        tuple(name for name in unpinned if name in exact_providers) if unpinned else ()
    )
    unpinned_widths = (
        (exact_info.vram_gb,)
        if not widths and unpinned_reachable and rents_arbitrary_card_counts(unpinned_reachable)
        else ()
    )
    # a class nobody configured can provision is blocked by the CONFIGURATION, not by its VRAM, and
    # no width or knob change can move that. reporting the shortfall first would answer a question
    # the user never reached: the same fleet with a smaller need already fails on reachability in
    # `allocate()`, so deciding it here keeps one root cause from producing two different errors
    # depending on how large the run happens to be.
    if not (reachable or unpinned_reachable):
        raise UnsupportedGpuError(f"exact GPU {exact!r} has no configured active provider")

    def _drop_pin_hint(above: int) -> str:
        return drop_pin_hint(
            exact,
            unpinned_widths,
            need,
            ceiling=widest_cap,
            above=above,
            executed_width=launched,
        )

    def _catalog_check_hint(above: int) -> str:
        return catalog_check_hint(
            exact,
            need,
            ceiling=widest_cap,
            above=above,
            # a freely-rentable width was already provable, or nothing carries this class: either
            # way the catalog question is not the one to ask. see `catalog_check_hint`.
            offerable=not (widths or unpinned_widths) and bool(reachable or unpinned_reachable),
            validated=exact_info.validated,
            executed_width=launched,
        )

    if exact_info.vram_gb < need and max_gpu_count <= 1:
        raise UnsupportedGpuError(
            f"exact GPU {exact!r} has {exact_info.vram_gb} GB VRAM, "
            f"but this run requires at least {need} GB"
            + (
                wider_shape_remedy(
                    widths, need, ceiling=widest_cap, above=1, executed_width=launched
                )
                or _drop_pin_hint(1)
                or _catalog_check_hint(1)
            )
        )
    # the widest shape providers actually rent for this ceiling, not the ceiling itself: a pin
    # that only fits at a non-rentable count (3) must be rejected here with a precise reason
    # rather than passing and dying later on a generic no-capacity error.
    if (
        exact_info.vram_gb < need
        and max_gpu_count > 1
        and combined_vram_gb(exact_info.vram_gb, launched(cap)) < need
    ):
        # the width the VRAM math above actually credited, which is what the message has to name.
        # naming `cap` instead points the operator at the card ceiling when the real limiter is the
        # batch: they raise `--gpus` and hit the identical failure.
        tried = launched(cap)
        raise UnsupportedGpuError(
            f"exact GPU {exact!r} cannot fit this run even as a {tried}-card combination"
            + (
                f" (of the {cap} cards allowed, only {tried} "
                f"{'joins' if tried == 1 else 'join'} this run"
                # the reason comes from the shared formatter rather than being spelled again here:
                # this copy said "sft" unconditionally, so a pinned small-batch grpo/opd run was sent
                # to a batch knob opd rejects at parse time and rows rl does not have. the dash form,
                # not the parenthetical: this clause is already inside parens.
                f"{batch_bound_width_note(algorithm=algorithm)})"
                if tried != cap
                else ""
            )
            + (
                wider_shape_remedy(
                    widths, need, ceiling=widest_cap, above=cap, executed_width=launched
                )
                or _drop_pin_hint(cap)
                or _catalog_check_hint(cap)
            )
        )
    return exact, reachable


def _narrow_to_pinned_gpus(
    gpu_type: str,
    gpu_type_fallbacks: tuple[str, ...],
    *,
    model_id: str,
    algorithm: str,
    need: float,
    max_gpu_count: int,
    model_revision: str,
    provider: str,
    available: tuple[str, ...],
    unpinned: tuple[str, ...] | None,
    executed_width,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Validate every acceptable class and narrow the provider search."""
    cap = geometry_safe_gpu_cap(
        model_id, max_gpu_count, model_revision=model_revision, certify=True
    )
    widest_cap = geometry_safe_gpu_cap(
        model_id, MAX_COMBINATION_CARDS, model_revision=model_revision, certify=True
    )
    resolved = [
        _resolve_exact_gpu(
            candidate,
            need=need,
            cap=cap,
            max_gpu_count=max_gpu_count,
            provider=provider,
            available=available,
            widest_cap=widest_cap,
            unpinned=unpinned,
            executed_width=executed_width,
            algorithm=algorithm,
        )
        for candidate in (gpu_type, *gpu_type_fallbacks)
    ]
    acceptable = tuple(exact for exact, _providers in resolved)
    reachable = {name for _exact, providers in resolved for name in providers}
    available = tuple(name for name in available if name in reachable)
    exact = acceptable[0] if len(acceptable) == 1 else ""
    return exact, acceptable, available


def _structural_gpu_names(available: tuple[str, ...], exact: str) -> tuple[str, ...]:
    """Validated classes the requested provider set can structurally provision."""
    return tuple(
        info.name
        for info in GPU_INFO.values()
        if info.validated
        and (not exact or info.name == exact)
        and any(provider in available for provider in providers_for(info.name))
    )


def _resolved_gpu_count(
    model_id: str,
    algorithm: str,
    *,
    need: float,
    requested_gpu_count: int | None,
    model_revision: str,
    available: tuple[str, ...],
    exact: str,
    unpinned: tuple[str, ...] | None = None,
    executed_width=None,
) -> int:
    """Resolve auto-size or validate that an authored ceiling can structurally fit.

    ``unpinned`` is the configured fleet before a provider pin narrowed ``available``, or ``None``
    when nothing was pinned. A rejection needs it to decide whether dropping the pin is a remedy:
    ``available`` alone cannot tell a pin from a plane that only ever configured one provider, and
    a pin on such a plane drops to the same pool and the same failure.

    ``executed_width`` (``_executed_width``) decides both the auto-sized ceiling and the
    authored-ceiling check on the width that will run. Without it an sft run the batch caps at one
    rank is told to raise `[gpu] count` to a width that changes nothing.

    Certifies the pin (``certify=True``): this runs inside ``allocate()``, which already does
    network i/o and can retry, and the width decided here is the one the run is really rented at. A
    hub failure degrades to the conservative ceiling rather than raising, so an outage can only
    narrow the shape, never reject a run that fits. The offline parse and quote paths deliberately
    do NOT certify -- see ``geometry_safe_gpu_cap``.
    """
    auto_cap = geometry_safe_gpu_cap(
        model_id, MAX_COMBINATION_CARDS, model_revision=model_revision, certify=True
    )
    gpu_names = _structural_gpu_names(available, exact)
    if requested_gpu_count is None:
        fitting_count = smallest_fitting_gpu_count(
            need, max_gpu_count=auto_cap, gpu_names=gpu_names, executed_width=executed_width
        )
        if fitting_count is not None:
            return fitting_count
        effective_count = auto_cap
    else:
        effective_count = geometry_safe_gpu_cap(
            model_id, requested_gpu_count, model_revision=model_revision, certify=True
        )
        if (
            smallest_fitting_gpu_count(
                need,
                max_gpu_count=effective_count,
                gpu_names=gpu_names,
                executed_width=executed_width,
            )
            is not None
        ):
            return effective_count
    raise UnsupportedGpuError(
        vram_fit_error_message(
            algorithm,
            need,
            requested_gpu_count=requested_gpu_count,
            effective_gpu_count=effective_count,
            max_gpu_count=auto_cap,
            gpu_names=gpu_names,
            providers=available,
            executed_width=executed_width,
            # the pin is worth dropping only if the fleet behind it still buys a wider shape, so
            # ask the same question of the unpinned pool that the pinned one just failed.
            widenable_without_pin=(
                None
                if unpinned is None
                else widenable_gpu_names(_structural_gpu_names(unpinned, exact), unpinned)
            ),
        )
    )


def _gather_candidates(
    available: tuple[str, ...],
    *,
    per_card_need: float,
    constraints: AllocationConstraints,
    acceptable: tuple[str, ...],
    provider: str,
) -> tuple[list[Candidate], bool, dict[str, UnsupportedGpuError]]:
    """Query every available provider for fitting shapes.

    Returns ``(candidates, lookup_failed, structurally_unsupported)``. The two failure records are
    what let an empty result be told apart from a genuine no-fit.

    ``acceptable`` is the authoritative class filter; empty means unpinned.
    """
    allowed = set(acceptable)
    queries = tuple(replace(constraints, gpu_type=gpu) for gpu in acceptable) or (constraints,)
    candidates: list[Candidate] = []
    lookup_failed = False
    structurally_unsupported: dict[str, UnsupportedGpuError] = {}
    for name in available:
        unsupported: list[UnsupportedGpuError] = []
        for query in queries:
            try:
                found = get_provider(name).live_candidates(per_card_need, query)
                candidates += [
                    candidate
                    for candidate in found
                    if candidate.provider == name and (not allowed or candidate.gpu in allowed)
                ]
            except UnsupportedGpuError as exc:
                unsupported.append(exc)
            except CapacityLookupError as exc:
                lookup_failed = True
                logger.warning(
                    "%s capacity lookup failed (%s); allocating without it", name, exc.__cause__
                )
        if len(unsupported) == len(queries):
            if provider:
                raise unsupported[0]
            structurally_unsupported[name] = unsupported[0]
            logger.info(
                "%s cannot offer any acceptable shape (%s); trying other providers",
                name,
                unsupported[0],
            )
    return list(dict.fromkeys(candidates)), lookup_failed, structurally_unsupported


def _raise_no_candidate_error(
    *,
    model_id: str,
    need: float,
    cap: int,
    exact: str,
    acceptable: tuple[str, ...],
    supported_available: tuple[str, ...],
    structurally_unsupported: dict[str, UnsupportedGpuError],
    lookup_failed: bool,
    executed_width,
) -> NoReturn:
    """Classify an empty candidate set as retryable capacity or a terminal structural miss.

    Retryable means "the shape exists but is sold out", so both branches below ask about the shape
    the run would EXECUTE (``_executed_width``): a width the filter deterministically rejects is a
    dead end, and calling it retryable spends the infra budget re-polling for capacity that would
    not help.
    """
    if not supported_available and structurally_unsupported:
        # Every configured provider rejected the shape structurally. Surface one provider's
        # concrete reason rather than misclassifying an impossible SKU as temporary capacity.
        raise next(iter(structurally_unsupported.values()))
    # THE precondition for every retryable answer below: a shape the run could execute on is
    # advertised somewhere, so "come back later" is a coherent thing to say. Evaluated once -- three
    # copies of this question is what let the sft width clamp reach one branch and not the others.
    could_fit = _structurally_fits(supported_available, need, cap, executed_width, acceptable)
    if lookup_failed and could_fit:
        # No candidate fit, but a live capacity lookup blipped and was the only possible source of one
        # -> retryable, NOT terminal: a Vast/Lambda-only run must ride out a market/API outage on its
        # infra budget instead of dying as if the job exceeds every GPU class. The class list stays
        # readable during the outage (it is static), so `could_fit` keeps a run the blip did not
        # cause and cannot cure from burning that budget.
        raise CapacityLookupError(
            f"no allocatable GPU (>= {need} GB VRAM for {model_id}): a provider's live capacity lookup "
            f"failed transiently and was the only source of a fitting class — retry may find hidden capacity"
        )
    # a provider whose capacity comes from a live market can be structurally able to rent a
    # shape while having none free right now (retryable), unlike one priced off a static table
    # where "no candidate" means the shape genuinely is not offered (terminal). this applies to
    # an unpinned search too: sold out is sold out whether or not the user named the class.
    live_only = bool(supported_available) and all(
        getattr(get_provider(name), "live_capacity", False) for name in supported_available
    )
    if exact:
        if live_only and could_fit:
            raise CapacityUnavailableError(
                f"exact GPU {exact!r} is structurally supported but currently has no capacity on "
                f"{', '.join(supported_available)}"
            )
        raise UnsupportedGpuError(
            f"exact GPU {exact!r} has no allocatable capacity on the requested active provider set "
            f"({', '.join(supported_available) or '(none)'})"
        )
    # unpinned: only retryable when SOME structurally-offered shape could have held the run.
    # without that guard a genuinely oversized run would retry until its infra budget ran out
    # instead of failing immediately with the reason.
    if live_only and could_fit:
        raise CapacityUnavailableError(
            f"no allocatable GPU (>= {need} GB VRAM for {model_id}) right now: a fitting class is "
            f"structurally offered on {', '.join(supported_available)} but has no capacity — "
            f"retry may find it"
        )
    raise UnsupportedGpuError(
        f"no allocatable GPU (>= {need} GB VRAM for {model_id}) on any available provider "
        f"({', '.join(supported_available) or '(none)'}); the run genuinely exceeds every "
        f"active GPU class"
    )


def allocate(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
    disk_gb: float = 0.0,
    max_wall_seconds: float = 0.0,
    provider: str = "",
    providers: tuple[str, ...] = (),
    gpu_type: str = "",
    gpu_type_fallbacks: tuple[str, ...] = (),
    model_revision: str = "",
    max_gpu_count: int | None = None,
    overrides: dict | None = None,
) -> Allocation:
    """Pick the cheapest fitting combination of (provider, GPU class, count) able to run the job.

    ``max_gpu_count=None`` auto-sizes to the smallest geometry-safe ceiling that can fit. an integer
    is an authored hard ceiling; fitting shapes up to that ceiling still compete on dollars per step.

    ``gpu_type`` and its fallbacks restrict the search without changing cheapest-cost ranking.
    """
    # the same profile knobs ranking prices on: VRAM must be sized for the work that will RUN, not
    # the authored request. an exact-unpacked run executes batch 1 at the measured length, so sizing
    # off the authored batch over-reserves (4 GB on a 4B at batch 8 / 4096) and can reject a card the
    # run would have fit on. `_fits` reads it too, so the requirement and the fit cannot disagree.
    sized_train = _overridden_train(train, overrides)
    # bound before the pinned-class checks below, which need it too: every question about a card
    # count in this function is asked through this one rule. see `_executed_width`.
    executed_width = _executed_width(algorithm, sized_train, overrides)
    need = required_vram_gb(
        model_id,
        algorithm,
        train=sized_train,
        thinking=thinking,
        model_revision=model_revision,
    )
    provider = (provider or "").strip().lower()
    if provider and provider not in PROVIDER_NAMES:
        raise UnsupportedGpuError(
            f"unknown provider {provider!r}; known providers: {', '.join(PROVIDER_NAMES)}"
        )
    try:
        providers = validated_provider_preferences(
            providers, allow_empty=isinstance(providers, tuple)
        )
    except (TypeError, ValueError) as exc:
        raise UnsupportedGpuError(str(exc)) from exc
    if provider and providers:
        raise UnsupportedGpuError("provider and providers cannot both be set")
    available = available_providers()
    # kept across the narrowing below so a rejection can ask what dropping the pin would restore.
    unpinned = None
    if provider:
        if provider not in available:
            raise UnsupportedGpuError(f"requested provider {provider!r} is not configured")
        unpinned = available
        available = (provider,)

    # allocate() is reachable directly, bypassing the parse gate entirely, so it resolves the
    # author's ceiling from the same shared predicate rather than trusting a caller to have applied
    # it. this has to precede the pinned-fit checks below, which read the ceiling to decide whether
    # a wider shape would fix the run. see `authored_gpu_ceiling` for what a bare pin means and why.
    max_gpu_count = authored_gpu_ceiling(gpu_type, max_gpu_count)
    exact = ""
    acceptable: tuple[str, ...] = ()
    if gpu_type:
        assert max_gpu_count is not None
        exact, acceptable, available = _narrow_to_pinned_gpus(
            gpu_type,
            gpu_type_fallbacks,
            model_id=model_id,
            algorithm=algorithm,
            need=need,
            max_gpu_count=max_gpu_count,
            model_revision=model_revision,
            provider=provider,
            available=available,
            unpinned=unpinned,
            executed_width=executed_width,
        )
    cap = _resolved_gpu_count(
        model_id,
        algorithm,
        need=need,
        requested_gpu_count=max_gpu_count,
        model_revision=model_revision,
        available=available,
        exact=exact,
        unpinned=unpinned,
        executed_width=executed_width,
    )

    constraints = AllocationConstraints(
        disk_gb=disk_gb,
        max_wall_seconds=max_wall_seconds,
        gpu_type=exact,
        required_vram_gb=need,
        max_gpu_count=cap,
    )
    # with multiple cards allowed, a card can contribute as one of N; query providers at the reduced
    # per-card floor so classes too small to fit alone still surface, then keep only the shapes that
    # actually fit below.
    per_card_need = need
    if cap > 1:
        per_card_need = max(1, math.ceil(need / (cap * SHARD_VRAM_EFFICIENCY)))
    candidates, lookup_failed, structurally_unsupported = _gather_candidates(
        available,
        per_card_need=per_card_need,
        constraints=constraints,
        acceptable=acceptable,
        provider=provider,
    )
    candidates = _fitting_candidates(candidates, need, executed_width)
    supported_available = tuple(name for name in available if name not in structurally_unsupported)
    if not candidates:
        _raise_no_candidate_error(
            model_id=model_id,
            need=need,
            cap=cap,
            exact=exact,
            acceptable=acceptable,
            supported_available=supported_available,
            structurally_unsupported=structurally_unsupported,
            lookup_failed=lookup_failed,
            executed_width=executed_width,
        )
    cost_per_step = _step_cost_ranker(
        model_id, algorithm, train, thinking, model_revision, overrides
    )
    provider_rank = {name: rank for rank, name in enumerate(providers)}
    return _cheapest_allocation(
        candidates,
        need=need,
        cost_per_step=cost_per_step,
        provider_rank=provider_rank,
    )


def _cheapest_allocation(
    candidates, *, need: float, cost_per_step, provider_rank: dict[str, int]
) -> Allocation:
    """The cheapest-JOB shape from a non-empty fitting set, plus the full ranking behind it.

    Cheapest job, not cheapest rental: rank on the dollars one step costs on each candidate (rate x
    how long that hardware takes), so a faster card wins whenever it finishes enough sooner to pay
    for itself. An authored provider preference ranks ahead of every cost key; unnamed providers
    share the final rank, so they remain eligible and retain their relative cost order. Within one
    provider rank, ties prefer fewer cards (less inter-card overhead), then combined VRAM, then class
    name. Sorting is stable, so provider and provider-local order apply only when every explicit key
    matches. A run the cost model cannot price (``cost_per_step`` is ``None``) falls back to total
    $/hr.
    """
    primary = cost_per_step if cost_per_step is not None else (lambda c: c.total_hourly_usd)
    unnamed_rank = len(provider_rank)
    ranked = sorted(
        candidates,
        key=lambda c: (
            provider_rank.get(c.provider, unnamed_rank),
            primary(c),
            c.total_hourly_usd,
            c.gpu_count,
            c.total_vram_gb,
            c.gpu,
        ),
    )
    best = ranked[0]
    return Allocation(
        provider=best.provider,
        gpu=best.gpu,
        hourly_usd=best.hourly_usd,
        min_vram_gb=need,
        candidates=tuple(ranked),
        gpu_count=best.gpu_count,
    )


def allocation_summary(a: Allocation) -> str:
    shape = f"{a.gpu_count}x {a.gpu}" if a.gpu_count > 1 else a.gpu
    total = a.gpu_count * a.hourly_usd
    head = f"allocated {shape} on {a.provider} at ${total:.2f}/hr (need >= {a.min_vram_gb} GB VRAM)"
    if len(a.candidates) > 1:
        nxt = a.candidates[1]
        nxt_shape = f"{nxt.gpu_count}x {nxt.gpu}" if nxt.gpu_count > 1 else nxt.gpu
        head += f"; next-best: {nxt_shape}@{nxt.provider} ${nxt.total_hourly_usd:.2f}/hr"
    return head

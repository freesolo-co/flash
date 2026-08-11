"""GPU allocation: cheapest fitting (provider, GPU class) across active providers."""

from __future__ import annotations

import math
from typing import NoReturn

from flash._internal.logging import get_logger
from flash.providers import PROVIDER_NAMES, available_providers, get_provider
from flash.providers.base import (
    GPU_INFO,
    SHARD_VRAM_EFFICIENCY,
    Allocation,
    AllocationConstraints,
    Candidate,
    CapacityLookupError,
    UnsupportedGpuError,
    _run_cost_key,
    canonical_gpu,
    combined_vram_gb,
    largest_rentable_count,
    providers_for,
    rentable_gpu_counts,
    run_config_for_ranking,
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


def profile_required_vram_gb() -> int:
    """VRAM a workload-profile job needs: none beyond the smallest rentable card.

    A profile job renders and tokenizes the exact dataset on cpu and exits before model weights or
    cuda are touched, so sizing it like the training run it measures would rent (and bill) a card the
    work never uses.
    """
    return 1


def _profile_cost_ranker():
    """``candidate -> dollars for the profile job``, which is rate alone.

    The profile's wall is a fixed cap rather than a function of the hardware, so no card finishes it
    sooner and the cheapest rentable shape always wins.
    """
    return lambda candidate: candidate.total_hourly_usd


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


def _fits(candidate: Candidate, need: int) -> bool:
    """Whether this rentable shape can actually hold the run.

    ``combined_vram_gb`` is the shared fit model, so a shape accepted here is a shape parse-time
    sizing also accepted -- the two must not be able to disagree.
    """
    return combined_vram_gb(candidate.vram_gb, candidate.gpu_count) >= need


def _structurally_fits(available, need: int, cap: int) -> bool:
    """Whether any provider OFFERS a class that could hold the run, ignoring current stock.

    Separates "sold out right now" (retryable) from "no such shape exists" (terminal) for an
    unpinned search. Reads each provider's advertised class list, never live capacity, so it stays
    truthful during the very outage it is called to interpret.
    """
    for name in available:
        try:
            classes = get_provider(name).gpu_classes()
        except Exception:  # a provider that cannot even list classes proves nothing either way
            continue
        for gpu_class in classes:
            for count in rentable_gpu_counts(cap):
                if combined_vram_gb(gpu_class.vram_gb, count) >= need:
                    return True
    return False


def geometry_safe_gpu_cap(model_id: str, max_gpu_count: int, *, model_revision: str = "") -> int:
    """Rentable ceiling whose sequence-parallel divisibility is known before paid allocation.

    The width becomes ``ulysses_sequence_parallel_size``, and verl requires
    ``num_attention_heads % sp_size == 0``, so a catalog row is only safe at the counts that divide
    its OWN head count. Curated membership is not uniform geometry: catalog head counts are 8, 8,
    16, 16, 24, and 16, so trusting membership alone accepted an 8-card width for the 27B (24 heads)
    that verl rejects at Ulysses init, after the box was already rented.

    The head count is READ from the row (``num_attention_heads``), never derived: ``hidden_size //
    head_dim`` is a different number on four of the six rows -- see ``_query_attention_heads``.

    A pinned or unreadable revision keeps the pre-existing four-card ceiling rather than renting 8
    cards verl may reject at startup, but that ceiling only NARROWS the divisor search; it is not a
    substitute for it. A ceiling is a bound, not a divisibility proof -- 4 divides 24 but not 20 --
    and SFT reaches allocation with the revision already resolved to a sha
    (``runner.submit.prepare_job`` -> ``_resolve_model_revision`` with ``required=True``), so a
    catalog row keyed on revision-emptiness alone would skip its own geometry on exactly the runs
    that need it. Match the row by id and check the heads either way.
    ALLOC-004 tracks validating arbitrary off-catalog head geometry at every width.
    """
    from flash.core.catalog import MODELS

    cap = largest_rentable_count(max_gpu_count)
    if model_revision or model_id not in MODELS:
        # an unvalidated revision keeps the pre-existing four-card ceiling, but that ceiling is a
        # BOUND, not a divisibility proof -- it happens to divide 24 and would not divide 20. So it
        # only narrows the search below, never substitutes for it.
        cap = min(cap, 4)
    info = MODELS.get(model_id)
    heads = _query_attention_heads(info) if info is not None else 0
    if heads <= 0:
        # geometry we cannot read is geometry we cannot certify, so a catalog row that records no
        # head count is treated exactly like an unvalidated revision rather than trusted for 8.
        return min(cap, 4)
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
) -> tuple[str, tuple[str, ...]]:
    """Validate an explicitly pinned GPU class and narrow ``available`` to providers that offer it."""
    exact = canonical_gpu(gpu_type)
    exact_info = GPU_INFO.get(exact)
    if exact_info is None or not exact_info.validated:
        raise UnsupportedGpuError(f"exact GPU {exact!r} is not an active validated GPU class")
    if exact_info.vram_gb < need and max_gpu_count <= 1:
        raise UnsupportedGpuError(
            f"exact GPU {exact!r} has {exact_info.vram_gb} GB VRAM, "
            f"but this run requires at least {need} GB"
        )
    # the widest shape providers actually rent for this ceiling, not the ceiling itself: a pin
    # that only fits at a non-rentable count (3) must be rejected here with a precise reason
    # rather than passing and dying later on a generic no-capacity error.
    if (
        exact_info.vram_gb < need
        and max_gpu_count > 1
        and combined_vram_gb(exact_info.vram_gb, cap) < need
    ):
        raise UnsupportedGpuError(
            f"exact GPU {exact!r} cannot fit this run even as a {cap}-card combination"
        )
    exact_providers = providers_for(exact)
    if provider and provider not in exact_providers:
        raise UnsupportedGpuError(f"provider {provider!r} cannot provision exact GPU {exact!r}")
    return exact, tuple(name for name in available if name in exact_providers)


def _gather_candidates(
    available: tuple[str, ...],
    *,
    per_card_need: float,
    constraints: AllocationConstraints,
    exact: str,
    provider: str,
) -> tuple[list[Candidate], bool, dict[str, UnsupportedGpuError]]:
    """Query every available provider for fitting shapes.

    Returns ``(candidates, lookup_failed, structurally_unsupported)``. The two failure records are
    what let an empty result be told apart from a genuine no-fit.
    """
    candidates: list[Candidate] = []
    lookup_failed = False
    structurally_unsupported: dict[str, UnsupportedGpuError] = {}
    # runpod prices off a static table (no live lookup), so it never blips; lambda/vast query live
    # capacity and can. a per-provider blip degrades to the others (we just skip it), but we remember it
    # so an empty result can be told apart from a genuine no-fit below.
    # runpod uses the same loop but does not raise CapacityLookupError.
    for name in available:
        try:
            found = get_provider(name).live_candidates(per_card_need, constraints)
            candidates += [
                candidate
                for candidate in found
                if candidate.provider == name and (not exact or candidate.gpu == exact)
            ]
        except UnsupportedGpuError as exc:
            # A count-specific SKU miss is provider-local during an automatic search. Lambda may not
            # sell 8x H100 while RunPod or Vast does; aborting here discards candidates already found
            # elsewhere. An explicitly selected provider still fails immediately with its precise
            # structural reason.
            if provider:
                raise
            structurally_unsupported[name] = exc
            logger.info("%s cannot offer this shape (%s); trying other providers", name, exc)
        except CapacityLookupError as exc:
            lookup_failed = True
            logger.warning(
                "%s capacity lookup failed (%s); allocating without it", name, exc.__cause__
            )
    return candidates, lookup_failed, structurally_unsupported


def _raise_no_candidate_error(
    *,
    model_id: str,
    need: float,
    cap: int,
    exact: str,
    supported_available: tuple[str, ...],
    structurally_unsupported: dict[str, UnsupportedGpuError],
    lookup_failed: bool,
) -> NoReturn:
    """Classify an empty candidate set as retryable capacity or a terminal structural miss."""
    if not supported_available and structurally_unsupported:
        # Every configured provider rejected the shape structurally. Surface one provider's
        # concrete reason rather than misclassifying an impossible SKU as temporary capacity.
        raise next(iter(structurally_unsupported.values()))
    if lookup_failed:
        # No candidate fit, but a live capacity lookup blipped and was the only possible source of one
        # -> retryable, NOT terminal: a Vast/Lambda-only run must ride out a market/API outage on its
        # infra budget instead of dying as if the job exceeds every GPU class.
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
        if live_only:
            raise CapacityLookupError(
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
    if live_only and _structurally_fits(supported_available, need, cap):
        raise CapacityLookupError(
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
    gpu_type: str = "",
    model_revision: str = "",
    max_gpu_count: int = 1,
    workload_profile: bool = False,
    overrides: dict | None = None,
) -> Allocation:
    """Pick the cheapest fitting combination of (provider, GPU class, count) able to run the job.

    With ``max_gpu_count=1`` (the default) this is exactly the classic cheapest single-class
    allocation. A caller whose algorithm can shard across cards passes a higher cap, and fitting
    multi-card combinations (same class x count) then compete on TOTAL hourly cost — e.g.
    2 x A100 beats 1 x H200 whenever 2 * $A100 < $H200 and the sharded fit clears the need.

    ``workload_profile=True`` allocates the cpu-only profile job instead of the run it measures: it
    needs no training VRAM and gains nothing from a faster card, so it ranks on rate alone.
    """
    if workload_profile:
        need = profile_required_vram_gb()
        max_gpu_count = 1
    else:
        need = required_vram_gb(
            model_id,
            algorithm,
            train=train,
            thinking=thinking,
            model_revision=model_revision,
        )
    provider = (provider or "").strip().lower()
    if provider and provider not in PROVIDER_NAMES:
        raise UnsupportedGpuError(
            f"unknown provider {provider!r}; known providers: {', '.join(PROVIDER_NAMES)}"
        )
    available = available_providers()
    if provider:
        if provider not in available:
            raise UnsupportedGpuError(f"requested provider {provider!r} is not configured")
        available = (provider,)

    cap = geometry_safe_gpu_cap(model_id, max_gpu_count, model_revision=model_revision)
    exact = ""
    if gpu_type:
        exact, available = _resolve_exact_gpu(
            gpu_type,
            need=need,
            cap=cap,
            max_gpu_count=max_gpu_count,
            provider=provider,
            available=available,
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
        exact=exact,
        provider=provider,
    )
    # providers report the shapes they can genuinely rent (RunPod takes a count, Lambda names it in
    # the instance type, Vast bakes it into the offer); the allocator owns only whether a shape fits.
    candidates = [c for c in candidates if _fits(c, need)]
    supported_available = tuple(name for name in available if name not in structurally_unsupported)
    if not candidates:
        _raise_no_candidate_error(
            model_id=model_id,
            need=need,
            cap=cap,
            exact=exact,
            supported_available=supported_available,
            structurally_unsupported=structurally_unsupported,
            lookup_failed=lookup_failed,
        )
    # cheapest JOB first, not cheapest rental: rank on the dollars one step costs on each candidate
    # (rate x how long that hardware takes), so a faster card wins whenever it finishes enough sooner
    # to pay for itself. ties prefer fewer cards (less inter-card overhead), then combined VRAM, then
    # class name. sorting is stable, so provider and provider-local order apply only when all key
    # fields match. a run the cost model cannot price falls back to total $/hr.
    cost_per_step = (
        _profile_cost_ranker()
        if workload_profile
        else _step_cost_ranker(model_id, algorithm, train, thinking, model_revision, overrides)
    )
    primary = cost_per_step if cost_per_step is not None else (lambda c: c.total_hourly_usd)
    ranked = sorted(
        candidates,
        key=lambda c: (primary(c), c.total_hourly_usd, c.gpu_count, c.total_vram_gb, c.gpu),
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

"""GPU allocation: cheapest fitting (provider, GPU class) across active providers."""

from __future__ import annotations

import math

from flash._logging import get_logger
from flash.providers import PROVIDER_NAMES, available_providers, get_provider
from flash.providers.base import (
    GPU_INFO,
    Allocation,
    AllocationConstraints,
    Candidate,
    CapacityLookupError,
    UnsupportedGpuError,
    _run_cost_key,
    canonical_gpu,
    providers_for,
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
    from flash.engine.vram import model_required_vram_gb

    return model_required_vram_gb(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        headroom=vram_headroom(),
        model_revision=model_revision,
    )


# FSDP shards parameters/optimizer across cards but replicates activations and adds collective
# buffers; a combination only counts as fitting when the combined VRAM clears the need with this
# margin. conservative on purpose: a too-optimistic shard model OOMs a paid run.
_SHARD_VRAM_EFFICIENCY = 0.85
# activations/buffers replicate PER CARD regardless of sharding; every card in a combination must
# individually hold this floor on top of its parameter shard, so tiny cards cannot fake a fit by
# count alone (e.g. 4 x 12 GB is not 40 GB of usable capacity for a 24 GB-activation job).
_REPLICATED_PER_CARD_GB = 8
# combinations larger than this are never proposed: shard-efficiency loss and inter-card overhead
# grow with count, and cost estimates get less reliable.
_MAX_COMBINATION_CARDS = 4


def _step_cost_ranker(model_id, algorithm, train, thinking, model_revision=""):
    """``candidate -> dollars for one optimizer step``, or None when the run cannot be priced.

    Wraps the shared per-step cost key with the multi-card speedup, which is the one thing the
    single-card paths (parse-time provisional, cost estimate) have no notion of: a combination
    bills every card but only the gpu-bound half of a step is divided among them.
    """
    cost_key = _run_cost_key(
        model_id, algorithm, train=train, thinking=thinking, model_revision=model_revision
    )
    if cost_key is None:
        logger.debug("total-cost ranking unavailable; ranking on $/hr")
        return None

    from flash.cost.analytical import seconds_per_step

    # the same one-step config the cost key was built from, so the single- and multi-card branches
    # below cannot price a run off different knobs.
    config = run_config_for_ranking(
        model_id, algorithm, train=train, thinking=thinking, model_revision=model_revision
    )

    def cost_per_step(candidate: Candidate) -> float:
        if candidate.gpu_count <= 1:
            # identical to the single-card key the preview and estimate use, so the three paths
            # agree exactly whenever one card is enough.
            return cost_key(candidate.gpu, candidate.hourly_usd)
        # scaling is per-class: an nvlink pair keeps ~0.88 of linear per card, a pcie pair ~0.71.
        # this is the same call the quote prices the chosen combination with, so ranking and
        # billing cannot disagree about what a second card buys.
        seconds = seconds_per_step(config, candidate.gpu, candidate.gpu_count)
        return candidate.total_hourly_usd * seconds / 3600.0

    return cost_per_step


def _combination_candidates(
    singles: list[Candidate], need: int, max_gpu_count: int
) -> list[Candidate]:
    """Expand single-card candidates into fitting multi-card combinations (same class x count).

    Per-card fields stay per-card; a combination fits when count * vram * _SHARD_VRAM_EFFICIENCY
    covers the need. Only the smallest fitting count per class is proposed (larger counts of the
    same class only cost more).
    """
    combos: list[Candidate] = []
    for c in singles:
        if c.vram_gb <= _REPLICATED_PER_CARD_GB:
            continue  # card too small to hold the replicated working set at all
        for count in range(2, min(max_gpu_count, _MAX_COMBINATION_CARDS) + 1):
            usable = count * (c.vram_gb - _REPLICATED_PER_CARD_GB) * _SHARD_VRAM_EFFICIENCY
            if usable + _REPLICATED_PER_CARD_GB >= need:
                combos.append(
                    Candidate(
                        provider=c.provider,
                        gpu=c.gpu,
                        hourly_usd=c.hourly_usd,
                        vram_gb=c.vram_gb,
                        gpu_count=count,
                    )
                )
                break
    return combos


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
) -> Allocation:
    """Pick the cheapest fitting combination of (provider, GPU class, count) able to run the job.

    With ``max_gpu_count=1`` (the default) this is exactly the classic cheapest single-class
    allocation. A caller whose algorithm can shard across cards passes a higher cap, and fitting
    multi-card combinations (same class x count) then compete on TOTAL hourly cost — e.g.
    2 x A100 beats 1 x H200 whenever 2 * $A100 < $H200 and the sharded fit clears the need.
    """
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

    exact = ""
    if gpu_type:
        exact = canonical_gpu(gpu_type)
        exact_info = GPU_INFO.get(exact)
        if exact_info is None or not exact_info.validated:
            raise UnsupportedGpuError(f"exact GPU {exact!r} is not an active validated GPU class")
        if exact_info.vram_gb < need and max_gpu_count <= 1:
            raise UnsupportedGpuError(
                f"exact GPU {exact!r} has {exact_info.vram_gb} GB VRAM, "
                f"but this run requires at least {need} GB"
            )
        _max_cards = min(max_gpu_count, _MAX_COMBINATION_CARDS)
        _pin_usable = (
            _max_cards
            * max(0, exact_info.vram_gb - _REPLICATED_PER_CARD_GB)
            * _SHARD_VRAM_EFFICIENCY
            + _REPLICATED_PER_CARD_GB
        )
        if exact_info.vram_gb < need and max_gpu_count > 1 and _pin_usable < need:
            raise UnsupportedGpuError(
                f"exact GPU {exact!r} cannot fit this run even as a "
                f"{min(max_gpu_count, _MAX_COMBINATION_CARDS)}-card combination"
            )
        exact_providers = providers_for(exact)
        if provider and provider not in exact_providers:
            raise UnsupportedGpuError(f"provider {provider!r} cannot provision exact GPU {exact!r}")
        available = tuple(name for name in available if name in exact_providers)

    constraints = AllocationConstraints(
        disk_gb=disk_gb,
        max_wall_seconds=max_wall_seconds,
        gpu_type=exact,
    )
    allow_combos = max_gpu_count > 1
    # with combinations allowed, a card can contribute as one of N; query providers at the reduced
    # per-card floor so classes too small to fit alone still surface, then split fit-alone singles
    # from combination material below.
    per_card_need = need
    if allow_combos:
        cap = min(max_gpu_count, _MAX_COMBINATION_CARDS)
        per_card_need = max(1, math.ceil(need / (cap * _SHARD_VRAM_EFFICIENCY)))
    candidates: list[Candidate] = []
    lookup_failed = False
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
        except CapacityLookupError as exc:
            lookup_failed = True
            logger.warning(
                "%s capacity lookup failed (%s); allocating without it", name, exc.__cause__
            )
    if allow_combos:
        fit_alone = [c for c in candidates if c.vram_gb >= need]
        undersized = [c for c in candidates if c.vram_gb < need]
        candidates = fit_alone + _combination_candidates(undersized, need, max_gpu_count)
    if not candidates:
        if lookup_failed:
            # No candidate fit, but a live capacity lookup blipped and was the only possible source of one
            # -> retryable, NOT terminal: a Vast/Lambda-only run must ride out a market/API outage on its
            # infra budget instead of dying as if the job exceeds every GPU class.
            raise CapacityLookupError(
                f"no allocatable GPU (>= {need} GB VRAM for {model_id}): a provider's live capacity lookup "
                f"failed transiently and was the only source of a fitting class — retry may find hidden capacity"
            )
        if exact:
            dynamic_capacity_providers = {"lambda", "vast"}
            if available and set(available).issubset(dynamic_capacity_providers):
                raise CapacityLookupError(
                    f"exact GPU {exact!r} is structurally supported but currently has no capacity on "
                    f"{', '.join(available)}"
                )
            raise UnsupportedGpuError(
                f"exact GPU {exact!r} has no allocatable capacity on the requested active provider set "
                f"({', '.join(available) or '(none)'})"
            )
        raise UnsupportedGpuError(
            f"no allocatable GPU (>= {need} GB VRAM for {model_id}) on any available provider "
            f"({', '.join(available) or '(none)'}); the run genuinely exceeds every active GPU class"
        )
    # cheapest JOB first, not cheapest rental: rank on the dollars one step costs on each candidate
    # (rate x how long that hardware takes), so a faster card wins whenever it finishes enough sooner
    # to pay for itself. ties prefer fewer cards (less inter-card overhead), then combined VRAM, then
    # class name. sorting is stable, so provider and provider-local order apply only when all key
    # fields match. a run the cost model cannot price falls back to total $/hr.
    cost_per_step = _step_cost_ranker(model_id, algorithm, train, thinking, model_revision)
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

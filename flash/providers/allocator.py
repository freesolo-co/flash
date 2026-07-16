"""GPU allocation: cheapest fitting (provider, GPU class) across active providers."""

from __future__ import annotations

from flash._logging import get_logger
from flash.providers import PROVIDER_NAMES, available_providers, get_provider
from flash.providers.base import (
    GPU_INFO,
    Allocation,
    AllocationConstraints,
    Candidate,
    CapacityLookupError,
    UnsupportedGpuError,
    canonical_gpu,
    providers_for,
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


def allocate(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
    disk_gb: float = 0.0,
    max_wall_seconds: float = 0.0,
    provider: str = "",
    exact_type: str = "",
    model_revision: str = "",
) -> Allocation:
    """Pick the cheapest fitting (provider, GPU class) able to run the job."""
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
    if exact_type:
        exact = canonical_gpu(exact_type)
        exact_info = GPU_INFO.get(exact)
        if exact_info is None or not exact_info.validated:
            raise UnsupportedGpuError(
                f"exact GPU {exact!r} is not an active validated GPU class"
            )
        if exact_info.vram_gb < need:
            raise UnsupportedGpuError(
                f"exact GPU {exact!r} has {exact_info.vram_gb} GB VRAM, "
                f"but this run requires at least {need} GB"
            )
        exact_providers = providers_for(exact)
        if provider and provider not in exact_providers:
            raise UnsupportedGpuError(
                f"provider {provider!r} cannot provision exact GPU {exact!r}"
            )
        available = tuple(name for name in available if name in exact_providers)

    constraints = AllocationConstraints(
        disk_gb=disk_gb,
        max_wall_seconds=max_wall_seconds,
        exact_type=exact,
    )
    candidates: list[Candidate] = []
    lookup_failed = False
    # runpod prices off a static table (no live lookup), so it never blips; lambda/vast query live
    # capacity and can. a per-provider blip degrades to the others (we just skip it), but we remember it
    # so an empty result can be told apart from a genuine no-fit below.
    # runpod uses the same loop but does not raise CapacityLookupError.
    for name in available:
        try:
            found = get_provider(name).live_candidates(need, constraints)
            candidates += [
                candidate
                for candidate in found
                if candidate.provider == name and (not exact or candidate.gpu == exact)
            ]
        except CapacityLookupError as exc:
            lookup_failed = True
            logger.warning("%s capacity lookup failed (%s); allocating without it", name, exc.__cause__)
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
    # cheapest first; ties break by VRAM, then GPU class name. sorting is stable, so provider and
    # provider-local order apply only when all three key fields match.
    ranked = sorted(candidates, key=lambda c: (c.hourly_usd, c.vram_gb, c.gpu))
    best = ranked[0]
    return Allocation(
        provider=best.provider,
        gpu=best.gpu,
        hourly_usd=best.hourly_usd,
        min_vram_gb=need,
        candidates=tuple(ranked),
    )


def allocation_summary(a: Allocation) -> str:
    head = (
        f"allocated {a.gpu} on {a.provider} at ${a.hourly_usd:.2f}/hr "
        f"(need >= {a.min_vram_gb} GB VRAM)"
    )
    if len(a.candidates) > 1:
        nxt = a.candidates[1]
        head += f"; next-best: {nxt.gpu}@{nxt.provider} ${nxt.hourly_usd:.2f}/hr"
    return head

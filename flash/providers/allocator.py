"""GPU allocation: cheapest fitting (provider, GPU class) across active providers."""

from __future__ import annotations

from flash._logging import get_logger
from flash.providers import available_providers, get_provider
from flash.providers.base import (
    Allocation,
    AllocationConstraints,
    Candidate,
    CapacityLookupError,
    UnsupportedGpuError,
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
) -> int:
    """VRAM required for the full run, sized to actual knobs via model_required_vram_gb."""
    from flash.engine.vram import model_required_vram_gb

    return model_required_vram_gb(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        headroom=vram_headroom(),
    )


def allocate(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
    disk_gb: float = 0.0,
    max_wall_seconds: float = 0.0,
) -> Allocation:
    """Pick the cheapest fitting (provider, GPU class) able to run the job."""
    need = required_vram_gb(model_id, algorithm, train=train, thinking=thinking)
    available = available_providers()
    constraints = AllocationConstraints(disk_gb=disk_gb, max_wall_seconds=max_wall_seconds)
    candidates: list[Candidate] = []
    lookup_failed = False
    # RunPod prices off a static table (no live lookup), so it never blips; Lambda/Vast query live
    # capacity and can. A per-provider blip degrades to the others (we just skip it), but we remember it
    # so an EMPTY result can be told apart from a genuine no-fit below. RunPod runs through the same
    # try harmlessly (it never raises CapacityLookupError), so a 4th provider needs no edit here.
    for name in available:
        try:
            candidates += get_provider(name).live_candidates(need, constraints)
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
        raise UnsupportedGpuError(
            f"no allocatable GPU (>= {need} GB VRAM for {model_id}) on any available provider "
            f"({', '.join(available) or '(none)'}); the run genuinely exceeds every active GPU class"
        )
    # Cheapest first; ties broken by VRAM (prefer smaller), then GPU class name. The tie-break is
    # provider-agnostic, so runpod/lambda/vast compete purely on price with no structural edge.
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
        f"(need >= {a.min_vram_gb} GB VRAM"
    )
    head += ")"
    if len(a.candidates) > 1:
        nxt = a.candidates[1]
        head += f"; next-best: {nxt.gpu}@{nxt.provider} ${nxt.hourly_usd:.2f}/hr"
    return head

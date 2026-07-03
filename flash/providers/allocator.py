"""GPU allocation: cheapest fitting (provider, GPU class) across active providers."""

from __future__ import annotations

from flash._logging import get_logger
from flash.providers import available_providers, get_provider
from flash.providers.base import (
    Allocation,
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


def _runpod_candidates(need: int) -> list[Candidate]:
    """RunPod validated classes fitting the VRAM requirement, priced by the static table."""
    provider = get_provider("runpod")
    return [
        Candidate("runpod", g.name, provider.hourly_rate(g.name), g.vram_gb)
        for g in provider.gpu_classes()
        if g.vram_gb >= need and g.validated
    ]


def _lambda_candidates(need: int) -> list[Candidate]:
    """Lambda classes with live regional capacity fitting the VRAM requirement.

    A capacity-lookup failure raises ``CapacityLookupError``; ``allocate`` degrades to the other
    providers, failing the run retryably only if this was the sole fitting source.
    """
    from flash.providers.lambdalabs.jobs import usable_instances

    provider = get_provider("lambda")
    out: list[Candidate] = []
    try:
        for g in provider.gpu_classes():
            if g.vram_gb < need:
                continue
            if usable_instances(g.name):
                out.append(Candidate("lambda", g.name, provider.hourly_rate(g.name), g.vram_gb))
    except Exception as exc:
        # Transient capacity-lookup blip -> signal allocate() so it degrades to the other providers but
        # can still tell "no fit" from "outage" if this was the only fitting source (see CapacityLookupError).
        raise CapacityLookupError("lambda live capacity lookup failed") from exc
    return out


def _vast_candidates(need: int, disk_gb: float = 0.0, max_wall_seconds: float = 0.0) -> list[Candidate]:
    """Vast's fitting classes that currently have a LIVE verified-datacenter offer, priced live.

    Capacity-aware like Lambda: a Vast class with no fitting offer on the market right now is EXCLUDED,
    so the allocator never hands the runner a Vast class that would immediately fail to rent. ONE market
    search covers every class (offers carry their own gpu_name -> class), so we search once at the
    smallest fitting class's VRAM and bucket the returned offers by class. A capacity-lookup failure
    (market/API blip) raises ``CapacityLookupError`` -> ``allocate`` degrades to the other providers,
    failing the run retryably (not terminally) only if Vast was the sole fitting source.

    ``disk_gb`` and ``max_wall_seconds`` are the run's requested disk and wall cap; the Vast package
    prices against the SAME effective disk/duration floors the submit path provisions with, so a
    high-disk or long run isn't advertised capacity it couldn't actually rent (an impossible attempt a
    max_retries=0 run never escapes).
    """
    from flash.providers.vast.pricing import live_candidate_rates

    provider = get_provider("vast")
    fitting = [g for g in provider.gpu_classes() if g.vram_gb >= need]
    if not fitting:
        return []
    try:
        # Search once at the smallest fitting class's VRAM; the market covers every class at/above it.
        rates = live_candidate_rates(min(g.vram_gb for g in fitting), disk_gb, max_wall_seconds)
    except Exception as exc:
        # Transient market/API blip -> signal allocate() (see CapacityLookupError): a Vast-only run must
        # infra-retry the outage, not terminally fail as if no GPU fit.
        raise CapacityLookupError("vast live capacity lookup failed") from exc
    fitting_names = {g.name: g.vram_gb for g in fitting}
    return [
        Candidate("vast", name, rate, fitting_names[name])
        for name, rate in rates.items()
        if name in fitting_names
    ]


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
    candidates: list[Candidate] = []
    lookup_failed = False
    # RunPod prices off a static table (no live lookup), so it never blips; Lambda/Vast query live
    # capacity and can. A per-provider blip degrades to the others (we just skip it), but we remember it
    # so an EMPTY result can be told apart from a genuine no-fit below.
    if "runpod" in available:
        candidates += _runpod_candidates(need)
    for name, produce in (
        ("lambda", lambda: _lambda_candidates(need)),
        ("vast", lambda: _vast_candidates(need, disk_gb, max_wall_seconds)),
    ):
        if name not in available:
            continue
        try:
            candidates += produce()
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

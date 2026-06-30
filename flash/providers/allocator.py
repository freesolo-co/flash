"""GPU allocation: cheapest fitting (provider, GPU class) across active providers."""

from __future__ import annotations

from flash._logging import get_logger
from flash.providers import PROVIDER_NAMES, available_providers, get_provider
from flash.providers.base import (
    Allocation,
    Candidate,
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

    Capacity-lookup failure degrades gracefully (returns []) so other providers still run.
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
        logger.warning("lambda capacity lookup failed (%s); allocating without lambda", exc)
        return []
    return out


def _vast_candidates(need: int, disk_gb: float = 0.0, max_wall_seconds: float = 0.0) -> list[Candidate]:
    """Vast's fitting classes that currently have a LIVE verified-datacenter offer, priced live.

    Capacity-aware like Lambda: a Vast class with no fitting offer on the market right now is EXCLUDED,
    so the allocator never hands the runner a Vast class that would immediately fail to rent. ONE market
    search covers every class (offers carry their own gpu_name -> class), so we search once at the
    smallest fitting class's VRAM and bucket the returned offers by class. A capacity-lookup failure
    (no key / network blip) degrades to the other providers — non-fatal as long as another can supply.

    ``disk_gb`` is the run's requested disk; the capacity search uses the SAME effective floor
    (``max(disk_gb, MIN_DISK_GB)``) the submit path provisions with, so a high-disk run isn't advertised
    Vast capacity that only exists at the 60 GB floor and then fails to rent (an impossible attempt that
    a max_retries=0 run never escapes). ``max_wall_seconds`` is likewise threaded into the SAME duration
    floor the submit path uses, so a long run isn't advertised short-lived offers that expire mid-run.
    """
    from flash.providers.vast.jobs import MIN_DISK_GB, usable_offers

    provider = get_provider("vast")
    fitting = [g for g in provider.gpu_classes() if g.vram_gb >= need]
    if not fitting:
        return []
    try:
        # Search once at the smallest fitting class's VRAM floor; usable_offers returns every managed
        # class at/above it, which we then restrict to the fitting set.
        floor = min(g.vram_gb for g in fitting)
        offers = usable_offers(
            floor,
            max(float(disk_gb or 0.0), MIN_DISK_GB),
            max_wall_seconds=float(max_wall_seconds or 0.0),
        )
    except Exception as exc:
        logger.warning("vast capacity lookup failed (%s); allocating without vast", exc)
        return []
    # Cheapest live offer per class (offers are price-sorted, so the first seen per class is cheapest).
    cheapest: dict[str, float] = {}
    for o in offers:
        cheapest.setdefault(o.gpu, o.dph_total)
    fitting_names = {g.name: g.vram_gb for g in fitting}
    return [
        Candidate("vast", name, rate, fitting_names[name])
        for name, rate in cheapest.items()
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
    if "runpod" in available:
        candidates += _runpod_candidates(need)
    if "lambda" in available:
        candidates += _lambda_candidates(need)
    if "vast" in available:
        candidates += _vast_candidates(need, disk_gb, max_wall_seconds)
    if not candidates:
        raise UnsupportedGpuError(
            f"no allocatable GPU (>= {need} GB VRAM for {model_id}) on any available provider "
            f"({', '.join(available) or '(none)'}); the run genuinely exceeds every active GPU class"
        )
    # Cheapest first; ties broken by VRAM (prefer smaller), then registry order.
    order = {n: i for i, n in enumerate(PROVIDER_NAMES)}
    ranked = sorted(candidates, key=lambda c: (c.hourly_usd, c.vram_gb, order.get(c.provider, 99)))
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

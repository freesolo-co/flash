"""GPU allocation: the cheapest fitting class across the active providers.

Given a base model (+ algorithm), compute the VRAM the FULL run needs — sized for the
heavier phase, GRPO, since the typical pipeline is SFT followed by GRPO — then rank every
fitting candidate by $/hr and pick the cheapest:

  runpod      every validated Flash-provisionable class (static $/hr)
  lambda      every fitting class that currently has LIVE regional capacity (live $/hr); opt-in,
              available only when LAMBDA_API_KEY is set on the control plane

RunPod's cheaper static rates almost always win on price, so the instance provider (Lambda)
joins the ranked list as a capacity COMPLEMENT: when RunPod's cheapest fitting class is
out of capacity (THROTTLED / queue backstop), the runner's gpu-walk steps down the ranked list and
reaches an in-capacity instance class. The instance provider is capacity-filtered up front
(``_lambda_candidates`` only offers a class a region can supply right now), so the walk never lands
on a class that would just fail to launch.

Allocation happens at SUBMIT time in the runner. The parse-time resolution in schema is a
RunPod-static provisional for validation/dry-run display.
"""

from __future__ import annotations

from flash._logging import get_logger
from flash.providers import PROVIDER_NAMES, available_providers, get_provider
from flash.providers.base import (
    Allocation,
    Candidate,
    UnsupportedGpuError,
)

logger = get_logger(__name__)

# "Comfortably" = the open-model VRAM estimate plus headroom, so a full SFT+GRPO run
# never lands in check_fit's "tight" band by construction. Curated catalog entries
# already carry measured minimums and are used as-is. The headroom (default 1.1 ==
# model_required_vram_gb's own default) is read at call time via vram_headroom() so allocate()
# and the parse-time provisional_gpu size identically.


def vram_headroom() -> float:
    """The sizing headroom multiplier, honored by both the submit-time allocator and the
    parse-time provisional_gpu so they never disagree. A constant."""
    return 1.1


def required_vram_gb(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
) -> int:
    """VRAM the full run needs, sized to the run's actual knobs (context length, LoRA
    rank, batch / group size, thinking) via the shared ``model_required_vram_gb`` matrix.

    Catalog GRPO floors stay hard floors (never under-provision a validated model); the
    matrix sizes up from there for big contexts/groups and down to a cheaper card for
    small runs. Unlisted open models size from HF metadata, falling back to the 24 GB tier
    when unreadable (handled inside model_required_vram_gb)."""
    from flash.engine.vram import model_required_vram_gb

    return model_required_vram_gb(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        headroom=vram_headroom(),
    )


def _runpod_candidates(need: int) -> list[Candidate]:
    """RunPod's fitting, validated classes priced by the static table.

    Restricted to the validated pool (``g.validated``): the deployed control plane rejects a submit
    for any non-validated class, so allocating one would only fail at submit time.
    """
    provider = get_provider("runpod")
    return [
        Candidate("runpod", g.name, provider.hourly_rate(g.name), g.vram_gb)
        for g in provider.gpu_classes()
        if g.vram_gb >= need and g.validated
    ]


def _lambda_candidates(need: int) -> list[Candidate]:
    """Lambda's fitting classes that currently have LIVE capacity, priced live.

    Capacity-aware by design: a Lambda class with no region advertising capacity is EXCLUDED, so
    the allocator never hands the runner a Lambda class that would immediately fail to launch (and
    burn a retry) — directly the "GPU allocation is good, doesn't randomly die" property. A Lambda
    capacity-lookup failure (no key / network blip) degrades to the other providers: it is
    non-fatal as long as another provider can supply a fitting class.
    """
    from flash.providers.lambdalabs.jobs import usable_instances

    provider = get_provider("lambda")
    out: list[Candidate] = []
    try:
        for g in provider.gpu_classes():
            if g.vram_gb < need:
                continue
            # usable_instances reads the cached /instance-types, so only the first call hits the API.
            if usable_instances(g.name):
                out.append(Candidate("lambda", g.name, provider.hourly_rate(g.name), g.vram_gb))
    except Exception as exc:
        logger.warning("lambda capacity lookup failed (%s); allocating without lambda", exc)
        return []
    return out


def _vast_candidates(need: int) -> list[Candidate]:
    """Vast's fitting classes that currently have a LIVE verified-datacenter offer, priced live.

    Capacity-aware like Lambda: a Vast class with no fitting offer on the market right now is EXCLUDED,
    so the allocator never hands the runner a Vast class that would immediately fail to rent. ONE market
    search covers every class (offers carry their own gpu_name -> class), so we search once at the
    smallest fitting class's VRAM and bucket the returned offers by class. A capacity-lookup failure
    (no key / network blip) degrades to the other providers — non-fatal as long as another can supply.
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
        offers = usable_offers(floor, MIN_DISK_GB)
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
) -> Allocation:
    """Pick the cheapest fitting (provider, GPU class) able to run the job.

    There is no GPU pin — every fitting class on every available provider is eligible, and the
    cheapest wins. RunPod is restricted to its validated pool (``GpuClass.validated``) because the
    deployed control plane rejects a submit for any non-validated class; the instance provider
    (Lambda via LAMBDA_API_KEY — opt-in) contributes
    its fitting classes that currently have live capacity. RunPod's cheaper static rates
    usually win, with Lambda joining as a capacity complement lower in the ranked list.
    ``train``/``thinking`` size the requirement to the run's actual knobs (context, group, rank,
    batch) via the matrix.
    """
    need = required_vram_gb(model_id, algorithm, train=train, thinking=thinking)
    available = available_providers()
    candidates: list[Candidate] = []
    if "runpod" in available:
        candidates += _runpod_candidates(need)
    if "lambda" in available:
        candidates += _lambda_candidates(need)
    if "vast" in available:
        candidates += _vast_candidates(need)
    if not candidates:
        raise UnsupportedGpuError(
            f"no allocatable GPU (>= {need} GB VRAM for {model_id}) on any available provider "
            f"({', '.join(available) or '(none)'}); the run genuinely exceeds every active GPU class"
        )
    # Cheapest first; equal rates prefer less VRAM (don't burn a big card on a small job),
    # then registry order.
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

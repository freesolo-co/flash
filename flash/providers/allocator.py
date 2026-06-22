"""Cross-provider GPU allocation: the cheapest class that comfortably fits the run.

Given a base model (+ algorithm), compute the VRAM the FULL run needs — sized for the
heavier phase, GRPO, since the typical pipeline is SFT followed by GRPO — then rank
every provisionable candidate across ALL registered providers by live $/hr and pick the
cheapest:

  runpod  every Flash-provisionable class (live pricing, cached; static fallback)
  vast    live verified-datacenter offers (usable_offers' quality floors applied)

Allocation happens at SUBMIT time in the runner (offers are a volatile market);
the parse-time resolution in schema is a RunPod-static provisional for
validation/dry-run display. With no live pricing/offers reachable (no network, or no
``VAST_API_KEY``) the allocator degrades to exactly ``cheapest_gpu``'s deterministic
static-rate answer (RunPod only — Vast is off without its key).

Provider-agnostic by construction: it walks the registered providers and asks each for
its ``gpu_classes()`` + ``hourly_rate()``; the only provider-specific knowledge is that
Vast classes come from a live offer book (collected through the provider's
``usable_offers`` and carried opaquely on ``Candidate.offer``).
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


def _runpod_candidates(need: int, exclude_gpu_classes: frozenset[str]) -> list[Candidate]:
    """RunPod's fitting, live-validated classes priced live (static fallback).

    Restricted to the validated pool (``g.validated``): the deployed control plane rejects a submit
    for any non-validated class, so allocating one would only fail at submit time. Any class in
    ``exclude_gpu_classes`` is dropped — used to walk OFF a MIG-prone class (a class RunPod has been
    seen fulfilling with a MIG slice) onto a different one on the infra-retry path."""
    provider = get_provider("runpod")
    return [
        Candidate("runpod", g.name, provider.hourly_rate(g.name), g.vram_gb)
        for g in provider.gpu_classes()
        if g.vram_gb >= need and g.validated and g.name not in exclude_gpu_classes
    ]


def _vast_candidates(
    need: int, disk_gb: int, exclude_machine_ids, exclude_gpu_classes: frozenset[str]
) -> tuple[list[Candidate], tuple]:
    """Vast's fitting, live-validated classes from the live offer book (cheapest per class).

    Returns (candidates, full_offer_book). Restricted to the validated pool (``GPU_INFO[gpu]
    .validated``) — the deployed control plane rejects a submit for any non-validated class. A Vast
    offer-search failure is caught and degrades to the other providers (RunPod): it is non-fatal AS
    LONG AS another provider can supply a fitting class. If Vast is the only available provider, the
    empty result means ``allocate`` then raises (nothing across any provider fits) — i.e. it is only
    fatal when Vast was the sole option.
    """
    from flash.providers.base import GPU_INFO
    from flash.providers.vast.jobs import MIN_DISK_GB, usable_offers

    book: list = []
    try:
        # The offer search must use the SAME disk floor instances are actually provisioned with
        # (a smaller requested ``disk_gb`` would surface offers that then fail to rent).
        book = usable_offers(
            need, max(float(disk_gb), MIN_DISK_GB), exclude_machine_ids=exclude_machine_ids
        )
    except Exception as exc:
        logger.warning("vast offer search failed (%s); allocating on runpod only", exc)
    out: list[Candidate] = []
    seen: set[str] = set()
    for o in book:
        if o.gpu in seen:  # offers are price-sorted; keep the cheapest per class
            continue
        if not GPU_INFO[o.gpu].validated:  # only offer live-validated classes the server accepts
            continue
        if o.gpu in exclude_gpu_classes:  # walked OFF this class (e.g. MIG-prone) on infra retry
            continue
        seen.add(o.gpu)
        out.append(Candidate("vast", o.gpu, o.dph_total, GPU_INFO[o.gpu].vram_gb, offer=o))
    return out, tuple(book)


def allocate(
    model_id: str,
    algorithm: str,
    *,
    disk_gb: int = 60,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
    exclude_gpu_classes: set[str] | frozenset[str] = frozenset(),
    train=None,
    thinking: bool = False,
) -> Allocation:
    """Pick the cheapest (provider, GPU class) able to run the job across ALL providers.

    There is no GPU pin and no provider pin — every fitting, LIVE-VALIDATED class on every live
    provider is eligible, and the cheapest wins. Allocation is restricted to the validated pool
    (``GpuClass.validated``) because the deployed control plane rejects a submit for any
    non-validated class, so picking the absolute-cheapest fitting class (e.g. an unvalidated "RTX
    2000 Ada") would just make the server refuse the run. ``train``/``thinking`` size the
    requirement to the run's actual knobs (context, group, rank, batch) via the matrix.

    ``exclude_gpu_classes`` drops whole GPU CLASSES from the candidate pool (across every provider).
    The infra-retry path uses it to walk OFF a class that returned a RETRIABLE_INFRA_GPU failure —
    e.g. an "RTX A5000" request RunPod fulfilled with a Blackwell MIG slice — so the retry
    re-allocates to a DIFFERENT validated class (a consumer card like RTX 4090/5090/3090 that can't
    be MIG-partitioned) instead of re-picking the same MIG-prone class and failing identically.
    """
    exclude_gpu_classes = frozenset(exclude_gpu_classes)
    need = required_vram_gb(model_id, algorithm, train=train, thinking=thinking)
    live = available_providers()
    candidates: list[Candidate] = []
    offer_book: tuple = ()
    if "runpod" in live:
        candidates += _runpod_candidates(need, exclude_gpu_classes)
    if "vast" in live:
        vcands, offer_book = _vast_candidates(
            need, disk_gb, exclude_machine_ids, exclude_gpu_classes
        )
        candidates += vcands
    if not candidates:
        excluded = (
            f" (excluding GPU classes {sorted(exclude_gpu_classes)})" if exclude_gpu_classes else ""
        )
        raise UnsupportedGpuError(
            f"no allocatable GPU (>= {need} GB VRAM for {model_id}) on any live provider "
            f"({', '.join(live) or '(none)'}){excluded}; add VAST_API_KEY for more classes, or the "
            "run genuinely exceeds every available GPU class"
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
        offer=best.offer,
        provider_offers=offer_book,
    )


def allocation_summary(a: Allocation) -> str:
    head = (
        f"allocated {a.gpu} on {a.provider} at ${a.hourly_usd:.2f}/hr "
        f"(need >= {a.min_vram_gb} GB VRAM"
    )
    # ``a.offer`` is an OPAQUE per-provider provisioning hint, not necessarily a Vast
    # offer — only format Vast specifics when the chosen provider is vast, so a future
    # provider's hint never misformats or raises on a missing attribute.
    if a.provider == "vast" and a.offer is not None:
        head += f", vast offer {a.offer.offer_id} in {a.offer.geolocation}"
    head += ")"
    if len(a.candidates) > 1:
        nxt = a.candidates[1]
        head += f"; next-best: {nxt.gpu}@{nxt.provider} ${nxt.hourly_usd:.2f}/hr"
    return head

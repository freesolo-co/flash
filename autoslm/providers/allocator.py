"""Cross-provider GPU allocation: the cheapest class that comfortably fits the run.

Given a base model (+ algorithm), compute the VRAM the FULL run needs — sized for the
heavier phase, GRPO, since the typical pipeline is SFT followed by GRPO — then rank
every provisionable candidate across ALL registered providers by live $/hr and pick the
cheapest:

  runpod  every Flash-provisionable class (live pricing, cached; static fallback)
  vast    live verified-datacenter offers (usable_offers' quality floors applied)

Allocation happens at SUBMIT time in the runner (offers are a volatile market);
the parse-time resolution in schema is a RunPod-static provisional for
validation/dry-run display. Offline (AUTOSLM_SKIP_NET) the allocator degrades to exactly
``cheapest_gpu``'s deterministic static-rate answer (RunPod only — Vast is offline-off).

Provider-agnostic by construction: it walks the registered providers and asks each for
its ``gpu_classes()`` + ``hourly_rate()``; the only provider-specific knowledge is that
Vast classes come from a live offer book (collected through the provider's
``usable_offers`` and carried opaquely on ``Candidate.offer``).
"""

from __future__ import annotations

import math
import os

from autoslm._logging import get_logger
from autoslm.providers import PROVIDER_NAMES, available_providers, get_provider
from autoslm.providers.base import (
    Allocation,
    Candidate,
    UnsupportedGpuError,
    canonical_gpu,
    unvalidated_allowed,
)

logger = get_logger(__name__)

# "Comfortably" = the open-model VRAM estimate plus headroom, so a full SFT+GRPO run
# never lands in check_fit's "tight" band by construction. Curated catalog entries
# already carry measured minimums and are used as-is.
VRAM_HEADROOM = float(os.environ.get("AUTOSLM_VRAM_HEADROOM", "1.15"))


def required_vram_gb(model_id: str, algorithm: str) -> int:
    """VRAM the full run needs. Catalog entries carry measured minimums; open models
    get the coarse estimate sized for GRPO (the heavier phase of the usual SFT+GRPO
    pipeline) plus headroom. Unknown sizes fall back to the 24 GB tier (same as
    ``resolve_gpu_policy``)."""
    from autoslm.catalog import catalog_min_vram_gb

    # Catalog entries carry measured minimums (shared with the parse-time
    # resolve_gpu_policy via catalog.catalog_min_vram_gb, incl. the grpo floor).
    catalog_vram = catalog_min_vram_gb(model_id, algorithm)
    if catalog_vram is not None:
        return catalog_vram
    from autoslm.engine.vram import estimate_vram_gb, fetch_hf_params_b

    params_b = fetch_hf_params_b(model_id)
    if params_b is None:
        return 24
    sizing_algo = algorithm if os.environ.get("AUTOSLM_SIZE_FOR_ALGO") == "job" else "grpo"
    return math.ceil(estimate_vram_gb(params_b, sizing_algo) * VRAM_HEADROOM)


def _runpod_candidates(need: int, pinned_gpu: str | None, allow_unval: bool) -> list[Candidate]:
    """RunPod's fitting classes priced live (static fallback)."""
    provider = get_provider("runpod")
    out: list[Candidate] = []
    for g in provider.gpu_classes():
        if g.vram_gb < need:
            continue
        if pinned_gpu and g.name != pinned_gpu:
            continue
        if "runpod" not in g.validated_on and not allow_unval:
            continue
        out.append(
            Candidate(
                "runpod",
                g.name,
                provider.hourly_rate(g.name),
                g.vram_gb,
                "runpod" in g.validated_on,
            )
        )
    return out


def _vast_candidates(
    need: int,
    pinned_gpu: str | None,
    allow_unval: bool,
    disk_gb: int,
    exclude_machine_ids,
    *,
    required: bool,
) -> tuple[list[Candidate], tuple]:
    """Vast's fitting classes from the live offer book (cheapest per class).

    Returns (candidates, full_offer_book). ``required`` (a hard ``provider="vast"``
    pin) re-raises a search failure; otherwise it degrades to RunPod-only.
    """
    from autoslm.providers.base import GPU_INFO
    from autoslm.providers.vast.jobs import MIN_DISK_GB, usable_offers

    # When a larger class is pinned for a small model, search at the PINNED class's VRAM,
    # not the (smaller) model requirement: the offer search returns the cheapest ``limit``
    # offers from a VRAM floor, so a search at ``need`` can fill that window entirely with
    # small cheap cards and never surface the pinned larger class. ``need`` is still the
    # validity floor (allocate() rejects an undersized pin before we get here).
    search_vram = max(need, GPU_INFO[pinned_gpu].vram_gb) if pinned_gpu else need
    book: list = []
    try:
        # The offer search must use the SAME disk floor instances are actually
        # provisioned with (``create_instance``/``_effective_disk_gb``); searching at a
        # smaller requested ``disk_gb`` would surface offers that then fail to rent.
        book = usable_offers(
            search_vram, max(float(disk_gb), MIN_DISK_GB), exclude_machine_ids=exclude_machine_ids
        )
    except Exception as exc:
        if required:
            raise UnsupportedGpuError(f"vast offer search failed: {exc}") from exc
        logger.warning("vast offer search failed (%s); allocating on runpod only", exc)
    out: list[Candidate] = []
    seen: set[str] = set()
    for o in book:
        if pinned_gpu and o.gpu != pinned_gpu:
            continue
        info = GPU_INFO[o.gpu]
        if "vast" not in info.validated_on and not allow_unval:
            continue
        if o.gpu in seen:  # offers are price-sorted; keep the cheapest per class
            continue
        seen.add(o.gpu)
        out.append(
            Candidate(
                "vast", o.gpu, o.dph_total, info.vram_gb, "vast" in info.validated_on, offer=o
            )
        )
    return out, tuple(book)


def allocate(
    model_id: str,
    algorithm: str,
    *,
    gpu: str | None = None,
    provider: str = "auto",
    disk_gb: int = 60,
    allow_unvalidated: bool | None = None,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
) -> Allocation:
    """Pick the cheapest (provider, GPU class) able to run the job across providers.

    ``gpu`` pins the class (the allocator then only picks the provider); ``provider``
    pins the substrate ("auto"/"runpod"/"vast"). Both default to fully automatic.
    """
    if provider not in ("auto", *PROVIDER_NAMES):
        raise UnsupportedGpuError(
            f"unknown provider {provider!r} (auto, {', '.join(PROVIDER_NAMES)})"
        )
    pinned_gpu = canonical_gpu(gpu) if gpu else None
    # The model's requirement is the floor regardless of a pin: an undersized concrete
    # pin (e.g. Qwen3-8B on a 24 GB card) must drop out of the candidate filter and
    # raise here, not provision a paid worker that OOMs. The pin only narrows WHICH
    # fitting class is chosen, never lowers the VRAM bar.
    need = required_vram_gb(model_id, algorithm)
    allow_unval = unvalidated_allowed(allow_unvalidated)
    live = available_providers()
    if provider != "auto" and provider not in live:
        raise UnsupportedGpuError(
            f"provider {provider!r} requested but not available on this control plane "
            f"(available: {', '.join(live) or '(none)'}; vast needs VAST_API_KEY)"
        )

    candidates: list[Candidate] = []
    offer_book: tuple = ()
    if provider in ("auto", "runpod") and "runpod" in live:
        candidates += _runpod_candidates(need, pinned_gpu, allow_unval)
    if provider in ("auto", "vast") and "vast" in live:
        vast_cands, offer_book = _vast_candidates(
            need,
            pinned_gpu,
            allow_unval,
            disk_gb,
            exclude_machine_ids,
            required=(provider == "vast"),
        )
        candidates += vast_cands

    if not candidates:
        constraint = (
            f"gpu pinned to {pinned_gpu}" if pinned_gpu else f">= {need} GB VRAM for {model_id}"
        )
        raise UnsupportedGpuError(
            f"no allocatable GPU ({constraint}, provider={provider}, "
            f"validated_only={not allow_unval}); widen with gpu.allow_unvalidated = true "
            f"or a different gpu.type/provider"
        )
    # Cheapest first; equal rates prefer less VRAM (don't burn a big card on a small
    # job), then registry order (runpod is the longest-validated substrate).
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

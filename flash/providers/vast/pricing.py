"""Vast.ai $/hr: cheapest live verified-datacenter offer per class, static fallback.

RunPod prices a fixed class catalog; Vast is a live market, so a class's "rate" is the
cheapest currently-usable offer for it (``usable_offers``). This module gives the
provider interface a uniform ``hourly_rate(gpu)`` and a ``live_rates()`` map for the
``flash gpus`` table. Offline-safe: without ``VAST_API_KEY`` (or on any failure) it falls
back to the shared catalog snapshot (``GpuClass.hourly_usd`` — the RunPod static rate,
NOT a live Vast price), a rough proxy that may not match the live Vast market. The
authoritative Vast rate is the live path; the fallback exists only so an offline cost
estimate / ``flash gpus`` render never crashes. (A dedicated Vast static snapshot, like
``lambdalabs/pricing.py`` carries, is the follow-up to make the offline number Vast-accurate.)
"""

from __future__ import annotations

import os
import time
from typing import Any

from flash._logging import get_logger

logger = get_logger(__name__)

# Cache the live market query (like Lambda's /instance-types cache) so repeated hourly_rate() lookups
# — e.g. the `flash gpus` table renders a row per class — share ONE Vast market fetch within the TTL
# instead of hammering the API. ``refresh=True`` bypasses it.
_RATES_TTL_S = 45.0
_rates_cache: dict[str, Any] = {"ts": 0.0, "data": None}


def _static_rates() -> dict[str, float]:
    """Offline fallback rate per class with a ``vast_name``. NOTE: this is the shared catalog
    ``GpuClass.hourly_usd`` (the RunPod static snapshot), NOT a live Vast price — a rough proxy used
    only when live pricing is unavailable, so consumers of ``live_rates()`` get a number rather than
    a crash. May not match the Vast market."""
    from flash.providers.base import GPU_INFO

    return {name: info.hourly_usd for name, info in GPU_INFO.items() if info.vast_name}


def live_rates(refresh: bool = False) -> dict[str, float]:
    """Friendly-name -> cheapest live verified-datacenter $/hr (static fallback).

    Cached for ``_RATES_TTL_S`` so repeated lookups share one market fetch; ``refresh=True`` bypasses
    the cache and forces a fresh query. Offline-safe: without ``VAST_API_KEY`` (or on any fetch
    failure) returns the static rates (and does not cache the failure).
    """
    static = _static_rates()
    if not os.environ.get("VAST_API_KEY"):
        return static
    now = time.monotonic()
    if not refresh and _rates_cache["data"] is not None and now - _rates_cache["ts"] < _RATES_TTL_S:
        return _rates_cache["data"]
    try:
        from flash.providers.vast.jobs import MIN_DISK_GB, usable_offers

        rates: dict[str, float] = {}
        # Gate on MIN_DISK_GB (what create() enforces): disk_gb=0 would disable disk filtering and
        # price the run off "cheapest" offers that aren't actually provisionable, making live pricing
        # optimistic and inconsistent with the allocator/submit paths (both pass MIN_DISK_GB).
        for offer in usable_offers(0, MIN_DISK_GB):  # offers are price-sorted, cheapest first
            rates.setdefault(offer.gpu, offer.dph_total)
        merged = {**static, **rates}
        _rates_cache.update(ts=now, data=merged)
        return merged
    except Exception as exc:
        logger.warning("live vast pricing unavailable (%s); using static rates", exc)
        return static


def hourly_rate(gpu_name: str) -> float:
    """$/hr for one friendly GPU name (cheapest live offer if available, else static)."""
    from flash.providers.base import canonical_gpu

    name = canonical_gpu(gpu_name)
    return live_rates().get(name) or _static_rates().get(name, 0.0)

"""Vast.ai $/hr: cheapest live verified-datacenter offer per class, static fallback.

Vast is a live market, so a class's rate is its cheapest currently-usable offer
(``usable_offers``). Offline-safe: without ``VAST_API_KEY`` (or on any failure) falls back to the
shared catalog snapshot (``GpuClass.hourly_usd`` — the RunPod static rate, NOT a live Vast price;
a rough proxy that may not match the market), so an offline estimate / ``flash gpus`` render never
crashes.
"""

from __future__ import annotations

import os
import time
from typing import Any

from flash._logging import get_logger

logger = get_logger(__name__)

# Cache the live market query so repeated hourly_rate() lookups (e.g. `flash gpus` renders a row per
# class) share ONE Vast market fetch within the TTL. ``refresh=True`` bypasses it.
_RATES_TTL_S = 45.0
_rates_cache: dict[str, Any] = {"ts": 0.0, "data": None}


def _static_rates() -> dict[str, float]:
    """Offline fallback rate per class with a ``vast_name``: the shared catalog ``GpuClass.hourly_usd``
    (RunPod snapshot), NOT a live Vast price — a rough proxy used only when live pricing is unavailable."""
    from flash.providers.base import GPU_INFO

    return {name: info.hourly_usd for name, info in GPU_INFO.items() if info.vast_name}


def _fetch_offer_rates(max_wall_seconds: float) -> dict[str, float]:
    """Friendly-name -> cheapest LIVE verified-datacenter $/hr for ONLY classes with a usable offer
    (NO static merge). Raises on fetch failure; assumes ``VAST_API_KEY`` is set (callers gate on it)."""
    from flash.providers.base import GPU_INFO
    from flash.providers.vast.jobs import MIN_DISK_GB, usable_offers

    rates: dict[str, float] = {}
    # Floor the market query at the smallest managed class's VRAM, not 0: min_vram_gb=0 returns the
    # cheapest offers across ALL sizes, so a flood of tiny unmanaged low-VRAM cards fills the price-sorted
    # page and crowds managed classes off it. No managed class is smaller than the floor, so none is
    # excluded; 0 if nothing is managed.
    vram_floor = int(min((i.vram_gb for i in GPU_INFO.values() if i.vast_name), default=0))
    # Gate on MIN_DISK_GB (what create() enforces): disk_gb=0 would price off "cheapest" offers that
    # aren't provisionable, inconsistent with the allocator/submit paths. Thread the wall cap so
    # duration-bound estimates match the offers a launch would accept.
    for offer in usable_offers(vram_floor, MIN_DISK_GB, max_wall_seconds=max_wall_seconds):
        rates.setdefault(offer.gpu, offer.dph_total)  # offers are price-sorted, cheapest first
    return rates


def live_rates(refresh: bool = False, max_wall_seconds: float = 0.0) -> dict[str, float]:
    """Friendly-name -> cheapest live verified-datacenter $/hr (static fallback).

    Cached for ``_RATES_TTL_S`` so repeated lookups share one market fetch; ``refresh=True`` forces a
    fresh query. Offline-safe: without ``VAST_API_KEY`` (or on any fetch failure) returns static rates.

    ``max_wall_seconds`` (>0) restricts the query to offers that outlast the run's whole wall cap (the
    same duration floor the allocator/submit path pass to ``usable_offers``), so a duration-bound
    estimate isn't set by a short-lived offer filtered out at launch. Such queries BYPASS the shared
    duration-agnostic cache — computed fresh and not stored.
    """
    static = _static_rates()
    if not os.environ.get("VAST_API_KEY"):
        return static
    now = time.monotonic()
    use_cache = max_wall_seconds <= 0  # duration-bound queries must not read/write the shared cache
    if (
        use_cache
        and not refresh
        and _rates_cache["data"] is not None
        and now - _rates_cache["ts"] < _RATES_TTL_S
    ):
        return _rates_cache["data"]
    try:
        merged = {**static, **_fetch_offer_rates(max_wall_seconds)}
    except Exception as exc:
        logger.warning("live vast pricing unavailable (%s); using static rates", exc)
        return static
    if use_cache:
        _rates_cache.update(ts=now, data=merged)
    return merged


def live_offer_rates(max_wall_seconds: float = 0.0) -> dict[str, float]:
    """Friendly-name -> cheapest live $/hr for ONLY classes with a rentable offer (NO static merge);
    ``{}`` offline / without ``VAST_API_KEY`` / on any fetch failure.

    Unlike ``live_rates`` (which merges static rates so ``flash gpus`` renders a row per class), this
    returns just the classes a launch could ACTUALLY rent under the wall cap. GPU selection uses it so a
    cheaper class with no surviving offer isn't chosen — and quoted — on its static (RunPod) rate. Never
    cached: selection always reflects the current market.
    """
    if not os.environ.get("VAST_API_KEY"):
        return {}
    try:
        return _fetch_offer_rates(max_wall_seconds)
    except Exception as exc:
        logger.warning("live vast offer rates unavailable (%s)", exc)
        return {}


def hourly_rate(gpu_name: str, max_wall_seconds: float = 0.0) -> float:
    """$/hr for one friendly GPU name (cheapest live offer if available, else static).

    ``max_wall_seconds`` (>0) prices against offers that outlast the run's wall cap (see ``live_rates``)
    so a long run is not underquoted by a short-lived offer that won't survive to launch."""
    from flash.providers.base import canonical_gpu

    name = canonical_gpu(gpu_name)
    return live_rates(max_wall_seconds=max_wall_seconds).get(name) or _static_rates().get(name, 0.0)

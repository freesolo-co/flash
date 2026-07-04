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
    from flash.providers.base import static_rates_for

    return static_rates_for("vast_name")


def live_candidate_rates(
    min_vram_gb: int, disk_gb: float = 0.0, max_wall_seconds: float = 0.0
) -> dict[str, float]:
    """Friendly-name -> cheapest LIVE verified-datacenter $/hr per managed class that currently has a
    rentable offer at/above ``min_vram_gb``, using the SAME effective disk floor
    (``max(disk_gb, MIN_DISK_GB)``) and duration floor the submit path provisions with — so a class is
    only priced/advertised when a launch could actually rent it. ONE market search; offers are
    price-sorted so the first seen per class is cheapest. Raises on a fetch failure (callers decide the
    fallback); assumes ``VAST_API_KEY`` is set (callers gate on it). Shared by the ``flash gpus`` /
    cost-estimate pricing path and the allocator's capacity check."""
    from flash.providers.vast.jobs import MIN_DISK_GB, usable_offers

    rates: dict[str, float] = {}
    for offer in usable_offers(
        min_vram_gb, max(float(disk_gb or 0.0), MIN_DISK_GB), max_wall_seconds=max_wall_seconds
    ):
        rates.setdefault(offer.gpu, offer.dph_total)  # price-sorted, first seen per class is cheapest
    return rates


def _fetch_offer_rates(max_wall_seconds: float, min_vram_gb: int = 0) -> dict[str, float]:
    """Friendly-name -> cheapest LIVE $/hr for the managed classes with a usable offer (NO static
    merge). Raises on fetch failure; assumes ``VAST_API_KEY`` is set (callers gate on it).

    ``min_vram_gb`` (>0) raises the market-search floor to the caller's required VRAM. The page is
    price-sorted and LIMITED, so flooring a high-VRAM job at the smallest managed class lets cheap
    small-card offers fill the page and crowd the big classes off it — omitting exactly the classes that
    job needs. Flooring at the required VRAM (parity with the launch allocator, which searches at the
    smallest FITTING class) keeps them on the page. Defaults to the smallest managed class.
    """
    from flash.providers.base import GPU_INFO

    # Floor the market query at the smallest managed class's VRAM, not 0: min_vram_gb=0 returns the
    # cheapest offers across ALL sizes, so a flood of tiny unmanaged low-VRAM cards fills the price-sorted
    # page and crowds managed classes off it. No managed class is smaller than the floor, so none is
    # excluded; 0 if nothing is managed. A caller-supplied ``min_vram_gb`` raises it further.
    vram_floor = int(min((i.vram_gb for i in GPU_INFO.values() if i.vast_name), default=0))
    return live_candidate_rates(max(vram_floor, int(min_vram_gb)), max_wall_seconds=max_wall_seconds)


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


def live_offer_rates(max_wall_seconds: float = 0.0, min_vram_gb: int = 0) -> dict[str, float]:
    """Friendly-name -> cheapest live $/hr for ONLY classes with a rentable offer (NO static merge);
    ``{}`` offline / without ``VAST_API_KEY`` / on any fetch failure.

    Unlike ``live_rates`` (which merges static rates so ``flash gpus`` renders a row per class), this
    returns just the classes a launch could ACTUALLY rent under the wall cap. GPU selection uses it so a
    cheaper class with no surviving offer isn't chosen — and quoted — on its static (RunPod) rate. Never
    cached: selection always reflects the current market.

    ``min_vram_gb`` floors the market search at the job's required VRAM so a high-VRAM selection isn't
    crowded off the price-sorted page by cheaper small-card offers (parity with the launch allocator).
    """
    if not os.environ.get("VAST_API_KEY"):
        return {}
    try:
        return _fetch_offer_rates(max_wall_seconds, min_vram_gb=min_vram_gb)
    except Exception as exc:
        logger.warning("live vast offer rates unavailable (%s)", exc)
        return {}


def hourly_rate(gpu_name: str, max_wall_seconds: float = 0.0, min_vram_gb: int = 0) -> float:
    """$/hr for one friendly GPU name (cheapest live offer if available, else static).

    ``max_wall_seconds`` (>0) prices against offers that outlast the run's wall cap (see ``live_rates``)
    so a long run is not underquoted by a short-lived offer that won't survive to launch.

    ``min_vram_gb`` (>0) floors the market search at the caller's required VRAM (selection parity): the
    price-sorted page is LIMITED, so searching from the smallest managed class lets cheap small-card
    offers crowd a high-VRAM class off it — it'd miss the live map and be misquoted on the static
    fallback. Only when the class has no rentable offer at that floor do we fall back to ``live_rates``."""
    from flash.providers.base import canonical_gpu

    name = canonical_gpu(gpu_name)
    if min_vram_gb > 0:
        floored = live_offer_rates(max_wall_seconds=max_wall_seconds, min_vram_gb=min_vram_gb)
        if name in floored:
            return floored[name]
    # live_rates already merges the static snapshot (and returns it wholesale offline), so every
    # vast_name class is present on every return path — no second static lookup needed.
    return live_rates(max_wall_seconds=max_wall_seconds).get(name) or 0.0

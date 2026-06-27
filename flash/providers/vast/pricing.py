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


def _fetch_offer_rates(max_wall_seconds: float) -> dict[str, float]:
    """Friendly-name -> cheapest LIVE verified-datacenter $/hr, for ONLY the classes that currently
    have a usable offer (NO static merge). Raises on a fetch failure (callers decide the fallback);
    assumes ``VAST_API_KEY`` is set (callers gate on it)."""
    from flash.providers.base import GPU_INFO
    from flash.providers.vast.jobs import MIN_DISK_GB, usable_offers

    rates: dict[str, float] = {}
    # Floor the market query at the SMALLEST managed Vast class's VRAM, NOT 0: with min_vram_gb=0
    # the server returns the cheapest offers across ALL sizes — a flood of tiny UNMANAGED low-VRAM
    # cards fills the fixed-size price-sorted page and crowds the managed classes off it, so
    # live pricing misses them and falls back to static (RunPod) rates even when live Vast offers
    # exist. The floor keeps it to ONE market query while making the page relevant; no managed class
    # is smaller than the floor, so none is excluded (Copilot Mtugt). 0 if nothing is managed.
    vram_floor = int(min((i.vram_gb for i in GPU_INFO.values() if i.vast_name), default=0))
    # Gate on MIN_DISK_GB (what create() enforces): disk_gb=0 would disable disk filtering and
    # price the run off "cheapest" offers that aren't actually provisionable, making live pricing
    # optimistic and inconsistent with the allocator/submit paths (both pass MIN_DISK_GB). Thread
    # the wall cap so duration-bound estimates match the offers a launch would actually accept.
    for offer in usable_offers(vram_floor, MIN_DISK_GB, max_wall_seconds=max_wall_seconds):
        rates.setdefault(offer.gpu, offer.dph_total)  # offers are price-sorted, cheapest first
    return rates


def live_rates(refresh: bool = False, max_wall_seconds: float = 0.0) -> dict[str, float]:
    """Friendly-name -> cheapest live verified-datacenter $/hr (static fallback).

    Cached for ``_RATES_TTL_S`` so repeated lookups share one market fetch; ``refresh=True`` bypasses
    the cache and forces a fresh query. Offline-safe: without ``VAST_API_KEY`` (or on any fetch
    failure) returns the static rates (and does not cache the failure).

    ``max_wall_seconds`` (>0) restricts the market query to offers available for the run's whole wall
    cap — the SAME duration floor the allocator and submit path pass to ``usable_offers`` — so a
    duration-bound cost estimate is not set by a cheap short-lived offer that gets filtered out at
    launch (Codex MtzrI). Duration-bound queries narrow the offer set, so they BYPASS the shared
    duration-agnostic cache (the ``flash gpus`` table path) — computed fresh and not stored.
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
    """Friendly-name -> cheapest live $/hr for ONLY the classes that currently have a rentable offer
    (NO static merge); ``{}`` offline / without ``VAST_API_KEY`` / on any fetch failure.

    Unlike ``live_rates`` (which merges static rates so the ``flash gpus`` table can render a row per
    managed class), this returns just the classes a Vast launch could ACTUALLY rent under the wall
    cap. Provider-specific GPU SELECTION uses it so a cheaper class with NO surviving offer is not
    chosen — and quoted — on its static (RunPod) rate when the launch-time ``usable_offers`` path
    would never rent it (Codex). Never cached: selection always reflects the current market.
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

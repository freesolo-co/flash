"""Vast.ai $/hr: cheapest live verified-datacenter offer per class, static fallback.

RunPod prices a fixed class catalog; Vast is a live market, so a class's "rate" is the
cheapest currently-usable offer for it (``usable_offers``). This module gives the
provider interface a uniform ``hourly_rate(gpu)`` and a ``live_rates()`` map for the
``flash gpus`` table. Offline-safe: without ``VAST_API_KEY`` (or on any failure) it falls
back to the static Vast snapshot carried on ``GpuClass.hourly_usd``.
"""

from __future__ import annotations

import os

from flash._logging import get_logger

logger = get_logger(__name__)


def _static_rates() -> dict[str, float]:
    """Static Vast snapshot rate per class with a ``vast_name`` (display-only fallback)."""
    from flash.providers.base import GPU_INFO

    return {name: info.hourly_usd for name, info in GPU_INFO.items() if info.vast_name}


def live_rates(refresh: bool = False) -> dict[str, float]:
    """Friendly-name -> cheapest live verified-datacenter $/hr (static fallback).

    Offline-safe: without ``VAST_API_KEY`` (or on any fetch failure) returns the static rates.
    """
    static = _static_rates()
    if not os.environ.get("VAST_API_KEY"):
        return static
    try:
        from flash.providers.vast.jobs import usable_offers

        rates: dict[str, float] = {}
        for offer in usable_offers(0, 0):  # offers are price-sorted, cheapest first
            rates.setdefault(offer.gpu, offer.dph_total)
        return {**static, **rates}
    except Exception as exc:
        logger.warning("live vast pricing unavailable (%s); using static rates", exc)
        return static


def hourly_rate(gpu_name: str) -> float:
    """$/hr for one friendly GPU name (cheapest live offer if available, else static)."""
    from flash.providers.base import canonical_gpu

    name = canonical_gpu(gpu_name)
    return live_rates().get(name) or _static_rates().get(name, 0.0)

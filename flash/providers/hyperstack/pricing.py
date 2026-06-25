"""Hyperstack $/hr: static list price per class (the flavors API carries no price field).

Unlike Lambda's ``/instance-types`` (which carries live prices), Hyperstack's ``/core/flavors``
exposes capacity (``stock_available``) but not price, so rates come from the published list-price
snapshot below. Offline-safe and stable (Hyperstack prices are fixed list rates, not an auction).
"""

from __future__ import annotations

# Hyperstack list prices (snapshot 2026-06-25, hyperstack.cloud/gpu-pricing), per single GPU.
_STATIC_RATES: dict[str, float] = {
    "RTX A6000": 0.50,
    "L40": 1.00,
    "A100 PCIe": 1.35,
    "H100": 1.90,
    "RTX Pro 6000": 1.80,
}


def hourly_rate(gpu_name: str) -> float:
    """$/hr for one friendly GPU name on Hyperstack (static list price)."""
    from flash.providers.base import GPU_INFO, canonical_gpu

    name = canonical_gpu(gpu_name)
    # Prefer the Hyperstack snapshot; fall back to the class nominal rate (keeps the call total).
    return _STATIC_RATES.get(name) or GPU_INFO[name].hourly_usd

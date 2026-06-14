"""LIVE (read-only): real Vast.ai offer search — verified-datacenter pool sanity.

Costs nothing (search only). Skipped offline (AUTOSLM_SKIP_NET) or without a key.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        bool(os.environ.get("AUTOSLM_SKIP_NET")) or not os.environ.get("VAST_API_KEY"),
        reason="live vast test needs network + VAST_API_KEY",
    ),
]


def test_live_search_returns_verified_datacenter_rows():
    from autoslm.providers import vast_api

    rows = vast_api.search_offers(int(12 * 1024 * 0.94), min_disk_gb=60)
    assert rows, "no verified-datacenter offers at all (pool empty?)"
    for r in rows:
        assert r.get("verification") == "verified"
        assert r.get("hosting_type") == 1  # datacenter, never consumer
        assert float(r.get("reliability2") or 0) >= 0.95


def test_live_usable_offers_quality_floors():
    from autoslm.flash.gpus import GPU_INFO
    from autoslm.providers.vast import MIN_INET_MBPS, usable_offers

    offers = usable_offers(12, disk_gb=60)
    assert offers, "no usable verified-datacenter offers >= 12 GB VRAM"
    prices = [o.dph_total for o in offers]
    assert prices == sorted(prices)  # cheapest first
    assert prices[0] < 3.0, f"cheapest 12GB-capable offer is suspiciously expensive: {prices[0]}"
    for o in offers:
        assert o.gpu in GPU_INFO  # canonical (Ampere+) classes only
        assert o.vram_gb >= 12
        assert o.disk_space >= 60
        assert o.inet_down >= MIN_INET_MBPS
    # the cheap sub-8B tier we built this for actually exists in the live pool
    assert any(o.vram_gb >= 24 and o.dph_total < 1.0 for o in offers)

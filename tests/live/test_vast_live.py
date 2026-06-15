"""Live, READ-ONLY Vast smoke (opt-in, no instance rental, no spend).

Only read paths: API auth/reachability via the instances list, the verified-datacenter
offer search, and the offer->class mapping / live pricing. Never rents an instance.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


def _require_key():
    if not os.environ.get("VAST_API_KEY"):
        pytest.skip("VAST_API_KEY not set")


def test_vast_auth_and_list_instances():
    """Auth + reachability: listing instances is a read-only account call."""
    _require_key()
    from autoslm.providers.vast import api as vast_api

    instances = vast_api.list_instances()
    assert isinstance(instances, list)  # 200 + a list == authed & reachable


def test_vast_search_offers_read_only():
    """The verified-datacenter offer search returns rows (read-only market query)."""
    _require_key()
    from autoslm.providers.vast import api as vast_api

    # A small VRAM floor so the verified-datacenter pool is non-trivial.
    rows = vast_api.search_offers((16 * 1024), min_disk_gb=40, min_reliability=0.90, limit=8)
    assert isinstance(rows, list)
    if rows:
        r = rows[0]
        assert "gpu_name" in r
        assert "dph_total" in r


def test_vast_usable_offers_and_pricing():
    """usable_offers maps live offers to managed classes; pricing reads the cheapest."""
    _require_key()
    from autoslm.providers import get_provider
    from autoslm.providers.vast.durable import usable_offers

    offers = usable_offers(16, disk_gb=40)
    assert isinstance(offers, list)
    for o in offers[:5]:
        assert o.gpu  # mapped to a canonical managed class
        assert o.dph_total > 0
    # the provider interface returns a rate (cheapest live offer, else static fallback)
    assert get_provider("vast").hourly_rate("RTX 5090") > 0

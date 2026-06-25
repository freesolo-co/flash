"""Live, READ-ONLY Hyperstack smoke (opt-in, no GPU rental, no spend).

Read paths only: API auth/reachability (regions) and the NEW weight-cache volume list
(``list_volumes`` — the call ``launch_and_submit`` makes to create-if-absent the per-region cache).
Never launches a VM or creates a volume.

The create -> ensure(idempotent) -> delete lifecycle of the cache volume was verified live
out-of-band (near-zero cost, confirmed deleted, no stranded resources); this committed smoke keeps
the read surface reproducible via FLASH_LIVE=1.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


def _require_key():
    if not os.environ.get("HYPERSTACK_API_KEY"):
        pytest.skip("HYPERSTACK_API_KEY not set")


def test_hyperstack_auth_and_list_regions():
    """Auth + reachability: listing regions is a read-only account call."""
    _require_key()
    from flash.providers.hyperstack import api as hs_api

    regions = hs_api._regions()
    assert isinstance(regions, list)  # 200 == authed & reachable
    assert regions  # non-empty region list


def test_hyperstack_list_volumes_is_reachable():
    """The weight-cache read path (list_volumes) authenticates and returns a list."""
    _require_key()
    from flash.providers.hyperstack import api as hs_api

    vols = hs_api.list_volumes()
    assert isinstance(vols, list)  # read-only; the cache create/delete lifecycle is exercised offline


def test_hyperstack_provider_hourly_rate():
    """The provider interface returns a positive static $/hr for a known class."""
    _require_key()
    from flash.providers import get_provider

    assert get_provider("hyperstack").hourly_rate("L40") > 0

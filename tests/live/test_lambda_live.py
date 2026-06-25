"""Live, READ-ONLY Lambda smoke (opt-in, no GPU rental, no spend).

Read paths only: API auth/reachability (instance types) and the NEW weight-cache filesystem list
(``list_filesystems`` — the call ``launch_and_submit`` makes to create-if-absent the per-region
cache). Never launches an instance or creates a filesystem.

The create -> ensure(idempotent) -> delete lifecycle of the cache filesystem was verified live
out-of-band (near-zero cost, confirmed deleted, no stranded resources); this committed smoke keeps
the read surface reproducible via FLASH_LIVE=1.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


def _require_key():
    if not os.environ.get("LAMBDA_API_KEY"):
        pytest.skip("LAMBDA_API_KEY not set")


def test_lambda_auth_and_list_instance_types():
    """Auth + reachability: listing instance types is a read-only account call."""
    _require_key()
    from flash.providers.lambdalabs import api as lambda_api

    types = lambda_api.list_instance_types()
    assert isinstance(types, dict)  # 200 + a dict == authed & reachable


def test_lambda_list_filesystems_is_reachable():
    """The weight-cache read path (list_filesystems) authenticates and returns a list."""
    _require_key()
    from flash.providers.lambdalabs import api as lambda_api

    fses = lambda_api.list_filesystems()
    assert isinstance(fses, list)  # read-only; the cache create/delete lifecycle is exercised offline


def test_lambda_provider_hourly_rate():
    """The provider interface returns a positive static $/hr for a known class."""
    _require_key()
    from flash.providers import get_provider

    assert get_provider("lambda").hourly_rate("A10") > 0

"""Live, READ-ONLY RunPod smoke (opt-in, no GPU rental, no spend).

Only read paths: API auth/reachability via the endpoints list, the GPU-types pricing
route, and the live-rate fetch with its static fallback. Never deploys/runs anything.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


def _require_key():
    if not os.environ.get("RUNPOD_API_KEY"):
        pytest.skip("RUNPOD_API_KEY not set")


def test_runpod_auth_and_list_endpoints():
    """Auth + reachability: listing endpoints is a read-only account call."""
    _require_key()
    from flash.providers.runpod import api as runpod_api

    endpoints = runpod_api.list_endpoints()
    assert isinstance(endpoints, list)  # 200 + a (possibly empty) list == authed & reachable


def test_runpod_live_gpu_pricing():
    """The GPU-types pricing route returns $/hr for at least one managed class."""
    _require_key()
    from flash.providers.runpod.pricing import _fetch_live_rates

    rates = _fetch_live_rates()
    assert isinstance(rates, dict)
    # Live pricing is best-effort per class; at least one managed class should resolve.
    assert any(v > 0 for v in rates.values()), f"no live RunPod rates returned: {rates}"


def test_runpod_provider_hourly_rate():
    """The provider interface returns a positive $/hr (live or static fallback)."""
    _require_key()
    from flash.providers import get_provider

    rate = get_provider("runpod").hourly_rate("RTX 5090")
    assert rate > 0

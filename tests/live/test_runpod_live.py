"""Live, READ-ONLY RunPod smoke (opt-in, no GPU rental, no spend).

Only read paths: API auth/reachability via the endpoints list and the provider's static
pricing interface. Never deploys/runs anything.
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


def test_runpod_provider_hourly_rate():
    """The provider interface returns a positive static $/hr."""
    _require_key()
    from flash.providers import get_provider

    rate = get_provider("runpod").hourly_rate("RTX 5090")
    assert rate > 0

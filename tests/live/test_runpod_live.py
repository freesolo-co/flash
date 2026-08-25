"""Live, read-only RunPod smoke for Pod inventory and static pricing."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


def _require_key():
    if not os.environ.get("RUNPOD_API_KEY"):
        pytest.skip("RUNPOD_API_KEY not set")


def test_runpod_auth_and_list_pods():
    """Authenticate every configured account through the read-only Pod inventory."""
    _require_key()
    from flash.providers.runpod.client import pods as runpod_pods

    pods_by_account, failed = runpod_pods.list_pods_by_key()
    assert failed == []
    assert isinstance(pods_by_account, dict)


def test_runpod_provider_hourly_rate():
    """The provider interface returns a positive static $/hr."""
    _require_key()
    from flash.providers.core.registry import get_provider

    rate = get_provider("runpod").hourly_rate("RTX 5090")
    assert rate > 0

"""Opt-in gate for the live, READ-ONLY provider smokes.

These hit the real RunPod / Vast APIs, so they are skipped unless ``AUTOSLM_LIVE=1``
(and the relevant provider key is present). They are strictly read-only — auth/health,
list GPU types / offers, fetch live pricing — and NEVER provision or rent anything.

Run them with:
    set -a && . /home/azureuser/workspace/.env && set +a && \
        AUTOSLM_LIVE=1 uv run pytest tests/live -q -m live
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _require_live_optin():
    if os.environ.get("AUTOSLM_LIVE") != "1":
        pytest.skip("live smoke disabled; set AUTOSLM_LIVE=1 (and provider creds) to run")
    # The live path must NOT be short-circuited by the offline marker.
    if os.environ.get("AUTOSLM_SKIP_NET"):
        pytest.skip("AUTOSLM_SKIP_NET set; live tests need the network")

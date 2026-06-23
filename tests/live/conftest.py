"""Opt-in gate for the live, READ-ONLY provider smokes.

These hit the real RunPod / Vast APIs, so they are skipped unless ``FLASH_LIVE=1``
(and the relevant provider key is present). They are strictly read-only — auth/health
and list GPU types / offers — and NEVER provision or rent anything.

Run them with:
    set -a && . /home/azureuser/workspace/.env && set +a && \
        FLASH_LIVE=1 uv run pytest tests/live -q -m live
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _require_live_optin():
    if os.environ.get("FLASH_LIVE") != "1":
        pytest.skip("live smoke disabled; set FLASH_LIVE=1 (and provider creds) to run")

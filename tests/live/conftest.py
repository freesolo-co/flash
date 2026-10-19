"""Opt-in gate for the live, READ-ONLY provider smokes.

These hit the real provider APIs, so they are skipped unless ``FLASH_LIVE=1``
(and the relevant provider key is present). They are strictly read-only — auth/health
and list calls — and NEVER provision or rent anything.

Run them with:
    set -a && . /home/azureuser/workspace/.env && set +a && \
        FLASH_LIVE=1 uv run pytest tests/live -q -m live
"""

from __future__ import annotations

import os

import pytest

_KEY_SNAPSHOT = {k: os.environ.get(k) for k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")}


@pytest.fixture(autouse=True)
def _require_live_optin(monkeypatch):
    if os.environ.get("FLASH_LIVE") != "1":
        pytest.skip("live smoke disabled; set FLASH_LIVE=1 (and provider creds) to run")
    # Runs AFTER the root _offline fixture (parent conftest), so restore the keys it deleted.
    for k, v in _KEY_SNAPSHOT.items():
        if v:
            monkeypatch.setenv(k, v)

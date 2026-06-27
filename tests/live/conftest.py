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

# The root ``tests/conftest._offline`` autouse fixture DELETES LAMBDA_API_KEY
# so the offline suite stays hermetic. Snapshot them at import time (before any fixture runs) so the
# live fixture below can restore them — otherwise every instance-provider live smoke would skip even
# with the keys sourced into the shell.
_KEY_SNAPSHOT = {
    k: os.environ.get(k)
    for k in ("LAMBDA_API_KEY", "RUNPOD_API_KEY")
}


@pytest.fixture(autouse=True)
def _require_live_optin(monkeypatch):
    if os.environ.get("FLASH_LIVE") != "1":
        pytest.skip("live smoke disabled; set FLASH_LIVE=1 (and provider creds) to run")
    # Runs AFTER the root _offline fixture (parent conftest), so restore the keys it deleted.
    for k, v in _KEY_SNAPSHOT.items():
        if v:
            monkeypatch.setenv(k, v)

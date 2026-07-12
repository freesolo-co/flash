"""Regression: /v1/health must be an async coroutine so the liveness probe is never starved.

Incident 2026-07-02 (see ISSUE.md): flash flapped Docker-"unhealthy" -> 502 for ~2h while the
process was alive. Root cause: `/v1/health` was a sync `def` handler, so it shared the single
anyio threadpool CapacityLimiter (default 40 tokens) with every other sync handler. When slow
sync handlers (chat->Modal 30-min timeout, auth/billing/HF calls) saturated that limiter under
upstream degradation, the trivial health handler was queued and its HEALTHCHECK timed out.

Declaring the handler `async def` makes it run on the event loop and never take a threadpool
token, so it stays responsive regardless of sync-handler load. These tests lock that in.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("fastapi")
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from flash import __version__
from flash.server.routes import meta


def test_health_handler_is_coroutine():
    # If this reverts to a plain `def`, the handler rejoins the shared sync threadpool and can be
    # starved into HEALTHCHECK timeouts again — the exact 2026-07-02 outage. Keep it async.
    assert inspect.iscoroutinefunction(meta.health), (
        "/v1/health must be `async def` so the liveness probe never acquires an anyio "
        "threadpool token and cannot be starved by slow sync handlers"
    )


def test_health_returns_ok():
    app = FastAPI()
    router = APIRouter()
    router.add_api_route("/v1/health", meta.health, methods=["GET"])
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "service": "flash", "version": __version__}

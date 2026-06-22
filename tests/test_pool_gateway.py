"""Gateway error-handling tests: which control-op HTTP failures are idempotent no-ops vs real errors.

Driven against tiny in-process ASGI apps via ``httpx.ASGITransport`` (no sockets, no torch) so the
exact status/body the backend returns is under test. The contract: vLLM returns a 400 whose body
says "already ... loaded" / "not found" on an idempotent re-load / double-unload (benign), but a
bare 404 means the dynamic-LoRA route itself is missing (wrong URL / a vLLM without the API) and
MUST surface as an error — otherwise the router believes an adapter is loaded and fails at generation.

Following the repo convention (see test_pool_router.py), async coroutines are driven with
``asyncio.run`` inside sync test functions rather than a pytest-asyncio marker.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response

from flash.pool.gateway import BackendGateway, GatewayError
from flash.pool.state import Backend


def _gateway_for(app: FastAPI) -> BackendGateway:
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://backend")
    gw = BackendGateway(client=client)
    # This client is created solely for the test gateway, so let gw.aclose() close it (production
    # aclose() only closes a client it OWNS, i.e. one it created — an injected client is the caller's).
    gw._owns_client = True
    return gw


def _backend() -> Backend:
    return Backend(id="b", url="http://backend", base_model="Q")


def test_load_lora_tolerates_already_loaded_400():
    app = FastAPI()

    @app.post("/v1/load_lora_adapter")
    async def load(body: dict) -> dict:
        raise HTTPException(status_code=400, detail=f"lora {body['lora_name']} has already been loaded")

    async def scenario():
        gw = _gateway_for(app)
        try:
            await gw.load_lora(_backend(), "run", "/lora/run")  # must NOT raise
        finally:
            await gw.aclose()

    asyncio.run(scenario())  # no GatewayError -> the idempotent 400 is tolerated


def test_load_lora_400_not_found_does_not_swallow():
    # load tolerates "already loaded" but NOT "not found": a 400 "not found" on LOAD means a bad LoRA
    # path/URI (a real failure) — swallowing it would let the router believe the adapter loaded.
    app = FastAPI()

    @app.post("/v1/load_lora_adapter")
    async def load(body: dict) -> dict:
        raise HTTPException(status_code=400, detail=f"lora path {body['lora_name']} not found")

    async def scenario():
        gw = _gateway_for(app)
        try:
            with pytest.raises(GatewayError):
                await gw.load_lora(_backend(), "run", "/bad/path")
        finally:
            await gw.aclose()

    asyncio.run(scenario())


def test_unload_lora_tolerates_not_found_400():
    app = FastAPI()

    @app.post("/v1/unload_lora_adapter")
    async def unload(body: dict) -> dict:
        raise HTTPException(status_code=400, detail=f"lora {body['lora_name']} not found")

    async def scenario():
        gw = _gateway_for(app)
        try:
            await gw.unload_lora(_backend(), "run")  # double-unload is a no-op
        finally:
            await gw.aclose()

    asyncio.run(scenario())  # no GatewayError -> the idempotent 400 is tolerated


def test_load_lora_404_missing_route_raises():
    # The dynamic-LoRA endpoint is absent (wrong backend URL / vLLM without the API). A 404 here is a
    # real misconfig and must raise — NOT be swallowed as an idempotent success.
    app = FastAPI()  # no /v1/load_lora_adapter route -> FastAPI returns 404

    async def scenario():
        gw = _gateway_for(app)
        try:
            await gw.load_lora(_backend(), "run", "/lora/run")
        finally:
            await gw.aclose()

    with pytest.raises(GatewayError) as ei:
        asyncio.run(scenario())
    assert ei.value.status == 404


def test_unload_lora_404_missing_route_raises():
    app = FastAPI()  # no unload route

    async def scenario():
        gw = _gateway_for(app)
        try:
            await gw.unload_lora(_backend(), "run")
        finally:
            await gw.aclose()

    with pytest.raises(GatewayError) as ei:
        asyncio.run(scenario())
    assert ei.value.status == 404


@pytest.mark.parametrize("status", [401, 403, 500, 502])
def test_load_lora_does_not_swallow_auth_or_server_errors_with_already_in_body(status):
    # A real auth/server failure (401/403/5xx) whose body happens to contain "already" must NOT be
    # treated as an idempotent no-op just because of the substring — the tolerance is gated on the
    # vLLM 400 contract only. Swallowing these would let the router believe the adapter is loaded and
    # then fail every generation against this backend.
    app = FastAPI()

    @app.post("/v1/load_lora_adapter")
    async def load(body: dict) -> dict:
        raise HTTPException(status_code=status, detail="lora has already been loaded (but auth failed)")

    async def scenario():
        gw = _gateway_for(app)
        try:
            await gw.load_lora(_backend(), "run", "/lora/run")
        finally:
            await gw.aclose()

    with pytest.raises(GatewayError) as ei:
        asyncio.run(scenario())
    assert ei.value.status == status


def test_chat_non_json_2xx_body_raises():
    # A JSON-expecting call (chat/completions, reward worker) that gets a 200 carrying a NON-JSON
    # body (HTML error page, proxy interstitial, truncated output) must FAIL FAST: blanket-succeeding
    # it would let the router treat a broken backend as a real result and surface a confusing
    # downstream error instead of engaging retry/failover.
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat() -> Response:
        return PlainTextResponse("<html>502 Bad Gateway</html>")  # 200 status, HTML body

    async def scenario():
        gw = _gateway_for(app)
        try:
            await gw.chat(_backend(), {"model": "Q", "messages": []})
        finally:
            await gw.aclose()

    with pytest.raises(GatewayError) as ei:
        asyncio.run(scenario())
    assert ei.value.status == 200
    assert "not JSON" in str(ei.value)
    assert "502 Bad Gateway" in str(ei.value)  # the real body is surfaced to the operator


def test_reward_worker_non_json_2xx_body_raises():
    # Same fail-fast contract for the reward-worker path (BackendGateway.post).
    app = FastAPI()

    @app.post("/score")
    async def score() -> Response:
        return PlainTextResponse("not json")

    async def scenario():
        gw = _gateway_for(app)
        try:
            await gw.post("reward", "http://backend/score", {"x": 1})
        finally:
            await gw.aclose()

    with pytest.raises(GatewayError):
        asyncio.run(scenario())


def test_load_lora_tolerates_empty_2xx_body():
    # Control ops (load/unload) may legitimately get an EMPTY 200 from vLLM's dynamic-LoRA endpoint
    # — that is a success, not a misconfigured backend. A truly-empty 2xx body must NOT raise.
    app = FastAPI()

    @app.post("/v1/load_lora_adapter")
    async def load(body: dict) -> Response:
        return Response(status_code=200)  # empty body, no JSON

    async def scenario():
        gw = _gateway_for(app)
        try:
            await gw.load_lora(_backend(), "run", "/lora/run")  # must NOT raise
        finally:
            await gw.aclose()

    asyncio.run(scenario())


def test_load_lora_nonempty_non_json_2xx_body_raises():
    # But a NON-empty non-JSON 2xx on a control op is still a misconfigured backend (wrong content
    # behind a success status) and must surface — the empty-body tolerance is exactly that narrow.
    app = FastAPI()

    @app.post("/v1/load_lora_adapter")
    async def load(body: dict) -> Response:
        return PlainTextResponse("<html>proxy</html>")

    async def scenario():
        gw = _gateway_for(app)
        try:
            await gw.load_lora(_backend(), "run", "/lora/run")
        finally:
            await gw.aclose()

    with pytest.raises(GatewayError):
        asyncio.run(scenario())


@pytest.mark.parametrize("status", [401, 500])
def test_unload_lora_does_not_swallow_auth_or_server_errors_with_not_found_in_body(status):
    # Same gating for unload: a 401/500 carrying "not found" is a real error, not a benign
    # double-unload.
    app = FastAPI()

    @app.post("/v1/unload_lora_adapter")
    async def unload(body: dict) -> dict:
        raise HTTPException(status_code=status, detail="lora not found (server error)")

    async def scenario():
        gw = _gateway_for(app)
        try:
            await gw.unload_lora(_backend(), "run")
        finally:
            await gw.aclose()

    with pytest.raises(GatewayError) as ei:
        asyncio.run(scenario())
    assert ei.value.status == status

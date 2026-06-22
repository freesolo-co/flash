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

from flash.pool.gateway import BackendGateway, GatewayError
from flash.pool.state import Backend


def _gateway_for(app: FastAPI) -> BackendGateway:
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://backend")
    return BackendGateway(client=client)


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

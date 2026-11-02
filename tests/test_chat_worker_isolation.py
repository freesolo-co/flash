"""Slow inference must not consume the API's authentication/status worker budget."""

import asyncio
import threading

import anyio
import httpx
from fastapi import FastAPI

from flash.server.platform.deps import require_key
from flash.server.routes import meta, serving, serving_chat


def test_forty_waiting_chats_do_not_block_authentication_or_another_chat(monkeypatch):
    started = 0
    lock = threading.Lock()
    release = threading.Event()
    ready = threading.Event()

    def forward(_run_id, payload, *_args):
        nonlocal started
        if payload.get("slow"):
            with lock:
                started += 1
                if started == 40:
                    ready.set()
            assert release.wait(10), "test did not release the pending model responses"
        return {"ok": True}

    monkeypatch.setattr(serving_chat, "managed_chat", forward)
    app = FastAPI()
    app.include_router(serving.router)
    app.include_router(meta.router)
    # A sync auth dependency exercises the same default pool as production auth.
    app.dependency_overrides[require_key] = lambda: {"id": 1, "key_prefix": "test"}

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            pending = [
                asyncio.create_task(client.post("/v1/runs/run/chat", json={"slow": True}))
                for _ in range(40)
            ]
            try:
                assert await asyncio.to_thread(ready.wait, 5)
                # On the old route both requests are stuck behind all forty model responses.
                async with asyncio.timeout(2):
                    status = await client.get("/v1/me")
                    fast = await client.post("/v1/runs/run/chat", json={})
                assert status.status_code == fast.status_code == 200
            finally:
                release.set()
                await asyncio.gather(*pending)

    asyncio.run(exercise())


def test_chat_forwarding_remains_bounded_without_blocking_status(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def forward(_run_id, payload, *_args):
        calls.append(payload)
        if payload.get("slow"):
            entered.set()
            assert release.wait(10)
        return {"ok": True}

    monkeypatch.setattr(serving_chat, "managed_chat", forward)
    app = FastAPI()
    app.include_router(serving.router)
    app.include_router(meta.router)
    app.dependency_overrides[require_key] = lambda: {"id": 1, "key_prefix": "test"}

    async def exercise():
        app.state.chat_forwarding_limiter = anyio.CapacityLimiter(1)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(client.post("/v1/runs/run/chat", json={"slow": True}))
            second = None
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                second = asyncio.create_task(client.post("/v1/runs/run/chat", json={}))
                async with asyncio.timeout(2):
                    await anyio.wait_all_tasks_blocked()
                    assert app.state.chat_forwarding_limiter.statistics().tasks_waiting == 1
                    assert (await client.get("/v1/me")).status_code == 200
                assert len(calls) == 1
            finally:
                release.set()
                await asyncio.gather(first, *([second] if second else []))
            assert len(calls) == 2

    asyncio.run(exercise())

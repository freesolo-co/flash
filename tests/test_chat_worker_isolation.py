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


def _slow_chat_app(monkeypatch, release, entered, target):
    calls = []
    lock = threading.Lock()

    def forward(_run_id, payload, *_args):
        with lock:
            calls.append(payload)
            if payload.get("slow") and len(calls) >= target:
                entered.set()
        if payload.get("slow"):
            assert release.wait(10)
        return {"ok": True}

    monkeypatch.setattr(serving_chat, "managed_chat", forward)
    app = FastAPI()
    app.include_router(serving.router)
    app.include_router(meta.router)
    app.dependency_overrides[require_key] = lambda: {"id": 1, "key_prefix": "test"}
    return app, calls


def test_saturated_chat_sheds_excess_with_fast_429_while_status_answers(monkeypatch):
    # Production saw ~1,800 sockets queued behind 128 slow forwards and no completions;
    # excess chats must be refused immediately so callers back off instead of piling up.
    monkeypatch.setenv("FLASH_CHAT_FORWARDING_CONCURRENCY", "4")
    monkeypatch.setenv("FLASH_CHAT_QUEUE_LIMIT", "2")
    release, entered = threading.Event(), threading.Event()
    app, calls = _slow_chat_app(monkeypatch, release, entered, target=4)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            pending = [
                asyncio.create_task(client.post("/v1/runs/run/chat", json={"slow": True}))
                for _ in range(6)
            ]
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                async with asyncio.timeout(2):
                    await anyio.wait_all_tasks_blocked()
                    stats = app.state.chat_forwarding_limiter.statistics()
                    assert (stats.borrowed_tokens, stats.tasks_waiting) == (4, 2)
                    shed = await client.post("/v1/runs/run/chat", json={"slow": True})
                    status = await client.get("/v1/me")
                assert shed.status_code == 429
                assert shed.headers["Retry-After"] == "5"
                assert status.status_code == 200
                assert len(calls) == 4
            finally:
                release.set()
                responses = await asyncio.gather(*pending)
            assert [r.status_code for r in responses] == [200] * 6

    asyncio.run(exercise())


def test_queued_chat_gives_up_with_503_after_bounded_wait(monkeypatch):
    monkeypatch.setenv("FLASH_CHAT_FORWARDING_CONCURRENCY", "1")
    monkeypatch.setenv("FLASH_CHAT_QUEUE_WAIT_SECONDS", "0.2")
    release, entered = threading.Event(), threading.Event()
    app, calls = _slow_chat_app(monkeypatch, release, entered, target=1)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(client.post("/v1/runs/run/chat", json={"slow": True}))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                async with asyncio.timeout(2):
                    queued = await client.post("/v1/runs/run/chat", json={})
                assert queued.status_code == 503
                assert queued.headers["Retry-After"] == "5"
                assert len(calls) == 1
            finally:
                release.set()
                assert (await first).status_code == 200
            assert app.state.chat_forwarding_limiter.statistics().borrowed_tokens == 0

    asyncio.run(exercise())

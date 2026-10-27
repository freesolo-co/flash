"""Admission is fenced while previously accepted streams finish cleanup."""

import asyncio

import httpx
import pytest

from flash.serving.deepinfer.app.http import build_app
from tests.serving.test_deepinfer import FakeRuntime, body


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_drain_rejects_new_work_and_waits_for_stream_cleanup():
    started, finish, cleaning, cleaned = (asyncio.Event() for _ in range(4))

    class Runtime(FakeRuntime):
        async def generate(self, request, tokens):
            self.calls += 1
            started.set()
            try:
                yield {"tokens": [2], "finish_reason": None}
                await finish.wait()
                yield {"tokens": [3, 1], "finish_reason": "stop"}
            finally:
                cleaning.set()
                await cleaned.wait()

    runtime = Runtime()
    transport = httpx.ASGITransport(app=build_app(runtime, manage_lifecycle=False))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"request_id": "old", "chat": body()}
        stream = asyncio.create_task(client.post("/internal/generate", json=payload))
        await asyncio.wait_for(started.wait(), 2)
        try:
            response = await client.post("/internal/drain")
            assert response.json() == {"draining": True, "active_http_requests": 1}
            rejected = await client.post("/internal/generate", json=payload)
            assert rejected.status_code == 503
            assert rejected.headers["X-DeepInfer-Retryable"] == "not-started"
            assert runtime.calls == 1
            finish.set()
            await asyncio.wait_for(cleaning.wait(), 2)
            assert (await client.post("/internal/drain")).json()["active_http_requests"] == 1
        finally:
            finish.set()
            cleaned.set()
        response = await asyncio.wait_for(stream, 2)
        assert response.status_code == 200
        assert '"type": "final"' in response.text
        assert (await client.post("/internal/drain")).json()["active_http_requests"] == 0


@pytest.mark.anyio
async def test_capacity_rejection_is_marked_safe_without_entering_runtime(monkeypatch):
    monkeypatch.setattr("flash.serving.deepinfer.app.http.MAX_INPUTS", 0)
    runtime = FakeRuntime()
    transport = httpx.ASGITransport(app=build_app(runtime, manage_lifecycle=False))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/internal/generate", json={"request_id": "no", "chat": body()}
        )
    assert response.status_code == 429
    assert response.headers["X-DeepInfer-Retryable"] == "not-started"
    assert runtime.calls == 0


@pytest.mark.anyio
async def test_restarted_replica_restores_its_drain_gate_before_listening(monkeypatch):
    from tests.serving.test_deepinfer_fleet import Store

    class Runtime(FakeRuntime):
        async def start(self):
            pass

        async def close(self):
            self.closed = True

    store = Store(fleet={"replicas": [{"name": "old", "phase": "draining"}]})
    monkeypatch.setenv("DEEPINFER_REPLICA_NAME", "old")
    monkeypatch.setattr("modal.Dict.from_name", lambda name: store)
    runtime = Runtime()
    app = build_app(runtime)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/internal/generate", json={})
        assert response.status_code == 503
        assert response.headers["X-DeepInfer-Retryable"] == "not-started"
        assert (await client.get("/health")).json()["draining"] is True
    assert runtime.closed
    assert runtime.calls == 0

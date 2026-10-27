"""HTTP deadlines include preparation and delivery, with time left for cleanup."""

import asyncio

import httpx
import pytest
from anyio import CancelScope

from flash.serving.deepinfer.app.http import _GenerationCount, build_app
from flash.serving.deepinfer.contract.config import DRAIN_SECONDS, REQUEST_SECONDS
from tests.serving.test_deepinfer import FakeRuntime, body


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def advance(monkeypatch):
    loop = asyncio.get_running_loop()
    clock = loop.time
    elapsed = [0]
    monkeypatch.setattr(loop, "time", lambda: clock() + elapsed[0])
    yield lambda seconds: elapsed.__setitem__(0, seconds)
    elapsed[0] = 0


def test_request_budget_leaves_five_minutes_inside_modal_draining():
    assert REQUEST_SECONDS == 55 * 60
    assert DRAIN_SECONDS - REQUEST_SECONDS == 5 * 60


@pytest.mark.anyio
async def test_preparation_cannot_start_a_fresh_generation_budget(advance):
    preparing = asyncio.Event()

    class Runtime(FakeRuntime):
        async def prepare_tokens(self, request):
            preparing.set()
            await asyncio.Event().wait()

    runtime = Runtime()
    transport = httpx.ASGITransport(app=build_app(runtime, manage_lifecycle=False))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task = asyncio.create_task(
            client.post("/internal/generate", json={"request_id": "slow-prepare", "chat": body()})
        )
        await preparing.wait()
        advance(REQUEST_SECONDS + 1)
        response = await asyncio.wait_for(task, 2)
        assert response.status_code == 504
        assert "X-DeepInfer-Retryable" not in response.headers
        assert runtime.calls == 0
        assert (await client.get("/health")).json()["active_http_requests"] == 0


@pytest.mark.anyio
@pytest.mark.parametrize("sse", [False, True])
async def test_started_stream_times_out_and_keeps_its_slot_until_cleanup(advance, sse):
    streaming, cleaning, release = (asyncio.Event() for _ in range(3))

    class Runtime(FakeRuntime):
        async def generate(self, request, tokens):
            self.calls += 1
            try:
                yield {"tokens": [2], "finish_reason": None}
                streaming.set()
                await asyncio.Event().wait()
            finally:
                # Mirror Runtime.generate's shielded native abort handshake.
                with CancelScope(shield=True):
                    cleaning.set()
                    await release.wait()
                    self.closed = True

    runtime = Runtime()
    transport = httpx.ASGITransport(app=build_app(runtime, manage_lifecycle=False))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task = asyncio.create_task(
            client.post(
                "/internal/generate",
                json={"request_id": "stream", "chat": body()},
                headers={"Accept": "text/event-stream" if sse else "application/x-ndjson"},
            )
        )
        await streaming.wait()
        advance(REQUEST_SECONDS + 1)
        try:
            await asyncio.wait_for(cleaning.wait(), 2)
            assert (await client.get("/health")).json()["active_http_requests"] == 1
        finally:
            release.set()
        response = await asyncio.wait_for(task, 2)
        assert response.status_code == 200  # Headers already went out; error terminates the stream.
        assert '"type": "ready"' in response.text
        expected = 'data: {"type":"error"}\n\n' if sse else '{"type":"error"}\n'
        assert response.text.endswith(expected)
        assert '"type": "final"' not in response.text
        assert runtime.closed
        assert (await client.get("/health")).json()["active_http_requests"] == 0


@pytest.mark.anyio
@pytest.mark.parametrize("stalled", ["upload", "delivery"])
async def test_http_deadline_covers_body_upload_and_backpressure(advance, stalled):
    blocked = asyncio.Event()
    messages = []
    counts = {"active": 0}

    async def receive():
        blocked.set()
        await asyncio.Event().wait()

    async def send(message):
        if message.get("body") == b"blocked":
            blocked.set()
            await asyncio.Event().wait()
        messages.append(message)

    async def app(scope, receive, send):
        if stalled == "upload":
            await receive()
        else:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"blocked", "more_body": True})

    task = asyncio.create_task(
        _GenerationCount(app, counts)(
            {"type": "http", "method": "POST", "path": "/internal/generate"}, receive, send
        )
    )
    await blocked.wait()
    advance(REQUEST_SECONDS + 1)
    await asyncio.wait_for(task, 2)
    assert messages[0]["status"] == (504 if stalled == "upload" else 200)
    assert messages[-1].get("more_body", False) is False
    assert counts["active"] == 0


@pytest.mark.anyio
async def test_capacity_rejection_is_safe_to_retry_before_entering_runtime(monkeypatch):
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

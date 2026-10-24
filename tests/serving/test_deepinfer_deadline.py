"""Generation deadlines survive task handoffs without releasing live native work."""

import asyncio
import queue
from types import SimpleNamespace

import pytest

from flash.serving.deepinfer.contract.config import MODEL_ID, REQUEST_SECONDS
from flash.serving.deepinfer.contract.protocol import parse_request
from flash.serving.deepinfer.runtime.bridge import Runtime
from flash.serving.src.http.stream_response import await_until_disconnect


@pytest.mark.anyio
@pytest.mark.parametrize("buffered", [False, True])
async def test_generation_deadline_survives_consumer_handoff(monkeypatch, buffered):
    loop = asyncio.get_running_loop()
    clock = loop.time
    elapsed = 0
    monkeypatch.setattr(loop, "time", lambda: clock() + elapsed)
    runtime = Runtime()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class Commands(queue.Queue):
        def put_nowait(self, command):
            super().put_nowait(command)
            if command["type"] == "generate":
                started.set()
            if command["type"] == "cancel":
                cancelled.set()

    runtime.incoming = [Commands() for _ in range(4)]
    request = parse_request({"model": MODEL_ID, "messages": [{"role": "user", "content": "hello"}]})
    generator = runtime.generate(request, [0])

    async def connected():
        await asyncio.Event().wait()

    first = asyncio.create_task(
        await_until_disconnect(SimpleNamespace(receive=connected), anext(generator))
    )
    await started.wait()
    request_id, pending = next(iter(runtime.pending.items()))
    event = {"type": "tokens", "id": request_id, "tokens": [2], "finish_reason": None}
    runtime._deliver(event)
    assert await first == event

    # A second consumer still gets events within the original generation budget.
    elapsed = REQUEST_SECONDS / 2
    runtime._deliver(event)
    assert await asyncio.create_task(anext(generator)) == event
    if buffered:
        runtime._deliver(event)
        elapsed = REQUEST_SECONDS + 1
    consumer = asyncio.create_task(anext(generator))
    await asyncio.sleep(0)
    elapsed = REQUEST_SECONDS + 1
    try:
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        assert runtime.loads[pending.rank] == 1
        assert request_id in runtime.pending
        runtime._deliver({"type": "cancelled", "id": request_id})
        with pytest.raises(TimeoutError):
            await consumer
        assert runtime.loads == [0] * 4
        assert runtime.pending == {}
        assert runtime.failure is None
    finally:
        pending.cancelled.set()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        await generator.aclose()

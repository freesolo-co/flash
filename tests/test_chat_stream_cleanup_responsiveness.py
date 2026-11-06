"""A stalled upstream close must not stall the control-plane event loop."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi.responses import StreamingResponse

from flash.server.routes.meta import health
from flash.server.routes.serving_chat import _UpstreamStreamingResponse


@pytest.mark.parametrize("disconnected", [False, True])
def test_health_responds_while_upstream_close_waits(monkeypatch, disconnected):
    entered = threading.Event()
    release = threading.Event()
    close_threads = []

    def close():
        close_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(3), "event loop did not release stalled cleanup"

    async def send_response(self, scope, receive, send):
        if disconnected:
            raise OSError("client disconnected")

    monkeypatch.setattr(StreamingResponse, "__call__", send_response)

    async def exercise():
        response = _UpstreamStreamingResponse(iter(()), upstream=SimpleNamespace(close=close))
        task = asyncio.create_task(response({}, None, None))
        try:
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 1), 2)
            assert (await asyncio.wait_for(health(), 0.2))["ok"]
            assert len(close_threads) == 1
            assert close_threads[0] != threading.get_ident()
        finally:
            release.set()
            if disconnected:
                with pytest.raises(OSError, match="client disconnected"):
                    await task
            else:
                await task

    asyncio.run(exercise())

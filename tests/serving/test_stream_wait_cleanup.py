import asyncio

import pytest

from flash.serving.src.io.openai_stream import _next_event_or_disconnect


def test_cancelled_stream_wait_stops_pending_engine_read():
    async def run():
        entered = asyncio.Event()
        closed = asyncio.Event()

        async def events():
            try:
                entered.set()
                await asyncio.Event().wait()
                yield {"type": "delta", "text": "unreachable"}
            finally:
                closed.set()

        source = events()
        disconnect = asyncio.create_task(asyncio.Event().wait())
        waiter = asyncio.create_task(_next_event_or_disconnect(source, disconnect))
        try:
            await entered.wait()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert closed.is_set(), "the engine read outlived its cancelled consumer"
            await source.aclose()
        finally:
            disconnect.cancel()
            await asyncio.gather(disconnect, return_exceptions=True)

    asyncio.run(run())

"""explicit lifetime ownership for hosted streaming response bodies."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from anyio import CancelScope
from fastapi import Request
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send


class ClosingStreamingResponse(StreamingResponse):
    def __init__(
        self,
        content: AsyncIterator[bytes],
        *,
        on_unstarted: Callable[[], Awaitable[None]] | None = None,
        **kwargs: Any,
    ) -> None:
        self._content = content
        self._started = False
        self._on_unstarted = on_unstarted
        super().__init__(self._iterate(), **kwargs)

    async def _iterate(self) -> AsyncIterator[bytes]:
        self._started = True
        async for chunk in self._content:
            yield chunk

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # own cleanup outside starlette's cancellable streaming child: a task
            # cancelled before its first step never enters its own finally block.
            with CancelScope(shield=True):
                try:
                    close = getattr(self._content, "aclose", None)
                    if close is not None:
                        await close()
                    if not self._started and self._on_unstarted is not None:
                        await self._on_unstarted()
                finally:
                    await self.body_iterator.aclose()


async def await_until_disconnect(request: Request, awaitable: Awaitable[Any]) -> Any:
    operation = asyncio.ensure_future(awaitable)
    disconnect = asyncio.create_task(wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait({operation, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        # `done` is a SET: if generation finished and the peer left in the same tick, both are in
        # it. deciding on the disconnect first discards a completed, durably finalized result.
        # resolve the tie toward the
        # operation, matching the packaged helper in flash/serve/app/http.py.
        if operation in done:
            return operation.result()
        disconnect.result()
        raise asyncio.CancelledError
    finally:
        disconnect.cancel()
        if not operation.done():
            operation.cancel()
        await asyncio.gather(operation, disconnect, return_exceptions=True)


async def wait_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return

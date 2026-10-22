"""explicit lifetime ownership for hosted streaming response bodies."""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from anyio import CancelScope
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

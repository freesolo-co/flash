"""ASGI request-body ceiling applied before FastAPI parses endpoint models."""

from __future__ import annotations

from collections import deque
from typing import Any

from starlette.responses import JSONResponse


class RequestBodyLimitMiddleware:
    def __init__(self, app: Any, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if _declared_oversize(scope.get("headers", []), self.max_bytes):
            await _too_large_response(scope, receive, send)
            return

        buffered: deque[dict[str, Any]] = deque()
        observed = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break
            observed += len(message.get("body", b""))
            if observed > self.max_bytes:
                await _too_large_response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay_receive() -> dict[str, Any]:
            if buffered:
                return buffered.popleft()
            return await receive()

        await self.app(scope, replay_receive, send)


def _declared_oversize(headers: list[tuple[bytes, bytes]], max_bytes: int) -> bool:
    for name, value in headers:
        if name.lower() != b"content-length":
            continue
        try:
            declared = int(value)
        except ValueError:
            continue
        if declared > max_bytes:
            return True
    return False


async def _too_large_response(scope: dict[str, Any], receive: Any, send: Any) -> None:
    response = JSONResponse({"detail": "request body too large"}, status_code=413)
    await response(scope, receive, send)

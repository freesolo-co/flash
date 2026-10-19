"""ASGI request-body ceiling applied before FastAPI parses endpoint models."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from fastapi.responses import JSONResponse


class _RequestBodyTooLarge(Exception):
    pass


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

        observed = 0
        overflowed = False
        rejection_sent = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal observed, overflowed
            message = await receive()
            if message["type"] == "http.request":
                observed += len(message.get("body", b""))
                if observed > self.max_bytes:
                    overflowed = True
                    raise _RequestBodyTooLarge
            return message

        async def limited_send(message: dict[str, Any]) -> None:
            nonlocal rejection_sent
            if not overflowed:
                await send(message)
                return
            if not rejection_sent:
                rejection_sent = True
                await _too_large_response(scope, receive, send)

        with suppress(_RequestBodyTooLarge):
            await self.app(scope, limited_receive, limited_send)
        if overflowed and not rejection_sent:
            await _too_large_response(scope, receive, send)


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
    # closing makes unread bytes unreachable, so do not drain and keep rejection work bounded.
    response = JSONResponse(
        {"detail": "request body too large"},
        status_code=413,
        headers={"connection": "close"},
    )
    await response(scope, receive, send)

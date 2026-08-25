"""configurable streaming transport fakes."""

from __future__ import annotations

from typing import Any


class StreamResponse:
    def __init__(
        self,
        *,
        byte_chunks=(),
        line_chunks=(),
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        json_payload: Any = None,
        status_error: BaseException | None = None,
    ) -> None:
        self.byte_chunks = byte_chunks
        self.line_chunks = line_chunks
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/event-stream"}
        self.json_payload = json_payload
        self.status_error = status_error

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error

    def iter_bytes(self):
        yield from self.byte_chunks

    def iter_lines(self):
        yield from self.line_chunks

    def read(self) -> bytes:
        return b""

    def json(self):
        return self.json_payload


class StreamContext:
    def __init__(self, response: StreamResponse, exits: list[tuple] | None = None) -> None:
        self.response = response
        self.exits = exits if exits is not None else []

    def __enter__(self) -> StreamResponse:
        return self.response

    def __exit__(self, *exc) -> bool:
        self.exits.append(exc)
        return False


class StreamClient:
    def __init__(self, context: StreamContext, seen: dict[str, Any] | None = None) -> None:
        self.context = context
        self.seen = seen if seen is not None else {}

    def stream(self, method: str, url: str, **kwargs):
        self.seen.update({"method": method, "url": url, **kwargs})
        return self.context

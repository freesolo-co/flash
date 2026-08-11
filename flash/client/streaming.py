"""Bounded response reads and upload progress for the control-plane client."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

ProgressCallback = Callable[[int, int], None]

_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_DEADLINE_CHUNK_BYTES = 8 * 1024
_MAX_JSON_RESPONSE_BYTES = 32 * 1024 * 1024


def _capped_timeout(timeout: float, deadline: float | None) -> float:
    """Shorten a socket timeout to the time remaining before a deadline."""
    if deadline is None:
        return timeout
    return min(timeout, max(0.0, deadline - time.monotonic()))


def _read_response_body(
    resp: object,
    *,
    max_bytes: int | None = None,
    deadline: float | None = None,
    path: str = "",
) -> bytes:
    """Read a response body with optional byte and wall-clock bounds."""
    if max_bytes is None and deadline is None:
        return resp.read()  # type: ignore[attr-defined]
    if deadline is not None and time.monotonic() > deadline:
        from flash.client.http import ClientError

        target = f" {path}" if path else ""
        raise ClientError(
            f"the control plane took too long to answer{target} and exceeded its overall deadline"
        )
    return _read_capped_response(
        resp,
        max_bytes if max_bytes is not None else _MAX_JSON_RESPONSE_BYTES,
        deadline=deadline,
    )


def _read_capped_response(resp: object, max_bytes: int, deadline: float | None = None) -> bytes:
    """Read a response body under a byte cap and optional absolute deadline."""
    from flash.client.http import ClientError

    read_size = _DOWNLOAD_CHUNK_BYTES if deadline is None else _DEADLINE_CHUNK_BYTES
    chunks: list[bytes] = []
    total = 0
    while True:
        if deadline is not None and time.monotonic() > deadline:
            raise ClientError(
                f"the control plane's response body stalled after {total} bytes "
                "and exceeded its overall deadline"
            )
        chunk = resp.read(read_size)  # type: ignore[attr-defined]
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ClientError(
                f"response body exceeded the maximum allowed size ({max_bytes} bytes); "
                "download aborted"
            )
        chunks.append(chunk)
    return b"".join(chunks)


class _ProgressReader:
    """File-like bytes wrapper that reports upload progress after each read."""

    def __init__(self, data: bytes, progress: ProgressCallback):
        self._data = data
        self._total = len(data)
        self._pos = 0
        self._progress = progress

    def __len__(self) -> int:
        return self._total

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk = self._data[self._pos :]
        else:
            chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        # a rendering failure must not abort an in-flight upload
        with contextlib.suppress(Exception):
            self._progress(self._pos, self._total)
        return chunk

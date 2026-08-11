"""Body-streaming mechanics for the control-plane client: capped reads and upload progress.

Split out of ``flash.client.http`` to keep that module under the file-size gate. Both halves are
about moving a body in bounded pieces rather than one slurp -- one for a response we read, one for a
request we send -- so they belong together and neither touches client state.

``http`` re-exports both. That matters for ``_ProgressReader`` specifically: tests patch it as
``http._ProgressReader`` and ``_request`` looks it up as a module global, so the re-export is what
keeps that seam live after the move.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

# the single definition. `http` re-exports it, because `cli.commands.env.push` imports the name from
# there; defining it in both places instead would leave two aliases free to drift apart.
ProgressCallback = Callable[[int, int], None]

_DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# an error body is read only to be shown: the caller uses the first few hundred characters as a
# message. Capping it costs nothing that is used, and leaving it uncapped means a drip-fed non-2xx
# body can stall the very code path that exists to REPORT the failure.
_MAX_ERROR_BODY_BYTES = 64 * 1024
_ERROR_BODY_DEADLINE_SECONDS = 20.0
# deliberately smaller than `_DOWNLOAD_CHUNK_BYTES`: the deadline can only be checked BETWEEN reads,
# so the read size is what bounds how long one blocked read can overshoot it. A single
# `HTTPResponse.read(n)` performs many socket receives internally, and urllib's timeout applies to
# each receive rather than to the call, so a peer that stays inside every window keeps one read
# blocked for far longer than the deadline. A smaller size narrows that window; it cannot close it,
# which is why the socket timeout remains the real backstop against a stalled peer.
_ERROR_BODY_CHUNK_BYTES = 8 * 1024


def _read_error_body(exc: object, deadline_seconds: float = _ERROR_BODY_DEADLINE_SECONDS) -> bytes:
    """Read a non-2xx body for display, bounded in bytes AND wall-clock.

    Lenient by design, mirroring the server-side helper of the same name: hitting the cap or the
    deadline truncates the body rather than raising. Losing the tail of an error body must never cost
    the status that classifies the failure -- the status is the useful part.

    A read that FAILS still raises, so the caller can distinguish "unreadable" from "truncated here"
    and say so.
    """
    import time

    fp = getattr(exc, "fp", None)
    if fp is None:
        return b""
    read = exc.read  # type: ignore[attr-defined]
    deadline = time.monotonic() + deadline_seconds
    chunks: list[bytes] = []
    total = 0
    while total < _MAX_ERROR_BODY_BYTES:
        if time.monotonic() > deadline:
            break
        chunk = read(min(_ERROR_BODY_CHUNK_BYTES, _MAX_ERROR_BODY_BYTES - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _read_capped_response(resp: object, max_bytes: int) -> bytes:
    from flash.client.http import ClientError

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
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
    """File-like wrapper over in-memory bytes that fires a progress callback on each read()."""

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
        # a rendering hiccup must never abort an in-flight upload
        with contextlib.suppress(Exception):
            self._progress(self._pos, self._total)
        return chunk

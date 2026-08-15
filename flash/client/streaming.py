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
    """Shorten a socket timeout to the time remaining before a deadline.

    Applied once, when the request is opened. `_cap_socket_timeout` re-applies the same idea to the
    live socket as the body is read, which is what keeps the two bounds from compounding.
    """
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


def _cap_socket_timeout(resp: object, remaining: float) -> None:
    """Lower the response socket's timeout to ``remaining`` seconds, if it can be reached.

    Best effort by design. The socket sits behind private attributes (`fp.raw._sock`), so a reader
    that does not expose them -- a test double, or a future urllib -- simply keeps the timeout it
    already had. That degrades to the previous behaviour rather than failing the read: the deadline
    check between reads still bounds the loop, it just cannot bound the single read in flight.

    Only ever lowers. Raising a caller's timeout here would hand a socket more patience than it was
    configured with, which is the opposite of the bound this is enforcing.
    """
    sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
    settimeout = getattr(sock, "settimeout", None)
    if settimeout is None:
        return
    try:
        current = sock.gettimeout()
        if current is None or current > remaining:
            settimeout(remaining)
    except OSError:
        # a socket that has already been torn down cannot be re-armed; the read that follows will
        # surface the real failure with a better message than anything raised from here.
        return


def _read_capped_response(resp: object, max_bytes: int, deadline: float | None = None) -> bytes:
    """Read a response body under a byte cap and optional absolute deadline."""
    from flash.client.http import ClientError

    read_size = _DOWNLOAD_CHUNK_BYTES if deadline is None else _DEADLINE_CHUNK_BYTES
    read = resp.read  # type: ignore[attr-defined]
    if deadline is not None:
        # the deadline is only checked BETWEEN reads, so it can only bound this loop if each read
        # returns promptly. `read(n)` on a buffered reader blocks until all n bytes arrive, so a
        # peer trickling a short body holds a single call open past the deadline and the check
        # never runs -- measured at 12s against a 2s deadline. `read1` returns what has already
        # arrived, which lets the check run. size 1 is the equivalent fallback for a reader
        # without it (the chat stream reader makes the same trade).
        read1 = getattr(resp, "read1", None)
        if read1 is not None:
            read = read1
        else:
            read_size = 1
    chunks: list[bytes] = []
    total = 0
    while True:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ClientError(
                    f"the control plane's response body stalled after {total} bytes "
                    "and exceeded its overall deadline"
                )
            # the check above only bounds the loop BETWEEN reads. `_capped_timeout` installs the
            # socket timeout once, when the request opens, and time spent connecting and waiting
            # for headers is charged to the deadline but not to that timeout -- so a read that
            # begins late is still entitled to block for longer than the deadline has left.
            # measured at 3.51s against a 2.0s deadline, and `flash env list` passes 230s as both
            # bounds. re-capping to the remaining budget makes the two agree instead of stacking.
            _cap_socket_timeout(resp, remaining)
        chunk = read(read_size)
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

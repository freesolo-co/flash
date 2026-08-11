"""Bounded body reads for the GitHub requests behind environment loading.

Split out of ``flash.envs.loader`` to keep that module under the file-size gate. Both helpers exist
for the same reason: a response body must be bounded in bytes AND in wall-clock, because urllib
applies its ``timeout`` to each blocking socket read rather than to the response as a whole, so a
peer that dribbles one chunk inside every window keeps a single attempt alive indefinitely.

They differ only in how strict they are, which follows from what the body is for -- see each
docstring. ``loader`` re-exports both, since that is the import site callers and tests already use.
"""

from __future__ import annotations

import time
import urllib.error
from collections.abc import Iterator

_MAX_ERROR_BODY_BYTES = 64 * 1024
# the read size used only when a deadline is in force, and only if it is SMALLER than the caller's
# configured chunk size -- a caller that already reads in smaller pieces keeps its own. Not folded
# into `_DOWNLOAD_CHUNK_BYTES`: that constant is the download granularity for the whole
# no-deadline path, and shrinking it would slow every large tarball fetch to buy a bound those
# callers never asked for.
_DEADLINE_CHUNK_BYTES = 8 * 1024


def _chunk_bytes() -> int:
    """The read size, resolved from ``loader`` on every call rather than bound at import.

    Not a module constant here: tests shrink the chunk size with
    ``monkeypatch.setattr(loader, "_DOWNLOAD_CHUNK_BYTES", n)`` to drive the cap with a small body,
    and a copy in this module would silently ignore that patch -- the reads below would keep using
    1 MiB while the test believed it had set 4. Resolving through ``loader`` keeps that seam pointing
    at the one binding it always pointed at.
    """
    from flash.envs import loader

    return loader._DOWNLOAD_CHUNK_BYTES


def _read_error_body(exc: urllib.error.HTTPError, body_deadline: float | None) -> str:
    """Read a non-2xx body for classification, bounded in both bytes and wall-clock.

    ``exc.read()`` is an unbounded read: ``HTTPResponse.read(None)`` streams a chunked body until it
    ends, so a slow 429/5xx whose chunks each arrive inside the socket timeout kept the caller's
    handler alive past the interactive list's deadline -- the client then timed out locally instead
    of receiving the controlled 429/502 the status was about to produce.

    Deliberately lenient where ``_iter_capped_chunks`` is strict: only the first few hundred
    characters are ever used (to classify, and as a message tail), so hitting the cap or the deadline
    truncates the body rather than raising. Losing the tail of an error body must not cost the status
    that classifies the failure.
    """
    if exc.fp is None:
        return ""
    deadline = time.monotonic() + body_deadline if body_deadline is not None else None
    chunks: list[bytes] = []
    total = 0
    while total < _MAX_ERROR_BODY_BYTES:
        if deadline is not None and time.monotonic() > deadline:
            break
        # bounded by the same reasoning as `_iter_capped_chunks` below, and for the same reason it is
        # only applied when a deadline is in force: without one there is nothing for a small read to
        # buy. The remaining-bytes term still applies, so the cap is never overshot either way.
        read_size = min(_chunk_bytes(), _MAX_ERROR_BODY_BYTES - total)
        if deadline is not None:
            read_size = min(read_size, _DEADLINE_CHUNK_BYTES)
        chunk = exc.read(read_size)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks).decode("utf-8", "replace")


def _iter_capped_chunks(
    resp: object, max_bytes: int, deadline: float | None = None
) -> Iterator[bytes]:
    # per-attempt accounting is also per-download accounting: _urlopen rewinds and truncates the
    # sink before every attempt, so the bytes this generator caps are exactly the bytes that
    # survive on disk. a retried download can never leave more than max_bytes behind.
    total = 0
    # the deadline can only be checked BETWEEN reads, so the read size is what bounds how far one
    # blocked read can overshoot it. A single `HTTPResponse.read(n)` performs many socket receives
    # internally and urllib's timeout applies per receive, not per call, so a peer that stays inside
    # every window keeps one read blocked far past the deadline: with a 1 MiB size that is thousands
    # of receives before control returns here. A deadline-bearing read therefore uses a smaller size,
    # so the check runs often enough for the bound to mean something.
    #
    # This NARROWS the overshoot; it does not eliminate it. The residual is "however long one read's
    # worth takes to arrive", which a slow enough peer still stretches -- bounding that properly needs
    # an interruptible read (a thread or async), which this blocking client has no way to express. The
    # socket timeout stays the real backstop against a peer that stalls outright, and the caller's own
    # fixed timeout bounds what the user ever waits.
    read_size = _chunk_bytes() if deadline is None else min(_chunk_bytes(), _DEADLINE_CHUNK_BYTES)
    while True:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(
                f"GitHub response body exceeded its overall deadline after {total} bytes"
            )
        chunk = resp.read(read_size)
        if not chunk:
            break
        if total + len(chunk) > max_bytes:
            raise RuntimeError(
                f"GitHub response body exceeded the maximum allowed size ({max_bytes} bytes); "
                "download aborted"
            )
        total += len(chunk)
        yield chunk

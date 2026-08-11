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

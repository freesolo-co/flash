"""Shared safety limits for Freesolo environment tar archives."""

from __future__ import annotations

import tarfile
from collections.abc import Callable
from typing import BinaryIO

ARCHIVE_MEMBER_LIMIT = 5000
ARCHIVE_SCAN_MEMBER_LIMIT = 200_000
TAR_METADATA_TYPES = frozenset(
    {
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
    }
)


def archive_stream_limit(content_limit: int, member_limit: int) -> int:
    """Backstop for decompressed tar bytes, including hidden header payloads."""
    return content_limit + member_limit * 1024 + (1 << 20)


class LimitedArchiveReader:
    """Reader wrapper that caps decompressed tar bytes, including header payloads."""

    def __init__(self, raw: BinaryIO, limit: int, error_factory: Callable[[], Exception]):
        self._raw = raw
        self._remaining = limit
        self._error_factory = error_factory

    def read(self, size: int = -1) -> bytes:
        want = self._remaining + 1 if size is None or size < 0 else min(size, self._remaining + 1)
        chunk = self._raw.read(want)
        self._remaining -= len(chunk)
        if self._remaining < 0:
            raise self._error_factory()
        return chunk


def tar_member_segments(name: str, *, unsafe_error: Callable[[str], Exception]) -> list[str]:
    """Normalize a tar member path into safe path segments."""
    segments: list[str] = []
    for segment in name.replace("\\", "/").split("/"):
        if not segment or segment == ".":
            continue
        if segment == "..":
            raise unsafe_error(name)
        segments.append(segment)
    return segments

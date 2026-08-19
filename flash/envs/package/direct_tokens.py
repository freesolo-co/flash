"""Bounded raw-byte scanning for direct environment package tokens."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterator
from pathlib import Path

_CHUNK_SIZE = 64 * 1024
_URLSAFE_CLASS = rb"A-Za-z0-9_-"
_ALPHANUMERIC_CLASS = rb"A-Za-z0-9"
_TOKEN_RULES = (
    (b"fslo_", _URLSAFE_CLASS, 45),
    (b"hf_", _ALPHANUMERIC_CLASS, 34),
    (b"pit_", _ALPHANUMERIC_CLASS, 64),
)
_TOKEN_PATTERNS = tuple(
    re.compile(
        rb"(?<!["
        + _URLSAFE_CLASS
        + rb"])(?:"
        + re.escape(prefix)
        + rb")(["
        + body_class
        + rb"]{"
        + str(body_length).encode()
        + rb"})(?!["
        + _URLSAFE_CLASS
        + rb"])"
    )
    for prefix, body_class, body_length in _TOKEN_RULES
)
_MAX_TOKEN_SIZE = max(len(prefix) + body_length for prefix, _, body_length in _TOKEN_RULES)
_OVERLAP = _MAX_TOKEN_SIZE + 1
_PLACEHOLDER_BODIES = frozenset(
    {
        b"exampletoken",
        b"notarealtoken",
        b"placeholdertoken",
        b"replacemewithtoken",
        b"testtoken",
        b"yourapitokenhere",
        b"yourtokenhere",
    }
)


class DirectTokenScanError(Exception):
    """A package could not be scanned without following or skipping a member."""

    def __init__(self) -> None:
        super().__init__("package scan failed")


def _is_obvious_placeholder(body: bytes) -> bool:
    normalized = body.lower().translate(None, b"_-")
    if normalized in _PLACEHOLDER_BODIES:
        return True
    return bool(normalized) and normalized[:1] in {b"0", b"x"} and len(set(normalized)) == 1


def _buffer_contains_direct_token(data: bytes, *, start_at: int, finalized_start_end: int) -> bool:
    for pattern in _TOKEN_PATTERNS:
        for match in pattern.finditer(data):
            if match.start() < start_at:
                continue
            if match.start() >= finalized_start_end:
                break
            if not _is_obvious_placeholder(match.group(1)):
                return True
    return False


def _file_contains_direct_token(path: Path) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise DirectTokenScanError
        with os.fdopen(fd, "rb") as stream:
            fd = None
            data = b""
            buffer_start = 0
            finalized_start = 0
            while chunk := stream.read(_CHUNK_SIZE):
                data += chunk
                start_at = finalized_start - buffer_start
                finalized_start_end = max(start_at, len(data) - _MAX_TOKEN_SIZE)
                if _buffer_contains_direct_token(
                    data,
                    start_at=start_at,
                    finalized_start_end=finalized_start_end,
                ):
                    return True
                finalized_start = buffer_start + finalized_start_end
                keep_from = max(0, len(data) - _OVERLAP)
                data = data[keep_from:]
                buffer_start += keep_from
            return _buffer_contains_direct_token(
                data,
                start_at=finalized_start - buffer_start,
                finalized_start_end=len(data),
            )
    finally:
        if fd is not None:
            os.close(fd)


def _regular_files(directory: Path) -> Iterator[Path]:
    with os.scandir(directory) as listing:
        entries = sorted(listing, key=lambda entry: entry.name)
    for entry in entries:
        if entry.is_symlink():
            raise DirectTokenScanError
        if entry.is_dir(follow_symlinks=False):
            yield from _regular_files(Path(entry.path))
        elif entry.is_file(follow_symlinks=False):
            yield Path(entry.path)
        else:
            raise DirectTokenScanError


def package_contains_direct_token(root: Path) -> bool:
    """Return whether a package tree contains a supported direct issued-token form."""
    try:
        root_stat = os.stat(root, follow_symlinks=False)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise DirectTokenScanError
        return any(_file_contains_direct_token(path) for path in _regular_files(root))
    except DirectTokenScanError:
        raise
    except OSError:
        pass
    raise DirectTokenScanError

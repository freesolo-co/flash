"""Small shared JSON file helpers for data under ``~/.flash``."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def reject_duplicate_keys(
    on_duplicate: Callable[[str], BaseException],
) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
    """Build a ``json.loads`` ``object_pairs_hook`` that rejects duplicate keys.

    ``json.loads`` keeps the LAST value for a repeated key, so a payload carrying the same key
    twice decodes to something neither writer meant, and a checked identity can be smuggled past
    a comparison. Each caller raises its own error type for the offending key: the decoders here
    sit on different trust boundaries (a served safetensors header, a resume marker, an API
    request), and the error each one raises is part of its own contract.
    """

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise on_duplicate(key)
            value[key] = item
        return value

    return hook


def read_json_or_empty(path: Path) -> dict:
    """Parse a JSON object file, returning ``{}`` on any error or if the root is not a dict."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def secure_json_write(path: Path, data: dict) -> None:
    """Atomically replace a JSON file with a private, fully written 0600 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path.parent, 0o700)
    if path.is_symlink():
        raise OSError(errno.ELOOP, "refusing to replace a credential symlink", str(path))
    # mkstemp creates a fresh 0600 inode. Never truncate the previous login or write a
    # new secret into an existing file whose permissions may still be public.
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as writer:
            json.dump(data, writer, indent=2, sort_keys=True)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)

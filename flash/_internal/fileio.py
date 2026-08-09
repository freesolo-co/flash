"""Small shared JSON file helpers for data under ``~/.flash``."""

from __future__ import annotations

import contextlib
import json
import os
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
    """Write ``data`` as JSON 0600. Opens with O_CREAT|0o600 — never write_text+chmod (TOCTOU). O_NOFOLLOW blocks symlink redirect."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path.parent, 0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)

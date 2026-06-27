"""Small shared file-IO helpers for credential/manifest JSON under ``~/.flash``."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path


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

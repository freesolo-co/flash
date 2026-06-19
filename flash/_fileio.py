"""Small shared file-IO helpers for credential/manifest JSON under ``~/.flash``."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path


def read_json_or_empty(path: Path) -> dict:
    """Parse a JSON object file, returning ``{}`` if it's missing or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def secure_json_write(path: Path, data: dict) -> None:
    """Write ``data`` as JSON with private permissions (the file may hold a secret).

    Creates the parent dir (0700) and opens the file 0600 from the start — never
    write_text + chmod, which leaves it umask-readable in between. ``O_NOFOLLOW``
    (where available) refuses to follow a symlink planted at ``path`` so the write
    can't be redirected to clobber an arbitrary file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path.parent, 0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)

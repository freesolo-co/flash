"""Publish complete environment trees without replacing concurrent cache hits."""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
from pathlib import Path

from flash.envs.meta.cache_security import trust_cache_entry

STAGING_PREFIX = ".flash-env-stage-"


def publish_cache_entry(source: Path, destination: Path, entrypoint: str) -> None:
    """Copy privately, then publish atomically; a complete concurrent publisher wins."""
    with tempfile.TemporaryDirectory(prefix=STAGING_PREFIX, dir=destination.parent) as tmp:
        staged = Path(tmp) / "contents"
        shutil.copytree(source, staged)
        try:
            os.rename(staged, destination)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            candidate = destination / entrypoint
            if not (
                trust_cache_entry(destination)
                and destination.is_dir()
                and trust_cache_entry(candidate)
                and candidate.is_file()
            ):
                raise RuntimeError(
                    "concurrent environment cache entry is incomplete or untrusted"
                ) from exc

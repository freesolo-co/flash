"""Publish complete environment trees without replacing concurrent cache hits."""

from __future__ import annotations

import errno
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

from flash.envs.meta import cache_security

STAGING_PREFIX = ".flash-env-stage-"
_STAGING_MAX_AGE_SECONDS = 24 * 3600


@contextmanager
def _file_lock(path: Path, *, blocking: bool = True) -> Iterator[bool]:
    """Hold an OS-released lock, including across a process crash; never unlink its file."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    acquired = False
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
            raise RuntimeError(f"untrusted environment cache lock: {path}")
        if os.name == "nt":
            import msvcrt

            if info.st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    if not blocking:
                        break
                    time.sleep(0.05)
        else:
            import fcntl

            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
                acquired = True
            except BlockingIOError:
                if blocking:
                    raise
        yield acquired
    finally:
        try:
            if acquired and os.name == "nt":
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)


def cache_lock(root: Path):
    """Serialize cache metadata changes, without holding the lock during downloads or copies."""
    return _file_lock(root / ".flash-env-cache.lock")


def find_cached_entry(destination: Path, entrypoint: str, default_entrypoint: str) -> Path | None:
    """Check trust and completeness, or remove an invalid entry, atomically with publication."""
    # Also reap on hits and failed downloads, which never reach post-publication eviction.
    with suppress(OSError):
        reap_abandoned_stages(destination.parent)
    with cache_lock(destination.parent):
        if os.path.lexists(destination) and not (
            cache_security.trust_cache_entry(destination) and destination.is_dir()
        ):
            cache_security.discard_untrusted_entry(destination)
        candidate = destination / entrypoint
        if candidate.is_dir():
            candidate /= default_entrypoint
        if candidate.is_file():
            if cache_security.trust_cache_entry(candidate):
                with suppress(OSError):
                    os.utime(destination)
                return candidate
            cache_security.discard_untrusted_entry(destination)
        if destination.exists():
            shutil.rmtree(destination)
    return None


@contextmanager
def _staging_directory(root: Path) -> Iterator[Path]:
    with ExitStack() as resources:
        # Creation and lease acquisition must be indivisible to the abandoned-stage reaper.
        with cache_lock(root):
            stage = Path(
                resources.enter_context(
                    tempfile.TemporaryDirectory(prefix=STAGING_PREFIX, dir=root)
                )
            )
            resources.enter_context(_file_lock(stage / ".active"))
        try:
            yield stage
        finally:
            # Close the lease before removal (required on Windows), under the root lock.
            with cache_lock(root):
                resources.close()


def reap_abandoned_stages(root: Path) -> None:
    """Remove old stages only when their copying process no longer holds the lease."""
    with cache_lock(root):
        cutoff = time.time() - _STAGING_MAX_AGE_SECONDS
        for stage in root.glob(f"{STAGING_PREFIX}*"):
            with suppress(OSError):
                if not cache_security.trust_cache_entry(stage) or not stage.is_dir():
                    continue
                if stage.stat().st_mtime >= cutoff:
                    continue
                with _file_lock(stage / ".active", blocking=False) as abandoned:
                    pass
                if abandoned:
                    shutil.rmtree(stage)


def publish_cache_entry(source: Path, destination: Path, entrypoint: str) -> None:
    """Copy privately, then publish atomically; a complete concurrent publisher wins."""
    with _staging_directory(destination.parent) as stage:
        staged = stage / "contents"
        shutil.copytree(source, staged)
        with cache_lock(destination.parent):
            try:
                os.rename(staged, destination)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                candidate = destination / entrypoint
                if not (
                    cache_security.trust_cache_entry(destination)
                    and destination.is_dir()
                    and cache_security.trust_cache_entry(candidate)
                    and candidate.is_file()
                ):
                    raise RuntimeError(
                        "concurrent environment cache entry is incomplete or untrusted"
                    ) from exc
            # copytree preserves archive timestamps; publication is a fresh cache use.
            with suppress(OSError):
                os.utime(destination)

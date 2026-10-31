"""Single active control plane per state directory.

A zero-downtime deploy starts the new plane while the old one drains in-flight requests, so for a
few minutes two planes share one state directory. Recovery and the background loops assume they
are alone: a second plane's recovery scan would reap instances the first is still launching. Only
the holder of this lease runs them. A plane without it still serves HTTP and leaves new runs queued
for the holder's recovery scan. The kernel drops the flock when the holder exits, even on SIGKILL.
"""

from __future__ import annotations

import asyncio
import fcntl
import os

from flash._internal.paths import data_dir

_fd: int | None = None


def lease_path() -> str:
    return os.path.join(str(data_dir()), "control-plane.lock")


def try_acquire() -> bool:
    """Take the lease without blocking; True if this process holds it afterwards."""
    global _fd
    if _fd is not None:
        return True
    path = lease_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return False
    except BaseException:
        os.close(fd)
        raise
    _fd = fd
    return True


async def acquire(poll_seconds: float = 1.0) -> None:
    """Wait until this process holds the lease."""
    while not try_acquire():  # noqa: ASYNC110 -- the holder is another process; flock has no event.
        await asyncio.sleep(poll_seconds)


def held() -> bool:
    return _fd is not None


def release() -> None:
    global _fd
    fd, _fd = _fd, None
    if fd is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

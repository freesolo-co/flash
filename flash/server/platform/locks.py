"""per-run deploy, undeploy, and export mutexes for the control plane."""

from __future__ import annotations

import fcntl
import os
import threading
import weakref
from typing import Self


class _RunLock:
    """Weak-referenceable mutex shared by threads and control-plane processes."""

    __slots__ = ("__weakref__", "_fd", "_lock", "_run_id", "_suffix")

    def __init__(self, run_id: str, *, suffix: str = ".deploy.lock") -> None:
        self._run_id = run_id
        self._suffix = suffix
        self._lock = threading.Lock()
        self._fd: int | None = None

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def acquire(self, blocking: bool = True) -> bool:
        """Acquire the thread mutex and the run's process-wide file lock."""
        if not self._lock.acquire(blocking):
            return False
        fd: int | None = None
        try:
            from flash.runner.lifecycle.state import RUNS_DIR, runs_file_path

            os.makedirs(RUNS_DIR, exist_ok=True)
            fd = os.open(runs_file_path(self._run_id, self._suffix), os.O_CREAT | os.O_RDWR, 0o600)
            operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(fd, operation)
        except BlockingIOError:
            if fd is not None:
                os.close(fd)
            self._lock.release()
            return False
        except BaseException:
            if fd is not None:
                os.close(fd)
            self._lock.release()
            raise
        self._fd = fd
        return True

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            raise RuntimeError("deploy lock is not held")
        self._fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
            self._lock.release()


# Serializes deploy, undeploy, and export per run id.
_DEPLOY_LOCKS: weakref.WeakValueDictionary[str, _RunLock] = weakref.WeakValueDictionary()
_RESUBMIT_LOCKS: weakref.WeakValueDictionary[str, _RunLock] = weakref.WeakValueDictionary()
_DEPLOY_LOCKS_GUARD = threading.Lock()


def _named_run_lock(
    run_id: str,
    locks: weakref.WeakValueDictionary[str, _RunLock],
    *,
    suffix: str,
) -> _RunLock:
    with _DEPLOY_LOCKS_GUARD:
        lock = locks.get(run_id)
        if lock is None:
            lock = _RunLock(run_id, suffix=suffix)
            locks[run_id] = lock
        return lock


def _deploy_lock(run_id: str) -> _RunLock:
    return _named_run_lock(run_id, _DEPLOY_LOCKS, suffix=".deploy.lock")


def _resubmit_lock(run_id: str) -> _RunLock:
    return _named_run_lock(run_id, _RESUBMIT_LOCKS, suffix=".resubmit.lock")

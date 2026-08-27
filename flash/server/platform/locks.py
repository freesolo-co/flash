"""per-run deploy, undeploy, and export mutexes for the control plane."""

from __future__ import annotations

import fcntl
import os
import threading
import weakref
from typing import Self


def _open_teacher_broker_lease() -> int:
    """Open the lease guarding the configured teacher ledger.

    Ledger recovery rewrites every live request, so it is safe only while no process is serving.
    The lease models that directly: exclusive means recovering, shared means serving. The path is
    resolved so symlinked or relative aliases of one ledger cannot yield independent leases.
    """
    from flash.server.platform.db import db_path

    path = f"{os.path.realpath(db_path())}.teacher-broker.lock"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return os.open(path, os.O_CREAT | os.O_RDWR, 0o600)


def _claim_teacher_recovery(fd: int) -> bool:
    """Whether this process may recover now: true only while no sibling recovers or serves."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _enter_teacher_serving(fd: int) -> None:
    """Hold the lease shared for the serving lifetime, waiting out any in-progress recovery.

    A later process therefore cannot claim recovery while this one is live, which is exactly the
    turnover that would rewrite our in-flight requests.
    """
    fcntl.flock(fd, fcntl.LOCK_SH)


def _release_teacher_broker_lease(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class _RunLock:
    """Weak-referenceable mutex shared by threads and control-plane processes."""

    __slots__ = ("__weakref__", "_fd", "_lock", "_run_id")

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
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
            fd = os.open(
                runs_file_path(self._run_id, ".deploy.lock"), os.O_CREAT | os.O_RDWR, 0o600
            )
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
_DEPLOY_LOCKS_GUARD = threading.Lock()


def _deploy_lock(run_id: str) -> _RunLock:
    with _DEPLOY_LOCKS_GUARD:
        lk = _DEPLOY_LOCKS.get(run_id)
        if lk is None:
            lk = _RunLock(run_id)
            _DEPLOY_LOCKS[run_id] = lk
        return lk

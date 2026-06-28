"""Per-run deploy/undeploy mutexes for the control plane."""

from __future__ import annotations

import threading
import weakref


class _RunLock:
    """Weak-referenceable mutex; threading.Lock() doesn't support weakref so we wrap it."""

    __slots__ = ("__weakref__", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def __enter__(self) -> _RunLock:
        self._lock.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self._lock.release()

    def acquire(self, blocking: bool = True) -> bool:
        """Acquire the underlying mutex; non-blocking mode reports whether it was free."""
        return self._lock.acquire(blocking)

    def release(self) -> None:
        self._lock.release()


# Serializes deploy, undeploy, export, and repo-GC deletes per run id.
_DEPLOY_LOCKS: weakref.WeakValueDictionary[str, _RunLock] = weakref.WeakValueDictionary()
_DEPLOY_LOCKS_GUARD = threading.Lock()


def _deploy_lock(run_id: str) -> _RunLock:
    with _DEPLOY_LOCKS_GUARD:
        lk = _DEPLOY_LOCKS.get(run_id)
        if lk is None:
            lk = _RunLock()
            _DEPLOY_LOCKS[run_id] = lk
        return lk


def _try_hold_deploy_lock(run_id: str) -> _RunLock | None:
    """Non-blocking acquire of a run's deploy/undeploy/export lock, for the repo GC.

    Returns the HELD lock — the caller MUST ``release()`` it once its critical section (the HF
    ``delete_repo``) is done — or ``None`` when a deploy/undeploy/export already holds it, so the GC
    spares that repo this cycle instead of blocking on a slow registration/export. Holding the SAME
    lock the deploy/undeploy/export endpoints take makes the destructive delete mutually exclusive
    with them, closing the start-after-check race a non-blocking read left open."""
    lk = _deploy_lock(run_id)
    return lk if lk.acquire(blocking=False) else None

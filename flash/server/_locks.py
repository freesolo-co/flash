"""Per-run deploy/undeploy mutexes for the control plane.

No fastapi dependency, so this module is safe to import at ``flash.server.app`` import
time (it must not pull in the optional server extras).
"""

from __future__ import annotations

import threading
import weakref


class _RunLock:
    """A weak-referenceable mutex usable as a context manager.

    ``threading.Lock()`` returns a ``_thread.lock`` that does NOT support weak references,
    so it can't live in a WeakValueDictionary directly — wrap it in a tiny object that does
    (and acquire/release via ``with``).
    """

    __slots__ = ("__weakref__", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def __enter__(self) -> _RunLock:
        self._lock.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self._lock.release()

    def acquire(self, blocking: bool = True) -> bool:
        """Acquire the underlying mutex; with ``blocking=False`` returns ``True`` iff it was free.

        The repo GC uses the non-blocking form to acquire-and-HOLD a run's lock ACROSS an HF
        ``delete_repo`` — making the destructive delete mutually exclusive with a concurrent
        deploy/undeploy/export of the same run, instead of merely *observing* the lock (a read leaves
        a start-after-check race where a deploy/export that began just after the read still raced the
        delete)."""
        return self._lock.acquire(blocking)

    def release(self) -> None:
        self._lock.release()


# Per-run lock serializing deploy vs undeploy vs export: registration with the freesolo serving app
# (deploy/undeploy) and the download-then-upload of the run's private artifact repo (export) are slow
# and run OUTSIDE the status lock, so without this they could interleave — a racing undeploy could
# leave a stale deployment record (registered with freesolo but unrecorded here, or vice-versa), a
# deploy's cleanup of a raced finalize could clobber another, and the repo GC could delete a run's HF
# source out from under an in-flight deploy/export. Serving is delegated to freesolo (scales to zero
# per base model), so there is no billable flash-side endpoint at stake — only the deployment
# record's consistency and the artifact repo's availability to in-flight readers.
# WeakValueDictionary so an entry is dropped once no request holds the lock — the map
# can't grow unboundedly with one entry per distinct run_id over the server's lifetime.
_DEPLOY_LOCKS: weakref.WeakValueDictionary[str, _RunLock] = weakref.WeakValueDictionary()
_DEPLOY_LOCKS_GUARD = threading.Lock()


def _deploy_lock(run_id: str) -> _RunLock:
    # The returned lock must be held by the caller (a `with` block) to keep it alive; once
    # released and unreferenced, the weak entry is garbage-collected.
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

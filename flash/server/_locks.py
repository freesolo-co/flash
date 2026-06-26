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


# Per-run lock serializing deploy vs undeploy: registration with the freesolo serving app
# is slow and runs OUTSIDE the status lock, so without this the two could interleave —
# a racing undeploy could leave a stale deployment record (registered with freesolo but
# unrecorded here, or vice-versa), or a deploy's cleanup of a raced finalize could clobber
# another. Serving is delegated to freesolo (scales to zero per base model), so there is no
# billable flash-side endpoint at stake — only the deployment record's consistency.
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

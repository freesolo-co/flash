"""Ownership of the short-lived child processes the deployment smoke spawns.

A child is owned from the moment it starts, not from the moment a reap fails. `spawn_owned`
publishes the process into the live set under the same lock that starts it, and `reap_owned`
releases it only after a confirmed exit. Anything still alive stays in the set, so the caller
cannot drop it by raising, by failing to close a pipe, or by never reaching its own reap at all:
`reap_live_children` is the single boundary that finishes the job, and the asgi lifespan drains it.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

_log = logging.getLogger("flash.server")

_LIVE_CHILDREN_LOCK = threading.Lock()
_LIVE_CHILDREN: set[Any] = set()
LIVE_CHILDREN_WAKE = threading.Event()

_JOIN_SECONDS = 0.1
_TERMINATE_SECONDS = 0.2
_KILL_SECONDS = 0.2


def spawn_owned(process) -> None:
    """start a child and publish it to the live set in one atomic step."""
    with _LIVE_CHILDREN_LOCK:
        process.start()
        _LIVE_CHILDREN.add(process)
        LIVE_CHILDREN_WAKE.set()


def _release(process) -> None:
    with _LIVE_CHILDREN_LOCK:
        _LIVE_CHILDREN.discard(process)
        if not _LIVE_CHILDREN:
            LIVE_CHILDREN_WAKE.clear()


def _bounded(step: float, deadline: float | None) -> float:
    if deadline is None:
        return step
    return min(step, max(0.0, deadline - time.monotonic()))


def reap_owned(process, *, terminate_first: bool = False, deadline: float | None = None) -> bool:
    """run the bounded shutdown ladder, releasing ownership only on a confirmed exit.

    returns whether the caller may close the process: true only for a child that ran and has now
    confirmably exited. a child that never started owns no os resources to release, and one that
    survives every step stays owned by the live set for `reap_live_children` to retry.
    """
    if process.pid is None:
        _release(process)
        return False
    if not terminate_first:
        process.join(timeout=_bounded(_JOIN_SECONDS, deadline))
    if process.is_alive():
        process.terminate()
        process.join(timeout=_bounded(_TERMINATE_SECONDS, deadline))
    if process.is_alive():
        process.kill()
        process.join(timeout=_KILL_SECONDS)
    if process.is_alive():
        return False
    _release(process)
    return True


def reap_live_children(timeout: float) -> bool:
    """retry the bounded ladder for every live child and report whether the set drained."""
    deadline = time.monotonic() + max(0.0, timeout)
    with _LIVE_CHILDREN_LOCK:
        processes = tuple(_LIVE_CHILDREN)
    for process in processes:
        try:
            if reap_owned(process, deadline=deadline):
                process.close()
        except Exception:
            _log.warning("live deployment child could not be reaped", exc_info=True)
    with _LIVE_CHILDREN_LOCK:
        return not _LIVE_CHILDREN

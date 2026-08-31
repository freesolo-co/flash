"""Ownership of the short-lived child processes the deployment smoke spawns.

A child is owned from the moment it starts, not from the moment a reap fails. `spawn_owned`
publishes the process into the live set under the same lock that starts it, and `reap_owned`
releases it only after a confirmed exit. Anything still alive stays in the set, so the caller
cannot drop it by raising, by failing to close a pipe, or by never reaching its own reap at all.

The set is a shutdown backstop, not a background reaper: a smoke child is allowed to run for its
whole budget, and only `reap_live_children` at the end of the asgi lifespan ends what is left.
Releasing is what confers the right to `close`, so exactly one of the owning thread and the
shutdown drain can close a given child.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

_log = logging.getLogger("flash.server")

_LIVE_CHILDREN_LOCK = threading.Lock()
_LIVE_CHILDREN: set[Any] = set()

_JOIN_SECONDS = 0.1
_TERMINATE_SECONDS = 0.2
_KILL_SECONDS = 0.2


def spawn_owned(process) -> None:
    """start a child and publish it to the live set in one atomic step."""
    with _LIVE_CHILDREN_LOCK:
        process.start()
        _LIVE_CHILDREN.add(process)


def _release(process) -> bool:
    """drop ownership, reporting whether this caller is the one that took it away."""
    with _LIVE_CHILDREN_LOCK:
        if process not in _LIVE_CHILDREN:
            return False
        _LIVE_CHILDREN.discard(process)
        return True


def _bounded(step: float, deadline: float | None) -> float:
    if deadline is None:
        return step
    return min(step, max(0.0, deadline - time.monotonic()))


def reap_owned(process, *, terminate_first: bool = False, deadline: float | None = None) -> bool:
    """run the bounded shutdown ladder, releasing ownership only on a confirmed exit.

    returns whether the caller may close the process: true only for the caller that ran a started
    child to a confirmed exit and took it out of the live set. a child that never started owns no
    os resources to release, one that survives every step stays owned for the shutdown drain to
    retry, and a second caller reaching an already released child is told no so it cannot close
    the same process twice.
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
    return _release(process)


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

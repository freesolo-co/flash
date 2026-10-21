"""Os-shared provider launch ownership leases."""

from __future__ import annotations

import contextlib
import fcntl
import os
import threading

from flash.runner.lifecycle import state

_CLAIM_LOCK_SUFFIX = ".launch-claim.lock"
_ACTIVE_CLAIM_FDS: dict[str, tuple[str, int]] = {}
_ACTIVE_CLAIM_FDS_LOCK = threading.Lock()


def try_acquire(run_id: str) -> int | None:
    """Acquire the run's launch lease, or return none while another owner is live."""
    os.makedirs(state.RUNS_DIR, exist_ok=True)
    fd = os.open(
        state.runs_file_path(run_id, _CLAIM_LOCK_SUFFIX),
        os.O_CREAT | os.O_RDWR,
        0o600,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def close(fd: int) -> None:
    """Unlock and close one unregistered or released lease descriptor."""
    with contextlib.suppress(OSError):
        fcntl.flock(fd, fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        os.close(fd)


def register(run_id: str, token: str, fd: int) -> None:
    """Associate a durable claim token with its locally held lease descriptor."""
    with _ACTIVE_CLAIM_FDS_LOCK:
        previous = _ACTIVE_CLAIM_FDS.pop(token, None)
        _ACTIVE_CLAIM_FDS[token] = (run_id, fd)
    if previous is not None:
        close(previous[1])


def owned_locally(run_id: str, token: str) -> bool:
    """Return whether this process owns the run lease under this exact token."""
    with _ACTIVE_CLAIM_FDS_LOCK:
        owned = _ACTIVE_CLAIM_FDS.get(token)
    return owned is not None and owned[0] == run_id


def release(run_id: str, token: str) -> None:
    """Release a locally held run lease by its durable token."""
    with _ACTIVE_CLAIM_FDS_LOCK:
        owned = _ACTIVE_CLAIM_FDS.get(token)
        if owned is None or owned[0] != run_id:
            return
        _ACTIVE_CLAIM_FDS.pop(token, None)
    close(owned[1])

"""exact process-group supervision for top-level managed workers."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time

_TERM_GRACE_S = 10.0
_KILL_GRACE_S = 5.0
_POLL_S = 0.1


def _group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _group_exists_after_reaping(process: subprocess.Popen, process_group_id: int) -> bool:
    process.poll()
    return _group_exists(process_group_id)


def terminate_process_group(
    process: subprocess.Popen,
    *,
    process_group_id: int,
    term_grace_s: float = _TERM_GRACE_S,
    kill_grace_s: float = _KILL_GRACE_S,
) -> None:
    """term, bounded wait, kill, and verify one captured process group disappears."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGTERM)
    term_deadline = time.monotonic() + max(0.0, term_grace_s)
    while (
        _group_exists_after_reaping(process, process_group_id) and time.monotonic() < term_deadline
    ):
        time.sleep(_POLL_S)
    if _group_exists_after_reaping(process, process_group_id):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)
    kill_deadline = time.monotonic() + max(0.0, kill_grace_s)
    while (
        _group_exists_after_reaping(process, process_group_id) and time.monotonic() < kill_deadline
    ):
        time.sleep(_POLL_S)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=0)
    if _group_exists(process_group_id):
        raise RuntimeError(f"process group {process_group_id} survived term and kill supervision")


def start_process_group(args, **kwargs) -> tuple[subprocess.Popen, int]:
    """start one process in a new session and return its captured group identity."""
    if "start_new_session" in kwargs:
        raise ValueError("start_new_session is owned by process-group supervision")
    process = subprocess.Popen(args, start_new_session=True, **kwargs)
    return process, process.pid

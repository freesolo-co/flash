"""exact process-group supervision for top-level managed workers."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time

_TERM_GRACE_S = 10.0
_KILL_GRACE_S = 5.0
_TERMINATION_SCHEDULING_RESERVE_S = 1.0
_POLL_S = 0.1


def worker_execution_deadline(
    upload_deadline_at: float,
    console_stop_timeout_s: float,
    console_final_timeout_s: float,
) -> float:
    """reserve complete worker teardown and console cleanup before the outer watchdog."""
    return (
        upload_deadline_at
        - _TERM_GRACE_S
        - _KILL_GRACE_S
        - _TERMINATION_SCHEDULING_RESERVE_S
        - console_stop_timeout_s
        - console_final_timeout_s
    )


def _group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
    while time.monotonic() < term_deadline:
        process.poll()
        if not _group_exists(process_group_id):
            break
        time.sleep(_POLL_S)
    process.poll()
    if _group_exists(process_group_id):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)
    kill_deadline = time.monotonic() + max(0.0, kill_grace_s)
    while time.monotonic() < kill_deadline:
        process.poll()
        if not _group_exists(process_group_id):
            break
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

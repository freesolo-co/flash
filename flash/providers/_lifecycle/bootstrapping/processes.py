"""Process-group teardown for the worker the bootstrap launches.

Stdlib only -- this ships inside the instance capsule alongside ``bootstrap.py`` and runs on the
rented host, where the ``flash`` package is not importable. The worker's own
``flash.engine.worker.train.entry.backend_common.kill_process_group`` solves the same problem one
layer down, but the bootstrap cannot import it, so the group teardown it needs lives here.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time

_TERM_GRACE_S = 10.0
_KILL_GRACE_S = 5.0
_POLL_S = 0.1


def _bounded_graces(
    term_grace_s: float, kill_grace_s: float, deadline_at: float | None
) -> tuple[float, float]:
    """Split the time actually left before ``deadline_at`` across the term and kill waits.

    The term wait yields first because the kill is the escalation that actually frees the gpu: with
    a tight budget it is better to signal SIGKILL early and spend what remains waiting for it than
    to burn the whole window on a SIGTERM the group is ignoring.
    """
    if deadline_at is None:
        return max(0.0, term_grace_s), max(0.0, kill_grace_s)
    remaining = max(0.0, deadline_at - time.time())
    kill_wait = min(max(0.0, kill_grace_s), remaining)
    term_wait = min(max(0.0, term_grace_s), remaining - kill_wait)
    return term_wait, kill_wait


def _group_exists(process_group_id: int) -> bool:
    """Report whether any process still belongs to the group."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # visible but not ours to signal, so treat it as alive rather than silently done.
        return True
    return True


def _group_exists_after_reaping(process: subprocess.Popen, process_group_id: int) -> bool:
    """Reap the leader before probing, so a zombie leader does not read as a live group."""
    process.poll()
    return _group_exists(process_group_id)


def _wait_for_group_exit(
    process: subprocess.Popen, process_group_id: int, timeout_s: float
) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        if not _group_exists_after_reaping(process, process_group_id):
            return True
        time.sleep(_POLL_S)
    return not _group_exists_after_reaping(process, process_group_id)


def terminate_process_group(
    process: subprocess.Popen,
    *,
    process_group_id: int,
    term_grace_s: float = _TERM_GRACE_S,
    kill_grace_s: float = _KILL_GRACE_S,
    deadline_at: float | None = None,
) -> None:
    """SIGTERM the group, wait, escalate to SIGKILL, and verify the group is gone.

    Signalling the group rather than the leader pid is the point: a training worker fans out into
    torchrun and vllm children that hold cuda contexts, and one survivor strands a paid gpu for
    every later run on that box.

    ``deadline_at`` caps the whole supervision against the wall clock. Without it a group that
    ignores SIGTERM costs the full term-then-kill grace, which is time the caller has already
    budgeted for what comes after teardown -- the final console upload that carries the survivor
    line. The kill still goes out when the term grace is spent, so the escalation is never skipped,
    only the waiting is bounded.
    """
    term_wait, kill_wait = _bounded_graces(term_grace_s, kill_grace_s, deadline_at)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process_group_id, signal.SIGTERM)
    if not _wait_for_group_exit(process, process_group_id, term_wait):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process_group_id, signal.SIGKILL)
        _wait_for_group_exit(process, process_group_id, kill_wait)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=0)
    if _group_exists(process_group_id):
        raise RuntimeError(f"process group {process_group_id} survived term and kill supervision")


def start_process_group(args, **kwargs) -> tuple[subprocess.Popen, int]:
    """Start one process as its own group leader and return it with its group id.

    The group id is captured at launch because ``os.getpgid`` stops working once the leader is
    reaped, which is exactly when the surviving descendants still need addressing.
    """
    if "start_new_session" in kwargs:
        raise ValueError("start_new_session is owned by process-group supervision")
    process = subprocess.Popen(args, start_new_session=True, **kwargs)
    return process, process.pid

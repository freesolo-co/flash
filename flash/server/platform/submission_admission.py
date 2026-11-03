"""Shared admission fence for HTTP submissions and restart recovery (no server extras)."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager

from flash.runner.lifecycle import state
from flash.server.platform import db


class SubmissionBusy(RuntimeError):
    """Another process is still admitting or cancelling this submission."""


@contextmanager
def submission_lock(run_id: str):
    # A slow owner cannot lose this fence to a timeout. The kernel releases on crash.
    os.makedirs(state.RUNS_DIR, exist_ok=True)
    fd = os.open(state.runs_file_path(run_id, ".submission.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SubmissionBusy(run_id) from None
        yield
    finally:
        os.close(fd)


def recovery_ready(run_id: str) -> bool:
    """Do not recover a status published by an unfinished submission.

    Recovery may run on a different plane while PUT still owns the admission fence.
    After a crash, the same fence lets recovery settle the durable cancellation intent
    (using normal resource teardown) or finish acknowledgement before taking over.
    Recheck at execution as well as classification: queued recovery tasks can be stale.
    """
    if db.submission_for_run(run_id) is None:
        return True  # Legacy POST runs have no submission admission protocol.
    try:
        with submission_lock(run_id):
            row = db.submission_for_run(run_id)
            if row is None:
                raise RuntimeError("durable submission disappeared during recovery")
            from flash.runner.lifecycle.status import get_status

            try:
                get_status(run_id)
            except FileNotFoundError:
                return False
            if row["cancelled"]:
                from flash.runner.supervise.deploy import cancel_run

                cancel_run(run_id)
                return False
            if not row["accepted"]:
                db.accept_submission(run_id)
            return True
    except SubmissionBusy:
        return False

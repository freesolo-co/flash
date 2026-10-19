"""Own the bounded deployment worker threads and their shutdown lifecycle."""

from __future__ import annotations

import os
import threading
import time

_DEPLOYMENT_JOBS_LOCK = threading.Lock()
_DEPLOYMENT_JOBS: set[threading.Thread] = set()
_DEPLOYMENT_JOBS_ACCEPTING = True
# a deployment job holds a thread for its whole lifecycle, and that lifecycle is minutes long: the
# readiness budget alone reaches 15 minutes. every accepted deploy for a distinct run got its own
# thread, and per-run locking only deduplicates one run id, so concurrent deploys across runs grew
# without any limit. cap the live set instead. the deploy route already turns a start failure into a
# retryable 503, so a full plane sheds load through the path it already has.
_MAX_ACTIVE_DEPLOYMENT_JOBS = 64


class DeploymentJobStartError(RuntimeError):
    pass


def _open_deployment_jobs() -> None:
    global _DEPLOYMENT_JOBS_ACCEPTING
    with _DEPLOYMENT_JOBS_LOCK:
        _DEPLOYMENT_JOBS_ACCEPTING = True


def _run_deployment_job(target, args, kwargs) -> None:
    try:
        target(*args, **kwargs)
    finally:
        with _DEPLOYMENT_JOBS_LOCK:
            _DEPLOYMENT_JOBS.discard(threading.current_thread())


def _wait_for_deployment_jobs(timeout: float) -> bool:
    global _DEPLOYMENT_JOBS_ACCEPTING
    deadline = time.monotonic() + timeout
    with _DEPLOYMENT_JOBS_LOCK:
        _DEPLOYMENT_JOBS_ACCEPTING = False
    while True:
        with _DEPLOYMENT_JOBS_LOCK:
            jobs = tuple(_DEPLOYMENT_JOBS)
        if not jobs:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        jobs[0].join(remaining)


def start_deployment_job(target, *args, **kwargs) -> bool:
    """Start a deployment lifecycle job.

    Returns True when the job ran synchronously (test mode), False when it was started in a
    background thread.
    """
    if os.environ.get("FLASH_DEPLOY_SYNC") == "1":
        target(*args, **kwargs)
        return True
    thread = threading.Thread(
        target=_run_deployment_job,
        args=(target, args, kwargs),
        daemon=True,
    )
    with _DEPLOYMENT_JOBS_LOCK:
        if not _DEPLOYMENT_JOBS_ACCEPTING:
            raise DeploymentJobStartError("deployment jobs are shutting down")
        if len(_DEPLOYMENT_JOBS) >= _MAX_ACTIVE_DEPLOYMENT_JOBS:
            raise DeploymentJobStartError(
                f"deployment job capacity reached ({_MAX_ACTIVE_DEPLOYMENT_JOBS} active)"
            )
        _DEPLOYMENT_JOBS.add(thread)
        try:
            thread.start()
        except Exception as exc:
            _DEPLOYMENT_JOBS.discard(thread)
            raise DeploymentJobStartError(str(exc)) from exc
    return False

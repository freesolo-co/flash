"""Background execution for deployment lifecycle jobs.

A deploy request persists its `queued` record and then hands the rest of the lifecycle -- smoke,
alias activation, ready commit -- to a job that outlives the response. The runner tracks every
live job so shutdown can wait for them instead of tearing down a half-activated deployment.

Instance-scoped: the control plane owns one runner, and a test owns its own. The accepting flag
and the live-job set are state on the object, so two planes in one process cannot close each
other's intake.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable


class DeploymentJobStartError(RuntimeError):
    """The job never ran, so whatever the caller was holding for it is still the caller's."""


class ThreadDeploymentJobRunner:
    """Runs each deployment lifecycle on a daemon thread, tracked until it finishes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: set[threading.Thread] = set()
        self._accepting = True

    def open(self) -> None:
        """Accept jobs again. Startup calls this so a restarted plane is not stuck draining."""
        with self._lock:
            self._accepting = True

    def start(self, target: Callable[..., object], *args, **kwargs) -> bool:
        """Start one lifecycle job.

        Returns True when it ran synchronously (test mode), False when it was handed to a thread.
        Raises ``DeploymentJobStartError`` when the job did not start, which is the caller's signal
        that it still owns the deploy lock.
        """
        if os.environ.get("FLASH_DEPLOY_SYNC") == "1":
            target(*args, **kwargs)
            return True
        thread = threading.Thread(target=self._run, args=(target, args, kwargs), daemon=True)
        with self._lock:
            if not self._accepting:
                raise DeploymentJobStartError("deployment jobs are shutting down")
            self._jobs.add(thread)
            try:
                thread.start()
            except Exception as exc:
                self._jobs.discard(thread)
                raise DeploymentJobStartError(str(exc)) from exc
        return False

    def _run(self, target: Callable[..., object], args: tuple, kwargs: dict) -> None:
        try:
            target(*args, **kwargs)
        finally:
            with self._lock:
                self._jobs.discard(threading.current_thread())

    def drain(self, timeout: float) -> bool:
        """Stop accepting jobs and wait for the live ones. Returns False on timeout."""
        deadline = time.monotonic() + timeout
        with self._lock:
            self._accepting = False
        while True:
            with self._lock:
                jobs = tuple(self._jobs)
            if not jobs:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            jobs[0].join(remaining)

    def live_jobs(self) -> tuple[threading.Thread, ...]:
        """A snapshot of the jobs still running, for shutdown diagnostics and tests."""
        with self._lock:
            return tuple(self._jobs)

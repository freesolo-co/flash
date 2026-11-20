"""terminal supervision signals."""


class _RunCancelled(RuntimeError):
    """user cancellation observed mid-run; terminal, never retried or overwritten."""


class _TerminalHandleRace(_RunCancelled):
    """a provider handle was created after the run became terminal."""


class _LaunchOwnershipLost(_RunCancelled):
    """the durable launch claim moved to another supervisor."""


class RunAttemptsFailed(RuntimeError):
    """the run's last attempt failed and the retry plan stopped; carries the attempt's detail.

    ``detail`` is ``"<failure>: <detail>"`` exactly as the attempt log prints it and as recovery
    persists it, built from Flash-authored provider details and the worker's sanitized error.
    """

    def __init__(self, run_id: str, detail: str | None):
        self.detail = detail or "job_failed: attempt failed"
        super().__init__(f"run {run_id} failed after retries: {self.detail}")

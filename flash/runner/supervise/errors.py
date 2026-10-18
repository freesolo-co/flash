"""terminal supervision signals."""


class _RunCancelled(RuntimeError):
    """user cancellation observed mid-run; terminal, never retried or overwritten."""


class _TerminalHandleRace(_RunCancelled):
    """a provider handle was created after the run became terminal."""


class _LaunchOwnershipLost(_RunCancelled):
    """the durable launch claim moved to another supervisor."""

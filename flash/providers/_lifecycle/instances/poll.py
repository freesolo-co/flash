"""shared resource poll-loop scaffolding for instance providers."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from flash.adapters.artifacts import MAX_ATTEMPT_ID
from flash.providers._lifecycle.net.deadline import remaining_seconds

PRELOAD_REAP_GRACE_S = 1800.0


def preload_instance_run_id(
    provider: str, region: str, reap_deadline_epoch: int, suffix: str
) -> str:
    """build a preload run id embedding its wall-clock reap deadline."""
    return f"flash-preload-d{int(reap_deadline_epoch)}-{provider}-{region.lower()}-{suffix}"


def preload_box_reap_due(
    name: str, now: float, grace_s: float = PRELOAD_REAP_GRACE_S
) -> bool:
    """return whether an embedded preload deadline elapsed beyond its grace."""
    match = re.search(r"-d(\d{10,})-", name)
    if not match:
        return False
    return float(match.group(1)) + grace_s < now


def make_say(log) -> Callable[[str], None]:
    """return a timestamped logger that no-ops when no stream is provided."""

    def say(message: str) -> None:
        if log is not None:
            print(f"[{time.strftime('%H:%M:%S')}] {message}", file=log, flush=True)

    return say


class PollErrorTracker:
    """count consecutive resource-status errors and apply bounded backoff."""

    def __init__(self, say: Callable[[str], None], interval_s: float, max_errors: int = 8) -> None:
        self._say = say
        self._interval_s = interval_s
        self._max_errors = max_errors
        self._count = 0

    def reset(self) -> None:
        self._count = 0

    def record(self, exc: Exception, *, deadline_at: float | None = None) -> bool:
        """register one status error and return whether observation should stop."""
        self._count += 1
        self._say(f"poll error ({self._count}): {type(exc).__name__}")
        if self._count >= self._max_errors:
            return True
        delay = min(60, self._interval_s * self._count)
        if deadline_at is not None:
            remaining = remaining_seconds(deadline_at)
            if remaining <= 0:
                return True
            delay = min(delay, remaining)
        if delay > 0:
            time.sleep(delay)
        return False


def _attempt_int(value: Any) -> int | None:
    """validate a bounded nonnegative integer attempt identity."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= MAX_ATTEMPT_ID else None

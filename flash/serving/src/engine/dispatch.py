"""Internal pre-header dispatch deadlines for hosted generation."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

PRE_HEADER_DISPATCH_TIMEOUT_SECONDS = 120.0
CAPACITY_RETRY_AFTER_SECONDS = 1


class PreHeaderDispatchExpired(RuntimeError):
    """The request waited too long to begin gpu work."""


def new_pre_header_dispatch_deadline(*, clock: Callable[[], float] | None = None) -> float:
    return (clock or time.time)() + PRE_HEADER_DISPATCH_TIMEOUT_SECONDS


def require_pre_header_dispatch_time(
    deadline: float | None,
    *,
    clock: Callable[[], float] | None = None,
) -> None:
    if deadline is None:
        return
    if not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
        raise ValueError("pre-header dispatch deadline must be a finite timestamp")
    normalized = float(deadline)
    if not math.isfinite(normalized):
        raise ValueError("pre-header dispatch deadline must be a finite timestamp")
    if (clock or time.time)() >= normalized:
        raise PreHeaderDispatchExpired("request expired before gpu generation began")

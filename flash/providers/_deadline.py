"""Private absolute run-deadline validation for provider operations."""

from __future__ import annotations

import inspect
import math
import time


def deadline_kwargs(call, deadline_at: float | None) -> dict[str, float | None]:
    """Return a deadline keyword only when the callable accepts it."""
    try:
        parameters = inspect.signature(call).parameters.values()
    except (TypeError, ValueError):
        return {"deadline_at": deadline_at}
    if any(
        parameter.name == "deadline_at" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    ):
        return {"deadline_at": deadline_at}
    return {}


def require_deadline_at(deadline_at: object) -> float:
    """Return a finite positive unix deadline or fail closed."""
    if isinstance(deadline_at, bool) or not isinstance(deadline_at, (int, float)):
        raise RuntimeError("run wall deadline is invalid")
    deadline = float(deadline_at)
    if not math.isfinite(deadline) or deadline <= 0:
        raise RuntimeError("run wall deadline is invalid")
    return deadline


def remaining_seconds(deadline_at: object, *, now: float | None = None) -> float:
    deadline = require_deadline_at(deadline_at)
    current_value = time.time() if now is None else now
    if isinstance(current_value, bool) or not isinstance(current_value, (int, float)):
        raise RuntimeError("current clock is invalid")
    current = float(current_value)
    if not math.isfinite(current) or current <= 0:
        raise RuntimeError("current clock is invalid")
    return deadline - current


def require_create_allowance(deadline_at: object, *, now: float | None = None) -> float:
    remaining = remaining_seconds(deadline_at, now=now)
    if remaining < 60.0:
        raise RuntimeError(
            "run wall deadline has less than the 60-second minimum provider allowance remaining"
        )
    return remaining

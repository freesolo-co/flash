"""Private absolute run-deadline validation for provider operations."""

from __future__ import annotations

import inspect
import math
import time

# Minimum a provider create needs left on the run deadline. Best-effort work that runs before a
# create must yield this much, or it becomes the reason the create is rejected.
CREATE_ALLOWANCE_S = 60.0


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


def require_finite_positive(value: object, invalid_message: str) -> float:
    """Return ``value`` as a finite positive float or fail closed with ``invalid_message``.

    ``bool`` is rejected explicitly: it is an ``int`` subclass, so ``True`` would otherwise pass
    as the timestamp 1.0 and silently arm a deadline that expired in 1970.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(invalid_message)
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(invalid_message)
    return number


def require_deadline_at(deadline_at: object) -> float:
    """Return a finite positive unix deadline or fail closed."""
    return require_finite_positive(deadline_at, "run wall deadline is invalid")


def remaining_seconds(deadline_at: object, *, now: float | None = None) -> float:
    deadline = require_deadline_at(deadline_at)
    current = require_finite_positive(
        time.time() if now is None else now, "current clock is invalid"
    )
    return deadline - current


def require_create_allowance(deadline_at: object, *, now: float | None = None) -> float:
    remaining = remaining_seconds(deadline_at, now=now)
    if remaining < CREATE_ALLOWANCE_S:
        raise RuntimeError(
            "run wall deadline has less than the 60-second minimum provider allowance remaining"
        )
    return remaining

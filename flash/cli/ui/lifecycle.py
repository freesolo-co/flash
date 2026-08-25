"""render attempt, progress, resource, and result observations without age inference."""

from __future__ import annotations

import math
import time
from collections.abc import Callable


def _age_seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return max(0.0, time.time() - number)


def _humanize_age(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds // 60)}m ago"
    return f"{seconds / 3600:.1f}h ago"


def live_attempt(obj: dict) -> tuple[int, int] | None:
    attempt = obj.get("attempt")
    if not isinstance(attempt, dict):
        return None
    attempt_id = attempt.get("attempt_id")
    fence = attempt.get("fence")
    if (
        isinstance(attempt_id, bool)
        or not isinstance(attempt_id, int)
        or attempt_id < 0
        or isinstance(fence, bool)
        or not isinstance(fence, int)
        or fence < 1
    ):
        return None
    return attempt_id, fence


def progress_is_current(obj: dict, progress: dict) -> bool:
    identity = live_attempt(obj)
    return bool(
        identity is not None
        and progress.get("attempt_id") == identity[0]
        and progress.get("fence") == identity[1]
    )


def _deadline_left(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds = max(0.0, float(value) - time.time())
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _metric_text(metrics: object) -> str | None:
    if not isinstance(metrics, dict) or not metrics:
        return None
    parts = []
    for key in sorted(metrics)[:12]:
        value = metrics[key]
        if isinstance(value, float):
            value = f"{value:.6g}"
        parts.append(f"{key}={value}")
    return " ".join(parts)


def _lifecycle_pairs(
    obj: dict,
    *,
    format_hint: Callable[[str], str] = str,
) -> list[tuple[str, str]]:
    """return observational lifecycle rows without deriving health from missing progress."""
    pairs: list[tuple[str, str]] = []
    attempt = obj.get("attempt")
    identity = live_attempt(obj)
    if isinstance(attempt, dict) and identity is not None:
        value = f"{identity[0]} / fence {identity[1]}"
        if attempt.get("state"):
            value += f" · {attempt['state']}"
        pairs.append(("attempt", value))
        remaining = _deadline_left(attempt.get("work_deadline_at"))
        if remaining is not None:
            pairs.append(("work deadline", f"{remaining} left"))
    resource = obj.get("resource")
    if isinstance(resource, dict) and (
        identity is None
        or (resource.get("attempt_id"), resource.get("fence")) == identity
    ):
        value = str(resource.get("state") or "unknown")
        provider = resource.get("provider")
        if provider:
            value += f" · {provider}"
        transport = resource.get("transport")
        if transport and transport != "ok":
            value += f" · transport={transport}"
        age = _humanize_age(_age_seconds(resource.get("observed_at")))
        if age is not None:
            value += f" · observed {age}"
        pairs.append(("resource", value))
    progress = obj.get("progress")
    if isinstance(progress, dict) and progress_is_current(obj, progress):
        value = str(progress.get("phase") or progress.get("kind") or "observed")
        steps = progress.get("completed_steps")
        if isinstance(steps, int) and not isinstance(steps, bool):
            value += f" · {steps} completed steps"
        pairs.append(("progress", value))
        occurred = _humanize_age(_age_seconds(progress.get("occurred_at")))
        observed = _humanize_age(_age_seconds(progress.get("observed_at")))
        if occurred is not None:
            pairs.append(("progress occurred", occurred))
        if observed is not None:
            pairs.append(("progress observed", observed))
        metrics = _metric_text(progress.get("metrics"))
        if metrics:
            pairs.append(("metrics", metrics))
        checkpoint = progress.get("checkpoint")
        if isinstance(checkpoint, dict) and checkpoint:
            pairs.append(("checkpoint", format_hint(str(checkpoint))))
    result = obj.get("result")
    if isinstance(result, dict) and (
        identity is None or (result.get("attempt_id"), result.get("fence")) == identity
    ):
        value = str(result.get("outcome") or "unknown")
        if result.get("failure_class"):
            value += f" · {result['failure_class']}"
        pairs.append(("result", value))
    return pairs

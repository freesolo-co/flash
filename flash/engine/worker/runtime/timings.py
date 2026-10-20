"""Thread-safe stage durations for identifying startup and artifact bottlenecks."""

from __future__ import annotations

import math
import threading

_LOCK = threading.Lock()
_STAGES: dict[str, dict[str, float | int]] = {}


def record(stage: str, seconds: float, *, failed: bool = False) -> None:
    if not math.isfinite(seconds) or seconds < 0:
        return
    with _LOCK:
        # Stage names are internal constants, but bound telemetry even if a new caller gets it wrong.
        if stage not in _STAGES and len(_STAGES) >= 128:
            return
        item = _STAGES.setdefault(
            stage, {"count": 0, "seconds": 0.0, "max_seconds": 0.0, "failures": 0}
        )
        item["count"] += 1
        item["seconds"] += seconds
        item["max_seconds"] = max(item["max_seconds"], seconds)
        item["failures"] += int(failed)


def snapshot() -> dict[str, dict[str, float | int]]:
    with _LOCK:
        return {stage: dict(values) for stage, values in sorted(_STAGES.items())}


def reset() -> None:
    """Clear attempt-local telemetry before a worker handles another run."""
    with _LOCK:
        _STAGES.clear()

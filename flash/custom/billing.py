"""Custom sessions charge measured session time against their frozen wall-time cap."""

import json
import math
from pathlib import Path


def session_charge(status, spec):
    from flash.runner.lifecycle.state import artifacts_dir

    seconds = 0.0
    try:
        metrics = json.loads((Path(artifacts_dir(spec)) / "metrics.json").read_text())
        seconds = metrics["session_seconds"]
    except (OSError, ValueError, KeyError):
        heartbeat = status.last_heartbeat or {}
        started = heartbeat.get("session_started_at")
        if isinstance(started, (int, float)) and not isinstance(started, bool):
            ended = getattr(status, "finished_at", None) or status.updated_at
            seconds = max(0.0, ended - started)
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
    ):
        raise ValueError("invalid custom session time")
    quote = status.estimated_cost_usd
    if (
        isinstance(quote, bool)
        or not isinstance(quote, (int, float))
        or not math.isfinite(quote)
        or quote < 0
    ):
        raise ValueError("custom run has no valid accepted session cap")
    return quote * min(1.0, max(0.0, seconds) / spec.custom["max_wall_seconds"])

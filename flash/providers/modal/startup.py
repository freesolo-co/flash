"""Attempt-scoped bootstrap evidence shared by supervision and cost accounting."""

import json
import math
import time

from flash.providers.artifacts.hf import make_hf_text_reader


def start_time_reader(remote, *, deadline_at=None):
    """Read the worker's timestamp, never the sandbox's pre-allocation creation time."""
    repo, run_id, phase = (remote.get(key) for key in ("hf_repo", "run_id", "phase"))
    attempt, launch = remote.get("attempt"), remote.get("started_ts")
    if (
        not all((repo, run_id, phase))
        or type(attempt) is not int
        or attempt < 0
        or type(launch) not in (int, float)
        or not math.isfinite(launch)
        or launch <= 0
    ):
        return lambda: None
    reader = make_hf_text_reader(
        repo,
        f"{phase}/{run_id}/modal_started_attempt{attempt}.json",
        deadline_at=deadline_at,
    )

    def read():
        try:
            marker = json.loads(reader() or "null")
            if not isinstance(marker, dict) or marker.get("run_id") != run_id:
                return None
            if type(marker.get("attempt")) is not int or marker["attempt"] != attempt:
                return None
            ts = marker.get("ts")
            if type(ts) not in (int, float) or not math.isfinite(ts):
                return None
            if not launch <= ts <= time.time():
                return None
            return ts
        except (ValueError, TypeError):
            return None

    return read

"""Region instance types Mithril just refused are skipped for a while, across runs and restarts.

A region whose VMs never boot is excluded outright (``prices.EXCLUDED_REGIONS``); a single boot
timeout does not bench a region, since one bad VM in a healthy region would stall every run.

State lives in its own directory under ``RUNS_DIR`` (never a top-level ``*.json``, which the run
scanners would read as a run record), guarded by an exclusive lock for read-modify-write.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX dev machines skip cross-process locking
    fcntl = None

# A bid Mithril definitively refused (and that reconciliation proved was never created) cools down
# only that region's instance type, and for less time: the refusal says nothing about other shapes
# there, and a quota or catalog change can clear it.
REFUSED_COOLDOWN_S = 3600.0


def _path() -> Path:
    from flash.runner.lifecycle import state

    return Path(state.RUNS_DIR) / ".spot-region-health" / "cooldown.json"


def _load(path: Path) -> dict[str, float]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        region: float(until)
        for region, until in raw.items()
        if isinstance(region, str) and isinstance(until, (int, float))
    }


@contextlib.contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.parent / ".lock", "a") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _refused_key(region: str, instance_type: str) -> str:
    # "/" never appears in a Mithril region name, so these keys cannot collide with a region entry.
    return f"{region}/{instance_type}"


def mark_refused(region: str, instance_type: str, *, now: float | None = None) -> None:
    """Skip one region's instance type for a while after Mithril refused a bid on it."""
    if not all(isinstance(value, str) and value for value in (region, instance_type)):
        return
    _mark(_refused_key(region, instance_type), REFUSED_COOLDOWN_S, now=now)


def refused(region: str, instance_type: str, *, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return _load(_path()).get(_refused_key(region, instance_type), 0.0) > now


def _mark(key: str, seconds: float, *, now: float | None = None) -> None:
    now = time.time() if now is None else now
    path = _path()
    with _locked(path):
        entries = {r: u for r, u in _load(path).items() if u > now}
        entries[key] = now + seconds
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".cooldown.")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(entries, stream)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

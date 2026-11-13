"""Regions whose allocated Spot VMs never booted are skipped for a while, across runs and restarts.

Mithril can list capacity and accept bids in a region whose instances then stay ``STATUS_NEW``
(af-south1-a, 2026-11-12: every VM there sat unbooted while us-central2 booted in minutes). Each such
bid holds a project quota slot for the whole boot window, so one boot timeout cools the region down
for every run rather than letting each run rediscover it.

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

COOLDOWN_S = 6 * 3600.0
# A definitive bid refusal (Mithril org/region limits that /quotas does not list) cools down only
# that auction shape, and briefly: the limit frees as soon as other bids on the shape end.
REFUSED_COOLDOWN_S = 15 * 60.0


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


def mark_unbootable(region: str, *, now: float | None = None, seconds: float = COOLDOWN_S) -> None:
    if not isinstance(region, str) or not region:
        return
    now = time.time() if now is None else now
    path = _path()
    with _locked(path):
        entries = {r: u for r, u in _load(path).items() if u > now}
        entries[region] = max(entries.get(region, 0.0), now + seconds)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".cooldown.")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(entries, stream)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def cooling_down(region: str, *, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return _load(_path()).get(region, 0.0) > now


def refused_key(region: str, instance_type: str) -> str:
    """Cooldown key for one auction shape (never a bare region, so region cooldowns are unaffected)."""
    return f"{region}/{instance_type}"


def mark_refused(region: str, instance_type: str, *, now: float | None = None) -> None:
    if not region or not instance_type:
        return
    mark_unbootable(refused_key(region, instance_type), now=now, seconds=REFUSED_COOLDOWN_S)

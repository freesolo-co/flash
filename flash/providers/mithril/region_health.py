"""Region instance types Mithril just refused are skipped for a while, across runs and restarts.

A region whose VMs never boot is excluded outright (``prices.EXCLUDED_REGIONS``); a single boot
timeout does not bench a region, since one bad VM in a healthy region would stall every run.
It does demote that region's instance type (its "pool"): ``boot_failures`` counts the pool's
recent boot timeouts, and the Spot queue bids on a pool with fewer of them first, within the same
GPU class. A demoted pool is still bid on when it is the only one with an offer, so no run is ever
left queued by it, and the next VM that boots there clears it.

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
# only that region's instance type, and only briefly: the refusal says nothing about other shapes
# there, and the usual cause, an organization quota or open-bid cap that Flash cannot read, clears
# as soon as other bids end. An hour-long cooldown left freed quota idle (2026-11-18: 13 H100s free
# while 50 queued runs waited), so queued runs rebid the shape every couple of minutes instead.
REFUSED_COOLDOWN_S = 120.0


# How long a boot timeout keeps demoting its pool. 2026-11-20: from ~21:00Z, 16 of 17 H200 VMs sat
# in booting_instance for the whole 30-minute window, attempt after attempt, while every H100 VM
# booted: ranking by price and region alone rebid each H200 run into the same failing pool.
BOOT_FAILURE_WINDOW_S = 6 * 3600.0


def _path() -> Path:
    from flash.runner.lifecycle import state

    return Path(state.RUNS_DIR) / ".spot-region-health" / "cooldown.json"


def _boot_path() -> Path:
    # its own file: cooldown.json holds expiry floats, and releases without this module read it.
    return _path().parent / "boot_failures.json"


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


def _load_boot(path: Path) -> dict[str, list[float]]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        key: [float(ts) for ts in stamps if isinstance(ts, (int, float))]
        for key, stamps in raw.items()
        if isinstance(key, str) and isinstance(stamps, list)
    }


def _update_boot(key: str, change, *, now: float) -> None:
    path = _boot_path()
    with _locked(path):
        entries = {
            k: kept
            for k, stamps in _load_boot(path).items()
            if (kept := [ts for ts in stamps if ts > now - BOOT_FAILURE_WINDOW_S])
        }
        updated = change(entries.get(key, []))
        if updated:
            entries[key] = updated
        else:
            entries.pop(key, None)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".boot.")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(entries, stream)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def _valid_pool(region, instance_type) -> bool:
    return all(isinstance(value, str) and value for value in (region, instance_type))


def mark_boot_failure(region: str, instance_type: str, *, now: float | None = None) -> None:
    """Record that a bid in this pool never produced a running VM within the boot window."""
    if not _valid_pool(region, instance_type):
        return
    now = time.time() if now is None else now
    _update_boot(_refused_key(region, instance_type), lambda stamps: [*stamps, now], now=now)


def mark_booted(region: str, instance_type: str, *, now: float | None = None) -> None:
    """A VM in this pool reached running: its earlier boot timeouts no longer demote it."""
    if not _valid_pool(region, instance_type):
        return
    now = time.time() if now is None else now
    if _refused_key(region, instance_type) not in _load_boot(_boot_path()):
        return  # the common case: no write, no lock
    _update_boot(_refused_key(region, instance_type), lambda stamps: [], now=now)


def boot_failures(region: str, instance_type: str, *, now: float | None = None) -> int:
    """Boot timeouts in this pool within the window and since its last successful boot."""
    now = time.time() if now is None else now
    stamps = _load_boot(_boot_path()).get(_refused_key(region, instance_type), [])
    return sum(1 for ts in stamps if ts > now - BOOT_FAILURE_WINDOW_S)

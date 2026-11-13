"""Regions whose filled Spot bids never booted are skipped for a while, across runs and restarts.

Mithril can list capacity and accept bids in a region whose instances then stay ``STATUS_NEW``
(af-south1-a, 2026-11-12: every VM there sat unbooted while us-central2 booted in minutes). Each such
bid holds a project quota slot for the whole boot window, so one boot timeout cools the region down
for every run rather than letting each run rediscover it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

COOLDOWN_S = 6 * 3600.0


def _path() -> Path:
    from flash.runner.lifecycle import state

    return Path(state.RUNS_DIR) / ".spot-region-cooldown.json"


def _load() -> dict[str, float]:
    try:
        raw = json.loads(_path().read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        region: float(until)
        for region, until in raw.items()
        if isinstance(region, str) and isinstance(until, (int, float))
    }


def mark_unbootable(region: str, *, now: float | None = None) -> None:
    if not isinstance(region, str) or not region:
        return
    now = time.time() if now is None else now
    entries = {r: u for r, u in _load().items() if u > now}
    entries[region] = now + COOLDOWN_S
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".spot-region-cooldown.")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(entries, stream)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def cooling_down(region: str, *, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return _load().get(region, 0.0) > now

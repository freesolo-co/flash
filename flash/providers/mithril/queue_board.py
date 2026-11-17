"""Flash's own view of its Spot queue: who is waiting, why, and how fast GPUs have been coming.

Mithril does not say where a bid would stand, so Flash keeps a small board of its own queued Spot
runs: when each joined (its position is its order among runs waiting for an overlapping GPU class),
when recent queued runs actually got a bid accepted (the only honest basis for an ETA), and when
Mithril last refused a bid for quota (Mithril's account quota is not visible through ``/quotas``
before a refusal, so a recent quota refusal is the evidence that the queue is quota-bound).

State lives under ``RUNS_DIR`` in its own directory (never a top-level ``*.json``, which the run
scanners would read as a run record), guarded by an exclusive lock for read-modify-write. Every
public function is diagnostic: a failure here must never disturb the queue itself.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX dev machines skip cross-process locking
    fcntl = None

# A waiting run refreshes its entry every queue pass (~60 s) and across a refused-bid requeue
# (~60 s pause plus the next create); an entry this old belongs to a run that is no longer waiting.
STALE_AFTER_S = 900.0
# Allocations older than this say little about the current queue.
ALLOCATION_WINDOW_S = 3 * 3600.0
# An ETA needs at least this many recent allocations; fewer is reported as unknown.
MIN_ALLOCATIONS_FOR_ETA = 2
# A quota refusal keeps describing the queue for as long as the refused shape is skipped.
QUOTA_REFUSAL_WINDOW_S = 3600.0

REASON_QUOTA = "gpu_quota_full"
REASON_PRICE = "no_gpu_below_price_cap"
REASON_LOOKUP = "capacity_lookup_failed"
REASON_TEXT = {
    REASON_QUOTA: "GPU quota full",
    REASON_PRICE: "no GPU below the price cap",
    REASON_LOOKUP: "Spot market temporarily unreadable",
}


def _path() -> Path:
    from flash.runner.lifecycle import state

    return Path(state.RUNS_DIR) / ".spot-queue" / "board.json"


def _empty() -> dict:
    return {"waiting": {}, "allocations": [], "quota_refusals": {}}


def _load(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return _empty()
    if not isinstance(raw, dict):
        return _empty()
    board = _empty()
    waiting = raw.get("waiting")
    if isinstance(waiting, dict):
        board["waiting"] = {
            run_id: entry
            for run_id, entry in waiting.items()
            if isinstance(run_id, str)
            and isinstance(entry, dict)
            and _number(entry.get("entered_at"))
            and _number(entry.get("seen_at"))
            and isinstance(entry.get("gpus"), list)
        }
    allocations = raw.get("allocations")
    if isinstance(allocations, list):
        board["allocations"] = [float(ts) for ts in allocations if _number(ts)]
    refusals = raw.get("quota_refusals")
    if isinstance(refusals, dict):
        board["quota_refusals"] = {
            gpu: float(ts) for gpu, ts in refusals.items() if isinstance(gpu, str) and _number(ts)
        }
    return board


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@contextlib.contextmanager
def _locked_board():
    """Yield the board for read-modify-write; it is saved when the block exits cleanly."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.parent / ".lock", "a") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            board = _load(path)
            yield board
            _prune(board, time.time())
            fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".board.")
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(board, stream)
                os.replace(temporary, path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        finally:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _prune(board: dict, now: float) -> None:
    board["waiting"] = {
        run_id: entry
        for run_id, entry in board["waiting"].items()
        if now - entry["seen_at"] <= STALE_AFTER_S
    }
    board["allocations"] = sorted(
        ts for ts in board["allocations"] if now - ts <= ALLOCATION_WINDOW_S
    )[-500:]
    board["quota_refusals"] = {
        gpu: ts for gpu, ts in board["quota_refusals"].items() if now - ts <= QUOTA_REFUSAL_WINDOW_S
    }


@dataclass(frozen=True)
class QueueView:
    """Where one run stands in Flash's Spot queue, as of ``at``."""

    at: float
    entered_at: float
    position: int
    waiting: int
    recent_allocations: int
    allocation_span_s: float
    quota_refused_at: float | None

    def eta_seconds(self) -> float | None:
        """Position over the recent allocation rate, or None when there is too little to go on."""
        if self.recent_allocations < MIN_ALLOCATIONS_FOR_ETA:
            return None
        # never extrapolate a burst: the rate is measured over at least ten minutes.
        span = max(self.allocation_span_s, 600.0)
        return self.position * span / self.recent_allocations

    def eta_text(self) -> str:
        eta = self.eta_seconds()
        if eta is None:
            return (
                "ETA unknown (too few recent Spot allocations to estimate from; Mithril does not "
                "publish queue positions)"
            )
        return (
            f"ETA roughly {_rough_duration(eta)} (rough: {self.recent_allocations} queued Flash "
            f"Spot runs got a GPU in the last {_rough_duration(self.allocation_span_s)}; "
            "not a promise)"
        )

    def eta_short(self) -> str:
        eta = self.eta_seconds()
        return "ETA unknown" if eta is None else f"ETA roughly {_rough_duration(eta)}"

    def public(self, reason: str, reason_detail: str) -> dict:
        eta = self.eta_seconds()
        return {
            "reason": reason,
            "reason_detail": reason_detail,
            "position": self.position,
            "waiting": self.waiting,
            "entered_at": self.entered_at,
            "eta_seconds": None if eta is None else int(eta),
            "eta": self.eta_text(),
        }


def _rough_duration(seconds: float) -> str:
    """Coarse on purpose: 5-minute steps under 90 minutes, half hours above."""
    minutes = max(0.0, seconds) / 60
    if minutes < 90:
        return f"{max(5, math.ceil(minutes / 5) * 5)} min"
    hours = math.ceil(minutes / 30) / 2
    return f"{hours:g} h"


def _overlaps(entry: dict, gpus: tuple[str, ...]) -> bool:
    return bool(set(entry.get("gpus") or ()) & set(gpus))


def _view(board: dict, run_id: str, gpus: tuple[str, ...], now: float) -> QueueView | None:
    entry = board["waiting"].get(run_id)
    if entry is None:
        return None
    rivals = [
        (other["entered_at"], other_id)
        for other_id, other in board["waiting"].items()
        if now - other["seen_at"] <= STALE_AFTER_S and _overlaps(other, gpus)
    ]
    ahead = sum(1 for key in rivals if key < (entry["entered_at"], run_id))
    recent = sorted(ts for ts in board["allocations"] if now - ts <= ALLOCATION_WINDOW_S)
    refusals = [
        ts
        for gpu, ts in board["quota_refusals"].items()
        if gpu in gpus and now - ts <= QUOTA_REFUSAL_WINDOW_S
    ]
    return QueueView(
        at=now,
        entered_at=entry["entered_at"],
        position=ahead + 1,
        waiting=max(len(rivals), ahead + 1),
        recent_allocations=len(recent),
        allocation_span_s=(now - recent[0]) if recent else 0.0,
        quota_refused_at=max(refusals) if refusals else None,
    )


def join(run_id: str, gpus) -> QueueView | None:
    """Record (or refresh) ``run_id`` as waiting for ``gpus``; keeps its original entry time."""
    gpus = (gpus,) if isinstance(gpus, str) else tuple(gpus)
    with contextlib.suppress(Exception):
        now = time.time()
        with _locked_board() as board:
            entry = board["waiting"].get(run_id)
            board["waiting"][run_id] = {
                "entered_at": entry["entered_at"] if entry else now,
                "seen_at": now,
                "gpus": sorted(set(gpus) | set(entry["gpus"] if entry else ())),
            }
            return _view(board, run_id, gpus, now)
    return None


def view(run_id: str) -> QueueView | None:
    """Where ``run_id`` stands now, without refreshing it; None if it is not waiting."""
    with contextlib.suppress(Exception):
        board = _load(_path())
        entry = board["waiting"].get(run_id)
        if entry is None:
            return None
        return _view(board, run_id, tuple(entry["gpus"]), time.time())
    return None


def leave(run_id: str, *, allocated: bool) -> None:
    """``run_id`` stopped waiting; ``allocated`` means Mithril accepted its bid (an ETA sample)."""
    with contextlib.suppress(Exception), _locked_board() as board:
        was_waiting = board["waiting"].pop(run_id, None) is not None
        if allocated and was_waiting:
            board["allocations"].append(time.time())


def record_quota_refusal(gpu: str) -> None:
    """Mithril refused a bid on ``gpu`` because an account or project GPU quota is full."""
    if not isinstance(gpu, str) or not gpu:
        return
    with contextlib.suppress(Exception), _locked_board() as board:
        board["quota_refusals"][gpu] = time.time()


def is_quota_refusal(text: object) -> bool:
    """True when a provider refusal names a GPU quota (e.g. "Self-serve GPU quota exceeded")."""
    return isinstance(text, str) and "quota" in text.lower()


def status_extra(view: QueueView | None, reason: str) -> str:
    """The short ``(...)`` suffix of the status detail line while queued."""
    text = f"Spot: {REASON_TEXT.get(reason, reason)}"
    if view is None:
        return text
    return f"{text}; position {view.position} of {view.waiting}; {view.eta_short()}"


def queue_line(view: QueueView | None, gpus, reason: str, reason_detail: str) -> str:
    """The queue's one-line log entry: why it waits, where it stands and a rough ETA."""
    gpus = (gpus,) if isinstance(gpus, str) else tuple(gpus)
    head = (
        f"Spot queued ({REASON_TEXT.get(reason, reason)}: {reason_detail}): "
        f"waiting for {' or '.join(gpus)}"
    )
    if view is None:
        return f"{head}; queue position unknown; ETA unknown"
    return (
        f"{head}; position {view.position} of {view.waiting} among Flash Spot runs waiting for "
        f"these GPUs; {view.eta_text()}"
    )

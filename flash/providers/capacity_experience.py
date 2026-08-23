"""Cross-run memory of GPU shapes that recently refused capacity.

This ledger is an allocation hint, not an eligibility or retry verdict. A recent refusal may move a
shape behind an unrefused peer inside the same authored provider preference rank, but it never removes
that shape and never enters the per-run capacity exhaustion tally.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

try:
    import fcntl
except ImportError:  # pragma: no cover - allocation degrades to no experience below
    fcntl = None

from flash._internal.logging import get_logger

logger = get_logger(__name__)

_LEDGER_NAME = "capacity-experience.json"
_LOCK_NAME = "capacity-experience.lock"
_LEDGER_VERSION = 1
_SHAPE_SEPARATOR = "\x1f"
_MAX_TRACKED_SHAPES = 256

# runpod and vast capacity commonly returns within minutes. twenty minutes is long enough to route
# around a repeatedly dry listing across nearby runs, but short enough that a recovered market gets
# today's exact cost ordering back without waiting through an operator-scale outage window.
CAPACITY_REFUSAL_TTL_S = 20 * 60

CapacityShape = tuple[str, str, int]


@dataclass(frozen=True)
class CapacityExperience:
    """Validated history for one provider, GPU class, and card-count shape."""

    last_refusal_at: float | None
    refusal_count: int
    last_success_at: float | None


def _paths() -> tuple[str, str, str]:
    from flash.runner import RUNS_DIR

    os.makedirs(RUNS_DIR, exist_ok=True)
    return (
        RUNS_DIR,
        os.path.join(RUNS_DIR, _LEDGER_NAME),
        os.path.join(RUNS_DIR, _LOCK_NAME),
    )


@contextmanager
def _locked_ledger(*, exclusive: bool) -> Iterator[tuple[str, str]]:
    if fcntl is None:
        raise RuntimeError("interprocess capacity-experience locking is unavailable")
    runs_dir, ledger_path, lock_path = _paths()
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield runs_dir, ledger_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _shape_key(shape: CapacityShape) -> str:
    provider, gpu, count = _validated_shape(shape)
    return _SHAPE_SEPARATOR.join((provider, gpu, str(count)))


def _shape_from_key(key: object) -> CapacityShape:
    if not isinstance(key, str):
        raise ValueError("capacity experience shape key must be a string")
    parts = key.split(_SHAPE_SEPARATOR)
    if len(parts) != 3:
        raise ValueError(f"invalid capacity experience shape key: {key!r}")
    provider, gpu, raw_count = parts
    try:
        count = int(raw_count)
    except ValueError:
        raise ValueError(f"invalid capacity experience shape count: {key!r}") from None
    if raw_count != str(count):
        raise ValueError(f"non-canonical capacity experience shape count: {key!r}")
    return _validated_shape((provider, gpu, count))


def _validated_shape(shape: object) -> CapacityShape:
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError(f"invalid capacity experience shape: {shape!r}")
    provider, gpu, count = shape
    if (
        not isinstance(provider, str)
        or not provider
        or _SHAPE_SEPARATOR in provider
        or not isinstance(gpu, str)
        or not gpu
        or _SHAPE_SEPARATOR in gpu
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise ValueError(f"invalid capacity experience shape: {shape!r}")
    return provider, gpu, count


def _optional_timestamp(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid capacity experience {field}")
    timestamp = float(value)
    if timestamp < 0 or not math.isfinite(timestamp):
        raise ValueError(f"invalid capacity experience {field}")
    return timestamp


def _entry_from_json(raw: object) -> CapacityExperience:
    if not isinstance(raw, dict) or set(raw) != {
        "last_refusal_at",
        "refusal_count",
        "last_success_at",
    }:
        raise ValueError("invalid capacity experience entry")
    refusal_count = raw["refusal_count"]
    if isinstance(refusal_count, bool) or not isinstance(refusal_count, int) or refusal_count < 0:
        raise ValueError("invalid capacity experience refusal count")
    refusal = _optional_timestamp(raw["last_refusal_at"], field="refusal timestamp")
    success = _optional_timestamp(raw["last_success_at"], field="success timestamp")
    if refusal is None and success is None:
        raise ValueError("capacity experience entry has no observation")
    active_refusal = refusal is not None and (success is None or refusal > success)
    if active_refusal != (refusal_count > 0):
        raise ValueError("capacity experience count disagrees with its latest observation")
    return CapacityExperience(refusal, refusal_count, success)


def _read_unlocked(path: str) -> dict[CapacityShape, CapacityExperience]:
    try:
        with open(path) as ledger_file:
            raw = json.load(ledger_file)
    except FileNotFoundError:
        return {}
    if not isinstance(raw, dict) or set(raw) != {"version", "shapes"}:
        raise ValueError(f"invalid capacity experience ledger: {path}")
    if raw["version"] != _LEDGER_VERSION or isinstance(raw["version"], bool):
        raise ValueError(f"invalid capacity experience ledger version: {path}")
    shapes = raw["shapes"]
    if not isinstance(shapes, dict) or len(shapes) > _MAX_TRACKED_SHAPES:
        raise ValueError(f"invalid capacity experience shape map: {path}")
    # one unreadable ENTRY drops that entry, not the file. the whole-ledger raise is reserved for a
    # structurally invalid container above, because the two failures have very different blast
    # radii: a single shape whose observations disagree costs one shape's hint, while discarding
    # the map costs every OTHER shape's hint too and the next write then persists that loss.
    entries: dict[CapacityShape, CapacityExperience] = {}
    for key, value in shapes.items():
        try:
            entries[_shape_from_key(key)] = _entry_from_json(value)
        except ValueError:
            logger.debug("dropping unreadable capacity experience entry %r", key)
    return entries


def _fsync_directory(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_unlocked(
    runs_dir: str,
    path: str,
    entries: dict[CapacityShape, CapacityExperience],
) -> None:
    payload = {
        "version": _LEDGER_VERSION,
        "shapes": {
            _shape_key(shape): {
                "last_refusal_at": entry.last_refusal_at,
                "refusal_count": entry.refusal_count,
                "last_success_at": entry.last_success_at,
            }
            for shape, entry in entries.items()
        },
    }
    fd, tmp = tempfile.mkstemp(dir=runs_dir, prefix=f"{os.path.basename(path)}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as ledger_file:
            json.dump(payload, ledger_file, indent=2, sort_keys=True)
            ledger_file.flush()
            os.fsync(ledger_file.fileno())
        os.replace(tmp, path)
        _fsync_directory(runs_dir)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def _observation_time(entry: CapacityExperience) -> float:
    return max(entry.last_refusal_at or 0.0, entry.last_success_at or 0.0)


def _bounded(
    entries: dict[CapacityShape, CapacityExperience],
) -> dict[CapacityShape, CapacityExperience]:
    newest = sorted(
        entries.items(),
        key=lambda item: (_observation_time(item[1]), _shape_key(item[0])),
        reverse=True,
    )[:_MAX_TRACKED_SHAPES]
    return dict(newest)


def read_capacity_experience() -> dict[CapacityShape, CapacityExperience]:
    """Read the plane-wide hint ledger, degrading any invalid or unavailable file to no experience.

    Verified revisions fail closed because they authorize deployment. This ledger only changes which
    eligible candidate is tried first, so refusing allocation over a truncated hint would invert its
    purpose. Strict parsing remains inside the lock; the public optimization boundary catches that
    failure and returns the exact no-ledger state.
    """
    try:
        with _locked_ledger(exclusive=False) as (_, path):
            return _read_unlocked(path)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "capacity experience unavailable (%s); using cost-only allocation",
            type(exc).__name__,
        )
        return {}


def recent_capacity_refusals(
    experience: dict[CapacityShape, CapacityExperience],
    *,
    now: float | None = None,
) -> frozenset[CapacityShape]:
    """Shapes whose latest observation is a refusal inside the twenty-minute routing window.

    The hard expiry is deliberate: after it every candidate receives the same experience tier, so the
    allocator's ordering is byte-for-byte the cost and preference ordering used before this ledger.
    """
    observed_at = time.time() if now is None else float(now)
    recent = set()
    for shape, entry in experience.items():
        refusal = entry.last_refusal_at
        if refusal is None or (
            entry.last_success_at is not None and entry.last_success_at >= refusal
        ):
            continue
        age = observed_at - refusal
        if 0 <= age < CAPACITY_REFUSAL_TTL_S:
            recent.add(shape)
    return frozenset(recent)


def _record(shape: CapacityShape, *, success: bool, now: float | None) -> None:
    shape = _validated_shape(shape)
    if now is not None and (float(now) < 0 or not math.isfinite(float(now))):
        raise ValueError(f"invalid capacity experience observation time: {now!r}")
    try:
        with _locked_ledger(exclusive=True) as (runs_dir, path):
            # sampled UNDER the lock, never before it. a timestamp taken outside the lock can be
            # merged with state a concurrent writer committed in between, which persists a refusal
            # older than the success it is merged with -- an entry whose count disagrees with its
            # latest observation, exactly what `_entry_from_json` rejects on the next read.
            observed_at = float(now) if now is not None else time.time()
            try:
                entries = _read_unlocked(path)
            except ValueError:
                # malformed optimization state must not become permanent. the next valid observation
                # repairs it under the same exclusive lock rather than preserving an unreadable file.
                entries = {}
            previous = entries.get(shape, CapacityExperience(None, 0, None))
            if success:
                # monotonic: an explicitly supplied older sample must not walk the success time
                # backwards past a refusal already recorded after it, which would leave the entry
                # claiming an active refusal with a zero count.
                last_success = max(observed_at, previous.last_success_at or observed_at)
                if previous.last_refusal_at is not None and last_success < previous.last_refusal_at:
                    # a success older than the standing refusal describes a market that has since
                    # refused this shape, so it says nothing new. zeroing the count here would
                    # persist an active refusal with a zero count -- the exact disagreement
                    # `_entry_from_json` rejects, which drops the entry on the next read. this is
                    # the mirror of the refusal branch below; both keep count and latest
                    # observation agreeing.
                    entries[shape] = previous
                else:
                    entries[shape] = CapacityExperience(
                        previous.last_refusal_at,
                        0,
                        last_success,
                    )
            else:
                # a refusal older than the last success describes a market that has since admitted
                # this shape, so it says nothing new. keeping the newer success (rather than
                # stamping the stale time) is what keeps `refusal_count` and the latest observation
                # agreeing, which is the invariant `_entry_from_json` enforces on the next read.
                last_refusal = max(observed_at, previous.last_refusal_at or observed_at)
                after_success = previous.last_success_at is not None and (
                    previous.last_refusal_at is None
                    or previous.last_success_at >= previous.last_refusal_at
                )
                if (
                    previous.last_success_at is not None
                    and last_refusal <= previous.last_success_at
                ):
                    entries[shape] = previous
                else:
                    entries[shape] = CapacityExperience(
                        last_refusal,
                        1 if after_success else previous.refusal_count + 1,
                        previous.last_success_at,
                    )
            _write_unlocked(runs_dir, path, _bounded(entries))
    except Exception as exc:
        # a lost hint is cheaper than turning an otherwise runnable job into a control-plane failure.
        logger.warning(
            "capacity experience write skipped (%s); continuing without durable memory",
            type(exc).__name__,
        )


def record_capacity_refusal(shape: CapacityShape, *, now: float | None = None) -> None:
    """Best-effort record that one exact provider, class, and count refused capacity."""
    _record(shape, success=False, now=now)


def record_capacity_success(shape: CapacityShape, *, now: float | None = None) -> None:
    """Best-effort record that one exact shape admitted work, immediately forgiving a refusal."""
    _record(shape, success=True, now=now)

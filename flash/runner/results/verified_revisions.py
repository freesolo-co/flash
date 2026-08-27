"""cross-process persistence for verified permanent checkpoints."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from flash.schema import parse_checkpoint_ref

_LEDGER_SUFFIX = ".verified-checkpoints"
_LOCK_SUFFIX = ".verified-checkpoints.lock"


def _paths(run_id: str) -> tuple[str, str, str]:
    from flash.runner.lifecycle.state import RUNS_DIR, runs_file_path

    os.makedirs(RUNS_DIR, exist_ok=True)
    return (
        RUNS_DIR,
        runs_file_path(run_id, _LEDGER_SUFFIX),
        runs_file_path(run_id, _LOCK_SUFFIX),
    )


@contextmanager
def _locked_ledger(run_id: str, *, exclusive: bool) -> Iterator[tuple[str, str]]:
    if fcntl is None:
        raise RuntimeError("interprocess verified-checkpoint locking is unavailable")
    runs_dir, ledger_path, lock_path = _paths(run_id)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield runs_dir, ledger_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_unlocked(path: str, run_id: str) -> tuple[int, list[str]]:
    try:
        with open(path) as ledger_file:
            raw = json.load(ledger_file)
    except FileNotFoundError:
        return 0, []
    if not isinstance(raw, dict):
        raise ValueError(f"invalid verified checkpoint ledger: {path}")
    generation = raw.get("generation")
    checkpoints = raw.get("checkpoints")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError(f"invalid verified checkpoint generation: {path}")
    if not isinstance(checkpoints, list):
        raise ValueError(f"invalid verified checkpoint membership: {path}")
    for checkpoint_id in checkpoints:
        parsed = parse_checkpoint_ref(checkpoint_id) if isinstance(checkpoint_id, str) else None
        if parsed is None or parsed[0] != run_id:
            raise ValueError(f"invalid verified checkpoint for run {run_id}: {checkpoint_id!r}")
    return generation, checkpoints


def _fsync_directory(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_unlocked(runs_dir: str, path: str, generation: int, checkpoints: list[str]) -> None:
    fd, tmp = tempfile.mkstemp(dir=runs_dir, prefix=f"{os.path.basename(path)}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as ledger_file:
            json.dump(
                {"generation": generation, "checkpoints": checkpoints},
                ledger_file,
                indent=2,
                sort_keys=True,
            )
            ledger_file.flush()
            os.fsync(ledger_file.fileno())
        os.replace(tmp, path)
        _fsync_directory(runs_dir)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def read_verified_checkpoints(run_id: str) -> frozenset[str]:
    """read the authoritative verified checkpoint membership for a run."""

    with _locked_ledger(run_id, exclusive=False) as (_, path):
        _, checkpoints = _read_unlocked(path, run_id)
        return frozenset(checkpoints)


def verified_checkpoint_generation(run_id: str) -> int:
    """read the current lifecycle generation for a run's checkpoint membership."""

    with _locked_ledger(run_id, exclusive=False) as (_, path):
        generation, _ = _read_unlocked(path, run_id)
        return generation


def commit_verified_checkpoint(
    run_id: str,
    checkpoint_id: str,
    *,
    expected_generation: int,
    commit: Callable[[], None],
    advance_generation: bool = False,
) -> bool:
    """persist verified membership before committing its ready status."""

    parsed = parse_checkpoint_ref(checkpoint_id)
    if parsed is None or parsed[0] != run_id:
        raise ValueError(f"invalid verified checkpoint for run {run_id}: {checkpoint_id!r}")
    if (
        isinstance(expected_generation, bool)
        or not isinstance(expected_generation, int)
        or expected_generation < 0
    ):
        raise ValueError(f"invalid verified checkpoint generation: {expected_generation!r}")
    with _locked_ledger(run_id, exclusive=True) as (runs_dir, path):
        generation, checkpoints = _read_unlocked(path, run_id)
        if generation != expected_generation:
            return False
        retained_generation = generation + 1 if advance_generation else generation
        retained = list(checkpoints)
        if checkpoint_id not in retained:
            retained.append(checkpoint_id)
        retained.sort()
        if retained_generation != generation or retained != checkpoints:
            _write_unlocked(runs_dir, path, retained_generation, retained)
        commit()
        return True


def add_verified_checkpoint(
    run_id: str,
    checkpoint_id: str,
    *,
    expected_generation: int,
) -> bool:
    return commit_verified_checkpoint(
        run_id,
        checkpoint_id,
        expected_generation=expected_generation,
        commit=lambda: None,
    )


def remove_verified_checkpoint(
    run_id: str,
    checkpoint_id: str,
    *,
    commit: Callable[[frozenset[str]], None],
) -> int:
    """remove one checkpoint and commit against the retained verified membership."""

    parsed = parse_checkpoint_ref(checkpoint_id)
    if parsed is None or parsed[0] != run_id:
        raise ValueError(f"invalid verified checkpoint for run {run_id}: {checkpoint_id!r}")
    with _locked_ledger(run_id, exclusive=True) as (runs_dir, path):
        generation, checkpoints = _read_unlocked(path, run_id)
        generation += 1
        retained = [value for value in checkpoints if value != checkpoint_id]
        _write_unlocked(runs_dir, path, generation, retained)
        commit(frozenset(retained))
        return generation


def invalidate_verified_checkpoints(
    run_id: str,
    *,
    commit: Callable[[], None],
) -> int:
    """clear run membership for internal administrative cleanup."""

    with _locked_ledger(run_id, exclusive=True) as (runs_dir, path):
        generation, _ = _read_unlocked(path, run_id)
        generation += 1
        _write_unlocked(runs_dir, path, generation, [])
        commit()
        return generation

"""Cross-process persistence for verified immutable adapter revisions."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from flash.schema import parse_adapter_revision

_LEDGER_SUFFIX = ".verified-revisions"
_LOCK_SUFFIX = ".verified-revisions.lock"


def _paths(run_id: str) -> tuple[str, str, str]:
    from flash.runner import RUNS_DIR, runs_file_path

    os.makedirs(RUNS_DIR, exist_ok=True)
    return (
        RUNS_DIR,
        runs_file_path(run_id, _LEDGER_SUFFIX),
        runs_file_path(run_id, _LOCK_SUFFIX),
    )


@contextmanager
def _locked_ledger(run_id: str, *, exclusive: bool) -> Iterator[tuple[str, str]]:
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
        raise ValueError(f"invalid verified revision ledger: {path}")
    generation = raw.get("generation")
    revisions = raw.get("revisions")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError(f"invalid verified revision generation: {path}")
    if not isinstance(revisions, list):
        raise ValueError(f"invalid verified revision membership: {path}")
    for revision in revisions:
        parsed = parse_adapter_revision(revision) if isinstance(revision, str) else None
        if parsed is None or parsed[0] != run_id:
            raise ValueError(f"invalid verified adapter revision for run {run_id}: {revision!r}")
    return generation, revisions


def _fsync_directory(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_unlocked(runs_dir: str, path: str, generation: int, revisions: list[str]) -> None:
    fd, tmp = tempfile.mkstemp(dir=runs_dir, prefix=f"{os.path.basename(path)}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as ledger_file:
            json.dump(
                {"generation": generation, "revisions": revisions},
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


def read_verified_adapter_revisions(run_id: str) -> frozenset[str]:
    """Read the authoritative verified revision membership for a run."""
    with _locked_ledger(run_id, exclusive=False) as (_, path):
        _, revisions = _read_unlocked(path, run_id)
        return frozenset(revisions)


def verified_adapter_revision_generation(run_id: str) -> int:
    """Read the current undeploy generation for a run."""
    with _locked_ledger(run_id, exclusive=False) as (_, path):
        generation, _ = _read_unlocked(path, run_id)
        return generation


def commit_verified_adapter_revision(
    run_id: str,
    revision: str,
    *,
    expected_generation: int,
    commit: Callable[[], None],
    retain_only_revision: bool = False,
    advance_generation: bool = False,
) -> bool:
    """Persist verified membership before committing its ready status."""
    parsed = parse_adapter_revision(revision)
    if parsed is None or parsed[0] != run_id:
        raise ValueError(f"invalid verified adapter revision for run {run_id}: {revision!r}")
    if (
        isinstance(expected_generation, bool)
        or not isinstance(expected_generation, int)
        or expected_generation < 0
    ):
        raise ValueError(f"invalid verified revision generation: {expected_generation!r}")
    with _locked_ledger(run_id, exclusive=True) as (runs_dir, path):
        generation, revisions = _read_unlocked(path, run_id)
        if generation != expected_generation:
            return False
        retained_generation = generation + 1 if advance_generation else generation
        retained_revisions = [revision] if retain_only_revision else list(revisions)
        if not retain_only_revision and revision not in retained_revisions:
            retained_revisions.append(revision)
        if retained_generation != generation or retained_revisions != revisions:
            _write_unlocked(runs_dir, path, retained_generation, retained_revisions)
        commit()
        return True


def add_verified_adapter_revision(
    run_id: str,
    revision: str,
    *,
    expected_generation: int,
) -> bool:
    """Atomically add one canonical same-run immutable revision."""
    return commit_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=expected_generation,
        commit=lambda: None,
    )


def invalidate_verified_adapter_revisions(
    run_id: str,
    *,
    commit: Callable[[], None],
) -> int:
    """Clear membership, advance the tombstone generation, then commit undeploy status."""
    with _locked_ledger(run_id, exclusive=True) as (runs_dir, path):
        generation, _ = _read_unlocked(path, run_id)
        generation += 1
        _write_unlocked(runs_dir, path, generation, [])
        commit()
        return generation


def clear_verified_adapter_revisions(run_id: str) -> int:
    """Atomically clear membership and advance the undeploy generation."""
    return invalidate_verified_adapter_revisions(run_id, commit=lambda: None)

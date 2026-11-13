"""Submission IDs that outlive one `flash train` process, so a rerun attaches instead of duplicating.

A paid submit whose outcome is uncertain (it timed out, lost its connection, or the process died
mid-request) may still create its run server-side. Retries inside one process reuse one ID, but a
user who reruns the same command used to get a fresh ID, and the plane then created a second paid
run. So the ID is recorded here before the first send, keyed by the request it was bound to, and a
later submit of the identical request reuses it: the plane returns the run it already created.

A record is dropped as soon as the plane answers definitively (a run, or a 4xx rejection), so a
deliberate resubmit of an identical config after a confirmed run gets a fresh ID and a new run.
Each record names the process holding it; a concurrent identical submit (a user launching repeats
in parallel) never borrows an ID still held by a live process, so it cannot collapse into one run.

The key mirrors the plane's binding: the body minus ``runtime_secrets`` (secrets rotate without
changing the job), scoped to the plane URL and a fingerprint of the API key, since IDs are scoped
per key server-side. This file is a convenience: any error reading or writing it degrades to a
fresh, unrecorded ID and never fails the submit.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import socket
import time
from pathlib import Path

from flash._internal.fileio import read_json_or_empty, secure_json_write
from flash._internal.paths import data_dir

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX machines skip cross-process locking
    fcntl = None

# A record older than this is never reused: past it, a rerun is a new submission, not a retry.
PENDING_TTL_SECONDS = 24 * 3600.0


def new_submission_id() -> str:
    """A fresh idempotency key (32 URL-safe characters, inside the plane's 16-128 limit)."""
    return secrets.token_urlsafe(24)


def _path() -> Path:
    return data_dir() / "submissions" / "pending.json"


def request_key(client, body: dict) -> str:
    """Hash of what the plane binds an ID to, scoped to this plane and API key."""
    api_key = getattr(client, "api_key", None) or ""
    identity = {
        "api_url": getattr(client, "api_url", None),
        "key": hashlib.sha256(api_key.encode()).hexdigest(),
        "body": {name: value for name, value in body.items() if name != "runtime_secrets"},
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


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


def _valid(entry: object, now: float) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("submission_id"), str):
        return False
    created = entry.get("created_at")
    return isinstance(created, (int, float)) and 0 <= now - created < PENDING_TTL_SECONDS


def _load(path: Path, now: float) -> dict[str, list[dict]]:
    """Unexpired, well-formed records; a corrupt file reads as empty."""
    raw = read_json_or_empty(path).get("pending")
    if not isinstance(raw, dict):
        return {}
    records: dict[str, list[dict]] = {}
    for key, entries in raw.items():
        if isinstance(entries, list):
            kept = [entry for entry in entries if _valid(entry, now)]
            if kept:
                records[key] = kept
    return records


def _held_by_live_process(entry: dict) -> bool:
    pid = entry.get("pid")
    if not isinstance(pid, int) or pid == os.getpid():
        return False
    if entry.get("host") != socket.gethostname():
        return True  # cannot probe another machine's process; leave its claim alone
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # exists but owned by someone else
    return True


def _update(mutate) -> object:
    path = _path()
    now = time.time()
    with _locked(path):
        records = _load(path, now)
        result = mutate(records, now)
        secure_json_write(path, {"pending": records})
    return result


def claim(key: str) -> tuple[str, bool]:
    """Return (submission_id, recorded): reuse an abandoned ID for this request or record a new one.

    ``recorded`` is False only when the record could not be written, in which case a rerun will
    not find this ID and the caller must not promise that it will.
    """

    def mutate(records: dict[str, list[dict]], now: float) -> str:
        owner = {"pid": os.getpid(), "host": socket.gethostname()}
        entries = records.setdefault(key, [])
        for entry in entries:
            if not _held_by_live_process(entry):
                entry.update(owner)
                return entry["submission_id"]
        submission_id = new_submission_id()
        entries.append({"submission_id": submission_id, "created_at": now, **owner})
        return submission_id

    try:
        return str(_update(mutate)), True
    except Exception:
        return new_submission_id(), False


def _edit_entry(key: str, submission_id: str, *, drop: bool) -> None:
    def mutate(records: dict[str, list[dict]], _now: float) -> None:
        entries = records.get(key, [])
        for entry in list(entries):
            if entry["submission_id"] == submission_id:
                if drop:
                    entries.remove(entry)
                else:
                    entry["pid"] = entry["host"] = None
        if not entries:
            records.pop(key, None)

    with contextlib.suppress(Exception):
        _update(mutate)


def forget(key: str, submission_id: str) -> None:
    """The plane answered definitively: an identical submit later is a new run, with a new ID."""
    _edit_entry(key, submission_id, drop=True)


def release(key: str, submission_id: str) -> None:
    """The outcome is uncertain: keep the ID for the next identical submit to attach to."""
    _edit_entry(key, submission_id, drop=False)

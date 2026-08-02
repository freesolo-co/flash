"""SQLite store for the managed control plane: API keys + run ownership."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  key_hash     TEXT NOT NULL UNIQUE,
  key_prefix   TEXT NOT NULL,
  email        TEXT,
  created_at   REAL NOT NULL,
  last_used_at REAL,
  disabled     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
  run_id     TEXT PRIMARY KEY,
  key_id     INTEGER NOT NULL REFERENCES api_keys(id),
  kind       TEXT NOT NULL DEFAULT 'train',
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_key_idx ON runs(key_id);
"""


# Tests override with monkeypatch.setattr(db, "DB_PATH", tmp).
DB_PATH = str(Path.home() / ".flash" / "server.db")
_LAST_USED_WRITE_INTERVAL_S = 60.0
_INITIALIZATION_LOCK = threading.Lock()
_INITIALIZED_DATABASES: dict[tuple[int, str], tuple[int, int]] = {}
_CONNECTIONS = threading.local()


def db_path() -> str:
    return DB_PATH


def _database_file_identity(path: str) -> tuple[int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_dev, stat.st_ino


def _initialize_database(path: str) -> None:
    database = (os.getpid(), path)
    identity = _database_file_identity(path)
    if identity is not None and _INITIALIZED_DATABASES.get(database) == identity:
        return
    _INITIALIZATION_LOCK.acquire()
    try:
        identity = _database_file_identity(path)
        if identity is not None and _INITIALIZED_DATABASES.get(database) == identity:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 30.0
        backoff = 0.01
        schema_identity = None
        while True:
            conn = None
            retry_delay = None
            try:
                remaining = max(deadline - time.monotonic(), 0.0)
                conn = sqlite3.connect(path, timeout=remaining)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
                # Record the identity of the file the schema actually ran on while
                # the connection is still open, so a file replaced at the same path
                # afterwards can't be cached as initialized without _SCHEMA on it.
                schema_identity = _database_file_identity(path)
            except sqlite3.OperationalError as exc:
                error_code = getattr(exc, "sqlite_errorcode", None)
                primary_error_code = error_code & 0xFF if isinstance(error_code, int) else None
                lock_message = "locked" in str(exc).lower() or "busy" in str(exc).lower()
                if (
                    primary_error_code not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
                    and not lock_message
                ):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                retry_delay = min(backoff, remaining)
            finally:
                if conn is not None:
                    conn.close()
            if retry_delay is None:
                break
            # Don't hold the initialization lock while backing off: another process
            # may finish creating the schema, and sibling threads shouldn't block on
            # in-process init for the whole deadline.
            _INITIALIZATION_LOCK.release()
            try:
                time.sleep(retry_delay)
            finally:
                _INITIALIZATION_LOCK.acquire()
            backoff = min(backoff * 2, 0.25)
            identity = _database_file_identity(path)
            if identity is not None and _INITIALIZED_DATABASES.get(database) == identity:
                return
        if schema_identity is None:
            _INITIALIZED_DATABASES.pop(database, None)
            raise sqlite3.OperationalError("database disappeared during schema initialization")
        _INITIALIZED_DATABASES[database] = schema_identity
    finally:
        _INITIALIZATION_LOCK.release()


def _connect() -> sqlite3.Connection:
    path = db_path()
    process_id = os.getpid()
    _initialize_database(path)
    file_identity = _database_file_identity(path)
    if file_identity is None:
        raise sqlite3.OperationalError("database disappeared after schema initialization")

    conn = getattr(_CONNECTIONS, "connection", None)
    if (
        conn is None
        or getattr(_CONNECTIONS, "path", None) != path
        or getattr(_CONNECTIONS, "process_id", None) != process_id
        or getattr(_CONNECTIONS, "file_identity", None) != file_identity
    ):
        # Clear the pooled handle before reopening so a failed reconnect never
        # leaves a stale/closed connection behind for this thread to reuse.
        _CONNECTIONS.connection = None
        _CONNECTIONS.path = None
        _CONNECTIONS.process_id = None
        _CONNECTIONS.file_identity = None
        if conn is not None:
            conn.close()
        conn = sqlite3.connect(path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        opened_identity = _database_file_identity(path)
        if opened_identity is None or opened_identity != file_identity:
            # The file was removed or replaced between the identity check and the
            # open, so this connection may point at a ghost inode. Discard it and
            # let the caller retry rather than caching a mismatched identity pair.
            conn.close()
            raise sqlite3.OperationalError("database changed while opening a pooled connection")
        _CONNECTIONS.connection = conn
        _CONNECTIONS.path = path
        _CONNECTIONS.process_id = process_id
        _CONNECTIONS.file_identity = opened_identity
    return conn


def hash_key(api_key: str) -> str:
    # High-entropy machine tokens — unsalted SHA-256 is fine; CodeQL password-hashing rule doesn't apply.
    return hashlib.sha256(api_key.encode()).hexdigest()


def ensure_internal_key(api_key: str) -> dict:
    """Provision a row for the shared freesolo internal/service key (idempotent)."""
    now = time.time()
    internal_email = "internal@freesolo.co"
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO api_keys (key_hash, key_prefix, email, created_at) "
            "VALUES (?, ?, ?, ?)",
            (hash_key(api_key), "internal", internal_email, now),
        )
        conn.execute(
            "UPDATE api_keys SET email = ? WHERE key_hash = ? AND key_prefix = ?",
            (internal_email, hash_key(api_key), "internal"),
        )
    row = lookup_key(api_key)
    if row is None:  # pragma: no cover - the row was just inserted
        raise RuntimeError("failed to provision the internal service key")
    return row


# The standalone plane's owner row is keyed on this sentinel instead of a key hash, so its `id`
# -- which every `runs.key_id` points at -- does NOT change when the operator rotates
# FREESOLO_INTERNAL_KEY. A sha256 digest is 64 hex chars, so this can never collide with one.
_STANDALONE_OWNER_HASH = "standalone-operator"


def ensure_standalone_owner() -> dict:
    """The single owner row for a standalone plane, independent of the operator key's VALUE.

    Standalone is single-tenant: the operator key is the only credential, so it owns every run.
    Deriving the owner row from the key's hash meant rotating that key (or regenerating it after
    a restart) minted a NEW row with a new id, and every existing run -- matched by
    ``runs.key_id`` -- became invisible: absent from the listing, 404 on status, logs, cancel.
    An in-flight job would keep burning GPU hours with no supported way for the new credential to
    stop it, which makes rotating a COMPROMISED key the thing that costs you control of the plane.

    Not reachable by presenting a token: nothing hashes to the sentinel, so this row is only ever
    returned to a caller who already matched the operator key in ``authenticate``.

    Runs already in the store are ADOPTED, not orphaned. A plane that ran managed first -- or ran
    on an earlier build of this code -- has runs pointing at whichever key row created them, and
    leaving them there would reproduce the bug this row exists to fix the moment standalone is
    switched on: empty listing, 404 on status/logs/cancel, an in-flight job still spending. Adopting
    is sound precisely because standalone is SINGLE-TENANT: there is exactly one principal, so
    every run in this store is already the operator's, and there is no second identity that
    reassignment could take a run away from.

    Called on EVERY authenticated request, so in the steady state this function must issue NO
    write statements at all. SQLite has a single write slot and takes it for any write regardless
    of how many rows the statement touches, so a no-op write still serializes concurrent
    status/logs/submit behind whichever request happens to be recording a run -- up to the
    connection's 30s timeout. Measured under WAL, both of the writes below block a second writer
    exactly as hard as a real INSERT does when they change nothing, so both are guarded by a read:

    - The `INSERT OR IGNORE` is read-guarded rather than fired blind. It is NOT a cost problem
      (the unique-index probe is cheaper per call than the SELECT that replaces it) -- it is
      purely that it takes the write slot. Kept as OR IGNORE in the miss path because two threads
      can both miss the SELECT and race to insert; OR IGNORE makes the loser a no-op instead of
      an IntegrityError, and the re-read then returns the winner's row.
    - The adoption `UPDATE` cannot use `runs_key_idx` (`WHERE key_id != ?` plans as `SCAN runs`),
      so unguarded it would also full-scan the whole run history every request. The `< or >`
      split is a MULTI-INDEX OR over the covering index that stops at the first foreign row: at
      50k runs, 1.22 ms/call becomes 0.003 ms/call. Standalone mints no new foreign rows --
      `record_run` takes the key_id of the authenticated identity, which is this row -- so the
      guard cannot miss a run that appears later.
    """
    now = time.time()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (_STANDALONE_OWNER_HASH,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT OR IGNORE INTO api_keys (key_hash, key_prefix, email, created_at) "
                "VALUES (?, ?, ?, ?)",
                (_STANDALONE_OWNER_HASH, "standalone", "operator@localhost", now),
            )
            row = conn.execute(
                "SELECT * FROM api_keys WHERE key_hash = ?", (_STANDALONE_OWNER_HASH,)
            ).fetchone()
        if row is None:  # pragma: no cover - the row was just inserted
            raise RuntimeError("failed to provision the standalone owner row")
        unowned = conn.execute(
            "SELECT 1 FROM runs WHERE key_id < ? OR key_id > ? LIMIT 1", (row["id"], row["id"])
        ).fetchone()
        if unowned is not None:
            conn.execute("UPDATE runs SET key_id = ? WHERE key_id != ?", (row["id"], row["id"]))
    return dict(row)


def ensure_external_key(
    api_key: str, *, key_prefix: str | None = None, email: str | None = None
) -> dict | None:
    """Provision a per-token row for a verified external (freesolo USER) key (idempotent).

    Returns None if the key exists but is disabled (revoked keys stay rejected, not revived)."""
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO api_keys (key_hash, key_prefix, email, created_at) "
            "VALUES (?, ?, ?, ?)",
            (hash_key(api_key), key_prefix or "freesolo", email, now),
        )
    return lookup_key(api_key)


def lookup_key(api_key: str) -> dict | None:
    """Resolve a presented key to its row (and touch last_used_at); None if unknown/disabled."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ? AND disabled = 0",
            (hash_key(api_key),),
        ).fetchone()
        if row is None:
            return None
        now = time.time()
        stale_before = now - _LAST_USED_WRITE_INTERVAL_S
        if row["last_used_at"] is None or row["last_used_at"] <= stale_before:
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? "
                "WHERE id = ? AND (last_used_at IS NULL OR last_used_at <= ?)",
                (now, row["id"], stale_before),
            )
        return dict(row)


def record_run(run_id: str, key_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, key_id, kind, created_at) VALUES (?, ?, ?, ?)",
            (run_id, key_id, "train", time.time()),
        )


def delete_run(run_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


def run_owner(run_id: str) -> int | None:
    with _connect() as conn:
        row = conn.execute("SELECT key_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row["key_id"] if row else None


def runs_for_key(key_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT run_id, kind, created_at FROM runs WHERE key_id = ? ORDER BY created_at",
            (key_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def all_runs() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT run_id, key_id, kind, created_at FROM runs").fetchall()
        return [dict(r) for r in rows]

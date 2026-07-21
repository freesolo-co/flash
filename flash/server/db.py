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
CREATE TABLE IF NOT EXISTS warm_endpoints (
  endpoint_id        TEXT PRIMARY KEY,
  name               TEXT NOT NULL,
  owning_fingerprint TEXT NOT NULL,
  owner_key_id       INTEGER,
  owner_org_id       TEXT NOT NULL DEFAULT '',
  compat_sig         TEXT NOT NULL,
  gpu_type           TEXT NOT NULL,
  expiry_ts          REAL NOT NULL,
  claimed_by         TEXT,
  created_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS warm_endpoints_reuse_idx
  ON warm_endpoints(owner_key_id, owner_org_id, compat_sig);
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


# --- keep-warm endpoint registry -------------------------------------------------------------
#
# A finished run whose spec set ``gpu.keep_alive_seconds > 0`` leaves its provider endpoint alive
# and records it here so a COMPATIBLE, SAME-OWNER next run can reuse it (skipping the cold start)
# instead of provisioning a fresh one. Reuse is fail-closed and owner-scoped: acquire matches the
# exact (owner_key_id, owner_org_id) security domain AND the compat signature. A record older than
# ``expiry_ts`` is dead: the reaper deletes the endpoint and drops the row. Reuse is best-effort --
# any miss, race, or unhealthy endpoint just falls back to a fresh deploy, which is always correct.


def register_warm_endpoint(
    *,
    endpoint_id: str,
    name: str,
    owning_fingerprint: str,
    owner_key_id: int | None,
    owner_org_id: str,
    compat_sig: str,
    gpu_type: str,
    expiry_ts: float,
) -> None:
    """Record an endpoint kept warm for reuse. Idempotent per endpoint: a run's teardown may fire
    more than once (completion + every server-restart recovery sweep), and re-inserting must NOT
    extend the keep-alive window (that would be runaway idle billing). A genuinely-reused endpoint
    is consumed by ``acquire_warm_endpoint`` first, so its next completion re-registers a fresh
    window cleanly."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO warm_endpoints (endpoint_id, name, owning_fingerprint, "
            "owner_key_id, owner_org_id, compat_sig, gpu_type, expiry_ts, claimed_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
            (
                endpoint_id,
                name,
                owning_fingerprint,
                owner_key_id,
                owner_org_id or "",
                compat_sig,
                gpu_type,
                float(expiry_ts),
                time.time(),
            ),
        )


def acquire_warm_endpoint(
    *,
    owner_key_id: int | None,
    owner_org_id: str,
    compat_sig: str,
    now: float,
) -> dict | None:
    """Atomically claim (by consuming) the freshest unexpired endpoint for this exact owner+signature.

    Owner scoping is the security boundary: a warm worker retains the prior run's fetched
    environment code and on-disk secrets, so it is reused ONLY within the identical
    (owner_key_id, owner_org_id) domain. The row is DELETED on claim: the reusing run takes the
    endpoint out of the pool (while it runs, the endpoint reads as busy so the idle reaper leaves it
    alone), and re-registers a fresh keep-alive window when it finishes. Returns the claimed record,
    or None on any miss/race (the caller then deploys a fresh endpoint, which is always safe).
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM warm_endpoints WHERE expiry_ts > ? "
            "AND compat_sig = ? AND owner_key_id IS ? AND owner_org_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (now, compat_sig, owner_key_id, owner_org_id or ""),
        ).fetchone()
        if row is None:
            return None
        claimed = conn.execute(
            "DELETE FROM warm_endpoints WHERE endpoint_id = ?", (row["endpoint_id"],)
        )
        if claimed.rowcount != 1:
            return None
        return dict(row)


def release_warm_endpoint(endpoint_id: str) -> None:
    """Drop a warm record (endpoint reused, torn down, or reaped)."""
    with _connect() as conn:
        conn.execute("DELETE FROM warm_endpoints WHERE endpoint_id = ?", (endpoint_id,))


def unexpired_warm_endpoint_names(now: float) -> set[str]:
    """Endpoint names the reaper must PROTECT: intentionally warm and not yet expired."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name FROM warm_endpoints WHERE expiry_ts > ?", (now,)
        ).fetchall()
        return {r["name"] for r in rows}


def unexpired_warm_endpoint_ids(now: float) -> set[str]:
    """Endpoint IDs the idle reaper must PROTECT (id-based, robust to a reused endpoint whose name
    was minted for an earlier run)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT endpoint_id FROM warm_endpoints WHERE expiry_ts > ?", (now,)
        ).fetchall()
        return {r["endpoint_id"] for r in rows}


def expired_warm_endpoints(now: float) -> list[dict]:
    """Records past their keep-alive window: the reaper tears these down and drops the rows."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM warm_endpoints WHERE expiry_ts <= ?", (now,)
        ).fetchall()
        return [dict(r) for r in rows]


def all_warm_endpoints() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM warm_endpoints ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

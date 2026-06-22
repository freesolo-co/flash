"""SQLite store for the managed control plane: API keys + run ownership.

Run *state* stays in the runner's JSON files (``runner.RUNS_DIR``) — the
battle-tested submit/attach/cancel paths all read those. This database is only the
key registry and the run -> key ownership index that makes the server multi-tenant.

Connections are opened per operation (cheap for SQLite, avoids cross-thread state;
the runner runs jobs in daemon threads inside the same process).
"""

from __future__ import annotations

import hashlib
import sqlite3
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


# Fixed location for the keys/run-ownership SQLite DB (not operator-configurable). Tests
# point it elsewhere with monkeypatch.setattr(db, "DB_PATH", tmp).
DB_PATH = str(Path.home() / ".flash" / "server.db")


def db_path() -> str:
    return DB_PATH


def _connect() -> sqlite3.Connection:
    path = db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def hash_key(api_key: str) -> str:
    # API keys are 192-bit random tokens (secrets.token_hex(24)), not passwords:
    # brute-forcing the keyspace is infeasible, so an unsalted fast hash is the
    # standard at-rest form and keeps O(1) lookup by hash. (CodeQL's
    # password-hashing rule does not apply to high-entropy machine tokens.)
    return hashlib.sha256(api_key.encode()).hexdigest()


def ensure_internal_key(api_key: str) -> dict:
    """Provision a row for the shared freesolo internal/service key (idempotent).

    The freesolo platform/SDK authenticate to the control plane with the same
    ``FREESOLO_INTERNAL_KEY`` they already hold. Backing it with a real row
    (inserted once, by hash) means run ownership and
    the runs.key_id foreign key work exactly as for a normal key — all
    internal-key runs share this single service identity (no per-user isolation;
    the platform scopes users upstream)."""
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO api_keys (key_hash, key_prefix, email, created_at) "
            "VALUES (?, ?, ?, ?)",
            (hash_key(api_key), "internal", "freesolo-internal", now),
        )
    row = lookup_key(api_key)
    if row is None:  # pragma: no cover - the row was just inserted
        raise RuntimeError("failed to provision the internal service key")
    return row


def ensure_external_key(
    api_key: str, *, key_prefix: str | None = None, email: str | None = None
) -> dict | None:
    """Provision a per-token row for a verified external (freesolo USER) key (idempotent).

    Unlike :func:`ensure_internal_key` (one shared service identity), this keys a distinct
    row by the presented token's hash, so each freesolo user key gets its OWN run-ownership
    identity (the runs.key_id foreign key then scopes runs per user). The full token is never
    stored — only its sha256.

    Returns ``None`` (not a row) when the token's row already exists but is DISABLED:
    ``INSERT OR IGNORE`` won't revive it and ``lookup_key`` filters disabled rows, so a
    revoked key is rejected (401) by the caller instead of surfacing as a 500."""
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
        conn.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (time.time(), row["id"]))
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

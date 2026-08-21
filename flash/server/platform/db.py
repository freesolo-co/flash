"""SQLite store for control-plane keys, run ownership, and teacher capabilities."""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from flash._internal.paths import data_dir
from flash.teacher.provider_status import validated_provider_status

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
CREATE TABLE IF NOT EXISTS teacher_capabilities (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  token_hash            TEXT NOT NULL UNIQUE,
  run_id                TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  attempt               INTEGER NOT NULL,
  teacher_alias         TEXT NOT NULL,
  provider              TEXT NOT NULL,
  model                 TEXT NOT NULL,
  scoring_mode          TEXT NOT NULL,
  expires_at            REAL NOT NULL,
  revoked_at            REAL,
  max_requests          INTEGER NOT NULL,
  max_score_items       INTEGER NOT NULL,
  max_request_bytes     INTEGER NOT NULL,
  max_response_bytes    INTEGER NOT NULL,
  max_concurrency       INTEGER NOT NULL,
  max_upstream_attempts INTEGER NOT NULL,
  max_request_tokens    INTEGER NOT NULL,
  max_total_tokens      INTEGER NOT NULL,
  request_count         INTEGER NOT NULL DEFAULT 0,
  score_item_count      INTEGER NOT NULL DEFAULT 0,
  token_count           INTEGER NOT NULL DEFAULT 0,
  in_flight             INTEGER NOT NULL DEFAULT 0,
  created_at            REAL NOT NULL,
  UNIQUE(run_id, attempt)
);
CREATE INDEX IF NOT EXISTS teacher_capabilities_run_idx
  ON teacher_capabilities(run_id, attempt);
CREATE TABLE IF NOT EXISTS teacher_score_requests (
  id                     INTEGER PRIMARY KEY AUTOINCREMENT,
  capability_id          INTEGER NOT NULL REFERENCES teacher_capabilities(id) ON DELETE CASCADE,
  request_id             TEXT NOT NULL,
  request_fingerprint    TEXT NOT NULL,
  request_bytes          INTEGER NOT NULL,
  score_items            INTEGER NOT NULL,
  state                  TEXT NOT NULL,
  upstream_attempt_count INTEGER NOT NULL DEFAULT 0,
  provider_status        INTEGER,
  error_class            TEXT,
  input_tokens           INTEGER,
  output_tokens          INTEGER,
  response_body          BLOB,
  created_at             REAL NOT NULL,
  updated_at             REAL NOT NULL,
  started_at             REAL,
  completed_at           REAL,
  UNIQUE(capability_id, request_id)
);
CREATE INDEX IF NOT EXISTS teacher_score_requests_state_idx
  ON teacher_score_requests(state, updated_at);
"""


# Tests override with monkeypatch.setattr(db, "DB_PATH", tmp).
DB_PATH = str(data_dir() / "server.db")

BUSY_TIMEOUT_ENV = "FLASH_SQLITE_BUSY_TIMEOUT_SECONDS"
_DEFAULT_BUSY_TIMEOUT_S = 30.0
_LAST_USED_WRITE_INTERVAL_S = 60.0
_INITIALIZATION_LOCK = threading.Lock()
_INITIALIZED_DATABASES: dict[tuple[int, str], tuple[int, int]] = {}
_CONNECTIONS = threading.local()


def db_path() -> str:
    return DB_PATH


def busy_timeout_s() -> float:
    """How long to wait for a lock held by another writer before giving up.

    Networked storage can hold a lock far longer than a local volume, and the only symptom is a
    "database is locked" error surfacing as a failed API call. This raises the patience, not the
    durability: journal mode, ``synchronous``, and foreign keys are correctness choices and are
    deliberately not configurable. A non-positive or unparseable value falls back to the default
    rather than disabling the wait, since a zero timeout turns ordinary contention into errors.
    """
    raw = os.environ.get(BUSY_TIMEOUT_ENV, "").strip()
    if not raw:
        return _DEFAULT_BUSY_TIMEOUT_S
    try:
        seconds = float(raw)
    except ValueError:
        return _DEFAULT_BUSY_TIMEOUT_S
    return seconds if seconds > 0 else _DEFAULT_BUSY_TIMEOUT_S


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
        deadline = time.monotonic() + busy_timeout_s()
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
                conn.commit()
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
        conn = sqlite3.connect(path, timeout=busy_timeout_s())
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


def record_run(run_id: str, key_id: int, *, kind: str = "train") -> None:
    if kind != "train":
        raise ValueError("run kind must be 'train'")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, key_id, kind, created_at) VALUES (?, ?, ?, ?)",
            (run_id, key_id, kind, time.time()),
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


class TeacherLedgerError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        retryable: bool = False,
        provider_status: int | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.provider_status = validated_provider_status(provider_status)


def _immediate(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")


def _teacher_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_teacher_capability(
    *,
    run_id: str,
    attempt: int,
    teacher_alias: str,
    provider: str,
    model: str,
    scoring_mode: str,
    expires_at: float,
    limits: dict[str, int],
    now: float | None = None,
) -> str:
    issued_at = time.time() if now is None else float(now)
    token = secrets.token_urlsafe(32)
    token_hash = _teacher_token_hash(token)
    conn = _connect()
    try:
        _immediate(conn)
        conn.execute(
            "UPDATE teacher_capabilities SET revoked_at = ? "
            "WHERE run_id = ? AND attempt != ? AND revoked_at IS NULL",
            (issued_at, run_id, attempt),
        )
        conn.execute(
            "INSERT INTO teacher_capabilities ("
            "token_hash, run_id, attempt, teacher_alias, provider, model, scoring_mode, "
            "expires_at, max_requests, max_score_items, max_request_bytes, max_response_bytes, "
            "max_concurrency, max_upstream_attempts, max_request_tokens, max_total_tokens, "
            "created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                token_hash,
                run_id,
                int(attempt),
                teacher_alias,
                provider,
                model,
                scoring_mode,
                float(expires_at),
                int(limits["max_requests"]),
                int(limits["max_score_items"]),
                int(limits["max_request_bytes"]),
                int(limits["max_response_bytes"]),
                int(limits["max_concurrency"]),
                int(limits["max_upstream_attempts"]),
                int(limits["max_request_tokens"]),
                int(limits["max_total_tokens"]),
                issued_at,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return token


def teacher_capability_binding(token: str) -> dict:
    token_hash = _teacher_token_hash(token)
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM teacher_capabilities WHERE token_hash = ?", (token_hash,)
        ).fetchone()
    if row is None:
        raise TeacherLedgerError("invalid_capability")
    return dict(row)


def active_teacher_capability(token: str, *, now: float | None = None) -> dict:
    capability = teacher_capability_binding(token)
    checked_at = time.time() if now is None else float(now)
    if capability["revoked_at"] is not None:
        raise TeacherLedgerError("revoked_capability")
    if checked_at >= capability["expires_at"]:
        raise TeacherLedgerError("expired_capability")
    return capability


def revoke_teacher_capability(token: str, *, now: float | None = None) -> bool:
    revoked_at = time.time() if now is None else float(now)
    conn = _connect()
    try:
        _immediate(conn)
        cursor = conn.execute(
            "UPDATE teacher_capabilities SET revoked_at = COALESCE(revoked_at, ?) "
            "WHERE token_hash = ?",
            (revoked_at, _teacher_token_hash(token)),
        )
        conn.commit()
        return cursor.rowcount > 0
    except Exception:
        conn.rollback()
        raise


def revoke_teacher_capabilities_for_run(run_id: str, *, now: float | None = None) -> int:
    revoked_at = time.time() if now is None else float(now)
    conn = _connect()
    try:
        _immediate(conn)
        cursor = conn.execute(
            "UPDATE teacher_capabilities SET revoked_at = ? "
            "WHERE run_id = ? AND revoked_at IS NULL",
            (revoked_at, run_id),
        )
        conn.commit()
        return cursor.rowcount
    except Exception:
        conn.rollback()
        raise


def _readmit_teacher_request(conn, capability, existing, *, admitted_at, charge_tokens) -> dict:
    """Re-open one existing ledger row as 'reserved' inside the caller's transaction.

    ``charge_tokens`` is True only for failures proven to precede provider dispatch, whose token
    reservation was released. A provider 429 keeps its existing reservation because the provider
    rejected it before execution and only in_flight must be re-acquired.
    """
    if existing["upstream_attempt_count"] >= capability["max_upstream_attempts"]:
        # the upstream budget is spent, so readmission is refused before any counter moves: a
        # readmitted-then-refused row would be bounced back through 'retryable' by the dispatch
        # path, releasing its held reservation and staying readmissible forever.
        raise TeacherLedgerError("upstream_attempt_quota_exhausted")
    if capability["in_flight"] >= capability["max_concurrency"]:
        raise TeacherLedgerError("broker_busy", retryable=True)
    token_delta = 0
    if charge_tokens:
        token_delta = existing["score_items"] * capability["max_request_tokens"]
        if capability["token_count"] + token_delta > capability["max_total_tokens"]:
            raise TeacherLedgerError("token_quota_exhausted")
    conn.execute(
        "UPDATE teacher_score_requests SET state = 'reserved', updated_at = ?, "
        "provider_status = NULL, error_class = NULL WHERE id = ?",
        (admitted_at, existing["id"]),
    )
    conn.execute(
        "UPDATE teacher_capabilities SET in_flight = in_flight + 1, "
        "token_count = token_count + ? WHERE id = ?",
        (token_delta, capability["id"]),
    )
    conn.commit()
    return {
        "capability": dict(capability),
        "request": {
            **dict(existing),
            "state": "reserved",
            "updated_at": admitted_at,
        },
    }


def _resume_existing_teacher_request(conn, capability, existing, *, admitted_at) -> dict:
    """Resolve one already-known request_id: readmit, replay, or refuse."""
    state = existing["state"]
    if state == "retryable":
        return _readmit_teacher_request(
            conn, capability, existing, admitted_at=admitted_at, charge_tokens=True
        )
    if state in {"reserved", "started"}:
        raise TeacherLedgerError("request_in_progress", retryable=True)
    # only a provider rejection proven to precede execution may dispatch again. currently that is a
    # conventional 429 recorded as transient. outcome_unknown is terminal because flash has no
    # upstream idempotency key and internal ledger accounting cannot prevent duplicate provider work.
    if state == "provider_rejected" and existing["error_class"] == "transient":
        return _readmit_teacher_request(
            conn, capability, existing, admitted_at=admitted_at, charge_tokens=False
        )
    if state == "succeeded":
        response_body = existing["response_body"]
        if (
            not isinstance(response_body, bytes)
            or not response_body
            or len(response_body) > capability["max_response_bytes"]
        ):
            raise TeacherLedgerError("replay_unavailable")
        conn.commit()
        return {
            "capability": dict(capability),
            "request": dict(existing),
            "response_body": response_body,
        }
    raise TeacherLedgerError(
        state,
        provider_status=existing["provider_status"] if state == "provider_rejected" else None,
    )


def reserve_teacher_request(
    *,
    token: str,
    request_id: str,
    request_fingerprint: str,
    request_bytes: int,
    score_items: int,
    expected_run_id: str,
    expected_attempt: int,
    now: float | None = None,
) -> dict:
    admitted_at = time.time() if now is None else float(now)
    conn = _connect()
    try:
        _immediate(conn)
        capability = conn.execute(
            "SELECT * FROM teacher_capabilities WHERE token_hash = ?",
            (_teacher_token_hash(token),),
        ).fetchone()
        if capability is None:
            raise TeacherLedgerError("invalid_capability")
        if capability["run_id"] != expected_run_id or capability["attempt"] != expected_attempt:
            raise TeacherLedgerError("capability_scope_mismatch")
        if capability["revoked_at"] is not None:
            raise TeacherLedgerError("revoked_capability")
        if admitted_at >= capability["expires_at"]:
            raise TeacherLedgerError("expired_capability")
        existing = conn.execute(
            "SELECT * FROM teacher_score_requests WHERE capability_id = ? AND request_id = ?",
            (capability["id"], request_id),
        ).fetchone()
        if existing is not None:
            if existing["request_fingerprint"] != request_fingerprint:
                raise TeacherLedgerError("request_body_changed")
            return _resume_existing_teacher_request(
                conn, capability, existing, admitted_at=admitted_at
            )
        if request_bytes > capability["max_request_bytes"]:
            raise TeacherLedgerError("request_too_large")
        request_token_limit = int(score_items) * capability["max_request_tokens"]
        if request_token_limit <= 0:
            raise TeacherLedgerError("invalid_score_items")
        if capability["token_count"] + request_token_limit > capability["max_total_tokens"]:
            raise TeacherLedgerError("token_quota_exhausted")
        if capability["request_count"] >= capability["max_requests"]:
            raise TeacherLedgerError("request_quota_exhausted")
        if capability["score_item_count"] + score_items > capability["max_score_items"]:
            raise TeacherLedgerError("score_item_quota_exhausted")
        if capability["in_flight"] >= capability["max_concurrency"]:
            raise TeacherLedgerError("broker_busy", retryable=True)
        cursor = conn.execute(
            "INSERT INTO teacher_score_requests ("
            "capability_id, request_id, request_fingerprint, request_bytes, score_items, state, "
            "created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, 'reserved', ?, ?)",
            (
                capability["id"],
                request_id,
                request_fingerprint,
                int(request_bytes),
                int(score_items),
                admitted_at,
                admitted_at,
            ),
        )
        conn.execute(
            "UPDATE teacher_capabilities SET request_count = request_count + 1, "
            "score_item_count = score_item_count + ?, in_flight = in_flight + 1, "
            "token_count = token_count + ? WHERE id = ?",
            (int(score_items), request_token_limit, capability["id"]),
        )
        conn.commit()
        request = conn.execute(
            "SELECT * FROM teacher_score_requests WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return {"capability": dict(capability), "request": dict(request)}
    except Exception:
        conn.rollback()
        raise


def mark_teacher_request_started(
    capability_id: int, request_id: str, *, now: float | None = None
) -> None:
    started_at = time.time() if now is None else float(now)
    conn = _connect()
    try:
        _immediate(conn)
        row = conn.execute(
            "SELECT r.state, r.upstream_attempt_count, c.max_upstream_attempts, c.revoked_at, "
            "c.expires_at FROM teacher_score_requests r JOIN teacher_capabilities c "
            "ON c.id = r.capability_id WHERE r.capability_id = ? AND r.request_id = ?",
            (capability_id, request_id),
        ).fetchone()
        if row is None or row["state"] != "reserved":
            raise TeacherLedgerError("request_not_reserved")
        if row["revoked_at"] is not None or started_at >= row["expires_at"]:
            raise TeacherLedgerError("capability_fenced")
        if row["upstream_attempt_count"] >= row["max_upstream_attempts"]:
            raise TeacherLedgerError("upstream_attempt_quota_exhausted")
        conn.execute(
            "UPDATE teacher_score_requests SET state = 'started', "
            "upstream_attempt_count = upstream_attempt_count + 1, started_at = ?, updated_at = ? "
            "WHERE capability_id = ? AND request_id = ?",
            (started_at, started_at, capability_id, request_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def complete_teacher_request(
    capability_id: int,
    request_id: str,
    *,
    state: str,
    provider_status: int | None = None,
    error_class: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    response_body: bytes | None = None,
    now: float | None = None,
) -> None:
    if state not in {
        "succeeded",
        "provider_rejected",
        "provider_contract_error",
        "outcome_unknown",
    }:
        raise ValueError(f"invalid teacher request terminal state: {state}")
    completed_at = time.time() if now is None else float(now)
    input_count = max(0, int(input_tokens or 0))
    output_count = max(0, int(output_tokens or 0))
    conn = _connect()
    try:
        _immediate(conn)
        row = conn.execute(
            "SELECT r.state, r.score_items, c.max_request_tokens, c.max_response_bytes "
            "FROM teacher_score_requests r JOIN teacher_capabilities c "
            "ON c.id = r.capability_id WHERE r.capability_id = ? AND r.request_id = ?",
            (capability_id, request_id),
        ).fetchone()
        if row is None or row["state"] not in {"reserved", "started"}:
            raise TeacherLedgerError("request_not_active")
        actual_tokens = input_count + output_count
        request_token_limit = row["score_items"] * row["max_request_tokens"]
        if state == "succeeded" and actual_tokens > request_token_limit:
            raise TeacherLedgerError("request_token_limit_exceeded")
        if state == "succeeded":
            if (
                not isinstance(response_body, bytes)
                or not response_body
                or len(response_body) > row["max_response_bytes"]
            ):
                raise TeacherLedgerError("invalid_replay_response")
        elif response_body is not None:
            raise ValueError("only succeeded teacher requests may persist a response")
        token_delta = actual_tokens - request_token_limit if state == "succeeded" else 0
        conn.execute(
            "UPDATE teacher_score_requests SET state = ?, provider_status = ?, error_class = ?, "
            "input_tokens = ?, output_tokens = ?, response_body = ?, completed_at = ?, "
            "updated_at = ? WHERE capability_id = ? AND request_id = ?",
            (
                state,
                provider_status,
                error_class,
                input_count,
                output_count,
                response_body,
                completed_at,
                completed_at,
                capability_id,
                request_id,
            ),
        )
        conn.execute(
            "UPDATE teacher_capabilities SET in_flight = MAX(0, in_flight - 1), "
            "token_count = MAX(0, token_count + ?) WHERE id = ?",
            (token_delta, capability_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def retry_teacher_request_before_dispatch(
    capability_id: int,
    request_id: str,
    *,
    error_class: str,
    now: float | None = None,
) -> None:
    updated_at = time.time() if now is None else float(now)
    conn = _connect()
    try:
        _immediate(conn)
        row = conn.execute(
            "SELECT r.state, r.score_items, c.max_request_tokens FROM teacher_score_requests r "
            "JOIN teacher_capabilities c ON c.id = r.capability_id "
            "WHERE r.capability_id = ? AND r.request_id = ?",
            (capability_id, request_id),
        ).fetchone()
        if row is None or row["state"] != "reserved":
            raise TeacherLedgerError("request_not_reserved")
        conn.execute(
            "UPDATE teacher_score_requests SET state = 'retryable', error_class = ?, updated_at = ? "
            "WHERE capability_id = ? AND request_id = ?",
            (error_class, updated_at, capability_id, request_id),
        )
        conn.execute(
            "UPDATE teacher_capabilities SET in_flight = MAX(0, in_flight - 1), "
            "token_count = MAX(0, token_count - ?) WHERE id = ?",
            (row["score_items"] * row["max_request_tokens"], capability_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def recover_teacher_request_ledger(*, now: float | None = None) -> dict[str, int]:
    recovered_at = time.time() if now is None else float(now)
    conn = _connect()
    try:
        _immediate(conn)
        conn.execute(
            "UPDATE teacher_capabilities SET token_count = MAX(0, token_count - "
            "max_request_tokens * COALESCE((SELECT SUM(r.score_items) "
            "FROM teacher_score_requests r WHERE r.capability_id = teacher_capabilities.id "
            "AND r.state = 'reserved'), 0))"
        )
        reserved = conn.execute(
            "UPDATE teacher_score_requests SET state = 'retryable', "
            "error_class = 'broker_restart_before_dispatch', updated_at = ? "
            "WHERE state = 'reserved'",
            (recovered_at,),
        ).rowcount
        started = conn.execute(
            "UPDATE teacher_score_requests SET state = 'outcome_unknown', "
            "error_class = 'broker_restart_after_dispatch', completed_at = ?, updated_at = ? "
            "WHERE state = 'started'",
            (recovered_at, recovered_at),
        ).rowcount
        conn.execute("UPDATE teacher_capabilities SET in_flight = 0")
        conn.commit()
        return {"retryable": reserved, "outcome_unknown": started}
    except Exception:
        conn.rollback()
        raise

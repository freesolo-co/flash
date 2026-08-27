"""Control-plane SQLite schema.

The DDL lives apart from the connection and query helpers in `db` so that adding a table
does not push the store module past the file-size gate.
"""

from __future__ import annotations

SCHEMA = """
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
CREATE TABLE IF NOT EXISTS run_submission_idempotency (
  key_id                  INTEGER NOT NULL,
  idempotency_key         TEXT NOT NULL,
  run_id                  TEXT NOT NULL UNIQUE,
  request_fingerprint     TEXT NOT NULL,
  phase                   TEXT NOT NULL CHECK (phase IN ('claimed', 'bound', 'disposed')),
  dry_run                 INTEGER NOT NULL,
  had_runtime_secrets     INTEGER NOT NULL,
  submitted_instance_providers TEXT NOT NULL,
  affordability_verified  INTEGER,
  disposed_reason         TEXT,
  created_at              REAL NOT NULL,
  updated_at              REAL NOT NULL,
  PRIMARY KEY (key_id, idempotency_key)
);
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

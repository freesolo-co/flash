"""Transactional admission, tenant round-robin scheduling, and execution attribution."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager

from flash.training.shared.accounting import estimate
from flash.training.shared.contracts import (
    AdmissionError,
    ConflictError,
    Limits,
    command_tokens,
    session_config,
)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Journal:
    """One durable pool journal; admission is atomic across API callers."""

    def __init__(
        self, path, *, limits=None, vocab_size=128, clock=time.time_ns, rate_microusd_per_hour=None
    ):
        self.limits = limits or Limits()
        self.vocab_size = vocab_size
        self.clock = clock
        if rate_microusd_per_hour is not None and (
            type(rate_microusd_per_hour) is not int or rate_microusd_per_hour < 0
        ):
            raise ValueError("resource rate must be nonnegative integer micro-USD/hour")
        self.rate = rate_microusd_per_hour
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS tenants (id TEXT PRIMARY KEY, turn INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, tenant TEXT NOT NULL, owner TEXT NOT NULL,
                config TEXT NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS commands (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                session TEXT NOT NULL REFERENCES sessions(id), tenant TEXT NOT NULL,
                request_key TEXT NOT NULL, fingerprint TEXT NOT NULL, operation TEXT NOT NULL,
                payload TEXT NOT NULL, tokens INTEGER NOT NULL, state TEXT NOT NULL,
                accepted_ns INTEGER NOT NULL, started_ns INTEGER, result TEXT,
                UNIQUE(session, request_key));
            CREATE TABLE IF NOT EXISTS usage (
                command TEXT PRIMARY KEY REFERENCES commands(id), tenant TEXT NOT NULL,
                execution_ns INTEGER, queue_ns INTEGER NOT NULL, outcome TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS ready_commands ON commands(state, tenant, seq);
            CREATE TABLE IF NOT EXISTS overhead (id INTEGER PRIMARY KEY, reason TEXT NOT NULL,
                elapsed_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS pricing (id INTEGER PRIMARY KEY CHECK(id=1), rate TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS residency (
                session TEXT PRIMARY KEY REFERENCES sessions(id), loaded_ns INTEGER NOT NULL,
                released_ns INTEGER, uncertain INTEGER NOT NULL DEFAULT 0);
        """)
        with self.transaction() as db:
            encoded_rate = encode(self.rate)
            db.execute("INSERT OR IGNORE INTO pricing VALUES (1,?)", (encoded_rate,))
            if db.execute("SELECT rate FROM pricing WHERE id=1").fetchone()[0] != encoded_rate:
                raise ValueError(
                    "resource rate differs from this journal's immutable pricing basis"
                )

    @contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
            except BaseException:
                self.db.rollback()
                raise
            else:
                self.db.commit()

    def _owned(self, db, principal, session):
        row = db.execute(
            "SELECT * FROM sessions WHERE id=? AND tenant=? AND owner=?",
            (session, principal.tenant, principal.owner),
        ).fetchone()
        if row is None:
            raise KeyError("unknown shared session")
        return row

    def create(self, principal, key, config):
        if not isinstance(key, str) or not 1 <= len(key) <= 128:
            raise ValueError("creation key must be 1..128 characters")
        config = session_config(config)
        # identity-derived handles make creation retries idempotent without trusting caller ids.
        session = hashlib.sha256(
            encode([principal.tenant, principal.owner, key]).encode()
        ).hexdigest()
        with self.transaction() as db:
            existing = db.execute("SELECT * FROM sessions WHERE id=?", (session,)).fetchone()
            if existing is not None:
                if existing["config"] != encode(config):
                    raise ConflictError("session creation key already used with different config")
                return session
            active = db.execute(
                "SELECT tenant FROM sessions WHERE state IN ('opening','active')"
            ).fetchall()
            if len(active) >= self.limits.adapters:
                raise AdmissionError("shared pool adapter capacity reached")
            if (
                sum(row["tenant"] == principal.tenant for row in active)
                >= self.limits.adapters_per_tenant
            ):
                raise AdmissionError("tenant adapter capacity reached")
            self._capacity(db, principal.tenant, 0)
            db.execute("INSERT OR IGNORE INTO tenants VALUES (?,0)", (principal.tenant,))
            db.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?)",
                (session, principal.tenant, principal.owner, encode(config), "opening"),
            )
            self._insert(db, session, principal.tenant, "__create__", "create", config, 0)
        return session

    def _capacity(self, db, tenant, tokens):
        rows = db.execute(
            "SELECT tenant,tokens FROM commands WHERE state IN ('queued','running')"
        ).fetchall()
        own = [row for row in rows if row["tenant"] == tenant]
        if len(rows) >= self.limits.queued_requests:
            raise AdmissionError("pool outstanding request capacity reached")
        if len(own) >= self.limits.queued_requests_per_tenant:
            raise AdmissionError("tenant outstanding request capacity reached")
        if sum(row["tokens"] for row in own) + tokens > self.limits.queued_tokens_per_tenant:
            raise AdmissionError("tenant outstanding token capacity reached")

    def _insert(self, db, session, tenant, key, operation, payload, tokens):
        command = uuid.uuid4().hex
        fingerprint = hashlib.sha256(encode([operation, payload]).encode()).hexdigest()
        db.execute(
            "INSERT INTO commands (id,session,tenant,request_key,fingerprint,operation,payload,"
            "tokens,state,accepted_ns) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                command,
                session,
                tenant,
                key,
                fingerprint,
                operation,
                encode(payload),
                tokens,
                "queued",
                self.clock(),
            ),
        )
        return command

    def submit(self, principal, session, key, operation, payload):
        if not isinstance(key, str) or not 1 <= len(key) <= 128 or key.startswith("__"):
            raise ValueError("request key must be 1..128 characters and not reserved")
        with self.transaction() as db:
            row = self._owned(db, principal, session)
            fingerprint = hashlib.sha256(encode([operation, payload]).encode()).hexdigest()
            previous = db.execute(
                "SELECT * FROM commands WHERE session=? AND request_key=?", (session, key)
            ).fetchone()
            if previous is not None:
                if previous["fingerprint"] != fingerprint:
                    raise ConflictError("request key already used with different work")
                return previous["id"]
            if row["state"] not in ("opening", "active"):
                raise ConflictError("session is no longer active")
            if db.execute(
                "SELECT 1 FROM commands WHERE session=? AND operation='close'", (session,)
            ).fetchone():
                raise ConflictError("session is closing")
            if operation == "restore":
                payload = self._restore_payload(db, principal, row, payload)
                tokens = 0
            else:
                algorithm = json.loads(row["config"])["algorithm"]
                tokens = command_tokens(
                    operation,
                    payload,
                    algorithm,
                    self.limits,
                    self.vocab_size,
                )
                if operation == "train" and algorithm == "grpo":
                    payload = {
                        **payload,
                        "normalizer": payload.get("normalizer", self.limits.context_tokens),
                    }
            # cleanup must remain possible when a tenant has exhausted its work allowance.
            if operation != "close":
                self._capacity(db, principal.tenant, tokens)
            command = self._insert(db, session, principal.tenant, key, operation, payload, tokens)
            db.execute("UPDATE commands SET fingerprint=? WHERE id=?", (fingerprint, command))
            return command

    def _restore_payload(self, db, principal, target, payload):
        from flash.training.shared.contracts import fields, integer

        fields(payload, ("source_session", "version"))
        if not isinstance(payload["source_session"], str):
            raise ValueError("source_session must be a session id")
        source = self._owned(db, principal, payload["source_session"])
        version = integer(payload["version"], "version", 0, 2**53)
        if source["config"] != target["config"]:
            raise ConflictError("restored session must use the original configuration")
        results = db.execute(
            "SELECT result FROM commands WHERE session=? AND operation='checkpoint' AND state='completed'",
            (source["id"],),
        ).fetchall()
        if not any(json.loads(r["result"])["version"] == version for r in results):
            raise KeyError("unknown completed checkpoint")
        return {"checkpoint": f"{source['id']}/{version}"}

    def claim(self):
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM commands WHERE state='running'").fetchone():
                return None
            row = db.execute("""
                SELECT c.* FROM commands c JOIN tenants t ON t.id=c.tenant
                JOIN sessions s ON s.id=c.session
                WHERE c.state='queued' AND s.state IN ('opening','active')
                ORDER BY t.turn,c.seq LIMIT 1
            """).fetchone()
            if row is None:
                return None
            turn = db.execute("SELECT COALESCE(MAX(turn),0)+1 FROM tenants").fetchone()[0]
            started = self.clock()
            db.execute("UPDATE tenants SET turn=? WHERE id=?", (turn, row["tenant"]))
            db.execute(
                "UPDATE commands SET state='running',started_ns=? WHERE id=?", (started, row["id"])
            )
            return {**dict(row), "started_ns": started, "payload": json.loads(row["payload"])}

    def finish(self, command, result, *, execution_ns, failed=False, lose_session=True):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM commands WHERE id=?", (command["id"],)).fetchone()
            if row is None or row["state"] != "running":
                raise ConflictError("command is not running")
            state = "failed" if failed else "completed"
            db.execute(
                "UPDATE commands SET state=?,result=? WHERE id=?",
                (state, encode(result), row["id"]),
            )
            db.execute(
                "INSERT INTO usage VALUES (?,?,?,?,?)",
                (
                    row["id"],
                    row["tenant"],
                    execution_ns,
                    max(0, row["started_ns"] - row["accepted_ns"]),
                    state,
                ),
            )
            if failed and lose_session:
                self._lose(db, row["session"])
            elif row["operation"] in ("create", "close"):
                db.execute(
                    "UPDATE sessions SET state=? WHERE id=?",
                    ("active" if row["operation"] == "create" else "closed", row["session"]),
                )
                if row["operation"] == "create":
                    db.execute(
                        "INSERT INTO residency(session,loaded_ns) VALUES (?,?)",
                        (row["session"], self.clock()),
                    )
                else:
                    db.execute(
                        "UPDATE residency SET released_ns=? WHERE session=?",
                        (self.clock(), row["session"]),
                    )

    def _lose(self, db, session):
        db.execute("UPDATE sessions SET state='lost' WHERE id=?", (session,))
        db.execute("UPDATE residency SET uncertain=1 WHERE session=?", (session,))
        db.execute(
            "UPDATE commands SET state='cancelled' WHERE session=? AND state='queued'", (session,)
        )

    def recover(self):
        """A new runtime cannot replay uncertain updates or retain the old GPU model state."""
        with self.transaction() as db:
            for row in db.execute("SELECT * FROM commands WHERE state='running'").fetchall():
                db.execute(
                    "INSERT OR IGNORE INTO usage VALUES (?,?,?,?,?)",
                    (
                        row["id"],
                        row["tenant"],
                        None,
                        max(0, row["started_ns"] - row["accepted_ns"]),
                        "unknown",
                    ),
                )
            db.execute("UPDATE commands SET state='unknown' WHERE state='running'")
            for row in db.execute(
                "SELECT id FROM sessions WHERE state IN ('opening','active')"
            ).fetchall():
                self._lose(db, row["id"])

    def status(self, principal, session):
        with self.lock:
            row = dict(self._owned(self.db, principal, session))
            commands = self.db.execute(
                "SELECT id,operation,state,result FROM commands WHERE session=? ORDER BY seq",
                (session,),
            ).fetchall()
            return {
                "session": session,
                "state": row["state"],
                "commands": [
                    {**dict(c), "result": json.loads(c["result"]) if c["result"] else None}
                    for c in commands
                ],
            }

    def cancel(self, principal, session, command_id):
        with self.transaction() as db:
            self._owned(db, principal, session)
            row = db.execute(
                "SELECT * FROM commands WHERE id=? AND session=?", (command_id, session)
            ).fetchone()
            if row is None:
                raise KeyError("unknown command")
            if row["operation"] in ("create", "close") or row["state"] == "running":
                raise ConflictError("lifecycle and executing commands cannot be cancelled")
            if row["state"] == "queued":
                db.execute("UPDATE commands SET state='cancelled' WHERE id=?", (command_id,))

    def usage(self, principal):
        with self.lock:
            rows = self.db.execute(
                "SELECT u.*,c.session,c.operation,c.tokens AS reserved_tokens,c.started_ns,c.result "
                "FROM usage u JOIN commands c ON c.id=u.command WHERE u.tenant=?",
                (principal.tenant,),
            ).fetchall()
            commands = []
            for row in rows:
                item = dict(row)
                output = json.loads(item.pop("result")) if row["result"] else {}
                item["tokens"] = {
                    key: output[key]
                    for key in ("input_tokens", "loss_tokens", "output_tokens")
                    if key in output
                }
                commands.append(item)
            result = {
                "execution_ns": sum(r["execution_ns"] or 0 for r in rows),
                "unknown_executions": sum(r["execution_ns"] is None for r in rows),
                "commands": commands,
                "adapters": [
                    dict(row)
                    for row in self.db.execute(
                        "SELECT r.*,s.state FROM residency r JOIN sessions s ON s.id=r.session "
                        "WHERE s.tenant=?",
                        (principal.tenant,),
                    )
                ],
            }
            result["resource_cost_estimate"] = estimate(result["execution_ns"], self.rate)
            return result

    def record_overhead(self, reason, elapsed_ns):
        with self.transaction() as db:
            db.execute("INSERT INTO overhead(reason,elapsed_ns) VALUES (?,?)", (reason, elapsed_ns))

    def pool_usage(self):
        with self.lock:
            row = self.db.execute(
                "SELECT COALESCE(SUM(execution_ns),0),SUM(execution_ns IS NULL) FROM usage"
            ).fetchone()
            overhead = self.db.execute(
                "SELECT COALESCE(SUM(elapsed_ns),0) FROM overhead"
            ).fetchone()[0]
            result = {
                "attributed_execution_ns": row[0],
                "unattributed_overhead_ns": overhead,
                "unknown_executions": row[1] or 0,
                "measured_pool_ns": row[0] + overhead,
            }
            if self.rate is not None:
                result["resource_cost_estimate"] = estimate(result["measured_pool_ns"], self.rate)
                result["overhead_cost_estimate"] = estimate(overhead, self.rate)
            return result

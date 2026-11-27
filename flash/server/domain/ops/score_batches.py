"""Score batches: run many chat requests against one model on Lambda and return them together.

A caller submits every request at once (``POST /v1/score-batches``) and polls one batch id. The
batch runs only on the Lambda GPU pool: each generation carries the serving provider pin, so it
never boots the Modal GPU fallback. Lambda does not always have capacity, so the batch first sends
one probe request and keeps resending it until Lambda serves it (the router itself waits up to an
hour for a Lambda worker per attempt). Only then does it fan out the rest. Results are returned
only when every request has finished, all at once.

Batches are durable. The row lives in the control-plane SQLite store and the requests and results
live under ``data_dir()/score_batches/<id>/``. Only the control-plane lease holder runs batches
(``ScoreBatchRunner``, started from the leader work). A runner heartbeats the batches it owns; a
batch whose owner stopped heartbeating (a restarted or demoted plane) is claimed again and resumes
with only the requests that have no recorded result.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import random
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from flash._internal.paths import data_dir
from flash.serve.contract.protocol import LAMBDA_SERVING_PROVIDER
from flash.server.platform import db

logger = logging.getLogger(__name__)

MAX_REQUESTS = 20_000
DEFAULT_CONCURRENCY = 32
MAX_CONCURRENCY = 64
MIN_DEADLINE_SECONDS = 60
MAX_DEADLINE_SECONDS = 24 * 60 * 60
# One generation may wait inside serving for a Lambda worker (the router's own bound), plus slack.
REQUEST_TIMEOUT_SECONDS = 2 * 60 * 60 + 60
# How long a batch keeps waiting for Lambda capacity before it fails its remaining requests.
CAPACITY_WAIT_ENV = "FLASH_SCORE_BATCH_CAPACITY_WAIT_SECONDS"
DEFAULT_CAPACITY_WAIT_SECONDS = 24 * 60 * 60
PROBE_RETRY_SECONDS = 60
# After capacity is confirmed, a request makes at most this many attempts on retryable failures.
ITEM_ATTEMPTS = 6
ITEM_BACKOFF_SECONDS = 5
ITEM_MAX_BACKOFF_SECONDS = 60
# Generations in flight across every batch on this plane.
MAX_INFLIGHT_ENV = "FLASH_SCORE_BATCH_MAX_INFLIGHT"
DEFAULT_MAX_INFLIGHT = 256
MAX_ACTIVE_BATCHES = 32
STALE_SECONDS = 120
POLL_SECONDS = 5
RETENTION_SECONDS = 7 * 24 * 60 * 60

ACTIVE_STATES = ("queued", "waiting_for_capacity", "running")
TERMINAL_STATES = ("done", "failed", "cancelled")
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS score_batches (
  batch_id        TEXT PRIMARY KEY,
  key_id          INTEGER NOT NULL,
  auth_kind       TEXT,
  api_key_id      TEXT,
  client_batch_id TEXT,
  org_id          TEXT,
  project_id      TEXT,
  run_id          TEXT,
  target          TEXT NOT NULL,
  total           INTEGER NOT NULL,
  concurrency     INTEGER NOT NULL,
  deadline_s      REAL,
  thinking        INTEGER,
  state           TEXT NOT NULL,
  owner           TEXT,
  heartbeat_at    REAL,
  created_at      REAL NOT NULL,
  capacity_at     REAL,
  finished_at     REAL,
  error           TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS score_batches_client_idx
  ON score_batches(key_id, client_batch_id) WHERE client_batch_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS score_batches_state_idx ON score_batches(state);
"""
_SCHEMA_READY: set[str] = set()
_SCHEMA_LOCK = threading.Lock()


class _Cancelled(RuntimeError):
    pass


class _DeadlineReached(RuntimeError):
    pass


def _conn() -> sqlite3.Connection:
    conn = db._connect()
    path = db.db_path()
    if path not in _SCHEMA_READY:
        with _SCHEMA_LOCK:
            if path not in _SCHEMA_READY:
                conn.executescript(_SCHEMA)
                conn.commit()
                _SCHEMA_READY.add(path)
    return conn


def batches_dir() -> Path:
    return data_dir() / "score_batches"


def _batch_dir(batch_id: str) -> Path:
    return batches_dir() / batch_id


def capacity_wait_seconds() -> float:
    raw = (os.environ.get(CAPACITY_WAIT_ENV) or "").strip()
    try:
        value = float(raw) if raw else DEFAULT_CAPACITY_WAIT_SECONDS
    except ValueError:
        value = DEFAULT_CAPACITY_WAIT_SECONDS
    return value if value > 0 else DEFAULT_CAPACITY_WAIT_SECONDS


def _max_inflight() -> int:
    raw = (os.environ.get(MAX_INFLIGHT_ENV) or "").strip()
    try:
        value = int(raw) if raw else DEFAULT_MAX_INFLIGHT
    except ValueError:
        value = DEFAULT_MAX_INFLIGHT
    return value if value > 0 else DEFAULT_MAX_INFLIGHT


# ---------------------------------------------------------------------------------------------
# validation and submission


def lambda_models() -> list[str]:
    from flash.serving.app.inference_provider import lambda_models as served

    return list(served())


def _bad(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _int_field(body: dict, name: str, default: int | None, low: int, high: int) -> int | None:
    value = body.get(name, default)
    if value is None:
        if default is not None:
            raise _bad(f"{name} must be an integer in [{low}, {high}]")
        return None
    if type(value) is not int or not low <= value <= high:
        raise _bad(f"{name} must be an integer in [{low}, {high}]")
    return value


def _request_items(body: dict) -> list[tuple[Any, dict]]:
    items = body.get("requests")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_REQUESTS:
        raise _bad(f"requests must be a list of 1-{MAX_REQUESTS} chat requests")
    out = []
    for number, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise _bad(f"request {number}: must be a JSON object")
        item = dict(item)
        identity = item.pop("id", None)
        if identity is not None and type(identity) not in (str, int):
            raise _bad(f"request {number}: id must be a string or integer")
        for selector in ("model", "checkpoint_id"):
            if selector in item:
                raise _bad(f"request {number}: {selector} is set once for the whole batch")
        if item.get("stream"):
            raise _bad(f"request {number}: score batch requests cannot stream")
        out.append((identity, item))
    return out


def _base_org(key: dict, org_header: str | None) -> str:
    from flash.server.platform.auth import serving_org_id

    if key.get("auth_kind") == "internal":
        org = serving_org_id(org_header)
    else:
        org = str(key.get("org_id") or "").strip()
    if not org:
        raise HTTPException(
            status_code=400,
            detail="base-model score batches need an organization to bill; send X-Freesolo-Org-Id",
        )
    return org


def _acting_api_key_id(key: dict) -> str | None:
    """The external key a base-model batch acts for, so serving keeps its per-key limits."""
    if key.get("auth_kind") == "internal":
        return None
    value = str(key.get("api_key_id") or "").strip()
    return value or None


def _validate_checkpoint_items(run_id, checkpoint, items, key, org, project) -> None:
    from flash.serve.request.openai import OpenAIRequestError, merge_stop_sequences
    from flash.serve.request.tool_calls import validate_tool_stop_sequences
    from flash.server.routes import serving_chat

    for number, (_, item) in enumerate(items, 1):
        try:
            request, _, spec, _, _ = serving_chat._resolve_chat_request(
                run_id, {**item, "checkpoint_id": checkpoint}, key, org, project
            )
            # The run's trained stops merge into every request at execution (managed_chat);
            # check them against tool calls now so no request runs before that rejection.
            mandatory = tuple(getattr(spec.train, "stop_sequences", ()) or ())
            try:
                validate_tool_stop_sequences(
                    merge_stop_sequences(mandatory, request.stop) or (),
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    error_type=OpenAIRequestError,
                )
            except OpenAIRequestError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException as exc:
            if exc.status_code == 400:
                raise _bad(f"request {number}: {exc.detail}. No requests were sent.") from None
            raise


def _validate_base_items(model, items, thinking) -> None:
    from flash.serve.request.openai import (
        OpenAIRequestError,
        parse_chat_request,
        reject_thinking_logprobs,
        reject_tool_capability,
    )
    from flash.serve.request.tool_calls import qualified_tool_parser

    enabled = bool(thinking)
    for number, (_, item) in enumerate(items, 1):
        try:
            parsed = parse_chat_request(
                {**item, "model": model}, require_model=True, allow_managed_selectors=False
            )
            # The same thinking-dependent checks serving applies, so no request runs (and is
            # charged) before a deterministic rejection of another one in the batch.
            reject_thinking_logprobs(thinking=enabled, logprobs=parsed.logprobs)
            reject_tool_capability(
                tools=parsed.tools,
                tool_choice=parsed.tool_choice,
                thinking=enabled,
                tool_parser=qualified_tool_parser(model),
            )
        except OpenAIRequestError as exc:
            raise _bad(f"request {number}: {exc}. No requests were sent.") from None


def submit(body: Any, key: dict, org: str | None, project: str | None) -> dict:
    """Validate every request, persist the batch, and return its id; nothing is sent here."""
    from flash.schema import parse_checkpoint_ref

    if not isinstance(body, dict):
        raise _bad("body must be a JSON object")
    unknown = sorted(
        set(body)
        - {
            "requests",
            "checkpoint_id",
            "model",
            "concurrency",
            "deadline_seconds",
            "thinking",
            "batch_id",
        }
    )
    if unknown:
        raise _bad(f"unsupported score batch field(s): {', '.join(unknown)}")
    client_batch_id = body.get("batch_id")
    if client_batch_id is not None and (
        not isinstance(client_batch_id, str) or not 1 <= len(client_batch_id) <= 128
    ):
        raise _bad("batch_id must be a string of 1-128 characters")
    if client_batch_id is not None:
        existing = _row_for_client(key["id"], client_batch_id)
        if existing is not None:
            return public(existing, include_results=False)
    concurrency = _int_field(body, "concurrency", DEFAULT_CONCURRENCY, 1, MAX_CONCURRENCY)
    deadline = _int_field(
        body, "deadline_seconds", None, MIN_DEADLINE_SECONDS, MAX_DEADLINE_SECONDS
    )
    thinking = body.get("thinking")
    if thinking is not None and type(thinking) is not bool:
        raise _bad("thinking must be a boolean")
    items = _request_items(body)
    checkpoint, model = body.get("checkpoint_id"), body.get("model")
    if (checkpoint is None) == (model is None):
        raise _bad(
            "name exactly one of checkpoint_id (a deployed checkpoint) or model (a base model)"
        )
    if checkpoint is not None:
        ref = parse_checkpoint_ref(checkpoint) if isinstance(checkpoint, str) else None
        if ref is None:
            raise _bad("checkpoint_id must be `<run_id>/final` or `<run_id>/step-N`")
        if thinking is not None:
            raise _bad("thinking comes from the run's spec for checkpoint batches; omit it")
        run_id = ref[0]
        _validate_checkpoint_items(run_id, checkpoint, items, key, org, project)
        target, scope_org = checkpoint, org
    else:
        if not isinstance(model, str) or model not in lambda_models():
            raise _bad(f"model must be a Lambda-served base model: {', '.join(lambda_models())}")
        _validate_base_items(model, items, thinking)
        run_id, target, scope_org = None, model, _base_org(key, org)
    batch_id = "sb-" + uuid.uuid4().hex
    folder = _batch_dir(batch_id)
    folder.mkdir(parents=True, exist_ok=False)
    _write_json(folder / "requests.json", [{"id": i, "request": r} for i, r in items])
    (folder / "results.jsonl").touch()
    now = time.time()
    conn = _conn()
    try:
        with conn:
            conn.execute(
                "INSERT INTO score_batches (batch_id, key_id, auth_kind, api_key_id,"
                " client_batch_id, org_id, project_id, run_id, target, total, concurrency,"
                " deadline_s, thinking, state, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    batch_id,
                    key["id"],
                    key.get("auth_kind"),
                    _acting_api_key_id(key),
                    client_batch_id,
                    scope_org,
                    project,
                    run_id,
                    target,
                    len(items),
                    concurrency,
                    deadline,
                    None if thinking is None else int(thinking),
                    "queued",
                    now,
                ),
            )
    except sqlite3.IntegrityError:
        # A concurrent retry with the same batch_id won the insert; return that batch.
        shutil.rmtree(folder, ignore_errors=True)
        existing = _row_for_client(key["id"], client_batch_id)
        if existing is None:
            raise
        return public(existing, include_results=False)
    return public(get_row(batch_id), include_results=False)


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------
# reads


def _row_for_client(key_id: int, client_batch_id: str) -> dict | None:
    row = (
        _conn()
        .execute(
            "SELECT * FROM score_batches WHERE key_id=? AND client_batch_id=?",
            (key_id, client_batch_id),
        )
        .fetchone()
    )
    return dict(row) if row else None


def get_row(batch_id: str) -> dict | None:
    row = _conn().execute("SELECT * FROM score_batches WHERE batch_id=?", (batch_id,)).fetchone()
    return dict(row) if row else None


def owned_row(batch_id: str, key: dict) -> dict:
    """The caller's batch by Flash's id or by the caller's own batch_id (idempotency key).

    The second form lets a caller whose submit reply was lost still poll or cancel the batch.
    """
    row = get_row(batch_id)
    if row is None or row["key_id"] != key["id"]:
        row = _row_for_client(key["id"], batch_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown score batch: {batch_id}")
    return row


def _results(batch_id: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    path = _batch_dir(batch_id) / "results.jsonl"
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return out
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a torn final line from a crash mid-write; that request runs again
        if isinstance(record, dict) and isinstance(record.get("index"), int):
            # First record wins: a late answer from a plane that lost the batch cannot replace
            # a result already recorded (and possibly already returned).
            out.setdefault(record["index"], record)
    return out


def public(row: dict, *, include_results: bool) -> dict:
    results = _results(row["batch_id"])
    failed = sum(1 for r in results.values() if "error" in r)
    now = time.time()
    waited_until = row["capacity_at"] or row["finished_at"] or now
    body = {
        "batch_id": row["batch_id"],
        "client_batch_id": row["client_batch_id"],
        "state": row["state"],
        "provider": LAMBDA_SERVING_PROVIDER,
        ("checkpoint_id" if row["run_id"] else "model"): row["target"],
        "total": row["total"],
        "completed": len(results),
        "succeeded": len(results) - failed,
        "failed": failed,
        "concurrency": row["concurrency"],
        "deadline_seconds": row["deadline_s"],
        "created_at": row["created_at"],
        "capacity_at": row["capacity_at"],
        "finished_at": row["finished_at"],
        "capacity_wait_seconds": round(max(0.0, waited_until - row["created_at"]), 3),
        "error": row["error"],
    }
    if include_results and row["state"] in TERMINAL_STATES:
        records = []
        for index in range(row["total"]):
            record = dict(results.get(index) or {"error": row["error"] or "not run"})
            record.pop("index", None)
            record.pop("owner", None)
            records.append(record)
        body["results"] = records
    return body


def cancel(batch_id: str, key: dict) -> dict:
    batch_id = owned_row(batch_id, key)["batch_id"]
    conn = _conn()
    with conn:
        conn.execute(
            "UPDATE score_batches SET state='cancelled', finished_at=?, error=COALESCE(error, ?)"
            " WHERE batch_id=? AND state IN (?,?,?)",
            (time.time(), "cancelled by the caller", batch_id, *ACTIVE_STATES),
        )
    return public(get_row(batch_id), include_results=True)


# ---------------------------------------------------------------------------------------------
# execution


def _status_of(exc: BaseException) -> int | None:
    if isinstance(exc, HTTPException):
        return exc.status_code
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None


def _retryable(exc: BaseException) -> bool:
    status = _status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    return isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError))


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        return f"serving HTTP {exc.response.status_code}" + (f": {detail}" if detail else "")
    return str(exc) or type(exc).__name__


class _Batch:
    """One claimed batch on this plane."""

    def __init__(self, runner: ScoreBatchRunner, row: dict) -> None:
        self.runner, self.row = runner, row
        self.batch_id = row["batch_id"]
        self.folder = _batch_dir(self.batch_id)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.stop_at: float | None = None

    # persistence -----------------------------------------------------------------------------

    def _update(self, **fields: Any) -> None:
        names = ", ".join(f"{name}=?" for name in fields)
        conn = _conn()
        with conn:
            updated = conn.execute(
                f"UPDATE score_batches SET {names} WHERE batch_id=? AND owner=? AND state IN (?,?,?)",
                (*fields.values(), self.batch_id, self.runner.owner, *ACTIVE_STATES),
            ).rowcount
        if not updated:
            # Cancelled (or claimed by another plane) since this runner last looked.
            raise _Cancelled("batch is no longer active here")
        self.row.update(fields)

    def _record(self, index: int, identity: Any, **result: Any) -> None:
        record = {"index": index, **({"id": identity} if identity is not None else {}), **result}
        line = json.dumps({**record, "owner": self.runner.owner}) + "\n"
        row = get_row(self.batch_id)
        if row is None or row["owner"] != self.runner.owner or row["state"] not in ACTIVE_STATES:
            # Released, cancelled or claimed elsewhere while this request ran: do not publish.
            raise _Cancelled("batch is no longer active here")
        with self.lock, (self.folder / "results.jsonl").open("a") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _check_live(self) -> None:
        if self.stop.is_set() or self.runner.stopping.is_set():
            raise _Cancelled("runner stopping")
        row = get_row(self.batch_id)
        if row is None or row["state"] == "cancelled" or row["owner"] != self.runner.owner:
            raise _Cancelled("batch cancelled or claimed elsewhere")

    # one request -----------------------------------------------------------------------------

    def _key(self) -> dict:
        return {"id": self.row["key_id"], "auth_kind": self.row["auth_kind"]}

    def send(self, request: dict) -> dict:
        with self.runner.inflight:
            # The slot may have been a long wait; never start a generation for a dead batch
            # or one past its deadline.
            self._check_live()
            if self.stop_at is not None and time.time() >= self.stop_at:
                raise _DeadlineReached
            return self.runner.send(self.row, self._key(), request)

    # phases ----------------------------------------------------------------------------------

    def run(self) -> None:
        try:
            self._run()
        except _Cancelled:
            return
        except Exception as exc:
            logger.exception("score_batch_failed batch=%s", self.batch_id)
            self._finish("failed", error=f"score batch failed: {_error_text(exc)}")

    def _repair_tail(self) -> None:
        """Drop a partly written final line so new records start on a record boundary."""
        path = self.folder / "results.jsonl"
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            path.touch()
            return
        if data and not data.endswith(b"\n"):
            with path.open("r+b") as handle:
                handle.truncate(data.rfind(b"\n") + 1)
                handle.flush()
                os.fsync(handle.fileno())

    def _run(self) -> None:
        requests = json.loads((self.folder / "requests.json").read_text())
        self._repair_tail()
        done = _results(self.batch_id)
        pending = [i for i in range(len(requests)) if i not in done]
        if self.row["capacity_at"] is None:
            self._update(state="waiting_for_capacity")
            pending = self._wait_for_capacity(requests, pending)
            if pending is None:
                return
        self._update(state="running")
        self._fan_out(requests, pending)
        self._check_live()
        self._finish("done")

    def _wait_for_capacity(self, requests: list, pending: list[int]) -> list[int] | None:
        """Send one request at a time until Lambda serves one; returns the requests still to run.

        A retryable failure moves that request to the back, so one request that fails for its own
        reasons cannot hold the batch; capacity is confirmed by whichever request succeeds first.
        """
        deadline = self.row["created_at"] + self.runner.capacity_wait()
        while pending:
            index = pending[0]
            entry = requests[index]
            self._check_live()
            try:
                response = self.send(entry["request"])
            except _Cancelled:
                raise
            except Exception as exc:
                if not _retryable(exc):
                    # This request is bad on its own; it is not evidence about capacity.
                    self._record(
                        index, entry["id"], error=_error_text(exc), status_code=_status_of(exc)
                    )
                    pending.pop(0)
                    continue
                remaining = deadline - self.runner.clock()
                if remaining <= 0:
                    message = (
                        "Lambda capacity was unavailable for "
                        f"{self.runner.capacity_wait() / 3600:g} hours; no requests ran"
                    )
                    self._finish("failed", error=message)
                    return None
                logger.warning(
                    "score_batch_waiting_for_lambda batch=%s error=%s retry_in=%ss",
                    self.batch_id,
                    _error_text(exc)[:200],
                    PROBE_RETRY_SECONDS,
                )
                pending.append(pending.pop(0))
                self._sleep(min(PROBE_RETRY_SECONDS, remaining))
                continue
            self._record(index, entry["id"], response=response)
            self._update(capacity_at=time.time())
            return pending[1:]
        self._update(capacity_at=time.time())
        return []

    def _fan_out(self, requests: list, pending: list[int]) -> None:
        if not pending:
            return
        deadline_s = self.row["deadline_s"]
        stop_at = None if deadline_s is None else self.row["capacity_at"] + deadline_s
        self.stop_at = stop_at
        cursor = iter(pending)
        cursor_lock = threading.Lock()

        def next_index() -> int | None:
            with cursor_lock:
                return next(cursor, None)

        def worker() -> None:
            while (index := next_index()) is not None:
                entry = requests[index]
                try:
                    if stop_at is not None and time.time() >= stop_at:
                        raise _DeadlineReached
                    self._run_item(index, entry)
                except _DeadlineReached:
                    self._record(
                        index,
                        entry["id"],
                        error="unfinished: the batch deadline was reached before this request ran",
                        unfinished=True,
                    )

        workers = min(self.row["concurrency"], len(pending))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=f"score-{self.batch_id[-6:]}"
        ) as pool:
            futures = [pool.submit(worker) for _ in range(workers)]
            for future in futures:
                future.result()

    def _run_item(self, index: int, entry: dict) -> None:
        last: BaseException | None = None
        for attempt in range(ITEM_ATTEMPTS):
            self._check_live()
            try:
                response = self.send(entry["request"])
            except (_Cancelled, _DeadlineReached):
                raise
            except Exception as exc:
                last = exc
                if not _retryable(exc):
                    break
                delay = min(ITEM_MAX_BACKOFF_SECONDS, ITEM_BACKOFF_SECONDS * 2**attempt)
                self._sleep(delay * random.uniform(0.5, 1.0))
                continue
            self._record(index, entry["id"], response=response)
            return
        assert last is not None
        self._record(index, entry["id"], error=_error_text(last), status_code=_status_of(last))

    def _sleep(self, seconds: float) -> None:
        if self.stop.wait(seconds) or self.runner.stopping.is_set():
            raise _Cancelled("runner stopping")

    def _finish(self, state: str, *, error: str | None = None) -> None:
        conn = _conn()
        with conn:
            conn.execute(
                "UPDATE score_batches SET state=?, finished_at=?, error=? "
                "WHERE batch_id=? AND owner=? AND state IN (?,?,?)",
                (state, time.time(), error, self.batch_id, self.runner.owner, *ACTIVE_STATES),
            )


def _send_request(row: dict, key: dict, request: dict) -> dict:
    """Forward one request to serving, pinned to Lambda. Raises on failure."""
    if row["run_id"]:
        from flash.server.routes import serving_chat

        return serving_chat.managed_chat(
            row["run_id"],
            {**request, "checkpoint_id": row["target"]},
            key,
            row["org_id"],
            row["project_id"],
            timeout_s=REQUEST_TIMEOUT_SECONDS,
            serving_provider=LAMBDA_SERVING_PROVIDER,
        )
    import flash.serve.deployment.deploy as serving_deploy
    from flash.serve.request.openai import parse_chat_request

    parsed = parse_chat_request(
        {**request, "model": row["target"]}, require_model=True, allow_managed_selectors=False
    )
    thinking = bool(row["thinking"]) if row["thinking"] is not None else False
    tools: dict[str, Any] = {}
    if parsed.tools is not None:
        tools = {
            "tools": parsed.tools,
            "tool_choice": parsed.tool_choice,
            "parallel_tool_calls": parsed.parallel_tool_calls,
        }
    logprob_ids = getattr(parsed, "logprob_token_ids", None)
    return serving_deploy.base_chat(
        row["target"],
        parsed.messages,
        org_id=row["org_id"],
        timeout_s=REQUEST_TIMEOUT_SECONDS,
        serving_provider=LAMBDA_SERVING_PROVIDER,
        api_key_id=row.get("api_key_id"),
        temperature=parsed.temperature,
        max_tokens=parsed.max_tokens,
        thinking=thinking,
        stop=list(parsed.stop) or None,
        structured_outputs=parsed.structured_outputs,
        top_p=parsed.top_p,
        chat_template_kwargs={**parsed.chat_template_kwargs, "enable_thinking": thinking},
        n=parsed.n,
        seed=parsed.seed,
        frequency_penalty=parsed.frequency_penalty,
        presence_penalty=parsed.presence_penalty,
        logprobs=parsed.logprobs,
        top_logprobs=parsed.top_logprobs,
        **tools,
        **({"logprob_token_ids": list(logprob_ids)} if logprob_ids is not None else {}),
    )


class ScoreBatchRunner:
    """Claims and runs score batches on the control-plane lease holder."""

    def __init__(self, *, send=_send_request, clock=time.time, capacity_wait=capacity_wait_seconds):
        self.owner = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self.send = send
        self.clock = clock
        self.capacity_wait = capacity_wait
        self.stopping = threading.Event()
        self.inflight = threading.BoundedSemaphore(_max_inflight())
        self.active: dict[str, tuple[_Batch, threading.Thread]] = {}
        self._thread: threading.Thread | None = None
        self._last_cleanup = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="score-batches", daemon=True)
        self._thread.start()

    def signal_stop(self) -> None:
        self.stopping.set()

    def release(self) -> None:
        """Stop running and give up every batch this runner owns, for the next lease holder."""
        self.stopping.set()
        for batch, _ in list(self.active.values()):
            batch.stop.set()
        if self._thread is not None:
            self._thread.join(POLL_SECONDS * 2)
        conn = _conn()
        with conn:
            conn.execute(
                "UPDATE score_batches SET owner=NULL, heartbeat_at=NULL WHERE owner=?",
                (self.owner,),
            )

    def _loop(self) -> None:
        while not self.stopping.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("score_batch_runner_tick_failed")
            self.stopping.wait(POLL_SECONDS)

    def tick(self) -> None:
        now = self.clock()
        for batch_id, (_batch, thread) in list(self.active.items()):
            if not thread.is_alive():
                del self.active[batch_id]
        conn = _conn()
        if self.active:
            with conn:
                conn.execute(
                    f"UPDATE score_batches SET heartbeat_at=? WHERE owner=? AND batch_id IN "
                    f"({','.join('?' * len(self.active))})",
                    (now, self.owner, *self.active),
                )
        for batch_id, (batch, _) in list(self.active.items()):
            row = get_row(batch_id)
            if row is None or row["state"] == "cancelled" or row["owner"] != self.owner:
                batch.stop.set()
        while len(self.active) < MAX_ACTIVE_BATCHES:
            row = self._claim(now)
            if row is None:
                break
            batch = _Batch(self, row)
            thread = threading.Thread(
                target=batch.run, name=f"score-batch-{row['batch_id'][-8:]}", daemon=True
            )
            self.active[row["batch_id"]] = (batch, thread)
            thread.start()
        if now - self._last_cleanup > 3600:
            self._last_cleanup = now
            self._cleanup(now)

    def _claim(self, now: float) -> dict | None:
        conn = _conn()
        with conn:
            row = conn.execute(
                "SELECT batch_id FROM score_batches WHERE state IN (?,?,?) AND "
                "(owner IS NULL OR heartbeat_at IS NULL OR heartbeat_at < ?) "
                "ORDER BY created_at LIMIT 1",
                (*ACTIVE_STATES, now - STALE_SECONDS),
            ).fetchone()
            if row is None:
                return None
            claimed = conn.execute(
                "UPDATE score_batches SET owner=?, heartbeat_at=? WHERE batch_id=? AND state IN "
                "(?,?,?) AND (owner IS NULL OR heartbeat_at IS NULL OR heartbeat_at < ?)",
                (self.owner, now, row["batch_id"], *ACTIVE_STATES, now - STALE_SECONDS),
            ).rowcount
        return get_row(row["batch_id"]) if claimed else None

    def _cleanup(self, now: float) -> None:
        conn = _conn()
        rows = conn.execute(
            "SELECT batch_id FROM score_batches WHERE state IN (?,?,?) AND finished_at < ?",
            (*TERMINAL_STATES, now - RETENTION_SECONDS),
        ).fetchall()
        for row in rows:
            shutil.rmtree(_batch_dir(row["batch_id"]), ignore_errors=True)
            with conn:
                conn.execute("DELETE FROM score_batches WHERE batch_id=?", (row["batch_id"],))

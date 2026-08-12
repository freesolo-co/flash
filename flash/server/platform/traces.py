"""Trace persistence and export for the control plane's recording proxy."""

from __future__ import annotations

import itertools
import json
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any

from flash.core.spec import require_project_id
from flash.server.platform import db

EXPORT_FORMATS = {"records", "prompts", "raw"}
MAX_EXPORT_TRACES = 1000
# a second cap, on the response rather than the row count. `MAX_EXPORT_TRACES` bounds traces, not
# bytes, and a payload string may legitimately reach 1 MB, so the trace cap alone permits a
# multi-gigabyte body that the plane builds in memory and the CLI holds whole. 64 MB is far above
# any ordinary project's export and far below what exhausts a self-hoster's box.
MAX_EXPORT_BYTES = 64 * 1024 * 1024

_MAX_ATTRIBUTE_COUNT = 128
_MAX_ATTRIBUTE_VALUE_LENGTH = 8_192
_MAX_ATTRIBUTE_DEPTH = 6
_MAX_SEQUENCE_LENGTH = 128
_MAX_TRACE_TITLE_LENGTH = 500
# model/provider/span-name columns. far above any real model id ("gpt-4o-2024-11-20" is 18 chars)
# and small enough that a caller cannot grow the database through them.
_MAX_IDENTIFIER_LENGTH = 500
# payloads are the product, not telemetry: a recorded completion is what `traces export` turns into
# a training row, so cutting it at the 8 KiB attribute bound would ship a truncated target with an
# ellipsis in it and no indication the text was ever longer. they still need A bound -- an untrimmed
# payload is unbounded rows in SQLite -- just one far above any real chat completion.
MAX_PAYLOAD_VALUE_LENGTH = 1_000_000
# far above a real chat exchange, but low enough that one authenticated caller cannot materialize
# an attacker-sized sqlite value or stall the single writer with it.
MAX_PAYLOAD_TOTAL_BYTES = 8 * 1024 * 1024
# likewise for nesting. an ordinary tool schema (`tools[] > function > parameters > properties >
# field > type`) already sits at depth 6, so the attribute depth would repr() the leaf and store a
# JSON schema as the string "{}".
_MAX_PAYLOAD_DEPTH = 24
# and for collection width. a long chat history past the 128-item attribute bound lost its TAIL --
# the newest turns, including the prompt the reply actually answers -- while the response was stored
# whole, so `records` paired a completion with a conversation that no longer led to it.
_MAX_PAYLOAD_COLLECTION = 100_000
_PAYLOAD_TRUNCATED_ATTRIBUTE = "payload_truncated"
# absent finish reasons are accepted for providers and older envelopes that never supplied one;
# explicit unknown reasons are rejected because they are not evidence of a clean terminal response.
_ACCEPTED_FINISH_REASONS = {None, "stop", "tool_calls"}


@dataclass
class _TruncationFlag:
    hit: bool = False


@dataclass
class TraceSpan:
    name: str | None = None
    provider: str | None = None
    model: str | None = None
    duration_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_payload: Any = None
    output_payload: Any = None
    attributes: dict[str, Any] | None = None
    status_code: str | None = None
    error: str | None = None


def sanitize_json_value(
    value: Any,
    *,
    depth: int = 0,
    max_string: int | None = None,
    max_depth: int | None = None,
    max_collection: int | None = None,
    flag: _TruncationFlag | None = None,
) -> Any:
    """Bound nested values before they enter the plane's durable trace store.

    `max_string` and `max_depth` select the bounds: metadata and attributes keep the tight attribute
    limits, while payloads pass the payload limits so an exported completion is the whole completion
    and a nested tool schema survives as a schema.
    """
    limit = _MAX_ATTRIBUTE_VALUE_LENGTH if max_string is None else max_string
    depth_limit = _MAX_ATTRIBUTE_DEPTH if max_depth is None else max_depth
    keys = _MAX_ATTRIBUTE_COUNT if max_collection is None else max_collection
    items = _MAX_SEQUENCE_LENGTH if max_collection is None else max_collection
    if depth >= depth_limit:
        return _truncate(repr(value), limit, flag=flag)
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _truncate(value, limit, flag=flag)
    if isinstance(value, dict):
        return {
            _truncate(str(key), limit, flag=flag): sanitize_json_value(
                item,
                depth=depth + 1,
                max_string=max_string,
                max_depth=max_depth,
                max_collection=max_collection,
                flag=flag,
            )
            for key, item in itertools.islice(value.items(), keys)
        }
    if isinstance(value, list | tuple):
        return [
            sanitize_json_value(
                item,
                depth=depth + 1,
                max_string=max_string,
                max_depth=max_depth,
                max_collection=max_collection,
                flag=flag,
            )
            for item in itertools.islice(value, items)
        ]
    return _truncate(repr(value), limit, flag=flag)


def store_trace(
    *,
    key_id: int,
    project_id: str,
    trace_title: str | None,
    metadata: dict[str, Any] | None,
    spans: list[TraceSpan],
) -> str:
    """Persist one trace and all of its spans atomically."""
    project_id = require_project_id(project_id)
    if not spans:
        raise ValueError("a trace requires at least one span")

    created_at = time.time()
    trace_id = str(uuid.uuid4())
    # bound the identifier columns. `model` is caller-supplied and lands in TWO columns outside the
    # payload bounds -- once per span and once on the trace -- and a span is stored even for an
    # upstream 4xx, so an authenticated caller could grow the shared database by an unbounded
    # amount without ever making a successful call. these are identifiers, not content: nothing
    # legitimate approaches the limit, unlike a payload.
    model = next((_bounded_identifier(span.model) for span in spans if span.model), None)

    # sanitize and serialize BEFORE taking the write lock. payloads are megabyte-scale by design
    # here, and doing this inside `BEGIN IMMEDIATE` held the single writer for the whole encode --
    # blocking unrelated writers (keys, runs, teacher ledgers) and pushing them toward the busy
    # timeout for work that touches no database state.
    serialized_metadata = _json_dump(metadata)
    serialized_spans = []
    for span in spans:
        truncation = _TruncationFlag()
        input_payload = _json_dump(
            span.input_payload,
            max_string=MAX_PAYLOAD_VALUE_LENGTH,
            max_depth=_MAX_PAYLOAD_DEPTH,
            max_collection=_MAX_PAYLOAD_COLLECTION,
            flag=truncation,
            max_bytes=MAX_PAYLOAD_TOTAL_BYTES,
        )
        output_payload = _json_dump(
            span.output_payload,
            max_string=MAX_PAYLOAD_VALUE_LENGTH,
            max_depth=_MAX_PAYLOAD_DEPTH,
            max_collection=_MAX_PAYLOAD_COLLECTION,
            flag=truncation,
            max_bytes=MAX_PAYLOAD_TOTAL_BYTES,
        )
        attributes = dict(span.attributes) if span.attributes is not None else None
        if truncation.hit:
            # the marker last, so a caller-supplied attribute of the same name cannot overwrite it.
            # it is the only signal that a stored payload is no longer the payload, and `records`
            # skips the row on it alone.
            attributes = {**(attributes or {}), _PAYLOAD_TRUNCATED_ATTRIBUTE: True}
        serialized_spans.append(
            (
                str(uuid.uuid4()),
                created_at,
                trace_id,
                _bounded_identifier(span.name),
                _bounded_identifier(span.provider),
                _bounded_identifier(span.model),
                span.duration_ms,
                span.input_tokens,
                span.output_tokens,
                input_payload,
                output_payload,
                _json_dump(attributes),
                span.status_code,
                span.error,
            )
        )
    serialized_title = _truncate(trace_title or "", _MAX_TRACE_TITLE_LENGTH) or None

    conn = db._connect()
    try:
        db._immediate(conn)
        conn.execute(
            "INSERT INTO llm_traces ("
            "id, created_at, key_id, project_id, model, trace_title, metadata"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                trace_id,
                created_at,
                int(key_id),
                project_id,
                model,
                serialized_title,
                serialized_metadata,
            ),
        )
        conn.executemany(
            "INSERT INTO llm_trace_spans ("
            "id, created_at, llm_trace_id, name, provider, model, duration_ms, input_tokens, "
            "output_tokens, input_payload, output_payload, attributes, status_code, error"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            serialized_spans,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return trace_id


def list_projects(*, key_id: int) -> list[dict[str, str]]:
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT project_id, MAX(created_at) AS updated_at FROM llm_traces "
            "WHERE key_id = ? GROUP BY project_id ORDER BY updated_at DESC, project_id",
            (int(key_id),),
        ).fetchall()
    return [{"id": str(row["project_id"]), "name": str(row["project_id"])} for row in rows]


def export_traces(
    *, key_id: int, project_id: str, export_format: str, limit: int
) -> dict[str, Any]:
    try:
        project_id = require_project_id(project_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    if export_format not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of: {', '.join(sorted(EXPORT_FORMATS))}")
    if limit <= 0 or limit > MAX_EXPORT_TRACES:
        raise ValueError(f"limit must be between 1 and {MAX_EXPORT_TRACES}")

    with db._connect() as conn:
        # one row past the limit, purely to tell "exactly `limit` traces exist" from "more exist".
        # counting `len(rows) >= limit` cannot separate them, so a project holding exactly the cap
        # would be reported incomplete and the CLI would claim older traces were missing.
        probe_rows = conn.execute(
            "SELECT * FROM llm_traces WHERE key_id = ? AND project_id = ? "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (int(key_id), project_id, int(limit) + 1),
        ).fetchall()
        truncated = len(probe_rows) > limit
        trace_rows = probe_rows[:limit]
        records: list[dict[str, Any]] = []
        # the trace cap alone does not bound the RESPONSE. payload strings are capped at 1 MB each
        # by design, so 1000 traces holding a request and a reply is a 2 GB JSON body -- built in
        # memory here, serialized again by the framework, then held whole by the client. cap the
        # bytes as well, and report the export truncated so the CLI says older traces are missing
        # instead of looking like it read the project whole.
        remaining_bytes = MAX_EXPORT_BYTES
        examined = 0
        for trace_row in trace_rows:
            span_rows = conn.execute(
                "SELECT * FROM llm_trace_spans WHERE llm_trace_id = ? ORDER BY created_at, rowid",
                (trace_row["id"],),
            ).fetchall()
            examined += 1
            record = _export_record(_raw_trace(trace_row, span_rows), export_format)
            if record is None:
                continue
            records.append(record)
            # measured after appending, so the first record is always emitted whole and the budget
            # can only overshoot by one row. a record is bounded by the payload caps, so that
            # overshoot is bounded too -- whereas refusing a too-large first record would return an
            # empty export the CLI reports as "no exportable traces".
            # encoded LENGTH, not character count: `ensure_ascii=False` keeps non-ASCII text as
            # itself, so one counted character can be up to four UTF-8 bytes on the wire. Counting
            # characters let an emoji-heavy export ship several times the nominal budget.
            remaining_bytes -= len(json.dumps(record, ensure_ascii=False).encode("utf-8"))
            if remaining_bytes <= 0:
                # exhausting the budget ON the last row is not truncation: everything was
                # returned. claiming otherwise sends the CLI and the env scaffold to warn about
                # older traces that do not exist.
                truncated = truncated or examined < len(trace_rows)
                break

    return {
        "records": records,
        "traces": examined,
        "skipped": 0 if export_format == "raw" else examined - len(records),
        "format": export_format,
        # the read is capped, so a project past the cap has older traces this response does not
        # contain. reporting the truncated count alone would read as "that is all of them".
        "truncated": truncated,
    }


def _export_record(raw: dict[str, Any], export_format: str) -> dict[str, Any] | None:
    """One export row in the requested shape, or None when the trace has nothing usable in it."""
    if export_format == "raw":
        return raw
    if any(_span_payload_was_truncated(span) for span in raw["spans"]):
        return None
    input_payload, output_payload = _training_pair(raw["spans"])
    if not _usable_payload(input_payload):
        return None
    if export_format == "prompts":
        return {"input": input_payload}
    if not _usable_payload(output_payload):
        return None
    return {"input": input_payload, "output": output_payload}


def _raw_trace(trace_row: Any, span_rows: list[Any]) -> dict[str, Any]:
    return {
        "id": trace_row["id"],
        "created_at": trace_row["created_at"],
        "project_id": trace_row["project_id"],
        "model": trace_row["model"],
        "trace_title": trace_row["trace_title"],
        "metadata": _json_load(trace_row["metadata"]),
        "spans": [
            {
                "id": span["id"],
                "created_at": span["created_at"],
                "name": span["name"],
                "provider": span["provider"],
                "model": span["model"],
                "duration_ms": span["duration_ms"],
                "input_tokens": span["input_tokens"],
                "output_tokens": span["output_tokens"],
                "input_payload": _json_load(span["input_payload"]),
                "output_payload": _json_load(span["output_payload"]),
                "attributes": _json_load(span["attributes"]),
                "status_code": span["status_code"],
                "error": span["error"],
            }
            for span in span_rows
        ],
    }


def _span_payload_was_truncated(span: dict[str, Any]) -> bool:
    attributes = span.get("attributes")
    return isinstance(attributes, dict) and attributes.get(_PAYLOAD_TRUNCATED_ATTRIBUTE) is True


def _training_pair(spans: list[dict[str, Any]]) -> tuple[Any, Any]:
    """Return a trainable pair, reducing chat envelopes to one user and assistant text pair."""
    input_payload = next(
        (span.get("input_payload") for span in spans if _usable_payload(span.get("input_payload"))),
        None,
    )
    output_payload = next(
        (
            span.get("output_payload")
            for span in reversed(spans)
            # an ERROR span's output is the provider's rejection -- a 429 body, a partial response
            # from an interrupted stream -- not a reply anyone wants to train toward. `raw` still
            # exports it; `records` must not present it as a desired completion.
            if span.get("status_code") != "ERROR" and _usable_payload(span.get("output_payload"))
        ),
        None,
    )
    # no fallback to the raw payload on either half. every stored span comes from the recording
    # proxy's chat.completions route, so a payload without the chat shape is not an alternative
    # encoding of a completion -- it is a malformed request, or a body that never was one: a
    # gateway's HTTP 200 login interstitial, an error object a provider returned without an error
    # status. those record as OK and would otherwise export whole, as the exact text `records`
    # trains the model to produce. `_chat_prompt`/`_chat_reply` return None there, and the row is
    # skipped as unusable.
    return _chat_prompt(input_payload), _chat_reply(output_payload)


def _chat_prompt(payload: Any) -> str | None:
    """The sole user turn of a chat-completions request, or None when context would be lost.

    `records` deliberately exports only the last user turn rather than a transcript containing prior
    answers. That conversion is correct only for a genuinely single-turn request: system or developer
    instructions, earlier turns, and trailing assistant prefills can all make the target unreachable
    from the exported text. This trades recall for correct training rows; `raw` still preserves every
    message, while converted formats skip any request whose user message is not the sole message.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        return None
    messages = payload["messages"]
    if len(messages) != 1:
        return None
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        return None
    return _message_text(message.get("content"))


def _chat_reply(payload: Any) -> str | None:
    """The assistant text of a chat-completions response, or None when it is not one.

    None rather than "" for an envelope carrying no assistant text -- an empty `choices` list, a
    choice without a message. The caller skips the row either way, but "" would also be the reply
    of a model that legitimately returned empty text, and the two want the same treatment here:
    neither is a completion worth training toward.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        return None
    if not payload["choices"]:
        return None
    choice = payload["choices"][0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        return None
    if choice.get("finish_reason") not in _ACCEPTED_FINISH_REASONS:
        return None
    return _message_text(choice["message"].get("content"))


def _message_text(content: Any) -> str | None:
    """The text of a message's content, or None when it cannot be represented as text.

    A content list holding a non-text part -- `image_url`, `input_audio`, a file -- returns None
    rather than the text of the remaining parts. `records` rows are text in and text out, so
    joining only the text would pair an answer that may depend entirely on the image with a prompt
    that no longer contains it: a row whose target is unreachable from its input. Skipping the
    example loses one row; keeping it corrupts the dataset in a way nothing downstream can detect.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            return None
        text = part.get("text")
        if not isinstance(text, str):
            return None
        parts.append(text)
    return "".join(parts) if parts else None


def _usable_payload(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _json_dump(
    value: Any,
    *,
    max_string: int | None = None,
    max_depth: int | None = None,
    max_collection: int | None = None,
    flag: _TruncationFlag | None = None,
    max_bytes: int | None = None,
) -> str | None:
    if value is None:
        return None
    serialized = json.dumps(
        sanitize_json_value(
            value,
            max_string=max_string,
            max_depth=max_depth,
            max_collection=max_collection,
            flag=flag,
        ),
        ensure_ascii=False,
        sort_keys=True,
    )
    encoded_bytes = len(serialized.encode("utf-8"))
    if max_bytes is None or encoded_bytes <= max_bytes:
        return serialized
    if flag is not None:
        flag.hit = True
    return json.dumps(
        {
            "flash_payload_dropped": {
                "bytes": encoded_bytes,
                "reason": "payload exceeded the stored size limit",
            }
        },
        sort_keys=True,
    )


def _json_load(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def _truncate(value: str, limit: int, *, flag: _TruncationFlag | None = None) -> str:
    if len(value) <= limit:
        return value
    if flag is not None:
        flag.hit = True
    return f"{value[: limit - 3]}..."


def _bounded_identifier(value: str | None) -> str | None:
    """Bound a caller-supplied identifier column (model, provider, span name).

    These sit outside the payload bounds and are stored even for a failed call, so an unbounded
    value is database growth an authenticated caller controls directly.
    """
    return None if value is None else _truncate(str(value), _MAX_IDENTIFIER_LENGTH)

"""Trace persistence and export for the control plane's recording proxy."""

from __future__ import annotations

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

_MAX_ATTRIBUTE_COUNT = 128
_MAX_ATTRIBUTE_VALUE_LENGTH = 8_192
_MAX_ATTRIBUTE_DEPTH = 6
_MAX_SEQUENCE_LENGTH = 128
_MAX_TRACE_TITLE_LENGTH = 500


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


def sanitize_json_value(value: Any, *, depth: int = 0) -> Any:
    """Bound nested values before they enter the plane's durable trace store."""
    if depth >= _MAX_ATTRIBUTE_DEPTH:
        return _truncate(repr(value), _MAX_ATTRIBUTE_VALUE_LENGTH)
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _truncate(value, _MAX_ATTRIBUTE_VALUE_LENGTH)
    if isinstance(value, dict):
        return {
            _truncate(str(key), _MAX_ATTRIBUTE_VALUE_LENGTH): sanitize_json_value(
                item, depth=depth + 1
            )
            for key, item in list(value.items())[:_MAX_ATTRIBUTE_COUNT]
        }
    if isinstance(value, list | tuple):
        return [
            sanitize_json_value(item, depth=depth + 1)
            for item in list(value)[:_MAX_SEQUENCE_LENGTH]
        ]
    return _truncate(repr(value), _MAX_ATTRIBUTE_VALUE_LENGTH)


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

    conn = db._connect()
    created_at = time.time()
    trace_id = str(uuid.uuid4())
    model = next((span.model for span in spans if span.model), None)
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
                _truncate(trace_title or "", _MAX_TRACE_TITLE_LENGTH) or None,
                _json_dump(metadata),
            ),
        )
        conn.executemany(
            "INSERT INTO llm_trace_spans ("
            "id, created_at, llm_trace_id, name, provider, model, duration_ms, input_tokens, "
            "output_tokens, input_payload, output_payload, attributes, status_code, error"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    str(uuid.uuid4()),
                    created_at,
                    trace_id,
                    span.name,
                    span.provider,
                    span.model,
                    span.duration_ms,
                    span.input_tokens,
                    span.output_tokens,
                    _json_dump(span.input_payload),
                    _json_dump(span.output_payload),
                    _json_dump(span.attributes),
                    span.status_code,
                    span.error,
                )
                for span in spans
            ],
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
        trace_rows = conn.execute(
            "SELECT * FROM llm_traces WHERE key_id = ? AND project_id = ? "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (int(key_id), project_id, int(limit)),
        ).fetchall()
        records: list[dict[str, Any]] = []
        for trace_row in trace_rows:
            span_rows = conn.execute(
                "SELECT * FROM llm_trace_spans WHERE llm_trace_id = ? ORDER BY created_at, rowid",
                (trace_row["id"],),
            ).fetchall()
            raw = _raw_trace(trace_row, span_rows)
            if export_format == "raw":
                records.append(raw)
                continue
            input_payload, output_payload = _training_pair(raw["spans"])
            if not _usable_payload(input_payload):
                continue
            if export_format == "prompts":
                records.append({"input": input_payload})
                continue
            if _usable_payload(output_payload):
                records.append({"input": input_payload, "output": output_payload})

    trace_count = len(trace_rows)
    return {
        "records": records,
        "traces": trace_count,
        "skipped": 0 if export_format == "raw" else trace_count - len(records),
        "format": export_format,
    }


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


def _training_pair(spans: list[dict[str, Any]]) -> tuple[Any, Any]:
    input_payload = next(
        (span.get("input_payload") for span in spans if _usable_payload(span.get("input_payload"))),
        None,
    )
    output_payload = next(
        (
            span.get("output_payload")
            for span in reversed(spans)
            if _usable_payload(span.get("output_payload"))
        ),
        None,
    )
    return input_payload, output_payload


def _usable_payload(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _json_dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(sanitize_json_value(value), ensure_ascii=False, sort_keys=True)


def _json_load(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[: limit - 3]}..."

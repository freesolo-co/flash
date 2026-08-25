from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from flash.serving.src.io.schemas import (
    AdapterRecord,
    PersistedAdapterRecord,
    internal_adapter_payload,
)
from flash.serving.src.store.persisted_columns import PERSISTED_COLUMNS
from flash.serving.src.store.settings import ADAPTER_TABLE, Settings
from flash.serving.src.store.supabase_rest import (
    postgrest_error,
    raise_for_supabase,
    supabase_headers,
    supabase_table_url,
)

FOREIGN_KEY_VIOLATION = "23503"
UNIQUE_VIOLATION = "23505"


class PersistenceConflict(RuntimeError):
    """The row already exists; the caller resolves it by reading the winner."""


class PersistenceReferenceError(RuntimeError):
    """A referenced row does not exist. Retrying cannot fix it -- the reference must change."""


class PersistenceRecordError(RuntimeError):
    pass


_ADAPTER_PAGE = 1000
_PERSISTED_ROW_FIELDS = frozenset(PERSISTED_COLUMNS.split(","))


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_adapters(settings: Settings) -> list[AdapterRecord]:
    if not settings.has_supabase:
        return []
    params = {
        "select": PERSISTED_COLUMNS,
        "status": "eq.ready",
        "order": "adapter_id.asc",
        "limit": str(_ADAPTER_PAGE),
    }
    records: list[AdapterRecord] = []
    cursor: str | None = None
    with httpx.Client(timeout=30.0) as client:
        while True:
            page_params = dict(params)
            if cursor is not None:
                page_params["adapter_id"] = f"gt.{cursor}"
            response = client.get(
                supabase_table_url(settings, ADAPTER_TABLE),
                params=page_params,
                headers=supabase_headers(settings, "flash"),
            )
            raise_for_supabase(response, "load hosted LoRA adapters")
            rows = response.json()
            if not isinstance(rows, list):
                raise PersistenceRecordError(
                    "Supabase load hosted LoRA adapters response must be a list"
                )
            records.extend(
                _records_from_response(
                    response,
                    "load hosted LoRA adapters",
                    skip_invalid=True,
                )
            )
            if len(rows) < _ADAPTER_PAGE:
                break
            last_row = rows[-1]
            if not isinstance(last_row, dict) or not isinstance(last_row.get("adapter_id"), str):
                raise PersistenceRecordError("Supabase adapter page has no cursor authority")
            cursor = last_row["adapter_id"]
    return records


def list_run_adapters(run_id: str, settings: Settings) -> list[AdapterRecord]:
    """load every persisted alias and revision for one run, regardless of lifecycle status."""
    if not settings.has_supabase:
        return []
    params = {
        "select": PERSISTED_COLUMNS,
        "metadata->>run_id": f"eq.{run_id}",
        "order": "adapter_id.asc",
        "limit": str(_ADAPTER_PAGE),
    }
    records: list[AdapterRecord] = []
    cursor: str | None = None
    with httpx.Client(timeout=30.0) as client:
        while True:
            page_params = dict(params)
            if cursor is not None:
                page_params["adapter_id"] = f"gt.{cursor}"
            response = client.get(
                supabase_table_url(settings, ADAPTER_TABLE),
                params=page_params,
                headers=supabase_headers(settings, "flash"),
            )
            raise_for_supabase(response, "load hosted LoRA run adapters")
            rows = response.json()
            if not isinstance(rows, list):
                raise PersistenceRecordError(
                    "Supabase load hosted LoRA run adapters response must be a list"
                )
            records.extend(_records_from_response(response, "load hosted LoRA run adapters"))
            if len(rows) < _ADAPTER_PAGE:
                break
            last_row = rows[-1]
            if not isinstance(last_row, dict) or not isinstance(last_row.get("adapter_id"), str):
                raise PersistenceRecordError("Supabase run adapter page has no cursor authority")
            cursor = last_row["adapter_id"]
    return records


def get_adapter(adapter_id: str, settings: Settings) -> AdapterRecord | None:
    if not settings.has_supabase:
        return None
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            supabase_table_url(settings, ADAPTER_TABLE),
            params={
                "select": PERSISTED_COLUMNS,
                "adapter_id": f"eq.{adapter_id}",
                "limit": "1",
            },
            headers=supabase_headers(settings, "flash"),
        )
    raise_for_supabase(response, "read hosted LoRA adapter")
    records = _records_from_response(response, "read hosted LoRA adapter")
    return records[0] if records else None


def insert_adapter(record: AdapterRecord, settings: Settings) -> AdapterRecord:
    now = utc_now_iso()
    if not settings.has_supabase:
        return PersistedAdapterRecord.model_validate(
            {
                **internal_adapter_payload(record),
                "created_at": record.created_at or now,
                "updated_at": record.updated_at or now,
                "deployment_generation": record.deployment_generation,
            }
        )
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            supabase_table_url(settings, ADAPTER_TABLE),
            params={"select": PERSISTED_COLUMNS},
            headers={**supabase_headers(settings, "flash"), "Prefer": "return=representation"},
            json=_adapter_to_row(record, now),
        )
    if response.status_code == 409:
        # postgrest returns 409 for BOTH a unique violation and a foreign key violation. only the
        # first means "someone else won the insert"; the second means org_id names an org that does
        # not exist, which no amount of reading back or retrying will resolve.
        code, detail = postgrest_error(response)
        if code == FOREIGN_KEY_VIOLATION:
            raise PersistenceReferenceError(
                f"hosted LoRA adapter references a row that does not exist "
                f"(org_id={record.org_id!r}): {detail or 'foreign key violation'}"
            )
        if code == UNIQUE_VIOLATION:
            raise PersistenceConflict("adapter row already exists")
    raise_for_supabase(response, "insert hosted LoRA adapter")
    records = _records_from_response(response, "insert hosted LoRA adapter")
    if len(records) != 1:
        raise RuntimeError("insert hosted LoRA adapter must return exactly one row")
    return records[0]


def replace_adapter_cas(
    record: AdapterRecord,
    *,
    expected_updated_at: str,
    settings: Settings,
) -> AdapterRecord | None:
    now = utc_now_iso()
    if not settings.has_supabase:
        return PersistedAdapterRecord.model_validate(
            {
                **internal_adapter_payload(record),
                "updated_at": now,
                "deployment_generation": record.deployment_generation,
            }
        )
    with httpx.Client(timeout=30.0) as client:
        response = client.patch(
            supabase_table_url(settings, ADAPTER_TABLE),
            params={
                "select": PERSISTED_COLUMNS,
                "adapter_id": f"eq.{record.adapter_id}",
                "updated_at": f"eq.{expected_updated_at}",
            },
            headers={**supabase_headers(settings, "flash"), "Prefer": "return=representation"},
            json=_adapter_to_row(record, now),
        )
    raise_for_supabase(response, "conditionally replace hosted LoRA adapter")
    records = _records_from_response(response, "conditionally replace hosted LoRA adapter")
    if len(records) > 1:
        raise RuntimeError("conditional adapter replacement returned more than one row")
    return records[0] if records else None


def _records_from_response(
    response: httpx.Response,
    operation: str,
    *,
    skip_invalid: bool = False,
) -> list[AdapterRecord]:
    rows = response.json()
    if not isinstance(rows, list):
        raise PersistenceRecordError(f"Supabase {operation} response must be a list")
    records: list[AdapterRecord] = []
    for row in rows:
        try:
            records.append(PersistedAdapterRecord.model_validate(_row_to_adapter(row)))
        except (RuntimeError, ValueError) as exc:
            if skip_invalid:
                continue
            raise PersistenceRecordError(
                f"Supabase {operation} returned an invalid adapter row"
            ) from exc
    return records


def _required_row_value(row: dict[str, Any], key: str) -> Any:
    if key not in row or row[key] is None:
        adapter_id = row.get("adapter_id") or "<unknown>"
        raise RuntimeError(f"Supabase adapter row {adapter_id!r} must include {key}")
    return row[key]


def _row_to_adapter(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise RuntimeError("Supabase adapter row must be an object")
    extra_fields = set(row) - _PERSISTED_ROW_FIELDS
    if extra_fields:
        adapter_id = row.get("adapter_id") or "<unknown>"
        extras = ", ".join(sorted(extra_fields))
        raise RuntimeError(f"Supabase adapter row {adapter_id!r} has unsupported fields: {extras}")
    raw_metadata = _required_row_value(row, "metadata")
    if not isinstance(raw_metadata, dict):
        adapter_id = row.get("adapter_id") or "<unknown>"
        raise RuntimeError(f"Supabase adapter row {adapter_id!r} metadata must be an object")
    metadata = dict(raw_metadata)
    raw_thinking = metadata.pop("thinking", None)
    if not isinstance(raw_thinking, bool):
        adapter_id = row.get("adapter_id") or "<unknown>"
        raise RuntimeError(
            f"Supabase adapter row {adapter_id!r} must include boolean metadata.thinking"
        )
    raw_structured_outputs = metadata.pop("structured_outputs", None)
    return {
        "adapter_id": _required_row_value(row, "adapter_id"),
        "repo_id": _required_row_value(row, "repo_id"),
        "org_id": _required_row_value(row, "org_id"),
        "url": row.get("url"),
        "base_model": _required_row_value(row, "base_model"),
        "subfolder": row.get("subfolder"),
        "repo_type": _required_row_value(row, "repo_type"),
        "checkpoint": row.get("checkpoint"),
        "private": _required_row_value(row, "private"),
        "thinking": raw_thinking,
        "structured_outputs": raw_structured_outputs,
        "status": _required_row_value(row, "status"),
        "metadata": metadata,
        "created_at": row.get("created_at"),
        "updated_at": _required_row_value(row, "updated_at"),
        "deployment_generation": row.get("deployment_generation"),
    }


def _adapter_to_row(record: AdapterRecord, now: str) -> dict[str, Any]:
    persisted = PersistedAdapterRecord.model_validate(
        {
            **internal_adapter_payload(record),
            "updated_at": record.updated_at or now,
            "deployment_generation": record.deployment_generation,
        }
    )
    metadata: dict[str, Any] = {**persisted.metadata, "thinking": persisted.thinking}
    if persisted.structured_outputs is not None:
        metadata["structured_outputs"] = persisted.structured_outputs
    return {
        "adapter_id": persisted.adapter_id,
        "repo_id": persisted.repo_id,
        "org_id": persisted.org_id,
        "url": persisted.url,
        "base_model": persisted.base_model,
        "subfolder": persisted.subfolder,
        "repo_type": persisted.repo_type,
        "checkpoint": persisted.checkpoint,
        "private": persisted.private,
        "status": persisted.status,
        "metadata": metadata,
        "created_at": persisted.created_at or now,
        "updated_at": now,
        "deployment_generation": persisted.deployment_generation,
    }

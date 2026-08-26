from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from flash.schema import format_checkpoint_ref, parse_checkpoint_ref
from flash.serving.src.io.schemas import AdapterRecord, PersistedAdapterRecord
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
    """the checkpoint binding already exists with different immutable facts."""


class PersistenceReferenceError(RuntimeError):
    """a referenced row does not exist."""


class PersistenceRecordError(RuntimeError):
    pass


_ADAPTER_PAGE = 1000
_PERSISTED_ROW_FIELDS = frozenset(PERSISTED_COLUMNS.split(","))


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_adapters(settings: Settings) -> list[AdapterRecord]:
    if not settings.has_supabase:
        return []
    return _list_adapters(
        settings,
        {"status": "eq.ready", "order": "org_id.asc,checkpoint_id.asc"},
        "load hosted checkpoints",
        skip_invalid=True,
    )


def list_run_adapters(org_id: str, run_id: str, settings: Settings) -> list[AdapterRecord]:
    """load every exact checkpoint for one tenant run, regardless of lifecycle status."""

    if not settings.has_supabase:
        return []
    return _list_adapters(
        settings,
        {
            "org_id": f"eq.{org_id}",
            "run_id": f"eq.{run_id}",
            "order": "checkpoint_id.asc",
        },
        "load hosted run checkpoints",
    )


def _list_adapters(
    settings: Settings,
    filters: dict[str, str],
    operation: str,
    *,
    skip_invalid: bool = False,
) -> list[AdapterRecord]:
    params = {"select": PERSISTED_COLUMNS, "limit": str(_ADAPTER_PAGE), **filters}
    records: list[AdapterRecord] = []
    offset = 0
    with httpx.Client(timeout=30.0) as client:
        while True:
            page_params = {**params, "offset": str(offset)}
            response = client.get(
                supabase_table_url(settings, ADAPTER_TABLE),
                params=page_params,
                headers=supabase_headers(settings, "flash"),
            )
            raise_for_supabase(response, operation)
            rows = response.json()
            if not isinstance(rows, list):
                raise PersistenceRecordError(f"Supabase {operation} response must be a list")
            records.extend(_records_from_response(response, operation, skip_invalid=skip_invalid))
            if len(rows) < _ADAPTER_PAGE:
                return records
            offset += len(rows)


def get_adapter(org_id: str, adapter_id: str, settings: Settings) -> AdapterRecord | None:
    if not settings.has_supabase:
        return None
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            supabase_table_url(settings, ADAPTER_TABLE),
            params={
                "select": PERSISTED_COLUMNS,
                "org_id": f"eq.{org_id}",
                "checkpoint_id": f"eq.{adapter_id}",
                "limit": "1",
            },
            headers=supabase_headers(settings, "flash"),
        )
    raise_for_supabase(response, "read hosted checkpoint")
    records = _records_from_response(response, "read hosted checkpoint")
    return records[0] if records else None


def insert_adapter(record: AdapterRecord, settings: Settings) -> AdapterRecord:
    now = utc_now_iso()
    if not settings.has_supabase:
        return PersistedAdapterRecord.model_validate(
            {
                **record.model_dump(mode="json"),
                "org_id": record.org_id,
                "created_at": record.created_at or now,
                "updated_at": record.updated_at or now,
                "deployment_generation": record.deployment_generation,
                "run_id": record.run_id,
                "checkpoint_step": record.checkpoint_step,
                "artifact_revision": record.artifact_revision,
                "artifact_digest": record.artifact_digest,
                "artifact_fingerprint": record.artifact_fingerprint,
                "lora_rank": record.lora_rank,
            }
        )
    payload = _bind_payload(record)
    url = f"{str(settings.supabase_url).rstrip('/')}/rest/v1/rpc/bind_hosted_checkpoint"
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            url,
            headers=supabase_headers(settings, "flash"),
            json=payload,
        )
    if response.status_code == 409:
        code, detail = postgrest_error(response)
        if code == FOREIGN_KEY_VIOLATION:
            raise PersistenceReferenceError(detail or "hosted checkpoint references a missing row")
        if code == UNIQUE_VIOLATION:
            raise PersistenceConflict(detail or "immutable checkpoint binding conflict")
    raise_for_supabase(response, "bind hosted checkpoint")
    records = _records_from_response(response, "bind hosted checkpoint")
    if len(records) != 1:
        raise PersistenceRecordError("bind hosted checkpoint must return exactly one row")
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
                **record.model_dump(mode="json"),
                "org_id": record.org_id,
                "updated_at": now,
                "deployment_generation": record.deployment_generation,
                "run_id": record.run_id,
                "checkpoint_step": record.checkpoint_step,
                "artifact_revision": record.artifact_revision,
                "artifact_digest": record.artifact_digest,
                "artifact_fingerprint": record.artifact_fingerprint,
                "lora_rank": record.lora_rank,
            }
        )
    with httpx.Client(timeout=30.0) as client:
        response = client.patch(
            supabase_table_url(settings, ADAPTER_TABLE),
            params={
                "select": PERSISTED_COLUMNS,
                "org_id": f"eq.{record.org_id}",
                "checkpoint_id": f"eq.{record.adapter_id}",
                "updated_at": f"eq.{expected_updated_at}",
            },
            headers={**supabase_headers(settings, "flash"), "Prefer": "return=representation"},
            json={
                "status": record.status,
                "url": record.url,
                "deployment_generation": record.deployment_generation,
            },
        )
    raise_for_supabase(response, "conditionally update hosted checkpoint lifecycle")
    records = _records_from_response(response, "conditionally update hosted checkpoint lifecycle")
    if len(records) > 1:
        raise PersistenceRecordError("conditional checkpoint update returned more than one row")
    return records[0] if records else None


def _adapter_to_row(record: AdapterRecord, now: str) -> dict[str, Any]:
    """serialize one exact checkpoint to the explicit hosted binding schema."""

    if record.run_id is None or record.checkpoint is None:
        raise PersistenceRecordError("checkpoint record is missing its permanent identity")
    parsed = parse_checkpoint_ref(record.checkpoint)
    if parsed is None or parsed[0] != record.run_id:
        raise PersistenceRecordError("checkpoint record has an invalid permanent identity")
    return {
        "org_id": record.org_id,
        "run_id": record.run_id,
        "checkpoint": "final" if parsed[1] is None else f"step-{parsed[1]}",
        "checkpoint_id": record.adapter_id,
        "source_repo_type": record.repo_type,
        "source_repository": record.repo_id,
        "source_revision": record.artifact_revision,
        "source_subfolder": record.subfolder,
        "artifact_digest": record.artifact_digest,
        "artifact_fingerprint": record.artifact_fingerprint,
        "base_model": record.base_model,
        "lora_config": {"rank": record.lora_rank},
        "serving_defaults": {
            "thinking": record.thinking,
            "structured_outputs": record.structured_outputs,
        },
        "url": record.url,
        "status": record.status,
        "deployment_generation": record.deployment_generation,
        "created_at": record.created_at or now,
        "updated_at": now,
    }


def _bind_payload(record: AdapterRecord) -> dict[str, Any]:
    if record.org_id is None or record.run_id is None:
        raise PersistenceRecordError("checkpoint registration requires org_id and run_id")
    return {
        "p_org_id": record.org_id,
        "p_run_id": record.run_id,
        "p_checkpoint_id": record.adapter_id,
        "p_source_repo_type": record.repo_type,
        "p_source_repository": record.repo_id,
        "p_source_revision": record.artifact_revision,
        "p_source_subfolder": record.subfolder,
        "p_artifact_digest": record.artifact_digest,
        "p_artifact_fingerprint": record.artifact_fingerprint,
        "p_base_model": record.base_model,
        "p_lora_config": {"rank": record.lora_rank},
        "p_serving_defaults": {
            "thinking": record.thinking,
            "structured_outputs": record.structured_outputs,
        },
    }


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
                f"Supabase {operation} returned an invalid checkpoint row"
            ) from exc
    return records


def _required_row_value(row: dict[str, Any], key: str) -> Any:
    if key not in row or row[key] is None:
        checkpoint_id = row.get("checkpoint_id") or "<unknown>"
        raise RuntimeError(f"Supabase checkpoint row {checkpoint_id!r} must include {key}")
    return row[key]


def _row_to_adapter(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise RuntimeError("Supabase checkpoint row must be an object")
    extra_fields = set(row) - _PERSISTED_ROW_FIELDS
    if extra_fields:
        extras = ", ".join(sorted(extra_fields))
        raise RuntimeError(f"Supabase checkpoint row has unsupported fields: {extras}")
    run_id = _required_row_value(row, "run_id")
    checkpoint_id = _required_row_value(row, "checkpoint_id")
    parsed = parse_checkpoint_ref(checkpoint_id)
    if parsed is None or parsed[0] != run_id:
        raise RuntimeError("Supabase checkpoint row has inconsistent checkpoint identity")
    if row.get("checkpoint") != ("final" if parsed[1] is None else f"step-{parsed[1]}"):
        raise RuntimeError("Supabase checkpoint row has inconsistent checkpoint selector")
    lora_config = _required_row_value(row, "lora_config")
    defaults = _required_row_value(row, "serving_defaults")
    if not isinstance(lora_config, dict) or not isinstance(defaults, dict):
        raise RuntimeError("Supabase checkpoint row config fields must be objects")
    return {
        "adapter_id": checkpoint_id,
        "repo_id": _required_row_value(row, "source_repository"),
        "org_id": _required_row_value(row, "org_id"),
        "url": row.get("url"),
        "base_model": _required_row_value(row, "base_model"),
        "subfolder": row.get("source_subfolder"),
        "repo_type": _required_row_value(row, "source_repo_type"),
        "checkpoint": format_checkpoint_ref(run_id, parsed[1]),
        "private": True,
        "thinking": defaults.get("thinking"),
        "structured_outputs": defaults.get("structured_outputs"),
        "status": _required_row_value(row, "status"),
        "created_at": row.get("created_at"),
        "updated_at": _required_row_value(row, "updated_at"),
        "deployment_generation": row.get("deployment_generation"),
        "run_id": run_id,
        "checkpoint_step": parsed[1],
        "artifact_revision": _required_row_value(row, "source_revision"),
        "artifact_digest": _required_row_value(row, "artifact_digest"),
        "artifact_fingerprint": _required_row_value(row, "artifact_fingerprint"),
        "lora_rank": lora_config.get("rank"),
    }

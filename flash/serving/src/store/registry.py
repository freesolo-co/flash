from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime
from pathlib import Path

from flash.serve.contract.provenance import CheckpointKey, checkpoint_key, record_key
from flash.serving.src.io.schemas import AdapterRecord

RegistryKey = tuple[str | None, str]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _download_ident(record: AdapterRecord) -> tuple[str, str, str, str | None]:
    return (record.repo_id, record.repo_type, record.artifact_revision or "", record.subfolder)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _record_key(record: AdapterRecord) -> RegistryKey:
    return (None, record.adapter_id) if record.serve_base_model else record_key(record)


def _lookup_key(org_id: str | None, adapter_id: str) -> RegistryKey:
    return (None, adapter_id) if org_id is None else checkpoint_key(org_id, adapter_id)


class AdapterRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[RegistryKey, AdapterRecord] = {}
        self._local_paths: dict[CheckpointKey, Path] = {}
        self._local_idents: dict[CheckpointKey, tuple[str, str, str, str | None]] = {}
        self._tombstones: dict[RegistryKey, str | None] = {}

    def _tombstone_blocks(self, key: RegistryKey, record: AdapterRecord) -> bool:
        if key not in self._tombstones:
            return False
        tombstoned_at = self._tombstones[key]
        record_at = _parse_iso(record.updated_at)
        tombstone_at = _parse_iso(tombstoned_at)
        if record_at is not None and tombstone_at is not None and record_at > tombstone_at:
            del self._tombstones[key]
            return False
        return True

    def hydrate(self, records: list[AdapterRecord]) -> None:
        with self._lock:
            self._records = {
                key: record
                for record in records
                if not self._tombstone_blocks((key := _record_key(record)), record)
            }

    def upsert(self, record: AdapterRecord, *, revive: bool = False) -> AdapterRecord:
        key = _record_key(record)
        with self._lock:
            if key in self._tombstones:
                if revive:
                    self._tombstones.pop(key, None)
                elif self._tombstone_blocks(key, record):
                    return record
            existing = self._records.get(key)
            if existing and not record.created_at:
                record = record.model_copy(update={"created_at": existing.created_at})
            if revive and _parse_iso(record.updated_at) is None:
                record = record.model_copy(update={"updated_at": _utc_now_iso()})
            if existing is not None and not revive:
                incoming_at = _parse_iso(record.updated_at)
                existing_at = _parse_iso(existing.updated_at)
                if (
                    incoming_at is not None
                    and existing_at is not None
                    and incoming_at < existing_at
                ):
                    return existing
            self._records[key] = record
            return record

    def get(self, org_id: str | None, adapter_id: str) -> AdapterRecord | None:
        with self._lock:
            return self._records.get(_lookup_key(org_id, adapter_id))

    def has(self, org_id: str | None, adapter_id: str) -> bool:
        with self._lock:
            return _lookup_key(org_id, adapter_id) in self._records

    def list_ready(self, *, org_id: str | None = None) -> list[AdapterRecord]:
        with self._lock:
            return sorted(
                (
                    record
                    for record in self._records.values()
                    if record.status == "ready" and (org_id is None or record.org_id == org_id)
                ),
                key=lambda record: (record.org_id or "", record.adapter_id),
            )

    def remove(self, org_id: str | None, adapter_id: str) -> AdapterRecord | None:
        key = _lookup_key(org_id, adapter_id)
        with self._lock:
            removed = self._records.pop(key, None)
            if removed is not None:
                self._tombstones[key] = removed.updated_at or _utc_now_iso()
            else:
                self._tombstones[key] = self._tombstones.get(key) or _utc_now_iso()
            if org_id is not None:
                checkpoint = checkpoint_key(org_id, adapter_id)
                self._local_paths.pop(checkpoint, None)
                self._local_idents.pop(checkpoint, None)
            return removed

    def local_path_is_stale(self, record: AdapterRecord) -> bool:
        if not record.is_checkpoint:
            return False
        key = record_key(record)
        with self._lock:
            return key in self._local_paths and self._local_idents.get(key) != _download_ident(
                record
            )

    def set_local_path(self, record: AdapterRecord, path: Path) -> None:
        if not record.is_checkpoint:
            raise ValueError("base-model records cannot acquire local adapter paths")
        key = record_key(record)
        with self._lock:
            self._local_paths[key] = path
            self._local_idents[key] = _download_ident(record)

    def local_path(self, record: AdapterRecord) -> Path | None:
        if not record.is_checkpoint:
            return None
        key = record_key(record)
        with self._lock:
            path = self._local_paths.get(key)
            if path is None:
                return None
            if self._local_idents.get(key) != _download_ident(record):
                self._local_paths.pop(key, None)
                self._local_idents.pop(key, None)
                return None
            return path


def lora_int_id(adapter_name: str) -> int:
    digest = hashlib.sha1(adapter_name.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") & 0x7FFFFFFF) or 1

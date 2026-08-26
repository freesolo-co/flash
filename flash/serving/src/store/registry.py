from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime
from pathlib import Path

from flash.serving.src.io.schemas import AdapterRecord


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _download_ident(record: AdapterRecord) -> tuple[str, str, str, str | None]:
    return (record.repo_id, record.repo_type, record.hf_revision or "", record.subfolder)


def _parse_iso(value: str | None) -> datetime | None:
    # Tolerant ISO-8601 parse (PostgREST's trailing Z vs isoformat()'s +00:00); naive -> UTC.
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


class AdapterRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, AdapterRecord] = {}
        self._local_paths: dict[str, Path] = {}
        self._local_idents: dict[str, tuple[str, str, str, str | None]] = {}
        # Tombstones: ids undeployed locally, mapped to the undeploy-time updated_at. They make
        # undeploy win over re-add paths that don't intend a redeploy (a stale status=ready row
        # from hydrate, or a lazy-load upsert forward). Cleared by an explicit redeploy
        # (revive=True) or a durably newer record, so undeploy is durable but not permanent.
        self._tombstones: dict[str, str | None] = {}

    def _tombstone_blocks(self, record: AdapterRecord) -> bool:
        # Caller holds self._lock. A record durably newer than the tombstone expires it.
        if record.adapter_id not in self._tombstones:
            return False
        tombstoned_at = self._tombstones[record.adapter_id]
        # Compare instants, not raw strings (mixed ISO renderings); can't prove newer -> stay blocked.
        record_at = _parse_iso(record.updated_at)
        tombstone_at = _parse_iso(tombstoned_at)
        if record_at is not None and tombstone_at is not None and record_at > tombstone_at:
            del self._tombstones[record.adapter_id]
            return False
        return True

    def hydrate(self, records: list[AdapterRecord]) -> None:
        with self._lock:
            # Drop records still suppressed by a local tombstone (undeployed here, not yet
            # flipped to disabled in shared storage) unless durably newer.
            self._records = {
                record.adapter_id: record
                for record in records
                if not self._tombstone_blocks(record)
            }

    def upsert(self, record: AdapterRecord, *, revive: bool = False) -> AdapterRecord:
        with self._lock:
            if record.adapter_id in self._tombstones:
                if revive:  # explicit (re)deploy wins even without a bumped timestamp
                    self._tombstones.pop(record.adapter_id, None)
                elif self._tombstone_blocks(record):
                    # Lazy-load / stale forward must not resurrect a locally undeployed adapter.
                    return record
            existing = self._records.get(record.adapter_id)
            if existing and not record.created_at:
                record = record.model_copy(update={"created_at": existing.created_at})
            if revive and _parse_iso(record.updated_at) is None:
                # An explicit (re)deploy must store a parseable, fresh ``updated_at`` so the
                # do-not-downgrade guard below can later reject a stale router's forward of an OLDER
                # record. ``persist_adapter`` stamps the DB row with its own ``now`` but never copies
                # it back into this record, so a redeploy whose POST body carried no timestamp would
                # otherwise be stored with ``updated_at=None`` — and a stale older forward (parseable
                # ts, but ``existing_at`` None) would slip past the guard and re-download stale weights.
                record = record.model_copy(update={"updated_at": _utc_now_iso()})
            # Do-not-downgrade guard. Modal runs several router containers; for up to
            # RELOAD_INTERVAL_SECONDS after a redeploy that reuses an id with a NEW source, a stale
            # router still holds the OLD record and forwards it to the engine via _lora_request's
            # lazy upsert (revive=False). Without an ordering check that older record would
            # overwrite the engine's NEWER one here, then local_path_is_stale() would evict the
            # freshly-loaded LoRA and re-download the OLD source — silently reverting a redeployed
            # adapter to stale weights (and flapping as fresh/stale routers alternate). An explicit
            # (re)deploy (revive=True) always wins; otherwise keep the existing record when the
            # incoming one is strictly older. Unknown/equal timestamps fall through (overwrite) so
            # legitimate lazy-load forwards of the same-or-newer record still win.
            if existing is not None and not revive:
                incoming_at = _parse_iso(record.updated_at)
                existing_at = _parse_iso(existing.updated_at)
                if (
                    incoming_at is not None
                    and existing_at is not None
                    and incoming_at < existing_at
                ):
                    return existing
            # The cached path is NOT invalidated here: local_path() catches an identity change
            # lazily so _ensure_adapter_local_locked can evict the old vLLM LoRA AND re-download.
            self._records[record.adapter_id] = record
            return record

    def get(self, adapter_id: str) -> AdapterRecord | None:
        with self._lock:
            return self._records.get(adapter_id)

    def has(self, adapter_id: str) -> bool:
        with self._lock:
            return adapter_id in self._records

    def list_ready(self) -> list[AdapterRecord]:
        with self._lock:
            return sorted(
                (record for record in self._records.values() if record.status == "ready"),
                key=lambda record: record.adapter_id,
            )

    def remove(self, adapter_id: str) -> AdapterRecord | None:
        with self._lock:
            removed = self._records.pop(adapter_id, None)
            # Tombstone at the undeploy-time updated_at (fall back to now: a None tombstone would
            # be non-durable). If we never held the record, keep any existing tombstone time.
            if removed is not None:
                self._tombstones[adapter_id] = removed.updated_at or _utc_now_iso()
            else:
                self._tombstones[adapter_id] = self._tombstones.get(adapter_id) or _utc_now_iso()
            self._local_paths.pop(adapter_id, None)
            self._local_idents.pop(adapter_id, None)
            return removed

    def discard_cached(self, adapter_id: str) -> AdapterRecord | None:
        """drop untrusted cached state without creating a durable undeploy tombstone."""
        with self._lock:
            removed = self._records.pop(adapter_id, None)
            self._local_paths.pop(adapter_id, None)
            self._local_idents.pop(adapter_id, None)
            return removed

    def local_path_is_stale(self, record: AdapterRecord) -> bool:
        if record.is_alias:
            return False
        # Cached path exists but was downloaded for a different repo/subfolder/repo_type — lets
        # the engine evict the old vLLM LoRA before re-downloading. Fresh (uncached) -> False.
        with self._lock:
            adapter_id = record.adapter_id
            if adapter_id not in self._local_paths:
                return False
            return self._local_idents.get(adapter_id) != _download_ident(record)

    def set_local_path(self, record: AdapterRecord, path: Path) -> None:
        if record.is_alias:
            raise ValueError("alias records cannot acquire local adapter paths")
        with self._lock:
            self._local_paths[record.adapter_id] = path
            self._local_idents[record.adapter_id] = _download_ident(record)

    def clear_local_path(self, adapter_id: str) -> None:
        with self._lock:
            self._local_paths.pop(adapter_id, None)
            self._local_idents.pop(adapter_id, None)

    def local_path(self, record: AdapterRecord) -> Path | None:
        if record.is_alias:
            return None
        # Cached path only if downloaded for the SAME identity; a changed source drops the stale
        # entry and returns None so the caller re-downloads instead of serving stale weights.
        with self._lock:
            adapter_id = record.adapter_id
            path = self._local_paths.get(adapter_id)
            if path is None:
                return None
            if self._local_idents.get(adapter_id) != _download_ident(record):
                self._local_paths.pop(adapter_id, None)
                self._local_idents.pop(adapter_id, None)
                return None
            return path


def lora_int_id(adapter_id: str) -> int:
    # Mask to 31 bits: vLLM stores lora_int_id in an int32 array, so a full uint32 hash overflows.
    digest = hashlib.sha1(adapter_id.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") & 0x7FFFFFFF) or 1

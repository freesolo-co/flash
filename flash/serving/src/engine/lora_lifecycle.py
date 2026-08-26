"""LoRA reservation, source-cache, and request lifecycle ownership."""

from __future__ import annotations

import asyncio
import inspect
import shutil
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from flash.serving.src.engine.support import (
    _adapter_cache_path,
    _adapter_cache_ready,
    _adapter_snapshot_size,
    _adapter_source_cache_dir,
    _adapter_source_ident,
    _assert_source_cache_containment,
    _materialize_adapter_snapshot,
    _replica_adapter_cache_dir,
)

SourceIdent = tuple[str, str, str, str | None]


class AdapterCacheCapacityError(RuntimeError):
    """the replica cannot materialize another source without deleting a live source."""


@dataclass(slots=True)
class _SourceState:
    directory: Path
    size_bytes: int
    last_used: float
    reserved_bytes: int = 0
    materializing: int = 0
    active: int = 0
    current: set[str] = field(default_factory=set)
    loaded: set[str] = field(default_factory=set)

    @property
    def protected(self) -> bool:
        return bool(self.materializing or self.active or self.current or self.loaded)

    @property
    def accounted_bytes(self) -> int:
        return max(self.size_bytes, self.reserved_bytes)


class ReplicaSourceCache:
    """track source leases and reclaim only inactive directories under one replica root."""

    def __init__(self, root: Path, max_bytes: int) -> None:
        self._root = root.resolve()
        self._max_bytes = max_bytes
        self._states: dict[SourceIdent, _SourceState] = {}
        self._current: dict[str, SourceIdent] = {}
        self._lock = asyncio.Lock()

    def _contained(self, directory: Path) -> Path:
        resolved = directory.resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise ValueError("adapter source is outside this replica cache") from exc
        return resolved

    @staticmethod
    def _directory_size(directory: Path) -> int:
        total = 0
        for path in directory.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def _state(self, ident: SourceIdent, directory: Path) -> _SourceState:
        state = self._states.get(ident)
        if state is None:
            state = _SourceState(directory, 0, time.monotonic())
            self._states[ident] = state
        else:
            state.directory = directory
            state.last_used = time.monotonic()
        return state

    async def require_replaceable(self, ident: SourceIdent) -> None:
        async with self._lock:
            state = self._states.get(ident)
            if state is not None and (state.active or state.current or state.loaded):
                raise AdapterCacheCapacityError("live adapter source cannot be replaced")

    @asynccontextmanager
    async def materializing(
        self,
        ident: SourceIdent,
        directory: Path,
        expected_bytes: int,
    ) -> AsyncIterator[None]:
        directory = self._contained(directory)
        if expected_bytes < 0:
            raise ValueError("adapter materialization reservation must be non-negative")
        existing_bytes = (
            await asyncio.to_thread(self._directory_size, directory) if directory.exists() else 0
        )
        async with self._lock:
            state = self._state(ident, directory)
            if state.materializing:
                raise RuntimeError("adapter source is already materializing")
            previous_reservation = state.reserved_bytes
            state.size_bytes = existing_bytes
            state.reserved_bytes = existing_bytes + expected_bytes
            state.materializing += 1
            await self._reclaim_locked(exclude={ident})
            if self._total_bytes() > self._max_bytes:
                state.materializing -= 1
                state.reserved_bytes = previous_reservation
                if not state.protected:
                    await asyncio.to_thread(shutil.rmtree, state.directory, True)
                    self._states.pop(ident, None)
                raise AdapterCacheCapacityError(
                    "replica adapter cache byte ceiling is occupied by live sources"
                )
        completed = False
        try:
            yield
            completed = True
        finally:
            async with self._lock:
                state = self._states.get(ident)
                if state is not None:
                    state.materializing -= 1
                    state.reserved_bytes = previous_reservation
                    state.last_used = time.monotonic()
                    if not completed and not state.protected:
                        await asyncio.to_thread(shutil.rmtree, state.directory, True)
                        self._states.pop(ident, None)
                    else:
                        await self._reclaim_locked()

    async def reconcile_materialization(
        self,
        ident: SourceIdent,
        directory: Path,
    ) -> None:
        directory = self._contained(directory)
        size_bytes = await asyncio.to_thread(self._directory_size, directory)
        async with self._lock:
            state = self._states.get(ident)
            if state is None or not state.materializing:
                raise RuntimeError("adapter materialization reservation is missing")
            state.size_bytes = size_bytes
            state.reserved_bytes = size_bytes
            state.last_used = time.monotonic()
            await self._reclaim_locked(exclude={ident})
            if self._total_bytes() > self._max_bytes:
                raise AdapterCacheCapacityError(
                    "adapter materialization exceeded its replica cache reservation"
                )

    async def bind_current(
        self,
        adapter_id: str,
        ident: SourceIdent,
        directory: Path,
    ) -> SourceIdent | None:
        directory = self._contained(directory)
        size_bytes = await asyncio.to_thread(self._directory_size, directory)
        async with self._lock:
            previous = self._current.get(adapter_id)
            state = self._states.get(ident)
            created = state is None
            if state is None:
                state = self._state(ident, directory)
            previous_directory = state.directory
            previous_size = state.size_bytes
            state.directory = directory
            state.size_bytes = size_bytes
            state.last_used = time.monotonic()
            await self._reclaim_locked(exclude={ident, previous} - {None})
            if self._total_bytes() > self._max_bytes:
                state.directory = previous_directory
                state.size_bytes = previous_size
                if created and not state.protected:
                    await asyncio.to_thread(shutil.rmtree, directory, True)
                    self._states.pop(ident, None)
                raise AdapterCacheCapacityError(
                    "replica adapter cache byte ceiling is occupied by live sources"
                )
            state.current.add(adapter_id)
            self._current[adapter_id] = ident
            if previous is not None and previous != ident:
                old = self._states.get(previous)
                if old is not None:
                    old.current.discard(adapter_id)
                    old.last_used = time.monotonic()
            return previous

    async def restore_current(
        self,
        adapter_id: str,
        previous: SourceIdent | None,
        failed: SourceIdent,
    ) -> None:
        async with self._lock:
            if (state := self._states.get(failed)) is not None:
                state.current.discard(adapter_id)
                state.last_used = time.monotonic()
            if previous is None:
                self._current.pop(adapter_id, None)
            else:
                self._current[adapter_id] = previous
                if old := self._states.get(previous):
                    old.current.add(adapter_id)
                    old.last_used = time.monotonic()
            await self._reclaim_locked()

    async def release_current(self, adapter_id: str) -> None:
        async with self._lock:
            ident = self._current.pop(adapter_id, None)
            if ident is not None and (state := self._states.get(ident)) is not None:
                state.current.discard(adapter_id)
                state.last_used = time.monotonic()
            await self._reclaim_locked()

    async def mark_loaded(self, adapter_id: str, ident: SourceIdent, directory: Path) -> None:
        directory = self._contained(directory)
        size_bytes = await asyncio.to_thread(self._directory_size, directory)
        async with self._lock:
            state = self._state(ident, directory)
            state.size_bytes = size_bytes
            state.loaded.add(adapter_id)

    @asynccontextmanager
    async def active(self, ident: SourceIdent, directory: Path) -> AsyncIterator[None]:
        directory = self._contained(directory)
        async with self._lock:
            state = self._states.get(ident)
            if state is None or not directory.exists():
                raise RuntimeError("adapter source is unavailable for generation")
            state.active += 1
            state.last_used = time.monotonic()
        try:
            yield
        finally:
            async with self._lock:
                state = self._states.get(ident)
                if state is not None:
                    state.active -= 1
                    state.last_used = time.monotonic()
                    await self._reclaim_locked()

    async def reclaim(self) -> None:
        async with self._lock:
            await self._reclaim_locked()

    def _total_bytes(self) -> int:
        return sum(state.accounted_bytes for state in self._states.values())

    async def _reclaim_locked(self, *, exclude: set[SourceIdent] | None = None) -> None:
        excluded = exclude or set()
        candidates = sorted(
            (
                (ident, state)
                for ident, state in self._states.items()
                if ident not in excluded and not state.protected
            ),
            key=lambda item: item[1].last_used,
        )
        for ident, state in candidates:
            if self._total_bytes() <= self._max_bytes:
                break
            await asyncio.to_thread(shutil.rmtree, state.directory, True)
            self._states.pop(ident, None)

    async def remove_if_unreferenced(self, ident: SourceIdent) -> None:
        async with self._lock:
            state = self._states.get(ident)
            if state is None or state.protected:
                return
            await asyncio.to_thread(shutil.rmtree, state.directory, True)
            self._states.pop(ident, None)

    async def close(self) -> None:
        root = self._root
        await asyncio.to_thread(shutil.rmtree, root, True)
        async with self._lock:
            self._states.clear()
            self._current.clear()


@dataclass(frozen=True, slots=True)
class _LoraEntry:
    source_ident: tuple[str, str, str, str | None]
    lora_request: Any
    state: Literal["reserved", "loaded", "unconfirmed"]
    in_flight: int = 0
    tombstoned: bool = False


def entries_for(owner: Any) -> dict[str, _LoraEntry]:
    """lazily initialize entries for modal instances whose base initializer never ran."""
    entries = getattr(owner, "_lora_entries", None)
    if entries is None:
        entries = {}
        owner._lora_entries = entries
    return entries


def cached_lora_request(owner: Any, record: Any, path: Path) -> Any:
    source_ident = _adapter_source_ident(record)
    adapter_id = record.adapter_id
    entries = entries_for(owner)
    entry = entries.get(adapter_id)
    if entry is not None:
        if entry.state == "unconfirmed":
            raise RuntimeError("LoRA registration is unconfirmed on this engine")
        if entry.source_ident == source_ident:
            return entry.lora_request
        raise RuntimeError("previous LoRA removal is unconfirmed on this engine")

    from vllm.lora.request import LoRARequest

    from flash.serving.src.store.registry import lora_int_id

    # reserve around int32 collisions, including ids that vllm may retain after failed removal.
    used = {entry.lora_request.lora_int_id for entry in entries.values()}
    int_id = lora_int_id(adapter_id)
    while int_id in used:
        int_id = int_id + 1 if int_id < 0x7FFFFFFF else 1

    request = LoRARequest(adapter_id, int_id, str(path))
    entries[adapter_id] = _LoraEntry(source_ident, request, "reserved")
    return request


class LoraLifecycleMixin:
    def _replica_identifier(self) -> str:
        replica_id = getattr(self, "_replica_id", None)
        if replica_id is None:
            replica_id = uuid.uuid4().hex
            self._replica_id = replica_id
        return replica_id

    def _adapter_materialization_dir(self) -> Path:
        path = getattr(self, "_adapter_cache_dir", None)
        if path is None:
            from flash.serving.src.store.settings import ADAPTER_CACHE_DIR

            path = _replica_adapter_cache_dir(ADAPTER_CACHE_DIR, self._replica_identifier())
            self._adapter_cache_dir = path
        return path

    def _source_cache_manager(self) -> Any:
        manager = getattr(self, "_source_cache", None)
        if manager is None:
            from flash.serving.src.store.settings import ADAPTER_CACHE_MAX_BYTES

            manager = ReplicaSourceCache(
                self._adapter_materialization_dir(),
                ADAPTER_CACHE_MAX_BYTES,
            )
            self._source_cache = manager
        return manager

    async def _adapter_lock(self, adapter_id: str) -> asyncio.Lock:
        async with self._adapter_locks_guard:
            lock = self._adapter_locks.get(adapter_id)
            if lock is None:
                lock = asyncio.Lock()
                self._adapter_locks[adapter_id] = lock
            return lock

    async def _source_lock(self, record: Any) -> asyncio.Lock:
        ident = _adapter_source_ident(record)
        async with self._source_locks_guard:
            lock = self._source_locks.get(ident)
            if lock is None:
                lock = asyncio.Lock()
                self._source_locks[ident] = lock
            return lock

    def _entries(self) -> dict[str, _LoraEntry]:
        return entries_for(self)

    async def _evict_loaded_lora(self, adapter_id: str) -> None:
        entries = self._entries()
        if (entry := entries.get(adapter_id)) is None:
            return
        if entry.state == "reserved" and entry.in_flight == 0:
            entries.pop(adapter_id)
            return
        if not entry.tombstoned:
            entries[adapter_id] = _LoraEntry(
                entry.source_ident,
                entry.lora_request,
                entry.state,
                entry.in_flight,
                True,
            )

    async def _pin_lora(self, lora_request: Any) -> None:
        pin = getattr(self.engine, "pin_lora", None)
        if pin is None:
            return
        result = pin(lora_request.lora_int_id)
        if inspect.isawaitable(result):
            await result

    async def _add_lora_locked(self, record: Any, path: Path) -> None:
        adapter_id = record.adapter_id
        entries = self._entries()
        entry_before = entries.get(adapter_id)
        lora_request = self._cached_lora_request_locked(record, path)
        attempt_entry = entries[adapter_id]
        created_reservation = entry_before is None

        def rollback_owned_reservation() -> None:
            current = entries.get(adapter_id)
            if (
                created_reservation
                and current is attempt_entry
                and current.state == "reserved"
                and current.in_flight == 0
            ):
                entries.pop(adapter_id)

        try:
            added = await self.engine.add_lora(lora_request)
        except Exception:
            rollback_owned_reservation()
            raise
        if added is False and attempt_entry.state == "reserved":
            rollback_owned_reservation()
            raise RuntimeError("vLLM rejected a new LoRA registration")
        entry = entries.get(adapter_id)
        if (
            entry is None
            or entry.source_ident != attempt_entry.source_ident
            or entry.lora_request is not lora_request
        ):
            raise RuntimeError("LoRA registration entry changed during explicit add")
        entries[adapter_id] = _LoraEntry(
            entry.source_ident,
            lora_request,
            "loaded",
            entry.in_flight,
            False,
        )
        manager = getattr(self, "_source_cache", None)
        if manager is not None:
            await manager.mark_loaded(
                adapter_id,
                entry.source_ident,
                path,
            )
        if self._pin_loras:  # capped pools stay unpinned so surplus adapters can lru-swap
            await self._pin_lora(lora_request)

    async def _bind_source_path(
        self, record: Any, path: Path
    ) -> tuple[str, str, str, str | None] | None:
        source_ident = _adapter_source_ident(record)
        previous = await self._source_cache_manager().bind_current(
            record.adapter_id,
            source_ident,
            _adapter_source_cache_dir(self._adapter_materialization_dir(), record),
        )
        self._source_paths[source_ident] = path
        self.registry.set_local_path(record, path)
        return previous

    async def _preload_cached_loras(self) -> None:
        for record in self.registry.list_ready():
            source_ident = _adapter_source_ident(record)
            local_dir = _adapter_source_cache_dir(self._adapter_materialization_dir(), record)
            subfolder = getattr(record, "subfolder", None)
            path = _adapter_cache_path(local_dir, subfolder)
            if not _adapter_cache_ready(path):
                continue
            lock = await self._adapter_lock(record.adapter_id)
            async with lock:
                previous = await self._bind_source_path(record, path)
                try:
                    await self._add_lora_locked(record, path)
                except Exception as exc:  # a bad cached LoRA must not kill startup
                    self.registry.clear_local_path(record.adapter_id)
                    self._source_paths.pop(source_ident, None)
                    await self._source_cache_manager().restore_current(
                        record.adapter_id,
                        previous,
                        source_ident,
                    )
                    await self._source_cache_manager().remove_if_unreferenced(source_ident)
                    print(
                        f"cached LoRA preload skipped for {record.adapter_id}: {exc!r}",
                        flush=True,
                    )

    async def _ensure_adapter_local_locked(self, record: Any) -> Path:
        # Download body; caller must already hold self._adapter_lock(record.adapter_id).
        import anyio
        from huggingface_hub import snapshot_download

        from flash.serving.src.store.settings import HF_HUB_CACHE_DIR

        adapter_id = record.adapter_id
        local_dir = _adapter_source_cache_dir(self._adapter_materialization_dir(), record)
        subfolder = getattr(record, "subfolder", None)
        cached_path = _adapter_cache_path(local_dir, subfolder)
        # Stale cached path (source changed) -> fence the mismatched immutable identity. Physical vllm
        # removal is intentionally deferred because it can race an active generation iterator.
        if self.registry.local_path_is_stale(record):
            await self._evict_loaded_lora(adapter_id)
        path = self.registry.local_path(record)
        if path is not None:
            _assert_source_cache_containment(local_dir, path)
            # this is the steady-state hit for an already-materialized adapter. keep the filesystem
            # validation off the event loop so local disk stalls cannot block co-resident requests.
            if await asyncio.to_thread(_adapter_cache_ready, path):
                return path

        source_ident = _adapter_source_ident(record)
        source_lock = await self._source_lock(record)
        async with source_lock:
            path = self.registry.local_path(record)
            if path is not None:
                _assert_source_cache_containment(local_dir, path)
                if _adapter_cache_ready(path):
                    return path
            if cached := self._source_paths.get(source_ident):
                _assert_source_cache_containment(local_dir, cached)
                if _adapter_cache_ready(cached):
                    await self._bind_source_path(record, cached)
                    return cached
                self._source_paths.pop(source_ident, None)

            manager = self._source_cache_manager()
            local_dir.parent.mkdir(parents=True, exist_ok=True)
            repo_type = getattr(record, "repo_type", "model") or "model"
            allow = [f"{subfolder}/**", f"{subfolder}/*"] if subfolder else None
            if _adapter_cache_ready(cached_path):
                await self._bind_source_path(record, cached_path)
                return cached_path
            if local_dir.exists():
                await manager.require_replaceable(source_ident)

            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    downloaded = await anyio.to_thread.run_sync(
                        lambda: snapshot_download(
                            repo_id=record.repo_id,
                            repo_type=repo_type,
                            revision=record.hf_revision,
                            cache_dir=str(HF_HUB_CACHE_DIR),
                            token=self.settings.hf_api_key,
                            allow_patterns=allow,
                        )
                    )
                    snapshot_root = Path(downloaded)
                    expected_bytes = await asyncio.to_thread(
                        _adapter_snapshot_size,
                        snapshot_root,
                        subfolder,
                    )
                    async with manager.materializing(
                        source_ident,
                        local_dir,
                        expected_bytes,
                    ):
                        path = await asyncio.to_thread(
                            _materialize_adapter_snapshot,
                            snapshot_root,
                            local_dir,
                            subfolder,
                        )
                        await manager.reconcile_materialization(source_ident, local_dir)
                        await self._bind_source_path(record, path)
                    return path
                except AdapterCacheCapacityError:
                    raise
                except Exception as exc:  # Hub/network errors are often transient
                    last_exc = exc
                    if attempt == 2:
                        break
                    await asyncio.sleep(0.5 * (2**attempt))

            assert last_exc is not None
            raise last_exc

    def _cached_lora_request_locked(self, record: Any, path: Path) -> Any:
        return cached_lora_request(self, record, path)

    async def _lora_request(
        self, adapter_id: str, record_dict: dict[str, Any] | None = None
    ) -> tuple[Any, Any]:
        """Resolve (LoRARequest, record) for ``adapter_id`` under the adapter lock.

        Returns the RESOLVED record alongside the request so the caller can bind the prompt's
        thinking default to the SAME record the weights came from — a later registry re-read could
        observe a different record after a concurrent same-id redeploy (see
        ``_effective_chat_template_kwargs``).
        """
        from flash.serving.src.io.schemas import AdapterRecord

        # Lock across read + download so a concurrent unregister can't slip in between.
        lock = await self._adapter_lock(adapter_id)
        async with lock:
            forwarded = None
            if record_dict is not None:
                forwarded = AdapterRecord.model_validate(record_dict)
                if forwarded.adapter_id != adapter_id or forwarded.base_model != self.base_model:
                    raise ValueError("forwarded adapter identity does not match engine dispatch")
                if not forwarded.serve_base_model and not forwarded.is_revision:
                    raise ValueError("forwarded adapter must be an immutable revision")
                self.registry.upsert(forwarded)
            record = self.registry.get(adapter_id)
            if record is None or record.status != "ready":
                raise ValueError(f"Unknown adapter id on {self.base_model}: {adapter_id}")
            if (
                forwarded is not None
                and record.immutable_fingerprint() != forwarded.immutable_fingerprint()
            ):
                raise ValueError(
                    "engine adapter identity differs from the routed immutable revision"
                )
            if not record.serve_base_model and not record.is_revision:
                raise ValueError(f"Unknown adapter id on {self.base_model}: {adapter_id}")
            if record.serve_base_model:
                # No LoRA to resolve: generate against the base weights the engine already has.
                return None, record
            path = await self._ensure_adapter_local_locked(record)
            lora_request = self._cached_lora_request_locked(record, path)
            entry = self._entries().get(adapter_id)
            if (
                entry is not None
                and entry.state == "loaded"
                and entry.tombstoned
                and entry.in_flight == 0
            ):
                self._entries()[adapter_id] = _LoraEntry(
                    entry.source_ident,
                    entry.lora_request,
                    entry.state,
                    entry.in_flight,
                    False,
                )
            return lora_request, record

    @asynccontextmanager
    async def _lora_request_in_flight(
        self,
        record: Any,
        lora_request: Any,
    ) -> AsyncIterator[None]:
        if lora_request is None:
            yield
            return
        adapter_id = record.adapter_id
        source_ident = _adapter_source_ident(record)
        lock = await self._adapter_lock(adapter_id)
        async with lock:
            entry = self._entries().get(adapter_id)
            if (
                entry is None
                or entry.tombstoned
                or entry.source_ident != source_ident
                or entry.lora_request is not lora_request
            ):
                raise RuntimeError("LoRA request is no longer current on this engine")
            self._entries()[adapter_id] = _LoraEntry(
                entry.source_ident,
                entry.lora_request,
                entry.state,
                entry.in_flight + 1,
                False,
            )
        try:
            yield
        finally:
            remove_source = False
            async with lock:
                entry = self._entries().get(adapter_id)
                if (
                    entry is not None
                    and entry.source_ident == source_ident
                    and entry.lora_request is lora_request
                    and entry.in_flight > 0
                ):
                    remaining = entry.in_flight - 1
                    if remaining == 0 and entry.tombstoned and entry.state == "reserved":
                        self._entries().pop(adapter_id)
                        remove_source = True
                    else:
                        self._entries()[adapter_id] = _LoraEntry(
                            entry.source_ident,
                            entry.lora_request,
                            entry.state,
                            remaining,
                            entry.tombstoned,
                        )
            if remove_source:
                manager = getattr(self, "_source_cache", None)
                if manager is not None:
                    await manager.remove_if_unreferenced(source_ident)

    async def _mark_lora_consumed(self, record: Any, lora_request: Any) -> None:
        if lora_request is None:
            return
        adapter_id = record.adapter_id
        source_ident = _adapter_source_ident(record)
        lock = await self._adapter_lock(adapter_id)
        async with lock:
            entry = self._entries().get(adapter_id)
            if entry is None:
                return
            if entry.source_ident != source_ident or entry.lora_request is not lora_request:
                raise RuntimeError("consumed LoRA does not match the in-flight adapter source")
            if entry.in_flight <= 0:
                raise RuntimeError("consumed LoRA has no in-flight request lease")
            if entry.state == "loaded":
                return
            if entry.state != "reserved":
                raise RuntimeError("consumed LoRA registration state is unconfirmed")
            manager = getattr(self, "_source_cache", None)
            if manager is not None:
                directory = _adapter_source_cache_dir(self._adapter_materialization_dir(), record)
                await manager.mark_loaded(adapter_id, source_ident, directory)
            self._entries()[adapter_id] = _LoraEntry(
                source_ident,
                lora_request,
                "loaded",
                entry.in_flight,
                entry.tombstoned,
            )

    @asynccontextmanager
    async def _source_generation_lease(
        self,
        record: Any,
        lora_request: Any,
    ) -> AsyncIterator[None]:
        if lora_request is None:
            yield
            return
        manager = getattr(self, "_source_cache", None)
        if manager is None:
            yield
            return
        ident = _adapter_source_ident(record)
        directory = _adapter_source_cache_dir(self._adapter_materialization_dir(), record)
        async with manager.active(ident, directory):
            yield

    async def _register(
        self,
        record_dict: dict[str, Any],
        deployment_generation: str | None = None,
    ) -> dict[str, Any]:
        """Download + register an adapter into this engine's cache."""
        from flash.serving.src.io.schemas import AdapterRecord

        record = AdapterRecord.model_validate(record_dict).model_copy(
            update={"deployment_generation": deployment_generation}
        )
        if not record.serve_base_model and not record.is_revision:
            raise ValueError("only immutable adapter revisions can be registered")
        lock = await self._adapter_lock(record.adapter_id)
        async with lock:  # _locked variant: we hold the lock (the public one would deadlock)
            if record.serve_base_model:
                # No LoRA to download or add — the base weights are already loaded; just track the id.
                self.registry.upsert(record, revive=True)
                return {"ok": True, "adapter_id": record.adapter_id, "base_model": self.base_model}
            previous_record = self.registry.get(record.adapter_id)
            previous_ident = (
                _adapter_source_ident(previous_record) if previous_record is not None else None
            )
            previous_path = (
                self.registry.local_path(previous_record) if previous_record is not None else None
            )
            source_ident = _adapter_source_ident(record)
            try:
                path = await self._ensure_adapter_local_locked(record)
                await self._add_lora_locked(record, path)
            except BaseException:
                self.registry.clear_local_path(record.adapter_id)
                self._source_paths.pop(source_ident, None)
                await self._source_cache_manager().restore_current(
                    record.adapter_id,
                    previous_ident,
                    source_ident,
                )
                if previous_record is not None and previous_path is not None:
                    self.registry.set_local_path(previous_record, previous_path)
                    self._source_paths[previous_ident] = previous_path
                await self._source_cache_manager().remove_if_unreferenced(source_ident)
                raise
            self.registry.upsert(record, revive=True)
        return {"ok": True, "adapter_id": record.adapter_id, "base_model": self.base_model}

    async def _unregister(
        self,
        adapter_id: str,
        expected_generation: str | None = None,
    ) -> dict[str, Any]:
        # undeploy under the per-adapter lock so register and stale background cleanup have one order.
        lock = await self._adapter_lock(adapter_id)
        async with lock:
            current = self.registry.get(adapter_id)
            current_generation = current.deployment_generation if current is not None else None
            stale = current is not None and (
                current_generation is not None
                if expected_generation is None
                else current_generation != expected_generation
            )
            if stale:
                return {
                    "ok": True,
                    "removed": None,
                    "skipped_stale_generation": True,
                    "base_model": self.base_model,
                    "cleanup_scope": "replica_local",
                    "engine_replica_id": self._replica_identifier(),
                }
            self.registry.remove(adapter_id)
            manager = getattr(self, "_source_cache", None)
            if manager is not None:
                await manager.release_current(adapter_id)
            await self._evict_loaded_lora(adapter_id)
        return {
            "ok": True,
            "removed": adapter_id,
            "base_model": self.base_model,
            "cleanup_scope": "replica_local",
            "engine_replica_id": self._replica_identifier(),
        }

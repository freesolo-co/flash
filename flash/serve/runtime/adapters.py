"""process-local lora registration, locking, identity, and unload."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path
from typing import Any

from flash.serve.request.runtime_support import is_adapter_tensor_file as _is_adapter_tensor_file

from .errors import (
    AdapterConflictError,
    AdapterError,
    AdapterNotFoundError,
    AdapterPathError,
    StaleIncarnationError,
)
from .types import AdapterSpec, EngineConfig

_MAX_LORA_ID = 0x7FFFFFFF


@dataclass(slots=True)
class AdapterBinding:
    """one locked adapter incarnation and its vllm request."""

    spec: AdapterSpec
    lora_request: Any


class _AdapterState(Enum):
    NOT_OWNED = auto()
    OWNED = auto()
    INDETERMINATE = auto()
    QUARANTINED = auto()


@dataclass(slots=True)
class _AdapterEntry:
    spec: AdapterSpec
    lora_request: Any
    state: _AdapterState = _AdapterState.NOT_OWNED


@dataclass(slots=True)
class _AdapterGate:
    """the per-adapter mutual exclusion between generations and lifecycle changes.

    Generations only read an incarnation, so any number may run at once -- vllm batches them, and
    holding a plain lock across a whole generation capped every adapter at one in-flight request
    regardless of `max_num_seqs`. Registration and unload mutate the incarnation vllm is serving,
    so they need the adapter to themselves.

    Writers are exclusive against each other via `lock`, and against readers via `drained`, which is
    set exactly while `readers == 0`. A writer holds `lock` for its whole operation, so no reader can
    enter behind it: `acquire` waits on `lock` before counting itself in.
    """

    lock: asyncio.Lock
    drained: asyncio.Event
    readers: int = 0

    def enter(self) -> None:
        self.readers += 1
        self.drained.clear()

    def leave(self) -> None:
        self.readers -= 1
        if self.readers == 0:
            self.drained.set()


def lora_int_id(adapter_id: str) -> int:
    """return a positive int32-compatible deterministic starting id."""
    digest = hashlib.sha1(adapter_id.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") & _MAX_LORA_ID) or 1


def validate_adapter_path(raw_path: str) -> Path:
    """return a resolved directory containing config and non-empty lora tensors."""
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise AdapterPathError(f"adapter path is not a directory: {path}")
    if not (path / "adapter_config.json").is_file():
        raise AdapterPathError(f"adapter path has no adapter_config.json: {path}")
    try:
        has_tensor = any(
            child.is_file() and _is_adapter_tensor_file(child) and child.stat().st_size > 0
            for child in path.iterdir()
        )
    except OSError as exc:
        raise AdapterPathError(f"adapter path cannot be inspected: {path}") from exc
    if not has_tensor:
        raise AdapterPathError(f"adapter path has no non-empty adapter tensor file: {path}")
    return path


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class AdapterManager:
    """serialize operations per adapter and keep lora ids collision-safe."""

    def __init__(self, engine: Any, config: EngineConfig) -> None:
        self._engine = engine
        self._config = config
        self._entries: dict[str, _AdapterEntry] = {}
        self._ids: dict[int, str] = {}
        self._gates: dict[str, _AdapterGate] = {}
        self._gates_guard = asyncio.Lock()
        self._state_lock = asyncio.Lock()

    @property
    def registered_count(self) -> int:
        return sum(entry.state is not _AdapterState.QUARANTINED for entry in self._entries.values())

    @property
    def loaded_count(self) -> int:
        return sum(entry.state is _AdapterState.OWNED for entry in self._entries.values())

    async def register(self, spec: AdapterSpec) -> bool:
        """load an exact incarnation, replacing the prior incarnation atomically per id."""
        path = await asyncio.to_thread(validate_adapter_path, spec.path)
        normalized = replace(spec, path=str(path))
        async with self._exclusive(normalized.adapter_id):
            current = self._entries.get(normalized.adapter_id)
            if current is not None:
                self._require_determinate_ownership(current)
            if current is not None and current.spec.incarnation == normalized.incarnation:
                if current.spec != normalized:
                    raise AdapterConflictError(
                        "one adapter incarnation token cannot identify different runtime state"
                    )
                if current.state is _AdapterState.NOT_OWNED:
                    await self._load_entry(current)
                return False
            int_id = await self._reserve_id(normalized.adapter_id, current)
            if current is not None and current.state is _AdapterState.OWNED:
                await self._remove_lora(current)
            request = self._new_lora_request(normalized, int_id)
            entry = _AdapterEntry(spec=normalized, lora_request=request)
            try:
                await self._load_entry(entry)
            except BaseException:
                if current is not None and entry.state is _AdapterState.INDETERMINATE:
                    current.state = _AdapterState.INDETERMINATE
                elif current is None and entry.state is _AdapterState.INDETERMINATE:
                    entry.state = _AdapterState.QUARANTINED
                    self._entries[normalized.adapter_id] = entry
                elif current is None:
                    await self._release_failed_reservation(normalized.adapter_id, int_id)
                raise
            self._entries[normalized.adapter_id] = entry
            return True

    @asynccontextmanager
    async def acquire(
        self,
        adapter_id: str,
        expected_incarnation: str | None,
    ) -> AsyncIterator[AdapterBinding]:
        """pin one incarnation for the full operation so replacement cannot cross-wire it.

        Held as a reader: concurrent generations on the same adapter run together, while a
        replacement or unload waits for them to finish rather than interleaving with them.
        """
        gate = await self._adapter_gate(adapter_id)
        async with gate.lock:
            entry = self._entries.get(adapter_id)
            if entry is None:
                raise AdapterNotFoundError(f"adapter is not registered: {adapter_id}")
            self._require_incarnation(entry, expected_incarnation)
            self._require_determinate_ownership(entry)
            if entry.state is _AdapterState.NOT_OWNED:
                await self._load_entry(entry)
            binding = AdapterBinding(spec=entry.spec, lora_request=entry.lora_request)
            gate.enter()
        try:
            yield binding
        finally:
            gate.leave()

    async def unload(
        self,
        adapter_id: str,
        expected_incarnation: str | None = None,
    ) -> bool:
        """remove one registration and all vllm lora state for its exact incarnation."""
        async with self._exclusive(adapter_id):
            entry = self._entries.get(adapter_id)
            if entry is None:
                return False
            self._require_incarnation(entry, expected_incarnation)
            if entry.state is not _AdapterState.NOT_OWNED:
                await self._remove_lora(entry)
            self._entries.pop(adapter_id)
            async with self._state_lock:
                self._ids.pop(entry.lora_request.lora_int_id, None)
            return True

    async def unload_all(self) -> None:
        for adapter_id in list(self._entries):
            await self.unload(adapter_id)

    async def _adapter_gate(self, adapter_id: str) -> _AdapterGate:
        async with self._gates_guard:
            gate = self._gates.get(adapter_id)
            if gate is None:
                gate = _AdapterGate(lock=asyncio.Lock(), drained=asyncio.Event())
                gate.drained.set()
                self._gates[adapter_id] = gate
            return gate

    @asynccontextmanager
    async def _exclusive(self, adapter_id: str) -> AsyncIterator[None]:
        """hold one adapter against every other operation, including in-flight generations.

        Taking `lock` first is what makes the wait terminate: it stops new readers from entering, so
        the reader count can only fall while we wait on `drained`.
        """

        gate = await self._adapter_gate(adapter_id)
        async with gate.lock:
            await gate.drained.wait()
            yield

    async def _reserve_id(
        self,
        adapter_id: str,
        current: _AdapterEntry | None,
    ) -> int:
        async with self._state_lock:
            if current is not None:
                return int(current.lora_request.lora_int_id)
            int_id = lora_int_id(adapter_id)
            while int_id in self._ids and self._ids[int_id] != adapter_id:
                int_id = int_id + 1 if int_id < _MAX_LORA_ID else 1
            self._ids[int_id] = adapter_id
            return int_id

    async def _release_failed_reservation(self, adapter_id: str, int_id: int) -> None:
        async with self._state_lock:
            if self._ids.get(int_id) == adapter_id:
                self._ids.pop(int_id, None)

    def _new_lora_request(self, spec: AdapterSpec, int_id: int) -> Any:
        from vllm.lora.request import LoRARequest

        return LoRARequest(spec.adapter_id, int_id, spec.path)

    async def _load_entry(self, entry: _AdapterEntry) -> None:
        should_pin = self._config.effective_pin_loras if entry.spec.pin is None else entry.spec.pin
        pin = getattr(self._engine, "pin_lora", None)
        if should_pin and pin is None:
            raise AdapterError("this vllm build does not expose pin_lora")
        entry.state = _AdapterState.INDETERMINATE
        added = await _maybe_await(self._engine.add_lora(entry.lora_request))
        if added is not True:
            raise AdapterError("vllm did not confirm lora registration")
        try:
            if should_pin:
                assert pin is not None
                await _maybe_await(pin(entry.lora_request.lora_int_id))
        except BaseException:
            await self._cleanup_failed_load(entry)
            raise
        entry.state = _AdapterState.OWNED

    async def _cleanup_failed_load(self, entry: _AdapterEntry) -> None:
        entry.state = _AdapterState.INDETERMINATE
        try:
            removed = await self._call_remove_lora(entry.lora_request.lora_int_id)
        except Exception:
            return
        if removed is True:
            entry.state = _AdapterState.NOT_OWNED

    async def _remove_lora(self, entry: _AdapterEntry) -> None:
        indeterminate = (
            _AdapterState.QUARANTINED
            if entry.state is _AdapterState.QUARANTINED
            else _AdapterState.INDETERMINATE
        )
        entry.state = indeterminate
        removed = await self._call_remove_lora(entry.lora_request.lora_int_id)
        if removed is not True:
            raise AdapterError("vllm did not confirm lora removal")
        entry.state = _AdapterState.NOT_OWNED

    async def _call_remove_lora(self, int_id: int) -> Any:
        remove = getattr(self._engine, "remove_lora", None)
        if remove is None:
            raise AdapterError("this vllm build does not expose remove_lora")
        return await _maybe_await(remove(int_id))

    @staticmethod
    def _require_determinate_ownership(entry: _AdapterEntry) -> None:
        if entry.state in {_AdapterState.INDETERMINATE, _AdapterState.QUARANTINED}:
            raise AdapterError(
                "vllm lora ownership is indeterminate; unload must confirm removal before reuse"
            )

    @staticmethod
    def _require_incarnation(
        entry: _AdapterEntry,
        expected_incarnation: str | None,
    ) -> None:
        if expected_incarnation is not None and entry.spec.incarnation != expected_incarnation:
            raise StaleIncarnationError(
                f"adapter {entry.spec.adapter_id!r} is no longer incarnation "
                f"{expected_incarnation!r}"
            )

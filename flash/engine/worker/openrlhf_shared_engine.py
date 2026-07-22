"""shared vLLM multi-LoRA rollout engine primitives.

this module owns adapter identity, publication, routing, and lifecycle only. shared
training state, scoring, and scheduling belong to later shared-controller layers.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_VLLM_LORA_ID = 0x7FFFFFFF
_ADAPTER_TENSOR_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


class AdapterRegistryError(RuntimeError):
    """base error for invalid shared-engine adapter operations."""


class AdapterVersionError(AdapterRegistryError):
    """raised when an adapter publication does not match the current version."""


class AdapterCapacityError(AdapterRegistryError):
    """raised when more logical runs are registered than the engine supports."""


class UnknownAdapterHandle(AdapterRegistryError):
    """raised when a rollout does not reference the current immutable handle."""


@dataclass(frozen=True, slots=True)
class AdapterHandle:
    """immutable routing identity for one published adapter version."""

    run_id: str
    run_slot: int
    version: int
    lora_int_id: int
    lora_name: str
    adapter_dir: str
    prefix_cache_namespace: str


@dataclass(frozen=True, slots=True)
class RolloutEnvelope:
    """a rollout result bound to the exact adapter version that generated it."""

    handle: AdapterHandle
    request_id: str
    output: Any


@dataclass(frozen=True, slots=True)
class AdapterHealth:
    """immutable adapter state returned by :meth:`health`."""

    handle: AdapterHandle
    in_flight: int
    current: bool
    stale: bool


@dataclass(frozen=True, slots=True)
class SharedEngineHealth:
    """snapshot of logical-run and hot-adapter occupancy."""

    run_capacity: int
    hot_slot_capacity: int
    adapters: tuple[AdapterHealth, ...]


@dataclass(slots=True)
class _AdapterState:
    handle: AdapterHandle
    lora_request: Any
    in_flight: int = 0
    stale: bool = False


def shared_vllm_engine_kwargs(
    base_kwargs: Mapping[str, Any],
    *,
    run_capacity: int,
    max_lora_rank: int,
    max_cpu_loras: int | None = None,
) -> dict[str, Any]:
    """apply shared-engine LoRA requirements to existing vLLM engine arguments.

    model-specific values already supplied by the OpenRLHF GRPO engine hooks are
    preserved. the shared engine adds one hot slot beyond the logical run capacity
    so an immutable new version can load before the draining version is removed.
    """

    if run_capacity < 1:
        raise ValueError("run_capacity must be positive")
    if max_lora_rank < 1:
        raise ValueError("max_lora_rank must be positive")
    hot_slot_capacity = run_capacity + 1
    cpu_capacity = hot_slot_capacity if max_cpu_loras is None else int(max_cpu_loras)
    if cpu_capacity < hot_slot_capacity:
        raise ValueError("max_cpu_loras must be at least run_capacity + 1")

    kwargs = dict(base_kwargs)
    kwargs.update(
        {
            "enable_lora": True,
            "max_loras": hot_slot_capacity,
            "max_cpu_loras": cpu_capacity,
            "max_lora_rank": int(max_lora_rank),
        }
    )
    return kwargs


def _default_lora_request_factory(name: str, int_id: int, path: str) -> Any:
    from vllm.lora.request import LoRARequest

    return LoRARequest(name, int_id, path)


def _validate_adapter_dir(adapter_dir: str | os.PathLike[str]) -> str:
    path = Path(adapter_dir).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"adapter path is not a directory: {path}")

    config_path = path / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"adapter directory has no adapter_config.json: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"adapter_config.json is invalid JSON: {path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"adapter_config.json must contain an object: {path}")

    tensor_path = next(
        (
            path / name
            for name in _ADAPTER_TENSOR_NAMES
            if (path / name).is_file() and (path / name).stat().st_size > 0
        ),
        None,
    )
    if tensor_path is None:
        raise ValueError(f"adapter directory has no non-empty adapter tensor file: {path}")
    return str(path)


class SharedMultiLoRARolloutEngine:
    """route concurrent rollouts through one vLLM engine by immutable handles.

    adapter additions and removals are serialized, while generation remains
    concurrent. each publication receives a fresh vLLM integer id, LoRA name, and
    prompt cache salt. an old version remains loaded until its in-flight count is
    zero, and the global ``run_capacity + 1`` limit permits exactly one draining
    version while every run retains its current version.
    """

    def __init__(
        self,
        engine: Any,
        *,
        run_capacity: int,
        lora_request_factory: Callable[[str, int, str], Any] | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if run_capacity < 1:
            raise ValueError("run_capacity must be positive")
        self._engine = engine
        self._run_capacity = int(run_capacity)
        self._hot_slot_capacity = self._run_capacity + 1
        self._lora_request_factory = lora_request_factory or _default_lora_request_factory
        self._request_id_factory = request_id_factory or (lambda: str(uuid.uuid4()))
        self._condition = asyncio.Condition()
        self._mutation_lock = asyncio.Lock()
        self._current_by_run: dict[str, AdapterHandle] = {}
        self._states: dict[int, _AdapterState] = {}
        self._run_slots: dict[str, int] = {}
        self._known_run_ids: set[str] = set()
        self._issued_lora_ids: set[int] = set()

    @property
    def run_capacity(self) -> int:
        return self._run_capacity

    @property
    def hot_slot_capacity(self) -> int:
        return self._hot_slot_capacity

    async def register_run(
        self,
        run_id: str,
        adapter_version: int,
        adapter_dir: str | os.PathLike[str],
    ) -> AdapterHandle:
        """load and register the initial immutable adapter for one logical run."""

        normalized_run_id = self._validate_identity(run_id, adapter_version)
        resolved_dir = _validate_adapter_dir(adapter_dir)
        while True:
            async with self._condition:
                if normalized_run_id in self._known_run_ids:
                    raise AdapterRegistryError(
                        f"run id was already registered: {normalized_run_id}"
                    )
                if len(self._current_by_run) >= self._run_capacity:
                    raise AdapterCapacityError(
                        f"shared engine supports at most {self._run_capacity} runs"
                    )
                await self._condition.wait_for(lambda: len(self._states) < self._hot_slot_capacity)

            async with self._mutation_lock:
                async with self._condition:
                    if normalized_run_id in self._known_run_ids:
                        raise AdapterRegistryError(
                            f"run id was already registered: {normalized_run_id}"
                        )
                    if len(self._current_by_run) >= self._run_capacity:
                        raise AdapterCapacityError(
                            f"shared engine supports at most {self._run_capacity} runs"
                        )
                    if len(self._states) >= self._hot_slot_capacity:
                        continue
                    run_slot = self._allocate_run_slot()
                    handle, request = self._new_handle(
                        normalized_run_id,
                        run_slot,
                        adapter_version,
                        resolved_dir,
                    )

                await self._add_lora(request)
                async with self._condition:
                    self._run_slots[normalized_run_id] = run_slot
                    self._known_run_ids.add(normalized_run_id)
                    self._states[handle.lora_int_id] = _AdapterState(handle, request)
                    self._current_by_run[normalized_run_id] = handle
                    self._condition.notify_all()
                return handle

    async def current_handle(self, run_id: str) -> AdapterHandle:
        """return the current immutable adapter handle for a run."""

        async with self._condition:
            try:
                return self._current_by_run[run_id]
            except KeyError as exc:
                raise AdapterRegistryError(f"unknown run: {run_id}") from exc

    async def publish_adapter(
        self,
        run_id: str,
        expected_old_version: int,
        new_version: int,
        adapter_dir: str | os.PathLike[str],
    ) -> AdapterHandle:
        """atomically publish a fresh adapter version for future rollouts.

        engine loading completes before the run pointer changes. a failed load
        therefore leaves the old handle current. the old version is evicted only
        after every rollout that acquired it has drained.
        """

        normalized_run_id = self._validate_identity(run_id, new_version)
        if new_version <= expected_old_version:
            raise AdapterVersionError("new adapter version must increase monotonically")
        resolved_dir = _validate_adapter_dir(adapter_dir)

        while True:
            async with self._condition:
                self._require_expected_version(normalized_run_id, expected_old_version)
                await self._condition.wait_for(lambda: len(self._states) < self._hot_slot_capacity)

            async with self._mutation_lock:
                async with self._condition:
                    old_handle = self._require_expected_version(
                        normalized_run_id, expected_old_version
                    )
                    if len(self._states) >= self._hot_slot_capacity:
                        continue
                    new_handle, request = self._new_handle(
                        normalized_run_id,
                        old_handle.run_slot,
                        new_version,
                        resolved_dir,
                    )

                await self._add_lora(request)
                async with self._condition:
                    old_state = self._states[old_handle.lora_int_id]
                    old_state.stale = True
                    self._states[new_handle.lora_int_id] = _AdapterState(new_handle, request)
                    self._current_by_run[normalized_run_id] = new_handle
                    evict_old = old_state.in_flight == 0
                    self._condition.notify_all()

                if evict_old:
                    await self._evict_state_locked(old_handle.lora_int_id)
                return new_handle

    async def generate(
        self,
        handle: AdapterHandle,
        prompt: Mapping[str, Any],
        sampling_params: Any,
        *,
        request_id: str | None = None,
        **engine_kwargs: Any,
    ) -> RolloutEnvelope:
        """generate with the exact current adapter referenced by ``handle``."""

        state = await self._acquire(handle)
        resolved_request_id = request_id or self._request_id_factory()
        prompt_with_namespace = dict(prompt)
        existing_salt = prompt_with_namespace.get("cache_salt")
        if existing_salt not in (None, handle.prefix_cache_namespace):
            await self._release(handle)
            raise ValueError("prompt cache_salt conflicts with the adapter handle")
        prompt_with_namespace["cache_salt"] = handle.prefix_cache_namespace

        try:
            stream = self._engine.generate(
                prompt_with_namespace,
                sampling_params,
                resolved_request_id,
                lora_request=state.lora_request,
                **engine_kwargs,
            )
            final_output = None
            if hasattr(stream, "__aiter__"):
                async for output in stream:
                    final_output = output
            elif inspect.isawaitable(stream):
                final_output = await stream
            else:
                final_output = stream
            if final_output is None:
                raise RuntimeError("vLLM returned no rollout output")
            return RolloutEnvelope(handle, resolved_request_id, final_output)
        finally:
            await self._release(handle)

    async def remove_run(self, run_id: str, expected_version: int) -> None:
        """stop admission, drain, and unload every live version for one run."""

        async with self._mutation_lock:
            async with self._condition:
                self._require_expected_version(run_id, expected_version)
                self._current_by_run.pop(run_id)
                states = [state for state in self._states.values() if state.handle.run_id == run_id]
                for state in states:
                    state.stale = True
                await self._condition.wait_for(
                    lambda: all(state.in_flight == 0 for state in states)
                )

            for state in states:
                await self._evict_state_locked(state.handle.lora_int_id)
            async with self._condition:
                self._run_slots.pop(run_id, None)
                self._condition.notify_all()

    async def health(self) -> SharedEngineHealth:
        """return an immutable registry and in-flight reference snapshot."""

        async with self._condition:
            adapters = tuple(
                AdapterHealth(
                    handle=state.handle,
                    in_flight=state.in_flight,
                    current=self._current_by_run.get(state.handle.run_id) == state.handle,
                    stale=state.stale,
                )
                for state in sorted(
                    self._states.values(),
                    key=lambda value: (value.handle.run_slot, value.handle.version),
                )
            )
            return SharedEngineHealth(
                run_capacity=self._run_capacity,
                hot_slot_capacity=self._hot_slot_capacity,
                adapters=adapters,
            )

    @staticmethod
    def _validate_identity(run_id: str, version: int) -> str:
        normalized_run_id = str(run_id).strip()
        if not normalized_run_id:
            raise ValueError("run_id must not be empty")
        if int(version) < 0:
            raise ValueError("adapter version must be non-negative")
        return normalized_run_id

    def _allocate_run_slot(self) -> int:
        used = set(self._run_slots.values())
        for slot in range(self._run_capacity):
            if slot not in used:
                return slot
        raise AdapterCapacityError(f"shared engine supports at most {self._run_capacity} runs")

    def _new_handle(
        self,
        run_id: str,
        run_slot: int,
        version: int,
        adapter_dir: str,
    ) -> tuple[AdapterHandle, Any]:
        digest = hashlib.sha256(f"{run_id}\0{version}".encode()).digest()
        int_id = (int.from_bytes(digest[:4], "big") & _MAX_VLLM_LORA_ID) or 1
        while int_id in self._issued_lora_ids:
            int_id = int_id + 1 if int_id < _MAX_VLLM_LORA_ID else 1
        self._issued_lora_ids.add(int_id)
        identity = hashlib.sha256(run_id.encode()).hexdigest()[:16]
        lora_name = f"flash-shared-{identity}-v{version}-id{int_id}"
        cache_namespace = f"{lora_name}-cache"
        handle = AdapterHandle(
            run_id=run_id,
            run_slot=run_slot,
            version=int(version),
            lora_int_id=int_id,
            lora_name=lora_name,
            adapter_dir=adapter_dir,
            prefix_cache_namespace=cache_namespace,
        )
        return handle, self._lora_request_factory(lora_name, int_id, adapter_dir)

    def _require_expected_version(self, run_id: str, expected_version: int) -> AdapterHandle:
        current = self._current_by_run.get(run_id)
        if current is None:
            raise AdapterRegistryError(f"unknown run: {run_id}")
        if current.version != expected_version:
            raise AdapterVersionError(
                f"run {run_id} is at adapter version {current.version}, not {expected_version}"
            )
        return current

    async def _add_lora(self, request: Any) -> None:
        try:
            result = self._engine.add_lora(request)
            if inspect.isawaitable(result):
                result = await result
            if result is False:
                raise AdapterRegistryError(f"vLLM rejected adapter id {request.lora_int_id}")

            pin = getattr(self._engine, "pin_lora", None)
            if pin is None:
                return
            result = pin(request.lora_int_id)
            if inspect.isawaitable(result):
                result = await result
            if result is False:
                raise AdapterRegistryError(f"vLLM could not pin adapter id {request.lora_int_id}")
        except BaseException:
            with contextlib.suppress(BaseException):
                await self._remove_lora(request.lora_int_id)
            raise

    async def _remove_lora(self, int_id: int) -> None:
        result = self._engine.remove_lora(int_id)
        if inspect.isawaitable(result):
            result = await result
        if result is False:
            raise AdapterRegistryError(f"vLLM could not remove adapter id {int_id}")

    async def _acquire(self, handle: AdapterHandle) -> _AdapterState:
        async with self._condition:
            state = self._states.get(handle.lora_int_id)
            current = self._current_by_run.get(handle.run_id)
            if state is None or state.handle != handle or current != handle or state.stale:
                raise UnknownAdapterHandle(
                    f"adapter handle is not current for run {handle.run_id} version {handle.version}"
                )
            state.in_flight += 1
            return state

    async def _release(self, handle: AdapterHandle) -> None:
        async with self._condition:
            state = self._states.get(handle.lora_int_id)
            if state is None or state.handle != handle:
                raise UnknownAdapterHandle(
                    f"adapter handle disappeared while in flight: {handle.run_id} v{handle.version}"
                )
            if state.in_flight < 1:
                raise AdapterRegistryError("adapter in-flight reference count underflow")
            state.in_flight -= 1
            should_evict = state.stale and state.in_flight == 0
            self._condition.notify_all()
        if should_evict:
            await self._evict_if_unused(handle.lora_int_id)

    async def _evict_if_unused(self, int_id: int) -> None:
        async with self._mutation_lock:
            async with self._condition:
                state = self._states.get(int_id)
                if state is None or not state.stale or state.in_flight != 0:
                    return
            await self._evict_state_locked(int_id)

    async def _evict_state_locked(self, int_id: int) -> None:
        async with self._condition:
            state = self._states.get(int_id)
            if state is None:
                return
            if not state.stale or state.in_flight != 0:
                raise AdapterRegistryError(f"adapter id {int_id} is not safe to evict")
        await self._remove_lora(int_id)
        async with self._condition:
            self._states.pop(int_id, None)
            self._condition.notify_all()

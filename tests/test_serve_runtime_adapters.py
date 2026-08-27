"""adapter concurrency, incarnation replacement, ids, pinning, reload, and unload."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from flash.serve.runtime import (
    AdapterConflictError,
    AdapterError,
    AdapterSpec,
    EngineConfig,
    StaleIncarnationError,
)
from flash.serve.runtime import adapters as adapters_module
from flash.serve.runtime.adapters import AdapterManager


class _LoRARequest:
    def __init__(self, name: str, lora_int_id: int, path: str) -> None:
        self.lora_name = name
        self.lora_int_id = lora_int_id
        self.lora_path = path


class _Engine:
    def __init__(self) -> None:
        self.added: list[_LoRARequest] = []
        self.pinned: list[int] = []
        self.removed: list[int] = []
        self.add_result: object = True
        self.remove_result: object = True
        self.remove_results: list[object | BaseException] = []
        self.remove_started: asyncio.Event | None = None
        self.allow_remove: asyncio.Event | None = None
        self.fail_add_after_append = False
        self.fail_pin = False
        self.fail_remove = False

    async def add_lora(self, request: _LoRARequest) -> object:
        await asyncio.sleep(0)
        self.added.append(request)
        if self.fail_add_after_append:
            raise RuntimeError("add failed after partial registration")
        return self.add_result

    async def pin_lora(self, int_id: int) -> None:
        if self.fail_pin:
            raise RuntimeError("pin failed")
        self.pinned.append(int_id)

    async def remove_lora(self, int_id: int) -> object:
        self.removed.append(int_id)
        if self.remove_started is not None:
            self.remove_started.set()
        if self.allow_remove is not None:
            await self.allow_remove.wait()
        if self.fail_remove:
            raise RuntimeError("remove failed")
        if self.remove_results:
            result = self.remove_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return self.remove_result


@pytest.fixture(autouse=True)
def _fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    lora = types.ModuleType("vllm.lora")
    request = types.ModuleType("vllm.lora.request")
    request.LoRARequest = _LoRARequest
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request)


@pytest.fixture
def adapter_dir(tmp_path: Path) -> Path:
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    return path


def _spec(path: Path, incarnation: str = "one", *, adapter_id: str = "adapter") -> AdapterSpec:
    return AdapterSpec(adapter_id=adapter_id, path=str(path), incarnation=incarnation)


def test_concurrent_registration_is_idempotent(adapter_dir: Path) -> None:
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    spec = _spec(adapter_dir)

    async def exercise() -> list[bool]:
        return await asyncio.gather(manager.register(spec), manager.register(spec))

    assert sorted(asyncio.run(exercise())) == [False, True]
    assert len(engine.added) == 1
    assert engine.pinned == [engine.added[0].lora_int_id]
    assert manager.registered_count == manager.loaded_count == 1


@pytest.mark.parametrize(
    "add_result",
    [False, None, [True, False], 1],
    ids=["false", "none", "mixed-list", "non-bool"],
)
def test_add_requires_exact_true_and_retains_quarantine(
    adapter_dir: Path,
    monkeypatch,
    add_result: object,
) -> None:
    monkeypatch.setattr(adapters_module, "lora_int_id", lambda _adapter_id: 7)
    engine = _Engine()
    engine.add_result = add_result
    manager = AdapterManager(engine, EngineConfig(model="model"))

    with pytest.raises(AdapterError, match="vllm did not confirm lora registration"):
        asyncio.run(manager.register(_spec(adapter_dir, adapter_id="a")))

    assert manager.registered_count == manager.loaded_count == 0
    assert engine.pinned == []
    assert engine.removed == []
    engine.add_result = True
    with pytest.raises(AdapterError, match="ownership is indeterminate"):
        asyncio.run(manager.register(_spec(adapter_dir, "two", adapter_id="a")))
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="b")))
    assert engine.added[-1].lora_int_id == 8

    assert asyncio.run(manager.unload("a", "one")) is True
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="c")))
    assert engine.added[-1].lora_int_id == 7


def test_replacement_waits_for_inflight_incarnation(adapter_dir: Path) -> None:
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))

    async def exercise() -> None:
        await manager.register(_spec(adapter_dir, "one"))
        acquired = asyncio.Event()
        release = asyncio.Event()

        async def hold_generation() -> None:
            async with manager.acquire("adapter", "one"):
                acquired.set()
                await release.wait()

        generation = asyncio.create_task(hold_generation())
        await acquired.wait()
        replacement = asyncio.create_task(manager.register(_spec(adapter_dir, "two")))
        await asyncio.sleep(0)
        assert replacement.done() is False
        release.set()
        await generation
        assert await replacement is True

    asyncio.run(exercise())
    assert engine.removed == [engine.added[0].lora_int_id]
    assert len(engine.added) == 2


def test_generations_on_one_incarnation_run_concurrently(adapter_dir: Path) -> None:
    """two generations on the same adapter must overlap, not queue behind each other.

    holding the adapter's lock across `acquire`'s yield capped every adapter at one in-flight
    request, so an engine configured for `max_num_seqs` batching served them one at a time. asserted
    by interleaving rather than by elapsed time so the test cannot go green on a slow machine.
    """

    manager = AdapterManager(_Engine(), EngineConfig(model="model"))
    order: list[str] = []

    async def exercise() -> None:
        await manager.register(_spec(adapter_dir, "one"))
        both_inside = asyncio.Event()
        inside = 0

        async def generation(name: str) -> None:
            nonlocal inside
            async with manager.acquire("adapter", "one"):
                order.append(f"{name}:enter")
                inside += 1
                if inside == 2:
                    both_inside.set()
                # a serializing gate never lets the second reader in, so this waits forever.
                await asyncio.wait_for(both_inside.wait(), timeout=5)
                order.append(f"{name}:exit")

        await asyncio.gather(generation("first"), generation("second"))

    asyncio.run(exercise())
    # both entered before either exited: that is the overlap, and it is what a serializing gate
    # cannot produce. the order the two then wake in is an asyncio scheduling detail that differs
    # between interpreter versions, so only the enter/exit split is pinned.
    assert set(order[:2]) == {"first:enter", "second:enter"}
    assert set(order[2:]) == {"first:exit", "second:exit"}


def test_unload_waits_for_inflight_generations(adapter_dir: Path) -> None:
    """concurrent readers must not let an unload pull lora state out from under a generation."""

    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))

    async def exercise() -> None:
        await manager.register(_spec(adapter_dir, "one"))
        acquired = asyncio.Event()
        release = asyncio.Event()

        async def hold_generation() -> None:
            async with manager.acquire("adapter", "one"):
                acquired.set()
                await release.wait()

        generation = asyncio.create_task(hold_generation())
        await acquired.wait()
        unloading = asyncio.create_task(manager.unload("adapter", "one"))
        await asyncio.sleep(0)
        assert unloading.done() is False
        assert engine.removed == []
        release.set()
        await generation
        assert await unloading is True

    asyncio.run(exercise())
    assert engine.removed == [engine.added[0].lora_int_id]


@pytest.mark.parametrize(
    "remove_result",
    [False, None, [True, False], "removed"],
    ids=["false", "none", "mixed-list", "non-bool"],
)
def test_unload_requires_exact_true_and_retains_quarantine_and_id(
    adapter_dir: Path,
    monkeypatch,
    remove_result: object,
) -> None:
    monkeypatch.setattr(adapters_module, "lora_int_id", lambda _adapter_id: 7)
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    asyncio.run(manager.register(_spec(adapter_dir)))
    engine.remove_result = remove_result

    with pytest.raises(AdapterError, match="vllm did not confirm lora removal"):
        asyncio.run(manager.unload("adapter", "one"))

    assert manager.registered_count == 1
    assert manager.loaded_count == 0

    async def acquire_quarantined() -> None:
        async with manager.acquire("adapter", "one"):
            raise AssertionError("indeterminate engine ownership must block acquire")

    with pytest.raises(AdapterError, match="ownership is indeterminate"):
        asyncio.run(acquire_quarantined())
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="other")))
    assert engine.added[-1].lora_int_id == 8

    engine.remove_result = True
    assert asyncio.run(manager.unload("adapter", "one")) is True
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="reused")))
    assert engine.added[-1].lora_int_id == 7


def test_unload_remove_exception_retains_quarantine_and_id(
    adapter_dir: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(adapters_module, "lora_int_id", lambda _adapter_id: 7)
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="a")))
    engine.remove_results = [RuntimeError("remove failed")]

    with pytest.raises(RuntimeError, match="remove failed"):
        asyncio.run(manager.unload("a", "one"))

    assert manager.registered_count == 1
    assert manager.loaded_count == 0
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="b")))
    assert engine.added[-1].lora_int_id == 8

    assert asyncio.run(manager.unload("a", "one")) is True
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="c")))
    assert engine.added[-1].lora_int_id == 7


def test_same_incarnation_rejects_different_runtime_state(adapter_dir: Path) -> None:
    other = adapter_dir.parent / "other"
    other.mkdir()
    (other / "adapter_config.json").write_text("{}")
    (other / "adapter_model.bin").write_bytes(b"weights")
    manager = AdapterManager(_Engine(), EngineConfig(model="model"))
    asyncio.run(manager.register(_spec(adapter_dir)))

    with pytest.raises(AdapterConflictError, match="incarnation token"):
        asyncio.run(manager.register(_spec(other)))


def test_replacement_reuses_id_and_stale_operations_fail(adapter_dir: Path) -> None:
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    first = _spec(adapter_dir, "one")
    second = _spec(adapter_dir, "two")
    asyncio.run(manager.register(first))
    first_id = engine.added[-1].lora_int_id
    asyncio.run(manager.register(second))

    assert engine.removed == [first_id]
    assert engine.added[-1].lora_int_id == first_id

    async def stale_acquire() -> None:
        async with manager.acquire("adapter", "one"):
            raise AssertionError("stale acquire must not enter")

    with pytest.raises(StaleIncarnationError):
        asyncio.run(stale_acquire())
    with pytest.raises(StaleIncarnationError):
        asyncio.run(manager.unload("adapter", "one"))
    assert asyncio.run(manager.unload("adapter", "two")) is True


@pytest.mark.parametrize(
    "remove_result",
    [False, None, [True, False], {"removed": True}],
    ids=["false", "none", "mixed-list", "non-bool"],
)
def test_replacement_requires_exact_remove_true_before_new_incarnation(
    adapter_dir: Path,
    remove_result: object,
) -> None:
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    asyncio.run(manager.register(_spec(adapter_dir, "one")))
    engine.remove_result = remove_result

    with pytest.raises(AdapterError, match="vllm did not confirm lora removal"):
        asyncio.run(manager.register(_spec(adapter_dir, "two")))

    assert manager.registered_count == 1
    assert manager.loaded_count == 0
    assert [request.lora_name for request in engine.added] == ["adapter"]

    async def acquire_old() -> None:
        async with manager.acquire("adapter", "one"):
            raise AssertionError("indeterminate engine ownership must block acquire")

    with pytest.raises(AdapterError, match="ownership is indeterminate"):
        asyncio.run(acquire_old())
    with pytest.raises(AdapterError, match="ownership is indeterminate"):
        asyncio.run(manager.register(_spec(adapter_dir, "two")))

    engine.remove_result = True
    assert asyncio.run(manager.unload("adapter", "one")) is True
    assert asyncio.run(manager.register(_spec(adapter_dir, "two"))) is True


def test_failed_replacement_quarantines_concurrent_acquire(adapter_dir: Path) -> None:
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))

    async def exercise() -> None:
        await manager.register(_spec(adapter_dir, "one"))
        engine.remove_result = None
        engine.remove_started = asyncio.Event()
        engine.allow_remove = asyncio.Event()

        replacement = asyncio.create_task(manager.register(_spec(adapter_dir, "two")))
        await engine.remove_started.wait()

        async def acquire_old() -> None:
            async with manager.acquire("adapter", "one"):
                raise AssertionError("indeterminate engine ownership must block acquire")

        acquiring = asyncio.create_task(acquire_old())
        await asyncio.sleep(0)
        assert acquiring.done() is False

        engine.allow_remove.set()
        with pytest.raises(AdapterError, match="did not confirm lora removal"):
            await replacement
        with pytest.raises(AdapterError, match="ownership is indeterminate"):
            await acquiring

        assert len(engine.added) == 1
        engine.remove_started = None
        engine.allow_remove = None
        engine.remove_result = True
        assert await manager.unload("adapter", "one") is True

    asyncio.run(exercise())


def test_replacement_remove_exception_quarantines_old_incarnation(
    adapter_dir: Path,
) -> None:
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    asyncio.run(manager.register(_spec(adapter_dir, "one")))
    engine.fail_remove = True

    with pytest.raises(RuntimeError, match="remove failed"):
        asyncio.run(manager.register(_spec(adapter_dir, "two")))

    assert manager.registered_count == 1
    assert manager.loaded_count == 0
    assert len(engine.added) == 1
    engine.fail_remove = False
    with pytest.raises(AdapterError, match="ownership is indeterminate"):
        asyncio.run(manager.register(_spec(adapter_dir, "two")))
    assert asyncio.run(manager.unload("adapter", "one")) is True


@pytest.mark.parametrize(
    "add_result",
    [False, None, [True, False], {"added": True}],
    ids=["false", "none", "mixed-list", "non-bool"],
)
def test_replacement_add_requires_exact_true_before_new_incarnation(
    adapter_dir: Path,
    monkeypatch,
    add_result: object,
) -> None:
    monkeypatch.setattr(adapters_module, "lora_int_id", lambda _adapter_id: 7)
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="a")))
    engine.add_result = add_result

    with pytest.raises(AdapterError, match="vllm did not confirm lora registration"):
        asyncio.run(manager.register(_spec(adapter_dir, "two", adapter_id="a")))

    assert manager.registered_count == 1
    assert manager.loaded_count == 0
    assert len(engine.added) == 2
    engine.add_result = True
    with pytest.raises(AdapterError, match="ownership is indeterminate"):
        asyncio.run(manager.register(_spec(adapter_dir, "three", adapter_id="a")))
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="b")))
    assert engine.added[-1].lora_int_id == 8

    assert asyncio.run(manager.unload("a", "one")) is True
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="c")))
    assert engine.added[-1].lora_int_id == 7
    assert engine.removed == [7, 7]


def test_indeterminate_replacement_blocks_acquire_until_exact_removal(
    adapter_dir: Path,
) -> None:
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    asyncio.run(manager.register(_spec(adapter_dir, "one")))
    engine.fail_add_after_append = True

    with pytest.raises(RuntimeError, match="add failed"):
        asyncio.run(manager.register(_spec(adapter_dir, "two")))

    async def acquire_old() -> None:
        async with manager.acquire("adapter", "one"):
            raise AssertionError("indeterminate engine ownership must block acquire")

    engine.fail_add_after_append = False
    with pytest.raises(AdapterError, match="ownership is indeterminate"):
        asyncio.run(acquire_old())
    assert len(engine.added) == 2

    engine.remove_result = True
    assert asyncio.run(manager.unload("adapter", "one")) is True


def test_indeterminate_replacement_blocks_same_name_replacement(
    adapter_dir: Path,
) -> None:
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    asyncio.run(manager.register(_spec(adapter_dir, "one")))
    engine.fail_add_after_append = True

    with pytest.raises(RuntimeError, match="add failed"):
        asyncio.run(manager.register(_spec(adapter_dir, "two")))

    engine.fail_add_after_append = False
    with pytest.raises(AdapterError, match="ownership is indeterminate"):
        asyncio.run(manager.register(_spec(adapter_dir, "three")))
    assert len(engine.added) == 2

    engine.remove_result = True
    assert asyncio.run(manager.unload("adapter", "one")) is True
    assert asyncio.run(manager.register(_spec(adapter_dir, "three"))) is True


def test_initial_failure_blocks_same_name_retry_until_exact_removal(
    adapter_dir: Path,
) -> None:
    engine = _Engine()
    engine.fail_add_after_append = True
    engine.remove_result = None
    manager = AdapterManager(engine, EngineConfig(model="model"))

    with pytest.raises(RuntimeError, match="add failed"):
        asyncio.run(manager.register(_spec(adapter_dir, "one")))

    engine.fail_add_after_append = False
    with pytest.raises(AdapterError, match="ownership is indeterminate"):
        asyncio.run(manager.register(_spec(adapter_dir, "two")))
    assert len(engine.added) == 1

    with pytest.raises(AdapterError, match="did not confirm lora removal"):
        asyncio.run(manager.unload("adapter", "one"))
    assert len(engine.added) == 1

    engine.remove_result = True
    assert asyncio.run(manager.unload("adapter", "one")) is True
    assert asyncio.run(manager.register(_spec(adapter_dir, "two"))) is True
    assert engine.added[0].lora_int_id == engine.added[1].lora_int_id


def test_add_exception_quarantines_old_incarnation_until_exact_cleanup(
    adapter_dir: Path,
) -> None:
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    asyncio.run(manager.register(_spec(adapter_dir, "one")))
    int_id = engine.added[0].lora_int_id
    engine.fail_add_after_append = True

    with pytest.raises(RuntimeError, match="add failed"):
        asyncio.run(manager.register(_spec(adapter_dir, "two")))

    assert manager.registered_count == 1
    assert manager.loaded_count == 0
    assert engine.removed == [int_id]
    assert len(engine.added) == 2

    async def acquire_old() -> None:
        async with manager.acquire("adapter", "one"):
            raise AssertionError("indeterminate engine ownership must block acquire")

    engine.fail_add_after_append = False
    with pytest.raises(AdapterError, match="ownership is indeterminate"):
        asyncio.run(acquire_old())
    assert asyncio.run(manager.unload("adapter", "one")) is True
    assert engine.removed == [int_id, int_id]


def test_pin_failure_rolls_back_new_lora_and_preserves_old_incarnation(
    adapter_dir: Path,
) -> None:
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))

    async def exercise() -> int:
        await manager.register(_spec(adapter_dir, "one"))
        int_id = engine.added[0].lora_int_id
        engine.remove_results = [True, True]
        engine.fail_pin = True
        with pytest.raises(RuntimeError, match="pin failed"):
            await manager.register(_spec(adapter_dir, "two"))
        assert manager.registered_count == 1
        assert manager.loaded_count == 0
        engine.fail_pin = False
        async with manager.acquire("adapter", "one") as binding:
            assert binding.spec.incarnation == "one"
        return int_id

    int_id = asyncio.run(exercise())
    assert engine.removed == [int_id, int_id]
    assert engine.pinned == [int_id, int_id]


def test_replacement_pin_cleanup_false_requires_confirmed_unload_before_id_reuse(
    adapter_dir: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(adapters_module, "lora_int_id", lambda _adapter_id: 7)
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="a")))
    engine.remove_results = [True, False, False]
    engine.fail_pin = True

    with pytest.raises(RuntimeError, match="pin failed"):
        asyncio.run(manager.register(_spec(adapter_dir, "two", adapter_id="a")))

    assert manager.registered_count == 1
    assert manager.loaded_count == 0
    engine.fail_pin = False
    with pytest.raises(AdapterError, match="did not confirm lora removal"):
        asyncio.run(manager.unload("a", "one"))

    assert manager.registered_count == 1
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="b")))
    assert engine.added[-1].lora_int_id == 8

    engine.remove_result = True
    assert asyncio.run(manager.unload("a", "one")) is True
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="c")))
    assert engine.added[-1].lora_int_id == 7
    assert engine.removed == [7, 7, 7, 7]


def test_replacement_pin_cleanup_error_requires_confirmed_unload_before_id_reuse(
    adapter_dir: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(adapters_module, "lora_int_id", lambda _adapter_id: 7)
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="a")))
    engine.remove_results = [True, RuntimeError("cleanup failed")]
    engine.fail_pin = True

    with pytest.raises(RuntimeError, match="pin failed"):
        asyncio.run(manager.register(_spec(adapter_dir, "two", adapter_id="a")))

    assert manager.registered_count == 1
    assert manager.loaded_count == 0
    engine.fail_pin = False
    engine.fail_remove = True
    with pytest.raises(RuntimeError, match="remove failed"):
        asyncio.run(manager.unload("a", "one"))

    assert manager.registered_count == 1
    engine.fail_remove = False
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="b")))
    assert engine.added[-1].lora_int_id == 8

    engine.remove_result = True
    assert asyncio.run(manager.unload("a", "one")) is True
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="c")))
    assert engine.added[-1].lora_int_id == 7
    assert engine.removed == [7, 7, 7, 7]


def test_replacement_pin_confirmed_cleanup_releases_id_on_unload(
    adapter_dir: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(adapters_module, "lora_int_id", lambda _adapter_id: 7)
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="a")))
    engine.remove_results = [True, True]
    engine.fail_pin = True

    with pytest.raises(RuntimeError, match="pin failed"):
        asyncio.run(manager.register(_spec(adapter_dir, "two", adapter_id="a")))

    assert manager.registered_count == 1
    assert manager.loaded_count == 0
    engine.fail_pin = False
    assert asyncio.run(manager.unload("a", "one")) is True
    assert engine.removed == [7, 7]

    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="b")))
    assert engine.added[-1].lora_int_id == 7


def test_failed_pin_cleanup_false_retains_id_ownership(
    adapter_dir: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(adapters_module, "lora_int_id", lambda _adapter_id: 7)
    engine = _Engine()
    engine.fail_pin = True
    engine.remove_result = False
    manager = AdapterManager(engine, EngineConfig(model="model"))

    with pytest.raises(RuntimeError, match="pin failed"):
        asyncio.run(manager.register(_spec(adapter_dir, adapter_id="a")))

    assert manager.registered_count == manager.loaded_count == 0
    assert engine.removed == [7]
    engine.fail_pin = False
    engine.remove_result = None
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="b")))
    assert engine.added[-1].lora_int_id == 8


def test_failed_pin_cleanup_error_retains_id_ownership(
    adapter_dir: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(adapters_module, "lora_int_id", lambda _adapter_id: 7)
    engine = _Engine()
    engine.fail_pin = True
    engine.fail_remove = True
    manager = AdapterManager(engine, EngineConfig(model="model"))

    with pytest.raises(RuntimeError, match="pin failed"):
        asyncio.run(manager.register(_spec(adapter_dir, adapter_id="a")))

    assert manager.registered_count == manager.loaded_count == 0
    assert engine.removed == [7]
    engine.fail_pin = False
    engine.fail_remove = False
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="b")))
    assert engine.added[-1].lora_int_id == 8


def test_failed_pin_confirmed_cleanup_releases_id(
    adapter_dir: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(adapters_module, "lora_int_id", lambda _adapter_id: 7)
    engine = _Engine()
    engine.fail_pin = True
    engine.remove_result = True
    manager = AdapterManager(engine, EngineConfig(model="model"))

    with pytest.raises(RuntimeError, match="pin failed"):
        asyncio.run(manager.register(_spec(adapter_dir, adapter_id="a")))

    assert manager.registered_count == manager.loaded_count == 0
    assert engine.removed == [7]
    engine.fail_pin = False
    engine.remove_result = None
    asyncio.run(manager.register(_spec(adapter_dir, adapter_id="b")))
    assert engine.added[-1].lora_int_id == 7


def test_collision_safe_ids_probe_without_cross_wiring(
    adapter_dir: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(adapters_module, "lora_int_id", lambda _adapter_id: 7)
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model"))

    async def exercise() -> None:
        await asyncio.gather(
            manager.register(_spec(adapter_dir, adapter_id="a")),
            manager.register(_spec(adapter_dir, adapter_id="b")),
        )

    asyncio.run(exercise())
    ids = {request.lora_name: request.lora_int_id for request in engine.added}
    assert set(ids) == {"a", "b"}
    assert set(ids.values()) == {7, 8}


def test_a_registration_left_unloaded_reloads_on_acquire_then_unloads(adapter_dir: Path) -> None:
    """a registration can outlive its vllm state, and the next acquire must restore it.

    A replacement that fails partway removes the prior incarnation's lora from vllm but keeps
    it registered, so the entry is left present-but-unloaded. That is the only way the runtime
    reaches this state, so the reload is driven through it rather than through a helper no
    caller invokes.
    """

    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model", pin_loras=True))
    asyncio.run(manager.register(_spec(adapter_dir, "one")))
    int_id = engine.added[0].lora_int_id

    engine.remove_results = [True, True]
    engine.fail_pin = True
    with pytest.raises(RuntimeError):
        asyncio.run(manager.register(_spec(adapter_dir, "two")))
    engine.fail_pin = False

    # the replacement lost its lora state, and "one" is still the registered incarnation.
    assert manager.registered_count == 1
    assert manager.loaded_count == 0

    async def stale_acquire() -> None:
        async with manager.acquire("adapter", "two"):
            raise AssertionError("the failed replacement must not become acquirable")

    with pytest.raises(StaleIncarnationError):
        asyncio.run(stale_acquire())

    async def acquire() -> None:
        async with manager.acquire("adapter", "one") as binding:
            assert binding.spec.incarnation == "one"
            assert binding.lora_request.lora_int_id == int_id

    asyncio.run(acquire())
    assert manager.loaded_count == 1
    assert engine.pinned == [int_id, int_id]
    assert asyncio.run(manager.unload("adapter", "one")) is True
    assert manager.registered_count == manager.loaded_count == 0


def test_per_adapter_pin_false_skips_pinning(adapter_dir: Path) -> None:
    engine = _Engine()
    manager = AdapterManager(engine, EngineConfig(model="model", pin_loras=True))
    spec = AdapterSpec(
        adapter_id="adapter",
        path=str(adapter_dir),
        incarnation="one",
        pin=False,
    )
    asyncio.run(manager.register(spec))
    assert engine.pinned == []


def test_default_pinning_preserves_cpu_eviction_capacity(adapter_dir: Path) -> None:
    eviction_engine = _Engine()
    eviction_manager = AdapterManager(
        eviction_engine,
        EngineConfig(model="model", max_loras=1, max_cpu_loras=2),
    )
    asyncio.run(eviction_manager.register(_spec(adapter_dir)))
    assert eviction_engine.pinned == []

    covered_engine = _Engine()
    covered_manager = AdapterManager(
        covered_engine,
        EngineConfig(model="model", max_loras=2, max_cpu_loras=2),
    )
    asyncio.run(covered_manager.register(_spec(adapter_dir)))
    assert covered_engine.pinned == [covered_engine.added[0].lora_int_id]

    explicit_engine = _Engine()
    explicit_manager = AdapterManager(
        explicit_engine,
        EngineConfig(model="model", max_loras=1, max_cpu_loras=2, pin_loras=True),
    )
    asyncio.run(explicit_manager.register(_spec(adapter_dir)))
    assert explicit_engine.pinned == [explicit_engine.added[0].lora_int_id]

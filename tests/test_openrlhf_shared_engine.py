"""cpu-only tests for the shared OpenRLHF multi-LoRA rollout engine."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Any

import pytest

from flash.engine.worker.openrlhf_shared_engine import (
    AdapterRegistryError,
    AdapterVersionError,
    SharedMultiLoRARolloutEngine,
    UnknownAdapterHandle,
    shared_vllm_engine_kwargs,
)


@dataclass(frozen=True)
class _FakeLoRARequest:
    lora_name: str
    lora_int_id: int
    lora_path: str


class _FakeEngine:
    def __init__(self, *, max_loaded: int | None = None) -> None:
        self.added: list[_FakeLoRARequest] = []
        self.pinned: list[int] = []
        self.removed: list[int] = []
        self.loaded: set[int] = set()
        self.generations: list[dict[str, Any]] = []
        self.started: dict[int, asyncio.Event] = {}
        self.release: dict[int, asyncio.Event] = {}
        self.add_started: dict[str, asyncio.Event] = {}
        self.add_release: dict[str, asyncio.Event] = {}
        self.fail_remove_count: dict[int, int] = {}
        self.fail_add_path: str | None = None
        self.max_loaded = max_loaded

    async def add_lora(self, request: _FakeLoRARequest) -> bool:
        if request.lora_path == self.fail_add_path:
            raise RuntimeError("load failed")
        if self.max_loaded is not None and len(self.loaded) >= self.max_loaded:
            raise RuntimeError("hot slots exhausted")
        self.added.append(request)
        self.loaded.add(request.lora_int_id)
        self.add_started.setdefault(request.lora_path, asyncio.Event()).set()
        blocker = self.add_release.get(request.lora_path)
        if blocker is not None:
            await blocker.wait()
        return True

    async def pin_lora(self, int_id: int) -> bool:
        self.pinned.append(int_id)
        return True

    async def remove_lora(self, int_id: int) -> bool:
        self.removed.append(int_id)
        remaining_failures = self.fail_remove_count.get(int_id, 0)
        if remaining_failures > 0:
            self.fail_remove_count[int_id] = remaining_failures - 1
            return False
        self.loaded.discard(int_id)
        return True

    async def generate(
        self,
        prompt: dict[str, Any],
        sampling_params: Any,
        request_id: str,
        *,
        lora_request: _FakeLoRARequest,
        **engine_kwargs: Any,
    ):
        record = {
            "prompt": prompt,
            "sampling_params": sampling_params,
            "request_id": request_id,
            "lora_request": lora_request,
            "engine_kwargs": engine_kwargs,
        }
        self.generations.append(record)
        self.started.setdefault(lora_request.lora_int_id, asyncio.Event()).set()
        blocker = self.release.get(lora_request.lora_int_id)
        if blocker is not None:
            await blocker.wait()
        yield {
            "request_id": request_id,
            "lora_int_id": lora_request.lora_int_id,
            "lora_name": lora_request.lora_name,
        }


def _adapter_dir(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    return path


def _manager(engine: _FakeEngine, *, run_capacity: int = 2) -> SharedMultiLoRARolloutEngine:
    return SharedMultiLoRARolloutEngine(
        engine,
        run_capacity=run_capacity,
        lora_request_factory=_FakeLoRARequest,
        request_id_factory=lambda: "generated-request-id",
    )


def test_shared_vllm_kwargs_preserve_existing_hooks_and_add_n_plus_one_slots():
    kwargs = shared_vllm_engine_kwargs(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "language_model_only": True,
            "attention_backend": "TRITON_ATTN",
            "enable_prefix_caching": True,
        },
        run_capacity=4,
        max_lora_rank=64,
    )

    assert kwargs == {
        "model": "Qwen/Qwen3.5-0.8B",
        "language_model_only": True,
        "attention_backend": "TRITON_ATTN",
        "enable_prefix_caching": True,
        "enable_lora": True,
        "max_loras": 5,
        "max_cpu_loras": 5,
        "max_lora_rank": 64,
    }


def test_adapter_handles_are_immutable_and_versions_increase(tmp_path):
    async def scenario():
        engine = _FakeEngine()
        manager = _manager(engine)
        first = await manager.register_run("run-a", 3, _adapter_dir(tmp_path, "a-v3"))
        second = await manager.publish_adapter("run-a", 3, 5, _adapter_dir(tmp_path, "a-v5"))

        assert first.version == 3
        assert second.version == 5
        assert first.run_slot == second.run_slot
        assert first.lora_int_id != second.lora_int_id
        assert first.lora_name != second.lora_name
        with pytest.raises(FrozenInstanceError):
            first.version = 4  # type: ignore[misc]
        with pytest.raises(AdapterVersionError, match="increase monotonically"):
            await manager.publish_adapter("run-a", 5, 5, _adapter_dir(tmp_path, "a-v5-copy"))

    asyncio.run(scenario())


def test_request_routes_to_the_correct_runs_adapter(tmp_path):
    async def scenario():
        engine = _FakeEngine()
        manager = _manager(engine)
        run_a = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        run_b = await manager.register_run("run-b", 0, _adapter_dir(tmp_path, "b-v0"))

        result_a, result_b = await asyncio.gather(
            manager.generate(run_a, {"prompt_token_ids": [1]}, {"temperature": 0}),
            manager.generate(run_b, {"prompt_token_ids": [2]}, {"temperature": 0}),
        )

        assert result_a.handle == run_a
        assert result_b.handle == run_b
        assert result_a.output["lora_int_id"] == run_a.lora_int_id
        assert result_b.output["lora_int_id"] == run_b.lora_int_id
        assert {call["lora_request"].lora_path for call in engine.generations} == {
            run_a.adapter_dir,
            run_b.adapter_dir,
        }

    asyncio.run(scenario())


def test_publish_while_draining_keeps_inflight_rollout_on_old_version(tmp_path):
    async def scenario():
        engine = _FakeEngine()
        manager = _manager(engine, run_capacity=1)
        old = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        engine.release[old.lora_int_id] = asyncio.Event()

        rollout = asyncio.create_task(
            manager.generate(old, {"prompt_token_ids": [7]}, {"temperature": 0})
        )
        await engine.started.setdefault(old.lora_int_id, asyncio.Event()).wait()
        new = await manager.publish_adapter("run-a", 0, 1, _adapter_dir(tmp_path, "a-v1"))

        assert await manager.current_handle("run-a") == new
        assert old.lora_int_id not in engine.removed
        health = await manager.health()
        old_health = next(item for item in health.adapters if item.handle == old)
        assert old_health.in_flight == 1
        assert old_health.stale is True
        assert old_health.current is False

        engine.release[old.lora_int_id].set()
        result = await rollout

        assert result.handle == old
        assert result.output["lora_int_id"] == old.lora_int_id
        assert old.lora_int_id in engine.removed
        next_result = await manager.generate(new, {"prompt_token_ids": [7]}, {"temperature": 0})
        assert next_result.output["lora_int_id"] == new.lora_int_id

    asyncio.run(scenario())


def test_stale_version_is_evicted_only_after_every_inflight_request_drains(tmp_path):
    async def scenario():
        engine = _FakeEngine()
        manager = _manager(engine, run_capacity=1)
        old = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        engine.release[old.lora_int_id] = asyncio.Event()

        first = asyncio.create_task(
            manager.generate(old, {"prompt_token_ids": [1]}, {"temperature": 0})
        )
        second = asyncio.create_task(
            manager.generate(old, {"prompt_token_ids": [2]}, {"temperature": 0})
        )
        while True:
            health = await manager.health()
            old_health = next(item for item in health.adapters if item.handle == old)
            if old_health.in_flight == 2:
                break
            await asyncio.sleep(0)

        await manager.publish_adapter("run-a", 0, 1, _adapter_dir(tmp_path, "a-v1"))
        assert old.lora_int_id not in engine.removed

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert old.lora_int_id not in engine.removed

        engine.release[old.lora_int_id].set()
        await second
        assert engine.removed.count(old.lora_int_id) == 1

    asyncio.run(scenario())


def test_n_plus_one_slot_blocks_another_publish_until_stale_version_drains(tmp_path):
    async def scenario():
        engine = _FakeEngine()
        manager = _manager(engine, run_capacity=2)
        run_a = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        run_b = await manager.register_run("run-b", 0, _adapter_dir(tmp_path, "b-v0"))
        engine.release[run_a.lora_int_id] = asyncio.Event()

        rollout = asyncio.create_task(
            manager.generate(run_a, {"prompt_token_ids": [1]}, {"temperature": 0})
        )
        await engine.started.setdefault(run_a.lora_int_id, asyncio.Event()).wait()
        await manager.publish_adapter("run-a", 0, 1, _adapter_dir(tmp_path, "a-v1"))

        publish_b = asyncio.create_task(
            manager.publish_adapter("run-b", 0, 1, _adapter_dir(tmp_path, "b-v1"))
        )
        await asyncio.sleep(0)
        assert publish_b.done() is False
        assert len((await manager.health()).adapters) == manager.hot_slot_capacity

        engine.release[run_a.lora_int_id].set()
        await rollout
        run_b_v1 = await publish_b
        assert run_b_v1.version == 1
        assert run_a.lora_int_id in engine.removed
        assert await manager.current_handle("run-b") == run_b_v1

    asyncio.run(scenario())


def test_hot_slots_are_reclaimed_across_repeated_transient_eviction_failures(tmp_path):
    async def scenario():
        engine = _FakeEngine(max_loaded=2)
        manager = _manager(engine, run_capacity=1)
        current = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))

        for version in range(1, 11):
            engine.fail_remove_count[current.lora_int_id] = 1
            current = await manager.publish_adapter(
                "run-a",
                version - 1,
                version,
                _adapter_dir(tmp_path, f"a-v{version}"),
            )
            assert await manager.current_handle("run-a") == current
            assert len(engine.loaded) <= manager.hot_slot_capacity

        current = await manager.publish_adapter(
            "run-a",
            10,
            11,
            _adapter_dir(tmp_path, "a-v11"),
        )
        health = await manager.health()
        assert [item.handle for item in health.adapters] == [current]
        assert engine.loaded == {current.lora_int_id}

    asyncio.run(scenario())


def test_generate_releases_reference_when_request_setup_raises(tmp_path):
    async def scenario():
        engine = _FakeEngine(max_loaded=2)
        manager = SharedMultiLoRARolloutEngine(
            engine,
            run_capacity=1,
            lora_request_factory=_FakeLoRARequest,
            request_id_factory=lambda: (_ for _ in ()).throw(RuntimeError("id failed")),
        )
        first = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))

        with pytest.raises(RuntimeError, match="id failed"):
            await manager.generate(first, {"prompt_token_ids": [1]}, {"temperature": 0})

        first_health = (await manager.health()).adapters[0]
        assert first_health.in_flight == 0
        second = await manager.publish_adapter("run-a", 0, 1, _adapter_dir(tmp_path, "a-v1"))
        assert [item.handle for item in (await manager.health()).adapters] == [second]

    asyncio.run(scenario())


def test_remove_run_drain_does_not_block_other_run_publication(tmp_path):
    async def scenario():
        engine = _FakeEngine(max_loaded=3)
        manager = _manager(engine, run_capacity=2)
        run_a = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        await manager.register_run("run-b", 0, _adapter_dir(tmp_path, "b-v0"))
        engine.release[run_a.lora_int_id] = asyncio.Event()

        rollout = asyncio.create_task(
            manager.generate(run_a, {"prompt_token_ids": [1]}, {"temperature": 0})
        )
        await engine.started.setdefault(run_a.lora_int_id, asyncio.Event()).wait()
        removal = asyncio.create_task(manager.remove_run("run-a", 0))
        while True:
            with contextlib.suppress(AdapterRegistryError):
                await manager.current_handle("run-a")
                await asyncio.sleep(0)
                continue
            break

        run_b_v1 = await asyncio.wait_for(
            manager.publish_adapter("run-b", 0, 1, _adapter_dir(tmp_path, "b-v1")),
            timeout=1,
        )
        assert run_b_v1.version == 1
        assert removal.done() is False

        engine.release[run_a.lora_int_id].set()
        await rollout
        await removal

    asyncio.run(scenario())


def test_replacement_run_can_register_while_removed_run_drains(tmp_path):
    async def scenario():
        engine = _FakeEngine(max_loaded=2)
        manager = _manager(engine, run_capacity=1)
        run_a = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        engine.release[run_a.lora_int_id] = asyncio.Event()

        rollout = asyncio.create_task(
            manager.generate(run_a, {"prompt_token_ids": [1]}, {"temperature": 0})
        )
        await engine.started.setdefault(run_a.lora_int_id, asyncio.Event()).wait()
        removal = asyncio.create_task(manager.remove_run("run-a", 0))
        while True:
            with contextlib.suppress(AdapterRegistryError):
                await manager.current_handle("run-a")
                await asyncio.sleep(0)
                continue
            break

        run_b = await manager.register_run("run-b", 0, _adapter_dir(tmp_path, "b-v0"))
        assert run_b.run_slot == run_a.run_slot
        assert removal.done() is False

        engine.release[run_a.lora_int_id].set()
        await rollout
        await removal
        assert await manager.current_handle("run-b") == run_b

    asyncio.run(scenario())


def test_cancelled_remove_finishes_drain_and_cleanup(tmp_path):
    async def scenario():
        engine = _FakeEngine(max_loaded=2)
        manager = _manager(engine, run_capacity=1)
        run_a = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        engine.release[run_a.lora_int_id] = asyncio.Event()

        rollout = asyncio.create_task(
            manager.generate(run_a, {"prompt_token_ids": [1]}, {"temperature": 0})
        )
        await engine.started.setdefault(run_a.lora_int_id, asyncio.Event()).wait()
        removal = asyncio.create_task(manager.remove_run("run-a", 0))
        while True:
            with contextlib.suppress(AdapterRegistryError):
                await manager.current_handle("run-a")
                await asyncio.sleep(0)
                continue
            break

        removal.cancel()
        engine.release[run_a.lora_int_id].set()
        await rollout
        with pytest.raises(asyncio.CancelledError):
            await removal

        assert (await manager.health()).adapters == ()
        assert run_a.lora_int_id not in engine.loaded

    asyncio.run(scenario())


def test_remove_run_does_not_swallow_eviction_cancellation(tmp_path):
    async def scenario():
        engine = _FakeEngine(max_loaded=2)
        manager = _manager(engine, run_capacity=1)
        await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        remove_started = asyncio.Event()
        release_remove = asyncio.Event()

        async def cancel_remove(_int_id: int) -> bool:
            remove_started.set()
            await release_remove.wait()
            raise asyncio.CancelledError

        async def wait_for_notification() -> None:
            async with manager._condition:
                await manager._condition.wait()

        engine.remove_lora = cancel_remove
        removal = asyncio.create_task(manager.remove_run("run-a", 0))
        await remove_started.wait()
        waiter = asyncio.create_task(wait_for_notification())
        await asyncio.sleep(0)
        release_remove.set()
        with pytest.raises(asyncio.CancelledError):
            await removal
        await asyncio.wait_for(waiter, timeout=1)

    asyncio.run(scenario())


def test_remove_eviction_failure_is_deferred_without_blocking_free_capacity(tmp_path):
    async def scenario():
        engine = _FakeEngine(max_loaded=3)
        manager = _manager(engine, run_capacity=2)
        run_a = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        await manager.register_run("run-b", 0, _adapter_dir(tmp_path, "b-v0"))
        engine.fail_remove_count[run_a.lora_int_id] = 2

        await manager.remove_run("run-a", 0)
        assert engine.fail_remove_count[run_a.lora_int_id] == 1
        with pytest.raises(AdapterRegistryError, match="unknown run"):
            await manager.current_handle("run-a")

        run_b_v1 = await manager.publish_adapter("run-b", 0, 1, _adapter_dir(tmp_path, "b-v1"))
        assert run_b_v1.version == 1
        assert engine.fail_remove_count[run_a.lora_int_id] == 1

    asyncio.run(scenario())


def test_run_id_normalization_is_consistent_for_every_operation(tmp_path):
    async def scenario():
        engine = _FakeEngine()
        manager = _manager(engine, run_capacity=1)
        first = await manager.register_run("  run-a  ", 0, _adapter_dir(tmp_path, "a-v0"))
        assert first.run_id == "run-a"
        assert await manager.current_handle(" run-a ") == first

        second = await manager.publish_adapter(" run-a ", 0, 1, _adapter_dir(tmp_path, "a-v1"))
        assert await manager.current_handle("  run-a") == second
        await manager.remove_run("run-a  ", 1)
        assert (await manager.health()).adapters == ()

    asyncio.run(scenario())


def test_cancelled_publication_finishes_load_to_registry_transition(tmp_path):
    async def scenario():
        engine = _FakeEngine(max_loaded=2)
        manager = _manager(engine, run_capacity=1)
        first = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        second_dir = _adapter_dir(tmp_path, "a-v1")
        resolved_second_dir = str(second_dir.resolve())
        engine.add_started[resolved_second_dir] = asyncio.Event()
        engine.add_release[resolved_second_dir] = asyncio.Event()

        publication = asyncio.create_task(manager.publish_adapter("run-a", 0, 1, second_dir))
        await engine.add_started[resolved_second_dir].wait()
        publication.cancel()
        engine.add_release[resolved_second_dir].set()
        with pytest.raises(asyncio.CancelledError):
            await publication

        current = await manager.current_handle("run-a")
        assert current.version == 1
        assert current.lora_int_id in engine.loaded
        assert first.lora_int_id not in engine.loaded
        assert [item.handle for item in (await manager.health()).adapters] == [current]

    asyncio.run(scenario())


def test_prefix_cache_namespace_isolated_for_every_adapter_version(tmp_path):
    async def scenario():
        engine = _FakeEngine()
        manager = _manager(engine, run_capacity=1)
        first = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        await manager.generate(first, {"prompt_token_ids": [1, 2]}, {"temperature": 0})
        second = await manager.publish_adapter("run-a", 0, 1, _adapter_dir(tmp_path, "a-v1"))
        await manager.generate(second, {"prompt_token_ids": [1, 2]}, {"temperature": 0})

        first_call, second_call = engine.generations
        assert first.prefix_cache_namespace != second.prefix_cache_namespace
        assert first_call["prompt"]["cache_salt"] == first.prefix_cache_namespace
        assert second_call["prompt"]["cache_salt"] == second.prefix_cache_namespace
        assert first_call["lora_request"].lora_name != second_call["lora_request"].lora_name
        assert first_call["lora_request"].lora_int_id != second_call["lora_request"].lora_int_id

    asyncio.run(scenario())


def test_failed_publication_leaves_old_handle_current_and_routable(tmp_path):
    async def scenario():
        engine = _FakeEngine()
        manager = _manager(engine, run_capacity=1)
        old = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        failed_dir = _adapter_dir(tmp_path, "a-v1")
        engine.fail_add_path = str(failed_dir.resolve())

        with pytest.raises(RuntimeError, match="load failed"):
            await manager.publish_adapter("run-a", 0, 1, failed_dir)

        assert await manager.current_handle("run-a") == old
        result = await manager.generate(old, {"prompt_token_ids": [3]}, {"temperature": 0})
        assert result.output["lora_int_id"] == old.lora_int_id
        assert old.lora_int_id not in engine.removed
        assert len(engine.removed) == 1

    asyncio.run(scenario())


def test_run_identity_and_lora_ids_are_never_reused_within_engine_lifetime(tmp_path):
    async def scenario():
        engine = _FakeEngine()
        manager = _manager(engine, run_capacity=1)
        old = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        await manager.remove_run("run-a", 0)

        with pytest.raises(AdapterRegistryError, match="already registered"):
            await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0-copy"))
        assert old.lora_int_id in engine.removed

    asyncio.run(scenario())


def test_published_old_handle_cannot_admit_a_new_rollout(tmp_path):
    async def scenario():
        engine = _FakeEngine()
        manager = _manager(engine, run_capacity=1)
        old = await manager.register_run("run-a", 0, _adapter_dir(tmp_path, "a-v0"))
        await manager.publish_adapter("run-a", 0, 1, _adapter_dir(tmp_path, "a-v1"))

        with pytest.raises(UnknownAdapterHandle, match="not current"):
            await manager.generate(old, {"prompt_token_ids": [3]}, {"temperature": 0})

    asyncio.run(scenario())


def test_adapter_directory_cannot_be_reused_for_another_immutable_version(tmp_path):
    async def scenario():
        engine = _FakeEngine()
        manager = _manager(engine, run_capacity=1)
        adapter_dir = _adapter_dir(tmp_path, "shared")
        await manager.register_run("run-a", 0, adapter_dir)

        with pytest.raises(AdapterRegistryError, match="already published"):
            await manager.publish_adapter("run-a", 0, 1, adapter_dir)

        first = await manager.current_handle("run-a")
        await manager.remove_run("run-a", 0)
        second = await manager.register_run("run-b", 0, adapter_dir)
        assert second.adapter_dir == first.adapter_dir
        assert second.lora_int_id != first.lora_int_id

    asyncio.run(scenario())


def test_adapter_directory_validation_is_cpu_only(tmp_path):
    async def scenario():
        engine = _FakeEngine()
        manager = _manager(engine, run_capacity=1)
        invalid = tmp_path / "invalid"
        invalid.mkdir()
        (invalid / "adapter_config.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="non-empty adapter tensor"):
            await manager.register_run("run-a", 0, invalid)
        with pytest.raises(AdapterRegistryError, match="unknown run"):
            await manager.current_handle("run-a")

    asyncio.run(scenario())

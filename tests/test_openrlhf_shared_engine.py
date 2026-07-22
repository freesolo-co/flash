"""cpu-only tests for the shared OpenRLHF multi-LoRA rollout engine."""

from __future__ import annotations

import asyncio
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
    def __init__(self) -> None:
        self.added: list[_FakeLoRARequest] = []
        self.pinned: list[int] = []
        self.removed: list[int] = []
        self.generations: list[dict[str, Any]] = []
        self.started: dict[int, asyncio.Event] = {}
        self.release: dict[int, asyncio.Event] = {}
        self.fail_add_path: str | None = None

    async def add_lora(self, request: _FakeLoRARequest) -> bool:
        if request.lora_path == self.fail_add_path:
            raise RuntimeError("load failed")
        self.added.append(request)
        return True

    async def pin_lora(self, int_id: int) -> bool:
        self.pinned.append(int_id)
        return True

    async def remove_lora(self, int_id: int) -> bool:
        self.removed.append(int_id)
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

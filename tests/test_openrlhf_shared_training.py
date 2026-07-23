"""cpu tests for isolated shared-base OpenRLHF LoRA training state."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from flash.engine.worker import openrlhf_shared_training as shared_training
from flash.engine.worker.openrlhf_shared_engine import (
    AdapterHandle,
    AdapterRegistryError,
    SharedMultiLoRARolloutEngine,
)
from flash.engine.worker.openrlhf_shared_training import (
    RunOptimizerConfig,
    SharedMultiLoRATrainingActor,
    TrainingIsolationError,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)


@dataclass(frozen=True)
class _FakeLoRARequest:
    lora_name: str
    lora_int_id: int
    lora_path: str


class _FakeRolloutEngine:
    def __init__(self) -> None:
        self.added: list[_FakeLoRARequest] = []
        self.removed: list[int] = []
        self.fail_adds = 0
        self.add_started: asyncio.Event | None = None
        self.block_add: asyncio.Event | None = None

    async def add_lora(self, request: _FakeLoRARequest) -> bool:
        if self.add_started is not None:
            self.add_started.set()
        if self.block_add is not None:
            await self.block_add.wait()
        if self.fail_adds:
            self.fail_adds -= 1
            raise RuntimeError("adapter publish failed")
        self.added.append(request)
        return True

    async def pin_lora(self, _int_id: int) -> bool:
        return True

    async def remove_lora(self, int_id: int) -> bool:
        self.removed.append(int_id)
        return True


def _rollout_manager(
    capacity: int, engine: _FakeRolloutEngine | None = None
) -> SharedMultiLoRARolloutEngine:
    return SharedMultiLoRARolloutEngine(
        engine or _FakeRolloutEngine(),
        run_capacity=capacity,
        lora_request_factory=_FakeLoRARequest,
    )


def _serialized_state(value: Any) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _is_directory(path: str) -> bool:
    return Path(path).is_dir()


def _parameter_bytes(parameters: tuple[Any, ...]) -> tuple[bytes, ...]:
    return tuple(
        parameter.detach().cpu().contiguous().view(-1).numpy().tobytes() for parameter in parameters
    )


def _build_actor(
    tmp_path: Path,
    capacity: int = 3,
    *,
    rollout_manager: SharedMultiLoRARolloutEngine | None = None,
    fail_first_attach: bool = False,
):
    import torch

    class _TinyAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_a = torch.nn.Linear(2, 1, bias=False)
            self.lora_b = torch.nn.Linear(1, 2, bias=False)
            torch.nn.init.constant_(self.lora_a.weight, 0.5)
            torch.nn.init.zeros_(self.lora_b.weight)

        def forward(self, inputs):
            return self.lora_b(self.lora_a(inputs))

    class _TinySharedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.zeros_(self.base.weight)
            self.adapters = torch.nn.ModuleDict()
            self.active_adapter: str | None = None
            self.eval()

        def add_adapter(self, adapter_name: str) -> None:
            self.adapters[adapter_name] = _TinyAdapter()

        def delete_adapter(self, adapter_name: str) -> None:
            del self.adapters[adapter_name]

        def set_adapter(self, adapter_name: str) -> None:
            self.active_adapter = adapter_name

        def forward(self, inputs):
            if self.active_adapter is None:
                raise RuntimeError("no active adapter")
            return self.base(inputs) + self.adapters[self.active_adapter](inputs)

    load_count = 0
    attach_failures = int(fail_first_attach)

    def load_base():
        nonlocal load_count
        load_count += 1
        return _TinySharedModel()

    def attach_adapter(model, adapter_name, _config):
        nonlocal attach_failures
        model.add_adapter(adapter_name)
        if attach_failures:
            attach_failures -= 1
            raise RuntimeError("adapter attach failed")
        return model

    def export_adapter(model, adapter_name, output_dir):
        adapter = model.adapters[adapter_name]
        (output_dir / "adapter_config.json").write_text(
            json.dumps({"r": 1, "adapter_name": adapter_name}), encoding="utf-8"
        )
        torch.save(adapter.state_dict(), output_dir / "adapter_model.bin")

    def optimizer_factory(parameters, config):
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    rollout_manager = rollout_manager or _rollout_manager(capacity)
    actor = SharedMultiLoRATrainingActor(
        load_base,
        rollout_manager,
        tmp_path / "published",
        optimizer_factory=optimizer_factory,
        adapter_attacher=attach_adapter,
        adapter_exporter=export_adapter,
    )
    return actor, lambda: load_count, rollout_manager


async def _register(actor, run_id: str):
    return await actor.register_run(
        run_id,
        optimizer_config=RunOptimizerConfig(learning_rate=0.1),
        dataloader=[f"{run_id}-prompt"],
        lora_config=object(),
    )


def _loss_for(target: float):
    import torch

    inputs = torch.tensor([[1.0, 1.0]])
    expected = torch.full((1, 2), target)

    def loss(model, _state):
        return torch.nn.functional.mse_loss(model(inputs), expected)

    return loss


@requires_torch
def test_registering_n_runs_loads_one_frozen_base_and_attaches_n_adapters(tmp_path):
    async def scenario():
        actor, load_count, _rollout_manager = _build_actor(tmp_path, capacity=3)
        states = [await _register(actor, f"run-{index}") for index in range(3)]

        assert load_count() == 1
        assert actor.run_ids == ("run-0", "run-1", "run-2")
        assert len(actor.model.adapters) == 3
        assert len({state.adapter_name for state in states}) == 3
        assert all(not parameter.requires_grad for parameter in actor.model.base.parameters())
        parameter_sets = [
            {id(parameter) for parameter in state.adapter_parameters} for state in states
        ]
        assert all(
            parameter_sets[index].isdisjoint(parameter_sets[other])
            for index in range(3)
            for other in range(index)
        )

    asyncio.run(scenario())


@requires_torch
def test_failed_adapter_attach_removes_partial_adapter_and_allows_retry(tmp_path):
    async def scenario():
        actor, _load_count, _rollout_manager = _build_actor(
            tmp_path, capacity=1, fail_first_attach=True
        )

        with pytest.raises(RuntimeError, match="adapter attach failed"):
            await _register(actor, "run-a")

        assert actor.run_ids == ()
        assert len(actor.model.adapters) == 0
        state = await _register(actor, "run-a")
        assert state.run_id == "run-a"
        assert len(actor.model.adapters) == 1

    asyncio.run(scenario())


@requires_torch
def test_cancelled_registration_commits_matching_actor_and_rollout_state(tmp_path):
    async def scenario():
        rollout_engine = _FakeRolloutEngine()
        rollout_engine.add_started = asyncio.Event()
        rollout_engine.block_add = asyncio.Event()
        rollout_manager = _rollout_manager(1, rollout_engine)
        actor, _load_count, _manager = _build_actor(
            tmp_path, capacity=1, rollout_manager=rollout_manager
        )

        registration = asyncio.create_task(_register(actor, "run-a"))
        await rollout_engine.add_started.wait()
        registration.cancel()
        rollout_engine.block_add.set()
        with pytest.raises(asyncio.CancelledError):
            await registration

        state = actor.run_state("run-a")
        assert state.adapter_version == 0
        assert state.handle == await rollout_manager.current_handle("run-a")
        assert _is_directory(state.handle.adapter_dir)
        assert len(actor.model.adapters) == 1

    asyncio.run(scenario())


@requires_torch
def test_remove_run_retires_training_rollout_and_export_state(tmp_path):
    async def scenario():
        actor, _load_count, rollout_manager = _build_actor(tmp_path, capacity=2)
        run_a = await _register(actor, "run-a")
        run_b = await _register(actor, "run-b")
        run_a_dir = Path(run_a.handle.adapter_dir).parent
        run_b_snapshot = actor.adapter_parameter_snapshot("run-b")

        await actor.remove_run("run-a")

        assert actor.run_ids == ("run-b",)
        assert run_a.adapter_name not in actor.model.adapters
        assert not run_a_dir.exists()
        with pytest.raises(KeyError, match="unknown training run"):
            actor.run_state("run-a")
        with pytest.raises(AdapterRegistryError, match="unknown run"):
            await rollout_manager.current_handle("run-a")
        assert actor.adapter_parameter_snapshot("run-b") == run_b_snapshot
        assert await rollout_manager.current_handle("run-b") == run_b.handle

    asyncio.run(scenario())


@requires_torch
def test_step_mutates_only_active_adapter_and_optimizer_then_publishes(tmp_path):
    async def scenario():
        actor, _load_count, rollout_manager = _build_actor(tmp_path, capacity=2)
        run_a = await _register(actor, "run-a")
        run_b = await _register(actor, "run-b")

        base_before = _parameter_bytes(tuple(actor.model.base.parameters()))
        a_parameters_before = actor.adapter_parameter_snapshot("run-a")
        b_parameters_before = actor.adapter_parameter_snapshot("run-b")
        a_optimizer_before = _serialized_state(run_a.optimizer.state_dict())
        b_optimizer_before = _serialized_state(run_b.optimizer.state_dict())

        result = await actor.step("run-a", _loss_for(1.0))

        assert actor.adapter_parameter_snapshot("run-a") != a_parameters_before
        assert _serialized_state(run_a.optimizer.state_dict()) != a_optimizer_before
        assert actor.adapter_parameter_snapshot("run-b") == b_parameters_before
        assert _serialized_state(run_b.optimizer.state_dict()) == b_optimizer_before
        assert _parameter_bytes(tuple(actor.model.base.parameters())) == base_before
        assert all(parameter.grad is None for parameter in run_b.adapter_parameters)
        assert all(not parameter.requires_grad for parameter in actor.model.parameters())
        assert actor.model.training is True
        assert isinstance(result.handle, AdapterHandle)
        assert result.handle.run_id == "run-a"
        assert result.handle.version == 1
        assert run_a.handle == result.handle
        assert await rollout_manager.current_handle("run-a") == result.handle

    asyncio.run(scenario())


@requires_torch
def test_failed_publish_retries_same_update_without_cross_run_mutation(tmp_path):
    async def scenario():
        rollout_engine = _FakeRolloutEngine()
        rollout_manager = _rollout_manager(2, rollout_engine)
        actor, _load_count, _manager = _build_actor(
            tmp_path, capacity=2, rollout_manager=rollout_manager
        )
        run_a = await _register(actor, "run-a")
        run_b = await _register(actor, "run-b")
        old_a_handle = run_a.handle
        base_before = _parameter_bytes(tuple(actor.model.base.parameters()))
        b_parameters_before = actor.adapter_parameter_snapshot("run-b")
        b_optimizer_before = _serialized_state(run_b.optimizer.state_dict())
        rollout_engine.fail_adds = 1

        with pytest.raises(RuntimeError, match="adapter publish failed"):
            await actor.step("run-a", _loss_for(1.0))

        a_parameters_after_update = actor.adapter_parameter_snapshot("run-a")
        a_optimizer_after_update = _serialized_state(run_a.optimizer.state_dict())
        assert run_a.adapter_version == 0
        assert run_a.global_step == 0
        assert run_a.handle == old_a_handle
        assert run_a.pending_publication is not None
        assert actor.adapter_parameter_snapshot("run-b") == b_parameters_before
        assert _serialized_state(run_b.optimizer.state_dict()) == b_optimizer_before
        assert _parameter_bytes(tuple(actor.model.base.parameters())) == base_before
        with pytest.raises(TrainingIsolationError, match="must publish adapter version 1"):
            await actor.step("run-a", _loss_for(-1.0))

        new_handle = await actor.publish_pending_adapter("run-a")

        assert new_handle.version == 1
        assert run_a.adapter_version == 1
        assert run_a.global_step == 1
        assert run_a.pending_publication is None
        assert actor.adapter_parameter_snapshot("run-a") == a_parameters_after_update
        assert _serialized_state(run_a.optimizer.state_dict()) == a_optimizer_after_update
        assert actor.adapter_parameter_snapshot("run-b") == b_parameters_before
        assert _serialized_state(run_b.optimizer.state_dict()) == b_optimizer_before
        assert _parameter_bytes(tuple(actor.model.base.parameters())) == base_before
        assert await rollout_manager.current_handle("run-a") == new_handle

    asyncio.run(scenario())


@requires_torch
def test_scheduler_failure_blocks_duplicate_optimizer_update(tmp_path):
    class _FailingScheduler:
        def __init__(self):
            self.failures = 1
            self.steps = 0

        def step(self):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("scheduler failed")
            self.steps += 1

        def state_dict(self):
            return {"steps": self.steps}

        def load_state_dict(self, state):
            self.steps = state["steps"]

    async def scenario():
        actor, _load_count, _rollout_manager = _build_actor(tmp_path, capacity=1)
        state = await actor.register_run(
            "run-a",
            optimizer_config=RunOptimizerConfig(learning_rate=0.1),
            dataloader=["run-a-prompt"],
            lora_config=object(),
            scheduler_factory=lambda _optimizer: _FailingScheduler(),
        )

        with pytest.raises(RuntimeError, match="scheduler failed"):
            await actor.step("run-a", _loss_for(1.0))

        assert state.pending_publication is not None
        assert state.pending_publication.scheduler_step_pending is True
        assert state.lr_scheduler.steps == 0
        assert state.adapter_version == 0
        assert state.global_step == 0
        with pytest.raises(TrainingIsolationError, match="must publish adapter version 1"):
            await actor.step("run-a", _loss_for(-1.0))

        handle = await actor.publish_pending_adapter("run-a")

        assert handle.version == 1
        assert state.pending_publication is None
        assert state.lr_scheduler.steps == 1
        assert state.adapter_version == 1
        assert state.global_step == 1

    asyncio.run(scenario())


@requires_torch
@pytest.mark.parametrize("fail_first_step", [False, True])
def test_scheduler_transition_state_is_prepared_before_advancing(
    monkeypatch, tmp_path, fail_first_step
):
    class _GuardedScheduler:
        def __init__(self):
            self.failures = int(fail_first_step)
            self.steps = 0

        def step(self):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("scheduler failed")
            self.steps += 1

        def state_dict(self):
            return {"steps": self.steps}

        def load_state_dict(self, state):
            self.steps = state["steps"]

    async def scenario():
        scheduler = _GuardedScheduler()
        actor, _load_count, _rollout_manager = _build_actor(tmp_path, capacity=1)
        state = await actor.register_run(
            "run-a",
            optimizer_config=RunOptimizerConfig(learning_rate=0.1),
            dataloader=["run-a-prompt"],
            lora_config=object(),
            scheduler_factory=lambda _optimizer: scheduler,
        )
        original_pending = shared_training._PendingAdapterPublication

        def guarded_pending(*args, **kwargs):
            if (
                scheduler.steps > 0
                and kwargs.get("adapter_dir") is None
                and kwargs.get("scheduler_step_pending") is False
            ):
                raise AssertionError("scheduler transition state was prepared too late")
            return original_pending(*args, **kwargs)

        monkeypatch.setattr(shared_training, "_PendingAdapterPublication", guarded_pending)
        if fail_first_step:
            with pytest.raises(RuntimeError, match="scheduler failed"):
                await actor.step("run-a", _loss_for(1.0))
            handle = await actor.publish_pending_adapter("run-a")
        else:
            handle = (await actor.step("run-a", _loss_for(1.0))).handle

        assert handle.version == 1
        assert scheduler.steps == 1
        assert state.pending_publication is None

    asyncio.run(scenario())


@requires_torch
def test_cancelled_step_publication_returns_committed_result(tmp_path):
    async def scenario():
        rollout_engine = _FakeRolloutEngine()
        rollout_manager = _rollout_manager(2, rollout_engine)
        actor, _load_count, _manager = _build_actor(
            tmp_path, capacity=2, rollout_manager=rollout_manager
        )
        run_a = await _register(actor, "run-a")
        run_b = await _register(actor, "run-b")
        base_before = _parameter_bytes(tuple(actor.model.base.parameters()))
        b_parameters_before = actor.adapter_parameter_snapshot("run-b")
        b_optimizer_before = _serialized_state(run_b.optimizer.state_dict())
        rollout_engine.add_started = asyncio.Event()
        rollout_engine.block_add = asyncio.Event()

        update = asyncio.create_task(actor.step("run-a", _loss_for(1.0)))
        await rollout_engine.add_started.wait()
        update.cancel()
        rollout_engine.block_add.set()
        result = await update

        assert result.global_step == 1
        assert run_a.adapter_version == 1
        assert run_a.global_step == 1
        assert run_a.pending_publication is None
        assert run_a.handle == await rollout_manager.current_handle("run-a")
        assert actor.adapter_parameter_snapshot("run-b") == b_parameters_before
        assert _serialized_state(run_b.optimizer.state_dict()) == b_optimizer_before
        assert _parameter_bytes(tuple(actor.model.base.parameters())) == base_before

    asyncio.run(scenario())


@requires_torch
def test_prompt_cursor_waits_for_checkpoint_save(monkeypatch, tmp_path):
    import torch

    async def scenario():
        actor, _load_count, _rollout_manager = _build_actor(tmp_path, capacity=1)
        await _register(actor, "run-a")
        save_started = threading.Event()
        save_release = threading.Event()
        original_save = shared_training._save_checkpoint_file

        def blocking_save(path, checkpoint):
            save_started.set()
            if not save_release.wait(timeout=5):
                raise TimeoutError("checkpoint save was not released")
            original_save(path, checkpoint)

        monkeypatch.setattr(shared_training, "_save_checkpoint_file", blocking_save)
        checkpoint_path = tmp_path / "cursor-save.pt"
        save = asyncio.create_task(actor.save_run_checkpoint("run-a", checkpoint_path))
        assert await asyncio.to_thread(save_started.wait, 5)
        advance = asyncio.create_task(actor.advance_prompt_cursor("run-a", 7))
        await asyncio.sleep(0)
        assert advance.done() is False

        save_release.set()
        checkpoint = await save
        assert await advance == 7
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["prompt_cursor"] == 0
        assert actor.run_state("run-a").prompt_cursor == 7

    asyncio.run(scenario())


@requires_torch
def test_cancelled_checkpoint_save_holds_lease_until_thread_finishes(monkeypatch, tmp_path):
    async def scenario():
        actor, _load_count, _rollout_manager = _build_actor(tmp_path, capacity=1)
        await _register(actor, "run-a")
        save_started = threading.Event()
        save_release = threading.Event()
        original_save = shared_training._save_checkpoint_file

        def blocking_save(path, checkpoint):
            save_started.set()
            if not save_release.wait(timeout=5):
                raise TimeoutError("checkpoint save was not released")
            original_save(path, checkpoint)

        monkeypatch.setattr(shared_training, "_save_checkpoint_file", blocking_save)
        checkpoint_path = tmp_path / "cancel-save.pt"
        save = asyncio.create_task(actor.save_run_checkpoint("run-a", checkpoint_path))
        assert await asyncio.to_thread(save_started.wait, 5)
        save.cancel()
        step = asyncio.create_task(actor.step("run-a", _loss_for(1.0)))
        await asyncio.sleep(0)
        assert save.done() is False
        assert step.done() is False

        save_release.set()
        with pytest.raises(asyncio.CancelledError):
            await save
        result = await step
        assert checkpoint_path.is_file()
        assert result.global_step == 1

    asyncio.run(scenario())


@requires_torch
def test_checkpoint_restore_is_independent_per_run(tmp_path):
    async def scenario():
        actor, _load_count, rollout_manager = _build_actor(tmp_path, capacity=2)
        run_a = await _register(actor, "run-a")
        run_b = await _register(actor, "run-b")

        await actor.step("run-a", _loss_for(1.0))
        await actor.advance_prompt_cursor("run-a", 7)
        checkpoint = await actor.save_run_checkpoint("run-a", tmp_path / "run-a.pt")
        saved_a_parameters = actor.adapter_parameter_snapshot("run-a")
        saved_a_optimizer = _serialized_state(run_a.optimizer.state_dict())
        saved_a_cursor = run_a.prompt_cursor
        saved_a_step = run_a.global_step

        await actor.step("run-a", _loss_for(-1.0))
        await actor.step("run-b", _loss_for(0.5))
        b_parameters_before_restore = actor.adapter_parameter_snapshot("run-b")
        b_optimizer_before_restore = _serialized_state(run_b.optimizer.state_dict())
        b_handle_before_restore = run_b.handle
        a_version_before_restore = run_a.adapter_version

        restored_handle = await actor.restore_run_checkpoint("run-a", checkpoint)

        assert actor.adapter_parameter_snapshot("run-a") == saved_a_parameters
        assert _serialized_state(run_a.optimizer.state_dict()) == saved_a_optimizer
        assert run_a.prompt_cursor == saved_a_cursor
        assert run_a.global_step == saved_a_step
        assert actor.adapter_parameter_snapshot("run-b") == b_parameters_before_restore
        assert _serialized_state(run_b.optimizer.state_dict()) == b_optimizer_before_restore
        assert run_b.handle == b_handle_before_restore
        assert restored_handle.version == a_version_before_restore + 1
        assert await rollout_manager.current_handle("run-a") == restored_handle
        assert await rollout_manager.current_handle("run-b") == b_handle_before_restore

    asyncio.run(scenario())


@requires_torch
def test_invalid_checkpoint_state_rolls_back_live_run_without_touching_peer(tmp_path):
    import torch

    async def scenario():
        actor, _load_count, rollout_manager = _build_actor(tmp_path, capacity=2)
        run_a = await _register(actor, "run-a")
        run_b = await _register(actor, "run-b")
        await actor.step("run-a", _loss_for(1.0))
        checkpoint = await actor.save_run_checkpoint("run-a", tmp_path / "invalid-run-a.pt")
        await actor.step("run-a", _loss_for(-1.0))
        await actor.step("run-b", _loss_for(0.5))

        checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checkpoint_state["optimizer"]["param_groups"] = []
        torch.save(checkpoint_state, checkpoint)
        a_parameters_before = actor.adapter_parameter_snapshot("run-a")
        a_optimizer_before = _serialized_state(run_a.optimizer.state_dict())
        a_scheduler_before = _serialized_state(run_a.lr_scheduler.state_dict())
        a_cursor_before = run_a.prompt_cursor
        a_step_before = run_a.global_step
        a_handle_before = run_a.handle
        b_parameters_before = actor.adapter_parameter_snapshot("run-b")
        b_optimizer_before = _serialized_state(run_b.optimizer.state_dict())
        b_handle_before = run_b.handle

        with pytest.raises(ValueError, match="different number of parameter groups"):
            await actor.restore_run_checkpoint("run-a", checkpoint)

        assert actor.adapter_parameter_snapshot("run-a") == a_parameters_before
        assert _serialized_state(run_a.optimizer.state_dict()) == a_optimizer_before
        assert _serialized_state(run_a.lr_scheduler.state_dict()) == a_scheduler_before
        assert run_a.prompt_cursor == a_cursor_before
        assert run_a.global_step == a_step_before
        assert run_a.handle == a_handle_before
        assert actor.adapter_parameter_snapshot("run-b") == b_parameters_before
        assert _serialized_state(run_b.optimizer.state_dict()) == b_optimizer_before
        assert run_b.handle == b_handle_before
        assert await rollout_manager.current_handle("run-a") == a_handle_before
        assert await rollout_manager.current_handle("run-b") == b_handle_before

    asyncio.run(scenario())


@requires_torch
def test_failed_restore_publish_rolls_back_and_can_retry(tmp_path):
    async def scenario():
        rollout_engine = _FakeRolloutEngine()
        rollout_manager = _rollout_manager(2, rollout_engine)
        actor, _load_count, _manager = _build_actor(
            tmp_path, capacity=2, rollout_manager=rollout_manager
        )
        run_a = await _register(actor, "run-a")
        run_b = await _register(actor, "run-b")
        await actor.step("run-a", _loss_for(1.0))
        checkpoint = await actor.save_run_checkpoint("run-a", tmp_path / "restore-run-a.pt")
        await actor.step("run-a", _loss_for(-1.0))
        await actor.step("run-b", _loss_for(0.5))
        a_parameters_before = actor.adapter_parameter_snapshot("run-a")
        a_optimizer_before = _serialized_state(run_a.optimizer.state_dict())
        a_scheduler_before = _serialized_state(run_a.lr_scheduler.state_dict())
        a_cursor_before = run_a.prompt_cursor
        a_step_before = run_a.global_step
        a_handle_before = run_a.handle
        b_parameters_before = actor.adapter_parameter_snapshot("run-b")
        b_optimizer_before = _serialized_state(run_b.optimizer.state_dict())
        b_handle_before = run_b.handle
        rollout_engine.fail_adds = 1

        with pytest.raises(RuntimeError, match="adapter publish failed"):
            await actor.restore_run_checkpoint("run-a", checkpoint)

        assert actor.adapter_parameter_snapshot("run-a") == a_parameters_before
        assert _serialized_state(run_a.optimizer.state_dict()) == a_optimizer_before
        assert _serialized_state(run_a.lr_scheduler.state_dict()) == a_scheduler_before
        assert run_a.prompt_cursor == a_cursor_before
        assert run_a.global_step == a_step_before
        assert run_a.handle == a_handle_before
        assert actor.adapter_parameter_snapshot("run-b") == b_parameters_before
        assert _serialized_state(run_b.optimizer.state_dict()) == b_optimizer_before
        assert run_b.handle == b_handle_before
        assert await rollout_manager.current_handle("run-a") == a_handle_before

        restored_handle = await actor.restore_run_checkpoint("run-a", checkpoint)

        assert restored_handle.version == a_handle_before.version + 1
        assert await rollout_manager.current_handle("run-a") == restored_handle
        assert await rollout_manager.current_handle("run-b") == b_handle_before

    asyncio.run(scenario())


@requires_torch
def test_cancelled_restore_publication_commits_restored_actor_and_rollout_state(tmp_path):
    async def scenario():
        rollout_engine = _FakeRolloutEngine()
        rollout_manager = _rollout_manager(2, rollout_engine)
        actor, _load_count, _manager = _build_actor(
            tmp_path, capacity=2, rollout_manager=rollout_manager
        )
        run_a = await _register(actor, "run-a")
        run_b = await _register(actor, "run-b")
        await actor.step("run-a", _loss_for(1.0))
        checkpoint = await actor.save_run_checkpoint("run-a", tmp_path / "cancel-restore-a.pt")
        restored_parameters = actor.adapter_parameter_snapshot("run-a")
        restored_optimizer = _serialized_state(run_a.optimizer.state_dict())
        restored_scheduler = _serialized_state(run_a.lr_scheduler.state_dict())
        restored_global_step = run_a.global_step
        await actor.step("run-a", _loss_for(-1.0))
        await actor.step("run-b", _loss_for(0.5))
        old_version = run_a.adapter_version
        b_parameters_before = actor.adapter_parameter_snapshot("run-b")
        b_optimizer_before = _serialized_state(run_b.optimizer.state_dict())
        b_handle_before = run_b.handle
        rollout_engine.add_started = asyncio.Event()
        rollout_engine.block_add = asyncio.Event()

        restore = asyncio.create_task(actor.restore_run_checkpoint("run-a", checkpoint))
        await rollout_engine.add_started.wait()
        restore.cancel()
        rollout_engine.block_add.set()
        with pytest.raises(asyncio.CancelledError):
            await restore

        assert actor.adapter_parameter_snapshot("run-a") == restored_parameters
        assert _serialized_state(run_a.optimizer.state_dict()) == restored_optimizer
        assert _serialized_state(run_a.lr_scheduler.state_dict()) == restored_scheduler
        assert run_a.global_step == restored_global_step
        assert run_a.adapter_version == old_version + 1
        assert run_a.handle == await rollout_manager.current_handle("run-a")
        assert actor.adapter_parameter_snapshot("run-b") == b_parameters_before
        assert _serialized_state(run_b.optimizer.state_dict()) == b_optimizer_before
        assert run_b.handle == b_handle_before

    asyncio.run(scenario())


@requires_torch
def test_default_peft_path_attaches_and_publishes_named_adapters(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")

    class _TinyBase(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)

        def forward(self, inputs):
            return self.linear(inputs)

    def optimizer_factory(parameters, config):
        return torch.optim.AdamW(parameters, lr=config.learning_rate)

    async def scenario():
        rollout_manager = _rollout_manager(2)
        actor = SharedMultiLoRATrainingActor(
            _TinyBase,
            rollout_manager,
            tmp_path / "peft-published",
            optimizer_factory=optimizer_factory,
        )
        config = peft.LoraConfig(
            r=1,
            lora_alpha=1,
            target_modules=["linear"],
            init_lora_weights=True,
        )
        run_a = await actor.register_run(
            "run-a",
            optimizer_config=RunOptimizerConfig(learning_rate=0.1),
            dataloader=["a"],
            lora_config=config,
        )
        run_b = await actor.register_run(
            "run-b",
            optimizer_config=RunOptimizerConfig(learning_rate=0.1),
            dataloader=["b"],
            lora_config=config,
        )
        b_before = actor.adapter_parameter_snapshot("run-b")

        result = await actor.step(
            "run-a", lambda model, _state: model(torch.ones(1, 2)).square().mean()
        )

        assert run_a.adapter_name in actor.model.peft_config
        assert run_b.adapter_name in actor.model.peft_config
        assert result.handle.version == 1
        assert actor.adapter_parameter_snapshot("run-b") == b_before
        assert await rollout_manager.current_handle("run-a") == result.handle

    asyncio.run(scenario())

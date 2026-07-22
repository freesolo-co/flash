"""cpu tests for isolated shared-base OpenRLHF LoRA training state."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from flash.engine.worker.openrlhf_shared_engine import (
    AdapterHandle,
    SharedMultiLoRARolloutEngine,
)
from flash.engine.worker.openrlhf_shared_training import (
    RunOptimizerConfig,
    SharedMultiLoRATrainingActor,
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

    async def add_lora(self, request: _FakeLoRARequest) -> bool:
        self.added.append(request)
        return True

    async def pin_lora(self, _int_id: int) -> bool:
        return True

    async def remove_lora(self, int_id: int) -> bool:
        self.removed.append(int_id)
        return True


def _rollout_manager(capacity: int) -> SharedMultiLoRARolloutEngine:
    return SharedMultiLoRARolloutEngine(
        _FakeRolloutEngine(),
        run_capacity=capacity,
        lora_request_factory=_FakeLoRARequest,
    )


def _serialized_state(value: Any) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _parameter_bytes(parameters: tuple[Any, ...]) -> tuple[bytes, ...]:
    return tuple(
        parameter.detach().cpu().contiguous().view(-1).numpy().tobytes() for parameter in parameters
    )


def _build_actor(tmp_path: Path, capacity: int = 3):
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

    def load_base():
        nonlocal load_count
        load_count += 1
        return _TinySharedModel()

    def attach_adapter(model, adapter_name, _config):
        model.add_adapter(adapter_name)
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

    rollout_manager = _rollout_manager(capacity)
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
        assert isinstance(result.handle, AdapterHandle)
        assert result.handle.run_id == "run-a"
        assert result.handle.version == 1
        assert run_a.handle == result.handle
        assert await rollout_manager.current_handle("run-a") == result.handle

    asyncio.run(scenario())


@requires_torch
def test_checkpoint_restore_is_independent_per_run(tmp_path):
    async def scenario():
        actor, _load_count, rollout_manager = _build_actor(tmp_path, capacity=2)
        run_a = await _register(actor, "run-a")
        run_b = await _register(actor, "run-b")

        await actor.step("run-a", _loss_for(1.0))
        actor.advance_prompt_cursor("run-a", 7)
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

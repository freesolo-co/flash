"""cpu parity and isolation tests for shared-engine GRPO integration."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from flash.engine.worker.grpo_openrlhf import (
    RewardBridge,
    RewardResult,
    openrlhf_grpo_loss,
)
from flash.engine.worker.openrlhf_shared_engine import SharedMultiLoRARolloutEngine
from flash.engine.worker.openrlhf_shared_grpo import (
    SharedGRPOConfig,
    SharedGRPOPrompt,
    SharedGRPORunAdapter,
)
from flash.engine.worker.openrlhf_shared_scheduler import RunPhase, SharedEngineRunController
from flash.engine.worker.openrlhf_shared_scoring import (
    ScoringBatchIdentity,
    ScoringKind,
    ScoringResult,
    SharedScoringPool,
)
from flash.engine.worker.openrlhf_shared_training import (
    RunOptimizerConfig,
    SharedMultiLoRATrainingActor,
)
from flash.engine.worker.rng import rollout_request_seed

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)


@dataclass(frozen=True)
class _FakeLoRARequest:
    lora_name: str
    lora_int_id: int
    lora_path: str


@dataclass(frozen=True)
class _SamplingParams:
    seed: int


class _FakeVLLMEngine:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def add_lora(self, _request: _FakeLoRARequest) -> bool:
        return True

    async def pin_lora(self, _int_id: int) -> bool:
        return True

    async def remove_lora(self, _int_id: int) -> bool:
        return True

    def generate(
        self,
        prompt,
        sampling_params,
        request_id,
        *,
        lora_request,
    ):
        version = int(lora_request.lora_name.split("-v", 1)[1].split("-", 1)[0])
        token_id = 10 + version + sampling_params.seed % 7
        self.requests.append(
            {
                "request_id": request_id,
                "lora_name": lora_request.lora_name,
                "seed": sampling_params.seed,
                "prompt_token_ids": tuple(prompt["prompt_token_ids"]),
            }
        )
        generated = SimpleNamespace(
            token_ids=[token_id, token_id + 1],
            logprobs=[
                {token_id: SimpleNamespace(logprob=-0.3)},
                {token_id + 1: SimpleNamespace(logprob=-0.4)},
            ],
            finish_reason="stop",
        )
        return SimpleNamespace(outputs=[generated])


class _TinyAdapterModel:
    def __init__(self, torch) -> None:
        self._torch = torch
        self.module = torch.nn.Module()
        self.module.base = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.constant_(self.module.base.weight, 0.05)
        self.module.adapters = torch.nn.ModuleDict()
        self.module.active_adapter = None

        def set_adapter(adapter_name: str) -> None:
            self.module.active_adapter = adapter_name

        def add_adapter(adapter_name: str) -> None:
            layer = torch.nn.Linear(2, 1, bias=False)
            torch.nn.init.constant_(layer.weight, 0.1)
            self.module.adapters[adapter_name] = layer

        def delete_adapter(adapter_name: str) -> None:
            del self.module.adapters[adapter_name]

        def forward(inputs):
            adapter_name = self.module.active_adapter
            if adapter_name is None:
                raise RuntimeError("no active adapter")
            return self.module.base(inputs) + self.module.adapters[adapter_name](inputs)

        self.module.set_adapter = set_adapter
        self.module.add_adapter = add_adapter
        self.module.delete_adapter = delete_adapter
        self.module.forward = forward

    def value(self):
        return self.module


def _optimizer_bytes(optimizer) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    return buffer.getvalue()


def _parameter_bytes(parameters) -> tuple[bytes, ...]:
    return tuple(parameter.detach().cpu().numpy().tobytes() for parameter in parameters)


def _build_runtime(tmp_path: Path, run_count: int):
    import torch

    backend = _FakeVLLMEngine()
    rollout_engine = SharedMultiLoRARolloutEngine(
        backend,
        run_capacity=run_count,
        lora_request_factory=_FakeLoRARequest,
    )

    def load_base():
        return _TinyAdapterModel(torch).value()

    def attach_adapter(model, adapter_name, _config):
        model.add_adapter(adapter_name)
        return model

    def export_adapter(model, adapter_name, output_dir):
        (output_dir / "adapter_config.json").write_text(
            json.dumps({"r": 1, "adapter_name": adapter_name}),
            encoding="utf-8",
        )
        torch.save(model.adapters[adapter_name].state_dict(), output_dir / "adapter_model.bin")

    def optimizer_factory(parameters, config):
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    actor = SharedMultiLoRATrainingActor(
        load_base,
        rollout_engine,
        tmp_path / "published",
        optimizer_factory=optimizer_factory,
        adapter_attacher=attach_adapter,
        adapter_exporter=export_adapter,
    )
    return actor, rollout_engine, backend


def _prompts(run_offset: int = 0) -> list[SharedGRPOPrompt]:
    return [
        SharedGRPOPrompt(
            rendered=f"{1 + run_offset} {2 + run_offset}",
            label=run_offset,
            token_ids=(1 + run_offset, 2 + run_offset),
        ),
        SharedGRPOPrompt(
            rendered=f"{3 + run_offset} {4 + run_offset}",
            label=run_offset + 1,
            token_ids=(3 + run_offset, 4 + run_offset),
        ),
    ]


async def _register(actor, run_id: str, prompts: list[SharedGRPOPrompt]):
    return await actor.register_run(
        run_id,
        optimizer_config=RunOptimizerConfig(learning_rate=0.01),
        dataloader=prompts,
        lora_config=object(),
    )


def _decode_tokens(tokens) -> str:
    return " ".join(str(token) for token in tokens)


def _policy_log_probs(model, _state, batch):
    import torch

    rows = []
    max_actions = batch.action_mask.shape[1]
    for row, (prompt_length, action_length) in enumerate(
        zip(batch.prompt_lengths, batch.action_lengths, strict=True)
    ):
        action_tokens = batch.sequences[row, prompt_length : prompt_length + action_length].float()
        features = torch.stack(
            (action_tokens / 100.0, torch.ones_like(action_tokens)),
            dim=-1,
        )
        values = model(features).squeeze(-1)
        if action_length < max_actions:
            values = torch.nn.functional.pad(values, (0, max_actions - action_length))
        rows.append(values)
    return torch.stack(rows)


def _config(seed: int = 123) -> SharedGRPOConfig:
    return SharedGRPOConfig(
        seed=seed,
        prompts_per_step=1,
        group_size=2,
        max_response_length=4,
        pad_token_id=0,
    )


def _driver(run_id, actor, engine, bridge, *, seed=123):
    return SharedGRPORunAdapter(
        run_id,
        config=_config(seed),
        training_actor=actor,
        rollout_engine=engine,
        reward_url=bridge.url,
        sampling_params_factory=_SamplingParams,
        decode_tokens=_decode_tokens,
        policy_log_probs=_policy_log_probs,
    )


@requires_torch
def test_single_run_shared_controller_matches_direct_grpo_loss_and_update(tmp_path):
    import torch

    async def scenario():
        torch.manual_seed(7)
        shared_actor, shared_engine, shared_backend = _build_runtime(tmp_path / "shared", 1)
        await _register(shared_actor, "run-a", _prompts())
        torch.manual_seed(7)
        direct_actor, direct_engine, direct_backend = _build_runtime(tmp_path / "direct", 1)
        await _register(direct_actor, "run-a", _prompts())

        def score(_label, completion, _prompt):
            reward = float(sum(int(part) for part in completion.split()))
            return RewardResult(reward, reward, {"exact": reward})

        with (
            RewardBridge(score, samples_per_step=2, first_step=1) as shared_bridge,
            RewardBridge(score, samples_per_step=2, first_step=1) as direct_bridge,
            SharedScoringPool(pool_size=1) as scoring_pool,
        ):
            shared_driver = _driver("run-a", shared_actor, shared_engine, shared_bridge)
            controller = SharedEngineRunController(scoring_pool, deficit_quantum_ms=1)
            shared_driver.add_to_controller(controller, total_steps=1)
            snapshots = await controller.drain(timeout_s=2)

            direct_driver = _driver("run-a", direct_actor, direct_engine, direct_bridge)
            rollout = await direct_driver.rollout("run-a", 0)
            rewards = direct_driver.score(direct_driver.scoring_payload("run-a", 0, rollout))
            reward_tensor = torch.tensor([result.reward for result in rewards.results])
            captured = []

            def direct_loss(model, state):
                model.eval()
                with torch.no_grad():
                    old_log_probs = _policy_log_probs(model, state, rollout.training_batch).detach()
                model.train()
                current_log_probs = _policy_log_probs(model, state, rollout.training_batch)
                objective = openrlhf_grpo_loss(
                    current_log_probs,
                    old_log_probs,
                    rollout.training_batch.rollout_log_probs,
                    reward_tensor,
                    rollout.training_batch.action_mask,
                    4,
                    2,
                )
                captured.append(objective)
                return objective.loss

            direct_step = await direct_actor.step("run-a", direct_loss)

        assert snapshots[0].phase is RunPhase.DONE
        assert shared_driver.last_update is not None
        assert shared_driver.last_update.training_step.loss == pytest.approx(
            direct_step.loss,
            abs=1e-6,
        )
        torch.testing.assert_close(
            shared_driver.last_update.objective.advantages,
            captured[0].advantages,
            rtol=0,
            atol=0,
        )
        assert shared_actor.adapter_parameter_snapshot(
            "run-a"
        ) == direct_actor.adapter_parameter_snapshot("run-a")
        expected_seeds = [rollout_request_seed(123, ordinal) for ordinal in range(2)]
        assert [request["seed"] for request in shared_backend.requests] == expected_seeds
        assert shared_backend.requests == direct_backend.requests
        assert shared_actor.run_state("run-a").handle.version == 1

    asyncio.run(scenario())


@requires_torch
def test_two_grpo_runs_use_own_adapters_and_bridges_while_slow_reward_waits(tmp_path):
    async def scenario():
        actor, engine, backend = _build_runtime(tmp_path, 2)
        state_a = await _register(actor, "run-a", _prompts())
        state_b = await _register(actor, "run-b", _prompts(10))
        initial_handle_a = state_a.handle
        initial_handle_b = state_b.handle
        score_a_started = threading.Event()
        release_score_a = threading.Event()

        def score_a(_label, _completion, _prompt):
            score_a_started.set()
            assert release_score_a.wait(timeout=2)
            return RewardResult(1.0, 1.0)

        def score_b(_label, _completion, _prompt):
            return RewardResult(2.0, 2.0)

        with (
            RewardBridge(score_a, samples_per_step=2, first_step=1) as bridge_a,
            RewardBridge(score_b, samples_per_step=2, first_step=1) as bridge_b,
            SharedScoringPool(pool_size=2) as scoring_pool,
        ):
            driver_a = _driver("run-a", actor, engine, bridge_a, seed=11)
            driver_b = _driver("run-b", actor, engine, bridge_b, seed=22)
            controller = SharedEngineRunController(scoring_pool, deficit_quantum_ms=1)
            driver_a.add_to_controller(controller, total_steps=1)
            driver_b.add_to_controller(controller, total_steps=1)

            first = await controller.step_the_world()
            assert first.gpu_run_id == "run-a"
            assert score_a_started.wait(timeout=1)
            second = await controller.step_the_world()
            assert second.gpu_run_id == "run-b"
            await asyncio.sleep(0.01)
            third = await controller.step_the_world()
            assert third.gpu_run_id == "run-b"
            assert actor.run_state("run-b").global_step == 1
            assert actor.run_state("run-a").global_step == 0
            assert release_score_a.is_set() is False

            release_score_a.set()
            snapshots = await controller.drain(timeout_s=2)

        assert [snapshot.phase for snapshot in snapshots] == [RunPhase.DONE, RunPhase.DONE]
        assert bridge_a.call_count == 2
        assert bridge_b.call_count == 2
        a_names = {
            request["lora_name"]
            for request in backend.requests
            if request["request_id"].startswith("run-a-")
        }
        b_names = {
            request["lora_name"]
            for request in backend.requests
            if request["request_id"].startswith("run-b-")
        }
        assert a_names == {initial_handle_a.lora_name}
        assert b_names == {initial_handle_b.lora_name}
        assert a_names.isdisjoint(b_names)

    asyncio.run(scenario())


@requires_torch
def test_grpo_update_leaves_peer_adapter_and_optimizer_byte_identical(tmp_path):
    async def scenario():
        actor, engine, _backend = _build_runtime(tmp_path, 2)
        await _register(actor, "run-a", _prompts())
        await _register(actor, "run-b", _prompts(10))

        def score(_label, completion, _prompt):
            reward = float(sum(int(part) for part in completion.split()))
            return RewardResult(reward, reward)

        with RewardBridge(score, samples_per_step=2, first_step=1) as bridge:
            driver = _driver("run-a", actor, engine, bridge)
            rollout = await driver.rollout("run-a", 0)
            reward_batch = driver.score(driver.scoring_payload("run-a", 0, rollout))
            identity = ScoringBatchIdentity("run-a", 0, "run-a-step-0")
            scoring_result = ScoringResult(identity, ScoringKind.REWARD, reward_batch)
            peer_state = actor.run_state("run-b")
            peer_parameters_before = _parameter_bytes(peer_state.adapter_parameters)
            peer_optimizer_before = _optimizer_bytes(peer_state.optimizer)

            await driver.update_and_publish("run-a", 0, rollout, scoring_result)
            next_rollout = await driver.rollout("run-a", 1)

        assert next_rollout.handle.version == 1
        assert _parameter_bytes(peer_state.adapter_parameters) == peer_parameters_before
        assert _optimizer_bytes(peer_state.optimizer) == peer_optimizer_before
        assert peer_state.global_step == 0
        assert peer_state.handle.version == 0
        assert actor.run_state("run-a").global_step == 1
        assert actor.run_state("run-a").handle.version == 1

    asyncio.run(scenario())

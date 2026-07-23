"""cpu parity and isolation tests for shared-engine OPD integration."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from flash.engine.worker import opd_openrlhf, openrlhf_shared_opd
from flash.engine.worker.opd import _gkd_loss_from_logps
from flash.engine.worker.openrlhf_shared_engine import SharedMultiLoRARolloutEngine
from flash.engine.worker.openrlhf_shared_opd import (
    SharedOPDConfig,
    SharedOPDPrompt,
    SharedOPDRunAdapter,
    _collate_training_batch,
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
from flash.engine.worker.teacher import TeacherToken

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)


@dataclass(frozen=True)
class _FakeLoRARequest:
    lora_name: str
    lora_int_id: int
    lora_path: str


@dataclass
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
        self.requests.append(
            {
                "request_id": request_id,
                "lora_name": lora_request.lora_name,
                "seed": sampling_params.seed,
                "prompt_token_ids": tuple(prompt["prompt_token_ids"]),
            }
        )
        generated = SimpleNamespace(
            token_ids=[20, 21, 2],
            finish_reason="stop",
        )
        return SimpleNamespace(outputs=[generated])


class _TinyAdapterModel:
    def __init__(self, torch) -> None:
        self.module = torch.nn.Module()
        self.module.base = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
        torch.nn.init.constant_(self.module.base.weight, 0.05)
        self.module.adapters = torch.nn.ModuleDict()
        self.module.active_adapter = None

        def set_adapter(adapter_name: str) -> None:
            self.module.active_adapter = adapter_name

        def add_adapter(adapter_name: str) -> None:
            layer = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
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


class _Tokenizer:
    eos_token_id = 2

    def decode(self, token_ids, *, skip_special_tokens):
        pieces = {
            10: "P",
            11: ":",
            12: "Q",
            13: ":",
            20: "a",
            21: "b",
            2: "" if skip_special_tokens else "</s>",
        }
        return "".join(pieces[int(token_id)] for token_id in token_ids)


class _Teacher:
    def __init__(
        self,
        *,
        expected_prompt: str,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.expected_prompt = expected_prompt
        self.started = started
        self.release = release
        self.calls = 0

    def score(self, prompt, completion):
        self.calls += 1
        assert prompt == self.expected_prompt
        assert completion == "ab"
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            assert self.release.wait(timeout=2)
        return [
            TeacherToken(text="a", logprob=-0.2, start=0, end=1),
            TeacherToken(text="b", logprob=-0.4, start=1, end=2),
        ]


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


def _prompt(example_index: int = 0, *, token_ids: tuple[int, ...] = (10, 11)):
    return SharedOPDPrompt(
        example_index=example_index,
        rendered="".join(str(token) for token in token_ids),
        token_ids=token_ids,
    )


def _bridge_prompt(token_ids: tuple[int, ...] = (10, 11), *, question: str = "question"):
    return opd_openrlhf._PromptRecord(
        messages=[{"role": "user", "content": question}],
        prompt_ids=token_ids,
        rendered="P:",
    )


async def _register(actor, run_id: str, prompts: list[SharedOPDPrompt]):
    return await actor.register_run(
        run_id,
        optimizer_config=RunOptimizerConfig(learning_rate=0.01),
        dataloader=prompts,
        lora_config=object(),
    )


def _config(seed: int = 123, *, no_signal_attempts: int = 1) -> SharedOPDConfig:
    return SharedOPDConfig(
        seed=seed,
        prompts_per_step=1,
        group_size=1,
        max_response_length=4,
        pad_token_id=0,
        kl_penalty_coef=0.37,
        no_signal_attempts=no_signal_attempts,
        eos_token_ids=(2,),
    )


def _policy_log_probs(gradient_log: list, output_log: list | None = None):
    def calculate(model, _state, batch):
        import torch

        target_tokens = batch.sequences[:, 1:].to(dtype=torch.float64)
        features = torch.stack(
            (target_tokens / 100.0, torch.ones_like(target_tokens)),
            dim=-1,
        )
        values = model(features).squeeze(-1)
        if output_log is not None:
            output_log.append(values.detach().clone())
        values.register_hook(lambda gradient: gradient_log.append(gradient.detach().clone()))
        return values

    return calculate


def _driver(
    run_id,
    actor,
    engine,
    teacher_url,
    *,
    seed=123,
    no_signal_attempts=1,
    gradient_log=None,
    request=opd_openrlhf.post_teacher_request_with_retry,
):
    return SharedOPDRunAdapter(
        run_id,
        config=_config(seed, no_signal_attempts=no_signal_attempts),
        training_actor=actor,
        rollout_engine=engine,
        teacher_url=teacher_url,
        sampling_params_factory=_SamplingParams,
        policy_log_probs=_policy_log_probs(gradient_log if gradient_log is not None else []),
        teacher_request=request,
    )


def _reference_loss(student_logprobs, batch, coefficient):
    row_losses = []
    for row in range(student_logprobs.shape[0]):
        selected = batch.response_mask[row] & batch.group_ids[row].ge(0)
        groups = []
        for group_id in batch.group_ids[row][selected].unique(sorted=True):
            indices = (selected & batch.group_ids[row].eq(group_id)).nonzero().flatten().tolist()
            groups.append((indices, float(batch.teacher_logsums[row, indices[0]].item())))
        if groups:
            row_losses.append(
                _gkd_loss_from_logps(student_logprobs[row], groups, kl_coef=coefficient)
            )
    return student_logprobs.new_tensor(0.0) if not row_losses else sum(row_losses) / len(row_losses)


@requires_torch
def test_single_run_shared_controller_matches_existing_opd_loss_and_gradients_to_1e12(tmp_path):
    import torch

    async def scenario():
        torch.manual_seed(7)
        shared_actor, shared_engine, shared_backend = _build_runtime(tmp_path / "shared", 1)
        await _register(shared_actor, "run-a", [_prompt()])
        torch.manual_seed(7)
        direct_actor, direct_engine, direct_backend = _build_runtime(tmp_path / "direct", 1)
        await _register(direct_actor, "run-a", [_prompt()])
        shared_gradients: list[Any] = []
        direct_gradients: list[Any] = []

        with (
            opd_openrlhf.TeacherAlignmentBridge(
                prompts=[_bridge_prompt()],
                tokenizer=_Tokenizer(),
                teacher=_Teacher(expected_prompt="User: question\nAssistant: "),
                thinking_prefill="",
                eos_token_ids=frozenset({2}),
                stop_sequences=(),
                token="shared-key",
            ) as shared_bridge,
            opd_openrlhf.TeacherAlignmentBridge(
                prompts=[_bridge_prompt()],
                tokenizer=_Tokenizer(),
                teacher=_Teacher(expected_prompt="User: question\nAssistant: "),
                thinking_prefill="",
                eos_token_ids=frozenset({2}),
                stop_sequences=(),
                token="direct-key",
            ) as direct_bridge,
            SharedScoringPool(pool_size=1) as scoring_pool,
        ):
            shared_driver = _driver(
                "run-a",
                shared_actor,
                shared_engine,
                shared_bridge.url,
                gradient_log=shared_gradients,
            )
            controller = SharedEngineRunController(scoring_pool, deficit_quantum_ms=1)
            shared_driver.add_to_controller(controller, total_steps=1)
            snapshots = await controller.drain(timeout_s=2)

            direct_driver = _driver(
                "run-a",
                direct_actor,
                direct_engine,
                direct_bridge.url,
                gradient_log=[],
            )
            rollout = await direct_driver.rollout("run-a", 0)
            teacher = direct_driver.score(direct_driver.scoring_payload("run-a", 0, rollout))
            batch = _collate_training_batch(rollout, teacher, pad_token_id=0)

            def direct_loss(model, state):
                student = _policy_log_probs(direct_gradients)(model, state, batch)
                return _reference_loss(student, batch, 0.37)

            direct_step = await direct_actor.step("run-a", direct_loss)

        assert snapshots[0].phase is RunPhase.DONE
        assert shared_driver.last_update is not None
        assert shared_driver.last_update.objective.item() == pytest.approx(
            direct_step.loss,
            abs=1e-12,
        )
        assert shared_driver.last_update.teacher_coverage == pytest.approx(1.0)
        torch.testing.assert_close(
            shared_gradients[0],
            direct_gradients[0],
            rtol=0,
            atol=1e-12,
        )
        assert shared_actor.adapter_parameter_snapshot(
            "run-a"
        ) == direct_actor.adapter_parameter_snapshot("run-a")
        assert shared_backend.requests == direct_backend.requests
        assert shared_actor.run_state("run-a").handle.version == 1

    asyncio.run(scenario())


@requires_torch
def test_two_opd_runs_use_own_adapters_and_teacher_keys_while_slow_teacher_waits(tmp_path):
    async def scenario():
        actor, engine, backend = _build_runtime(tmp_path, 2)
        state_a = await _register(actor, "run-a", [_prompt()])
        state_b = await _register(actor, "run-b", [_prompt(0, token_ids=(12, 13))])
        initial_handle_a = state_a.handle
        initial_handle_b = state_b.handle
        score_a_started = threading.Event()
        release_score_a = threading.Event()
        teacher_a = _Teacher(
            expected_prompt="User: question\nAssistant: ",
            started=score_a_started,
            release=release_score_a,
        )
        teacher_b = _Teacher(expected_prompt="User: second\nAssistant: ")
        called_urls: list[str] = []
        called_urls_lock = threading.Lock()

        def request(url, payload):
            with called_urls_lock:
                called_urls.append(url)
            return opd_openrlhf.post_teacher_request_with_retry(url, payload, timeout=5)

        with (
            opd_openrlhf.TeacherAlignmentBridge(
                prompts=[_bridge_prompt()],
                tokenizer=_Tokenizer(),
                teacher=teacher_a,
                thinking_prefill="",
                eos_token_ids=frozenset({2}),
                stop_sequences=(),
                token="key-a",
            ) as bridge_a,
            opd_openrlhf.TeacherAlignmentBridge(
                prompts=[_bridge_prompt((12, 13), question="second")],
                tokenizer=_Tokenizer(),
                teacher=teacher_b,
                thinking_prefill="",
                eos_token_ids=frozenset({2}),
                stop_sequences=(),
                token="key-b",
            ) as bridge_b,
            SharedScoringPool(pool_size=2) as scoring_pool,
        ):
            driver_a = _driver(
                "run-a",
                actor,
                engine,
                bridge_a.url,
                seed=11,
                no_signal_attempts=3,
                request=request,
            )
            driver_b = _driver(
                "run-b",
                actor,
                engine,
                bridge_b.url,
                seed=22,
                no_signal_attempts=3,
                request=request,
            )
            controller = SharedEngineRunController(scoring_pool, deficit_quantum_ms=1)
            driver_a.add_to_controller(controller, total_steps=1)
            driver_b.add_to_controller(controller, total_steps=1)

            first = await controller.step_the_world()
            assert first.gpu_run_id == "run-a"
            assert score_a_started.wait(timeout=1)
            run_a_requests_while_waiting = [
                request
                for request in backend.requests
                if request["request_id"].startswith("run-a-")
            ]
            assert len(run_a_requests_while_waiting) == 3

            second = await controller.step_the_world()
            assert second.gpu_run_id == "run-b"
            await asyncio.sleep(0.01)
            third = await controller.step_the_world()
            assert third.gpu_run_id == "run-b"
            assert actor.run_state("run-b").global_step == 1
            assert actor.run_state("run-a").global_step == 0
            assert release_score_a.is_set() is False
            assert (
                len(
                    [
                        request
                        for request in backend.requests
                        if request["request_id"].startswith("run-a-")
                    ]
                )
                == 3
            )

            release_score_a.set()
            snapshots = await controller.drain(timeout_s=2)

        assert [snapshot.phase for snapshot in snapshots] == [RunPhase.DONE, RunPhase.DONE]
        assert teacher_a.calls == 1
        assert teacher_b.calls == 1
        assert set(called_urls) == {bridge_a.url, bridge_b.url}
        assert all("key-a" in url or "key-b" in url for url in called_urls)
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
def test_opd_update_leaves_peer_adapter_and_optimizer_byte_identical(tmp_path):
    async def scenario():
        actor, engine, _backend = _build_runtime(tmp_path, 2)
        await _register(actor, "run-a", [_prompt()])
        await _register(actor, "run-b", [_prompt(0, token_ids=(12, 13))])

        with opd_openrlhf.TeacherAlignmentBridge(
            prompts=[_bridge_prompt()],
            tokenizer=_Tokenizer(),
            teacher=_Teacher(expected_prompt="User: question\nAssistant: "),
            thinking_prefill="",
            eos_token_ids=frozenset({2}),
            stop_sequences=(),
            token="isolation-key",
        ) as bridge:
            driver = _driver("run-a", actor, engine, bridge.url)
            rollout = await driver.rollout("run-a", 0)
            teacher = driver.score(driver.scoring_payload("run-a", 0, rollout))
            identity = ScoringBatchIdentity("run-a", 0, "run-a-step-0")
            scoring_result = ScoringResult(identity, ScoringKind.TEACHER, teacher)
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


@requires_torch
def test_opd_no_signal_uses_bounded_attempt_identity_before_update(tmp_path):
    async def scenario():
        actor, engine, _backend = _build_runtime(tmp_path, 1)
        await _register(actor, "run-a", [_prompt()])
        seen_attempts = []

        def request(_url, payload):
            attempt = payload["label"]["no_signal_attempt"]
            seen_attempts.append(attempt)
            prompt_length = payload["prompt_length"]
            action_length = len(payload["sequence_ids"]) - 1
            has_signal = attempt == 1
            group_ids = [-1] * action_length
            teacher_logsums = [0.0] * action_length
            signal_mask = [False] * action_length
            if has_signal:
                response_offset = prompt_length - 1
                group_ids[response_offset] = 0
                teacher_logsums[response_offset] = -0.2
                signal_mask[response_offset] = True
            return {
                "teacher_group_ids": group_ids,
                "teacher_logsums": teacher_logsums,
                "teacher_signal_mask": signal_mask,
                "signal_count": sum(signal_mask),
                "coverage": float(has_signal),
            }

        driver = _driver(
            "run-a",
            actor,
            engine,
            "http://127.0.0.1:1/teacher/run-key",
            no_signal_attempts=3,
            request=request,
        )
        rollout = await driver.rollout("run-a", 0)
        teacher = driver.score(driver.scoring_payload("run-a", 0, rollout))

        assert seen_attempts == [0, 1]
        assert teacher.samples[0].attempt_index == 1
        assert teacher.samples[0].request_id.endswith("attempt-1")

    asyncio.run(scenario())


@requires_torch
def test_opd_all_no_signal_advances_without_optimizer_or_adapter_mutation(tmp_path):
    async def scenario():
        actor, engine, _backend = _build_runtime(tmp_path, 1)
        state = await _register(actor, "run-a", [_prompt()])
        initial_handle = state.handle
        parameters_before = _parameter_bytes(state.adapter_parameters)
        optimizer_before = _optimizer_bytes(state.optimizer)

        def request(_url, payload):
            action_length = len(payload["sequence_ids"]) - 1
            return {
                "teacher_group_ids": [-1] * action_length,
                "teacher_logsums": [0.0] * action_length,
                "teacher_signal_mask": [False] * action_length,
                "signal_count": 0,
                "coverage": 0.0,
            }

        driver = _driver(
            "run-a",
            actor,
            engine,
            "http://127.0.0.1:1/teacher/run-key",
            no_signal_attempts=3,
            request=request,
        )
        rollout = await driver.rollout("run-a", 0)
        teacher = driver.score(driver.scoring_payload("run-a", 0, rollout))
        scoring_result = ScoringResult(
            ScoringBatchIdentity("run-a", 0, "run-a-step-0"),
            ScoringKind.TEACHER,
            teacher,
        )

        result = await driver.update_and_publish("run-a", 0, rollout, scoring_result)
        next_rollout = await driver.rollout("run-a", 1)

        assert result.aligned_samples == 0
        assert result.objective.item() == 0.0
        assert result.training_step.handle == initial_handle
        assert next_rollout.handle == initial_handle
        assert state.global_step == 1
        assert state.handle.version == 0
        assert _parameter_bytes(state.adapter_parameters) == parameters_before
        assert _optimizer_bytes(state.optimizer) == optimizer_before

    asyncio.run(scenario())


def test_teacher_request_retry_matches_transport_only_single_run_contract(monkeypatch):
    calls = []

    def request(_url, _payload, *, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise opd_openrlhf.OpenRLHFTeacherBridgeError(
                "transport",
                classification="transient",
                retry_transport=True,
            )
        return {"ok": True}

    monkeypatch.setattr(opd_openrlhf, "post_teacher_request", request)
    monkeypatch.setattr(opd_openrlhf.time, "sleep", lambda _delay: None)

    assert opd_openrlhf.post_teacher_request_with_retry(
        "http://127.0.0.1:1/teacher/key",
        {},
        timeout=7,
    ) == {"ok": True}
    assert calls == [7, 7, 7]


def test_teacher_request_retry_does_not_retry_typed_teacher_failure(monkeypatch):
    calls = []

    def request(_url, _payload, *, timeout):
        calls.append(timeout)
        raise opd_openrlhf.OpenRLHFTeacherBridgeError(
            "teacher unavailable",
            classification="transient",
        )

    monkeypatch.setattr(opd_openrlhf, "post_teacher_request", request)

    with pytest.raises(opd_openrlhf.OpenRLHFTeacherBridgeError, match="teacher unavailable"):
        opd_openrlhf.post_teacher_request_with_retry(
            "http://127.0.0.1:1/teacher/key",
            {},
            timeout=7,
        )

    assert calls == [7]


def test_shared_opd_has_no_cross_step_prefetch_surface():
    source = inspect.getsource(openrlhf_shared_opd).lower()

    assert "generate_ahead" not in source
    assert "generate-ahead" not in source

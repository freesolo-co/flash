"""cpu coverage for the OpenRLHF OPD core."""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import io
import json
import os
import sys
import threading
import types
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.engine.worker import opd, opd_openrlhf
from flash.engine.worker.opd import _gkd_loss_from_logps
from flash.engine.worker.teacher import TeacherError, TeacherToken

_OPENRLHF_SOURCE = Path(os.environ.get("FLASH_TEST_OPENRLHF_SOURCE", "/mnt/resource/openrlhf-src"))
requires_openrlhf_source = pytest.mark.skipif(
    not _OPENRLHF_SOURCE.joinpath("openrlhf/trainer/ppo_trainer.py").is_file(),
    reason="pinned OpenRLHF source is unavailable",
)
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is unavailable in offline CI",
)


def _value(args: list[str], flag: str) -> str:
    index = args.index(flag)
    return args[index + 1]


def _install_ray_shaped_opd_extension(
    monkeypatch,
    *,
    warmstart: bool = False,
    torch_module=None,
):
    class _ActorPPOTrainer:
        def ppo_train(self, *_args, **_kwargs):
            return dict(self.original_actor_status)

    class _Actor:
        def __init__(self, *_args, **_kwargs):
            self.model = SimpleNamespace()

    class _SingleTurnAgentExecutor:
        def __init__(self, _remote=None):
            pass

    class _SamplesGenerator:
        pass

    class _ExperienceMaker:
        pass

    class _PPOTrainerRuntime:
        def save_logs_and_checkpoints(self, global_step, logs_dict=None, client_states=None):
            self.original_saves.append((global_step, logs_dict, client_states))

        def fit(self, *args, **kwargs):
            self.original_fit_calls.append((args, kwargs))
            return "fit-result"

    native_fit = _PPOTrainerRuntime.fit

    @functools.wraps(native_fit)
    def grpo_warmstart_fit(self, *args, **kwargs):
        if warmstart:
            self.broadcast_to_vllm()
        return native_fit(self, *args, **kwargs)

    _PPOTrainerRuntime.fit = grpo_warmstart_fit

    class _PolicyModelActorRuntime:
        def save_checkpoint(self, tag, client_states=None, metric_value=None, metric_key=None):
            return (tag, client_states, metric_value, metric_key)

    class _PolicyModelActor:
        __ray_metadata__ = SimpleNamespace(modified_class=_PolicyModelActorRuntime)

    class _PPOTrainer:
        __ray_metadata__ = SimpleNamespace(modified_class=_PPOTrainerRuntime)

    ppo_trainer = types.ModuleType("openrlhf.trainer.ppo_trainer")
    ppo_trainer.PPOTrainer = _PPOTrainer
    ppo_trainer.prepare_datasets = lambda strategy, tokenizer: (strategy, tokenizer)
    modules = {
        "torch": torch_module or types.ModuleType("torch"),
        "openrlhf": types.ModuleType("openrlhf"),
        "openrlhf.utils": types.ModuleType("openrlhf.utils"),
        "openrlhf.utils.agent": types.ModuleType("openrlhf.utils.agent"),
        "openrlhf.trainer": types.ModuleType("openrlhf.trainer"),
        "openrlhf.trainer.ppo_trainer": ppo_trainer,
        "openrlhf.trainer.ppo_utils": types.ModuleType("openrlhf.trainer.ppo_utils"),
        "openrlhf.trainer.ppo_utils.samples_generator": types.ModuleType(
            "openrlhf.trainer.ppo_utils.samples_generator"
        ),
        "openrlhf.trainer.ppo_utils.experience_maker": types.ModuleType(
            "openrlhf.trainer.ppo_utils.experience_maker"
        ),
    }
    for name in ("openrlhf", "openrlhf.utils", "openrlhf.trainer", "openrlhf.trainer.ppo_utils"):
        modules[name].__path__ = []
    modules["openrlhf.utils.agent"].SingleTurnAgentExecutor = _SingleTurnAgentExecutor
    modules["openrlhf.trainer"].ppo_trainer = ppo_trainer
    modules["openrlhf.trainer.ppo_utils.samples_generator"].SamplesGenerator = _SamplesGenerator
    modules["openrlhf.trainer.ppo_utils.experience_maker"].RemoteExperienceMaker = _ExperienceMaker
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("FLASH_OPENRLHF_OPD_BRIDGE_URL", "http://127.0.0.1:1/teacher")
    monkeypatch.setenv("FLASH_OPENRLHF_OPD_KL_COEF", "0.37")
    monkeypatch.setenv("FLASH_OPENRLHF_OPD_SEED", "42")
    if warmstart:
        monkeypatch.setenv("FLASH_OPENRLHF_WARMSTART_ADAPTER", "/work/warmstart")
    else:
        monkeypatch.delenv("FLASH_OPENRLHF_WARMSTART_ADAPTER", raising=False)
    namespace = {
        "__name__": "sitecustomize",
        "__file__": "sitecustomize.py",
        "os": os,
        "functools": functools,
        "_ppo_actor_module": SimpleNamespace(PolicyModelActor=_PolicyModelActor),
        "_ActorPPOTrainer": _ActorPPOTrainer,
        "_Actor": _Actor,
        "_original_execute": lambda *_args, **_kwargs: None,
        "_original_process_response": lambda *_args, **_kwargs: None,
    }
    exec(
        compile(opd_openrlhf._opd_sitecustomize_extension(), "sitecustomize.py", "exec"),
        namespace,
    )
    return namespace, _ActorPPOTrainer, _PPOTrainer, _PPOTrainerRuntime


def _config(**overrides) -> opd_openrlhf.OpenRLHFOPDConfig:
    values = {
        "model_path": "/cache/models--Qwen--Qwen3.5-0.8B/snapshots/" + "a" * 40,
        "dataset_path": "/work/train.jsonl",
        "teacher_url": "http://127.0.0.1:1234/teacher/token",
        "output_dir": "/work/final",
        "checkpoint_dir": "/work/checkpoints",
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_revision": "a" * 40,
        "max_length": 4096,
        "max_completion": 320,
        "prompts_per_step": 16,
        "group_size": 8,
        "scheduled_prompt_count": 48,
        "learning_rate": 1e-5,
        "temperature": 0.7,
        "top_p": 0.95,
        "seed": 42,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_target_modules": ("all-linear",),
        "kl_penalty_coef": 0.3,
        "save_every": 20,
        "save_at_steps": (),
        "final_step": 3,
        "gpu_count": 2,
        "qwen35_language_model_only": True,
    }
    values.update(overrides)
    return opd_openrlhf.OpenRLHFOPDConfig(**values)


class _Tokenizer:
    eos_token_id = 2

    def decode(self, token_ids, *, skip_special_tokens):
        pieces = {10: "P", 11: ":", 20: "a", 21: "b", 2: "" if skip_special_tokens else "</s>"}
        return "".join(pieces[int(token_id)] for token_id in token_ids)


class _Teacher:
    def score(self, prompt, completion):
        assert prompt == "User: question\nAssistant: "
        assert completion == "ab"
        return [
            TeacherToken(text="a", logprob=-0.2, start=0, end=1),
            TeacherToken(text="b", logprob=-0.4, start=1, end=2),
        ]


def test_resolve_inputs_keeps_configured_batch_for_small_prompt_pool(monkeypatch):
    import flash.multimodal as multimodal

    class _Env:
        multi_turn = False
        is_tool_env = False

        def dataset(self):
            return [{"id": index} for index in range(3)]

        def prompt_messages(self, example):
            return [{"role": "user", "content": f"prompt {example['id']}"}]

    class _Tokenizer:
        pad_token = None
        eos_token = "</s>"

        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"]

        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(input_ids=[1, 2])

    knobs = SimpleNamespace(
        max_length=16,
        max_completion=4,
        prompts_per_step=4,
        epochs=1,
        max_steps=None,
        save_at_steps=(),
        group_size=1,
        temperature=0.7,
        top_p=0.95,
        kl_coef=0.37,
        learning_rate=1e-5,
        save_every=-1,
        stop_sequences=(),
    )
    spec = SimpleNamespace(
        algorithm="opd",
        train=SimpleNamespace(structured_outputs="", max_examples=0),
        model="Qwen/Qwen3.5-0.8B",
        model_revision="a" * 40,
        gpu=SimpleNamespace(count=2),
        seed=7,
    )
    monkeypatch.setattr(opd_openrlhf._w, "JOB_SPEC", spec)
    monkeypatch.setattr(opd_openrlhf._w, "THINKING", False)
    monkeypatch.setattr(opd_openrlhf._w, "require_active_env", lambda: _Env())
    monkeypatch.setattr(opd_openrlhf._w, "load_tokenizer", lambda *_args, **_kwargs: _Tokenizer())
    monkeypatch.setattr(opd_openrlhf, "_resolve_opd_knobs", lambda: knobs)
    monkeypatch.setattr(opd_openrlhf, "_thinking_prefill_text", lambda _tokenizer: "")
    monkeypatch.setattr(opd_openrlhf, "backend_seed", lambda seed: seed)
    monkeypatch.setattr(multimodal, "record_has_images", lambda *_args, **_kwargs: False)

    inputs = opd_openrlhf._resolve_single_turn_inputs()

    assert len(inputs["prompts"]) == 3
    assert inputs["prompts_per_step"] == 4
    assert inputs["prompts_per_step"] * inputs["group_size"] % inputs["gpu_count"] == 0
    assert inputs["steps"] == 1


@requires_torch
def test_reverse_kl_value_and_gradients_match_trl_reference():
    torch = pytest.importorskip("torch")
    student = torch.tensor(
        [[-0.2, -0.7, -1.1, -0.3], [-0.4, -0.8, -0.2, -0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )
    group_ids = torch.tensor([[0, 0, 1, -1], [0, 1, 1, 1]], dtype=torch.long)
    teacher_logsums = torch.tensor(
        [[-1.4, -1.4, -0.6, 0.0], [-0.7, -2.1, -2.1, -2.1]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
    coefficient = 0.37

    actual = opd_openrlhf.flash_groupwise_reverse_kl(
        student,
        teacher_logsums,
        group_ids,
        mask,
        coefficient,
    )
    actual_grad = torch.autograd.grad(actual, student, retain_graph=True)[0]

    row0 = _gkd_loss_from_logps(
        student[0],
        [([0, 1], -1.4), ([2], -0.6)],
        kl_coef=coefficient,
    )
    row1 = _gkd_loss_from_logps(
        student[1],
        [([0], -0.7), ([1, 2, 3], -2.1)],
        kl_coef=coefficient,
    )
    expected = (row0 + row1) / 2
    expected_grad = torch.autograd.grad(expected, student)[0]

    assert actual.item() == pytest.approx(expected.item(), abs=1e-12)
    assert torch.allclose(actual_grad, expected_grad, atol=1e-12, rtol=0)


@requires_torch
def test_reverse_kl_excludes_empty_signal_rows_from_sequence_mean():
    torch = pytest.importorskip("torch")
    student = torch.tensor([[-0.2, -0.7], [-0.3, -0.4]], dtype=torch.float64, requires_grad=True)
    group_ids = torch.tensor([[0, 0], [-1, -1]])
    teacher = torch.tensor([[-1.5, -1.5], [0.0, 0.0]], dtype=torch.float64)
    mask = torch.ones_like(group_ids, dtype=torch.bool)

    actual = opd_openrlhf.flash_groupwise_reverse_kl(student, teacher, group_ids, mask, 1.0)
    expected = _gkd_loss_from_logps(student[0], [([0, 1], -1.5)], kl_coef=1.0)

    assert actual.item() == pytest.approx(expected.item(), abs=1e-12)


def test_deterministic_rollout_seed_uses_full_identity_and_retry():
    base = opd_openrlhf.deterministic_rollout_seed(42, 3, 7, 2)

    assert base == opd_openrlhf.deterministic_rollout_seed(42, 3, 7, 2)
    assert base != opd_openrlhf.deterministic_rollout_seed(42, 4, 7, 2)
    assert base != opd_openrlhf.deterministic_rollout_seed(42, 3, 8, 2)
    assert base != opd_openrlhf.deterministic_rollout_seed(42, 3, 7, 3)
    assert base != opd_openrlhf.deterministic_rollout_seed(42, 3, 7, 2, no_signal_attempt_ordinal=1)


def test_teacher_bridge_round_trip_returns_aligned_action_tensors():
    prompt = opd_openrlhf._PromptRecord(
        messages=[{"role": "user", "content": "question"}],
        prompt_ids=(10, 11),
        rendered="P:",
    )
    identity = json.dumps(
        {
            "global_step": 0,
            "example_index": 0,
            "rollout_ordinal": 1,
            "no_signal_attempt": 0,
        }
    )
    with opd_openrlhf.TeacherAlignmentBridge(
        prompts=[prompt],
        tokenizer=_Tokenizer(),
        teacher=_Teacher(),
        thinking_prefill="",
        eos_token_ids=frozenset({2}),
        stop_sequences=(),
        token="test-token",
    ) as bridge:
        result = opd_openrlhf.post_teacher_request(
            bridge.url,
            {
                "label": identity,
                "prompt_length": 2,
                "sequence_ids": [10, 11, 20, 21, 2],
                "terminated": True,
            },
        )

    assert result["rewards"] == 0.0
    assert result["scores"] == 0.0
    assert result["teacher_group_ids"] == [-1, 0, 1, -1]
    assert result["teacher_logsums"] == [0.0, -0.2, -0.4, 0.0]
    assert result["teacher_signal_mask"] == [False, True, True, False]
    assert result["signal_count"] == 2
    assert bridge.snapshot()["teacher_ok"] == 1


def test_teacher_bridge_scores_requests_concurrently():
    class _ConcurrentTeacher:
        def __init__(self):
            self.barrier = threading.Barrier(2)
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def score(self, _prompt, _completion):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                self.barrier.wait(timeout=2)
                return [
                    TeacherToken(text="a", logprob=-0.2, start=0, end=1),
                    TeacherToken(text="b", logprob=-0.4, start=1, end=2),
                ]
            finally:
                with self.lock:
                    self.active -= 1

    teacher = _ConcurrentTeacher()
    prompt = opd_openrlhf._PromptRecord(
        messages=[{"role": "user", "content": "question"}],
        prompt_ids=(10, 11),
        rendered="P:",
    )

    def score(rollout_ordinal):
        return opd_openrlhf.post_teacher_request(
            bridge.url,
            {
                "label": json.dumps(
                    {
                        "global_step": 0,
                        "example_index": 0,
                        "rollout_ordinal": rollout_ordinal,
                        "no_signal_attempt": 0,
                    }
                ),
                "prompt_length": 2,
                "sequence_ids": [10, 11, 20, 21, 2],
                "terminated": True,
            },
            timeout=5,
        )

    with (
        opd_openrlhf.TeacherAlignmentBridge(
            prompts=[prompt],
            tokenizer=_Tokenizer(),
            teacher=teacher,
            thinking_prefill="",
            eos_token_ids=frozenset({2}),
            stop_sequences=(),
        ) as bridge,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        results = list(pool.map(score, (0, 1)))

    assert [result["signal_count"] for result in results] == [2, 2]
    assert teacher.max_active == 2


@pytest.mark.parametrize(
    ("permanent", "classification"),
    [(True, "permanent"), (False, "transient")],
)
def test_teacher_bridge_preserves_typed_teacher_error_classification(permanent, classification):
    class _FailingTeacher:
        def score(self, _prompt, _completion):
            raise TeacherError("teacher unavailable", permanent=permanent)

    prompt = opd_openrlhf._PromptRecord(
        messages=[{"role": "user", "content": "question"}],
        prompt_ids=(10, 11),
        rendered="P:",
    )
    identity = json.dumps(
        {
            "global_step": 0,
            "example_index": 0,
            "rollout_ordinal": 0,
            "no_signal_attempt": 0,
        }
    )
    with (
        opd_openrlhf.TeacherAlignmentBridge(
            prompts=[prompt],
            tokenizer=_Tokenizer(),
            teacher=_FailingTeacher(),
            thinking_prefill="",
            eos_token_ids=frozenset({2}),
            stop_sequences=(),
        ) as bridge,
        pytest.raises(opd_openrlhf.OpenRLHFTeacherBridgeError) as error,
    ):
        opd_openrlhf.post_teacher_request(
            bridge.url,
            {
                "label": identity,
                "prompt_length": 2,
                "sequence_ids": [10, 11, 20, 21, 2],
                "terminated": True,
            },
        )

    assert error.value.classification == classification
    assert bridge.teacher_failure == (classification, "teacher unavailable")


def test_teacher_bridge_fails_closed_on_prompt_identity_mismatch():
    prompt = opd_openrlhf._PromptRecord(
        messages=[{"role": "user", "content": "question"}],
        prompt_ids=(10, 11),
        rendered="P:",
    )
    identity = json.dumps(
        {
            "global_step": 0,
            "example_index": 0,
            "rollout_ordinal": 0,
            "no_signal_attempt": 0,
        }
    )
    with (
        opd_openrlhf.TeacherAlignmentBridge(
            prompts=[prompt],
            tokenizer=_Tokenizer(),
            teacher=_Teacher(),
            thinking_prefill="",
            eos_token_ids=frozenset({2}),
            stop_sequences=(),
        ) as bridge,
        pytest.raises(opd_openrlhf.OpenRLHFTeacherBridgeError) as error,
    ):
        opd_openrlhf.post_teacher_request(
            bridge.url,
            {
                "label": identity,
                "prompt_length": 2,
                "sequence_ids": [10, 99, 20, 21, 2],
                "terminated": True,
            },
        )

    assert error.value.classification == "permanent"


def test_teacher_bridge_mutation_callback_is_idempotent():
    calls = []
    prompt = opd_openrlhf._PromptRecord(messages=[], prompt_ids=(10,), rendered="P")
    with opd_openrlhf.TeacherAlignmentBridge(
        prompts=[prompt],
        tokenizer=_Tokenizer(),
        teacher=_Teacher(),
        thinking_prefill="",
        eos_token_ids=frozenset({2}),
        stop_sequences=(),
        mutation_callback=lambda: calls.append(True),
    ) as bridge:
        assert opd_openrlhf.post_teacher_request(bridge.url, {"mutation": True}) == {"ok": True}
        assert opd_openrlhf.post_teacher_request(bridge.url, {"mutation": True}) == {"ok": True}

    assert calls == [True]


def test_checkpoint_state_is_valid_full_resume_sidecar():
    state = opd_openrlhf._checkpoint_state(
        step=2,
        seed=42,
        prompt_pool_fingerprint="a" * 64,
        prompts_per_step=3,
        group_size=4,
        accounting={
            "generated_tokens": 20,
            "teacher_input_tokens": 40,
            "truncated_rollouts": 1,
            "aligned_sequences": 18,
            "empty_alignments": 2,
            "coverage_sum": 15.0,
            "teacher_ok": 18,
            "teacher_transient": 1,
            "teacher_error": 0,
            "samples_seen": 20,
            "no_signal_resamples": 2,
        },
        loss_curve=[0.4, 0.3],
        coverage_curve=[0.7, 0.8],
        train_wall_seconds=12.5,
    )

    assert state["opt_steps"] == 2
    assert state["rollout_seed_ordinal"] == 24
    assert state["loss_curve"] == [0.4, 0.3]
    assert state["coverage_curve"] == [0.7, 0.8]
    assert state["skip_counts"] == {"empty_alignment": 2}


def test_teacher_bridge_runs_checkpoint_callback_before_reply():
    calls = []
    prompt = opd_openrlhf._PromptRecord(messages=[], prompt_ids=(10,), rendered="P")
    with opd_openrlhf.TeacherAlignmentBridge(
        prompts=[prompt],
        tokenizer=_Tokenizer(),
        teacher=_Teacher(),
        thinking_prefill="",
        eos_token_ids=frozenset({2}),
        stop_sequences=(),
        checkpoint_callback=lambda step: calls.append(step),
    ) as bridge:
        assert opd_openrlhf.post_teacher_request(bridge.url, {"checkpoint": 3}) == {"ok": True}

    assert calls == [3]


def test_teacher_bridge_delivers_metrics_before_checkpoint():
    calls = []
    prompt = opd_openrlhf._PromptRecord(messages=[], prompt_ids=(10,), rendered="P")
    with opd_openrlhf.TeacherAlignmentBridge(
        prompts=[prompt],
        tokenizer=_Tokenizer(),
        teacher=_Teacher(),
        thinking_prefill="",
        eos_token_ids=frozenset({2}),
        stop_sequences=(),
        metrics_callback=lambda step, loss, coverage: calls.append(
            ("metrics", step, loss, coverage)
        ),
        checkpoint_callback=lambda step: calls.append(("checkpoint", step)),
    ) as bridge:
        assert opd_openrlhf.post_teacher_request(
            bridge.url,
            {"metrics": {"step": 3, "loss": 0.2, "coverage": 0.75}},
        ) == {"ok": True}
        assert opd_openrlhf.post_teacher_request(bridge.url, {"checkpoint": 3}) == {"ok": True}

    assert calls == [("metrics", 3, 0.2, 0.75), ("checkpoint", 3)]


def test_teacher_bridge_classifies_retriable_callback_failure_as_transient():
    prompt = opd_openrlhf._PromptRecord(messages=[], prompt_ids=(10,), rendered="P")

    def fail(_step):
        raise opd_openrlhf._w.RetriableInfraError("temporary upload failure")

    with (
        opd_openrlhf.TeacherAlignmentBridge(
            prompts=[prompt],
            tokenizer=_Tokenizer(),
            teacher=_Teacher(),
            thinking_prefill="",
            eos_token_ids=frozenset({2}),
            stop_sequences=(),
            checkpoint_callback=fail,
        ) as bridge,
        pytest.raises(opd_openrlhf.OpenRLHFTeacherBridgeError) as error,
    ):
        opd_openrlhf.post_teacher_request(bridge.url, {"checkpoint": 3})

    assert error.value.classification == "transient"


def test_child_ray_actor_runtime_writes_flash_rng_after_native_checkpoint(monkeypatch, tmp_path):
    namespace, *_ = _install_ray_shaped_opd_extension(monkeypatch)
    events = []
    checkpoint_root = tmp_path / "checkpoints"
    runtime = namespace["_FlashPolicyModelActorRuntime"]
    assert namespace["_FlashPolicyModelActor"].__ray_metadata__.modified_class is runtime
    assert runtime.save_checkpoint is namespace["_flash_policy_save_checkpoint"]
    actor = runtime()
    actor.strategy = SimpleNamespace(
        args=SimpleNamespace(ckpt=SimpleNamespace(path=str(checkpoint_root))),
        is_rank_0=lambda: True,
    )
    namespace["_original_policy_save_checkpoint"] = lambda *_args, **_kwargs: events.append(
        "native"
    )
    namespace["_flash_capture_training_rng_state"] = lambda: {"source": "child-trainer"}
    namespace["torch"].save = lambda state, path: events.append((state, path))

    actor.save_checkpoint("global_step2", client_states={"global_step": 2})

    assert events == [
        "native",
        (
            {"source": "child-trainer"},
            str(checkpoint_root / "_actor" / "global_step2" / "rng_state.pth"),
        ),
    ]


def test_required_checkpoint_uploads_resume_before_deployable(monkeypatch):
    events = []
    publisher = object.__new__(opd_openrlhf._OpenRLHFOPDCheckpointPublisher)
    publisher.required_steps = frozenset({2})
    publisher.save_every = 20
    publisher.final_step = 3
    monkeypatch.setattr(publisher, "_wait_for_checkpoint", lambda _step: ("actor", "hf"))
    monkeypatch.setattr(publisher, "_stage", lambda *_args: "/staged/checkpoint-2")

    def upload(step, stage, *, after_upload=None):
        assert (step, stage) == (2, "/staged/checkpoint-2")
        events.append("resume")
        after_upload()
        return True

    monkeypatch.setattr(opd_openrlhf._w, "upload_resume_checkpoint", upload)
    monkeypatch.setattr(
        opd_openrlhf._w,
        "publish_deployable_checkpoint",
        lambda path, step, **_kwargs: events.append(("deployable", path, step)),
    )

    publisher._publish(2)

    assert events == [
        "resume",
        ("deployable", "/staged/checkpoint-2/_adapter_export", 2),
    ]


def test_periodic_checkpoint_publishes_best_effort_deployable(monkeypatch):
    events = []
    publisher = object.__new__(opd_openrlhf._OpenRLHFOPDCheckpointPublisher)
    publisher.required_steps = frozenset()
    publisher.save_every = 2
    publisher.final_step = 5
    monkeypatch.setattr(publisher, "_wait_for_checkpoint", lambda _step: ("actor", "hf"))
    monkeypatch.setattr(publisher, "_stage", lambda *_args: "/staged/checkpoint-2")

    def upload(_step, _stage, *, after_upload=None):
        events.append("resume")
        after_upload()
        return True

    monkeypatch.setattr(opd_openrlhf._w, "upload_resume_checkpoint", upload)
    monkeypatch.setattr(
        opd_openrlhf._w,
        "publish_deployable_checkpoint",
        lambda path, _step, **kwargs: events.append(("deployable", path, kwargs["required"])),
    )

    publisher._publish(2)

    assert events == [
        "resume",
        ("deployable", "/staged/checkpoint-2/_adapter_export", False),
    ]


def test_checkpoint_stage_copies_child_trainer_rng_blob(monkeypatch, tmp_path):
    actor_tag = tmp_path / "actor"
    hf_export = tmp_path / "hf"
    actor_tag.mkdir()
    hf_export.mkdir()
    actor_tag.joinpath("zero_optim_states.pt").write_bytes(b"optimizer")
    actor_tag.joinpath("rng_state.pth").write_bytes(b"child-trainer-rng")
    hf_export.joinpath("adapter_config.json").write_text("{}", encoding="utf-8")
    hf_export.joinpath("adapter_model.bin").write_bytes(b"adapter")

    def export(_source, destination, *_args):
        os.makedirs(destination)
        Path(destination, "adapter_config.json").write_text("{}", encoding="utf-8")
        Path(destination, "adapter_model.bin").write_bytes(b"adapter")

    monkeypatch.setattr(opd_openrlhf, "export_openrlhf_adapter", export)
    publisher = object.__new__(opd_openrlhf._OpenRLHFOPDCheckpointPublisher)
    publisher.staging_root = str(tmp_path / "staging")
    publisher.model_id = "Qwen/Qwen3.5-0.8B"
    publisher.model_revision = "a" * 40
    publisher.python_bin = "/usr/bin/python"
    publisher.state_for_step = lambda step: {"step": step}

    stage = Path(publisher._stage(2, str(actor_tag), str(hf_export)))

    assert stage.joinpath("optimizer.pt").read_bytes() == b"optimizer"
    assert stage.joinpath("rng_state.pth").read_bytes() == b"child-trainer-rng"
    deployable = stage / "_adapter_export"
    assert deployable.joinpath("adapter_config.json").is_file()
    assert deployable.joinpath("adapter_model.bin").is_file()
    assert not deployable.joinpath("_actor").exists()


def test_final_resume_checkpoint_is_required_without_extra_deployable(monkeypatch):
    publisher = object.__new__(opd_openrlhf._OpenRLHFOPDCheckpointPublisher)
    publisher.required_steps = frozenset()
    publisher.save_every = 20
    publisher.final_step = 3
    monkeypatch.setattr(publisher, "_wait_for_checkpoint", lambda _step: ("actor", "hf"))
    monkeypatch.setattr(publisher, "_stage", lambda *_args: "/staged/checkpoint-3")
    monkeypatch.setattr(
        opd_openrlhf._w, "upload_resume_checkpoint", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        opd_openrlhf._w,
        "publish_deployable_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("final deployable is published during finalization"),
    )

    with pytest.raises(
        opd_openrlhf._w.RetriableInfraError,
        match="required OpenRLHF OPD checkpoint step 3",
    ):
        publisher._publish(3)


def test_resume_reconciles_missing_required_deployable(monkeypatch):
    calls = []
    monkeypatch.setattr(opd_openrlhf, "_deployable_adapter_on_hf", lambda _step: False)
    monkeypatch.setattr(
        opd_openrlhf._w,
        "publish_deployable_checkpoint",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    opd_openrlhf._reconcile_required_deployable(
        "/resume/checkpoint-2",
        {"opt_steps": 2},
        (2,),
    )

    assert calls == [
        (
            ("/resume/checkpoint-2/_adapter_export", 2),
            {"required": True, "_provenance_ready": True},
        )
    ]


def test_resume_replayed_step_checkpoint_reuses_restored_metrics():
    resume_state = {
        "opt_steps": 2,
        "loss_curve": [0.4, 0.3],
        "coverage_curve": [0.7, 0.8],
    }

    step_states = opd_openrlhf._initial_checkpoint_step_states(resume_state)

    assert step_states == {2: resume_state}
    assert step_states[2] is not resume_state


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [
        (opd_openrlhf._OPENRLHF_TRANSIENT_TEACHER_EXIT, "transient"),
        (opd_openrlhf._OPENRLHF_PERMANENT_TEACHER_EXIT, "permanent"),
    ],
)
def test_training_failure_maps_child_teacher_exit_codes(returncode, expected):
    error_type = opd_openrlhf._w.RetriableInfraError if expected == "transient" else RuntimeError

    with pytest.raises(error_type, match=expected):
        opd_openrlhf._raise_training_failure(returncode, None)


def test_ray_modified_class_checkpoint_and_warmstart_fit_hooks_execute(monkeypatch):
    namespace, actor_trainer, ray_wrapper, runtime = _install_ray_shaped_opd_extension(
        monkeypatch, warmstart=True
    )
    callbacks = []
    namespace["_flash_post_teacher"] = lambda payload: callbacks.append(payload)
    trainer = runtime()
    trainer.args = SimpleNamespace(ckpt=SimpleNamespace(save_steps=1, load_enable=False))
    trainer.original_saves = []
    trainer.original_fit_calls = []
    trainer.broadcast_to_vllm = lambda: callbacks.append("broadcast")

    logs = {"policy_loss": 0.2, "teacher_coverage": 0.75}
    trainer.save_logs_and_checkpoints(1, logs, {"global_step": 1})
    result = trainer.fit()

    assert actor_trainer.training_step is namespace["_flash_opd_training_step"]
    assert ray_wrapper.__ray_metadata__.modified_class is runtime
    assert runtime.save_logs_and_checkpoints is namespace["_flash_save_logs_and_checkpoints"]
    assert runtime.fit is namespace["_flash_ppo_fit"]
    assert trainer.original_saves == [(1, logs, {"global_step": 1})]
    assert callbacks == [
        {"metrics": {"step": 1, "loss": 0.2, "coverage": 0.75}},
        {"checkpoint": 1},
        "broadcast",
    ]
    assert trainer.original_fit_calls == [((), {})]
    assert result == "fit-result"


def test_checkpoint_hook_does_not_post_after_save_failure(monkeypatch):
    namespace, _, _, runtime = _install_ray_shaped_opd_extension(monkeypatch)
    callbacks = []
    namespace["_flash_post_teacher"] = lambda payload: callbacks.append(payload)

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("checkpoint save failed")

    namespace["_original_save_logs_and_checkpoints"] = fail_save
    trainer = runtime()
    trainer.args = SimpleNamespace(ckpt=SimpleNamespace(save_steps=1, load_enable=False))
    logs = {"policy_loss": 0.2, "teacher_coverage": 0.75}

    with pytest.raises(RuntimeError, match="checkpoint save failed"):
        trainer.save_logs_and_checkpoints(1, logs, {"global_step": 1})

    assert callbacks == [{"metrics": {"step": 1, "loss": 0.2, "coverage": 0.75}}]
    assert trainer.args.ckpt.save_steps == 1


def test_checkpoint_hook_uses_exact_steps_plus_final_boundary(monkeypatch):
    monkeypatch.setenv("FLASH_OPENRLHF_OPD_EXACT_SAVE_STEPS", "[2]")
    monkeypatch.setenv("FLASH_OPENRLHF_OPD_FINAL_STEP", "3")
    namespace, _, _, runtime = _install_ray_shaped_opd_extension(monkeypatch)
    callbacks = []
    namespace["_flash_post_teacher"] = lambda payload: callbacks.append(payload)
    trainer = runtime()
    trainer.args = SimpleNamespace(ckpt=SimpleNamespace(save_steps=float("inf"), load_enable=False))
    trainer.original_saves = []
    trainer.original_fit_calls = []
    logs = {"policy_loss": 0.2, "teacher_coverage": 0.75}

    trainer.save_logs_and_checkpoints(1, logs, {"global_step": 1})
    trainer.save_logs_and_checkpoints(2, logs, {"global_step": 2})
    trainer.save_logs_and_checkpoints(3, logs, {"global_step": 3})

    assert callbacks == [
        {"metrics": {"step": 1, "loss": 0.2, "coverage": 0.75}},
        {"metrics": {"step": 2, "loss": 0.2, "coverage": 0.75}},
        {"checkpoint": 2},
        {"metrics": {"step": 3, "loss": 0.2, "coverage": 0.75}},
        {"checkpoint": 3},
    ]
    assert trainer.args.ckpt.save_steps == float("inf")


def test_checkpoint_hook_keeps_negative_periodic_interval_disabled(monkeypatch):
    namespace, _, _, runtime = _install_ray_shaped_opd_extension(monkeypatch)
    callbacks = []
    observed_save_steps = []
    namespace["_flash_post_teacher"] = lambda payload: callbacks.append(payload)

    def save(trainer, *_args, **_kwargs):
        observed_save_steps.append(trainer.args.ckpt.save_steps)

    namespace["_original_save_logs_and_checkpoints"] = save
    trainer = runtime()
    trainer.args = SimpleNamespace(ckpt=SimpleNamespace(save_steps=-1, load_enable=False))
    logs = {"policy_loss": 0.2, "teacher_coverage": 0.75}

    trainer.save_logs_and_checkpoints(1, logs, {"global_step": 1})

    assert observed_save_steps == [float("inf")]
    assert callbacks == [{"metrics": {"step": 1, "loss": 0.2, "coverage": 0.75}}]
    assert trainer.args.ckpt.save_steps == -1


def test_ray_modified_class_fit_does_not_prebroadcast_resumed_warmstart(monkeypatch):
    _, _, _, runtime = _install_ray_shaped_opd_extension(monkeypatch, warmstart=True)
    trainer = runtime()
    trainer.args = SimpleNamespace(ckpt=SimpleNamespace(save_steps=1, load_enable=True))
    trainer.original_fit_calls = []
    broadcasts = []
    trainer.broadcast_to_vllm = lambda: broadcasts.append(True)

    trainer.fit()

    assert broadcasts == []
    assert trainer.original_fit_calls == [((), {})]


def test_opd_prompt_dataloader_forces_authoritative_file_order(monkeypatch):
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)
    calls = []

    class _Strategy:
        def setup_dataloader(self, *_args, **kwargs):
            calls.append(kwargs["shuffle"])
            return "loader"

    namespace["_original_prepare_datasets"] = lambda strategy, _tokenizer: (
        strategy.setup_dataloader(object(), 1, shuffle=True)
    )

    assert namespace["_flash_prepare_datasets"](_Strategy(), object()) == "loader"
    assert calls == [False]


def test_child_seed_matches_optimizer_example_rollout_retry_identity(monkeypatch):
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)
    identity = {
        "global_step": 3,
        "example_index": 7,
        "rollout_ordinal": 2,
        "no_signal_attempt": 1,
    }

    actual = namespace["_flash_seed"](namespace["_flash_identity"](identity))

    assert actual == opd_openrlhf.deterministic_rollout_seed(
        42,
        3,
        7,
        2,
        no_signal_attempt_ordinal=1,
    )


def test_child_preserves_scheduled_ordinal_base_for_group_samples(monkeypatch):
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)
    identities = []

    async def execute(*_args, **_kwargs):
        return {
            "action_ranges": [(2, 4)],
            "observation_tokens": [10, 11, 20, 21],
            "truncated": False,
        }

    namespace["_original_execute"] = execute
    namespace["_flash_post_teacher"] = lambda payload: (
        identities.append(json.loads(payload["label"]))
        or {
            "signal_count": 1,
            "teacher_group_ids": [-1, 0, 1],
            "teacher_logsums": [0.0, -0.2, -0.4],
            "teacher_signal_mask": [False, True, True],
            "coverage": 1.0,
        }
    )
    label = json.dumps(
        {
            "global_step": 0,
            "example_index": 0,
            "rollout_ordinal": 8,
            "no_signal_attempt": 0,
        }
    )
    executor = SimpleNamespace()

    for _ in range(2):
        asyncio.run(
            namespace["_flash_opd_execute"](
                executor,
                "prompt",
                label,
                SimpleNamespace(),
                10,
                object(),
                object(),
            )
        )

    assert [identity["rollout_ordinal"] for identity in identities] == [8, 9]


def test_child_passes_native_non_length_termination_to_teacher_bridge(monkeypatch):
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)
    payloads = []

    async def execute(*_args, **_kwargs):
        return {
            "action_ranges": [(2, 4)],
            "observation_tokens": [10, 11, 20, 21],
            "truncated": False,
        }

    namespace["_original_execute"] = execute
    namespace["_flash_post_teacher"] = lambda payload: (
        payloads.append(payload)
        or {
            "signal_count": 1,
            "teacher_group_ids": [-1, 0, 1],
            "teacher_logsums": [0.0, -0.2, -0.4],
            "teacher_signal_mask": [False, True, True],
            "coverage": 1.0,
        }
    )
    label = json.dumps(
        {
            "global_step": 0,
            "example_index": 0,
            "rollout_ordinal": 0,
            "no_signal_attempt": 0,
        }
    )

    result = asyncio.run(
        namespace["_flash_opd_execute"](
            SimpleNamespace(),
            "prompt",
            label,
            SimpleNamespace(),
            10,
            object(),
            object(),
        )
    )

    assert result["teacher_coverage"] == 1.0
    assert payloads[0]["terminated"] is True


def test_child_drops_no_signal_rollout_after_bounded_resampling(monkeypatch):
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)
    attempts = []

    async def execute(*_args, **_kwargs):
        return {
            "action_ranges": [(2, 4)],
            "observation_tokens": [10, 11, 20, 21],
            "truncated": False,
        }

    namespace["_original_execute"] = execute
    namespace["_flash_post_teacher"] = lambda payload: (
        attempts.append(payload)
        or {
            "signal_count": 0,
            "teacher_group_ids": [-1, -1, -1],
            "teacher_logsums": [0.0, 0.0, 0.0],
            "teacher_signal_mask": [False, False, False],
            "coverage": 0.0,
        }
    )
    label = json.dumps(
        {
            "global_step": 0,
            "example_index": 0,
            "rollout_ordinal": 0,
            "no_signal_attempt": 0,
        }
    )

    result = asyncio.run(
        namespace["_flash_opd_execute"](
            SimpleNamespace(),
            "prompt",
            label,
            SimpleNamespace(),
            10,
            object(),
            object(),
        )
    )

    assert result["teacher_signal_mask"] == [False, False, False]
    assert result["teacher_coverage"] == 0.0
    assert len(attempts) == opd_openrlhf._OPENRLHF_NO_SIGNAL_ATTEMPTS
    assert [json.loads(attempt["label"])["no_signal_attempt"] for attempt in attempts] == [0, 1, 2]


@requires_torch
def test_child_process_response_drops_no_signal_action_tokens(monkeypatch):
    torch = pytest.importorskip("torch")
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)
    namespace["torch"] = torch
    experience = SimpleNamespace(
        action_mask=torch.ones((1, 3), dtype=torch.bool),
        info={},
    )
    namespace["_original_process_response"] = lambda *_args, **_kwargs: experience

    result = namespace["_flash_opd_process_response"](
        object(),
        {
            "teacher_group_ids": [-1, -1, -1],
            "teacher_logsums": [0.0, 0.0, 0.0],
            "teacher_signal_mask": [False, False, False],
            "teacher_coverage": 0.0,
        },
    )

    assert not bool(result.action_mask.any().item())


@requires_torch
def test_child_pads_variable_length_per_sample_teacher_tensors(monkeypatch):
    torch = pytest.importorskip("torch")
    namespace, _, _, _ = _install_ray_shaped_opd_extension(
        monkeypatch,
        torch_module=torch,
    )
    action_mask = torch.ones((2, 4), dtype=torch.bool)

    padded = namespace["_flash_pad_info"](
        [
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([[3, 4, 5]], dtype=torch.long),
        ],
        action_mask,
        -1,
    )

    assert torch.equal(
        padded,
        torch.tensor(
            [
                [1, 2, -1, -1],
                [3, 4, 5, -1],
            ],
            dtype=torch.long,
        ),
    )


def test_child_mutation_marker_requires_signal_in_accumulation_window(monkeypatch):
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)
    trainer = SimpleNamespace()
    should_mark = namespace["_flash_should_mark_mutation"]

    assert should_mark(trainer, 0, 2, True) is False
    assert should_mark(trainer, 1, 2, False) is True
    assert should_mark(trainer, 0, 2, False) is False
    assert should_mark(trainer, 1, 2, False) is False
    with pytest.raises(RuntimeError, match="gradient accumulation must be positive"):
        should_mark(trainer, 0, 0, True)


def test_child_counts_only_samples_with_aligned_tokens(monkeypatch):
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)
    selected = SimpleNamespace(
        any=lambda dim: SimpleNamespace(
            sum=lambda: SimpleNamespace(item=lambda: 0 if dim == -1 else 99)
        )
    )

    assert namespace["_flash_aligned_sample_count"](selected) == 0.0


def test_child_gates_sample_metrics_on_current_global_microbatch(monkeypatch):
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)

    class _Strategy:
        def __init__(self, count):
            self.count = count

        def all_reduce(self, values, *, op):
            assert values == {"flash_aligned_samples": 0.0}
            assert op == "sum"
            return {"flash_aligned_samples": self.count}

    has_signal = namespace["_flash_current_batch_has_signal"]

    assert has_signal(_Strategy(0), 0.0) is False
    assert has_signal(_Strategy(2), 0.0) is True


def test_child_scales_local_loss_to_global_sequence_mean(monkeypatch):
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)
    scale = namespace["_flash_global_sequence_mean_loss"]

    assert scale(2.0, 2.0, 2.0, 1) == 2.0
    assert scale(2.0, 1.0, 1.5, 3) == 4.0
    assert scale(2.0, 0.0, 1.5, 3) == 0.0


def test_child_empty_window_advances_without_optimizer_update(monkeypatch):
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)
    events = []

    class _Optimizer:
        def zero_grad(self):
            events.append("zero_grad")

    class _Engine:
        def __init__(self):
            self._is_gradient_accumulation_boundary = None
            self.optimizer = _Optimizer()

        def set_gradient_accumulation_boundary(self, value):
            self._is_gradient_accumulation_boundary = value
            events.append(("boundary", value))

    engine = _Engine()

    class _Strategy:
        def optimizer_step(self, *_args, **_kwargs):
            events.append(("step", engine._is_gradient_accumulation_boundary))

    trainer = SimpleNamespace(
        actor=SimpleNamespace(model=engine),
        actor_optim=object(),
        actor_scheduler=object(),
        strategy=_Strategy(),
    )

    namespace["_flash_advance_empty_accumulation"](trainer)

    assert events == [
        ("boundary", False),
        ("step", False),
        ("boundary", None),
        "zero_grad",
    ]


def test_child_actor_status_defaults_empty_signal_metrics(monkeypatch):
    _, actor_trainer, _, _ = _install_ray_shaped_opd_extension(monkeypatch)
    trainer = actor_trainer()
    trainer.original_actor_status = {"actor_lr": 1e-5}

    status = trainer.ppo_train(0.0)

    assert status == {
        "actor_lr": 1e-5,
        "policy_loss": 0.0,
        "distillation_loss": 0.0,
        "teacher_coverage": 0.0,
    }


def test_child_bridge_retries_transport_then_exits_transient(monkeypatch):
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)
    attempts = []
    sleeps = []

    class _Exit(RuntimeError):
        def __init__(self, code):
            super().__init__(str(code))
            self.code = code

    def fail_urlopen(*_args, **_kwargs):
        attempts.append(True)
        raise urllib.error.URLError("connection refused")

    namespace["urllib"] = SimpleNamespace(
        error=urllib.error,
        request=SimpleNamespace(Request=lambda *_args, **_kwargs: object(), urlopen=fail_urlopen),
    )
    namespace["time"] = SimpleNamespace(sleep=lambda delay: sleeps.append(delay))
    namespace["os"] = SimpleNamespace(_exit=lambda code: (_ for _ in ()).throw(_Exit(code)))

    with pytest.raises(_Exit) as error:
        namespace["_flash_post_teacher"]({"mutation": True})

    assert error.value.code == opd_openrlhf._OPENRLHF_TRANSIENT_TEACHER_EXIT
    assert len(attempts) == opd_openrlhf._OPENRLHF_BRIDGE_TRANSPORT_ATTEMPTS
    assert sleeps == [0.25, 0.5]


def test_child_bridge_exits_permanent_for_provider_classification(monkeypatch):
    namespace, _, _, _ = _install_ray_shaped_opd_extension(monkeypatch)

    class _Exit(RuntimeError):
        def __init__(self, code):
            super().__init__(str(code))
            self.code = code

    body = json.dumps(
        {"error": {"classification": "permanent", "message": "invalid request"}}
    ).encode()
    http_error = urllib.error.HTTPError(
        namespace["_FLASH_OPD_BRIDGE_URL"],
        422,
        "unprocessable",
        {},
        io.BytesIO(body),
    )
    namespace["urllib"] = SimpleNamespace(
        error=urllib.error,
        request=SimpleNamespace(
            Request=lambda *_args, **_kwargs: object(),
            urlopen=lambda *_args, **_kwargs: (_ for _ in ()).throw(http_error),
        ),
    )
    namespace["os"] = SimpleNamespace(_exit=lambda code: (_ for _ in ()).throw(_Exit(code)))

    with pytest.raises(_Exit) as error:
        namespace["_flash_post_teacher"]({"mutation": True})

    assert error.value.code == opd_openrlhf._OPENRLHF_PERMANENT_TEACHER_EXIT


def test_warmstart_config_preserves_source_lora_shape(tmp_path):
    tmp_path.joinpath("adapter_config.json").write_text(
        json.dumps(
            {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )

    assert opd_openrlhf._warmstart_config(str(tmp_path), "Qwen/Qwen3.5-0.8B") == (
        16,
        32,
        ("q_proj", "v_proj"),
    )


def test_scheduled_dataset_carries_stable_step_and_example_identity(tmp_path):
    prompts = [
        opd_openrlhf._PromptRecord(messages=[], prompt_ids=(1,), rendered="one"),
        opd_openrlhf._PromptRecord(messages=[], prompt_ids=(2,), rendered="two"),
    ]
    path = tmp_path / "train.jsonl"

    count = opd_openrlhf._write_scheduled_dataset(
        str(path),
        prompts,
        steps=2,
        prompts_per_step=3,
        group_size=2,
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    identities = [json.loads(row["label"]) for row in rows]

    assert count == 6
    assert [row["input"] for row in rows] == ["one", "two", "one", "two", "one", "two"]
    assert identities == [
        {"example_index": 0, "global_step": 0, "no_signal_attempt": 0, "rollout_ordinal": 0},
        {"example_index": 1, "global_step": 0, "no_signal_attempt": 0, "rollout_ordinal": 0},
        {"example_index": 0, "global_step": 0, "no_signal_attempt": 0, "rollout_ordinal": 2},
        {"example_index": 1, "global_step": 1, "no_signal_attempt": 0, "rollout_ordinal": 0},
        {"example_index": 0, "global_step": 1, "no_signal_attempt": 0, "rollout_ordinal": 0},
        {"example_index": 1, "global_step": 1, "no_signal_attempt": 0, "rollout_ordinal": 2},
    ]


def test_build_openrlhf_opd_args_maps_distillation_job():
    args = opd_openrlhf.build_openrlhf_opd_args(_config())

    assert _value(args, "--actor.model_name_or_path").endswith("/" + "a" * 40)
    assert _value(args, "--reward.remote_url").startswith("http://127.0.0.1:")
    assert _value(args, "--data.input_key") == "input"
    assert _value(args, "--data.label_key") == "label"
    assert _value(args, "--rollout.max_new_tokens") == "320"
    assert _value(args, "--rollout.batch_size") == "16"
    assert _value(args, "--rollout.n_samples_per_prompt") == "8"
    assert _value(args, "--train.batch_size") == "128"
    assert _value(args, "--algo.advantage.estimator") == "reinforce"
    assert _value(args, "--algo.kl.init_coef") == "0.0"
    assert _value(args, "--actor.adam.lr") == "1e-05"
    assert args[args.index("--actor.adam.betas") + 1 : args.index("--actor.adam.betas") + 3] == [
        "0.9",
        "0.999",
    ]
    assert _value(args, "--actor.adam.eps") == "1e-8"
    assert _value(args, "--actor.adam.weight_decay") == "0.01"
    assert "--train.full_determinism_enable" not in args
    assert _value(args, "--ds.lora.rank") == "32"
    assert _value(args, "--ds.lora.alpha") == "64"
    assert _value(args, "--ds.lora.target_modules") == "all-linear"
    assert "--actor.gradient_checkpointing_reentrant" in args
    assert "--ds.attn_implementation" in args
    assert "--train.colocate_all" in args


@pytest.mark.parametrize(
    ("prompts_per_step", "group_size", "gpu_count"),
    [(1, 1, 2), (3, 1, 2)],
)
def test_build_openrlhf_opd_args_rejects_invalid_actor_global_batch(
    prompts_per_step, group_size, gpu_count
):
    with pytest.raises(ValueError, match="completion batch must be at least and divisible"):
        opd_openrlhf.build_openrlhf_opd_args(
            _config(
                prompts_per_step=prompts_per_step,
                group_size=group_size,
                gpu_count=gpu_count,
            )
        )


def test_build_openrlhf_opd_args_enables_native_resume():
    args = opd_openrlhf.build_openrlhf_opd_args(replace(_config(), resume=True))

    assert "--ckpt.load_enable" in args


def test_build_openrlhf_opd_args_disables_periodic_saves_for_exact_schedule():
    args = opd_openrlhf.build_openrlhf_opd_args(_config(save_at_steps=(2,)))

    assert _value(args, "--ckpt.save_steps") == "-1"


def test_build_openrlhf_opd_args_rejects_nonpositive_objective_scale():
    with pytest.raises(ValueError, match="kl_penalty_coef must be positive"):
        opd_openrlhf.build_openrlhf_opd_args(_config(kl_penalty_coef=0.0))


def test_child_env_excludes_fireworks_and_other_provider_secrets(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    monkeypatch.setenv("FIREWORKS_API_KEY", "do-not-forward")
    monkeypatch.setenv("RUNPOD_API_KEY", "do-not-forward")
    monkeypatch.setenv("HF_TOKEN", "do-not-forward")

    child = opd_openrlhf.build_openrlhf_opd_child_env(
        plugin_dir="/work/plugin",
        max_response_length=320,
        language_model_only=True,
        bridge_url="http://127.0.0.1:1234/teacher/token",
        kl_penalty_coef=0.3,
        seed=42,
        stop_sequences=("</answer>",),
        eos_token_ids=frozenset({2, 3}),
        save_at_steps=(2,),
        final_step=3,
        warmstart_adapter="/work/warmstart",
    )

    assert child["PATH"] == "/usr/bin"
    assert child["NCCL_DEBUG"] == "WARN"
    assert child["FLASH_OPENRLHF_OPD_KL_COEF"] == "0.3"
    assert child["FLASH_OPENRLHF_OPD_SEED"] == "42"
    assert child["FLASH_OPENRLHF_WARMSTART_ADAPTER"] == "/work/warmstart"
    assert json.loads(child["FLASH_OPENRLHF_OPD_STOP_SEQUENCES"]) == ["</answer>"]
    assert json.loads(child["FLASH_OPENRLHF_OPD_EOS_TOKEN_IDS"]) == [2, 3]
    assert json.loads(child["FLASH_OPENRLHF_OPD_EXACT_SAVE_STEPS"]) == [2]
    assert child["FLASH_OPENRLHF_OPD_FINAL_STEP"] == "3"
    assert "FIREWORKS_API_KEY" not in child
    assert "RUNPOD_API_KEY" not in child
    assert "HF_TOKEN" not in child


@requires_openrlhf_source
@requires_torch
def test_pinned_openrlhf_hooks_target_concrete_ray_runtime(monkeypatch):
    helper_path = Path(__file__).with_name("test_grpo_openrlhf.py")
    spec = importlib.util.spec_from_file_location("_flash_grpo_openrlhf_test_helpers", helper_path)
    assert spec is not None
    assert spec.loader is not None
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    namespace, _, ppo_actor_module, _, _ = helpers._install_pinned_sitecustomize_modules(
        monkeypatch
    )

    ray = sys.modules["ray"]

    def wrap_remote(target):
        return type(
            f"RayWrapped{target.__name__}",
            (),
            {"__ray_metadata__": SimpleNamespace(modified_class=target)},
        )

    def remote(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return wrap_remote(args[0])
        return wrap_remote

    ray.remote = remote
    datasets = types.ModuleType("openrlhf.datasets")
    datasets.__path__ = []
    datasets.PromptDataset = type("PromptDataset", (), {})
    monkeypatch.setitem(sys.modules, "openrlhf.datasets", datasets)
    dataset_utils = types.ModuleType("openrlhf.datasets.utils")
    dataset_utils.blending_datasets = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "openrlhf.datasets.utils", dataset_utils)
    experience_maker = types.ModuleType("openrlhf.trainer.ppo_utils.experience_maker")
    experience_maker.RemoteExperienceMaker = type("RemoteExperienceMaker", (), {})
    monkeypatch.setitem(
        sys.modules, "openrlhf.trainer.ppo_utils.experience_maker", experience_maker
    )
    kl_controller = types.ModuleType("openrlhf.trainer.ppo_utils.kl_controller")
    kl_controller.AdaptiveKLController = type("AdaptiveKLController", (), {})
    kl_controller.FixedKLController = type("FixedKLController", (), {})
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_utils.kl_controller", kl_controller)
    launcher = sys.modules["openrlhf.trainer.ray.launcher"]
    launcher.RayActorGroup = type("RayActorGroup", (), {})
    logging_utils = sys.modules["openrlhf.utils.logging_utils"]
    logging_utils.TensorboardLogger = type("TensorboardLogger", (), {})
    logging_utils.WandbLogger = type("WandbLogger", (), {})
    sys.modules["openrlhf.utils.utils"].get_tokenizer = lambda *_args, **_kwargs: None
    ppo_trainer_module = helpers._source_module(
        monkeypatch,
        "openrlhf.trainer.ppo_trainer",
        _OPENRLHF_SOURCE / "openrlhf/trainer/ppo_trainer.py",
    )
    sys.modules["openrlhf.trainer"].ppo_trainer = ppo_trainer_module

    monkeypatch.setenv("FLASH_OPENRLHF_OPD_BRIDGE_URL", "http://127.0.0.1:1/teacher")
    monkeypatch.setenv("FLASH_OPENRLHF_OPD_KL_COEF", "0.37")
    monkeypatch.setenv("FLASH_OPENRLHF_OPD_SEED", "42")
    monkeypatch.setenv("FLASH_OPENRLHF_WARMSTART_ADAPTER", "/work/warmstart")
    exec(
        compile(opd_openrlhf._opd_sitecustomize_extension(), "sitecustomize.py", "exec"),
        namespace,
    )

    runtime = ppo_trainer_module.PPOTrainer.__ray_metadata__.modified_class
    callbacks = []
    namespace["_flash_post_teacher"] = lambda payload: callbacks.append(payload)
    trainer = object.__new__(runtime)
    trainer.args = SimpleNamespace(
        logger=SimpleNamespace(logging_steps=2),
        ckpt=SimpleNamespace(save_steps=1, load_enable=False),
    )
    trainer.wandb_logger = None
    trainer.tensorboard_logger = None
    trainer.actor_model_group = SimpleNamespace(async_run_method=lambda **_kwargs: [])
    trainer.critic_model_group = None
    trainer._latest_eval_metric_value = None
    trainer.best_eval_metric_key = "none"
    trainer.broadcast_to_vllm = lambda: callbacks.append("broadcast")
    namespace["_original_ppo_fit"] = lambda self, *args, **kwargs: "fit-result"

    trainer.save_logs_and_checkpoints(
        1,
        {"distillation_loss": 0.2, "teacher_coverage": 0.75},
        {"global_step": 1},
    )
    result = trainer.fit()

    assert ppo_actor_module.ActorPPOTrainer.training_step is namespace["_flash_opd_training_step"]
    assert runtime.save_logs_and_checkpoints is namespace["_flash_save_logs_and_checkpoints"]
    assert runtime.fit is namespace["_flash_ppo_fit"]
    assert callbacks == [
        {"metrics": {"step": 1, "loss": 0.2, "coverage": 0.75}},
        {"checkpoint": 1},
        "broadcast",
    ]
    assert result == "fit-result"


@requires_torch
def test_sitecustomize_training_step_backpropagates_exact_reverse_kl(monkeypatch):
    torch = pytest.importorskip("torch")

    class _ActorPPOTrainer:
        def ppo_train(self, *_args, **_kwargs):
            return dict(self.original_actor_status)

    class _Actor:
        def __init__(self, *_args, **_kwargs):
            self.model = SimpleNamespace()

    class _SingleTurnAgentExecutor:
        def __init__(self, _remote=None):
            pass

    class _SamplesGenerator:
        pass

    class _ExperienceMaker:
        pass

    class _PPOTrainerRuntime:
        def save_logs_and_checkpoints(self, _global_step, logs_dict=None, client_states=None):
            return None

        def fit(self, *args, **kwargs):
            return None

    class _PolicyModelActorRuntime:
        def save_checkpoint(self, tag, client_states=None, metric_value=None, metric_key=None):
            return (tag, client_states, metric_value, metric_key)

    class _PolicyModelActor:
        __ray_metadata__ = SimpleNamespace(modified_class=_PolicyModelActorRuntime)

    class _PPOTrainer:
        __ray_metadata__ = SimpleNamespace(modified_class=_PPOTrainerRuntime)

    modules = {
        "openrlhf": types.ModuleType("openrlhf"),
        "openrlhf.utils": types.ModuleType("openrlhf.utils"),
        "openrlhf.utils.agent": types.ModuleType("openrlhf.utils.agent"),
        "openrlhf.trainer": types.ModuleType("openrlhf.trainer"),
        "openrlhf.trainer.ppo_trainer": types.ModuleType("openrlhf.trainer.ppo_trainer"),
        "openrlhf.trainer.ppo_utils": types.ModuleType("openrlhf.trainer.ppo_utils"),
        "openrlhf.trainer.ppo_utils.samples_generator": types.ModuleType(
            "openrlhf.trainer.ppo_utils.samples_generator"
        ),
        "openrlhf.trainer.ppo_utils.experience_maker": types.ModuleType(
            "openrlhf.trainer.ppo_utils.experience_maker"
        ),
    }
    modules["openrlhf"].__path__ = []
    modules["openrlhf.utils"].__path__ = []
    modules["openrlhf.trainer"].__path__ = []
    modules["openrlhf.trainer.ppo_utils"].__path__ = []
    modules["openrlhf.utils.agent"].SingleTurnAgentExecutor = _SingleTurnAgentExecutor
    modules["openrlhf.trainer.ppo_trainer"].PPOTrainer = _PPOTrainer
    modules["openrlhf.trainer.ppo_trainer"].prepare_datasets = lambda strategy, tokenizer: (
        strategy,
        tokenizer,
    )
    modules["openrlhf.trainer"].ppo_trainer = modules["openrlhf.trainer.ppo_trainer"]
    modules["openrlhf.trainer.ppo_utils.samples_generator"].SamplesGenerator = _SamplesGenerator
    modules["openrlhf.trainer.ppo_utils.experience_maker"].RemoteExperienceMaker = _ExperienceMaker
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("FLASH_OPENRLHF_OPD_BRIDGE_URL", "http://127.0.0.1:1/teacher")
    monkeypatch.setenv("FLASH_OPENRLHF_OPD_KL_COEF", "0.37")
    monkeypatch.setenv("FLASH_OPENRLHF_OPD_SEED", "42")
    namespace = {
        "__name__": "sitecustomize",
        "__file__": "sitecustomize.py",
        "os": os,
        "functools": functools,
        "_ppo_actor_module": SimpleNamespace(PolicyModelActor=_PolicyModelActor),
        "_ActorPPOTrainer": _ActorPPOTrainer,
        "_Actor": _Actor,
        "_original_execute": lambda *_args, **_kwargs: None,
        "_original_process_response": lambda *_args, **_kwargs: None,
    }
    exec(
        compile(opd_openrlhf._opd_sitecustomize_extension(), "sitecustomize.py", "exec"),
        namespace,
    )
    namespace["_flash_post_teacher"] = lambda _payload: {"ok": True}

    student = torch.tensor(
        [[-0.2, -0.7, -1.1], [-0.4, -0.8, -0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    group_ids = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)
    teacher_logsums = torch.tensor(
        [[-1.4, -1.4, -0.6], [-0.7, -1.8, -1.8]],
        dtype=torch.float64,
    )
    signal_mask = torch.ones_like(group_ids, dtype=torch.bool)

    class _ActorModel:
        def train(self):
            return None

        def __call__(self, *_args, **_kwargs):
            return student, SimpleNamespace()

    class _Strategy:
        ring_attn_group = None
        accumulated_gradient = 1
        world_size = 1

        def all_reduce(self, values, **_kwargs):
            return values

        def backward(self, loss, *_args):
            self.loss = loss

        def optimizer_step(self, *_args, **_kwargs):
            return None

        def get_grad_norm(self, _actor):
            return torch.tensor(0.0)

    trainer = _ActorPPOTrainer()
    trainer.actor = _ActorModel()
    trainer.actor_optim = object()
    trainer.actor_scheduler = SimpleNamespace(get_last_lr=lambda: [1e-5])
    trainer.strategy = _Strategy()
    experience = SimpleNamespace(
        sequences=torch.ones((2, 4), dtype=torch.long),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
        action_mask=torch.ones((2, 3), dtype=torch.bool),
        info={
            "flash_teacher_group_ids": group_ids,
            "flash_teacher_logsums": teacher_logsums,
            "flash_teacher_signal_mask": signal_mask,
            "teacher_coverage": torch.tensor([1.0, 1.0]),
        },
    )

    status = trainer.training_step(
        experience,
        0.0,
        0,
        {"global_batch_size": torch.tensor(2.0)},
    )
    expected = opd_openrlhf.flash_groupwise_reverse_kl(
        student,
        teacher_logsums,
        group_ids,
        signal_mask,
        0.37,
    )

    assert trainer.strategy.loss.item() == pytest.approx(expected.item(), abs=1e-12)
    assert status["metrics"]["distillation_loss"].item() == pytest.approx(
        expected.item(), abs=1e-12
    )
    expected_grad = torch.autograd.grad(expected, student, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(trainer.strategy.loss, student)[0]
    assert torch.allclose(actual_grad, expected_grad, atol=1e-12, rtol=0)


def test_sitecustomize_carries_reverse_kl_teacher_and_lora_hooks():
    source = opd_openrlhf._sitecustomize_source()

    compile(source, "sitecustomize.py", "exec")
    assert "_ActorPPOTrainer.training_step = _flash_opd_training_step" in source
    assert "student_logprobs[row][group_mask].detach().sum()" in source
    assert "torch.stack(row_losses).mean()" in source
    assert "has_window_signal" not in source
    assert "self.strategy.accumulated_gradient,\n        has_current_signal," in source
    assert "self._flash_opd_rollout_ordinals = rollout_ordinals" in source
    assert "_FlashExperienceMaker.make_experience_batch = _flash_make_experience_batch" in source
    assert (
        "_FlashPPOTrainerRuntime.save_logs_and_checkpoints = _flash_save_logs_and_checkpoints"
        in source
    )
    assert "_FlashPPOTrainerRuntime.fit = _flash_ppo_fit" in source
    assert 'getattr(_flash_ray_metadata, "modified_class", None)' in source
    assert "_flash_ppo_trainer_module.prepare_datasets = _flash_prepare_datasets" in source
    assert "shuffle=False" in source
    assert "_FLASH_BRIDGE_TRANSPORT_ATTEMPTS = 3" in source
    assert "FLASH_OPENRLHF_WARMSTART_ADAPTER" in source
    assert 'getattr(result, "missing_keys", [])' in source
    assert 'name.endswith("lora_B.default.weight")' in source
    assert "FIREWORKS_API_KEY" not in source


def test_parent_worker_keeps_filtering_liveness_and_exact_final_checkpoint_gate():
    source = Path(opd_openrlhf.__file__).read_text(encoding="utf-8")

    assert 'liveness_heartbeat("opd_filtering_prompts", progress=lambda: scanned)' in source
    assert "heartbeat=lambda:" not in source
    assert 'if final_save_due(last_step[0], inputs["save_at_steps"]):' in source


def test_run_opd_dispatches_to_openrlhf(monkeypatch):
    calls = []
    monkeypatch.setenv("FLASH_RL_BACKEND", "openrlhf")
    monkeypatch.setattr(opd_openrlhf, "run_opd_openrlhf", lambda: calls.append(True))

    opd.run_opd()

    assert calls == [True]


def test_run_opd_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("FLASH_RL_BACKEND", "unknown")

    with pytest.raises(RuntimeError, match="not a known opd backend"):
        opd.run_opd()

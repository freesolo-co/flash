"""CPU contracts for the OPD migration to verl 0.8.0."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.engine.worker.opd import _gkd_loss_from_logps
from flash.engine.worker.opd_verl import (
    _BridgePrompt,
    _build_opd_child_env,
    _TeacherAlignmentBridge,
    build_opd_verl_overrides,
    encode_shifted_group_metadata,
)
from flash.engine.worker.opd_verl_plugin import (
    _flash_groupwise_reverse_kl_values,
    _full_sequence_signal_sequences,
    _set_current_global_batch_info,
    _signal_sequences,
    deterministic_rollout_seed,
)
from flash.engine.worker.tokenizer_align import TeacherToken


def _aggregate_seq_mean_token_mean(values, response_mask):
    counts = response_mask.sum(dim=-1)
    sequence_losses = (values * response_mask).sum(dim=-1) / counts
    return sequence_losses.mean()


def test_groupwise_reverse_kl_scalar_and_analytic_gradient_match_legacy():
    torch = pytest.importorskip("torch")
    student = torch.tensor(
        [[-0.4, -0.8, -1.2, -2.0], [-0.3, -0.9, -1.1, -1.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher_logsums = torch.tensor(
        [[-0.6, -2.3, -2.3, 0.0], [-0.5, -1.4, -1.4, 0.0]],
        dtype=torch.float64,
    )
    group_ids = torch.tensor([[0, 1, 1, -1], [0, 1, 1, -1]])
    response_mask = torch.ones_like(student, dtype=torch.bool)
    coef = 0.7

    values = _flash_groupwise_reverse_kl_values(
        student, teacher_logsums, group_ids, response_mask, coef
    )
    verl_loss = _aggregate_seq_mean_token_mean(values, response_mask)
    verl_gradient = torch.autograd.grad(verl_loss, student, retain_graph=True)[0]

    legacy_losses = [
        _gkd_loss_from_logps(
            student[row],
            [([0], float(teacher_logsums[row, 0])), ([1, 2], float(teacher_logsums[row, 1]))],
            kl_coef=coef,
        )
        for row in range(2)
    ]
    legacy_loss = torch.stack(legacy_losses).mean()
    legacy_gradient = torch.autograd.grad(legacy_loss, student)[0]

    assert torch.allclose(verl_loss, legacy_loss, atol=1e-12, rtol=1e-12)
    assert torch.allclose(verl_gradient, legacy_gradient, atol=1e-12, rtol=1e-12)
    assert torch.equal(verl_gradient[:, 3], torch.zeros(2, dtype=torch.float64))


def test_dynamic_microbatches_refresh_global_batch_info_and_match_flash_sequence_mean():
    torch = pytest.importorskip("torch")
    steps = [
        {
            "student": [
                [-0.4, -0.8, -1.2, -2.0],
                [-0.3, -0.9, -1.7, -2.1],
                [-0.6, -1.4, -2.2, -2.8],
            ],
            "teacher": [
                [-0.6, -2.3, -2.3, 0.0],
                [-0.5, -1.5, 0.0, 0.0],
                [-0.9, 0.0, 0.0, 0.0],
            ],
            "groups": [[0, 1, 1, -1], [0, 1, -1, -1], [0, -1, -1, -1]],
            "mask": [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]],
            "splits": [(0, 2), (2, 3)],
        },
        {
            "student": [[-0.2, -0.7, -1.3, -1.9], [-0.5, -1.0, -1.6, -2.4]],
            "teacher": [[-0.4, -1.8, -1.8, 0.0], [-0.8, -1.1, 0.0, 0.0]],
            "groups": [[0, 1, 1, -1], [0, 1, -1, -1]],
            "mask": [[1, 1, 1, 1], [1, 1, 1, 0]],
            "splits": [(0, 1), (1, 2)],
        },
    ]
    coef = 0.7
    observed = []
    expected = []

    for step in steps:
        student = torch.tensor(step["student"], dtype=torch.float64, requires_grad=True)
        teacher = torch.tensor(step["teacher"], dtype=torch.float64)
        group_ids = torch.tensor(step["groups"])
        response_mask = torch.tensor(step["mask"], dtype=torch.bool)
        config = SimpleNamespace(
            global_batch_info={
                "dp_size": 99,
                "batch_num_tokens": 1,
                "global_batch_size": 99,
                "loss_scale_factor": 99,
            },
            loss_scale_factor=None,
        )
        micro_losses = []
        global_batch_size = student.shape[0]
        batch_num_tokens = int(response_mask.sum().item())
        for start, end in step["splits"]:
            data = {
                "dp_size": 1,
                "batch_num_tokens": batch_num_tokens,
                "global_batch_size": global_batch_size,
            }
            _set_current_global_batch_info(config, data)
            values = _flash_groupwise_reverse_kl_values(
                student[start:end],
                teacher[start:end],
                group_ids[start:end],
                response_mask[start:end],
                coef,
            )
            counts = response_mask[start:end].sum(dim=-1)
            sequence_losses = (values * response_mask[start:end]).sum(dim=-1) / counts
            micro_losses.append(
                sequence_losses.sum()
                / config.global_batch_info["global_batch_size"]
                * config.global_batch_info["dp_size"]
            )
            _set_current_global_batch_info(config, data)
        actual = torch.stack(micro_losses).sum()
        legacy = torch.stack(
            [
                _gkd_loss_from_logps(
                    student[row],
                    [
                        (
                            torch.nonzero(group_ids[row].eq(group_id), as_tuple=False)
                            .flatten()
                            .tolist(),
                            float(teacher[row][group_ids[row].eq(group_id)][0]),
                        )
                        for group_id in torch.unique(
                            group_ids[row][group_ids[row].ge(0)], sorted=True
                        )
                    ],
                    kl_coef=coef,
                )
                for row in range(student.shape[0])
            ]
        ).mean()
        actual_gradient = torch.autograd.grad(actual, student, retain_graph=True)[0]
        legacy_gradient = torch.autograd.grad(legacy, student)[0]

        assert config.global_batch_info == {
            "dp_size": 1,
            "batch_num_tokens": batch_num_tokens,
            "global_batch_size": global_batch_size,
            "loss_scale_factor": None,
        }
        assert torch.allclose(actual, legacy, atol=1e-12, rtol=1e-12)
        assert torch.allclose(actual_gradient, legacy_gradient, atol=1e-12, rtol=1e-12)
        observed.append(float(actual))
        expected.append(float(legacy))

    assert observed == pytest.approx(expected, abs=1e-12, rel=1e-12)
    assert len(steps[0]["student"]) != len(steps[1]["student"])


def test_no_signal_sequence_is_excluded_before_actor_training():
    torch = pytest.importorskip("torch")
    group_ids = torch.tensor([[0, -1, -1], [-1, -1, -1], [2, 2, -1]])
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    assert _signal_sequences(group_ids, response_mask).tolist() == [True, False, True]
    full_sequence_ids = torch.tensor(
        [[-1, -1, 0, -1], [-1, -1, -1, -1], [-1, 2, 2, -1]]
    ).unsqueeze(-1)
    assert _full_sequence_signal_sequences(full_sequence_ids).tolist() == [True, False, True]


def test_shifted_group_metadata_uses_verl_prediction_layout():
    teacher_ids, teacher_logprobs = encode_shifted_group_metadata(
        prompt_length=3,
        response_length=4,
        groups=[([0], -0.5), ([1, 2], -1.75)],
    )
    assert teacher_ids == [-1, -1, 0, 1, 1, -1, -1]
    assert teacher_logprobs == [0.0, 0.0, -0.5, -1.75, -1.75, 0.0, 0.0]
    assert teacher_ids[2:6] == [0, 1, 1, -1]


class _BridgeTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, *, skip_special_tokens):
        mapping = {65: "A", 66: "B", 99: "" if skip_special_tokens else "<eos>"}
        return "".join(mapping[int(token_id)] for token_id in token_ids)


class _BridgeTeacher:
    def score(self, prompt_text, completion_text):
        assert prompt_text == "User: question\nAssistant: "
        assert completion_text == "AB"
        return [
            TeacherToken(text="A", logprob=-0.4, start=0, end=1),
            TeacherToken(text="B", logprob=-0.7, start=1, end=2),
        ]


def test_bridge_verifies_prompt_and_serializes_aligned_native_fields():
    bridge = _TeacherAlignmentBridge(
        prompts=[
            _BridgePrompt(
                messages=[{"role": "user", "content": "question"}],
                prompt_ids=(10, 11),
            )
        ],
        tokenizer=_BridgeTokenizer(),
        teacher=_BridgeTeacher(),
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=lambda: None,
    )
    encoded = bridge.score(0, [10, 11, 65, 66, 99])
    assert encoded["teacher_ids"] == [-1, 0, 1, -1, -1]
    assert encoded["teacher_logprobs"] == [0.0, -0.4, -0.7, 0.0, 0.0]
    assert bridge.aligned_sequences == 1
    assert bridge.generated_tokens == 3
    with pytest.raises(ValueError, match="prompt ids"):
        bridge.score(0, [10, 12, 65, 99])


def _config(**overrides):
    config = {
        "train_files": ["/w/train.parquet"],
        "val_files": ["/w/val.parquet"],
        "train_batch_size": 8,
        "max_prompt_length": 1024,
        "max_response_length": 512,
        "model_path": "/models/student",
        "lora_rank": 32,
        "lora_alpha": 64,
        "target_modules": "all-linear",
        "learning_rate": 1e-5,
        "local_dir": "/w/checkpoints",
        "save_freq": 20,
        "n_gpus_per_node": 4,
        "ulysses_sequence_parallel_size": 4,
        "seed": 42,
        "project_name": "flash",
        "experiment_name": "opd-test",
        "total_training_steps": 10,
        "group_size": 2,
        "bridge_url": "http://127.0.0.1:1234",
        "bridge_token": "token",
        "kl_penalty_coef": 0.5,
    }
    config.update(overrides)
    return config


def test_overrides_match_verl_0_8_sync_distillation_contract():
    overrides = dict(value.split("=", 1) for value in build_opd_verl_overrides(_config()))
    assert overrides["distillation._target_"] == "flash_opd_verl_plugin.FlashRemoteDistillationConfig"
    assert overrides["distillation.distillation_loss.loss_mode"] == "flash_groupwise_reverse_kl"
    assert overrides["distillation.distillation_loss.use_policy_gradient"] == "false"
    assert overrides["distillation.distillation_loss.use_task_rewards"] == "false"
    assert overrides["actor_rollout_ref.actor.loss_agg_mode"] == "seq-mean-token-mean"
    assert overrides["actor_rollout_ref.actor.use_kl_loss"] == "false"
    assert overrides["algorithm.use_kl_in_reward"] == "false"
    assert overrides["actor_rollout_ref.model.use_remove_padding"] == "true"
    assert overrides["actor_rollout_ref.rollout.tensor_model_parallel_size"] == "4"
    assert overrides["actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size"] == "4"
    assert overrides["actor_rollout_ref.rollout.max_model_len"] == "32768"
    assert overrides["distillation.n_gpus_per_node"] == "0"
    assert overrides["distillation.nnodes"] == "0"
    assert overrides["distillation.teacher_key"] == "index"
    assert "actor.engine.ulysses_sequence_parallel_size" not in overrides
    assert "ref_log_prob" not in " ".join(overrides)


def test_child_environment_keeps_bridge_but_excludes_teacher_key(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWORKS_API_KEY", "teacher-secret")
    monkeypatch.setenv("HF_TOKEN", "hub-secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    child = _build_opd_child_env(
        shim_dir=str(tmp_path),
        wandb_enabled=False,
        bridge_url="http://127.0.0.1:4444",
        bridge_token="bridge-token",
        seed=42,
        stop_sequences=("</answer>",),
        eos_token_ids=frozenset({1, 2}),
    )
    assert child["FLASH_OPD_BRIDGE_URL"] == "http://127.0.0.1:4444"
    assert child["FLASH_OPD_BRIDGE_TOKEN"] == "bridge-token"
    assert child["VERL_USE_EXTERNAL_MODULES"] == "flash_opd_verl_plugin"
    assert child["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "FIREWORKS_API_KEY" not in child
    assert "HF_TOKEN" not in child


def test_runner_rejects_structured_outputs_before_preparation():
    from flash import runner
    from flash.spec import JobSpec, TrainSpec

    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(structured_outputs='{"json":{"type":"object"}}'),
    )
    with pytest.raises(ValueError, match="forced-position metadata"):
        runner._require_supported_opd_verl_spec(spec)


def test_runner_rejects_multiturn_tool_flags_before_preparation():
    from flash import runner
    from flash.spec import EnvironmentSpec, JobSpec

    for flag in ("multi_turn", "is_tool_env", "tool_calling"):
        spec = JobSpec(
            model="Qwen/Qwen3.5-4B",
            algorithm="opd",
            environment=EnvironmentSpec(id="local", params={flag: True}),
        )
        with pytest.raises(ValueError, match="not yet supported on the verl OPD backend"):
            runner._require_supported_opd_verl_spec(spec)


@pytest.mark.parametrize(
    ("multi_turn", "is_tool_env", "records", "message"),
    [
        (True, False, [], "multi-turn and tool-calling"),
        (False, True, [], "multi-turn and tool-calling"),
        (
            False,
            False,
            [{"input": [{"role": "user", "content": "look"}], "image": "image.png"}],
            "multimodal OPD",
        ),
    ],
)
def test_runner_environment_gate_rejects_before_provisioning(
    monkeypatch, multi_turn, is_tool_env, records, message
):
    from flash.envs import registry
    from flash.runner import lifecycle
    from flash.spec import EnvironmentSpec, JobSpec, TrainSpec

    class FakeEnvironment:
        def __init__(self):
            self.multi_turn = multi_turn
            self.is_tool_env = is_tool_env

        def dataset(self):
            return records

        def prompt_messages(self, record):
            return record.get("input", [])

    calls = []
    monkeypatch.setattr(
        registry,
        "load_environment",
        lambda *_args, **_kwargs: calls.append("load") or FakeEnvironment(),
    )
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        environment=EnvironmentSpec(id="local"),
        train=TrainSpec(max_examples=1),
    )

    with pytest.raises(ValueError, match=message):
        lifecycle._preflight_opd_verl_environment(spec)
    assert calls == ["load"]


def test_deterministic_seed_uses_every_rollout_identity_component():
    baseline = deterministic_rollout_seed(42, 3, 7, 1)
    assert baseline == deterministic_rollout_seed(42, 3, 7, 1)
    assert len(
        {
            baseline,
            deterministic_rollout_seed(43, 3, 7, 1),
            deterministic_rollout_seed(42, 4, 7, 1),
            deterministic_rollout_seed(42, 3, 8, 1),
            deterministic_rollout_seed(42, 3, 7, 2),
        }
    ) == 5
    assert 0 <= baseline < 2**63


def test_worker_dispatch_and_multi_gpu_route_to_verl():
    import inspect

    import flash.engine.worker as worker
    from flash import runner

    assert worker.run_opd.__module__ == "flash.engine.worker.opd_verl"
    assert worker.run_opd.__name__ == "run_opd_verl"
    assert "opd" in runner._MULTI_GPU_ALGORITHMS
    assert "run_verl_training" in inspect.getsource(worker.run_opd)


def test_plugin_identifiers_remain_provider_neutral():
    import inspect

    import flash.engine.worker.opd_verl_plugin as plugin

    source = inspect.getsource(plugin).lower()
    forbidden = ("parasail", "fireworks")
    for name in forbidden:
        assert f"class {name}" not in source
        assert f"def {name}" not in source
        assert f"_{name}" not in source

"""Active provider-neutral worker contracts retained after Serverless deletion."""

from __future__ import annotations

import inspect

import pytest


def _spec(*, algorithm: str = "grpo", secrets: tuple[str, ...] = ()):
    from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec

    return JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm=algorithm,
        environment=EnvironmentSpec(id="owner/env", secrets=secrets),
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )


def test_worker_env_owns_allocator_policy_and_filters_removed_knobs(monkeypatch):
    from flash.providers._lifecycle.net.worker import build_worker_env

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:999")
    monkeypatch.setenv("VLLM_USE_V1", "0")
    grpo = build_worker_env(_spec(), 0)
    sft = build_worker_env(_spec(algorithm="sft"), 0)

    assert "expandable_segments" not in grpo["PYTORCH_CUDA_ALLOC_CONF"]
    assert grpo["PYTORCH_ALLOC_CONF"] == grpo["PYTORCH_CUDA_ALLOC_CONF"]
    assert sft["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert "VLLM_USE_V1" not in grpo


def test_opd_worker_env_uses_sleep_safe_allocator_and_bounded_teacher_transport():
    from flash.providers._lifecycle.net.worker import build_worker_env

    env = build_worker_env(
        _spec(algorithm="opd"),
        0,
        runtime_secrets={
            "FLASH_PUBLIC_URL": "https://broker.example",
            "FLASH_TEACHER_CAPABILITY": "capability-test-value",
            "PARASAIL_API_KEY": "must-not-forward",
        },
    )

    assert "expandable_segments" not in env["PYTORCH_CUDA_ALLOC_CONF"]
    assert env["FLASH_PUBLIC_URL"] == "https://broker.example"
    assert env["FLASH_TEACHER_CAPABILITY"] == "capability-test-value"
    assert "PARASAIL_API_KEY" not in env


def test_worker_env_forwards_only_declared_runtime_secrets_and_lists_redaction_names():
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV
    from flash.providers._lifecycle.net.worker import build_worker_env

    env = build_worker_env(
        _spec(secrets=("SERPAPI_API_KEY", "AWS_SECRET_ACCESS_KEY")),
        0,
        runtime_secrets={
            "SERPAPI_API_KEY": "serp-user",
            "AWS_SECRET_ACCESS_KEY": "aws-user",
            "UNDECLARED_API_KEY": "must-not-forward",
        },
    )

    assert env["SERPAPI_API_KEY"] == "serp-user"
    assert env["AWS_SECRET_ACCESS_KEY"] == "aws-user"
    assert "UNDECLARED_API_KEY" not in env
    assert set(env[SECRET_ENV_KEYS_ENV].split(",")) == {
        "AWS_SECRET_ACCESS_KEY",
        "SERPAPI_API_KEY",
    }


def test_removed_worker_keys_cannot_reenter_through_declared_secrets():
    from flash.providers._lifecycle.net.worker import build_worker_env

    env = build_worker_env(
        _spec(secrets=("FLASH_CHALK_SPEC", "FLASH_TRITON_LORA", "MY_TOKEN")),
        0,
        runtime_secrets={
            "FLASH_CHALK_SPEC": "freesolo-chalk==0.5.7",
            "FLASH_TRITON_LORA": "1",
            "MY_TOKEN": "keep-me",
        },
    )

    assert "FLASH_CHALK_SPEC" not in env
    assert "FLASH_TRITON_LORA" not in env
    assert env["MY_TOKEN"] == "keep-me"


def test_worker_and_control_plane_share_attempt_scoped_artifact_names():
    from flash.engine.worker.io.hf import error_artifact_name as worker_name
    from flash.providers.artifacts.hf import error_artifact_name as plane_name

    for phase in ("sft", "rl", "opd"):
        for attempt in (0, 1, 7):
            assert worker_name(phase, attempt) == plane_name(phase, attempt)
    assert worker_name("sft", 0) != worker_name("sft", 1)
    with pytest.raises(ValueError, match="attempt must be"):
        plane_name("sft", True)


def test_sft_training_keeps_active_optimization_wiring():
    from flash.engine.profiling import sft_image_rows, sft_workload
    from flash.engine.worker.entry import sft
    from flash.engine.worker.train.entry import sft_train
    from flash.engine.worker.train.sft.child import plugin as sft_plugin
    from flash.engine.worker.train.sft.setup import config as sft_config

    assert "run_sft_train()" in inspect.getsource(sft.run_sft)
    assert "_pretokenize_completion_only(" in inspect.getsource(sft_workload)
    assert "completion_mask_from_ids(" in inspect.getsource(sft_image_rows)
    train_source = inspect.getsource(sft_train) + inspect.getsource(sft_config)
    assert "resolve_vocab_size(" in train_source
    assert "sft_grad_accum(" in train_source
    assert "grad_checkpointing_on(" in train_source
    assert "grpo_use_reentrant(" in train_source
    assert "create_loraplus_optimizer" in inspect.getsource(sft_plugin)


def test_snapshot_weight_validation_rejects_config_only_snapshot(tmp_path):
    from flash.engine.worker.io.hf import _snapshot_has_weights

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    assert not _snapshot_has_weights(str(snapshot))

    (snapshot / "model.safetensors-00001-of-00001.safetensors").write_text("weights")
    assert _snapshot_has_weights(str(snapshot))

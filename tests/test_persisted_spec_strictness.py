"""Strict persisted JobSpec types, resolved semantics, and activation boundaries."""

from __future__ import annotations

import math
from copy import deepcopy

import pytest

from flash.core.spec import JobSpec

PROJECT = "11111111-1111-4111-8111-111111111111"


def _payload(*, algorithm: str = "grpo") -> dict:
    train = {
        "epochs": 1,
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-5,
        "max_context_tokens": 1024,
        "max_steps": 4,
        "max_examples": 8,
    }
    if algorithm == "sft":
        train["batch_size"] = 2
    else:
        train.update(
            prompts_per_step=8,
            group_size=4,
            temperature=0.7,
            max_completion_tokens=128,
            kl_penalty_coef=0.1,
        )
    if algorithm == "opd":
        train["teacher_model"] = "glm-5.2"
    return {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": algorithm,
        "project": PROJECT,
        "environment": {
            "id": "owner/project/example",
            "params": {"difficulty": "easy"},
            "pip": ["requests"],
            "secrets": ["TOKEN"],
        },
        "train": train,
        "gpu": {
            "type": ["H100", "A100 PCIe"],
            "disk_gb": 120,
            "max_wall_seconds": 3600,
            "max_retries": 2,
            "network_volume": "weights",
            "network_volume_gb": 100,
            "count": 1,
        },
        "run_id": "strict-persisted",
        "seed": 42,
        "thinking": False,
        "wandb": {"project": "tests", "run_name": None},
        "model_revision": "a" * 40,
        "model_revision_auto": True,
        "model_revision_force_pin": False,
        "gpu_count_auto": False,
        "workload_profile_input_digest": "",
        "workload_profile_producer_version": "",
        "workload_profile": {},
    }


def _set(payload: dict, path: str, value: object) -> dict:
    changed = deepcopy(payload)
    section, _, field = path.partition(".")
    if field:
        changed[section][field] = value
    else:
        changed[section] = value
    return changed


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        ("thinking", "false"),
        ("model_revision_auto", 1),
        ("gpu_count_auto", None),
        ("seed", True),
        ("train.epochs", "1"),
        ("train.lora_rank", 16.0),
        ("train.max_examples", False),
        ("gpu.count", 1.0),
        ("gpu.max_retries", "2"),
        ("train.learning_rate", "0.1"),
        ("train.temperature", True),
        ("train.kl_penalty_coef", math.inf),
        ("model", 7),
        ("train.teacher_model", None),
        ("environment.params", None),
        ("workload_profile", []),
        ("wandb", None),
        ("environment.pip", "requests"),
        ("environment.secrets", ["TOKEN", 7]),
        ("train.stop_sequences", "END"),
        ("train.save_at_steps", [1, True]),
        ("train.credit_assignment", "PER_TURN"),
        ("algorithm", "GRPO"),
        ("gpu.type_fallbacks", None),
    ],
)
def test_persisted_fields_reject_coercion_and_wrong_json_types(path, invalid) -> None:
    payload = _payload()
    if path == "gpu.type_fallbacks":
        payload["gpu"]["type"] = "H100"
    with pytest.raises((TypeError, ValueError)):
        JobSpec.from_dict(_set(payload, path, invalid))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("train.epochs", None),
        ("train.learning_rate", None),
        ("train.max_examples", None),
        ("gpu.network_volume", None),
        ("wandb.project", None),
    ],
)
def test_null_is_retained_only_for_optional_fields(path, value) -> None:
    spec = JobSpec.from_dict(_set(_payload(), path, value))
    section, _, field = path.partition(".")
    target = getattr(spec, section) if field else spec
    assert getattr(target, field or section) is None


def test_retained_sequence_and_numeric_alternatives_round_trip() -> None:
    payload = _payload()
    payload["environment"]["pip"] = ("requests", "httpx")
    payload["environment"]["secrets"] = ("TOKEN",)
    payload["train"]["stop_sequences"] = ["END"]
    payload["gpu"]["type"] = ("h100", "a100")
    payload["train"]["temperature"] = 1

    spec = JobSpec.from_dict(payload)

    assert spec.environment.pip == ("requests", "httpx")
    assert spec.environment.secrets == ("TOKEN",)
    assert spec.train.stop_sequences == ("END",)
    assert spec.gpu.acceptable_types == ("H100", "A100 PCIe")
    assert spec.train.temperature == 1.0


@pytest.mark.parametrize(
    ("algorithm", "path", "value", "match"),
    [
        ("sft", "train.epochs", 0, "epochs must be positive"),
        ("sft", "train.lora_rank", 0, "lora_rank must be positive"),
        ("sft", "train.lora_alpha", 0, "lora_alpha must be positive"),
        ("sft", "train.learning_rate", 0.0, "learning_rate must be positive"),
        ("sft", "train.batch_size", 0, "batch_size must be positive"),
        ("grpo", "train.prompts_per_step", 0, "prompts_per_step must be positive"),
        ("grpo", "train.max_context_tokens", 0, "max_context_tokens must be positive"),
        ("grpo", "train.max_completion_tokens", 0, "max_completion_tokens must be positive"),
        ("sft", "train.max_examples", -1, "max_examples must be nonnegative"),
        ("sft", "train.max_steps", -1, "max_steps must be nonnegative"),
        ("grpo", "train.temperature", -0.1, "temperature must be nonnegative"),
        ("grpo", "train.kl_penalty_coef", -0.1, "kl_penalty_coef must be nonnegative"),
        ("grpo", "train.entropy_quantile", 1.1, "entropy_quantile must be between"),
        (
            "grpo",
            "train.thinking_length_penalty_coef",
            -0.1,
            "thinking_length_penalty_coef must be between",
        ),
        ("sft", "gpu.disk_gb", 0, "disk_gb must be positive"),
        ("sft", "gpu.max_wall_seconds", 0, "max_wall_seconds must be positive"),
        ("sft", "gpu.max_retries", -1, "max_retries must be nonnegative"),
        ("opd", "train.kl_penalty_coef", 0.0, "kl_penalty_coef must be positive for opd"),
    ],
)
def test_resolved_semantic_bounds(algorithm, path, value, match) -> None:
    with pytest.raises(ValueError, match=match):
        JobSpec.from_dict(_set(_payload(algorithm=algorithm), path, value))


def test_max_steps_zero_persists_as_the_derived_horizon_sentinel() -> None:
    spec = JobSpec.from_dict(_set(_payload(algorithm="sft"), "train.max_steps", 0))
    assert spec.train.max_steps == 0
    assert spec.to_internal_dict()["train"]["max_steps"] == 0


@pytest.mark.parametrize(
    ("algorithm", "path", "value"),
    [
        ("sft", "train.prompts_per_step", 8),
        ("sft", "train.group_size", 4),
        ("sft", "train.temperature", 0.7),
        ("sft", "train.teacher_model", "glm-5.2"),
        ("grpo", "train.batch_size", 2),
        ("grpo", "train.teacher_model", "glm-5.2"),
        ("opd", "train.batch_size", 2),
        ("opd", "train.entropy_quantile", 0.5),
    ],
)
def test_resolved_algorithm_applicability(algorithm, path, value) -> None:
    with pytest.raises(ValueError, match="does not apply"):
        JobSpec.from_dict(_set(_payload(algorithm=algorithm), path, value))


@pytest.mark.parametrize("group_size", [1, 3, 6, 32])
def test_persisted_grpo_rejects_unsupported_group_sizes(group_size) -> None:
    with pytest.raises(ValueError, match="group_size must be one of"):
        JobSpec.from_dict(_set(_payload(), "train.group_size", group_size))


def test_persisted_grpo_rejects_excess_completion_product() -> None:
    payload = _payload()
    payload["train"].update(prompts_per_step=64, group_size=16)
    with pytest.raises(ValueError, match=r"prompts_per_step \* train.group_size must be <= 512"):
        JobSpec.from_dict(payload)


def test_warm_start_persisted_topology_remains_resolved() -> None:
    payload = _payload(algorithm="grpo")
    payload["train"].update(
        init_from_adapter="Freesolo-Co/run-artifacts:rl/source-run",
        init_from_adapter_revision="b" * 40,
        lora_rank=64,
        lora_alpha=96,
    )

    spec = JobSpec.from_dict(payload)

    assert spec.train.init_from_adapter
    assert spec.train.lora_rank == 64
    assert spec.train.lora_alpha == 96
    assert "lora_rank" not in spec.to_dict()["train"]
    assert spec.to_internal_dict()["train"]["lora_rank"] == 64
    assert spec.to_internal_dict()["train"]["lora_alpha"] == 96


def test_activation_enforces_current_known_model_lora_cap() -> None:
    import flash.runner.lifecycle.status as runner_status
    from flash.runner.lifecycle import preparation
    from flash.runner.lifecycle.state import RunStatus

    payload = _payload()
    payload.update(model_revision="", model_revision_auto=False)
    payload["train"].update(lora_rank=129, lora_alpha=258)
    spec = JobSpec.from_dict(payload)
    status = RunStatus(
        run_id=spec.run_id,
        state="running",
        spec=spec.to_dict(),
        effective_preparation={
            "version": 1,
            "worker_spec": spec.to_internal_dict(),
            "preparation_digest": preparation._preparation_digest(spec, spec, None),
        },
    )

    with pytest.raises(ValueError, match="serving max_lora_rank=128"):
        runner_status.effective_spec_from_status(status)


def test_exact_internal_round_trip_remains_stable_for_strict_values() -> None:
    spec = JobSpec.from_dict(_payload())
    assert JobSpec.from_dict(spec.to_internal_dict()).to_internal_dict() == spec.to_internal_dict()
    assert JobSpec.from_json(spec.to_json()).to_json() == spec.to_json()

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from flash.core.grpo import (
    DEFAULT_GRPO_GROUP_SIZE,
    DEFAULT_GRPO_PROMPTS_PER_STEP,
    MAX_GRPO_COMPLETIONS_PER_STEP,
    SUPPORTED_GRPO_GROUP_SIZES,
    resolve_grpo_rollout_shape,
)
from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
from flash.cost.types import RunConfig
from flash.engine.plan.recipe import RECIPE
from flash.schema import spec_from_dict
from flash.schema.fields import ConfigError


def _public_grpo_train(**train):
    return {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "environment": {"id": "owner/project/env"},
        "train": {"epochs": 1, **train},
    }


def _internal_grpo_train(**train):
    return {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "environment": {"id": "owner/project/env"},
        "train": {"epochs": 1, **train},
    }


@pytest.mark.parametrize("group_size", SUPPORTED_GRPO_GROUP_SIZES)
def test_grpo_rollout_contract_accepts_supported_groups(group_size):
    shape = resolve_grpo_rollout_shape(1, group_size)
    assert (shape.prompts_per_step, shape.group_size) == (1, group_size)


@pytest.mark.parametrize("group_size", [0, 1, 3, 5, 6, 7, 9, 16, 1024])
def test_grpo_rollout_contract_rejects_every_other_group(group_size):
    with pytest.raises(ValueError, match="must be one of"):
        resolve_grpo_rollout_shape(1, group_size)


@pytest.mark.parametrize("value", [False, True, 1.0, "8", [], {}])
def test_grpo_rollout_contract_rejects_non_integer_groups(value):
    with pytest.raises(TypeError, match="group_size must be an integer"):
        resolve_grpo_rollout_shape(1, value)


@pytest.mark.parametrize("value", [False, True, 1.0, "64", [], {}])
def test_grpo_rollout_contract_rejects_non_integer_prompts(value):
    with pytest.raises(TypeError, match="prompts_per_step must be an integer"):
        resolve_grpo_rollout_shape(value, 8)


@pytest.mark.parametrize("value", [0, -1, -64])
def test_grpo_rollout_contract_rejects_nonpositive_prompts(value):
    with pytest.raises(ValueError, match="must be positive"):
        resolve_grpo_rollout_shape(value, 8)


@pytest.mark.parametrize("group_size", SUPPORTED_GRPO_GROUP_SIZES)
def test_grpo_rollout_contract_accepts_exact_ceiling_and_rejects_first_over(group_size):
    prompts = MAX_GRPO_COMPLETIONS_PER_STEP // group_size
    assert resolve_grpo_rollout_shape(prompts, group_size).completions_per_step == 512
    with pytest.raises(ValueError, match=r"must be <= 512"):
        resolve_grpo_rollout_shape(prompts + 1, group_size)


def test_grpo_rollout_contract_resolves_omitted_defaults_without_mutation():
    authored = {"prompts_per_step": None, "group_size": None}
    before = dict(authored)
    shape = resolve_grpo_rollout_shape(**authored)
    assert shape.prompts_per_step == DEFAULT_GRPO_PROMPTS_PER_STEP
    assert shape.group_size == DEFAULT_GRPO_GROUP_SIZE
    assert authored == before


def test_recipe_uses_shared_grpo_defaults():
    assert RECIPE.rl.prompts_per_step == DEFAULT_GRPO_PROMPTS_PER_STEP
    assert RECIPE.rl.group_size == DEFAULT_GRPO_GROUP_SIZE


def test_public_parser_rejects_shape_before_gpu_validation(monkeypatch):
    import flash.schema as schema

    calls = []
    monkeypatch.setattr(
        schema,
        "_validate_gpu_section",
        lambda *_args, **_kwargs: calls.append("gpu") or pytest.fail("gpu validation ran"),
    )
    with pytest.raises(ConfigError, match=r"must be <= 512"):
        spec_from_dict(_public_grpo_train(prompts_per_step=65, group_size=8))
    assert calls == []


def test_public_parser_preserves_authored_shape_exactly():
    spec = spec_from_dict(_public_grpo_train(prompts_per_step=127, group_size=4))
    assert spec.train.prompts_per_step == 127
    assert spec.train.group_size == 4


def test_direct_jobspec_and_persisted_decode_reject_unsupported_historical_shapes():
    with pytest.raises(ValueError, match="must be one of"):
        JobSpec(
            algorithm="grpo",
            environment=EnvironmentSpec(id="owner/project/env"),
            train=TrainSpec(prompts_per_step=64, group_size=16),
        )
    with pytest.raises(ValueError, match="must be one of"):
        JobSpec.from_dict(_internal_grpo_train(prompts_per_step=64, group_size=16))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("prompts_per_step", True),
        ("prompts_per_step", "64"),
        ("group_size", False),
        ("group_size", "8"),
    ],
)
def test_persisted_grpo_decode_rejects_permissive_integer_coercions(key, value):
    with pytest.raises(TypeError, match=f"{key} must be an integer"):
        JobSpec.from_dict(_internal_grpo_train(**{key: value}))


def test_direct_jobspec_omitted_shape_keeps_serialized_fields_omitted():
    spec = JobSpec(
        algorithm="grpo",
        environment=EnvironmentSpec(id="owner/project/env"),
        train=TrainSpec(),
    )
    assert spec.train.prompts_per_step is None
    assert spec.train.group_size is None
    assert spec.to_dict()["train"]["prompts_per_step"] is None
    assert spec.to_dict()["train"]["group_size"] is None
    assert asdict(spec.train)["prompts_per_step"] is None


def test_direct_runconfig_validates_and_normalizes_shared_grpo_shape():
    config = RunConfig("Qwen/Qwen3.5-4B", "grpo", 1)
    normalized = config.normalized()
    assert normalized.batch_size == DEFAULT_GRPO_PROMPTS_PER_STEP
    assert normalized.group_size == DEFAULT_GRPO_GROUP_SIZE
    with pytest.raises(ValueError, match=r"must be <= 512"):
        RunConfig("Qwen/Qwen3.5-4B", "grpo", 1, batch_size=129, group_size=4)


def test_allocator_rejects_before_sizing_provider_or_capacity_calls(monkeypatch):
    import flash.providers.allocator as allocator

    calls = []
    monkeypatch.setattr(
        allocator,
        "required_vram_gb",
        lambda *_args, **_kwargs: calls.append("sizing") or 1,
    )
    monkeypatch.setattr(
        allocator,
        "available_providers",
        lambda: calls.append("providers") or (),
    )
    monkeypatch.setattr(
        allocator,
        "_gather_candidates",
        lambda *_args, **_kwargs: calls.append("capacity") or ([], False, {}),
    )
    with pytest.raises(ValueError, match=r"must be <= 512"):
        allocator.allocate(
            "Qwen/Qwen3.5-4B",
            "grpo",
            train={"prompts_per_step": 257, "group_size": 2},
        )
    assert calls == []


@pytest.mark.parametrize(
    ("algorithm", "prompts_per_step", "group_size"),
    [("opd", 8, 1), ("sft", None, None)],
)
def test_non_grpo_jobspec_behavior_is_unchanged(algorithm, prompts_per_step, group_size):
    spec = JobSpec(
        algorithm=algorithm,
        environment=EnvironmentSpec(id="owner/project/env"),
        train=TrainSpec(prompts_per_step=prompts_per_step, group_size=group_size),
    )
    assert spec.algorithm == algorithm


def test_grpo_worker_env_reasserts_managed_native_thread_policy(monkeypatch):
    from flash.core.grpo import GRPO_NATIVE_THREAD_ENV
    from flash.providers._lifecycle.worker import build_worker_env

    spec = JobSpec(
        algorithm="grpo",
        environment=EnvironmentSpec(id="owner/project/env", secrets=tuple(GRPO_NATIVE_THREAD_ENV)),
        train=TrainSpec(prompts_per_step=64, group_size=8),
    )
    hostile = dict.fromkeys(GRPO_NATIVE_THREAD_ENV, "999")
    env = build_worker_env(spec, spec.seed, runtime_secrets=hostile)
    assert {key: env[key] for key in GRPO_NATIVE_THREAD_ENV} == dict(GRPO_NATIVE_THREAD_ENV)
    secret_names = set(env.get("FLASH_SECRET_ENV_KEYS", "").split(","))
    assert not (secret_names & set(GRPO_NATIVE_THREAD_ENV))


def test_grpo_child_env_reasserts_native_thread_policy_last(monkeypatch):
    from flash.core.grpo import GRPO_NATIVE_THREAD_ENV
    from flash.engine.worker import rl_train

    hostile = dict.fromkeys(GRPO_NATIVE_THREAD_ENV, "999")
    monkeypatch.setattr(rl_train, "_build_verl_child_env", lambda **_kwargs: dict(hostile))
    env = rl_train._build_rl_child_env(
        {"multi_turn": False},
        {
            "shim_dir": "/tmp/shim",
            "plugin_config_path": "/tmp/plugin.json",
            "rank_device_claims": "/tmp/claims",
        },
        [],
        "http://127.0.0.1:1",
    )
    assert {key: env[key] for key in GRPO_NATIVE_THREAD_ENV} == dict(GRPO_NATIVE_THREAD_ENV)


def test_sft_worker_env_does_not_gain_grpo_native_policy():
    from flash.core.grpo import GRPO_NATIVE_THREAD_ENV
    from flash.providers._lifecycle.worker import build_worker_env

    spec = JobSpec(
        algorithm="sft",
        environment=EnvironmentSpec(id="owner/project/env"),
    )
    env = build_worker_env(spec, spec.seed)
    assert not (set(env) & set(GRPO_NATIVE_THREAD_ENV))


def test_training_guide_describes_authored_admission_and_retained_execution_clamp():
    from pathlib import Path

    guide = (Path(__file__).parents[1] / "flash/cli/scaffold/TRAINING.md").read_text()
    assert "preserves the authored positive integer in the job spec and admission checks" in guide
    assert "effective value clamps to the number of retained valid prompts" in guide
    assert "`group_size` is never changed" in guide
    assert "A later retained-data clamp cannot rescue an oversized authored shape" in guide


def test_worker_option_resolver_rejects_before_dataset_loading(monkeypatch):
    from flash.engine.worker.train.rl import inputs

    train = SimpleNamespace(
        structured_outputs="",
        stop_sequences=(),
        credit_assignment="per_episode",
        prompts_per_step=65,
        learning_rate=None,
        lora_rank=32,
        lora_alpha=64,
    )
    monkeypatch.setattr(inputs._w, "grpo_overrides", lambda: {"group_size": 8})
    with pytest.raises(ValueError, match=r"must be <= 512"):
        inputs._resolve_grpo_options(train, RECIPE.rl, False)

    source = __import__("inspect").getsource(inputs._resolve_grpo_inputs)
    assert source.index("_resolve_grpo_options") < source.index("_load_training_records")

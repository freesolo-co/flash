from __future__ import annotations

import math
import re
from dataclasses import asdict

import pytest

import flash.engine.worker.train.entry.rl_train_runner as rl_train_runner
import flash.runner.accounting.costs as runner_costs
import flash.runner.lifecycle.state as runner_state
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


def _unsupported_group_message() -> str:
    """the rejection message as a regex, derived from the contract rather than spelled out.

    hardcoding the set here means widening `SUPPORTED_GRPO_GROUP_SIZES` leaves a test asserting the
    old wording, which passes only while the tuple never changes.
    """
    allowed = ", ".join(str(value) for value in SUPPORTED_GRPO_GROUP_SIZES)
    return re.escape(f"must be one of {{{allowed}}}")


def _public_grpo_train(**train):
    return {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "environment": {"id": "owner/project/env"},
        "train": {"epochs": 1, **train},
    }


def _internal_grpo_train(**train):
    return {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "environment": {"id": "owner/project/env"},
        "train": {"epochs": 1, **train},
    }


@pytest.mark.parametrize("group_size", SUPPORTED_GRPO_GROUP_SIZES)
def test_grpo_rollout_contract_accepts_supported_groups(group_size):
    shape = resolve_grpo_rollout_shape(1, group_size)
    assert (shape.prompts_per_step, shape.group_size) == (1, group_size)


@pytest.mark.parametrize("group_size", [0, 1, 3, 5, 6, 7, 9, 15, 17, 32, 1024])
def test_grpo_rollout_contract_rejects_every_other_group(group_size):
    with pytest.raises(ValueError, match="must be one of"):
        resolve_grpo_rollout_shape(1, group_size)


def test_group_size_16_is_authorable_and_bounded_by_the_completion_ceiling():
    """16 is a supported group, and widening the set did not widen the concurrency bound.

    Named rather than left to the parametrized cases above so narrowing the tuple fails loudly
    here instead of silently shrinking what those cases cover. `prompts_per_step` is what has to
    absorb the wider group: the ceiling bounds `prompts * group`, so 16 buys a larger group at
    proportionally fewer prompts, never more total completions in a step.
    """
    assert 16 in SUPPORTED_GRPO_GROUP_SIZES
    assert resolve_grpo_rollout_shape(8, 16).completions_per_step == 128
    assert resolve_grpo_rollout_shape(32, 16).completions_per_step == MAX_GRPO_COMPLETIONS_PER_STEP
    with pytest.raises(ValueError, match=r"must be <= 512"):
        resolve_grpo_rollout_shape(33, 16)


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


@pytest.mark.parametrize("group_size", [32, 3, 0, 10**9])
def test_public_parser_rejects_unsupported_group_before_gpu_validation(monkeypatch, group_size):
    import flash.schema as schema

    calls = []
    monkeypatch.setattr(
        schema,
        "_validate_gpu_section",
        lambda *_args, **_kwargs: calls.append("gpu") or pytest.fail("gpu validation ran"),
    )
    with pytest.raises(ConfigError, match=_unsupported_group_message()):
        spec_from_dict(_public_grpo_train(prompts_per_step=4, group_size=group_size))
    assert calls == []


def test_public_parser_rejects_completion_ceiling_before_gpu_validation(monkeypatch):
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


def test_persisted_decode_reads_back_shapes_an_older_flash_accepted():
    """The persisted decoder must not apply the authored-shape admission rule.

    `spec_persistence` states the split: authored config goes through the STRICT
    `schema.spec_from_dict`, while `JobSpec.from_dict` reads back immutable history that an older
    Flash already wrote and never rewrites. The supported-group and completion-ceiling rules bound
    what may be newly SUBMITTED; applying them on read makes a run recorded under the old schema
    (every value >= 2, no completion ceiling) undecodable, and status, recovery, retry, deploy and
    teardown all reconstruct existing runs through this decoder -- so an upgrade would strand them
    with their endpoints still billing.
    """
    legacy_group = JobSpec.from_dict(_internal_grpo_train(prompts_per_step=64, group_size=32))
    assert legacy_group.train.group_size == 32
    assert legacy_group.train.prompts_per_step == 64

    # and a historical run over today's completion ceiling reads back at its recorded shape.
    legacy_width = JobSpec.from_dict(_internal_grpo_train(prompts_per_step=128, group_size=8))
    assert legacy_width.train.prompts_per_step * legacy_width.train.group_size == 1024


def test_authoring_still_rejects_the_shapes_persistence_reads_back():
    """Relaxing the decoder must not relax what a user can newly submit."""
    for train in (
        {"prompts_per_step": 64, "group_size": 32},
        {"prompts_per_step": 128, "group_size": 8},
    ):
        with pytest.raises(ConfigError):
            spec_from_dict(_public_grpo_train(**train))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("prompts_per_step", True),
        ("prompts_per_step", "64"),
        ("group_size", False),
        ("group_size", "8"),
    ],
)
def test_persisted_grpo_decode_still_rejects_non_integer_shapes(key, value):
    """Tolerating a historical VALUE is not tolerating a malformed TYPE.

    A bool or string here was never written by any Flash; it is a corrupt record, and decoding it
    would put a non-integer into arithmetic the runner does on these fields.
    """
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


def test_runconfig_normalizes_defaults_and_preserves_persisted_grpo_shapes():
    config = RunConfig("Qwen/Qwen3.5-9B", "grpo", 1)
    normalized = config.normalized()
    assert normalized.batch_size == DEFAULT_GRPO_PROMPTS_PER_STEP
    assert normalized.group_size == DEFAULT_GRPO_GROUP_SIZE

    for prompts_per_step, group_size in ((4, 3), (65, 8)):
        legacy = RunConfig(
            "Qwen/Qwen3.5-9B",
            "grpo",
            10,
            batch_size=prompts_per_step,
            group_size=group_size,
        ).normalized()
        assert (legacy.batch_size, legacy.group_size) == (prompts_per_step, group_size)


@pytest.mark.parametrize(("prompts_per_step", "group_size"), [(4, 3), (65, 8)])
def test_persisted_legacy_shapes_price_finite_nonzero(prompts_per_step, group_size):
    spec = JobSpec.from_dict(
        _internal_grpo_train(
            max_steps=10,
            prompts_per_step=prompts_per_step,
            group_size=group_size,
        )
    )

    full = runner_costs.charge_usd_for_spec(spec, fallback=float("nan"))
    partial = runner_costs.charge_usd_for_spec(spec, steps=5, fallback=float("nan"))
    assert math.isfinite(full)
    assert full > 0
    assert math.isfinite(partial)
    assert 0 < partial < full


def test_cancellation_billing_does_not_zero_a_completed_legacy_run():
    spec = JobSpec.from_dict(_internal_grpo_train(max_steps=10, prompts_per_step=4, group_size=3))
    status = runner_state.RunStatus(
        run_id="legacy-cancel",
        state="cancelled",
        spec={},
        estimated_cost_usd=8.0,
    )

    charge = runner_costs.cancelled_charge_usd(status, spec, steps=5)
    assert math.isfinite(charge)
    assert 0 < charge < status.estimated_cost_usd


def test_supported_grpo_shape_pricing_is_numerically_unchanged():
    from flash.cost import estimate_cost

    expected = {
        2: 0.2334652531216931,
        4: 0.2734888395767196,
        8: 0.35353601248677247,
        16: 0.5136303583068783,
    }

    actual = {
        group_size: estimate_cost(
            RunConfig(
                "Qwen/Qwen3.5-9B",
                "grpo",
                10,
                batch_size=4,
                group_size=group_size,
                gpu_type="A100 PCIe",
            )
        ).total_usd
        for group_size in SUPPORTED_GRPO_GROUP_SIZES
    }
    assert actual == expected


@pytest.mark.parametrize(
    ("prompts_per_step", "group_size", "message"),
    [
        (4, 3, _unsupported_group_message()),
        (4, 32, _unsupported_group_message()),
        (65, 8, r"must be <= 512"),
    ],
)
def test_authoring_still_rejects_every_legacy_shape(prompts_per_step, group_size, message):
    with pytest.raises(ConfigError, match=message):
        spec_from_dict(
            _public_grpo_train(
                prompts_per_step=prompts_per_step,
                group_size=group_size,
            )
        )


@pytest.mark.parametrize("group_size", SUPPORTED_GRPO_GROUP_SIZES)
def test_supported_authored_groups_reach_allocation(group_size):
    import flash.providers.core.allocator as allocator

    spec = spec_from_dict(
        _public_grpo_train(prompts_per_step=4, group_size=group_size),
        run_id=f"supported-{group_size}",
    )
    allocation = allocator.allocate(
        spec.model,
        spec.algorithm,
        train=spec.train,
        providers=("runpod",),
        gpu_type="H100",
    )

    assert spec.train.group_size == group_size
    assert allocation.provider == "runpod"
    assert allocation.gpu == "H100"


@pytest.mark.parametrize(
    ("prompts_per_step", "group_size"),
    [
        (4, 3),
        (65, 8),
        (64, 32),
    ],
)
def test_persisted_legacy_shapes_reach_allocation(prompts_per_step, group_size):
    import flash.providers.core.allocator as allocator

    spec = JobSpec.from_dict(
        _internal_grpo_train(
            prompts_per_step=prompts_per_step,
            group_size=group_size,
            max_context_tokens=4096,
            max_completion_tokens=2048,
        )
    )
    expected_vram = allocator.required_vram_gb(spec.model, spec.algorithm, train=spec.train)
    allocation = allocator.allocate(
        spec.model,
        spec.algorithm,
        train=spec.train,
        providers=("runpod",),
        gpu_type="H100",
        max_gpu_count=2,
    )

    assert allocation.provider == "runpod"
    assert allocation.gpu == "H100"
    assert allocation.min_vram_gb == expected_vram
    if group_size == 32:
        default_vram = allocator.required_vram_gb(
            spec.model,
            spec.algorithm,
            train={**asdict(spec.train), "group_size": DEFAULT_GRPO_GROUP_SIZE},
        )
        assert expected_vram > default_vram


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
    from flash.providers._lifecycle.net.worker import build_worker_env

    spec = JobSpec(
        algorithm="grpo",
        environment=EnvironmentSpec(id="owner/project/env", secrets=tuple(GRPO_NATIVE_THREAD_ENV)),
        train=TrainSpec(prompts_per_step=64, group_size=8),
    )
    hostile = dict.fromkeys(GRPO_NATIVE_THREAD_ENV, "999")
    env = build_worker_env(spec, runtime_secrets=hostile)
    assert {key: env[key] for key in GRPO_NATIVE_THREAD_ENV} == dict(GRPO_NATIVE_THREAD_ENV)
    secret_names = set(env.get("FLASH_SECRET_ENV_KEYS", "").split(","))
    assert not (secret_names & set(GRPO_NATIVE_THREAD_ENV))


def test_grpo_child_env_reasserts_native_thread_policy_last(monkeypatch):
    from flash.core.grpo import GRPO_NATIVE_THREAD_ENV

    hostile = dict.fromkeys(GRPO_NATIVE_THREAD_ENV, "999")
    monkeypatch.setattr(rl_train_runner, "_build_verl_child_env", lambda **_kwargs: dict(hostile))
    env = rl_train_runner._build_rl_child_env(
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
    from flash.providers._lifecycle.net.worker import build_worker_env

    spec = JobSpec(
        algorithm="sft",
        environment=EnvironmentSpec(id="owner/project/env"),
    )
    env = build_worker_env(spec)
    assert not (set(env) & set(GRPO_NATIVE_THREAD_ENV))


def test_training_guide_describes_authored_admission_and_retained_execution_clamp():
    from pathlib import Path

    guide = (Path(__file__).parents[1] / "flash/cli/scaffold/TRAINING.md").read_text()
    assert "preserves the authored positive integer in the job spec and admission checks" in guide
    assert "effective value clamps to the number of retained valid prompts" in guide
    assert "`group_size` is never changed" in guide
    assert "A later retained-data clamp cannot rescue an oversized authored shape" in guide


@pytest.mark.parametrize(
    ("prompts_per_step", "group_size"),
    [
        (4, 3),
        (65, 8),
    ],
)
def test_persisted_legacy_shapes_reach_worker_option_resolution(
    monkeypatch, prompts_per_step, group_size
):
    from flash.engine.worker.train.rl.launch import inputs

    spec = JobSpec.from_dict(
        _internal_grpo_train(prompts_per_step=prompts_per_step, group_size=group_size)
    )
    monkeypatch.setattr(
        inputs._worker_config,
        "grpo_overrides",
        lambda: {"group_size": spec.train.group_size},
    )

    options = inputs._resolve_grpo_options(spec.train, RECIPE.rl, False)

    assert (options["prompts_per_step"], options["group_size"]) == (
        prompts_per_step,
        group_size,
    )


@pytest.mark.parametrize("group_size", SUPPORTED_GRPO_GROUP_SIZES)
def test_supported_shapes_reach_worker_option_resolution_exactly(monkeypatch, group_size):
    from flash.engine.worker.train.rl.launch import inputs

    spec = spec_from_dict(_public_grpo_train(prompts_per_step=4, group_size=group_size))
    monkeypatch.setattr(
        inputs._worker_config,
        "grpo_overrides",
        lambda: {"group_size": spec.train.group_size},
    )

    options = inputs._resolve_grpo_options(spec.train, RECIPE.rl, False)

    assert (options["prompts_per_step"], options["group_size"]) == (4, group_size)

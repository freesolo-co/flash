"""LoRA+ retirement: plain-LoRA clients work and old optimizer state is never reinterpreted."""

import pytest

from flash.core.spec import JobSpec, TrainSpec, require_supported_sft_optimizer
from flash.schema import (
    TRAIN_SCHEMA_KEYS,
    ConfigError,
    spec_and_train_keys_from_file,
    spec_from_dict,
    train_schema_metadata,
)


def _raw(train, algorithm="sft"):
    return {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": algorithm,
        "project": "11111111-1111-4111-8111-111111111111",
        "environment": {"id": "owner/project/env"},
        "train": train,
    }


@pytest.mark.parametrize("train", [{}, {"loraplus_ratio": 1}, {"loraplus_ratio": 1.0}])
def test_plain_lora_round_trips_and_preserves_old_client_compatibility(train):
    spec = spec_from_dict(_raw({**train, "learning_rate": 1e-5}))
    for result in (spec, spec_from_dict(spec.to_dict()), JobSpec.from_json(spec.to_json())):
        require_supported_sft_optimizer(result)
        assert result.train.loraplus_ratio == 1.0
        assert result.train.learning_rate == 1e-5
    assert "loraplus_ratio" not in TRAIN_SCHEMA_KEYS
    assert "loraplus_ratio" not in train_schema_metadata()


@pytest.mark.parametrize("ratio", [None, 3.5, 16, 32])
@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_public_parser_rejects_retired_optimizer_selection(ratio, algorithm):
    with pytest.raises(ConfigError, match="LoRA\\+ has been removed"):
        spec_from_dict(_raw({"loraplus_ratio": ratio}, algorithm))


@pytest.mark.parametrize("ratio", [None, 1.0, 3.5, 16.0])
def test_historical_records_remain_readable_without_changing_serialized_shape(ratio):
    train = {} if ratio is None else {"loraplus_ratio": ratio}
    spec = JobSpec.from_dict(_raw(train))
    assert spec.train.loraplus_ratio == ratio
    for serialized in (spec.to_dict(), spec.to_internal_dict()):
        if ratio is None:
            assert "loraplus_ratio" not in serialized["train"]
        else:
            assert serialized["train"]["loraplus_ratio"] == ratio
        restored = JobSpec.from_dict(serialized)
        assert restored.to_dict() == spec.to_dict()
        if ratio == 1.0:
            require_supported_sft_optimizer(restored)
        else:
            with pytest.raises(ValueError, match="original Flash version"):
                require_supported_sft_optimizer(restored)


@pytest.mark.parametrize(
    "ratio",
    [
        True,
        False,
        "16",
        [],
        0,
        -1,
        0.5,
        float("nan"),
        float("inf"),
        -float("inf"),
        10**400,
        -(10**400),
    ],
)
def test_invalid_legacy_ratio_is_rejected_at_public_and_internal_boundaries(ratio):
    with pytest.raises(ConfigError, match="loraplus_ratio"):
        spec_from_dict(_raw({"loraplus_ratio": ratio}))
    with pytest.raises((TypeError, ValueError), match="loraplus_ratio"):
        JobSpec.from_dict(_raw({"loraplus_ratio": ratio}))
    with pytest.raises((TypeError, ValueError), match="loraplus_ratio"):
        TrainSpec(loraplus_ratio=ratio)


@pytest.mark.parametrize("ratio", [None, 3.5, 16.0])
def test_legacy_submission_fails_before_preparation_side_effects(monkeypatch, ratio):
    from flash.runner.lifecycle import preparation
    from flash.runner.lifecycle.submit import prepare_job

    monkeypatch.setattr(
        preparation,
        "_inherit_warmstart_revision",
        lambda *_a, **_kw: pytest.fail("retired optimizer must fail before preparation"),
    )
    spec = JobSpec.from_dict(_raw({} if ratio is None else {"loraplus_ratio": ratio}))
    with pytest.raises(ValueError, match="LoRA\\+ training has been removed"):
        prepare_job(spec)


@pytest.mark.parametrize("ratio", [None, 16.0])
def test_prepared_legacy_submission_cannot_bypass_retirement(ratio):
    from types import SimpleNamespace

    from flash.runner.lifecycle.submit import submit_job

    spec = JobSpec.from_dict(_raw({} if ratio is None else {"loraplus_ratio": ratio}))
    prepared = SimpleNamespace(public_spec=spec, worker_spec=spec)
    with pytest.raises(ValueError, match="LoRA\\+ training has been removed"):
        submit_job(spec, prepared_job=prepared)


def test_toml_and_cli_override_cannot_reenable_loraplus(tmp_path):
    config = tmp_path / "flash.toml"
    config.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'algorithm = "sft"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        '[environment]\nid = "owner/project/env"\n'
        "[train]\nlearning_rate = 0.00001\n"
    )
    spec, keys = spec_and_train_keys_from_file(str(config), run_id="test")
    require_supported_sft_optimizer(spec)
    assert "loraplus_ratio" not in keys
    compatible, keys = spec_and_train_keys_from_file(
        str(config), run_id="test", overrides=["train.loraplus_ratio=1.0"]
    )
    require_supported_sft_optimizer(compatible)
    assert keys <= TRAIN_SCHEMA_KEYS
    assert "loraplus_ratio" not in keys
    with pytest.raises(ConfigError, match="LoRA\\+ has been removed"):
        spec_and_train_keys_from_file(
            str(config), run_id="test", overrides=["train.loraplus_ratio=16.0"]
        )
    config.write_text(config.read_text() + "loraplus_ratio = 16.0\n")
    with pytest.raises(ConfigError, match="LoRA\\+ has been removed"):
        spec_and_train_keys_from_file(str(config), run_id="test")


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_new_programmatic_specs_use_supported_optimizer_without_a_ratio_knob(algorithm):
    spec = JobSpec(algorithm=algorithm, train=TrainSpec())
    require_supported_sft_optimizer(spec)
    payload = spec.to_internal_dict()
    if algorithm != "sft":
        assert "loraplus_ratio" not in payload["train"]
    require_supported_sft_optimizer(JobSpec.from_dict(payload))

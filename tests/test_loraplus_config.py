"""user-authored lora+ ratios survive cli and worker spec boundaries."""

import pytest

from flash.core.spec import JobSpec, TrainSpec
from flash.schema import ConfigError, spec_and_train_keys_from_file, spec_from_dict


def _raw(train, algorithm="sft"):
    return {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": algorithm,
        "project": "11111111-1111-4111-8111-111111111111",
        "environment": {"id": "owner/project/env"},
        "train": train,
    }


@pytest.mark.parametrize("ratio", [1, 3.5, 16, 32])
def test_loraplus_ratio_survives_public_and_worker_round_trips(ratio):
    spec = spec_from_dict(_raw({"loraplus_ratio": ratio, "learning_rate": 1e-5}))
    public = spec_from_dict(spec.to_dict())
    worker = JobSpec.from_json(spec.to_json())
    for result in (spec, public, worker):
        assert result.train.loraplus_ratio == ratio
        assert result.train.learning_rate == 1e-5


@pytest.mark.parametrize("train", [{}, {"loraplus_ratio": None}])
def test_new_unset_ratio_is_frozen_for_future_retries(train):
    spec = spec_from_dict(_raw(train))
    assert spec.train.loraplus_ratio == 1.0
    assert spec.to_dict()["train"]["loraplus_ratio"] == 1.0
    assert JobSpec.from_json(spec.to_json()).train.loraplus_ratio == 1.0


def test_legacy_unset_ratio_preserves_historical_payload_shape():
    spec = JobSpec.from_dict(_raw({}))
    assert spec.train.loraplus_ratio is None
    assert "loraplus_ratio" not in spec.to_dict()["train"]
    assert "loraplus_ratio" not in spec.to_internal_dict()["train"]
    assert JobSpec.from_json(spec.to_json()).train.loraplus_ratio is None


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
def test_invalid_ratio_is_rejected_at_public_and_internal_boundaries(ratio):
    with pytest.raises(ConfigError, match="loraplus_ratio"):
        spec_from_dict(_raw({"loraplus_ratio": ratio}))
    with pytest.raises((TypeError, ValueError), match="loraplus_ratio"):
        JobSpec.from_dict(_raw({"loraplus_ratio": ratio}))
    with pytest.raises((TypeError, ValueError), match="loraplus_ratio"):
        TrainSpec(loraplus_ratio=ratio)


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_ratio_is_rejected_for_algorithms_without_loraplus(algorithm):
    raw = _raw({"loraplus_ratio": 4}, algorithm)
    with pytest.raises(ConfigError, match="only applies to sft"):
        spec_from_dict(raw)
    with pytest.raises(ValueError, match="only applies to sft"):
        JobSpec.from_dict(raw)


def test_toml_ratio_and_cli_override_are_user_authored_keys(tmp_path):
    config = tmp_path / "flash.toml"
    config.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'algorithm = "sft"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        '[environment]\nid = "owner/project/env"\n'
        "[train]\nloraplus_ratio = 4.0\nlearning_rate = 0.00001\n"
    )
    spec, keys = spec_and_train_keys_from_file(str(config), run_id="test")
    assert spec.train.loraplus_ratio == 4.0
    assert "loraplus_ratio" in keys
    spec, keys = spec_and_train_keys_from_file(
        str(config), run_id="test", overrides=["train.loraplus_ratio=1.0"]
    )
    assert spec.train.loraplus_ratio == 1.0
    assert spec.train.learning_rate == 1e-5
    assert "loraplus_ratio" in keys

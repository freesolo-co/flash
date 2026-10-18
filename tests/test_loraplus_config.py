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
def test_legacy_loraplus_ratio_survives_worker_round_trips(ratio):
    spec = JobSpec.from_dict(_raw({"loraplus_ratio": ratio, "learning_rate": 1e-5}))
    worker = JobSpec.from_json(spec.to_json())
    assert worker.train.loraplus_ratio == ratio
    assert worker.train.learning_rate == 1e-5


@pytest.mark.parametrize("ratio", [3.5, 16, 32])
def test_new_nondefault_loraplus_ratio_is_retired(ratio):
    with pytest.raises(ConfigError, match="loraplus_ratio is retired"):
        spec_from_dict(_raw({"loraplus_ratio": ratio}))


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
    with pytest.raises(ConfigError, match="loraplus_ratio is retired"):
        spec_and_train_keys_from_file(str(config), run_id="test")
    spec, _ = spec_and_train_keys_from_file(
        str(config), run_id="test", overrides=["train.loraplus_ratio=1.0"]
    )
    assert spec.train.loraplus_ratio == 1.0


@pytest.mark.parametrize("quantile", [0.0, 0.5, 1.0])
def test_entropy_filter_is_retired_for_new_runs_but_preserved_for_legacy_runs(quantile):
    raw = _raw({"entropy_quantile": quantile}, "grpo")
    with pytest.raises(ConfigError, match="entropy_quantile is retired"):
        spec_from_dict(raw)
    spec = JobSpec.from_dict(raw)
    assert JobSpec.from_json(spec.to_json()).train.entropy_quantile == quantile

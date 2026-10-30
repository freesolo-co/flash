"""Removed experiments have no authored or persisted compatibility surface."""

from dataclasses import fields

import pytest

from flash.core.spec import JobSpec, TrainSpec
from flash.schema import (
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


@pytest.mark.parametrize("key", ["loraplus_ratio", "entropy_quantile"])
@pytest.mark.parametrize("value", [None, 0.0, 0.5, 1.0, 16.0])
@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
def test_removed_fields_are_rejected_at_public_and_persisted_boundaries(key, value, algorithm):
    raw = _raw({key: value}, algorithm)
    with pytest.raises(ConfigError, match=key):
        spec_from_dict(raw)
    with pytest.raises(ValueError, match=key):
        JobSpec.from_dict(raw)
    with pytest.raises(TypeError, match=key):
        TrainSpec(**{key: value})


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
def test_current_specs_round_trip_without_removed_settings(algorithm):
    spec = spec_from_dict(_raw({}, algorithm))
    restored = JobSpec.from_json(spec.to_json())
    assert restored == spec
    removed = {"loraplus_ratio", "entropy_quantile"}
    assert removed.isdisjoint(spec.to_dict()["train"])
    assert removed.isdisjoint(spec.to_internal_dict()["train"])
    assert removed.isdisjoint(train_schema_metadata())
    assert removed.isdisjoint(field.name for field in fields(TrainSpec))


@pytest.mark.parametrize("key", ["loraplus_ratio", "entropy_quantile"])
def test_removed_toml_and_cli_overrides_are_rejected(tmp_path, key):
    config = tmp_path / "flash.toml"
    config.write_text(
        'model = "Qwen/Qwen3.5-9B"\nalgorithm = "sft"\n[environment]\nid = "owner/project/env"\n'
    )
    with pytest.raises(ConfigError, match=key):
        spec_and_train_keys_from_file(str(config), overrides=[f"train.{key}=1.0"])
    with config.open("a") as stream:
        stream.write(f"[train]\n{key}=1.0\n")
    with pytest.raises(ConfigError, match=key):
        spec_and_train_keys_from_file(str(config))

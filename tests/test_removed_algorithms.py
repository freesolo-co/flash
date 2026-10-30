"""Removed training algorithms cannot enter configuration or persisted-job execution."""

import pytest

from flash.core.catalog import normalize_algorithm, public_model_rows
from flash.core.spec import JobSpec
from flash.cost.types import RunConfig
from flash.schema import ConfigError, parse_adapter_storage_ref, spec_from_dict


@pytest.mark.parametrize("algorithm", ["opd", "OPD", "ppo"])
def test_removed_algorithm_rejected_at_all_spec_boundaries(algorithm):
    with pytest.raises(ValueError, match="unsupported algorithm"):
        normalize_algorithm(algorithm)
    with pytest.raises(ConfigError, match="unsupported algorithm"):
        spec_from_dict({"model": "Qwen/Qwen3.5-9B", "algorithm": algorithm})
    with pytest.raises(ValueError, match="unsupported algorithm"):
        JobSpec.from_dict({"algorithm": algorithm})
    with pytest.raises(ValueError, match="unsupported algorithm"):
        RunConfig(model_id="Qwen/Qwen3.5-9B", method=algorithm, steps=1)


def test_removed_algorithm_has_no_model_or_checkpoint_capability():
    assert all(set(row["algos"]) <= {"sft", "grpo", "custom"} for row in public_model_rows())
    assert parse_adapter_storage_ref("owner/repo:opd/old-run") is None


def test_removed_managed_teacher_setting_is_not_accepted():
    with pytest.raises(ConfigError, match="teacher_model"):
        spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "environment": {"id": "org/project/env"},
                "train": {"teacher_model": "glm-5.3"},
            }
        )

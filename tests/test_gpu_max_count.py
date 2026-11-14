"""`[gpu] max_count` is an authored card ceiling enforced at submit and at allocation."""

import json

import pytest

from flash.core.spec import GpuSpec, JobSpec
from flash.schema import ConfigError, spec_from_dict

RAW = {
    "model": "Qwen/Qwen3.5-9B-Base",
    "algorithm": "grpo",
    "project": "11111111-1111-4111-8111-111111111111",
    "environment": {"id": "freesolo/math-agent/gsm8k"},
    "train": {"epochs": 1, "max_examples": 10, "group_size": 8, "max_context_tokens": 24576},
    "gpu": {"spot": True},
}


def _raw(gpu=None, **train):
    raw = json.loads(json.dumps(RAW))
    raw["gpu"].update(gpu or {})
    raw["train"].update(train)
    return raw


def test_unset_ceiling_auto_sizes_and_keeps_serialized_specs_unchanged():
    spec = spec_from_dict(_raw())
    assert spec.gpu.max_count == 0
    assert "max_count" not in spec.to_internal_dict()["gpu"]
    assert "max_count" not in spec.to_dict()["gpu"]
    assert spec.authored_gpu_count is None


def test_ceiling_round_trips_and_caps_allocation():
    spec = spec_from_dict(_raw({"max_count": 1}))
    assert spec.gpu.max_count == 1
    assert spec.authored_gpu_count == 1
    assert spec.to_dict()["gpu"]["max_count"] == 1
    restored = JobSpec.from_dict(spec.to_internal_dict())
    assert restored.gpu.max_count == 1
    assert restored.authored_gpu_count == 1
    assert spec_from_dict({**spec.to_dict()}).gpu.max_count == 1


@pytest.mark.parametrize("value", [0, 9, -1, True, 1.0, "1"])
def test_invalid_ceilings_are_rejected(value):
    with pytest.raises(ConfigError, match="max_count"):
        spec_from_dict(_raw({"max_count": value}))
    if type(value) is int and not 0 <= value <= 8:
        with pytest.raises(ValueError, match="max_count"):
            GpuSpec(max_count=value)


def test_job_that_fits_one_card_is_accepted_with_a_one_card_ceiling():
    spec = spec_from_dict(_raw({"max_count": 1}, max_context_tokens=12288))
    assert spec.authored_gpu_count == 1


def test_job_too_large_for_one_card_is_rejected_at_submit():
    large = {"max_context_tokens": 131072, "group_size": 16, "prompts_per_step": 8}
    # the same job is valid without a ceiling (Flash may rent several cards)
    assert spec_from_dict(_raw(**large)).authored_gpu_count is None
    with pytest.raises(ConfigError):
        spec_from_dict(_raw({"max_count": 1}, **large))

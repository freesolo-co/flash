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
    assert spec.gpu_count_ceiling is None


def test_ceiling_round_trips_and_caps_allocation():
    spec = spec_from_dict(_raw({"max_count": 1}))
    assert spec.gpu.max_count == 1
    assert spec.gpu_count_ceiling == 1
    assert spec.to_dict()["gpu"]["max_count"] == 1
    restored = JobSpec.from_dict(spec.to_internal_dict())
    assert restored.gpu.max_count == 1
    assert restored.gpu_count_ceiling == 1
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
    assert spec.gpu_count_ceiling == 1


def test_job_too_large_for_one_card_is_rejected_at_submit():
    large = {"max_context_tokens": 131072, "group_size": 16, "prompts_per_step": 8}
    # the same job is valid without a ceiling (Flash may rent several cards)
    assert spec_from_dict(_raw(**large)).authored_gpu_count is None
    with pytest.raises(ConfigError):
        spec_from_dict(_raw({"max_count": 1}, **large))


def test_auto_sized_width_within_the_ceiling_passes_the_preparation_check():
    """A ceiling of 2 admits a 2-card allocation; the public count stays the placeholder 1."""
    from dataclasses import replace

    from flash.runner.lifecycle.preparation import _validate_effective_spec

    public = spec_from_dict(_raw({"max_count": 2}, max_context_tokens=8192))
    reparsed_public = JobSpec.from_dict(public.to_dict())
    assert reparsed_public.gpu.count == 1
    _validate_effective_spec(reparsed_public, replace(public, gpu=replace(public.gpu, count=2)))
    with pytest.raises(ValueError, match="does not match the public run"):
        _validate_effective_spec(reparsed_public, replace(public, gpu=replace(public.gpu, count=3)))


def test_allocation_width_ceiling_is_separate_from_the_authored_pin():
    spec = spec_from_dict(_raw({"max_count": 2}, max_context_tokens=8192))
    assert spec.gpu_count_ceiling == 2
    assert spec.authored_gpu_count is None  # wire/provenance meaning unchanged
    assert spec_from_dict(_raw()).gpu_count_ceiling is None


def test_custom_ceiling_is_clipped_to_the_custom_spot_placement():
    raw = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "custom",
        "project": "11111111-1111-4111-8111-111111111111",
        "custom": {"file": "algorithm.py"},
        "train": {"max_steps": 6},
        "gpu": {"spot": True, "max_count": 2},
    }
    spec = spec_from_dict(raw)
    assert spec.gpu.max_count == 1  # custom Spot runs place on exactly one card
    assert spec.gpu_count_ceiling == 1


def test_unresolved_cost_quote_honours_the_ceiling():
    from flash.cost.spec import run_config_from_spec

    bounded = spec_from_dict(_raw({"max_count": 1}, max_context_tokens=12288))
    assert run_config_from_spec(bounded).gpu_count == 1
    assert run_config_from_spec(spec_from_dict(_raw())).gpu_count is None

"""OPD/DPO spec plumbing (CPU-only)."""

from __future__ import annotations

import pytest

from autoslm.config_schema import ConfigError, spec_from_dict
from autoslm.worker_spec import ALGORITHMS, JobSpec


def test_algorithms_registry():
    assert set(ALGORITHMS) == {"sft", "grpo", "opd", "dpo"}


def test_opd_spec_roundtrip():
    raw = {
        "model": "Qwen/Qwen3-0.6B",
        "algorithm": "opd",
        "distill": {"teacher": "Qwen/Qwen3-4B-Instruct-2507", "lmbda": 0.7},
        "train": {"steps": 10},
    }
    spec = spec_from_dict(raw, run_id="x")
    assert spec.algorithm == "opd"
    assert spec.phase == "opd"
    assert spec.distill.teacher == "Qwen/Qwen3-4B-Instruct-2507"
    assert spec.distill.lmbda == pytest.approx(0.7)
    spec2 = JobSpec.from_dict(spec.to_dict())
    assert spec2.distill.teacher == spec.distill.teacher


def test_opd_requires_teacher():
    raw = {"model": "Qwen/Qwen3-0.6B", "algorithm": "opd", "train": {"steps": 10}}
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(raw, run_id="x")
    assert "teacher" in str(ei.value)


def test_dpo_spec():
    raw = {
        "model": "Qwen/Qwen3-0.6B",
        "algorithm": "dpo",
        "environment": {"id": "gsm8k", "params": {"preference_dataset": "org/prefs"}},
        "train": {"epochs": 1},
    }
    spec = spec_from_dict(raw, run_id="x")
    assert spec.algorithm == "dpo"
    assert spec.environment.params["preference_dataset"] == "org/prefs"


def test_unknown_algorithm_rejected():
    with pytest.raises(ConfigError):
        spec_from_dict({"model": "Qwen/Qwen3-0.6B", "algorithm": "ppo"}, run_id="x")


def test_grpo_capability_still_enforced():
    # Qwen3.5-9B is SFT-class only; opd/dpo are allowed (trainer-only) but grpo is not.
    raw = {"model": "Qwen/Qwen3.5-9B", "algorithm": "grpo", "train": {"steps": 1}}
    with pytest.raises(ConfigError):
        spec_from_dict(raw, run_id="x")
    raw_opd = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "opd",
        "distill": {"teacher": "Qwen/Qwen3-4B-Instruct-2507"},
        "train": {"steps": 1},
    }
    assert spec_from_dict(raw_opd, run_id="x").algorithm == "opd"

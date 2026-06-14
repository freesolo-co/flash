"""Algorithm spec plumbing (CPU-only)."""

from __future__ import annotations

import pytest

from autoslm.config_schema import ConfigError, spec_from_dict
from autoslm.worker_spec import ALGORITHMS


def test_algorithms_registry():
    assert set(ALGORITHMS) == {"sft", "grpo"}


def test_unknown_algorithm_rejected():
    with pytest.raises(ConfigError):
        spec_from_dict({"model": "Qwen/Qwen3-0.6B", "algorithm": "ppo"}, run_id="x")


def test_grpo_capability_still_enforced():
    # Qwen3.5-9B is SFT-class only; grpo (which needs the colocated engine) is rejected.
    raw = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "environment": {"id": "owner/env"},
        "train": {"steps": 1},
    }
    with pytest.raises(ConfigError):
        spec_from_dict(raw, run_id="x")
    # SFT on the same model is allowed.
    raw_sft = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "sft",
        "environment": {"id": "owner/env"},
        "train": {"epochs": 1},
    }
    assert spec_from_dict(raw_sft, run_id="x").algorithm == "sft"

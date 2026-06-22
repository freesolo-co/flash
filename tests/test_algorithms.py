"""Algorithm spec plumbing (CPU-only)."""

from __future__ import annotations

import pytest

from flash.catalog import ALGORITHMS
from flash.schema import ConfigError, spec_from_dict


def test_algorithms_registry():
    assert set(ALGORITHMS) == {"sft", "grpo"}


def test_unknown_algorithm_rejected():
    with pytest.raises(ConfigError):
        spec_from_dict({"model": "Qwen/Qwen3.5-0.8B", "algorithm": "ppo"}, run_id="x")


def test_grpo_capability_still_enforced():
    # The guardrail: an SFT-only model rejects GRPO through the config path. No catalog
    # entry is SFT-only anymore, so inject a temporary one.
    from flash import catalog
    from flash.catalog import ModelInfo

    catalog.MODELS["test/sft-only"] = ModelInfo(
        id="test/sft-only",
        display_name="sft only",
        params="1B",
        algos=("sft",),
        min_vram_gb=12,
    )
    try:
        with pytest.raises(ConfigError):
            spec_from_dict(
                {
                    "model": "test/sft-only",
                    "algorithm": "grpo",
                    "environment": {"id": "owner/env"},
                    "train": {"steps": 1, "hf_repo": "owner/runs"},
                },
                run_id="x",
            )
        # SFT on the same model is allowed.
        sft = spec_from_dict(
            {
                "model": "test/sft-only",
                "algorithm": "sft",
                "environment": {"id": "owner/env"},
                "train": {"epochs": 1, "hf_repo": "owner/runs"},
            },
            run_id="x",
        )
        assert sft.algorithm == "sft"
    finally:
        catalog.MODELS.pop("test/sft-only", None)


def test_qwen35_9b_now_supports_grpo():
    # Qwen3.5-9B is GRPO-capable now (normal bf16 LoRA; auto-routed to an 80 GB A100).
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "environment": {"id": "owner/env"},
            "train": {"steps": 1, "hf_repo": "owner/runs"},
            "gpu": {"type": "A100 PCIe"},
        },
        run_id="x",
    )
    assert spec.algorithm == "grpo"

"""Algorithm spec plumbing (CPU-only)."""

from __future__ import annotations

import pytest

from flash.core.catalog import ALGORITHMS, normalize_algorithm
from flash.schema import ConfigError, spec_from_dict


def test_algorithms_registry():
    assert set(ALGORITHMS) == {"sft", "grpo", "opd"}


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithm_names_normalize_case_insensitively(algorithm):
    # config values reach the catalog verbatim from user toml, so every supported algorithm must
    # survive an uppercase spelling.
    assert normalize_algorithm(algorithm.upper()) == algorithm


def test_unknown_algorithm_rejected():
    with pytest.raises(ConfigError):
        spec_from_dict({"model": "Qwen/Qwen3.5-9B", "algorithm": "ppo"}, run_id="x")


def test_opd_algorithm_accepted():
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "opd",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 1, "max_examples": 10},
        },
        run_id="x",
    )
    assert spec.algorithm == "opd"
    # phase drives RUN_MODE + the artifact path segment; grpo->rl, everything else->itself.
    assert spec.phase == "opd"
    # the teacher provider key is control-plane-only, not a user-declared environment secret.
    assert "PARASAIL_API_KEY" not in spec.environment.secrets


def test_opd_capability_gated_per_model():
    # A model whose algos lack "opd" rejects a opd run through the config path.
    from flash.core import catalog
    from flash.core.catalog import ModelInfo

    catalog.MODELS["test/no-opd"] = ModelInfo(
        id="test/no-opd",
        display_name="no opd",
        params="1B",
        params_b=1.0,
        algos=("sft", "grpo"),
        min_vram_gb=12,
    )
    try:
        with pytest.raises(ConfigError):
            spec_from_dict(
                {
                    "model": "test/no-opd",
                    "algorithm": "opd",
                    "environment": {"id": "github:owner/repo@main:env/environment.py"},
                    "train": {"epochs": 1, "max_examples": 1},
                },
                run_id="x",
            )
    finally:
        catalog.MODELS.pop("test/no-opd", None)


def test_grpo_capability_still_enforced():
    # The guardrail: an SFT-only model rejects GRPO through the config path. No catalog
    # entry is SFT-only anymore, so inject a temporary one.
    from flash.core import catalog
    from flash.core.catalog import ModelInfo

    catalog.MODELS["test/sft-only"] = ModelInfo(
        id="test/sft-only",
        display_name="sft only",
        params="1B",
        params_b=1.0,
        algos=("sft",),
        min_vram_gb=12,
    )
    try:
        with pytest.raises(ConfigError):
            spec_from_dict(
                {
                    "model": "test/sft-only",
                    "algorithm": "grpo",
                    "environment": {"id": "github:owner/repo@main:env/environment.py"},
                    "train": {"epochs": 1, "max_examples": 1},
                },
                run_id="x",
            )
        # SFT on the same model is allowed.
        sft = spec_from_dict(
            {
                "model": "test/sft-only",
                "algorithm": "sft",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"epochs": 1, "max_examples": 8},
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
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 1, "max_examples": 1},
            "gpu": {"type": "A100 PCIe"},
        },
        run_id="x",
    )
    assert spec.algorithm == "grpo"

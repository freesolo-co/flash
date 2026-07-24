"""Algorithm spec plumbing (CPU-only)."""

from __future__ import annotations

import pytest

from flash.catalog import ALGORITHMS, normalize_algorithm
from flash.schema import ConfigError, spec_from_dict


def test_algorithms_registry():
    assert set(ALGORITHMS) == {"sft", "grpo", "opd", "opsd"}


def test_unknown_algorithm_rejected():
    with pytest.raises(ConfigError):
        spec_from_dict({"model": "Qwen/Qwen3.5-0.8B", "algorithm": "ppo"}, run_id="x")


def test_opd_algorithm_accepted():
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "opd",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 1, "max_examples": 10, "hf_repo": "owner/runs"},
        },
        run_id="x",
    )
    assert spec.algorithm == "opd"
    # phase drives RUN_MODE + the artifact path segment; grpo->rl, everything else->itself.
    assert spec.phase == "opd"
    # The teacher key is platform-managed (injected into the worker env by the control plane, like
    # HF_TOKEN) — NOT a user-declared secret, so it must not be auto-added to environment.secrets.
    assert "FIREWORKS_API_KEY" not in spec.environment.secrets


def test_opsd_algorithm_accepted_and_normalized():
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "OPSD",
            # opsd runs only in reasoning mode; thinking = true is now required by the config gate.
            "thinking": True,
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 1, "max_examples": 10, "hf_repo": "owner/runs"},
        },
        run_id="x",
    )
    assert normalize_algorithm("OPSD") == "opsd"
    assert spec.algorithm == "opsd"
    assert spec.phase == "opsd"


def test_opsd_registration_boundaries_are_on_policy_local_teacher_and_text_only():
    from flash.catalog import samples_on_policy, validate_model_for_algorithm
    from flash.cost.spec import runconfig_from_spec, spec_steps
    from flash.lora_rank import _effective_train_context
    from flash.multimodal import validate_multimodal_training
    from flash.schema import parse_adapter_storage_ref
    from flash.spec import JobSpec, TrainSpec

    spec = JobSpec(
        model="Qwen/Qwen3.6-27B",
        algorithm="opsd",
        train=TrainSpec(
            epochs=2,
            batch_size=2,
            max_examples=4,
            max_context_tokens=128,
            max_completion_tokens=32,
        ),
    )
    assert samples_on_policy("opsd") is True
    assert validate_model_for_algorithm(spec.model, "opsd").id == spec.model
    assert parse_adapter_storage_ref("owner/repo:opsd/run-1") == (
        "owner/repo",
        "opsd/run-1",
    )
    assert spec_steps(spec) == 4
    assert runconfig_from_spec(spec).teacher_model == ""
    assert _effective_train_context(spec) == (
        128,
        "train.max_context_tokens / train.max_completion_tokens (OPSD rollout prompt+completion)",
    )
    with pytest.raises(ValueError, match="does not support image-bearing"):
        validate_multimodal_training(spec.model, "opsd")


def test_opd_capability_gated_per_model():
    # A model whose algos lack "opd" rejects a opd run through the config path.
    from flash import catalog
    from flash.catalog import ModelInfo

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
                    "train": {"epochs": 1, "max_examples": 1, "hf_repo": "owner/runs"},
                },
                run_id="x",
            )
    finally:
        catalog.MODELS.pop("test/no-opd", None)


def test_grpo_capability_still_enforced():
    # The guardrail: an SFT-only model rejects GRPO through the config path. No catalog
    # entry is SFT-only anymore, so inject a temporary one.
    from flash import catalog
    from flash.catalog import ModelInfo

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
                    "train": {"epochs": 1, "max_examples": 1, "hf_repo": "owner/runs"},
                },
                run_id="x",
            )
        # SFT on the same model is allowed.
        sft = spec_from_dict(
            {
                "model": "test/sft-only",
                "algorithm": "sft",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"epochs": 1, "max_examples": 8, "hf_repo": "owner/runs"},
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
            "train": {"epochs": 1, "max_examples": 1, "hf_repo": "owner/runs"},
            "gpu": {"type": "A100 PCIe"},
        },
        run_id="x",
    )
    assert spec.algorithm == "grpo"

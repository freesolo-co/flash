"""Guardrails on the model catalog and the package-wide default model.

These keep the curated catalog internally consistent and make sure the out-of-the-box
default an average developer hits is a real, supported model.
"""

from __future__ import annotations

import os

from flash.catalog import DEFAULT_MODEL, MODELS, get_model
from flash.providers.base import KNOWN, canonical_gpu


def test_recommended_gpu_is_supported():
    # Every catalog entry must recommend a GPU Flash actually manages.
    for model_id, info in MODELS.items():
        assert canonical_gpu(info.recommended_gpu) in KNOWN, (
            f"{model_id} recommends unsupported GPU {info.recommended_gpu!r}"
        )


def test_default_model_is_supported():
    info = get_model(DEFAULT_MODEL)
    # The default is GRPO+SFT capable so both algorithms work out of the box.
    assert "grpo" in info.algos
    assert "sft" in info.algos


def test_recipe_and_jobspec_defaults_match_catalog_default():
    from flash.engine.recipe import RECIPE
    from flash.spec import JobSpec

    # When BENCH_HF_MODEL isn't overriding it, the recipe + JobSpec default to the catalog default.
    if not os.environ.get("BENCH_HF_MODEL"):
        assert RECIPE.hf_model_id == DEFAULT_MODEL
    assert JobSpec().model == DEFAULT_MODEL


def test_thinking_capability_values_are_valid():
    # The config validator branches on these exact values; "unknown" is reserved for
    # open-model-policy entries and must not appear in the curated catalog.
    for model_id, info in MODELS.items():
        assert info.thinking in ("none", "hybrid", "always"), (model_id, info.thinking)


def test_serving_capacity_matches_validated_matrix():
    expected = {
        "Qwen/Qwen3.5-4B": {
            "gpu": "L40S",
            "serve_model_id": "lovedheart/Qwen3.5-4B-FP8",
            "max_loras": 256,
            "max_lora_rank": 32,
            "max_model_len": 8192,
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.98,
        },
        "Qwen/Qwen3.5-9B": {
            "gpu": "L40S",
            "serve_model_id": "lovedheart/Qwen3.5-9B-FP8",
            "max_loras": 176,
            "max_lora_rank": 32,
            "max_model_len": 8192,
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.98,
        },
        "Qwen/Qwen3.6-35B-A3B": {
            "gpu": "A100-80GB",
            "serve_model_id": "Qwen/Qwen3.6-35B-A3B-FP8",
            "max_loras": 12,
            "max_lora_rank": 32,
            "max_model_len": 8192,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.98,
        },
    }
    for model_id, values in expected.items():
        serving = get_model(model_id).serving
        assert serving is not None
        for key, value in values.items():
            assert getattr(serving, key) == value


def test_public_rows_include_serving_capacity():
    row = get_model("Qwen/Qwen3.5-4B").to_dict()
    assert row["serving"]["gpu"] == "L40S"
    assert row["serving"]["max_loras"] == 256


def test_default_model_is_thinking_capable():
    # The thinking flag defaults OFF, but the default model should stay thinking-capable
    # ("hybrid" or "always") so a plain default run can still opt into thinking = true
    # without being rejected by config_schema. (DEFAULT_MODEL is currently hybrid.)
    assert get_model(DEFAULT_MODEL).thinking in ("hybrid", "always")


def test_default_model_is_a_dense_text_model():
    # Guard against regressing the default back to a multimodal / novel-arch model: the
    # default should be the proven dense instruction model.
    assert DEFAULT_MODEL == "Qwen/Qwen3.5-4B"


def test_every_catalog_entry_sets_params_b():
    # params_b is the authoritative numeric model size — the VRAM/disk/cost terms read it DIRECTLY
    # (nothing parses the `params` display string any more). It is a required ModelInfo field, but 0.0
    # is a legal value only for the open-model policy's "unknown size" path. This guards that every
    # CURATED entry states a real, positive size, so a new entry can never silently size as 0/unknown.
    for model_id, info in MODELS.items():
        assert info.params_b > 0, f"{model_id} must set params_b > 0 (curated authoritative size)"

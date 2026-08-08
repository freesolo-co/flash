"""Guardrails on the model catalog and the package-wide default model.

These keep the curated catalog internally consistent and make sure the out-of-the-box
default an average developer hits is a real, supported model.
"""

from __future__ import annotations

import os

from flash.core.catalog import (
    DEFAULT_MODEL,
    MODELS,
    SERVING_MODEL_REPOS,
    get_model,
)
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
    from flash.core.spec import JobSpec
    from flash.engine.plan.recipe import RECIPE

    # When BENCH_HF_MODEL isn't overriding it, the recipe + JobSpec default to the catalog default.
    if not os.environ.get("BENCH_HF_MODEL"):
        assert RECIPE.hf_model_id == DEFAULT_MODEL
    assert JobSpec().model == DEFAULT_MODEL


def test_opd_mamba_block_size_is_catalogued_only_for_hybrid_rollout_model():
    assert MODELS["Qwen/Qwen3.6-35B-A3B"].mamba_block_size == 1072
    assert all(
        info.mamba_block_size == 0
        for model_id, info in MODELS.items()
        if model_id != "Qwen/Qwen3.6-35B-A3B"
    )


def test_thinking_capability_values_are_valid():
    # The config validator branches on these exact values; "unknown" is reserved for
    # a synthesized entry's placeholder, and must not appear in the curated catalog.
    for model_id, info in MODELS.items():
        assert info.thinking in ("none", "hybrid", "always"), (model_id, info.thinking)


def test_serving_capacity_matches_validated_matrix():
    # `expected` is a transcription guard, not a cross-repo lockstep check. flash CI has no
    # freesolo tree; compare both repos in the pinned freesolo-co/tests checkout.
    expected = {
        "Qwen/Qwen3.5-0.8B": {
            "gpu": "L4",
            "serve_model_id": "Freesolo-Co/Qwen3.5-0.8B-FP8",
            "max_loras": 16,
            "max_lora_rank": 128,
            "max_model_len": 32768,
        },
        "Qwen/Qwen3.5-2B": {
            "gpu": "L4",
            "serve_model_id": "Freesolo-Co/Qwen3.5-2B-FP8",
            "max_loras": 16,
            "max_lora_rank": 128,
            "max_model_len": 32768,
        },
        "Qwen/Qwen3.5-4B": {
            "gpu": "L4",
            "serve_model_id": "Freesolo-Co/Qwen3.5-4B-FP8",
            "max_loras": 16,
            "max_lora_rank": 128,
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "gpu_memory_utilization": 0.98,
        },
        "Qwen/Qwen3.5-9B": {
            "gpu": "L40S",
            "serve_model_id": "Freesolo-Co/Qwen3.5-9B-FP8",
            "max_loras": 16,
            "max_lora_rank": 128,
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "gpu_memory_utilization": 0.90,
        },
        "Qwen/Qwen3.6-27B": {
            "gpu": "H100",
            "serve_model_id": "Freesolo-Co/Qwen3.6-27B-FP8",
            "max_loras": 16,
            "max_lora_rank": 64,
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "gpu_memory_utilization": 0.90,
        },
        "Qwen/Qwen3.6-35B-A3B": {
            "gpu": "H200",
            "serve_model_id": "Qwen/Qwen3.6-35B-A3B",
            "max_loras": 6,
            "max_lora_rank": 64,
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.90,
        },
    }
    for model_id, values in expected.items():
        serving = get_model(model_id).serving
        assert serving is not None
        for key, value in values.items():
            assert getattr(serving, key) == value


def test_public_rows_include_serving_capacity():
    row = get_model("Qwen/Qwen3.5-4B").to_dict()
    assert row["serving"]["gpu"] == "L4"
    assert row["serving"]["max_loras"] == 16
    assert row["serving"]["max_lora_rank"] == 128
    assert row["serving"]["serve_model_id"] == "Freesolo-Co/Qwen3.5-4B-FP8"


def test_public_rows_prune_unset_serving_capacity_fields():
    row = get_model("Qwen/Qwen3.5-0.8B").to_dict()
    # serve_model_id survives while zero-valued optional capacity fields are pruned.
    assert row["serving"] == {
        "gpu": "L4",
        "serve_model_id": "Freesolo-Co/Qwen3.5-0.8B-FP8",
        "max_loras": 16,
        "max_lora_rank": 128,
        "max_model_len": 32768,
    }


def test_serving_repos_match_current_serving_matrix() -> None:
    # dense models serve freesolo-owned fp8 checkpoints; the qwen3.6 moe serves base bf16 on h200.
    assert SERVING_MODEL_REPOS["Qwen/Qwen3.5-0.8B"] == "Freesolo-Co/Qwen3.5-0.8B-FP8"
    assert SERVING_MODEL_REPOS["Qwen/Qwen3.5-2B"] == "Freesolo-Co/Qwen3.5-2B-FP8"
    assert SERVING_MODEL_REPOS["Qwen/Qwen3.5-4B"] == "Freesolo-Co/Qwen3.5-4B-FP8"
    assert SERVING_MODEL_REPOS["Qwen/Qwen3.5-9B"] == "Freesolo-Co/Qwen3.5-9B-FP8"
    assert SERVING_MODEL_REPOS["Qwen/Qwen3.6-27B"] == "Freesolo-Co/Qwen3.6-27B-FP8"
    assert SERVING_MODEL_REPOS["Qwen/Qwen3.6-35B-A3B"] == "Qwen/Qwen3.6-35B-A3B"


def test_qwen36_27b_geometry_is_dense_hybrid():
    info = MODELS["Qwen/Qwen3.6-27B"]
    assert info.vocab_size == 248_320
    assert info.hidden_size == 5120
    assert info.num_layers == 64
    assert info.active_params_b == 0.0
    assert info.is_moe is False
    assert info.thinking == "hybrid"
    assert info.params_b == 27.0


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
    # is the dataclass default. This guards that every entry states a real, positive size, so a new
    # entry added when forking to extend the catalog can never silently size as 0/unknown.
    for model_id, info in MODELS.items():
        assert info.params_b > 0, f"{model_id} must set params_b > 0 (curated authoritative size)"

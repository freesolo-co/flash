"""Guardrails on the model catalog and the package-wide default model.

These keep the curated catalog internally consistent and make sure the out-of-the-box
default an average developer hits is a real, supported model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from flash.core.catalog import (
    DEFAULT_MODEL,
    MODELS,
    SERVING_MODEL_REPOS,
    get_model,
)
from flash.providers.core.base import KNOWN, canonical_gpu


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
        "Qwen/Qwen3.5-9B": {
            "gpu": "B200",
            "serve_model_id": "Freesolo-Co/Qwen3.5-9B-FP8",
            "max_loras": 16,
            "max_cpu_loras": 16,
            "max_lora_rank": 128,
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "image_limit": 4,
        },
        "Qwen/Qwen3.8-27B": {
            "gpu": "B200",
            "serve_model_id": "Qwen/Qwen3.8-27B-FP8",
            "max_loras": 16,
            "max_cpu_loras": 16,
            "max_lora_rank": 64,
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "image_limit": 4,
        },
        "Qwen/Qwen3.6-35B-A3B": {
            "gpu": "B200",
            "serve_model_id": "Qwen/Qwen3.6-35B-A3B",
            "max_loras": 6,
            "max_cpu_loras": 6,
            "max_lora_rank": 64,
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 4096,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "image_limit": 4,
        },
    }
    for model_id, values in expected.items():
        serving = get_model(model_id).serving
        assert serving is not None
        for key, value in values.items():
            assert getattr(serving, key) == value


def test_public_rows_include_serving_capacity():
    row = get_model("Qwen/Qwen3.5-9B").to_dict()
    assert row["serving"]["gpu"] == "B200"
    assert row["serving"]["max_loras"] == 16
    assert row["serving"]["max_lora_rank"] == 128
    assert row["serving"]["serve_model_id"] == "Freesolo-Co/Qwen3.5-9B-FP8"


def test_public_rows_prune_unset_serving_capacity_fields():
    row = get_model("Qwen/Qwen3.5-9B").to_dict()
    # serve_model_id survives while zero-valued optional capacity fields are pruned.
    assert row["serving"] == {
        "gpu": "B200",
        "serve_model_id": "Freesolo-Co/Qwen3.5-9B-FP8",
        "max_loras": 16,
        "max_cpu_loras": 16,
        "max_lora_rank": 128,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
        "image_limit": 4,
    }


def test_serving_repos_cover_public_catalog_and_hosted_activation() -> None:
    assert SERVING_MODEL_REPOS == {
        "Qwen/Qwen3.5-9B": "Freesolo-Co/Qwen3.5-9B-FP8",
        "Qwen/Qwen3.8-27B": "Qwen/Qwen3.8-27B-FP8",
        "Qwen/Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B",
    }
    assert MODELS["Qwen/Qwen3.8-27B"].serving is not None
    from flash.serving.src.engine.model_config import SERVING_MODELS

    assert "Qwen/Qwen3.8-27B" in {entry["base_model"] for entry in SERVING_MODELS}


def test_qwen38_27b_fixture_binds_checkpoint_metadata() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "qwen38_27b_target_metadata.json").read_text()
    )
    info = MODELS["Qwen/Qwen3.8-27B"]
    assert fixture["base"]["model"] == info.id
    assert fixture["base"]["revision"] == info.managed_revision
    assert fixture["base"]["architecture"] == "Qwen3_5ForConditionalGeneration"
    assert fixture["base"]["model_type"] == "qwen3_5"
    assert fixture["base"]["parameter_count"] == 27_781_427_952
    assert fixture["base"]["weight_bytes"] == 55_562_855_904
    assert fixture["base"]["parameter_count"] / 1e9 == info.params_b
    assert tuple(tuple(row) for row in fixture["base"]["target_shapes"]) == info.lora_target_shapes
    assert fixture["base"]["target_count"] == sum(row[2] for row in info.lora_target_shapes)
    assert fixture["base"]["geometry"] == {
        "attention_heads": 24,
        "full_attention_layers": 16,
        "hidden_size": 5120,
        "key_value_heads": 4,
        "layers": 64,
        "linear_attention_layers": 48,
        "vision_layers": 27,
        "vocab_size": info.vocab_size,
    }
    assert fixture["tokenizer"]["class"] == "Qwen2Tokenizer"
    assert fixture["tokenizer"]["vocab_size"] == 248044
    assert fixture["tokenizer"]["length"] == 248077
    assert len(fixture["tokenizer"]["added_tokens"]) == 33
    assert fixture["tokenizer"]["preserve_thinking_default"] is True
    assert fixture["tokenizer"]["chat_template_sha256"] == (
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
    )
    assert fixture["tokenizer"]["length"] != fixture["base"]["geometry"]["vocab_size"]
    assert fixture["processor"] == {
        "class": "Qwen3VLProcessor",
        "image_processor_class": "Qwen2VLImageProcessorFast",
    }
    assert fixture["fp8"] == {
        "activation_scheme": "dynamic",
        "format": "e4m3",
        "model": "Qwen/Qwen3.8-27B-FP8",
        "quant_method": "fp8",
        "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "weight_block_size": [128, 128],
    }


def test_qwen38_27b_geometry_is_dense_hybrid():
    info = MODELS["Qwen/Qwen3.8-27B"]
    assert info.vocab_size == 248_320
    assert info.hidden_size == 5120
    assert info.num_layers == 64
    assert info.active_params_b == 0.0
    assert info.is_moe is False
    assert info.thinking == "hybrid"
    assert info.params_b == 27.781427952
    assert info.managed_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert "managed_revision" not in info.to_dict()


@pytest.mark.parametrize(
    "model_id",
    ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-27B"],
)
def test_retired_models_are_absent_from_every_active_catalog(model_id):
    assert model_id not in MODELS
    assert model_id not in SERVING_MODEL_REPOS
    with pytest.raises(ValueError, match="unsupported model"):
        get_model(model_id)


def test_default_model_is_thinking_capable():
    # The thinking flag defaults OFF, but the default model should stay thinking-capable
    # ("hybrid" or "always") so a plain default run can still opt into thinking = true
    # without being rejected by config_schema. (DEFAULT_MODEL is currently hybrid.)
    assert get_model(DEFAULT_MODEL).thinking in ("hybrid", "always")


def test_default_model_is_a_dense_text_model():
    # Guard against regressing the default back to a multimodal / novel-arch model: the
    # default should be the proven dense instruction model.
    assert DEFAULT_MODEL == "Qwen/Qwen3.5-9B"


def test_every_catalog_entry_sets_params_b():
    # params_b is the authoritative numeric model size — the VRAM/disk/cost terms read it DIRECTLY
    # (nothing parses the `params` display string any more). It is a required ModelInfo field, but 0.0
    # is the dataclass default. This guards that every entry states a real, positive size, so a new
    # entry added when forking to extend the catalog can never silently size as 0/unknown.
    for model_id, info in MODELS.items():
        assert info.params_b > 0, f"{model_id} must set params_b > 0 (curated authoritative size)"

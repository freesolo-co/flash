"""Per-base-model serving config: catalog shape."""

from __future__ import annotations

import pytest

from flash.serving.src.engine.model_config import (
    SERVING_MODELS,
    base_models,
    engine_overrides_for,
    gpu_for,
    image_limit_for,
    is_supported_base_model,
    reasoning_parser_for,
    serve_model_for,
    supports_image_input,
    tool_parser_for,
)


def test_catalog_has_only_canary_qualified_active_models() -> None:
    assert set(base_models()) == {
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.6-35B-A3B",
    }


def test_catalog_entries_are_wellformed() -> None:
    for entry in SERVING_MODELS:
        assert entry.get("base_model")
        assert "image_input_limit" in entry


def test_every_catalog_model_has_an_intentional_image_classification() -> None:
    image_models = {model for model in base_models() if supports_image_input(model)}
    assert image_models == {
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.6-35B-A3B",
    }
    assert all(image_limit_for(model) == 4 for model in image_models)


def test_uncataloged_base_models_are_not_supported() -> None:
    assert is_supported_base_model("Qwen/Qwen3.5-9B") is True
    assert is_supported_base_model("openai/gpt-oss-20b") is False
    with pytest.raises(ValueError, match="Unsupported base model"):
        gpu_for("openai/gpt-oss-20b")


def test_reasoning_parser_is_configured_for_every_model() -> None:
    for model in base_models():
        assert reasoning_parser_for(model) == "qwen3"


def test_tool_parser_is_qualified_only_for_exact_qwen35() -> None:
    assert tool_parser_for("Qwen/Qwen3.5-9B") == "qwen3_coder"
    assert tool_parser_for("Qwen/Qwen3.6-35B-A3B") is None


def test_no_model_runs_language_model_only() -> None:
    # No tier sets language_model_only: flash adapters adapt the FULL multimodal tree, so their
    # vision-tower LoRA keys need the vision encoder loaded to bind to. Every base loads the whole
    # VL model (the 35B included) — none carries the text-only skip.
    assert "language_model_only" not in engine_overrides_for("Qwen/Qwen3.6-35B-A3B")
    assert "language_model_only" not in engine_overrides_for("Qwen/Qwen3.5-9B")


def test_35b_serves_bf16_base_not_fp8() -> None:
    # The 35B serves the BASE bf16 weights, NOT the FP8 checkpoint, so full all-expert LoRA + CUDA
    # graphs can coexist (bf16 sidesteps the A100-only fp8e4nv fused-MoE-LoRA kernel). Its engine dict
    # pins serve_model_id to the bf16 base and quantization=None, overriding the injected FP8 default.
    ov = engine_overrides_for("Qwen/Qwen3.6-35B-A3B")
    assert ov.get("serve_model_id") == "Qwen/Qwen3.6-35B-A3B"  # bf16 base, not the FP8 checkpoint
    assert ov.get("quantization") is None  # explicit bf16 (overrides the injected FP8 default)
    assert "moe_backend" not in ov
    assert "kv_cache_dtype" not in ov  # KV inherits the global fp8


def test_35b_does_not_pin_loras() -> None:
    # max_loras (GPU slots) is capped below max_cpu_loras for the MoE while max_cpu_loras stays at 256,
    # so the 35B opts OUT of pinning — otherwise the first max_loras adapters would lock the GPU slots
    # and the surplus could never load. Unpinned, vLLM LRU-swaps from the CPU cache on demand.
    assert engine_overrides_for("Qwen/Qwen3.6-35B-A3B").get("pin_loras") is False
    # Dense text models carry no explicit pin override; modal_app derives the default from whether
    # max_loras covers max_cpu_loras.
    assert "pin_loras" not in engine_overrides_for("Qwen/Qwen3.5-9B")


def test_dense_models_serve_fp8_and_35b_serves_bf16_base() -> None:
    # every dense catalog model loads a pre-quantized FP8 checkpoint with no online quantization.
    # serve_model_for() resolves it and engine_overrides_for() injects it as serve_model_id.
    expected_fp8 = {
        "Qwen/Qwen3.5-9B": "Freesolo-Co/Qwen3.5-9B-FP8",
        "Qwen/Qwen3.8-27B": "Qwen/Qwen3.8-27B-FP8",
    }
    for base, ckpt in expected_fp8.items():
        assert serve_model_for(base) == ckpt
        assert engine_overrides_for(base)["serve_model_id"] == ckpt
    # The 35B MoE is the exception: serve_model_for() still resolves the official FP8 checkpoint as the
    # DEFAULT, but the 35B's engine dict overrides serve_model_id to the BASE bf16 weights (full-expert
    # LoRA + graphs need bf16). So the INJECTED serve_model_id is the bf16 base, not the FP8 checkpoint.
    assert serve_model_for("Qwen/Qwen3.6-35B-A3B") == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert engine_overrides_for("Qwen/Qwen3.6-35B-A3B")["serve_model_id"] == "Qwen/Qwen3.6-35B-A3B"
    # Unknown models are rejected instead of silently using the L4 default tier.
    with pytest.raises(ValueError, match="Unsupported base model"):
        serve_model_for("some/unlisted-model")


def test_no_bitsandbytes_anywhere() -> None:
    # bitsandbytes is the slowest inference quant and was deleted from the catalog.
    for entry in SERVING_MODELS:
        assert entry.get("quantization") != "bitsandbytes"
        assert entry.get("load_format") != "bitsandbytes"


def test_9b_has_b200_rank128_serving_overrides() -> None:
    ov = engine_overrides_for("Qwen/Qwen3.5-9B")
    assert ov["max_loras"] == 16
    assert ov["max_lora_rank"] == 128
    assert ov["max_model_len"] == 32768
    assert ov["max_num_seqs"] == 8
    # the 9B serves on B200: 132.87 GiB of KV and 217x concurrency at 32k with rank-128 x 16.
    assert gpu_for("Qwen/Qwen3.5-9B") == "B200"
    assert ov["gpu_memory_utilization"] == 0.90
    assert (
        ov["enforce_eager"] is False
    )  # CUDA graphs on (~10x faster decode on this hybrid GDN model)


def test_qwen38_27b_is_active_on_b200_with_pinned_immutable_revisions() -> None:
    # 27B is a FULLY ACTIVE hosted tier: every active lookup resolves it, and its served checkpoint,
    # tokenizer and processor stay pinned to immutable revisions so a served engine cannot follow an
    # upstream retag. The rank-64 x 16 LoRA shape at 32k measured 112.66 GiB of KV on the B200.
    base_model = "Qwen/Qwen3.8-27B"

    assert base_model in base_models()
    assert is_supported_base_model(base_model) is True
    assert gpu_for(base_model) == "B200"
    assert image_limit_for(base_model) == 4
    assert serve_model_for(base_model) == "Qwen/Qwen3.8-27B-FP8"

    ov = engine_overrides_for(base_model)
    assert ov["serve_model_id"] == "Qwen/Qwen3.8-27B-FP8"
    assert ov["tokenizer_model"] == base_model
    assert ov["max_loras"] == 16
    assert ov["max_lora_rank"] == 64
    assert ov["max_model_len"] == 32768
    assert ov["max_num_seqs"] == 8
    assert ov["gpu_memory_utilization"] == 0.90
    assert ov["enforce_eager"] is False
    assert ov["reasoning_parser"] == "qwen3"
    assert ov["model_revision"] == "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
    assert {
        key: ov[key] for key in ("model_revision", "tokenizer_revision", "processor_revision")
    } == {
        "model_revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "processor_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    }
    assert "language_model_only" not in ov


def test_35b_has_b200_bf16_rank64_six_loras_overrides() -> None:
    # the 35B MoE runs bf16 on the B200 with rank 64 at 6 hot slots and CUDA graphs on. bf16 is the
    # only config where full all-expert LoRA and graphs coexist (FP8 forces eager on the A100 and
    # fails the fp8e4nv fused-MoE-LoRA kernel); see SERVING_MODELS for the full rationale.
    assert gpu_for("Qwen/Qwen3.6-35B-A3B") == "B200"
    ov = engine_overrides_for("Qwen/Qwen3.6-35B-A3B")
    assert ov["max_loras"] == 6
    assert ov["max_lora_rank"] == 64
    assert ov["max_model_len"] == 32768
    assert ov["max_num_seqs"] == 8
    assert ov["max_num_batched_tokens"] == 4096
    assert (
        ov["gpu_memory_utilization"] == 0.90
    )  # headroom above the weights + 6 x 64 LoRA buffer for KV + graphs
    assert ov["enforce_eager"] is False  # CUDA graphs on for the bf16/B200 path

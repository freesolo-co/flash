from __future__ import annotations

import math

import pytest

from flash.catalog import MODELS
from flash.engine.vram import (
    _KV_BLOCK_TOKENS,
    _LORA_ADAMW_BYTES_PER_PARAM,
    _architecture_kv_raw_gb,
    _legacy_lora_floor_gb,
    _lora_memory_gb,
    model_required_vram_gb,
)

_EXPECTED_LORA_TARGET_COUNTS = {
    "openbmb/MiniCPM5-1B": 168,
    "Qwen/Qwen3.5-0.8B": 236,
    "Qwen/Qwen3.5-2B": 284,
    "Qwen/Qwen3.5-4B": 346,
    "Qwen/Qwen3.5-9B": 358,
    "Qwen/Qwen3.6-35B-A3B": 460,
    "Qwen/Qwen3.6-27B": 606,
}


def test_curated_architecture_geometry_covers_every_loaded_lora_target():
    for model_id, info in MODELS.items():
        # only models with curated arch-aware geometry are covered here; a model registered
        # without a declared LoRA-target-count is not an arch-aware-sizing target and is skipped
        if model_id not in _EXPECTED_LORA_TARGET_COUNTS:
            continue
        assert info.num_attention_layers > 0
        assert info.num_attention_layers + info.num_linear_attention_layers == info.num_layers
        assert info.num_key_value_heads > 0
        assert info.head_dim > 0
        assert info.lora_target_shapes
        assert (
            sum(count for _in_features, _out_features, count in info.lora_target_shapes)
            == (_EXPECTED_LORA_TARGET_COUNTS[model_id])
        )
        assert all(
            in_features > 0 and out_features > 0 and count > 0
            for in_features, out_features, count in info.lora_target_shapes
        )


@pytest.mark.parametrize("model_id", tuple(MODELS))
def test_shape_sizing_keeps_the_measured_lora_floor(model_id: str):
    info = MODELS[model_id]
    rank = 128
    effective_params_b = info.active_params_b or info.params_b
    target_dims = sum(
        (in_features + out_features) * count
        for in_features, out_features, count in info.lora_target_shapes
    )
    exact_opd_gb = rank * target_dims * _LORA_ADAMW_BYTES_PER_PARAM / 1e9
    sized_gb = _lora_memory_gb(rank, effective_params_b, "opd", info)

    assert sized_gb >= exact_opd_gb
    assert sized_gb >= _legacy_lora_floor_gb(rank, effective_params_b)


def test_gdn_state_page_and_attention_kv_use_real_geometry():
    dense = MODELS["openbmb/MiniCPM5-1B"]
    dense_raw = _architecture_kv_raw_gb(dense, 4096, 8, False)
    dense_expected = (
        dense.num_attention_layers
        * 8
        * 4096
        * 2
        * dense.num_key_value_heads
        * dense.head_dim
        * 2
        / 1e9
    )
    assert dense_raw == pytest.approx(dense_expected)

    moe = MODELS["Qwen/Qwen3.6-35B-A3B"]
    state_elements = (
        moe.linear_num_value_heads * moe.linear_key_head_dim * moe.linear_value_head_dim
    )
    state_elements += (
        moe.linear_num_key_heads * moe.linear_key_head_dim
        + moe.linear_num_value_heads * moe.linear_value_head_dim
    ) * moe.linear_conv_kernel_dim
    state_bytes = state_elements * 2
    fp8_attention_bytes_per_token = 2 * moe.num_key_value_heads * moe.head_dim
    derived_block = math.ceil(state_bytes / fp8_attention_bytes_per_token)
    derived_block = math.ceil(derived_block / _KV_BLOCK_TOKENS) * _KV_BLOCK_TOKENS

    assert state_bytes == 1_097_728
    assert derived_block == moe.mamba_block_size == 1072
    assert _architecture_kv_raw_gb(moe, 4096, 8, True) < _architecture_kv_raw_gb(
        moe, 4096, 8, False
    )


def test_pinned_grpo_resident_check_uses_generic_geometry(monkeypatch):
    import flash.engine.vram as vram_mod

    captured = {}

    def _capture_estimate(params_b, *_args, **kwargs):
        captured["params_b"] = params_b
        captured.update(kwargs)
        return 1.0

    monkeypatch.setattr(
        vram_mod,
        "resolve_params_b",
        lambda _model_id, revision="": 35.5 if revision == "a" * 40 else None,
    )
    monkeypatch.setattr(vram_mod, "estimate_vram_gb", _capture_estimate)

    assert vram_mod.grpo_fits_resident(
        "Qwen/Qwen3.6-35B-A3B",
        card_vram_gb=180,
        revision="a" * 40,
    )
    assert captured["params_b"] == 35.5
    assert captured["active_params_b"] == 0.0
    assert captured["model_info"] is None


def test_sizing_accuracy_matrix_preserves_safe_boundaries_and_removes_overrouting():
    cases = {
        "gqa_dense_sft": (
            "openbmb/MiniCPM5-1B",
            "sft",
            {"max_context_tokens": 4096, "lora_rank": 32},
            18,
        ),
        "gdn_vl_small_grpo": (
            "Qwen/Qwen3.5-0.8B",
            "grpo",
            {"max_context_tokens": 32768, "group_size": 8, "lora_rank": 32},
            72,
        ),
        "gdn_vl_high_rank_sft": (
            "Qwen/Qwen3.5-2B",
            "sft",
            {"max_context_tokens": 4096, "batch_size": 4, "lora_rank": 128},
            32,
        ),
        "gdn_vl_4b_grpo": (
            "Qwen/Qwen3.5-4B",
            "grpo",
            {
                "max_context_tokens": 16384,
                "max_completion_tokens": 4096,
                "group_size": 8,
                "lora_rank": 32,
            },
            98,
        ),
        "gdn_vl_9b_grpo": (
            "Qwen/Qwen3.5-9B",
            "grpo",
            {"max_context_tokens": 4096, "group_size": 8, "lora_rank": 64},
            80,
        ),
        "moe_sft": (
            "Qwen/Qwen3.6-35B-A3B",
            "sft",
            {"max_context_tokens": 4096, "batch_size": 4, "lora_rank": 64},
            103,
        ),
        "moe_grpo": (
            "Qwen/Qwen3.6-35B-A3B",
            "grpo",
            {
                "max_context_tokens": 4096,
                "max_completion_tokens": 384,
                "group_size": 8,
                "lora_rank": 16,
            },
            180,
        ),
    }

    sized = {
        name: model_required_vram_gb(model_id, algorithm, train=train)
        for name, (model_id, algorithm, train, _old_need) in cases.items()
    }
    for name, (_model_id, _algorithm, _train, old_need) in cases.items():
        assert sized[name] <= old_need

    assert sized["gqa_dense_sft"] <= 24
    assert sized["gdn_vl_small_grpo"] <= 24
    assert sized["gdn_vl_high_rank_sft"] <= 32
    assert model_required_vram_gb("Qwen/Qwen3.5-2B", "grpo") <= 32
    assert model_required_vram_gb("Qwen/Qwen3.5-4B", "grpo", train={"group_size": 8}) <= 48
    assert sized["gdn_vl_9b_grpo"] <= 80
    assert sized["moe_sft"] <= 141
    assert sized["moe_grpo"] <= 180

    assert sized["gdn_vl_small_grpo"] < cases["gdn_vl_small_grpo"][3]
    assert sized["gdn_vl_4b_grpo"] < cases["gdn_vl_4b_grpo"][3]


def test_sizing_accuracy_matrix_does_not_accept_known_oom_boundaries():
    assert model_required_vram_gb("Qwen/Qwen3.5-0.8B", "grpo") > 20
    assert model_required_vram_gb("Qwen/Qwen3.5-2B", "grpo") > 24
    assert (
        model_required_vram_gb(
            "Qwen/Qwen3.5-4B",
            "grpo",
            train={"max_context_tokens": 4096, "group_size": 16},
        )
        > 31
    )
    assert (
        model_required_vram_gb(
            "Qwen/Qwen3.5-4B",
            "grpo",
            train={
                "max_context_tokens": 32768,
                "max_completion_tokens": 8192,
                "group_size": 8,
            },
        )
        > 32
    )
    # NOTE: 4B SFT @ 8192 is NOT a >32GB OOM boundary anymore -- chunked-NLL (enabled+validated for
    # Qwen3.5-4B via #582) drops dense logits, so it fits a 32GB card (~27GB). That fits-32GB behavior is
    # asserted by test_qwen4b_sft_8192_chunked_nll_routes_to_32gb_card; keeping a >32 boundary here would
    # test a boundary chunked-NLL legitimately removed.
    assert model_required_vram_gb("Qwen/Qwen3.5-9B", "grpo") > 48
    assert (
        model_required_vram_gb(
            "Qwen/Qwen3.6-35B-A3B",
            "grpo",
            train={
                "max_context_tokens": 8192,
                "max_completion_tokens": 384,
                "group_size": 8,
                "lora_rank": 16,
            },
        )
        > 180
    )

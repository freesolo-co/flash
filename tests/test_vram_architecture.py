from __future__ import annotations

import math

import pytest

from flash.catalog import MODELS, serving_lora_rank_cap
from flash.engine.vram import (
    _KV_BLOCK_TOKENS,
    _LORA_ADAMW_BYTES_PER_PARAM,
    _architecture_kv_raw_gb,
    _legacy_lora_floor_gb,
    _lora_memory_gb,
    model_required_vram_gb,
)

_EXPECTED_LORA_TARGET_COUNTS = {
    "Qwen/Qwen3.5-0.8B": 236,
    "Qwen/Qwen3.5-2B": 284,
    "Qwen/Qwen3.5-4B": 346,
    "Qwen/Qwen3.5-9B": 358,
    # 460 ordinary linears + both fused routed-expert tensors (40 layers x 256 experts each),
    # which peft wraps as 10,240 rank-r slices per tensor.
    "Qwen/Qwen3.6-35B-A3B": 460 + 2 * 10_240,
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


@pytest.mark.parametrize("algorithm", ["sft", "opd"])
@pytest.mark.parametrize("model_id", tuple(MODELS))
def test_sizing_covers_fp32_adamw_for_sft_and_opd(model_id: str, algorithm: str):
    """SFT and OPD size on fp32 AdamW. verl builds torch.optim.AdamW for both -- OPD inherits verl's
    own default and SFT names it explicitly for FSDP2/DTensor safety -- and the backend lives in the
    spec's [worker_env], which never reaches this sizing path. so the estimate must cover the AdamW
    footprint unconditionally; sizing on the 8-bit paged optimizer under-reserves a verl run and
    places it on a card that cannot hold its optimizer state.

    GRPO is excluded on purpose: verl GRPO runs AdamW too, but its estimate is already at the B200
    ceiling (the 35B MoE at 4096 ctx sizes to 180 GB exactly), so widening it rejects a configuration
    that runs today. that gap is tracked separately and needs the resident-peak model retuned."""
    info = MODELS[model_id]
    rank = serving_lora_rank_cap(model_id) or 64
    target_dims = sum(
        (in_features + out_features) * count
        for in_features, out_features, count in info.lora_target_shapes
    )
    adamw_gb = rank * target_dims * _LORA_ADAMW_BYTES_PER_PARAM / 1e9
    effective_params_b = info.active_params_b or info.params_b

    assert _lora_memory_gb(rank, effective_params_b, algorithm, info) >= adamw_gb
    # and the whole-run estimate the allocator ranks on must carry it too
    assert model_required_vram_gb(model_id, algorithm, train={"lora_rank": rank}) >= adamw_gb


def test_9b_rank64_sft_is_sized_off_the_rtx_5090():
    """the exact configuration the paged-optimizer estimate mis-sized: 9B SFT at rank 64 (its catalog
    cap) fits a 32 GB RTX 5090 only when the adapter is sized on the 8-bit paged optimizer. under
    verl's AdamW it needs 33 GB, so a paged-based estimate would place a run on a card 1 GB short."""
    need = model_required_vram_gb("Qwen/Qwen3.5-9B", "sft", train={"lora_rank": 64})
    assert need > 32, f"9B rank-64 SFT sized at {need} GB would still be placed on a 32 GB RTX 5090"


def test_gdn_state_page_and_attention_kv_use_real_geometry():
    gdn = MODELS["Qwen/Qwen3.5-0.8B"]
    gdn_raw = _architecture_kv_raw_gb(gdn, 4096, 8, False)
    attention_expected = (
        gdn.num_attention_layers * 8 * 4096 * 2 * gdn.num_key_value_heads * gdn.head_dim * 2 / 1e9
    )
    assert gdn_raw is not None
    assert gdn_raw > attention_expected

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
        # the two moe ceilings sit above the pre-expert numbers (103 and 180) on purpose: the routed
        # experts are real trainable parameters, so an estimate that excluded them under-reserved.
        "moe_sft": (
            "Qwen/Qwen3.6-35B-A3B",
            "sft",
            {"max_context_tokens": 4096, "batch_size": 4, "lora_rank": 64},
            154,
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
            190,
        ),
    }

    sized = {
        name: model_required_vram_gb(model_id, algorithm, train=train)
        for name, (model_id, algorithm, train, _old_need) in cases.items()
    }
    for name, (_model_id, _algorithm, _train, old_need) in cases.items():
        assert sized[name] <= old_need

    # 0.8B @ 32k ctx is NOT a 24/32 GB run despite the tiny weights: the long-context KV dominates.
    # The resident peak is ~38.5 GB (grpo_fits_resident rejects 24 and 32, admits 48) and the sleep
    # pool the worker reserves -- max(_KV_CAP, 1.5 * arch KV) at group 8 -- is likewise > 32 GB. The
    # old <= 24 bound encoded the sleep-KV under-count this PR removes (preflight would have admitted a
    # 24 GB card the sleep-mode vLLM executor then OOMs); the run correctly sizes onto the 48 GB tier.
    assert 32 < sized["gdn_vl_small_grpo"] <= 48
    assert sized["gdn_vl_high_rank_sft"] <= 32
    assert model_required_vram_gb("Qwen/Qwen3.5-2B", "grpo") <= 32
    assert model_required_vram_gb("Qwen/Qwen3.5-4B", "grpo", train={"group_size": 8}) <= 48
    assert sized["gdn_vl_9b_grpo"] <= 80
    # training the routed experts adds ~3.7B trainable parameters at rank 64, so the 35B no longer
    # fits one card for these shapes; it routes to two. sizing it back under a single-card tier would
    # mean under-reserving a run that really does need the memory.
    assert sized["moe_sft"] <= 2 * 141
    assert sized["moe_grpo"] <= 2 * 180

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

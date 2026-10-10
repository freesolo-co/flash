from __future__ import annotations

import math

import pytest

from flash.core.catalog import MODELS, serving_lora_rank_cap
from flash.engine.plan.vram import (
    _KV_BLOCK_TOKENS,
    _LORA_ADAMW_BYTES_PER_PARAM,
    _architecture_kv_raw_gb,
    _legacy_lora_floor_gb,
    _lora_memory_gb,
    _lora_parameter_count,
    _lora_weight_memory_gb,
    model_required_vram_gb,
)

_EXPECTED_LORA_TARGET_COUNTS = {
    "Qwen/Qwen3.5-9B": 358,
    # 460 ordinary linears + both fused routed-expert tensors (40 layers x 256 experts each),
    # which peft wraps as 10,240 rank-r slices per tensor.
    "Qwen/Qwen3.6-35B-A3B": 460 + 2 * 10_240,
    "Qwen/Qwen3.8-27B": 606,
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


@pytest.mark.parametrize(
    ("rank", "expected_params", "expected_bf16_gb"),
    [
        (16, 953_178_240, 1.90635648),
        (32, 1_906_356_480, 3.81271296),
        (64, 3_812_712_960, 7.62542592),
    ],
)
def test_qwen36_moe_lora_footprint_matches_fused_expert_geometry(
    rank: int, expected_params: int, expected_bf16_gb: float
):
    info = MODELS["Qwen/Qwen3.6-35B-A3B"]
    fused_count = info.num_layers * info.lora_expert_count

    assert info.lora_expert_count == 256
    assert (
        sum(count == fused_count for _input_dim, _output_dim, count in info.lora_target_shapes) == 2
    )
    assert _lora_parameter_count(rank, info) == expected_params
    assert _lora_weight_memory_gb(rank, info) == pytest.approx(expected_bf16_gb)


def test_qwen36_moe_tp2_lora_footprint_keeps_one_projection_factor_replicated():
    info = MODELS["Qwen/Qwen3.6-35B-A3B"]
    full_params = _lora_parameter_count(64, info)
    rank_params = _lora_parameter_count(64, info, tensor_parallel=2)

    # vllm keeps the shared-expert gate's min-dim-1 factor whole on every rank. the gate shape costs
    # 5,245,440 parameters rather than the fractional-shard result of 5,244,160.
    gate_shape_params = 64 * 40 * (2048 + 1)
    assert gate_shape_params == 5_245_440
    assert full_params == 3_812_712_960
    assert rank_params == 3_295_638_016
    assert rank_params > full_params / 2
    assert _lora_weight_memory_gb(64, info, tensor_parallel=2) == pytest.approx(6.591276032)


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
    own default and SFT names it explicitly for FSDP2/DTensor safety. the managed backend is not a
    sizing input, so the estimate must cover the AdamW footprint unconditionally; sizing on the
    8-bit paged optimizer under-reserves a verl run and
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
    gdn = MODELS["Qwen/Qwen3.5-9B"]
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


def test_9b_grpo_stays_within_its_validated_placement_tier(monkeypatch):
    from flash.cost import RunConfig, estimate_cost
    from flash.providers.core import allocator

    train = {
        "max_context_tokens": 4096,
        "max_completion_tokens": 320,
        "group_size": 8,
        "lora_rank": 64,
    }
    need = model_required_vram_gb("Qwen/Qwen3.5-9B", "grpo", train=train)
    monkeypatch.setattr(allocator, "available_providers", lambda: ("runpod",))
    allocation = allocator.allocate("Qwen/Qwen3.5-9B", "grpo", train=train)
    quote = estimate_cost(
        RunConfig(
            "Qwen/Qwen3.5-9B",
            "grpo",
            1,
            seq_len=train["max_context_tokens"],
            completion_len=train["max_completion_tokens"],
            group_size=train["group_size"],
            lora_rank=train["lora_rank"],
            provider="runpod",
        )
    )

    assert 48 < need <= 80
    assert allocation.min_vram_gb == quote.required_vram_gb == need
    assert (allocation.gpu, allocation.gpu_count) == (quote.gpu, quote.gpu_count)
    assert allocation.gpu == "A100 PCIe"


def test_grpo_estimator_grows_with_context_and_completion():
    from flash.engine.plan.vram import estimate_vram_gb

    short_context = estimate_vram_gb(
        9.7, "grpo", seq_len=1024, max_tokens=512, group_size=8, use_vllm=False
    )
    long_context = estimate_vram_gb(
        9.7, "grpo", seq_len=8192, max_tokens=512, group_size=8, use_vllm=False
    )
    short_completion = estimate_vram_gb(
        9.7, "grpo", seq_len=4096, max_tokens=128, group_size=8, use_vllm=False
    )
    long_completion = estimate_vram_gb(
        9.7, "grpo", seq_len=4096, max_tokens=2048, group_size=8, use_vllm=False
    )

    assert long_context > short_context
    assert long_completion > short_completion


def test_sizing_accuracy_matrix_does_not_accept_known_oom_boundaries():
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

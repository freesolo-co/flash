"""CPU test for ``colocate_kv_util`` — the need-based vLLM KV-pool utilization for colocated GRPO.

``gpu_memory_utilization`` is vLLM's whole model-executor budget (weights + KV), so the helper budgets
both, scales the KV with context + the generation group, and caps the utilization at 0.45.
"""

from __future__ import annotations

from flash.engine.vram import _KV_CAP, _resident_kv_gb, colocate_kv_util


def test_validated_4b_config_is_025():
    # The GPU-validated operating point (Qwen3.5-4B, group 8, 2k ctx, 80 GB): 0.25 vs old blanket 0.45.
    assert round(colocate_kv_util(4.0, 2048, 80.0, sleep_mode=True, num_generations=8), 2) == 0.25
    assert colocate_kv_util(4.0, 2048, 80.0, sleep_mode=True, num_generations=8) < 0.45


def test_budgets_weights_so_bigger_models_get_a_bigger_budget():
    # gpu_memory_utilization = weights + KV, so a bigger model's larger weight copy raises the budget.
    # (Budgeting KV alone would starve the weights — e.g. 9B would get the same 0.25 -> ~2 GB KV.)
    u4 = colocate_kv_util(4.0, 2048, 141.0, sleep_mode=True, num_generations=8)
    u9 = colocate_kv_util(9.0, 2048, 141.0, sleep_mode=True, num_generations=8)
    assert u9 > u4


def test_scales_with_generation_group():
    # More concurrent sequences (group_size) -> proportionally more KV needed.
    u8 = colocate_kv_util(4.0, 2048, 80.0, sleep_mode=True, num_generations=8)
    u16 = colocate_kv_util(4.0, 2048, 80.0, sleep_mode=True, num_generations=16)
    assert u16 > u8


def test_preserves_kv_for_long_contexts():
    # Long context -> bigger pool (the KV is NOT capped at the estimate; the 0.45 util cap bounds it).
    u_short = colocate_kv_util(4.0, 2048, 80.0, sleep_mode=True, num_generations=8)
    u_long = colocate_kv_util(4.0, 4096, 80.0, sleep_mode=True, num_generations=8)
    assert u_long > u_short


def test_non_sleep_path_budgets_weights_plus_kv():
    # gpu_memory_utilization is the WHOLE executor budget, so the non-sleep path budgets the bf16
    # weight copy + the resident-KV target (_KV_CAP). The old _KV_CAP/total budgeted KV ALONE, which
    # starved the weights -- vLLM raised "No available memory for the cache blocks" on >=3B models
    # whose weights already exceed an 8 GB total budget (hit once the 4B resident path skips sleep).
    for params_b, card in [(0.8, 80.0), (2.0, 48.0), (4.0, 80.0), (4.7, 48.0)]:
        u = colocate_kv_util(params_b, 2048, card, sleep_mode=False)
        assert u * card >= params_b * 2.0  # the engine can actually load its weight copy (the fix)
        # weights + resident KV (context+group scaled, floored at _KV_CAP for the short-ctx lean point)
        kv = max(_KV_CAP, _resident_kv_gb(params_b, 2048))
        assert u == max(0.10, min(0.45, (params_b * 2.0 + kv) / card))


def test_non_sleep_kv_scales_with_long_context():
    # vLLM's cache blocks must cover vllm_max_model_length, so a long-context resident run needs a
    # bigger KV budget than a short one -- the old flat _KV_CAP starved the cache blocks (Codex P2).
    short = colocate_kv_util(4.0, 2048, 80.0, sleep_mode=False)
    long = colocate_kv_util(4.0, 32768, 80.0, sleep_mode=False)
    assert long > short


def test_non_sleep_bigger_model_gets_a_bigger_budget():
    # the weight copy lives in the non-sleep budget too, so a bigger model gets more headroom.
    big = colocate_kv_util(4.0, 2048, 80.0, sleep_mode=False)
    small = colocate_kv_util(0.8, 2048, 80.0, sleep_mode=False)
    assert big > small


def test_caps_at_045_on_small_or_weight_heavy_configs():
    assert colocate_kv_util(4.0, 2048, 32.0, sleep_mode=True, num_generations=8) == 0.45  # small card
    assert colocate_kv_util(9.0, 2048, 80.0, sleep_mode=True, num_generations=8) == 0.45  # weight-heavy


def test_robust_to_missing_params_and_zero_context():
    u = colocate_kv_util(None, 0, 80.0, sleep_mode=True, num_generations=8)
    assert 0.0 < u <= 0.45


def test_moe_sizes_kv_on_active_backbone_not_total():
    """Cursor MsXcx: an MoE's resident KV pool must scale with the ACTIVE backbone (so the runtime
    colocate budget matches grpo_fits_resident's sleep-mode gate, which sizes its KV on active), while
    the bf16 weight copy stays on the FULL total. Keying KV off the 35B total would budget a bigger
    pool than the gate counted -> the gate could disable sleep while the engine reserves more KV than
    it sized (the near-margin 35B-A3B GRPO mismatch)."""
    card = 400.0  # large enough that the 0.45 cap doesn't mask the active-vs-total KV difference
    # 35B-A3B, non-sleep resident path: KV on the active 3B, weights on the full 35B.
    u_active = colocate_kv_util(35.0, 2048, card, sleep_mode=False, active_params_b=3.0)
    kv_active = max(_KV_CAP, _resident_kv_gb(3.0, 2048))
    assert u_active == max(0.10, min(0.45, (35.0 * 2.0 + kv_active) / card))
    # active unset -> the whole thing keys off total, giving a bigger KV (sqrt(35) vs sqrt(3)) ->
    # a bigger budget. The two diverging is exactly the resident-gate/runtime-budget mismatch.
    u_total = colocate_kv_util(35.0, 2048, card, sleep_mode=False)
    kv_total = max(_KV_CAP, _resident_kv_gb(35.0, 2048))
    assert u_total == max(0.10, min(0.45, (35.0 * 2.0 + kv_total) / card))
    assert u_active < u_total
    # dense (active unset/0) is unchanged: every term keys off the single params_b.
    assert colocate_kv_util(4.0, 2048, 80.0, sleep_mode=False, active_params_b=0.0) == colocate_kv_util(
        4.0, 2048, 80.0, sleep_mode=False
    )

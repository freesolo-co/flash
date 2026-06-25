"""CPU test for ``colocate_kv_util`` — the need-based vLLM KV-pool utilization for colocated GRPO.

``gpu_memory_utilization`` is vLLM's whole model-executor budget (weights + KV), so the helper budgets
both, scales the KV with context + the generation group, and caps the utilization at 0.45.
"""

from __future__ import annotations

from flash.engine.vram import _KV_CAP, colocate_kv_util


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


def test_non_sleep_path_unchanged():
    # Non-sleep keeps the existing resident-KV target (_KV_CAP), now via the named constant.
    assert colocate_kv_util(0.8, 2048, 80.0, sleep_mode=False) == min(0.45, _KV_CAP / 80.0)
    assert colocate_kv_util(2.0, 4096, 24.0, sleep_mode=False) == min(0.45, _KV_CAP / 24.0)


def test_caps_at_045_on_small_or_weight_heavy_configs():
    assert colocate_kv_util(4.0, 2048, 32.0, sleep_mode=True, num_generations=8) == 0.45  # small card
    assert colocate_kv_util(9.0, 2048, 80.0, sleep_mode=True, num_generations=8) == 0.45  # weight-heavy


def test_robust_to_missing_params_and_zero_context():
    u = colocate_kv_util(None, 0, 80.0, sleep_mode=True, num_generations=8)
    assert 0.0 < u <= 0.45

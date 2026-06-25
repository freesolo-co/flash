"""CPU test for ``colocate_kv_util`` — the need-based vLLM KV-pool utilization for colocated GRPO.

The blanket sleep-path 0.45 reserved ~36 GB of KV on an 80 GB A100 (the measured dominant allocation
at the GRPO step peak). The helper sizes it from flash's per-model KV estimate instead.
"""

from __future__ import annotations

from flash.engine.vram import _KV_CAP, _KV_COEF, colocate_kv_util


def test_sleep_path_frees_kv_on_large_cards():
    # 4B on an 80 GB A100: old blanket 0.45 (=36 GB) -> need-based 0.25 (=20 GB), freeing ~16 GB,
    # yet still 2.5x flash's ~8 GB KV estimate.
    u = colocate_kv_util(params_b=4.0, vllm_max_len=2048, total_vram_gb=80.0, sleep_mode=True)
    assert round(u, 2) == 0.25
    assert u < 0.45  # below the old blanket
    est_gb = min(_KV_COEF * (2048 / 1024) * (4.0**0.5), _KV_CAP)  # = 8 GB
    assert u * 80.0 >= 2.5 * est_gb  # generous margin over the estimate (20 >= 20)


def test_non_sleep_path_unchanged():
    # Non-sleep keeps the existing flat 8 GB target (KV stays resident through the backward).
    assert colocate_kv_util(0.8, 2048, 80.0, sleep_mode=False) == min(0.45, 8.0 / 80.0)
    assert colocate_kv_util(2.0, 4096, 24.0, sleep_mode=False) == min(0.45, 8.0 / 24.0)


def test_caps_at_045_on_small_cards():
    # On a small card the target exceeds the cap -> stays 0.45 (behaviour unchanged, no regression).
    assert colocate_kv_util(4.0, 2048, 32.0, sleep_mode=True) == 0.45  # 20/32 > 0.45
    assert colocate_kv_util(9.0, 4096, 40.0, sleep_mode=True) == 0.45  # 20/40 = 0.5 -> capped


def test_frees_more_on_bigger_cards():
    # 35B on an H200 (141 GB): ~20 GB target is a small fraction -> frees tens of GB vs 0.45 (=63 GB).
    u = colocate_kv_util(35.0, 4096, 141.0, sleep_mode=True)
    assert u < 0.20
    assert u * 141.0 >= 12.0  # still above the floor


def test_robust_to_missing_params_and_zero_context():
    # Defensive: no params / zero context never yields >0.45 or a sub-floor pool.
    u = colocate_kv_util(None, 0, 80.0, sleep_mode=True)
    assert 12.0 / 80.0 <= u <= 0.45

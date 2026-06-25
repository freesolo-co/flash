"""vLLM sleep-mode gate for colocated GRPO (CPU-only, no GPU/network).

Sleep mode (offload the rollout engine between steps) is only NEEDED when a GRPO run can't
fit RESIDENT on the card. On the large-model path the sleep/wake cycle stalls the colocated
rollout, so the worker skips sleep mode whenever the policy + rollout engine + training peak
fit on the live card. These cover the pure sizing logic that gate uses (the GPU trainer
wiring itself is exercised by the live smokes).
"""

from __future__ import annotations

from flash.engine.vram import estimate_vram_gb, grpo_fits_resident


def test_resident_peak_is_at_least_sleep_peak():
    # Without sleep offload the rollout engine stays resident through the backward, so the
    # peak is the SUM of the two phases, never less than the sleep-mode max().
    common = {"seq_len": 1024, "max_tokens": 64, "group_size": 8, "lora_rank": 32}
    for params_b in (0.9, 2.3, 4.7, 9.7):
        sleep = estimate_vram_gb(params_b, "grpo", "bf16", sleep_offload=True, **common)
        resident = estimate_vram_gb(params_b, "grpo", "bf16", sleep_offload=False, **common)
        assert resident >= sleep
    # sleep_offload only affects GRPO; SFT is unchanged.
    sft_a = estimate_vram_gb(4.7, "sft", "bf16", seq_len=1024, sleep_offload=True)
    sft_b = estimate_vram_gb(4.7, "sft", "bf16", seq_len=1024, sleep_offload=False)
    assert sft_a == sft_b


def test_4b_grpo_fits_resident_on_roomy_cards_not_tight_ones():
    # The 4B GRPO run (1024 ctx, group 8) fits resident on the cards the allocator sends it to
    # (A6000 48 GB / A100-H100 80 GB) -> sleep mode can be skipped; it does NOT fit a 24/32 GB
    # card -> sleep stays on there.
    kw = {"seq_len": 1024, "max_tokens": 64, "group_size": 8, "lora_rank": 32}
    assert grpo_fits_resident("Qwen/Qwen3.5-4B", card_vram_gb=80, **kw) is True
    assert grpo_fits_resident("Qwen/Qwen3.5-4B", card_vram_gb=48, **kw) is True
    assert grpo_fits_resident("Qwen/Qwen3.5-4B", card_vram_gb=32, **kw) is False
    assert grpo_fits_resident("Qwen/Qwen3.5-4B", card_vram_gb=24, **kw) is False


def test_fits_resident_is_conservative_when_unknown():
    # Unknown card VRAM (0) or an unlisted/open model (no catalog params) -> keep the safe
    # sleep default (return False), never disable sleep on a guess.
    assert grpo_fits_resident("Qwen/Qwen3.5-4B", card_vram_gb=0) is False
    assert grpo_fits_resident("some/unlisted-model", card_vram_gb=80) is False

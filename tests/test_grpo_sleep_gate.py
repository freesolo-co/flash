"""vLLM sleep-mode gate for colocated GRPO (CPU-only, no GPU/network).

Sleep mode (offload the rollout engine between steps) is only NEEDED when a GRPO run can't
fit RESIDENT on the card. On the large-model path the sleep/wake cycle stalls the colocated
rollout, so the worker skips sleep mode whenever the policy + rollout engine + training peak
fit on the live card. These cover the pure sizing logic that gate uses (the GPU trainer
wiring itself is exercised by the live smokes).
"""

from __future__ import annotations

from flash.engine.vram import estimate_vram_gb, grpo_fits_resident, grpo_rollout_seq_len


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


def test_rollout_seq_len_mirrors_run_rl_defaults():
    # When [train].max_length is unset, the gate must size to the engine context run_rl() launches
    # (max(1024, prompt+completion)), not a flat 1024 -- the Codex P2 fix.
    from flash.engine.recipe import RECIPE

    rl = RECIPE.rl
    assert grpo_rollout_seq_len(0) == max(1024, rl.max_prompt_len + rl.max_completion_len)
    assert grpo_rollout_seq_len(0, thinking=True) == max(
        1024, rl.max_prompt_len + rl.max_completion_len_thinking
    )
    assert grpo_rollout_seq_len(4096) == 4096  # explicit max_length wins
    assert grpo_rollout_seq_len(0, max_tokens=128) == max(1024, rl.max_prompt_len + 128)


def test_resident_estimate_sizes_to_real_default_not_1024():
    # The gate's resident estimate at the REAL default rollout length is >= the (too-small) 1024-token
    # estimate, so a marginal card is not wrongly told the run fits resident.
    kw = {"max_tokens": 64, "group_size": 8, "lora_rank": 32}
    big = estimate_vram_gb(
        4.7, "grpo", "bf16", seq_len=grpo_rollout_seq_len(0), sleep_offload=False, **kw
    )
    small = estimate_vram_gb(4.7, "grpo", "bf16", seq_len=1024, sleep_offload=False, **kw)
    assert big >= small


def test_resident_kv_uncapped_for_long_context():
    # The resident estimate's rollout KV must grow with context (vLLM holds it through the backward);
    # a 32k run estimates materially higher than a 1k run, so grpo_fits_resident won't admit a
    # long-context run that the flat-_KV_CAP estimate used to wave through.
    kw = {"max_tokens": 64, "group_size": 8, "sleep_offload": False}
    assert estimate_vram_gb(4.7, "grpo", "bf16", seq_len=32768, **kw) > estimate_vram_gb(
        4.7, "grpo", "bf16", seq_len=1024, **kw
    )


def test_fits_resident_is_conservative_when_unknown():
    # Unknown card VRAM (0) or an unlisted/open model (no catalog params) -> keep the safe
    # sleep default (return False), never disable sleep on a guess.
    assert grpo_fits_resident("Qwen/Qwen3.5-4B", card_vram_gb=0) is False
    assert grpo_fits_resident("some/unlisted-model", card_vram_gb=80) is False


def test_moe_grpo_fits_resident_sizes_compute_on_active_params():
    # Cursor High: grpo_fits_resident must size the resident peak's COMPUTE terms (KV pool,
    # activations, rank-linear LoRA) on the MoE's ~3B ACTIVE backbone — like model_required_vram_gb
    # does — not the 35B TOTAL. Keying them on the total inflates the resident estimate above the
    # 180 GB B200 (~186 GB w/ margin) and wrongly forces vLLM sleep mode on a B200 MoE GRPO run,
    # where the sleep/wake cycle stalls the colocated rollout — the very failure the gate prevents.
    from flash.catalog import MODELS, vocab_size_for

    moe = "Qwen/Qwen3.6-35B-A3B"
    info = MODELS[moe]
    assert info.active_params_b  # it's an MoE...
    assert info.active_params_b < info.params_b  # ...with active < total
    kw = {"seq_len": 1024, "max_tokens": 64, "group_size": 8, "lora_rank": 32}
    # active-aware (the fix) admits the run resident on the 180 GB B200; the 141 GB H200 still can't
    # hold the two ~70 GB weight copies, so sleep stays on there.
    assert grpo_fits_resident(moe, card_vram_gb=180, **kw) is True
    assert grpo_fits_resident(moe, card_vram_gb=141, **kw) is False
    # Had the gate kept sizing compute on the 35B total, the 1.15-margined resident estimate would
    # exceed 180 and the B200 case above would be False. Prove the active-aware estimate is materially
    # leaner than the (buggy) total-based one — that gap is what flips the B200 verdict.
    active_aware = estimate_vram_gb(
        info.params_b, "grpo", "bf16", sleep_offload=False,
        active_params_b=info.active_params_b, vocab=vocab_size_for(moe), **kw,
    )
    total_based = estimate_vram_gb(
        info.params_b, "grpo", "bf16", sleep_offload=False,
        active_params_b=None, vocab=vocab_size_for(moe), **kw,
    )
    assert active_aware < total_based
    assert active_aware * 1.15 <= 180 < total_based * 1.15  # only the active-aware fit clears the B200


def test_sleep_gate_resolves_unset_max_length_against_real_rollout_length():
    # Cursor High / Codex P2: a sub-3B model with [train].max_length UNSET (0) but a real rollout
    # (max_tokens) must NOT short-circuit the size/context pre-filter as a 0-length "short" run --
    # the effective rollout (~2112 tokens here, >= the 2048 long-context threshold) makes the gate
    # reach the resident-fit check. Before the fix `_memory_mode(model, 0)` was False for a sub-3B
    # model, so grpo_sleep_mode returned False (sleep OFF) on EVERY card, even one too small to fit
    # the run resident -> OOM risk on the real long rollout.
    from flash.engine.worker.perf import grpo_sleep_mode

    model = "Qwen/Qwen3.5-2B"  # sub-3B: the large-model Liger default does not mask the bug
    # tight card: the run doesn't fit resident -> sleep ON (the regression the old gate missed)
    assert grpo_sleep_mode(model, max_length=0, max_tokens=64, card_vram_gb=16) is True
    # roomy card: fits resident -> sleep OFF (skip the slow/buggy sleep-wake cycle)
    assert grpo_sleep_mode(model, max_length=0, max_tokens=64, card_vram_gb=80) is False
    # unknown card VRAM -> fall back to the size/context gate, which now sees the real long rollout
    assert grpo_sleep_mode(model, max_length=0, max_tokens=64, card_vram_gb=0) is True

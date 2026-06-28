"""vLLM sleep-mode gate for colocated GRPO (CPU-only, no GPU/network).

Sleep mode (offload the rollout engine between steps) is only NEEDED when a GRPO run can't
fit RESIDENT on the card. On the large-model path the sleep/wake cycle stalls the colocated
rollout, so the worker skips sleep mode whenever the policy + rollout engine + training peak
fit on the live card. These cover the pure sizing logic that gate uses (the GPU trainer
wiring itself is exercised by the live smokes).
"""

from __future__ import annotations

from flash.engine.vram import estimate_vram_gb, grpo_fits_resident, grpo_rollout_seq_len


def test_fp8_kv_unlocks_longer_resident_context():
    # fp8 KV halves the resident KV bytes, so the resident-fit gate (which sizes the KV) must admit a
    # longer context than the bf16 sizing would. The 35B-A3B on a 180 GB B200 at group 8 fits ctx 2048
    # either way, but ctx 4096 fits ONLY with fp8 KV factored (else it's wrongly routed to the broken
    # sleep path). Without this awareness the gate over-reserves bf16 KV and rejects a run fp8 KV fits.
    mid = "Qwen/Qwen3.6-35B-A3B"

    def f(ctx, fp8):
        return grpo_fits_resident(
            mid, seq_len=ctx, max_tokens=384, lora_rank=16, group_size=8, card_vram_gb=179.06, fp8_kv=fp8
        )

    assert f(2048, False)  # short ctx fits either way
    assert f(2048, True)
    assert not f(4096, False), "bf16-KV sizing should (conservatively) reject group8 ctx4096"
    assert f(4096, True), "fp8 KV halves the pool -> group8 ctx4096 fits resident"


def test_sleep_unsupported_model_rejects_instead_of_sleeping():
    # The 35B-A3B is resident-only (vLLM sleep HANGS its wake). A config that doesn't fit resident must
    # be REJECTED -- not routed to the hanging sleep path. (a) the worker gate RAISES rather than
    # returning True; (b) it still routes a fitting config resident (returns False, no raise).
    import pytest

    from flash.engine.worker.perf import grpo_sleep_mode

    mid = "Qwen/Qwen3.6-35B-A3B"
    # Fits resident (group 8, ctx 4096 with fp8 KV) -> resident, no raise.
    assert (
        grpo_sleep_mode(mid, max_length=4096, group_size=8, max_tokens=384, lora_rank=16, card_vram_gb=179.06, fp8_kv=True)
        is False
    )
    # Too long to fit resident -> hard reject (NOT True / sleep).
    with pytest.raises(ValueError, match="sleep mode HANGS"):
        grpo_sleep_mode(mid, max_length=8192, group_size=8, max_tokens=384, lora_rank=16, card_vram_gb=179.06, fp8_kv=True)


def test_sleep_unsupported_model_rejected_at_parse_time_for_long_context():
    # Submit-time: a sleep-broken model is sized on its RESIDENT peak, so a context too long to fit
    # resident pushes the requirement PAST the biggest GPU (180 GB B200) -> rejected before launch.
    from flash.engine.vram import model_required_vram_gb

    mid = "Qwen/Qwen3.6-35B-A3B"
    fits = {"max_length": 4096, "group_size": 8, "max_tokens": 384, "lora_rank": 16}
    over = {"max_length": 8192, "group_size": 8, "max_tokens": 384, "lora_rank": 16}
    assert model_required_vram_gb(mid, "grpo", train=fits) <= 180  # admitted on the B200
    assert model_required_vram_gb(mid, "grpo", train=over) > 180  # past every GPU -> rejected


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


def test_moe_resident_fit_sizes_activations_on_active_params():
    # Codex P2 (Mrkfb): the 35B-A3B MoE resident-fit estimate must size activations/KV on the ACTIVE
    # backbone (~3B), not the dense 35B total, so a B200 that genuinely fits the run resident (full
    # 140 GB of weights + active-sized activations/KV under the margin) is admitted resident instead
    # of being pinned to the stall-prone sleep/wake path. The weights term stays on the full 35B, so
    # this never under-counts the two resident weight copies.
    kw = {"seq_len": 2048, "max_tokens": 384, "group_size": 8, "lora_rank": 32}
    moe = "Qwen/Qwen3.6-35B-A3B"
    # active-sized estimate fits a 180 GB B200 resident; the same gate would NOT have, sizing as 35B.
    assert grpo_fits_resident(moe, card_vram_gb=180, **kw) is True
    # a too-small card still keeps sleep mode on (the active sizing only shrinks activations/KV, not
    # the full 140 GB of resident weights, so a 80 GB card cannot hold the two bf16 copies).
    assert grpo_fits_resident(moe, card_vram_gb=80, **kw) is False
    # the active-sized resident estimate is strictly below a (wrong) dense-35B estimate of the same run
    active = estimate_vram_gb(35.0, "grpo", "bf16", sleep_offload=False, active_params_b=3.0, **kw)
    dense = estimate_vram_gb(35.0, "grpo", "bf16", sleep_offload=False, **kw)
    assert active < dense


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


def test_finalize_alloc_conf_passes_fp8_kv_in_parity_with_run_rl():
    # finalize_alloc_conf_for_sleep documents that it resolves the sleep decision "EXACTLY as run_rl
    # does". run_rl threads fp8_kv (device capability >= (8, 9)) into grpo_sleep_mode so the
    # resident-fit gate admits the longer context fp8 KV unlocks; finalize MUST pass it too, or a
    # long-context MoE that fits resident ONLY with fp8 KV resolves sleep ON at boot (keeping the
    # non-expandable alloc conf) while training runs sleep OFF -- the very divergence this guards.
    # AST-based (the called keyword is the invariant) so it survives reformatting.
    import ast
    import inspect

    import flash.engine.worker as w
    from flash.engine.worker import gpu_setup

    def _sleep_call_keywords(fn) -> set[str] | None:
        tree = ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name == "grpo_sleep_mode":
                return {kw.arg for kw in node.keywords}
        return None

    rl_kws = _sleep_call_keywords(w.run_rl)
    assert rl_kws is not None, "run_rl no longer calls grpo_sleep_mode"
    assert "fp8_kv" in rl_kws, "run_rl must pass fp8_kv to grpo_sleep_mode"
    fin_kws = _sleep_call_keywords(gpu_setup.finalize_alloc_conf_for_sleep)
    assert fin_kws is not None, "finalize_alloc_conf_for_sleep no longer calls grpo_sleep_mode"
    assert "fp8_kv" in fin_kws, (
        "finalize_alloc_conf_for_sleep must thread fp8_kv into grpo_sleep_mode (parity with run_rl) -- "
        "else the boot-time alloc-conf sleep decision diverges from the trainer's for an fp8-KV-only "
        "resident fit"
    )

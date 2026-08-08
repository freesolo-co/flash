"""Resident-fit sizing for colocated GRPO (CPU-only, no GPU/network).

A GRPO run holds the policy, the rollout engine and the training peak on one card. These cover the
pure sizing logic that decides whether a given config fits there -- which is what the parse-time
gate uses to reject a run before renting a GPU, and what pins a sleep_unsupported model resident.
"""

from __future__ import annotations

import pytest

from flash.engine.plan.vram import estimate_vram_gb, grpo_fits_resident, grpo_rollout_seq_len


def test_fp8_kv_unlocks_longer_resident_context():
    # Without this awareness the gate over-reserves bf16 KV and rejects a run fp8 KV fits. the
    # budget is load-bearing and deliberately NOT a round multiple of a card: training the
    # routed experts pushed this run past one card, and only ~190-195 GB still separates bf16
    # from fp8 here. a full two-card 358 GB budget admits every case above, which would make
    # this test unfailable.
    mid = "Qwen/Qwen3.6-35B-A3B"

    def f(ctx, fp8):
        return grpo_fits_resident(
            mid,
            seq_len=ctx,
            max_tokens=384,
            lora_rank=16,
            group_size=8,
            card_vram_gb=190.0,
            fp8_kv=fp8,
        )

    assert f(2048, False)  # short ctx fits either way
    assert f(2048, True)
    assert not f(4096, False), "bf16-KV sizing should (conservatively) reject group8 ctx4096"
    assert f(4096, True), "fp8 KV halves the pool -> group8 ctx4096 fits resident"


def test_sleep_unsupported_model_is_sized_resident_not_slept():
    # The 35B-A3B is resident-only (vLLM sleep HANGS its wake), so the fit question for it is
    # always "does it fit RESIDENT" -- there is no sleeping fallback to fall back to. group 8
    # / ctx 4096 fits with fp8 KV; doubling the context does not, and the parse-time gate
    # below turns that into a rejection rather than a run that hangs on its first wake. same
    # load-bearing budget as the fp8 test above: a full two-card 358 GB budget fits ctx 8192
    # too, which would erase the ceiling this test exists to prove.
    mid = "Qwen/Qwen3.6-35B-A3B"

    def fits(ctx):
        return grpo_fits_resident(
            mid,
            seq_len=ctx,
            max_tokens=384,
            lora_rank=16,
            group_size=8,
            card_vram_gb=190.0,
            fp8_kv=True,
        )

    assert fits(4096)
    assert not fits(8192)


def test_sleep_unsupported_model_rejected_at_parse_time_for_long_context():
    # Submit-time: a sleep-broken model is sized on its RESIDENT peak, so a longer context
    # pushes the requirement past the allocation it would otherwise get -> rejected before
    # launch. the 195 GB bound is between the two sized values (190 and 200) on purpose.
    # training the routed experts moved both past one 180 GB B200, and a full two-card 360 GB
    # bound would sit above BOTH, so the test would pass no matter how badly the long-context
    # case were sized.
    from flash.engine.plan.vram import model_required_vram_gb

    mid = "Qwen/Qwen3.6-35B-A3B"
    fits = {
        "max_context_tokens": 4096,
        "group_size": 8,
        "max_completion_tokens": 384,
        "lora_rank": 16,
    }
    over = {
        "max_context_tokens": 8192,
        "group_size": 8,
        "max_completion_tokens": 384,
        "lora_rank": 16,
    }
    assert model_required_vram_gb(mid, "grpo", train=fits) <= 195  # admitted
    assert model_required_vram_gb(mid, "grpo", train=over) > 195  # sized past it -> rejected


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
    # The 4B GRPO run (1024 ctx, group 8) fits resident on a roomy card (>=48 GB; the allocator
    # now sends it to the 80 GB A100-H100 tier) -> sleep mode can be skipped; it does NOT fit a
    # 24/32 GB card -> sleep stays on there.
    kw = {"seq_len": 1024, "max_tokens": 64, "group_size": 8, "lora_rank": 32}
    assert grpo_fits_resident("Qwen/Qwen3.5-4B", card_vram_gb=80, **kw) is True
    assert grpo_fits_resident("Qwen/Qwen3.5-4B", card_vram_gb=48, **kw) is True
    assert grpo_fits_resident("Qwen/Qwen3.5-4B", card_vram_gb=32, **kw) is False
    assert grpo_fits_resident("Qwen/Qwen3.5-4B", card_vram_gb=24, **kw) is False


def test_moe_resident_fit_sizes_activations_on_active_params():
    # the 35B-A3B MoE resident-fit estimate must size activations/KV on the ACTIVE backbone (~3B),
    # not the dense 35B total, so a B200 that genuinely fits the run resident (full 140 GB of
    # weights + active-sized activations/KV under the margin) is admitted resident instead of being
    # pinned to the stall-prone sleep/wake path. The weights term stays on the full 35B, so this
    # never under-counts the two resident weight copies.
    kw = {"seq_len": 2048, "max_tokens": 384, "group_size": 8, "lora_rank": 32}
    moe = "Qwen/Qwen3.6-35B-A3B"
    # active-sized estimate fits resident on two B200s; the same gate would NOT have, sizing as 35B.
    assert grpo_fits_resident(moe, card_vram_gb=2 * 180, **kw) is True
    # a too-small card still keeps sleep mode on (the active sizing only shrinks activations/KV, not
    # the full 140 GB of resident weights, so a 80 GB card cannot hold the two bf16 copies).
    assert grpo_fits_resident(moe, card_vram_gb=80, **kw) is False
    # the active-sized resident estimate is strictly below a (wrong) dense-35B estimate of the same run
    active = estimate_vram_gb(35.0, "grpo", "bf16", sleep_offload=False, active_params_b=3.0, **kw)
    dense = estimate_vram_gb(35.0, "grpo", "bf16", sleep_offload=False, **kw)
    assert active < dense


def test_rollout_seq_len_mirrors_run_rl_defaults():
    # When [train].max_context_tokens is unset, the gate must size to the engine context run_rl() launches
    # (max(1024, prompt+completion)), not a flat 1024 -- the Codex P2 fix.
    from flash.engine.plan.recipe import RECIPE

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
    # vLLM holds rollout KV through backward, so resident sizing must grow with context and reject a
    # 32k run that does not fit.
    kw = {"max_tokens": 64, "group_size": 8, "sleep_offload": False}
    assert estimate_vram_gb(4.7, "grpo", "bf16", seq_len=32768, **kw) > estimate_vram_gb(
        4.7, "grpo", "bf16", seq_len=1024, **kw
    )


def test_pinned_revision_is_sized_on_the_revisions_own_geometry(monkeypatch):
    # a pinned commit must be sized on the weights the worker will actually load, not on the
    # catalog's default-revision numbers. otherwise a revision that grew (more params, wider vocab)
    # is admitted onto a card the run cannot fit, and the resident-fit verdict is made against a
    # model that was never loaded.
    import flash.engine.plan.vram as vram_mod

    seen = {}

    def _capture(model_id, revision, info):
        seen["model_id"] = model_id
        seen["revision"] = revision
        return 400.0, 200_000  # a far larger model than the catalog entry

    monkeypatch.setattr(vram_mod, "_validated_revision_geometry", _capture)

    pinned = vram_mod.model_required_vram_gb("Qwen/Qwen3.5-4B", "grpo", model_revision="a" * 40)
    assert seen["revision"] == "a" * 40, "the pinned revision never reached the sizing path"
    # and the substituted geometry actually moved the answer -- otherwise the assert above would
    # pass even if the result were computed from the catalog numbers regardless.
    assert pinned > vram_mod.model_required_vram_gb("Qwen/Qwen3.5-4B", "grpo")


def test_fits_resident_is_conservative_when_unknown():
    # Unknown card VRAM (0) or an id the catalog does not list (no params) -> keep the safe
    # sleep default (return False), never disable sleep on a guess.
    assert grpo_fits_resident("Qwen/Qwen3.5-4B", card_vram_gb=0) is False
    assert grpo_fits_resident("some/unlisted-model", card_vram_gb=80) is False


def test_moe_grpo_fits_resident_sizes_compute_on_active_params():
    # grpo_fits_resident must size the resident peak's COMPUTE terms (KV pool, activations,
    # rank-linear LoRA) on the MoE's ~3B ACTIVE backbone — like model_required_vram_gb does — not
    # the 35B TOTAL. Keying them on the total inflates the resident estimate above the 180 GB B200
    # (~186 GB w/ margin) and wrongly forces vLLM sleep mode on a B200 MoE GRPO run, where the
    # sleep/wake cycle stalls the colocated rollout — the very failure the gate prevents.
    from flash.core.catalog import MODELS, vocab_size_for

    moe = "Qwen/Qwen3.6-35B-A3B"
    info = MODELS[moe]
    assert info.active_params_b  # it's an MoE...
    assert info.active_params_b < info.params_b  # ...with active < total
    kw = {"seq_len": 1024, "max_tokens": 64, "group_size": 8, "lora_rank": 32}
    # active-aware (the fix) admits the run resident on two 180 GB B200s; one card no longer holds it
    # now that the routed experts are trained, and the 141 GB H200 never could.
    assert grpo_fits_resident(moe, card_vram_gb=2 * 180, **kw) is True
    assert grpo_fits_resident(moe, card_vram_gb=141, **kw) is False
    # the weight terms now decide the card count, so prove the active-aware/total-based gap directly
    # rather than through the fit verdict: 180 still lies between the two margined estimates, so this
    # stays a real discriminator. sizing compute on the 35B total would push it over that line.
    active_aware = estimate_vram_gb(
        info.params_b,
        "grpo",
        "bf16",
        sleep_offload=False,
        active_params_b=info.active_params_b,
        vocab=vocab_size_for(moe),
        **kw,
    )
    total_based = estimate_vram_gb(
        info.params_b,
        "grpo",
        "bf16",
        sleep_offload=False,
        active_params_b=None,
        vocab=vocab_size_for(moe),
        **kw,
    )
    assert active_aware < total_based
    assert (
        active_aware * 1.15 <= 180 < total_based * 1.15
    )  # only the active-aware fit clears the B200


def test_unset_max_length_still_resolves_the_real_rollout_length():
    # [train].max_context_tokens UNSET (0) does not mean a 0-length run. the rollout still generates
    # max_tokens, so the effective sequence length must be derived from it -- sizing a run at 0
    # would treat a real long rollout as free and admit it onto a card that cannot hold its KV.
    assert grpo_rollout_seq_len(0, 64, False) >= 2048
    # and an explicit context wins over the derived floor when it is the larger of the two.
    assert grpo_rollout_seq_len(8192, 64, False) >= 8192


# ---------------------------------------------------------------------------
# expandable_segments vs verl's rollout allocator. verl's GRPO and OPD trainers both run a vLLM
# rollout and leave rollout.enable_sleep_mode defaulted True, so the engine ALWAYS builds a
# CuMemAllocator -- and CuMemAllocator asserts outright on "expandable_segments:True"
# (vllm/device_allocator/cumem.py:132, pytorch#147851). The launcher's per-algorithm choice is now
# the only route to that conf, so it carries the whole invariant.
# ---------------------------------------------------------------------------
def _worker_runtime_for(algorithm: str) -> dict[str, str] | None:
    if algorithm != "opd":
        return None
    return {
        "FLASH_CONTROL_PANEL_URL": "https://broker.example",
        "FLASH_TEACHER_CAPABILITY": "capability-test-value",
    }


def _worker_env_for(algorithm: str, phase: str) -> dict:
    from flash.core.spec import JobSpec
    from flash.providers._lifecycle.worker import build_worker_env

    spec = JobSpec.from_dict({"model": "m", "seed": 0, "algorithm": algorithm})
    assert spec.phase == phase, "phase is derived from algorithm; the mapping moved"
    return build_worker_env(spec, 0, runtime_secrets=_worker_runtime_for(algorithm))


@pytest.mark.parametrize(("algorithm", "phase"), [("grpo", "rl"), ("opd", "opd")])
def test_verl_rollout_never_gets_expandable_segments(algorithm, phase):
    """GRPO and OPD are the algorithms that build a vLLM rollout, so they are the ones whose
    CuMemAllocator asserts on expandable_segments. SFT is covered below: it has no rollout and must
    KEEP the expandable conf."""
    env = _worker_env_for(algorithm, phase)
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        assert "expandable_segments" not in env[key], (
            f"verl {algorithm} would build a CuMemAllocator against {env[key]}"
        )


def test_verl_sft_keeps_the_expandable_allocator():
    """verl.trainer.sft_trainer is a pure FSDP trainer: no rollout, no vLLM engine, no CuMemAllocator.
    so the carve-out must not reach it -- switching SFT to the non-expandable conf would regress the
    large-tensor SFT path onto a fragmentation-prone allocator and OOM jobs that fit today. the
    predicate is "generates through verl", not "runs on verl"."""
    assert _worker_env_for("sft", "sft")["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


@pytest.mark.parametrize("stale_backend", ["trl", "verl", "bogus"])
def test_a_stale_backend_key_cannot_change_the_allocator(stale_backend):
    """[worker_env] no longer selects a backend: every phase delegates to verl unconditionally. A
    stale key left in a config must therefore be inert -- honoring one would pick an allocator for a
    trainer that cannot run, and "trl" on grpo/opd would hand verl's rollout the expandable conf that
    trips the CuMemAllocator assert before step 1."""
    from flash.core.spec import JobSpec
    from flash.providers._lifecycle.worker import build_worker_env

    for algorithm, phase, expect_expandable in (
        ("grpo", "rl", False),
        ("opd", "opd", False),
        ("sft", "sft", True),
    ):
        spec = JobSpec.from_dict(
            {
                "model": "m",
                "seed": 0,
                "algorithm": algorithm,
                "worker_env": {f"FLASH_{phase.upper()}_BACKEND": stale_backend},
            }
        )
        env = build_worker_env(
            spec,
            0,
            runtime_secrets=_worker_runtime_for(algorithm),
        )
        for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            assert ("expandable_segments" in env[key]) is expect_expandable, (
                f"{algorithm} allocator moved under a stale {stale_backend!r} key"
            )

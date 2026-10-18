"""Resident-fit sizing for colocated GRPO (CPU-only, no GPU/network).

A GRPO run holds the policy, the rollout engine and the training peak on one card. These cover the
pure sizing logic that decides whether a given config fits there -- which is what the parse-time
gate uses to reject a run before renting a GPU, and what pins a sleep_unsupported model resident.
"""

from __future__ import annotations

import pytest

from flash.engine.plan.vram import estimate_vram_gb, grpo_rollout_seq_len


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

    pinned = vram_mod.model_required_vram_gb("Qwen/Qwen3.5-9B", "grpo", model_revision="a" * 40)
    assert seen["revision"] == "a" * 40, "the pinned revision never reached the sizing path"
    # and the substituted geometry actually moved the answer -- otherwise the assert above would
    # pass even if the result were computed from the catalog numbers regardless.
    assert pinned > vram_mod.model_required_vram_gb("Qwen/Qwen3.5-9B", "grpo")


def test_moe_grpo_resident_estimate_sizes_compute_on_active_params():
    from flash.core.catalog import MODELS, vocab_size_for

    moe = "Qwen/Qwen3.6-35B-A3B"
    info = MODELS[moe]
    assert info.active_params_b
    assert info.active_params_b < info.params_b
    kw = {"seq_len": 1024, "max_tokens": 64, "group_size": 8, "lora_rank": 32}
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
    assert active_aware * 1.15 <= 180 < total_based * 1.15


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
        "FLASH_PUBLIC_URL": "https://broker.example",
        "FLASH_TEACHER_CAPABILITY": "capability-test-value",
    }


def _built_env_for(algorithm: str, phase: str) -> dict:
    from flash.core.spec import JobSpec
    from flash.providers._lifecycle.net.worker import build_worker_env

    spec = JobSpec.from_dict({"model": "m", "seed": 0, "algorithm": algorithm})
    assert spec.phase == phase, "phase is derived from algorithm; the mapping moved"
    return build_worker_env(spec, runtime_secrets=_worker_runtime_for(algorithm))


@pytest.mark.parametrize(("algorithm", "phase"), [("grpo", "rl"), ("opd", "opd")])
def test_verl_rollout_never_gets_expandable_segments(algorithm, phase):
    """GRPO and OPD are the algorithms that build a vLLM rollout, so they are the ones whose
    CuMemAllocator asserts on expandable_segments. SFT is covered below: it has no rollout and must
    KEEP the expandable conf."""
    env = _built_env_for(algorithm, phase)
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        assert "expandable_segments" not in env[key], (
            f"verl {algorithm} would build a CuMemAllocator against {env[key]}"
        )


def test_verl_sft_keeps_the_expandable_allocator():
    """verl.trainer.sft_trainer is a pure FSDP trainer: no rollout, no vLLM engine, no CuMemAllocator.
    so the carve-out must not reach it -- switching SFT to the non-expandable conf would regress the
    large-tensor SFT path onto a fragmentation-prone allocator and OOM jobs that fit today. the
    predicate is "generates through verl", not "runs on verl"."""
    assert _built_env_for("sft", "sft")["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"

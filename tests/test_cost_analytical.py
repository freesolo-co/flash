"""Cost estimator: the analytical ground-truth model. Qualitative invariants (cost scales with
steps, GRPO > SFT, bigger model costs more, the wall cap bounds runs) + arithmetic consistency."""

from __future__ import annotations

import pytest

from flash.cost import RunConfig, estimate_cost
from flash.cost.analytical import DEFAULT_WALL_CAP_S, seconds_per_step, select_gpu, setup_seconds

SMALL = "Qwen/Qwen3.5-0.8B"
MID = "Qwen/Qwen3.5-4B"
BIG = "Qwen/Qwen3.5-9B"


def test_estimate_is_positive_and_self_consistent():
    e = estimate_cost(RunConfig(MID, "sft", 200))
    assert e.total_usd > 0
    assert e.wall_clock_seconds == pytest.approx(e.setup_seconds + e.train_seconds)
    # total = wall-clock hours x hourly rate
    assert e.total_usd == pytest.approx(e.wall_clock_hours * e.gpu_hourly_usd)
    # chosen card actually fits the run's requirement
    assert e.gpu_vram_gb >= e.required_vram_gb


def test_cost_increases_with_steps():
    costs = [estimate_cost(RunConfig(SMALL, "sft", s)).total_usd for s in (100, 500, 1000)]
    assert costs[0] < costs[1] < costs[2]


def test_grpo_costs_more_than_sft():
    gpu = "RTX 5090"
    # Per-step: GRPO (rollout + update) dominates SFT on the same card.
    assert seconds_per_step(RunConfig(MID, "grpo", 1), gpu) > seconds_per_step(
        RunConfig(MID, "sft", 1), gpu
    )
    # End to end (each on its own chosen card): GRPO is the costlier run.
    sft = estimate_cost(RunConfig(MID, "sft", 150))
    grpo = estimate_cost(RunConfig(MID, "grpo", 150))
    assert grpo.total_usd > sft.total_usd


def test_bigger_model_costs_more_per_step():
    gpu = "RTX 5090"
    small = seconds_per_step(RunConfig(SMALL, "sft", 1), gpu)
    big = seconds_per_step(RunConfig(BIG, "sft", 1), gpu)
    assert big > small


def test_grpo_requires_at_least_as_much_vram_as_sft():
    for model in (SMALL, MID, BIG):
        _, sft_need = select_gpu(RunConfig(model, "sft", 100))
        _, grpo_need = select_gpu(RunConfig(model, "grpo", 100))
        assert grpo_need >= sft_need


def test_omitted_sft_batch_sizes_like_the_real_allocator():
    # An omitted SFT batch sizes VRAM like the allocator (micro-batch 4), not the recipe batch (32).
    from flash.providers.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "sft", 100)  # batch_size omitted
    _, need = select_gpu(cfg)
    real = alloc_required_vram_gb(MID, "sft", train={}, thinking=False)
    assert need == real
    assert "batch_size" not in cfg.train_knobs()  # omitted batch isn't forwarded (would inflate)


def test_explicit_sft_batch_is_still_forwarded_for_sizing():
    # An EXPLICIT batch_size is honored for VRAM sizing (and matches the allocator at that
    # batch), so a deliberately large batch still routes to the right card.
    from flash.providers.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "sft", 100, batch_size=32)
    _, need = select_gpu(cfg)
    real = alloc_required_vram_gb(MID, "sft", train={"batch_size": 32}, thinking=False)
    assert cfg.train_knobs()["batch_size"] == 32
    assert need == real


def test_grpo_colocate_routes_4b_to_a_bigger_card_than_sft():
    # 4.7B SFT fits a 32 GB 5090; GRPO's 2nd weight copy + logits push it past 32 GB.
    sft = estimate_cost(RunConfig(MID, "sft", 100))
    grpo = estimate_cost(RunConfig(MID, "grpo", 100))
    assert grpo.required_vram_gb > sft.required_vram_gb
    assert grpo.gpu_vram_gb >= grpo.required_vram_gb


def test_setup_grpo_exceeds_sft_and_scales_with_model_size():
    # GRPO pays an extra vLLM-init cost; bigger models download longer.
    assert setup_seconds(RunConfig(MID, "grpo", 1)) > setup_seconds(RunConfig(MID, "sft", 1))
    assert setup_seconds(RunConfig(BIG, "sft", 1)) > setup_seconds(RunConfig(SMALL, "sft", 1))


def test_cold_start_calibrated_to_real_short_sft_run():
    # Calibration anchor: a real fresh-worker run (0.8B SFT, 391 examples -> 26 priced steps at
    # the recipe batch) was cold-start-dominated (a fresh worker spent ~12.5 min in model load).
    # Static pricing picks the cheapest fitting class; the cheapest managed card is the 48 GB
    # RTX A6000 ($0.49). 26 = ceil(391 / 32) * 2 epochs.
    e = estimate_cost(RunConfig(SMALL, "sft", 26))
    assert e.gpu == "RTX A6000"
    assert e.gpu_hourly_usd == pytest.approx(0.49, abs=1e-3)
    assert e.total_usd == pytest.approx(0.0773, rel=0.10)
    # Model load (not boot/deps) is the dominant cold-start term for a short job.
    assert e.setup_seconds > e.train_seconds  # cold start dominates this short run


def test_cold_start_negligible_for_long_runs():
    # The bigger cold start must NOT regress long runs: when training wall dominates, setup is a
    # small single-digit fraction of the bill.
    e = estimate_cost(RunConfig(SMALL, "sft", 5000))
    assert e.setup_seconds / e.wall_clock_seconds < 0.05


def test_wall_clock_cap_bounds_runaway_runs():
    e = estimate_cost(RunConfig(BIG, "grpo", 100_000))
    assert e.wall_capped is True
    assert e.wall_clock_seconds == pytest.approx(DEFAULT_WALL_CAP_S)
    # The cap shows up in the notes and bounds the bill.
    assert any("cap" in n.lower() for n in e.notes)
    uncapped_like = estimate_cost(RunConfig(BIG, "grpo", 100_000), wall_cap_s=10**9)
    assert uncapped_like.total_usd > e.total_usd


def test_config_max_wall_seconds_overrides_default_cap():
    # A spec-pinned per-run wall cap (gpu.max_wall_seconds) is honored instead of the 24h
    # default, so a run with a short explicit cap isn't priced against the 24h ceiling.
    uncapped = estimate_cost(RunConfig(BIG, "grpo", 100_000))
    assert uncapped.wall_capped is True  # binds the 24h default
    short = estimate_cost(RunConfig(BIG, "grpo", 100_000, max_wall_seconds=3600))
    assert short.wall_capped is True
    assert short.wall_clock_seconds == pytest.approx(3600.0)
    assert short.total_usd < uncapped.total_usd


def test_wall_cap_is_applied_per_seed_not_to_the_aggregate():
    # The cap is per-seed, so a runaway 3-seed run bills 3x the per-seed cap, not one aggregate cap.
    cfg = RunConfig(BIG, "grpo", 300_000, setup_repeats=3, max_wall_seconds=3600)
    e = estimate_cost(cfg)
    assert e.wall_capped is True
    assert e.wall_clock_seconds == pytest.approx(3 * 3600.0)


def test_sub_60s_wall_cap_is_floored_to_the_runner_minimum():
    # The runner floors the cap to max(60, ...); the estimate mirrors it (else a ~$0 quote).
    e10 = estimate_cost(RunConfig(BIG, "grpo", 100_000, max_wall_seconds=10))
    e30 = estimate_cost(RunConfig(BIG, "grpo", 100_000, max_wall_seconds=30))
    assert e10.wall_clock_seconds == pytest.approx(60.0)
    assert e30.wall_clock_seconds == pytest.approx(60.0)
    assert e10.total_usd > 0.0


def test_nonpositive_max_wall_seconds_is_accepted_and_floored():
    # A 0/negative cap is accepted (the runner floors it to 60s), not rejected -- so --cost can
    # price configs the runner accepts; estimate_cost mirrors the 60s floor -> positive quote.
    for cap in (0, -5):
        cfg = RunConfig(BIG, "grpo", 100_000, max_wall_seconds=cap)
        assert cfg.max_wall_seconds == cap
        e = estimate_cost(cfg)
        assert e.wall_clock_seconds == pytest.approx(60.0)
        assert e.total_usd > 0.0
    one = RunConfig(BIG, "grpo", 100_000, setup_repeats=1, max_wall_seconds=3600)
    assert estimate_cost(one).wall_clock_seconds == pytest.approx(3600.0)


def test_select_gpu_picks_cheapest_including_unvalidated():
    # No validation gate: select_gpu picks the cheapest fitting class (validated or not) at the
    # static rate, and nothing fitting is cheaper.
    from flash.cost.facts import gpu_hourly_usd, pick_gpu
    from flash.providers.base import GPU_INFO

    gpu, need = select_gpu(RunConfig(MID, "sft", 100))
    assert gpu == pick_gpu(need)
    cheaper = [
        g for g in GPU_INFO.values()
        if g.vram_gb >= need and gpu_hourly_usd(g.name) < gpu_hourly_usd(gpu)
    ]
    assert not cheaper, f"{cheaper} cheaper than {gpu} for {need} GB"


def test_9b_bf16_grpo_needs_an_80gb_class():
    # Qwen3.5-9B is bf16 (QLoRA was dropped: the 4-bit vLLM-rollout merge collapsed the GRPO
    # importance-sampling ratio -> no learning). Colocated bf16 GRPO needs an 80 GB-class card.
    e = estimate_cost(RunConfig(BIG, "grpo", 100))
    assert e.required_vram_gb >= 80
    assert e.gpu_vram_gb >= 80


def test_35b_moe_long_context_grpo_sized_past_the_b200():
    # The 35B MoE colocated GRPO holds two ~70 GB weight copies + a KV pool on ONE 180 GB B200, so a
    # long rollout context overflows it. The grpo_min_vram_gb floor engages the seq escalation, keyed
    # on the ~3B ACTIVE params (not the 35B total): default/moderate GRPO stays on the B200, but a
    # 32k-token run is sized PAST 180 GB so it's rejected at parse time rather than booted-then-OOM'd.
    from flash.providers.allocator import required_vram_gb as alloc_required_vram_gb

    moe = "Qwen/Qwen3.6-35B-A3B"
    # default + moderate context fit the single B200 (<= 180 GB).
    assert alloc_required_vram_gb(moe, "grpo", train={}, thinking=False) <= 180
    assert alloc_required_vram_gb(moe, "grpo", train={"max_length": 8192}, thinking=False) <= 180
    # a 32k-token rollout is sized ABOVE the 180 GB B200 (the biggest single card) -> nothing fits.
    assert alloc_required_vram_gb(moe, "grpo", train={"max_length": 32768}, thinking=False) > 180
    # The GRPO escalation is GRPO-only: default SFT stays at its 180 GB floor (fits the B200). (Long-
    # context SFT has its OWN large-vocab fp32-logits growth, independent of this grpo escalation.)
    assert alloc_required_vram_gb(moe, "sft", train={}, thinking=False) <= 180


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="steps must be"):
        RunConfig(MID, "sft", 0)
    with pytest.raises(ValueError, match="unsupported algorithm"):
        RunConfig(MID, "ppo", 100)


def test_omitted_grpo_max_length_sizes_like_the_real_allocator():
    # An omitted GRPO max_length mirrors the worker's max(1024, max_prompt_len + completion), not
    # bare max_prompt_len -- else the estimate under-sizes VRAM by the completion budget.
    from flash.engine.recipe import RECIPE
    from flash.providers.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "grpo", 100)  # max_length / seq_len omitted
    worker_len = max(1024, RECIPE.rl.max_prompt_len + RECIPE.rl.max_completion_len)
    assert cfg.normalized().seq_len == worker_len
    assert cfg.train_knobs()["max_length"] == worker_len
    _, need = select_gpu(cfg)
    real = alloc_required_vram_gb(MID, "grpo", train={}, thinking=False)
    assert need == real
    # ...and never under-sizes vs the old bare-max_prompt_len default.
    old = alloc_required_vram_gb(MID, "grpo", train={"max_length": RECIPE.rl.max_prompt_len}, thinking=False)
    assert need >= old


def test_omitted_grpo_max_length_mirrors_worker_with_thinking():
    # The thinking completion budget (larger) feeds the same worker-mirrored default.
    from flash.engine.recipe import RECIPE

    cfg = RunConfig(MID, "grpo", 100, thinking=True)
    worker_len = max(1024, RECIPE.rl.max_prompt_len + RECIPE.rl.max_completion_len_thinking)
    assert cfg.normalized().seq_len == worker_len
    assert worker_len > max(1024, RECIPE.rl.max_prompt_len + RECIPE.rl.max_completion_len)


def test_explicit_grpo_max_length_still_wins():
    # An explicitly pinned seq_len (engine length) is honored verbatim, not overridden by the
    # worker-mirrored default, and matches the allocator at that same pinned max_length.
    from flash.providers.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "grpo", 100, seq_len=8192)
    assert cfg.normalized().seq_len == 8192
    assert cfg.train_knobs()["max_length"] == 8192
    _, need = select_gpu(cfg)
    real = alloc_required_vram_gb(MID, "grpo", train={"max_length": 8192}, thinking=False)
    assert need == real


@pytest.mark.parametrize(
    "knob",
    ["seq_len", "batch_size", "group_size", "completion_len", "lora_rank"],
)
@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_run_knobs_rejected(knob, bad):
    # A <= 0 run knob produces a bogus quote, so it's rejected up front (like steps >= 1).
    with pytest.raises(ValueError, match=f"{knob} must be"):
        RunConfig(MID, "grpo", 100, **{knob: bad})

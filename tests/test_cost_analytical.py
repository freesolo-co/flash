"""Cost estimator: the analytical ground-truth model.

No network. Checks the qualitative invariants the model must satisfy (cost scales with
steps, GRPO costs more than SFT, bigger models cost more per step, the wall-clock cap
bounds runaway runs) plus end-to-end arithmetic consistency.
"""

from __future__ import annotations

import pytest

from flash.cost import RunConfig, estimate_cost, seconds_per_step, select_gpu
from flash.cost.analytical import DEFAULT_WALL_CAP_S, setup_seconds

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
    # When SFT batch_size is omitted, GPU sizing must match the REAL allocator, which
    # defaults an omitted batch to the worker's per-device micro-batch (_sft_per_device_bs() = 4
    # in model_required_vram_gb) -- NOT the recipe effective batch (32). The recipe batch is still
    # the right compute-throughput model in seconds_per_step; it just must not over-provision VRAM
    # relative to the allocator.
    from flash.providers.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "sft", 100)  # batch_size omitted
    _, need = select_gpu(cfg)
    real = alloc_required_vram_gb(MID, "sft", train={}, thinking=False)  # allocator defaults to micro-batch 4
    assert need == real
    # batch_size must NOT be forwarded for sizing when omitted (it would inflate the need).
    assert "batch_size" not in cfg.train_knobs()


def test_explicit_sft_batch_is_still_forwarded_for_sizing():
    # An EXPLICIT batch_size is honored for VRAM sizing (and matches the allocator at that
    # batch), so a deliberately large batch still routes to the right card.
    from flash.providers.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "sft", 100, batch_size=32)
    _, need = select_gpu(cfg)
    real = alloc_required_vram_gb(MID, "sft", train={"batch_size": 32}, thinking=False)
    assert cfg.train_knobs()["batch_size"] == 32
    assert need == real


def test_offline_open_model_sized_from_parsed_id_not_24gb_fallback():
    # Offline (FLASH_SKIP_NET), HF metadata is unreadable, so the allocator's open-model
    # path returns a flat 24 GB. But model_specs parses the size from the id, so an unlisted
    # 13B model must size WELL above 24 GB and route to a big card -- not a 24 GB A5000.
    big_open = RunConfig("vendor/foo-13b", "sft", 100)
    _, need = select_gpu(big_open)
    assert need > 24
    # A 13B run does not fit a 24 GB card.
    e = estimate_cost(big_open)
    assert e.gpu_vram_gb >= e.required_vram_gb
    assert e.required_vram_gb > 24


def test_offline_open_model_helper_defers_for_catalog_models():
    # The offline open-model override ONLY replaces the flat-24 fallback for unlisted models;
    # a catalog model defers to the allocator (returns None), so curated sizing is unchanged.
    from flash.cost.analytical import _offline_open_model_vram_gb

    assert _offline_open_model_vram_gb(RunConfig(MID, "grpo", 100)) is None
    assert _offline_open_model_vram_gb(RunConfig(SMALL, "sft", 100)) is None


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
    # The runner provisions each seed as its own job and applies max_wall_seconds PER SEED.
    # So a 3-seed run that wall-caps at a per-seed limit bills 3x the per-seed capped wall,
    # NOT a single aggregate cap. With a 1h per-seed cap on a runaway 3-seed run, total wall
    # is ~3h (3 capped seeds), not 1h.
    cfg = RunConfig(BIG, "grpo", 300_000, setup_repeats=3, max_wall_seconds=3600)
    e = estimate_cost(cfg)
    assert e.wall_capped is True
    assert e.wall_clock_seconds == pytest.approx(3 * 3600.0)


def test_sub_60s_wall_cap_is_floored_to_the_runner_minimum():
    # The runner enforces max(60, max_wall_seconds); the estimate must mirror it so a sub-60s
    # cap isn't priced below what the run actually bills (otherwise a ~$0 estimate). A 10s and a
    # 30s cap both floor to 60s of (capped) wall, and the dollar figure is strictly positive.
    e10 = estimate_cost(RunConfig(BIG, "grpo", 100_000, max_wall_seconds=10))
    e30 = estimate_cost(RunConfig(BIG, "grpo", 100_000, max_wall_seconds=30))
    assert e10.wall_clock_seconds == pytest.approx(60.0)
    assert e30.wall_clock_seconds == pytest.approx(60.0)
    assert e10.total_usd > 0.0


def test_nonpositive_max_wall_seconds_is_rejected():
    # A 0/negative cap would drive estimate_cost to a negative wall and negative total_usd, so
    # RunConfig rejects it up front (a sub-60s but positive cap is fine -- it's floored above).
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_wall_seconds must be >= 1"):
            RunConfig(BIG, "grpo", 10, max_wall_seconds=bad)
    # One seed of the same shape caps at exactly one per-seed window.
    one = RunConfig(BIG, "grpo", 100_000, setup_repeats=1, max_wall_seconds=3600)
    assert estimate_cost(one).wall_clock_seconds == pytest.approx(3600.0)


def test_allow_unvalidated_widens_gpu_selection():
    # When the spec permits unvalidated classes, GPU selection mirrors the allocator's
    # widened pool: an unvalidated class can be picked when it's the cheapest that fits.
    from flash.cost.facts import pick_gpu

    cfg_strict = RunConfig(MID, "sft", 100, allow_unvalidated=False)
    cfg_open = RunConfig(MID, "sft", 100, allow_unvalidated=True)
    strict_gpu, need = select_gpu(cfg_strict)
    open_gpu = select_gpu(cfg_open)[0]
    # Selection routes the allow_unvalidated flag through to pick_gpu: the open pick equals
    # pick_gpu(need, allow_unvalidated=True) and the strict pick equals the validated-only pool.
    assert open_gpu == pick_gpu(need, allow_unvalidated=True)
    assert strict_gpu == pick_gpu(need, allow_unvalidated=False)


def test_unspecified_allow_unvalidated_resolves_like_submit_time(monkeypatch):
    # ``allow_unvalidated`` is tri-state: an UNSPECIFIED value (None, the default and what a
    # spec with no gpu.allow_unvalidated key reconstructs to) is resolved by select_gpu via the
    # SAME helper the submit-time allocator uses (providers.base.unvalidated_allowed), so the
    # estimate prices against exactly the pool a run actually allocates. Flash is fully managed
    # now -- ``unvalidated_allowed`` honors the per-run flag only, with NO global env override --
    # so an unspecified (None) flag resolves to the validated-only pool regardless of any
    # FLASH_GPU_ALLOW_UNVALIDATED in the environment.
    from flash.cost.facts import pick_gpu
    from flash.providers.base import unvalidated_allowed

    cfg_unspecified = RunConfig(MID, "sft", 100)  # allow_unvalidated defaults to None
    assert cfg_unspecified.allow_unvalidated is None
    need = select_gpu(RunConfig(MID, "sft", 100, allow_unvalidated=False))[1]

    # None resolves through unvalidated_allowed -> validated-only (managed default), and an
    # ambient FLASH_GPU_ALLOW_UNVALIDATED does NOT widen it (no global env override anymore).
    assert unvalidated_allowed(None) is False
    monkeypatch.delenv("FLASH_GPU_ALLOW_UNVALIDATED", raising=False)
    assert select_gpu(cfg_unspecified)[0] == pick_gpu(need, allow_unvalidated=False)

    monkeypatch.setenv("FLASH_GPU_ALLOW_UNVALIDATED", "1")
    assert unvalidated_allowed(None) is False
    assert select_gpu(cfg_unspecified)[0] == pick_gpu(need, allow_unvalidated=False)


def test_qlora_model_fits_a_smaller_card_than_bf16_would():
    # Qwen3.5-9B is 4-bit QLoRA; its GRPO still fits a 32 GB 5090, not an 80 GB A100.
    e = estimate_cost(RunConfig(BIG, "grpo", 100))
    assert e.gpu_vram_gb <= 32
    assert any("qlora" in n.lower() for n in e.notes)


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="steps must be"):
        RunConfig(MID, "sft", 0)
    with pytest.raises(ValueError, match="unsupported algorithm"):
        RunConfig(MID, "ppo", 100)


@pytest.mark.parametrize(
    "knob",
    ["seq_len", "batch_size", "group_size", "completion_len", "lora_rank"],
)
@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_run_knobs_rejected(knob, bad):
    # A <= 0 run knob produces a bogus quote (zero/negative tokens or completions),
    # so it's rejected up front -- consistent with the existing steps >= 1 check and the
    # real parser's train.lora_rank < 1 rejection.
    with pytest.raises(ValueError, match=f"{knob} must be"):
        RunConfig(MID, "grpo", 100, **{knob: bad})

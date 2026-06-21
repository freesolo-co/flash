"""Cost estimator: the analytical ground-truth model.

No network. Checks the qualitative invariants the model must satisfy (cost scales with
steps, GRPO costs more than SFT, bigger models cost more per step, the wall-clock cap
bounds runaway runs) plus end-to-end arithmetic consistency.
"""

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


def test_nonpositive_max_wall_seconds_is_accepted_and_floored():
    # A 0/negative max_wall_seconds is ACCEPTED, mirroring the runner: submit/run floor it with
    # max(60, int(spec.gpu.max_wall_seconds)), so the runner accepts a non-positive cap and runs
    # it for 60s of wall. RunConfig must NOT reject it (else --cost can't price configs the
    # runner accepts), and estimate_cost's cap_s = max(60.0, ...) floor turns it into a 60s wall
    # with a strictly-positive total_usd -- NOT a negative/zero quote.
    for cap in (0, -5):
        cfg = RunConfig(BIG, "grpo", 100_000, max_wall_seconds=cap)  # accepted, no raise
        assert cfg.max_wall_seconds == cap
        e = estimate_cost(cfg)
        assert e.wall_clock_seconds == pytest.approx(60.0)  # floored to the runner's 60s minimum
        assert e.total_usd > 0.0  # positive, not negative
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


def test_omitted_grpo_max_length_sizes_like_the_real_allocator():
    # When GRPO max_length (seq_len) is omitted, the engine context length must MIRROR the worker:
    # the allocator sizes an unset [train].max_length to max(1024, max_prompt_len + completion)
    # (flash/engine/vram.py:243, worker/__init__.py:1478), NOT bare max_prompt_len -- otherwise the
    # estimate under-sizes VRAM by the completion budget and can pick/price an undersized GPU.
    from flash.engine.recipe import RECIPE
    from flash.providers.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "grpo", 100)  # max_length / seq_len omitted
    worker_len = max(1024, RECIPE.rl.max_prompt_len + RECIPE.rl.max_completion_len)
    assert cfg.normalized().seq_len == worker_len  # mirrors the worker exactly
    assert cfg.train_knobs()["max_length"] == worker_len  # forwarded as [train].max_length
    # GPU sizing matches the allocator's own omitted-max_length default (same engine length).
    _, need = select_gpu(cfg)
    real = alloc_required_vram_gb(MID, "grpo", train={}, thinking=False)
    assert need == real
    # ...and is >= what the OLD bare-max_prompt_len default would have produced (never under-sizes).
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


def test_estimate_does_no_network_for_unlisted_model(monkeypatch):
    # The estimator's contract is fully local. select_gpu() threads skip_net=True through the VRAM
    # sizing stack (required_vram_gb -> model_required_vram_gb -> fetch_hf_params_b), so an UNLISTED
    # model never constructs the HF network client: fetch_hf_params_b's skip_net guard short-circuits
    # (return None) BEFORE it imports/instantiates HfApi. Wire HfApi to flip a flag if ever built and
    # assert it stays untouched, and that the estimate still succeeds via the offline fallback (the
    # 24 GB tier). (This is purely the explicit param -- there is no global offline env switch.)
    import huggingface_hub

    built = []

    class _NoNetHfApi:  # pragma: no cover - must never be constructed/called
        def __init__(self, *a, **k):
            built.append(True)
            raise AssertionError("estimator constructed the HF network client (HfApi)")

        def model_info(self, *a, **k):
            raise AssertionError("estimator hit the HF network (model_info)")

    monkeypatch.setattr(huggingface_hub, "HfApi", _NoNetHfApi)
    e = estimate_cost(RunConfig("some-org/totally-unlisted-7b", "grpo", 50))
    assert not built  # the network client was never even constructed
    assert e.required_vram_gb >= 24  # sane offline VRAM estimate (unlisted -> 24 GB tier)
    assert e.total_usd > 0


def test_skip_net_param_bypasses_hf(monkeypatch):
    # Unit-level: the threaded skip_net=True bypasses the HF probe for an unlisted model -- proving
    # the explicit param (not any global state) is what forces offline sizing.
    from flash.engine.vram import fetch_hf_params_b, model_required_vram_gb

    def _boom(*a, **k):  # any HF construction/lookup is a failure
        raise AssertionError("hit the HF network despite skip_net=True")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _boom)
    # Direct: skip_net short-circuits to None (offline) without constructing HfApi.
    assert fetch_hf_params_b("some-org/unlisted", skip_net=True) is None
    # And the sizing matrix falls back to the 24 GB tier for the unreadable unlisted model.
    assert model_required_vram_gb("some-org/unlisted", "grpo", skip_net=True) == 24


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

"""Cost estimator: the analytical ground-truth model. Qualitative invariants (cost scales with
steps, GRPO > SFT, bigger model costs more, the wall cap bounds runs) + arithmetic consistency."""

from __future__ import annotations

import math

import pytest

from flash.cost import RunConfig, estimate_cost
from flash.cost.analytical import (
    DEFAULT_WALL_CAP_S,
    VLLM_INIT_S,
    _offline_gpu_shape,
    seconds_per_step,
    setup_seconds,
)

SMALL = "Qwen/Qwen3.5-9B"
MID = "Qwen/Qwen3.5-9B"
BIG = "Qwen/Qwen3.5-9B"


def test_estimate_is_positive_and_self_consistent():
    e = estimate_cost(RunConfig(MID, "sft", 200))
    assert e.total_usd > 0
    assert e.wall_clock_seconds == pytest.approx(e.setup_seconds + e.train_seconds)
    # total = billable training hours x hourly rate; setup is elapsed time only.
    assert e.total_usd == pytest.approx(e.billable_hours * e.gpu_hourly_usd)
    assert e.billable_hours == pytest.approx(e.train_seconds / 3600.0)
    # chosen card actually fits the run's requirement
    assert e.gpu_vram_gb >= e.required_vram_gb


def test_cost_increases_with_steps():
    costs = [estimate_cost(RunConfig(SMALL, "sft", s)).total_usd for s in (100, 500, 1000)]
    assert costs[0] < costs[1] < costs[2]


def test_sft_train_tokens_price_actual_tokens_instead_of_padded_slots():
    padded = estimate_cost(RunConfig(MID, "sft", 10, batch_size=16, seq_len=2048))
    actual = estimate_cost(
        RunConfig(MID, "sft", 10, batch_size=16, seq_len=2048, train_tokens=50_000)
    )
    assert actual.train_seconds < padded.train_seconds
    assert actual.total_usd < padded.total_usd
    assert any("50,000 actual train tokens" in n for n in actual.notes)


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
    gpu = "H100"
    small = seconds_per_step(RunConfig("Qwen/Qwen3.5-9B", "sft", 1), gpu)
    big = seconds_per_step(RunConfig("Qwen/Qwen3.8-27B", "sft", 1), gpu)
    assert big > small


def test_grpo_requires_at_least_as_much_vram_as_sft():
    for model in (SMALL, MID, BIG):
        _, sft_need, *_ = _offline_gpu_shape(RunConfig(model, "sft", 100))
        _, grpo_need, *_ = _offline_gpu_shape(RunConfig(model, "grpo", 100))
        assert grpo_need >= sft_need


def test_omitted_sft_batch_sizes_like_the_real_allocator():
    # An omitted SFT batch sizes VRAM like the allocator (micro-batch 4), not the recipe batch (32).
    from flash.providers.core.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "sft", 100)  # batch_size omitted
    _, need, *_ = _offline_gpu_shape(cfg)
    real = alloc_required_vram_gb(MID, "sft", train={}, thinking=False)
    assert need == real
    assert "batch_size" not in cfg.train_knobs()  # omitted batch isn't forwarded (would inflate)


def test_explicit_sft_batch_is_still_forwarded_for_sizing():
    # An EXPLICIT batch_size is honored for VRAM sizing (and matches the allocator at that
    # batch), so a deliberately large batch still routes to the right card.
    from flash.providers.core.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "sft", 100, batch_size=32)
    _, need, *_ = _offline_gpu_shape(cfg)
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
    # grpo pays an extra vllm-init cost; bigger models download longer.
    assert setup_seconds(RunConfig(MID, "grpo", 1)) > setup_seconds(RunConfig(MID, "sft", 1))
    assert setup_seconds(RunConfig("Qwen/Qwen3.8-27B", "sft", 1)) > setup_seconds(
        RunConfig("Qwen/Qwen3.5-9B", "sft", 1)
    )


def test_setup_opd_includes_vllm_init():
    # OPD now starts a colocated vLLM rollout engine too; wall-cap estimates must include that setup.
    assert setup_seconds(RunConfig(MID, "opd", 1)) == pytest.approx(
        setup_seconds(RunConfig(MID, "sft", 1)) + VLLM_INIT_S
    )


def test_cold_start_calibrated_to_real_short_sft_run(monkeypatch):
    # the historical 0.9b calibration is a generic size anchor, not an executable catalog model.
    from flash.core.catalog import MODELS, ModelInfo

    model_id = "test/cold-start-anchor"
    monkeypatch.setitem(
        MODELS,
        model_id,
        ModelInfo(
            id=model_id,
            display_name="synthetic 0.9b calibration anchor",
            params="0.9B",
            params_b=0.9,
            algos=("sft",),
            min_vram_gb=12,
        ),
    )
    e = estimate_cost(RunConfig(model_id, "sft", 26))
    assert e.gpu == "RTX 4090"
    assert e.gpu_hourly_usd == pytest.approx(0.69, abs=1e-3)
    assert e.total_usd == pytest.approx(e.billable_hours * e.gpu_hourly_usd)
    assert e.total_usd < e.wall_clock_hours * e.gpu_hourly_usd
    assert e.setup_seconds > e.train_seconds


def test_cold_start_negligible_for_long_runs():
    # The bigger cold start must NOT regress long runs: when training wall dominates, setup is a
    # small single-digit fraction of elapsed wall time.
    e = estimate_cost(RunConfig(SMALL, "sft", 5000))
    assert e.setup_seconds / e.wall_clock_seconds < 0.05


def test_wall_clock_cap_bounds_runaway_runs():
    e = estimate_cost(RunConfig(BIG, "grpo", 100_000))
    assert e.wall_capped is True
    assert e.wall_clock_seconds == pytest.approx(DEFAULT_WALL_CAP_S)
    # The cap shows up in the notes and bounds billable training time.
    assert any("cap" in n.lower() for n in e.notes)
    assert e.total_usd == pytest.approx(e.billable_hours * e.gpu_hourly_usd)
    assert e.total_usd < e.wall_clock_hours * e.gpu_hourly_usd
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


def test_sub_60s_wall_cap_is_floored_to_the_runner_minimum():
    # The runner floors the cap to max(60, ...); the estimate mirrors it (else a ~$0 quote).
    e10 = estimate_cost(RunConfig(BIG, "grpo", 100_000, max_wall_seconds=10))
    e30 = estimate_cost(RunConfig(BIG, "grpo", 100_000, max_wall_seconds=30))
    assert e10.wall_clock_seconds == pytest.approx(60.0)
    assert e30.wall_clock_seconds == pytest.approx(60.0)
    assert e10.train_seconds == pytest.approx(0.0)
    assert e10.total_usd == pytest.approx(0.0)


def test_nonpositive_max_wall_seconds_is_accepted_and_floored():
    # A 0/negative cap is accepted (the runner floors it to 60s), not rejected -- so --cost can
    # price configs the runner accepts; estimate_cost mirrors the 60s floor -> positive quote.
    for cap in (0, -5):
        cfg = RunConfig(BIG, "grpo", 100_000, max_wall_seconds=cap)
        assert cfg.max_wall_seconds == cap
        e = estimate_cost(cfg)
        assert e.wall_clock_seconds == pytest.approx(60.0)
        assert e.train_seconds == pytest.approx(0.0)
        assert e.total_usd == pytest.approx(0.0)
    capped = RunConfig(BIG, "grpo", 100_000, max_wall_seconds=3600)
    assert estimate_cost(capped).wall_clock_seconds == pytest.approx(3600.0)


def test_9b_bf16_grpo_needs_an_80gb_class():
    # Qwen3.5-9B is bf16 (QLoRA was dropped: the 4-bit vLLM-rollout merge collapsed the GRPO
    # importance-sampling ratio -> no learning). Colocated bf16 GRPO needs an 80 GB-class card.
    e = estimate_cost(RunConfig(BIG, "grpo", 100))
    assert e.required_vram_gb >= 80
    assert e.gpu_vram_gb >= 80


def test_35b_moe_long_context_grpo_sized_past_the_resident_wall():
    # the 35b moe is resident-only for grpo because vllm sleep hangs its wake. 205 gb separates
    # moderate from long contexts using the resident peak, preventing admission into the broken sleep
    # path; it is intentionally not a card size.
    from flash.providers.core.allocator import required_vram_gb as alloc_required_vram_gb

    moe = "Qwen/Qwen3.6-35B-A3B"
    # default + moderate context stay under the wall.
    assert alloc_required_vram_gb(moe, "grpo", train={}, thinking=False) <= 205
    assert (
        alloc_required_vram_gb(moe, "grpo", train={"max_context_tokens": 4096}, thinking=False)
        <= 205
    )
    # past the resident wall -> sized above it -> rejected (NOT routed to broken sleep).
    assert (
        alloc_required_vram_gb(moe, "grpo", train={"max_context_tokens": 8192}, thinking=False)
        > 205
    )
    assert (
        alloc_required_vram_gb(moe, "grpo", train={"max_context_tokens": 32768}, thinking=False)
        > 205
    )
    # The GRPO escalation is GRPO-only: default SFT stays at its 180 GB floor (fits the B200). (Long-
    # context SFT has its OWN large-vocab fp32-logits growth, independent of this grpo escalation.)
    assert alloc_required_vram_gb(moe, "sft", train={}, thinking=False) <= 180


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="steps must be"):
        RunConfig(MID, "sft", 0)
    with pytest.raises(ValueError, match="unsupported algorithm"):
        RunConfig(MID, "ppo", 100)


def test_omitted_grpo_context_sizes_like_the_real_allocator():
    # An omitted GRPO context mirrors the worker's max(1024, max_prompt_len + completion), not
    # bare max_prompt_len -- else the estimate under-sizes VRAM by the completion budget.
    from flash.engine.plan.recipe import RECIPE
    from flash.providers.core.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "grpo", 100)  # max_context_tokens / seq_len omitted
    worker_len = max(1024, RECIPE.rl.max_prompt_len + RECIPE.rl.max_completion_len)
    assert cfg.normalized().seq_len == worker_len
    assert cfg.train_knobs()["max_context_tokens"] == worker_len
    _, need, *_ = _offline_gpu_shape(cfg)
    real = alloc_required_vram_gb(MID, "grpo", train={}, thinking=False)
    assert need == real
    # ...and never under-sizes vs the old bare-max_prompt_len default.
    old = alloc_required_vram_gb(
        MID, "grpo", train={"max_context_tokens": RECIPE.rl.max_prompt_len}, thinking=False
    )
    assert need >= old


def test_omitted_grpo_context_mirrors_worker_with_thinking():
    # The thinking completion budget (larger) feeds the same worker-mirrored default.
    from flash.engine.plan.recipe import RECIPE

    cfg = RunConfig(MID, "grpo", 100, thinking=True)
    worker_len = max(1024, RECIPE.rl.max_prompt_len + RECIPE.rl.max_completion_len_thinking)
    assert cfg.normalized().seq_len == worker_len
    assert worker_len > max(1024, RECIPE.rl.max_prompt_len + RECIPE.rl.max_completion_len)


def test_explicit_grpo_context_still_wins():
    # An explicitly pinned seq_len (engine length) is honored verbatim, not overridden by the
    # worker-mirrored default, and matches the allocator at that same pinned context.
    from flash.providers.core.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "grpo", 100, seq_len=8192)
    assert cfg.normalized().seq_len == 8192
    assert cfg.train_knobs()["max_context_tokens"] == 8192
    _, need, *_ = _offline_gpu_shape(cfg)
    real = alloc_required_vram_gb(MID, "grpo", train={"max_context_tokens": 8192}, thinking=False)
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


@pytest.mark.parametrize(
    ("multi_turn", "max_turns", "multiplier"),
    [(False, None, 3), (True, 24, 72), (True, 64, 192), (True, None, 192)],
)
def test_opd_teacher_cost_uses_authoritative_request_multiplier(multi_turn, max_turns, multiplier):
    from flash.cost.facts import teacher_token_cost_usd

    config = RunConfig(
        MID,
        "opd",
        1,
        seq_len=1024,
        batch_size=2,
        group_size=1,
        teacher_model="glm-5.2",
        opd_multi_turn=multi_turn,
        opd_max_turns=max_turns,
    )

    estimate = estimate_cost(config)

    input_tokens = 2 * 1024 * multiplier
    output_tokens = 2 * multiplier
    assert estimate.teacher_api_usd == pytest.approx(
        teacher_token_cost_usd(input_tokens, output_tokens, "glm-5.2")
    )


@pytest.mark.parametrize(
    ("completions", "multi_turn", "max_turns", "expected_scored_requests"),
    [
        # single-turn: every completion costs OPD_NO_SIGNAL_ATTEMPTS (3) potentially-scored
        # requests, because a bounded no-signal replacement consumes a slot whether or not it is
        # needed. so 8 completions -> 24 requests, 9 -> 27, 16 -> 48.
        (8, False, None, 24),
        (9, False, None, 27),
        (16, False, None, 48),
        # multi-turn with no explicit cap bounds at OPD_MAX_EPISODE_TURNS (64) assistant turns,
        # each of which is separately scored and separately retried: 1 * 64 * 3 = 192.
        (1, True, None, 192),
    ],
)
def test_opd_teacher_latency_uses_conservative_retry_and_turn_wave_policy(
    completions,
    multi_turn,
    max_turns,
    expected_scored_requests,
):
    from flash.cost.analytical import step_seconds_split
    from flash.cost.facts import teacher_seconds_per_completion
    from flash.teacher.limits import OPD_TEACHER_SCORING_CONCURRENCY

    config = RunConfig(
        MID,
        "opd",
        1,
        batch_size=completions,
        group_size=1,
        opd_multi_turn=multi_turn,
        opd_max_turns=max_turns,
    )

    _gpu_seconds, fixed_seconds = step_seconds_split(config, "RTX 5090")

    # hand-write the policy request count, including retries and per-turn scoring, but derive waves
    # from the measured concurrency constant. using the production count helper would make this test
    # self-fulfilling.
    expected_waves = math.ceil(expected_scored_requests / OPD_TEACHER_SCORING_CONCURRENCY)
    assert fixed_seconds == pytest.approx(expected_waves * teacher_seconds_per_completion())


def test_revision_pinned_sizing_flows_into_setup_and_required_save(monkeypatch, tmp_path):
    # regression (PR #538 finding 0): cold-start setup + required-save cost must size the PINNED
    # commit's weights, not the catalog default-revision param count, so a run whose revision differs
    # from the catalog is priced on the checkpoint the worker actually downloads and serializes.
    import json
    from types import SimpleNamespace

    from flash.core.catalog import MODELS, ModelInfo
    from flash.cost.analytical import required_save_overhead_seconds
    from flash.cost.facts import download_weight_gb, total_params_b

    model_id = "test/revision-sized-model"
    monkeypatch.setitem(
        MODELS,
        model_id,
        ModelInfo(
            id=model_id,
            display_name="synthetic revision-sized model",
            params="0.9B",
            params_b=0.9,
            algos=("sft",),
            min_vram_gb=12,
            vocab_size=248_320,
        ),
    )

    # a pinned commit the hub reports at 0.87b, within the 5% catalog-drift gate (catalog is 0.9b).
    # vocab must equal the catalog to clear the fail-closed validation for a cataloged model.
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"vocab_size": 248320}))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(config))

    class Api:
        def __init__(self, *, token):
            pass

        def model_info(self, model, **kwargs):
            return SimpleNamespace(safetensors=SimpleNamespace(total=int(0.87e9)))

    monkeypatch.setattr("huggingface_hub.HfApi", Api)

    rev = "d" * 40
    # the facts accessors resolve the pinned commit's size, distinct from the catalog default.
    assert total_params_b(model_id, rev) == pytest.approx(0.87)
    assert total_params_b(model_id) == pytest.approx(0.9)
    assert download_weight_gb(model_id, rev) == pytest.approx(1.74)

    # both cold-start setup (download term) and required-save (serialize term) track the smaller pinned
    # weights, so the 0.87B revision quotes strictly cheaper than the 0.9B catalog default.
    pinned = RunConfig(model_id, "sft", 100, model_revision=rev, save_at_steps=(100,))
    default = RunConfig(model_id, "sft", 100, save_at_steps=(100,))
    assert setup_seconds(pinned) < setup_seconds(default)
    assert required_save_overhead_seconds(pinned) < required_save_overhead_seconds(default)


def test_offline_quote_and_allocator_agree_on_the_executed_sft_width():
    """A quoted shape must be one the allocator will actually accept.

    Both sides credit multi-card VRAM from sharding, so both must credit the same number of ranks.
    The offline quote used the BILLED count while the allocator credits the launched one, so a 27B
    at 128k over 10 retained rows was quoted 4x H200 -- 460 GB credited against a 422 GB need --
    while the allocator launches 2 ranks (10 rows cannot split 4 ways), credits 234 GB, and rejects
    it. The run was priced as feasible and then refused at submit.

    Asserted as agreement rather than against a literal shape: the requirement moves with the vram
    model, but the two paths must never disagree about the same run.
    """
    import pytest

    from flash.cost.analytical import _offline_gpu_shape, executed_gpu_count
    from flash.cost.types import RunConfig
    from flash.providers.core.allocator import _fits
    from flash.providers.core.base import GPU_INFO, Candidate

    def quoted_shape_is_allocatable(**kwargs):
        config = RunConfig("Qwen/Qwen3.8-27B", "sft", 10, **kwargs)
        gpu, need, count, _provider, rate = _offline_gpu_shape(config)
        candidate = Candidate(
            provider="runpod",
            gpu=gpu,
            hourly_usd=rate,
            vram_gb=GPU_INFO[gpu].vram_gb,
            gpu_count=count,
        )
        return _fits(candidate, need, executed_gpu_count(config, count))

    # the shape that exposed the disagreement: rows bind the width below the quoted card count.
    with pytest.raises(ValueError, match="VRAM"):
        quoted_shape_is_allocatable(seq_len=131072, batch_size=8, sft_retained_examples=10)

    # and a run that genuinely fits must still be quoted, or the clamp would reject everything.
    assert quoted_shape_is_allocatable(seq_len=4096, batch_size=8, sft_retained_examples=64)

    # the shared helper is the reason they cannot drift: sft narrows, everything else does not.
    grpo = RunConfig("Qwen/Qwen3.8-27B", "grpo", 10, batch_size=8, sft_retained_examples=10)
    assert executed_gpu_count(grpo, 4) == 4
    sft_rows_bound = RunConfig(
        "Qwen/Qwen3.8-27B", "sft", 10, batch_size=8, sft_retained_examples=10
    )
    assert executed_gpu_count(sft_rows_bound, 4) == 2

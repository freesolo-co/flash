"""Cost estimator: the analytical ground-truth model. Qualitative invariants (cost scales with
steps, GRPO > SFT, bigger model costs more, the wall cap bounds runs) + arithmetic consistency."""

from __future__ import annotations

import itertools

import pytest

from flash.cost import RunConfig, estimate_cost
from flash.cost.analytical import (
    DEFAULT_WALL_CAP_S,
    SFT_RUN_STARTUP_K,
    VLLM_INIT_S,
    multi_card_speedup,
    run_startup_seconds,
    seconds_per_step,
    select_gpu,
    setup_seconds,
    step_seconds_split,
)

SMALL = "Qwen/Qwen3.5-0.8B"
MID = "Qwen/Qwen3.5-4B"
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


def test_run_startup_is_charged_per_method_and_per_card():
    # verl launch + model load + framework init land inside the billed train wall for every method.
    # an earlier revision charged this to sft ONLY, on the reasoning that grpo and opd already
    # carried it in their per-step floor. that was wrong: the per-step floor WAS this block divided
    # by the step count of the arms it was fitted on, which is why it is charged here now and no
    # longer charged per step.
    sft_block = run_startup_seconds(RunConfig(MID, "sft", 32), "H100")
    assert sft_block > 0.0
    # sft's block has no per-class table, so it is card-invariant (it DOES scale with model size).
    assert run_startup_seconds(RunConfig(MID, "sft", 32), "RTX 5090") == sft_block

    # rollout methods build a vLLM engine and do a first weight sync on top, so their block is both
    # larger than sft's and card-shaped.
    for method in ("grpo", "opd"):
        h100 = run_startup_seconds(RunConfig(MID, method, 32), "H100")
        rtx = run_startup_seconds(RunConfig(MID, method, 32), "RTX 5090")
        assert h100 > sft_block, method
        assert rtx > 0.0, method
        assert h100 != rtx, method


def test_run_startup_is_zero_for_a_resident_rollout_engine():
    # a sleep_unsupported model pins the rollout engine resident, so the engine build this block is
    # mostly made of never runs. see facts.run_block_seconds for what that zero gives up.
    resident = RunConfig("Qwen/Qwen3.6-35B-A3B", "grpo", 32)
    assert run_startup_seconds(resident, "B200") == 0.0
    assert run_startup_seconds(RunConfig(MID, "grpo", 32), "B200") > 0.0


def test_sft_run_startup_scales_with_model_size():
    # a flat block was falsified: it charges a 0.8B run and a 27B run the same 68.4s, while the
    # realized arms need ~81s and ~437s. the block is model-shaped, so the quote must be too.
    small = run_startup_seconds(RunConfig("Qwen/Qwen3.5-0.8B", "sft", 32), "H100")
    mid = run_startup_seconds(RunConfig("Qwen/Qwen3.5-4B", "sft", 32), "H100")
    big = run_startup_seconds(RunConfig("Qwen/Qwen3.6-27B", "sft", 32), "H100")
    assert small < mid < big
    # sublinear: a 34x parameter span must not become a 34x block, or short big-model runs blow up.
    assert big / small < 10.0
    assert SFT_RUN_STARTUP_K > 0.0


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


def test_setup_opd_includes_vllm_init():
    # OPD now starts a colocated vLLM rollout engine too; wall-cap estimates must include that setup.
    assert setup_seconds(RunConfig(MID, "opd", 1)) == pytest.approx(
        setup_seconds(RunConfig(MID, "sft", 1)) + VLLM_INIT_S
    )


def test_cold_start_calibrated_to_real_short_sft_run():
    # Calibration anchor: a real fresh-worker run (0.8B SFT, 391 examples -> 26 priced steps at
    # the recipe batch) was cold-start-dominated (a fresh worker spent ~12.5 min in model load).
    # Static pricing picks the cheapest fitting class; the cheapest managed card is the 24 GB
    # RTX 4090 ($0.69). 26 = ceil(391 / 32) * 2 epochs.
    e = estimate_cost(RunConfig(SMALL, "sft", 26))
    assert e.gpu == "RTX 4090"
    assert e.gpu_hourly_usd == pytest.approx(0.69, abs=1e-3)
    assert e.total_usd == pytest.approx(e.billable_hours * e.gpu_hourly_usd)
    assert e.total_usd < e.wall_clock_hours * e.gpu_hourly_usd
    # Model load (not boot/deps) is the dominant cold-start term for a short job.
    assert e.setup_seconds > e.train_seconds  # cold start dominates this short run


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


def test_select_gpu_picks_cheapest_including_unvalidated():
    # No validation gate: select_gpu picks the cheapest fitting class (validated or not) at the
    # static rate, and nothing fitting is cheaper.
    from flash.cost.facts import gpu_hourly_usd, pick_gpu
    from flash.providers.base import GPU_INFO

    gpu, need = select_gpu(RunConfig(MID, "sft", 100))
    assert gpu == pick_gpu(need)
    cheaper = [
        g
        for g in GPU_INFO.values()
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
    # The 35B MoE is RESIDENT-ONLY for GRPO (sleep_unsupported: vLLM sleep HANGS its wake), so it's
    # sized on the RESIDENT peak (two ~70 GB weight copies + KV pool, fp8 KV). Default/moderate context
    # fits the single 180 GB B200, but anything past the resident wall (~4-5k tok at group 8) is sized
    # PAST 180 GB -> REJECTED at parse time, rather than admitted-then-HUNG in the broken sleep path
    # (the old sleep estimate wrongly admitted up to ~16k, then the worker stalled).
    from flash.providers.allocator import required_vram_gb as alloc_required_vram_gb

    moe = "Qwen/Qwen3.6-35B-A3B"
    # default + moderate context fit the single B200 (<= 180 GB).
    assert alloc_required_vram_gb(moe, "grpo", train={}, thinking=False) <= 180
    assert (
        alloc_required_vram_gb(moe, "grpo", train={"max_context_tokens": 4096}, thinking=False)
        <= 180
    )
    # past the resident wall -> sized ABOVE the 180 GB B200 -> rejected (NOT routed to broken sleep).
    assert (
        alloc_required_vram_gb(moe, "grpo", train={"max_context_tokens": 8192}, thinking=False)
        > 180
    )
    assert (
        alloc_required_vram_gb(moe, "grpo", train={"max_context_tokens": 32768}, thinking=False)
        > 180
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
    from flash.engine.recipe import RECIPE
    from flash.providers.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "grpo", 100)  # max_context_tokens / seq_len omitted
    worker_len = max(1024, RECIPE.rl.max_prompt_len + RECIPE.rl.max_completion_len)
    assert cfg.normalized().seq_len == worker_len
    assert cfg.train_knobs()["max_context_tokens"] == worker_len
    _, need = select_gpu(cfg)
    real = alloc_required_vram_gb(MID, "grpo", train={}, thinking=False)
    assert need == real
    # ...and never under-sizes vs the old bare-max_prompt_len default.
    old = alloc_required_vram_gb(
        MID, "grpo", train={"max_context_tokens": RECIPE.rl.max_prompt_len}, thinking=False
    )
    assert need >= old


def test_omitted_grpo_context_mirrors_worker_with_thinking():
    # The thinking completion budget (larger) feeds the same worker-mirrored default.
    from flash.engine.recipe import RECIPE

    cfg = RunConfig(MID, "grpo", 100, thinking=True)
    worker_len = max(1024, RECIPE.rl.max_prompt_len + RECIPE.rl.max_completion_len_thinking)
    assert cfg.normalized().seq_len == worker_len
    assert worker_len > max(1024, RECIPE.rl.max_prompt_len + RECIPE.rl.max_completion_len)


def test_explicit_grpo_context_still_wins():
    # An explicitly pinned seq_len (engine length) is honored verbatim, not overridden by the
    # worker-mirrored default, and matches the allocator at that same pinned context.
    from flash.providers.allocator import required_vram_gb as alloc_required_vram_gb

    cfg = RunConfig(MID, "grpo", 100, seq_len=8192)
    assert cfg.normalized().seq_len == 8192
    assert cfg.train_knobs()["max_context_tokens"] == 8192
    _, need = select_gpu(cfg)
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


def test_opd_bills_the_completion_cap_not_the_context_capacity():
    # OPD used to bill completions x seq_len, i.e. max_context_tokens -- the engine's context
    # CAPACITY. Capacity is a memory-sizing bound: raising it does not make a step generate or train
    # on more tokens, so a quote that tracks it moves with a config knob rather than with the work.
    # Measured on 65 held-out arms, capacity-billing ran a 2.452x geometric over-quote against
    # 1.520x for cap-billing, so this pins the basis rather than the magnitude.
    from flash.cost.analytical import _opd_step_shape

    wide = RunConfig(MID, "opd", 10, batch_size=4, group_size=2, completion_len=256, seq_len=8192)
    narrow = RunConfig(MID, "opd", 10, batch_size=4, group_size=2, completion_len=256, seq_len=512)
    assert _opd_step_shape(wide.normalized()) == _opd_step_shape(narrow.normalized()), (
        "raising max_context_tokens must not change the billed token count"
    )
    # and the count is the generation cap x completions, the same basis GRPO bills.
    completions, tokens = _opd_step_shape(wide.normalized())
    assert completions == 8
    assert tokens == 8 * 256
    # the cap MUST still move the bill, or the term would be insensitive to the actual workload.
    longer = RunConfig(MID, "opd", 10, batch_size=4, group_size=2, completion_len=512, seq_len=8192)
    assert _opd_step_shape(longer.normalized())[1] == 2 * tokens


def test_opd_teacher_scoring_bills_serial_batched_round_trips():
    # Teacher scoring is batched AND serial: _TextTeacherBatcher (opd_train.py) fills a batch of
    # OPD_TEACHER_BATCH_SIZE and hands it to ONE daemon thread, whose loop blocks in _score_batch
    # (a single score_many = a single echo POST) before taking the next batch. So a step's teacher
    # wall is ceil(completions / batch) latencies -- not one wave, and not completions x latency.
    # Hold seq_tokens (hence gen_s/update_s) constant while doubling the completion count so the
    # only thing that can move the delta is per-completion scaling.
    from flash.cost.analytical import OPD_TEACHER_BATCH_SIZE, _opd_step_shape, seconds_per_step
    from flash.cost.facts import ROLLOUT_SECONDS_PER_COMPLETION, teacher_seconds_per_completion

    gpu = "RTX 5090"
    teacher_lat = teacher_seconds_per_completion()
    assert teacher_lat > 0  # else the isolation below is vacuous
    # completions x completion_len is identical (8*2048 == 16*1024), so gen_s/update_s match and only
    # the completion count (hence teacher round trips + rollout slope) differs. The cancelling knob is
    # completion_len, NOT seq_len: OPD bills the generation cap, so holding seq_len equal would leave
    # the FLOPs term free to move and the delta below would no longer isolate the teacher.
    few = RunConfig(MID, "opd", 10, batch_size=8, group_size=1, completion_len=2048, seq_len=4096)
    many = RunConfig(MID, "opd", 10, batch_size=16, group_size=1, completion_len=1024, seq_len=4096)
    # the isolation is the whole point of this fixture, so assert it rather than trusting the
    # arithmetic above: equal billed tokens is what makes the delta a pure teacher+rollout quantity.
    assert _opd_step_shape(few.normalized())[1] == _opd_step_shape(many.normalized())[1], (
        "fixture broken: the two configs must bill identical tokens per step"
    )
    # 8 completions fill exactly one batch; 16 need two. pin that the fixture actually straddles a
    # batch boundary, or the assertion below passes for the wrong reason.
    assert 8 % OPD_TEACHER_BATCH_SIZE == 0, "8 completions must fill whole batches exactly"
    assert 16 // OPD_TEACHER_BATCH_SIZE == 2, "16 completions must need exactly two batches"
    delta = seconds_per_step(many, gpu) - seconds_per_step(few, gpu)
    # the rollout wall scales per completion (vllm samples and detokenizes each sequence separately,
    # whatever the token total), and the teacher adds exactly ONE more round trip across this
    # boundary -- not eight (per-completion serial) and not zero (one parallel wave).
    expected = 8 * ROLLOUT_SECONDS_PER_COMPLETION + teacher_lat
    assert delta == pytest.approx(expected, abs=1e-6), (
        "teacher scoring bills ceil(completions/batch) serial round trips; crossing one batch "
        "boundary at equal total tokens must add the rollout slope plus exactly one latency"
    )
    assert delta < teacher_lat * 8  # a per-completion serial teacher would swamp the slope


def test_opd_teacher_batch_size_matches_the_worker():
    # flash.cost must not import the training worker (it prices runs on machines without the
    # training stack), so OPD_TEACHER_BATCH_SIZE is a MIRROR of the worker's _TEXT_TEACHER_BATCH_SIZE.
    # A mirror that nothing pins silently goes stale, and the teacher term would then quote a batch
    # shape the worker no longer runs. Read the worker's literal from source rather than importing
    # it, since importing opd_train pulls in the training dependencies this test must not require.
    import ast
    import pathlib

    from flash.cost.analytical import OPD_TEACHER_BATCH_SIZE

    src = pathlib.Path(__file__).resolve().parents[1] / "flash/engine/worker/opd_train.py"
    tree = ast.parse(src.read_text())
    worker_value = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_TEXT_TEACHER_BATCH_SIZE" for t in node.targets
        ):
            worker_value = ast.literal_eval(node.value)
    assert worker_value is not None, (
        "worker no longer defines _TEXT_TEACHER_BATCH_SIZE at module level"
    )
    assert worker_value == OPD_TEACHER_BATCH_SIZE, (
        f"cost model mirrors a stale teacher batch size ({OPD_TEACHER_BATCH_SIZE}) - "
        f"the worker now batches {worker_value}"
    )


def test_revision_pinned_sizing_flows_into_setup_and_required_save(monkeypatch, tmp_path):
    # regression (PR #538 finding 0): cold-start setup + required-save cost must size the PINNED
    # commit's weights, not the catalog default-revision param count, so a run whose revision differs
    # from the catalog is priced on the checkpoint the worker actually downloads and serializes.
    import json
    from types import SimpleNamespace

    from flash.cost.analytical import required_save_overhead_seconds
    from flash.cost.facts import download_weight_gb, total_params_b

    # a pinned commit the Hub reports at 0.87B, within the 5% catalog-drift gate (catalog is 0.9B).
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
    assert total_params_b(SMALL, rev) == pytest.approx(0.87)
    assert total_params_b(SMALL) == pytest.approx(0.9)
    assert download_weight_gb(SMALL, rev) == pytest.approx(1.74)

    # both cold-start setup (download term) and required-save (serialize term) track the smaller pinned
    # weights, so the 0.87B revision quotes strictly cheaper than the 0.9B catalog default.
    pinned = RunConfig(SMALL, "sft", 100, model_revision=rev, save_at_steps=(100,))
    default = RunConfig(SMALL, "sft", 100, save_at_steps=(100,))
    assert setup_seconds(pinned) < setup_seconds(default)
    assert required_save_overhead_seconds(pinned) < required_save_overhead_seconds(default)


def test_sft_run_startup_is_charged_once_per_run_not_per_step(monkeypatch):
    """The SFT startup block must reach the billed wall, be SFT-only, and not scale with steps.

    Asserting the constant equals its own value would pass even if nothing consumed it, so this
    perturbs the constant and requires the quote to move by exactly that delta -- it fails if the
    term is ever unwired from estimate_cost, and fails differently if it is multiplied by steps.
    """
    from flash.cost import analytical

    base_32 = estimate_cost(RunConfig(SMALL, "sft", 32)).train_seconds
    base_64 = estimate_cost(RunConfig(SMALL, "sft", 64)).train_seconds
    grpo_before = estimate_cost(RunConfig(SMALL, "grpo", 32)).train_seconds

    bump = 100.0
    # perturb via the multiplicative constant, then convert to the wall delta it implies.
    scale = (analytical.SFT_RUN_STARTUP_K + bump) / analytical.SFT_RUN_STARTUP_K
    block = analytical.run_startup_seconds(RunConfig(SMALL, "sft", 32), "H100")
    delta = block * (scale - 1.0)
    monkeypatch.setattr(analytical, "SFT_RUN_STARTUP_K", analytical.SFT_RUN_STARTUP_K + bump)

    # charged once: the same delta at 32 and at 64 steps. a per-step term would double.
    assert estimate_cost(RunConfig(SMALL, "sft", 32)).train_seconds == pytest.approx(
        base_32 + delta
    )
    assert estimate_cost(RunConfig(SMALL, "sft", 64)).train_seconds == pytest.approx(
        base_64 + delta
    )
    # SFT-only: grpo and opd already price this through their per-step non-shardable floor, so
    # charging them again would double-count it.
    assert estimate_cost(RunConfig(SMALL, "grpo", 32)).train_seconds == pytest.approx(grpo_before)


def test_sft_run_startup_reaches_the_token_budgeted_branch(monkeypatch):
    """train_tokens takes a second raw_train path; the startup block must be charged there too."""
    from flash.cost import analytical

    cfg = RunConfig(SMALL, "sft", 32, train_tokens=200_000)
    before = estimate_cost(cfg).train_seconds
    block = analytical.run_startup_seconds(cfg, "H100")
    scale = (analytical.SFT_RUN_STARTUP_K + 100.0) / analytical.SFT_RUN_STARTUP_K
    monkeypatch.setattr(analytical, "SFT_RUN_STARTUP_K", analytical.SFT_RUN_STARTUP_K + 100.0)
    assert estimate_cost(cfg).train_seconds == pytest.approx(before + block * (scale - 1.0))


def test_extra_cards_shorten_the_quoted_wall():
    """A billed card that buys no modelled time is a card charged for nothing.

    total_usd multiplies by gpu_count, so if the wall were card-invariant the quote would rank one
    card cheapest by construction rather than by any measurement. Regression for the state before
    this change, where wall_s was bit-identical at 1/2/4 cards while cost scaled exactly 4.00x.
    """
    for method in ("sft", "grpo", "opd"):
        walls = [
            estimate_cost(RunConfig(SMALL, method, 32, gpu_count=n)).wall_clock_seconds
            for n in (1, 2, 4)
        ]
        assert walls[1] < walls[0], f"{method}: a 2nd card must shorten the modelled wall"
        assert walls[2] < walls[1], f"{method}: a 4th card must shorten the modelled wall"


def test_extra_cards_shorten_the_token_budgeted_wall():
    """train_tokens takes the other raw_train branch; sharding must reach it too."""
    tokens = 2_000_000
    one = estimate_cost(
        RunConfig(SMALL, "sft", 32, gpu_count=1, train_tokens=tokens)
    ).wall_clock_seconds
    two = estimate_cost(
        RunConfig(SMALL, "sft", 32, gpu_count=2, train_tokens=tokens)
    ).wall_clock_seconds
    assert two < one


def test_only_the_gpu_bound_half_of_a_step_shards():
    """The non-shardable floor (teacher waits, reward grading, MoE routing) is paid on every card.

    Pins the exact split rather than just "smaller", so a change that sharded the whole step -- and
    would model n cards as n times faster -- fails here instead of quietly overcrediting wide combos.
    """
    cfg = RunConfig(MID, "grpo", 32)
    for gpu in ("RTX 4090", "A100 SXM"):
        gpu_bound, fixed = step_seconds_split(cfg, gpu)
        assert fixed > 0, "this config must have a non-shardable half for the test to mean anything"
        for n in (2, 4):
            expected = gpu_bound / multi_card_speedup(n, gpu) + fixed
            assert seconds_per_step(cfg, gpu, n) == pytest.approx(expected)
            # the floor bounds it from below no matter how many cards are added
            assert seconds_per_step(cfg, gpu, n) > fixed


def test_ranker_and_quote_price_a_combination_identically():
    """The allocator picks the combination and the quote bills it; they must agree on its wall.

    They disagreed by the full speedup factor before this change (ranker 1.41x of one card for a
    2-card 4090, quote 2.00x with an identical wall), because only the ranker applied the speedup.
    Asserting through the allocator's own ranker is what makes this catch a future divergence: a
    test that re-derived the arithmetic locally would pass while the two drifted apart.

    Both sides are anchored to SHIPPED output, not to a locally recomputed formula: the ranker end
    is the allocator's real closure, and the quote end is estimate_cost's own train_seconds. Pinning
    only ``seconds_per_step`` on both sides would leave the quote free to stop calling it (exactly
    the defect being fixed) while this still passed.

    The ranker prices the whole billed wall over the step count, not the marginal step alone: the
    per-run blocks (compile, framework/engine startup, required saves) are amortized into it so the
    key orders candidates exactly as total job cost does. So parity here is against
    ``(per_run / steps + per_step)``, and a ranker that dropped the amortized term would fail this.
    """
    from flash.cost.analytical import (
        compile_seconds,
        required_save_overhead_seconds,
        run_startup_seconds,
    )
    from flash.providers.allocator import _step_cost_ranker
    from flash.providers.base import Candidate, run_config_for_ranking

    model, method = MID, "sft"
    ranker = _step_cost_ranker(model, method, train={}, thinking=False)
    assert ranker is not None, "ranker must be available or this test proves nothing"

    config = run_config_for_ranking(model, method, train={}, thinking=False)
    for gpu, hourly, vram in (("RTX 4090", 0.69, 24), ("A100 SXM", 1.89, 80)):
        for n in (1, 2, 4):
            candidate = Candidate(
                provider="probe", gpu=gpu, hourly_usd=hourly, vram_gb=vram, gpu_count=n
            )
            per_run = (
                compile_seconds(config, gpu)
                + run_startup_seconds(config, gpu)
                + required_save_overhead_seconds(config)
            )
            wall = per_run / config.steps + seconds_per_step(config, gpu, n)
            quoted = wall * candidate.total_hourly_usd / 3600.0
            assert ranker(candidate) == pytest.approx(quoted)

    # and the QUOTE has to be on that same curve. estimate_cost is the shipped surface; comparing
    # its realized per-step wall against the shared helper is what fails if the quote ever stops
    # calling it, which the ranker-side assertion above cannot see on its own.
    steps = 64
    single = estimate_cost(RunConfig(MID, method, steps, gpu_count=1))
    for n in (2, 4):
        e = estimate_cost(RunConfig(MID, method, steps, gpu_count=n))
        assert e.gpu == single.gpu, "same class, or the deltas below compare different hardware"
        # estimate_cost REDEFINES seconds_per_step as raw_train/steps, so it also carries the
        # per-run compile/startup/save blocks -- which do not shard. those are identical across
        # card counts, so they cancel in a DIFFERENCE while corrupting a ratio.
        got = single.train_seconds - e.train_seconds
        want = steps * (seconds_per_step(config, e.gpu, 1) - seconds_per_step(config, e.gpu, n))
        assert got == pytest.approx(want, rel=1e-6)
        assert got > 0, "adding cards must remove time from the quoted wall"


def test_ranking_order_is_identical_to_total_job_cost_order():
    """The ranking key must order candidates EXACTLY as total job cost does.

    This is the contract the key exists to satisfy, and it is a property, not an example: for every
    (method, model, shape, horizon) the argmin of ``key`` must be the argmin of what the job costs.
    A per-step key has that property only while every term is per-step. Once-per-run terms break it
    -- they are paid once regardless of length, so a key that adds one whole gives it the weight of
    a single step and over-weights it by a factor of ``steps``.

    Measured over this grid before the fix: 133 of 264 configurations picked a card that was not the
    cheapest, worst case 1.40x overpay. After amortizing every per-run term: 0. Sweeping the STEP
    COUNT is what makes it falsifiable -- at any single horizon the two agree often enough to look
    fine, and the shipped key was wrong specifically at long ones.
    """
    from flash.cost.analytical import (
        compile_seconds,
        required_save_overhead_seconds,
        run_startup_seconds,
        seconds_per_step,
        step_cost_key,
    )

    # real classes spanning a ~6x rate range and both interconnect families, so a mis-weighted term
    # has somewhere to show up.
    fleet = {
        "RTX 4090": 0.69,
        "RTX 5090": 0.99,
        "A100 PCIe": 1.29,
        "A100 SXM": 1.89,
        "H100": 3.29,
        "H200": 3.99,
    }

    def job_usd(config, gpu: str, rate: float) -> float:
        """What the run bills end to end -- the same terms estimate_cost assembles."""
        per_run = (
            compile_seconds(config, gpu)
            + run_startup_seconds(config, gpu)
            + required_save_overhead_seconds(config)
        )
        return rate * (per_run + config.steps * seconds_per_step(config, gpu, 1)) / 3600.0

    checked = 0
    for method in ("sft", "grpo", "opd"):
        for model in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-35B-A3B"):
            for wants_save in (False, True):
                for steps in (1, 2, 4, 8, 16, 32, 64, 128, 200):
                    # a mid-run save, clamped into the horizon under test -- save_at_steps may not
                    # exceed steps, and the point is to exercise the term at every length.
                    saves = (max(1, steps // 2),) if wants_save else ()
                    config = RunConfig(
                        model,
                        method,
                        steps,
                        seq_len=512,
                        completion_len=128,
                        batch_size=8,
                        group_size=4,
                        save_at_steps=saves,
                    )
                    key = step_cost_key(config)
                    assert key is not None, model

                    by_key = sorted(fleet, key=lambda g: (key(g, fleet[g]), g))
                    by_job = sorted(fleet, key=lambda g: (job_usd(config, g, fleet[g]), g))
                    assert by_key == by_job, f"{method} {model} steps={steps} saves={saves}"
                    checked += 1

    assert checked == 162  # the grid really ran, rather than an empty loop passing vacuously


def test_ranking_key_is_exactly_job_cost_per_step():
    """Not just order-preserving: the key IS the job cost divided by the step count.

    The order test above passes for any monotone transform of job cost, so it cannot tell a correct
    key from one that happens to sort the same way on this fleet. This pins the actual value, which
    is also what makes the key comparable across candidates with different step counts.
    """
    from flash.cost.analytical import (
        compile_seconds,
        required_save_overhead_seconds,
        run_startup_seconds,
        seconds_per_step,
        step_cost_key,
    )

    for method in ("sft", "grpo", "opd"):
        for steps in (1, 8, 64):
            config = RunConfig(
                MID, method, steps, seq_len=512, completion_len=128, batch_size=8, group_size=4
            )
            key = step_cost_key(config)
            for gpu, rate in (("H100", 3.29), ("RTX 5090", 0.99)):
                per_run = (
                    compile_seconds(config, gpu)
                    + run_startup_seconds(config, gpu)
                    + required_save_overhead_seconds(config)
                )
                job = rate * (per_run + steps * seconds_per_step(config, gpu, 1)) / 3600.0
                assert key(gpu, rate) == pytest.approx(job / steps), f"{method} {gpu} {steps}"


def test_ranking_choice_moves_with_the_horizon():
    """The chosen card must be allowed to CHANGE with the step count, and must settle.

    A guard against the fix being silently undone in either direction. If the block were re-divided
    into every step, the key would be horizon-invariant and the winner would never move -- the old
    behaviour. If the block were dropped entirely, the winner would also never move. Only charging
    it once per run makes the short-run and long-run answers legitimately differ.

    It must also CONVERGE: as steps grow the block's share falls monotonically toward zero, so past
    some horizon the ranking stops changing. A key that kept moving would mean a term scaling with
    the step count that should not.
    """
    from flash.cost.analytical import step_cost_key

    fleet = {"RTX 4090": 0.69, "RTX 5090": 0.99, "H100": 3.29, "H200": 3.99}

    def winner(steps: int) -> str:
        config = RunConfig(
            "Qwen/Qwen3.5-0.8B",
            "grpo",
            steps,
            seq_len=512,
            completion_len=128,
            batch_size=8,
            group_size=4,
        )
        key = step_cost_key(config)
        return min(fleet, key=lambda g: (key(g, fleet[g]), g))

    horizons = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    picks = [winner(s) for s in horizons]

    # the block is real: the short-run and long-run winners are not the same card.
    assert picks[0] != picks[-1], f"winner never moved across {horizons}: {picks}"
    # ... and it converges: no further change once the block's share is small.
    assert picks[-4:] == [picks[-1]] * 4, f"ranking still moving at long horizons: {picks}"
    # the winner changes at most once, monotonically -- no oscillation.
    switches = sum(1 for a, b in itertools.pairwise(picks) if a != b)
    assert switches == 1, f"expected a single crossover, saw {switches}: {picks}"

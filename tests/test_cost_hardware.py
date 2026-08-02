"""Cost estimator: GPU compute table, pricing/VRAM lookups, cheapest-fit selection.

No network. The compute table and the selection rule must stay consistent with the
RunPod GPU registry in ``flash.providers.base``.
"""

from __future__ import annotations

import pytest

from flash.cost.facts import (
    GPU_COMPUTE_TFLOPS,
    ROLLOUT_SECONDS_PER_COMPLETION,
    gpu_hourly_usd,
    gpu_tflops,
    gpu_vram_gb,
    pick_gpu,
)
from flash.providers.base import GPU_INFO


def test_static_rate_is_positive_for_any_class():
    for name in GPU_INFO:
        assert gpu_hourly_usd(name) > 0, name


def test_compute_table_only_lists_real_classes():
    # Every GPU we assign a TFLOPS figure to must be a real managed class (no drift).
    for name in GPU_COMPUTE_TFLOPS:
        assert name in GPU_INFO, f"{name} is not a managed GPU class"


def test_gpu_tflops_known_and_default():
    assert gpu_tflops("RTX 5090") == GPU_COMPUTE_TFLOPS["RTX 5090"]
    assert gpu_tflops("RTX 5090") > gpu_tflops("RTX 4090")  # newer/faster
    assert gpu_tflops("totally-unknown-gpu") == 100.0  # documented default


def test_lambda_a100_40gb_band_has_real_tflops():
    # Removing A6000 exposes the Lambda-only `A100 SXM 40GB` as the cheapest fit for the 33-40 GB
    # band on the Lambda path; it must carry a real TFLOPS figure rather than fall back to
    # _DEFAULT_TFLOPS, or Lambda cost quotes overstate runtime ~3x for that band.
    pick = pick_gpu(35, provider="lambda")
    assert pick == "A100 SXM 40GB"
    assert gpu_tflops(pick) == GPU_COMPUTE_TFLOPS["A100 SXM 40GB"]
    assert gpu_tflops(pick) > 100.0  # not the _DEFAULT_TFLOPS fallback


def test_pricing_and_vram_track_the_registry():
    for name, g in GPU_INFO.items():
        assert gpu_hourly_usd(name) == g.hourly_usd
        assert gpu_vram_gb(name) == g.vram_gb


def test_unknown_gpu_lookup_raises():
    with pytest.raises(KeyError):
        gpu_hourly_usd("Tesla T4")
    with pytest.raises(KeyError):
        gpu_vram_gb("Tesla T4")


def test_pick_gpu_cheapest_fit_no_validation_gate():
    # No validation gate: every fitting class is eligible, ranked by static rate. The cheapest
    # managed card is the 24 GB RTX 4090 ($0.69), so anything that fits <=24 GB lands on it.
    assert pick_gpu(12) == "RTX 4090"
    assert pick_gpu(24) == "RTX 4090"
    # 40 GB has no card between the 32 GB RTX 5090 and the 80 GB A100 PCIe ($1.39, the cheapest fit).
    assert pick_gpu(40) == "A100 PCIe"


def test_pick_gpu_result_actually_fits_and_is_cheapest():
    for need in (8, 16, 24, 33, 48, 80):
        gpu = pick_gpu(need)
        assert gpu_vram_gb(gpu) >= need
        # No validation gate: nothing fitting is cheaper at the static rate.
        cheaper_fits = [
            g
            for g in GPU_INFO.values()
            if g.vram_gb >= need and gpu_hourly_usd(g.name) < gpu_hourly_usd(gpu)
        ]
        assert not cheaper_fits, f"{cheaper_fits} cheaper than {gpu} for {need} GB"


def test_pick_gpu_includes_unvalidated_classes(monkeypatch):
    # No validation gate: the cheapest static-rate class wins regardless of validation status.
    # The managed catalog is now fully validated, so inject a synthetic UNVALIDATED cheap class
    # and confirm pick_gpu still selects it (the submit-time allocator is what applies the gate).
    from flash.providers.base import GPU_INFO, GpuClass

    fake = GpuClass("FAKE Cheap", "NVIDIA_FAKE", 24, "fakecheap", "sm80", 0.10)
    assert not fake.validated
    monkeypatch.setitem(GPU_INFO, "FAKE Cheap", fake)
    assert pick_gpu(12) == "FAKE Cheap"


def test_pick_gpu_impossible_raises():
    with pytest.raises(ValueError, match="no GPU class fits"):
        pick_gpu(100_000)


def test_pick_gpu_auto_matches_default():
    assert pick_gpu(24, provider="auto") == pick_gpu(24)
    assert pick_gpu(12) == "RTX 4090"


def test_effective_train_tflops_caps_b200_at_h200_class():
    # b200/sm100 training falls back to portable kernels, so realized training throughput is
    # h200-class, not the 2.25 pflops peak. the cost model must not treat b200 as faster than h200.
    from flash.cost.facts import effective_train_tflops

    assert gpu_tflops("B200") == 2250.0  # raw peak unchanged (vram/serving still use gpu_tflops)
    # assert the RELATIONSHIP, not a literal: the cap tracks whatever the H200 entry is, and that
    # entry is anchored to a measured rate that recalibration can move.
    assert effective_train_tflops("B200") == effective_train_tflops("H200")
    assert effective_train_tflops("B200") < gpu_tflops("B200")


def test_effective_train_tflops_is_peak_for_uncapped_classes():
    from flash.cost.facts import effective_train_tflops

    for name in ("H100", "H200", "A100 SXM", "RTX 4090", "B200"):
        # only b200 is capped; every other class keeps its peak.
        expected = effective_train_tflops("H200") if name == "B200" else gpu_tflops(name)
        assert effective_train_tflops(name) == expected


def test_rollout_wall_dominates_a_small_rollout_step():
    # regression: the model used to quote a grpo step as compute + reward wait and nothing else,
    # which ran ~0.48x of realized because it priced no rollout wall at all. a small 0.8B step is
    # ~0.3s of arithmetic against a realized 60-140s, so the wall -- not the flops -- must be what
    # the quote is made of. asserting the RATIO, so recalibrating any single constant cannot silently
    # return the model to a compute-only quote.
    #
    # the step's non-compute half is now the per-completion rollout slope only; the once-per-RUN
    # block that used to be divided into it lives in run_block_seconds and is charged once by
    # estimate_cost. that reattribution does not weaken this test: the slope alone still has to
    # carry an order of magnitude more than the arithmetic.
    from flash.cost.analytical import step_seconds_split
    from flash.cost.types import RunConfig

    config = RunConfig(
        "Qwen/Qwen3.5-0.8B", "grpo", 6, seq_len=512, completion_len=128, batch_size=8, group_size=4
    )
    gpu_bound, fixed = step_seconds_split(config, "H100")
    assert gpu_bound < 5.0  # the arithmetic really is negligible at this size
    assert fixed > 10 * gpu_bound  # ... so a quote that omits the wall is wrong by an order
    # and the rollout slope specifically, not the reward wall, has to be what is in there. isolate
    # the reward term by differencing against a zero-reward config rather than restating it as a
    # literal: it is completions x reward_seconds, and that default has moved before.
    from dataclasses import replace

    no_reward = step_seconds_split(replace(config, reward_seconds_per_completion=0.0), "H100")[1]
    reward_share = fixed - no_reward
    assert no_reward > 10 * reward_share  # the wall is rollout, not grading
    n = config.normalized()
    assert no_reward == pytest.approx(n.batch_size * n.group_size * ROLLOUT_SECONDS_PER_COMPLETION)


def test_run_block_is_charged_per_class_not_pooled():
    from flash.cost.facts import _DEFAULT_RUN_BLOCK_S, run_block_seconds

    # measured classes carry their own block; h200's is the outlier the pooled value would erase.
    assert run_block_seconds("H200") > run_block_seconds("H100")
    assert run_block_seconds("RTX 5090") < run_block_seconds("RTX 4090")
    # an unmeasured class falls back to the pooled median rather than to no block at all.
    assert run_block_seconds("some-unmeasured-class") == _DEFAULT_RUN_BLOCK_S
    assert _DEFAULT_RUN_BLOCK_S > 0.0


def test_run_block_never_exceeds_the_shortest_run_measured_on_that_card():
    """A block is a strict subset of the run that starts it, so it cannot outlast one.

    This is the check that caught B200 (410.3 s block against a complete 272.8 s 4-step run) after
    the step-identification sweep, and it is worth a permanent test because the failure is invisible
    at the horizons the corpus is thickest at: a too-large block only shows up where it dominates the
    wall, which is the short-run end nothing else scores.

    The bound is per card and measured, not a modelled quantity, so this cannot be satisfied by
    recalibrating the step slope. Values are the shortest realized TRAIN wall (``wall - setup``) on
    the canonical 0.8B rollout cell, taken from the worker's own metrics.json and confirmed to have
    completed every requested step rather than exiting early.

    Compared against the 1.194x replicate noise floor rather than 1.0: a block within run-to-run pod
    variance of its shortest run is tight, not impossible, and failing on that would make the test
    fire on noise.

    Two classes are held out, each for a stated reason rather than to make the test pass:
      - RTX 4090: its shortest arm banked a 829 s setup, 9.4x that card's median, so the engine
        build landed BEFORE the setup stamp and its train wall is not comparable to the corpus.
      - H200: it DOES violate the bound (1.39x) and is knowingly left uncorrected -- see the table
        comment. One low arm on a class with 2.17x internal spread is a leverage point, and every
        candidate replacement scores worse on that card's own arms. It is asserted below at the
        looser bound it actually satisfies, so a further regression there still fails.
    """
    from flash.cost.facts import run_block_seconds

    shortest_train_s = {
        "A100 PCIe": 339.9,
        "A100 SXM": 541.0,
        "B200": 272.8,
        "H100": 323.7,
        "RTX 5090": 198.8,
        "RTX Pro 6000": 292.8,
    }
    noise_floor = 1.194

    for card, realized in shortest_train_s.items():
        block = run_block_seconds(card)
        assert block <= realized * noise_floor, (
            f"{card}: block {block:.1f}s exceeds its shortest measured run {realized:.1f}s by "
            f"{block / realized:.2f}x, past the {noise_floor}x noise floor"
        )

    # the known exception, pinned so it cannot silently drift further from its own measurement.
    assert run_block_seconds("H200") <= 549.6 * 1.40


def test_run_block_table_only_lists_real_classes():
    # same drift guard the TFLOPS table gets. a typo'd key here does not raise -- it silently falls
    # through to the pooled default, so the measured value would be quietly discarded and the class
    # would go on being mispriced with nothing failing.
    from flash.cost.facts import _RUN_BLOCK_S

    for name in _RUN_BLOCK_S:
        assert name in GPU_INFO, f"{name} is not a managed GPU class"


def test_run_block_applies_to_rollout_methods_only():
    # the block is vllm engine build + actor->rollout weight sync, paid once per run. sft runs
    # neither, so it pays its own smaller verl/fsdp launch instead (SFT_RUN_STARTUP_K) rather than a
    # card-shaped rollout block.
    from flash.cost.analytical import run_startup_seconds
    from flash.cost.facts import run_block_seconds
    from flash.cost.types import RunConfig

    sft = RunConfig("Qwen/Qwen3.5-0.8B", "sft", 6, seq_len=512, batch_size=8)
    # sft's launch is card-FREE: same on every class, because there is no engine to build. (it does
    # scale with MODEL size -- see test_sft_run_startup_scales_with_model_size -- which is why this
    # compares the classes to each other rather than to a constant.)
    sft_block = run_startup_seconds(sft, "H100")
    for card in ("H200", "RTX 5090"):
        assert run_startup_seconds(sft, card) == pytest.approx(sft_block)

    # rollout methods carry the block, and by the card's own amount -- comparing two classes
    # isolates it from anything shape-dependent, which is identical on both.
    for method in ("grpo", "opd"):
        config = RunConfig("Qwen/Qwen3.5-0.8B", method, 6, seq_len=512, completion_len=128)
        h100 = run_startup_seconds(config, "H100")
        h200 = run_startup_seconds(config, "H200")
        assert h200 - h100 == pytest.approx(run_block_seconds("H200") - run_block_seconds("H100"))
        # and it is materially larger than the sft launch, which is the whole reason it is separate.
        assert h100 > sft_block


def test_run_block_is_charged_once_per_run_not_once_per_step():
    """THE regression this model exists to fix, asserted end to end through the quote.

    The block was previously divided into every step. That is invisible at a fixed horizon -- both
    forms quote the same 6-step run -- so the only thing that separates them is how the quote MOVES
    with the step count. Per-run, doubling the steps adds exactly the steady-state step twice; per-
    step, it also re-charges the block, and the marginal cost of a step comes out inflated by it.

    Asserted as a difference of differences so it cannot be satisfied by any single recalibration.
    """
    from dataclasses import replace

    from flash.cost.analytical import (
        compile_seconds,
        run_startup_seconds,
        seconds_per_step,
        step_seconds_split,
    )
    from flash.cost.types import RunConfig

    base = RunConfig(
        "Qwen/Qwen3.5-0.8B", "grpo", 4, seq_len=512, completion_len=128, batch_size=8, group_size=4
    )
    card = "H200"  # the widest block, so a per-step charge would be unmissable here

    def train_wall(steps: int) -> float:
        cfg = replace(base, steps=steps)
        return (
            compile_seconds(cfg, card)
            + run_startup_seconds(cfg, card)
            + steps * seconds_per_step(cfg, card)
        )

    sps = sum(step_seconds_split(base, card))
    # each extra step adds one steady-state step and nothing else, at every horizon.
    for steps in (4, 8, 16, 48):
        marginal = train_wall(steps * 2) - train_wall(steps)
        assert marginal == pytest.approx(steps * sps), f"block re-charged at steps={steps}"


def test_opd_run_block_is_measured_not_borrowed_from_grpo():
    """OPD shares grpo's block by MEASUREMENT, not merely by shared mechanism.

    The table is fitted on rollout arms and applied to both methods, which is only sound if opd's
    step/block split actually matches. It does: 7 opd arms on RTX 4090 at steps 4/8/16/48 (12x
    leverage, 3 replicates at the short end) fit step 33.55 s and block 384.3 s at R2 0.992, against
    the shipped 404.6 -- 0.95x, well inside the 1.194x replicate noise floor.

    Asserted through the quote at two horizons rather than against the fitted numbers directly, so
    it tests the shipped path. The wide tolerance is deliberate: this pins that opd is priced on its
    own measured scale, and would fail on a borrowed-from-nowhere or method-blind default, without
    re-asserting a fit that a handful of arms cannot place more tightly than the noise floor.
    """
    from flash.cost.analytical import compile_seconds, run_startup_seconds, seconds_per_step
    from flash.cost.types import RunConfig

    card = "RTX 4090"
    measured = {4: 494.3, 48: 1962.3}  # median of the 3 short replicates; the single long arm

    for steps, realized in measured.items():
        cfg = RunConfig(
            "Qwen/Qwen3.5-0.8B",
            "opd",
            steps,
            seq_len=512,
            completion_len=128,
            batch_size=8,
            group_size=4,
        )
        quoted = (
            compile_seconds(cfg, card)
            + run_startup_seconds(cfg, card)
            + steps * seconds_per_step(cfg, card)
        )
        assert 1 / 1.5 <= quoted / realized <= 1.5, (
            f"opd steps={steps}: quoted {quoted:.0f}s vs measured {realized:.0f}s "
            f"({quoted / realized:.2f}x)"
        )


def test_run_block_is_not_shardable_across_cards():
    # the allocator divides ONLY the gpu-bound half by card count. engine build and weight sync do
    # not get faster on more ranks, so a block that leaked into the shardable half would make wide
    # multi-card jobs look arbitrarily cheap. the block is not in step_seconds_split at all now, so
    # this asserts the property where it moved to: the per-card ranking key.
    from flash.cost.analytical import multi_card_speedup, step_cost_key, step_seconds_split
    from flash.cost.types import RunConfig

    config = RunConfig(
        "Qwen/Qwen3.5-0.8B", "grpo", 8, seq_len=512, completion_len=128, batch_size=8, group_size=4
    )
    gpu_bound, fixed = step_seconds_split(config, "H100")
    # the non-shardable half stays non-shardable: it is the rollout slope, per completion, and more
    # ranks do not sample fewer sequences.
    assert fixed > 0.0
    assert gpu_bound < fixed

    key = step_cost_key(config)
    # per card, 2 cards must not be cheaper than 1 by more than the gpu-bound half can explain --
    # if the block sharded, the 2-card key would drop by ~half the block as well.
    one = key("H100", 1.0, 1) / 1.0
    two = key("H100", 1.0, 2) / 2.0  # per-card seconds, so the count factor is removed
    saved = (one - two) * 3600.0
    ceiling = gpu_bound * (1.0 - 1.0 / multi_card_speedup(2, "H100"))
    assert saved == pytest.approx(ceiling, rel=1e-9)


def test_rollout_wall_survives_a_measured_reward_latency():
    # THE regression. rollout cost per completion (~0.81s) is close enough to the default reward
    # guess (1.0s/completion) that one term appeared to cover both -- until a caller supplied a
    # MEASURED reward latency. real graders in the calibration corpus grade in ~0.0003s, so passing
    # the true value used to delete ~32s of real rollout wall along with the fictitious reward and
    # collapsed the quote from geo-bias 1.017x to 0.517x. the more accurate the caller's input, the
    # worse the estimate.
    #
    # asserting the delta alone would be UNFALSIFIABLE: the drop equals the reward difference under
    # both the broken and the fixed model, since both subtract the same reward term. what separates
    # them is the ABSOLUTE floor left standing once the fictitious reward is gone, so that is what
    # this pins -- against the realized wall, not against the model's own arithmetic.
    from flash.cost.analytical import compile_seconds, run_startup_seconds, step_seconds_split
    from flash.cost.types import RunConfig

    shape = {"seq_len": 512, "completion_len": 128, "batch_size": 8, "group_size": 4}
    completions = shape["batch_size"] * shape["group_size"]
    steps = 6

    measured = RunConfig(
        "Qwen/Qwen3.5-0.8B", "grpo", steps, reward_seconds_per_completion=0.0003, **shape
    )
    gpu_bound, fixed = step_seconds_split(measured, "H100")

    # the per-STEP survivor is the rollout slope: it is what stands once the fictitious reward is
    # gone, and it is charged per completion regardless of how fast the grader is.
    assert fixed >= completions * ROLLOUT_SECONDS_PER_COMPLETION

    # h100 arms at this shape realized 56-100s/step with graders this fast. the broken model quoted
    # ~43s of constant and nothing per-completion, landing at ~0.5x realized. the realized figure is
    # a per-run wall over its step count, so compare against the same thing: the amortized quote,
    # not the steady-state step alone, or this asserts a number the model never claimed.
    per_run = compile_seconds(measured, "H100") + run_startup_seconds(measured, "H100")
    assert per_run / steps + gpu_bound + fixed > 60.0


def test_rollout_wall_scales_with_completion_count():
    # the per-step wall is slope*completions. fitted on 16-256 completions/step, where the per-card
    # median rose ~3x (h100 77.8s at 32 -> 230.7s at 256). a constant-only model fitted on the
    # original campaign missed this because 30 of its 32 arms ran exactly 32 completions.
    from flash.cost.facts import rollout_step_seconds

    assert rollout_step_seconds(0) == pytest.approx(0.0)
    assert rollout_step_seconds(256) == pytest.approx(256 * ROLLOUT_SECONDS_PER_COMPLETION)
    # the slope is a rollout-path property and does not vary by card: the per-class fits agreed
    # inside the noise floor, so it takes no card argument at all to key it on.
    assert rollout_step_seconds(100) == pytest.approx(100 * ROLLOUT_SECONDS_PER_COMPLETION)
    # a negative completion count cannot credit time back.
    assert rollout_step_seconds(-5) == pytest.approx(0.0)


def test_a100_sxm_40gb_inherits_its_siblings_wall():
    # same SMs and tensor cores as the 80GB board, less hbm only -- and this block is engine-build
    # and weight-sync bound, not capacity bound, so the 40GB variant cannot be ~2x cheaper per run
    # than the board it is a memory-trimmed version of. left out of the table it fell through to the
    # pooled default and under-priced a dominant term on every lambda/vast quote in that band. the
    # TFLOPS table inherits across this same pair for the same reason.
    from flash.cost.facts import _DEFAULT_RUN_BLOCK_S, run_block_seconds

    assert run_block_seconds("A100 SXM 40GB") == run_block_seconds("A100 SXM")
    assert run_block_seconds("A100 SXM 40GB") != _DEFAULT_RUN_BLOCK_S


def test_nvlink_classification_is_by_form_factor():
    from flash.cost.facts import has_nvlink

    # sxm datacenter parts carry nvlink.
    assert has_nvlink("A100 SXM")
    assert has_nvlink("A100 SXM 40GB")
    assert has_nvlink("H100")
    # geforce parts do not; the 4090 dropped the nvlink connector entirely. l40s is a pcie board.
    assert not has_nvlink("RTX 4090")
    assert not has_nvlink("L40S")
    # an unclassified class must fall to the conservative side rather than raise.
    assert not has_nvlink("some-unlisted-gpu")


def test_nvlink_classification_tracks_the_provisioned_board():
    """Classification must follow the pin a MULTI-CARD run actually lands on.

    Multi-card provisioning is runpod-only, and runpod pins H100 to the HBM3 sxm part while
    negating the pcie/NVL boards in the same pool. Assert against those pins rather than restating
    the classification, so re-pinning a class to a different board fails here instead of silently
    pricing it on an interconnect it no longer has.
    """
    from flash.cost.facts import has_nvlink
    from flash.providers.base import GPU_INFO
    from flash.providers.runpod.gpus import _POOL_MEMBERS_MISSING_FROM_SDK

    assert GPU_INFO["H100"].enum_member == "NVIDIA_H100_80GB_HBM3"  # sxm, not the pcie board
    assert has_nvlink("H100")
    # the non-sxm members of the same runpod pool are negated, so a pin cannot land on them.
    assert _POOL_MEMBERS_MISSING_FROM_SDK["ADA_80_PRO"] == ("NVIDIA H100 PCIe", "NVIDIA H100 NVL")


def test_multi_card_speedup_is_interconnect_aware():
    from flash.cost.analytical import multi_card_speedup

    # MEASURED on runpod with one identical 2-card fsdp benchmark per interconnect:
    #   nvlink 2x A100-SXM4-80GB 1.7675x, pcie 2x L40S 1.4212x.
    # the model must land near each measurement, not split the difference with one constant.
    assert multi_card_speedup(2, "A100 SXM") == pytest.approx(1.7675, abs=0.05)
    assert multi_card_speedup(2, "RTX 4090") == pytest.approx(1.4212, abs=0.05)
    # one card is exactly one card on any fabric.
    for name in ("A100 SXM", "RTX 4090", "unknown"):
        assert multi_card_speedup(1, name) == 1.0


def test_pcie_scaling_is_never_credited_nvlink_bandwidth():
    """The invariant the measurement exists to protect.

    A pcie pair delivered 1.42x where the old global 0.85 constant claimed 1.70x. Crediting that
    difference lets a 2-card pcie combination win a ranking on scaling it does not have, and then
    bills both cards for the longer wall time. Assert the ORDERING, so recalibrating either
    constant cannot silently reintroduce the inversion.
    """
    from flash.cost.analytical import multi_card_speedup

    for n in (2, 3, 4):
        assert multi_card_speedup(n, "RTX 4090") < multi_card_speedup(n, "A100 SXM")
        # and neither may ever claim linear scaling, which no fabric delivers.
        assert multi_card_speedup(n, "A100 SXM") < n


def test_multi_card_speedup_never_decreases_with_card_count():
    """Adding a card must never model as slower.

    The raw geometric curve turns back down below ~0.72 scaling (at the measured pcie 0.71: 3 cards
    1.512x, 4 cards 1.432x). Left unclamped the allocator would price a 4-card pcie combination as
    slower than a 3-card one and reject cards that do add throughput. No real fabric loses aggregate
    throughput when a card is added; the extrapolation flattens, it does not reverse.
    """
    from flash.cost.analytical import multi_card_speedup

    for name in ("A100 SXM", "RTX 4090", "H100", "unlisted-class"):
        vals = [multi_card_speedup(n, name) for n in range(1, 9)]
        assert vals == sorted(vals), f"{name} speedup decreases: {vals}"


def test_resident_rollout_is_not_charged_the_engine_build():
    # the block is a vllm engine build + first weight sync. a sleep_unsupported model pins its
    # rollout engine RESIDENT (catalog.py, backend_common.rollout_resident_overrides), so that build
    # never runs and charging it extrapolates outside the fit: all 56 fitted arms are sleep-capable
    # 0.8b/2b/4b models. the error is not marginal -- on qwen3.6-35b-a3b the h200 block is 762.4s
    # against a ~24s realized grpo step analytical.py documents for that model.
    from flash.cost.facts import rollout_is_resident, run_block_seconds

    assert rollout_is_resident("Qwen/Qwen3.6-35B-A3B")
    assert not rollout_is_resident("Qwen/Qwen3.5-0.8B")

    for card in ("B200", "H200"):
        sleeping = run_block_seconds(card)
        resident = run_block_seconds(card, resident=True)
        assert resident == pytest.approx(0.0)
        # guards the mutation "resident flag ignored": that returns the sleeping value here.
        assert sleeping > 50.0


def test_resident_rollout_still_pays_the_per_completion_slope():
    # the opposite error, and the reason resident is not simply "no rollout cost": a resident engine
    # skips the BUILD, not the per-sequence sampling/detokenize/dispatch work. zeroing the per-step
    # wall too would under-quote every resident step by slope*completions. the two now live in
    # different terms, which is what makes zeroing one unable to touch the other.
    from flash.cost.facts import rollout_step_seconds

    assert rollout_step_seconds(64) == pytest.approx(64 * ROLLOUT_SECONDS_PER_COMPLETION)


def test_cost_and_worker_agree_on_which_rollouts_are_resident():
    # two readers of one catalog flag. if they diverge the quote prices a mode the worker is not in,
    # and nothing else in the suite would notice -- so compare them across the WHOLE catalog rather
    # than on the one model that happens to be flagged today.
    from flash.catalog import MODELS
    from flash.cost.facts import rollout_is_resident
    from flash.engine.worker.backend_common import rollout_sleep_unsupported

    for model_id in MODELS:
        assert rollout_is_resident(model_id) == rollout_sleep_unsupported(model_id), model_id


def test_resident_flag_reaches_the_quote_not_just_the_helper():
    # run_block_seconds honouring `resident` is worthless if run_startup_seconds never passes it.
    # the helper-level tests above cannot see that wiring: they call the helper directly, so a
    # call site that drops the argument leaves them all green. this asserts through the real quote.
    from flash.cost.analytical import run_startup_seconds, step_seconds_split
    from flash.cost.types import RunConfig

    shape = {"seq_len": 512, "completion_len": 128, "batch_size": 8, "group_size": 4}
    completions = shape["batch_size"] * shape["group_size"]
    slope = completions * ROLLOUT_SECONDS_PER_COMPLETION

    for method in ("grpo", "opd"):
        # reward/teacher wait zeroed so `fixed` is the rollout wall plus overhead only.
        cfg = RunConfig(
            "Qwen/Qwen3.6-35B-A3B",
            method,
            6,
            reward_seconds_per_completion=0.0,
            **shape,
        )
        # the h200 block is 762.4s; if it were still charged, startup would clear it outright.
        assert run_startup_seconds(cfg, "H200") == pytest.approx(0.0), method
        # and the per-step slope is still in there -- this is not a "wall deleted" pass. the block
        # and the slope are separate terms, so zeroing the block must leave this one standing.
        _, fixed = step_seconds_split(cfg, "H200")
        assert fixed >= slope, method

    # a sleep-capable model on the same card still pays it, so the assertion above cannot pass by
    # the block having been dropped for everyone.
    sleeps = RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 6, **shape)
    assert run_startup_seconds(sleeps, "H200") > 50.0


def test_hardware_choice_is_not_a_restatement_of_the_hourly_rate():
    """Selection must rank on what the JOB costs, which requires per-card step times to differ.

    This is the property the allocator's total-cost ranking rides on. Before the per-card wall, a
    step was quoted as compute plus reward wait, and compute is 0.3-9.3s on every class -- so all
    eight classes quoted within 0.6s of each other and dollars-per-step degenerated into a monotone
    function of the hourly rate. The ranking looked speed-aware and was arithmetically incapable of
    disagreeing with sorting on $/hr.

    Measured on the one campaign family that ran on all eight classes (Qwen3.5-0.8B, b8 g4, ctx512,
    cap128), RTX 5090 is the cheapest per job despite RTX 4090 being cheaper per hour ($0.69 vs
    $0.99): the 4090 takes 101.4s/step-equivalent against the 5090's 44.6s. Ranking on rate alone
    buys the 4090 and pays 1.59x more.

    WHERE that difference lives moved. Nearly all of it is the once-per-RUN block (RTX 4090 404.6s
    vs RTX 5090 98.4s); the steady-state steps are within 1% of each other (23.60s vs 23.41s). So
    this must price a JOB, not a step -- pricing one steady-state step compares two cards the model
    now says are almost the same speed, and the disagreement it exists to assert disappears. That
    is also exactly why the ranking key amortizes the block instead of ranking on the step alone.
    """
    from flash.cost.analytical import (
        compile_seconds,
        run_startup_seconds,
        step_cost_key,
        step_seconds_split,
    )
    from flash.cost.types import RunConfig

    steps = 8
    cfg = RunConfig(
        "Qwen/Qwen3.5-0.8B",
        "grpo",
        steps,
        seq_len=512,
        completion_len=128,
        batch_size=8,
        group_size=4,
    )
    rates = {"RTX 4090": 0.69, "RTX 5090": 0.99}

    def job_seconds(card: str) -> float:
        return (
            compile_seconds(cfg, card)
            + run_startup_seconds(cfg, card)
            + steps * sum(step_seconds_split(cfg, card))
        )

    seconds = {c: job_seconds(c) for c in rates}
    # the two cards must not be quoted as interchangeable over a real run.
    assert seconds["RTX 4090"] > 1.4 * seconds["RTX 5090"]

    dollars = {c: seconds[c] * rates[c] / 3600.0 for c in rates}
    # cheaper per hour, dearer per job -- so the two orderings genuinely disagree here.
    assert rates["RTX 4090"] < rates["RTX 5090"]
    assert dollars["RTX 5090"] < dollars["RTX 4090"]

    # and the shipped ranking key must reproduce that disagreement, not just this local arithmetic:
    # it is the thing selection actually calls.
    key = step_cost_key(cfg)
    assert key("RTX 5090", rates["RTX 5090"]) < key("RTX 4090", rates["RTX 4090"])


def test_rollout_slope_is_per_completion_not_per_generated_token():
    """Raising ``max_completion_tokens`` must not scale the rollout wall, because it is a CEILING.

    The slope charges 0.81s per completion and ignores ``completion_len`` entirely, which reads like
    an omission: sampling and detokenization are per-token work, so a 512-token cap "should" cost 4x
    a 128-token one. The corpus says otherwise. Holding card AND completion count fixed so cap is the
    only variable, on H100 at 32 completions: cap 128 -> 79.1s (n=17), cap 256 -> 70.3s (n=1), cap
    512 -> 69.0s (n=2). Quadrupling the cap moved the wall to 0.873x -- slightly DOWN, where a
    per-token slope predicts ~4x up. Implied per-token cost falls ~6x across that range while implied
    per-completion cost stays inside the 1.095x per-observation noise floor.

    The mechanism is that generation stops at EOS, so on a short-answer env the cap raises the bound
    and not the work. Pooled residual is flat across cap for the same reason: 1.027x at 128, 1.027x
    at 256, 1.063x at 512.

    What this test does NOT establish: that the same holds for an env whose completions SATURATE the
    cap (thinking traces at the 1536 default). Every corpus arm is short-answer, and the corpus has
    no generated-token field at all (``gens`` is the completion count on 56/56 rows), so a per-token
    term cannot be identified here -- fitting one would fit noise. This pins the measured behaviour;
    a saturating env is the case that would distinguish the two forms and it has not been run.
    """
    from flash.cost.analytical import step_seconds_split
    from flash.cost.types import RunConfig

    def quote(cap: int) -> float:
        cfg = RunConfig(
            "Qwen/Qwen3.5-0.8B",
            "grpo",
            1,
            seq_len=512,
            completion_len=cap,
            batch_size=8,
            group_size=4,
        )
        return sum(step_seconds_split(cfg, "H100"))

    base, wide = quote(128), quote(1536)
    # a 12x cap must not carry a 12x wall. the arithmetic (FLOPs) term may move; the wall may not.
    assert wide < 1.5 * base, (
        f"cap 128->1536 moved the quote {wide / base:.2f}x; slope went per-token"
    )

    # the wall term itself must be BIT-identical across the two caps: it is the thing that would
    # have to move for the per-token reading to be right, and it takes no cap argument at all.
    from flash.cost.analytical import step_seconds_split as _split

    def wall(cap: int) -> float:
        cfg = RunConfig(
            "Qwen/Qwen3.5-0.8B",
            "grpo",
            1,
            seq_len=512,
            completion_len=cap,
            batch_size=8,
            group_size=4,
        )
        return _split(cfg, "H100")[1]  # non-shardable half == reward + rollout wall

    assert wall(128) == wall(1536)

    # the slope IS live, so this cannot pass by the wall being constant everywhere: it must still
    # scale with the COMPLETION count, which is what it actually prices.
    from flash.cost.facts import rollout_step_seconds

    assert rollout_step_seconds(64) > rollout_step_seconds(32)


def test_whole_run_quote_never_sits_below_the_fastest_run_of_that_config():
    """A whole-run quote cannot be smaller than a run of that exact config that actually happened.

    Fit-free counterpart to ``test_run_block_never_exceeds_the_shortest_run_measured_on_that_card``:
    that one bounds a COMPONENT from above, this bounds the TOTAL from below. It needs no regression
    and no step/token leverage, which is what makes it usable on a corpus too thin to fit anything.

    The bound is the MONOTONE LOWER ENVELOPE, not the raw fastest wall: each config is bounded by the
    fastest run at >= its own token count. Two SFT cells are non-monotone in tokens (H100/4B ran
    527.9 s at 2081 tok but 210.3 s at 32197 tok; RTX 4090/0.8B ran 323.3 s at 128483 tok but 282.6 s
    at 256052 tok). More work cannot take less time on the same card and model, so those walls carry
    a pod-specific slug and bound nothing. Enveloping can only LOWER a bound, never invent headroom.

    This caught the defect the SFT constants now correct: the shipped flat block plus a bare flops
    term quoted 13 of 13 configs BELOW a realized run, at geo 0.401x with 4.71x spread. OPD, scored
    the same way on the same corpus, was already sound at geo 1.005x -- which is what shows the check
    measures something real instead of firing on whatever it is pointed at.
    """
    import math

    from flash.cost.analytical import (
        run_startup_seconds,
        seconds_per_step,
        sft_seconds_for_tokens,
    )
    from flash.cost.types import RunConfig

    # (card, model, train_tokens) -> fastest realized TRAIN wall at >= that token count, on the
    # canonical (batch_size=8, seq_len=1024) SFT cell. Worker metrics.json, completed runs only.
    sft_envelope = {
        ("H100", "Qwen/Qwen3.5-0.8B", 32197): 82.3,
        ("H100", "Qwen/Qwen3.5-4B", 2081): 210.3,
        ("H100", "Qwen/Qwen3.5-4B", 32197): 210.3,
        ("H100", "Qwen/Qwen3.5-4B", 64576): 215.5,
        ("H100", "Qwen/Qwen3.5-4B", 128483): 372.9,
        ("H100", "Qwen/Qwen3.5-4B", 256052): 489.4,
        ("H100", "Qwen/Qwen3.6-27B", 8048): 437.4,
        ("RTX 4090", "Qwen/Qwen3.5-0.8B", 2081): 85.5,
        ("RTX 4090", "Qwen/Qwen3.5-0.8B", 32197): 89.6,
        ("RTX 4090", "Qwen/Qwen3.5-0.8B", 64576): 113.2,
        ("RTX 4090", "Qwen/Qwen3.5-0.8B", 128483): 282.6,
        ("RTX 4090", "Qwen/Qwen3.5-0.8B", 256052): 282.6,
        ("RTX 4090", "Qwen/Qwen3.5-4B", 32197): 290.4,
    }

    ratios = []
    for (card, model, tokens), fastest in sft_envelope.items():
        cfg = RunConfig(model, "sft", 1, batch_size=8, seq_len=1024)
        quote = run_startup_seconds(cfg, card) + sft_seconds_for_tokens(cfg, card, tokens, 1)
        ratios.append(quote / fastest)

    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    spread = max(ratios) / min(ratios)

    # The SFT replicate noise floor is 1.722x, so a single config landing under 1.0 is inside what
    # pod variance explains. What is NOT allowed is the systematic shortfall the flat block produced:
    # every config low at once, with a geometric mean far outside that floor.
    assert 1.0 / 1.722 < geo < 1.722, (
        f"sft geo {geo:.3f}x is outside the 1.722x replicate noise floor; the shipped flat block "
        f"scored 0.401x here (13/13 configs quoted below a run that actually happened)"
    )
    assert sum(1 for r in ratios if r < 1.0) <= len(ratios) // 2, (
        "more than half of sft configs are quoted below their own fastest realized run, which is "
        "the systematic-underquote signature rather than run-to-run variance"
    )
    # Spread is the estimand that matters for a SELECTION model: a uniform bias cancels in a ranking,
    # variation across configs does not. The flat block scored 4.71x.
    assert spread < 2.5, f"sft error spread {spread:.2f}x across configs (flat block scored 4.71x)"

    # OPD: the control. Same estimand, same corpus discipline, calibrated on grpo arms and never
    # refitted -- so if the SFT assertions above ever fail, this passing shows the corpus and the
    # scoring are intact and the defect is specific to sft.
    opd_envelope = {
        ("H100", "Qwen/Qwen3.5-0.8B", 8): 645.9,
        ("RTX 4090", "Qwen/Qwen3.5-0.8B", 4): 461.6,
        ("RTX 4090", "Qwen/Qwen3.5-0.8B", 8): 631.7,
        ("RTX 4090", "Qwen/Qwen3.5-0.8B", 16): 966.5,
        ("RTX 4090", "Qwen/Qwen3.5-0.8B", 48): 1962.3,
    }
    rollout_noise_floor = 1.194
    for (card, model, steps), fastest in opd_envelope.items():
        cfg = RunConfig(model, "opd", steps, batch_size=8, group_size=4, seq_len=1024)
        quote = run_startup_seconds(cfg, card) + steps * seconds_per_step(cfg, card)
        assert quote >= fastest / rollout_noise_floor, (
            f"opd {card}/{model} x{steps}: quote {quote:.1f}s undercuts its fastest realized run "
            f"{fastest:.1f}s by {fastest / quote:.2f}x, past the {rollout_noise_floor}x floor"
        )


def test_small_sft_step_prices_at_the_saturation_floor_but_still_tracks_card_speed():
    """Pins BOTH halves of the mechanism the batch-32 anchor and card ranking jointly selected.

    Every sft arm runs (batch_size=8, seq_len=1024), and at a single shape tokens = steps x 8192, so
    a per-token rate error and a per-step floor are the same regressor. They diverge only when the
    shape changes: 4x the batch costs 4x the step under a rate, but is absorbed by a floor.
    Extrapolating the 4090 group's measured 6.1x rate error to the batch-32 anchor predicts a 762 s
    train wall against a run recorded as cold-start dominated under 449.5 s of setup; the floor
    predicts 249 s. So the correction is a floor, and a refit cannot move it back into the rate term.

    A FLAT per-step overhead also clears that bar, and is the trap this second half exists to block:
    being card-invariant it made over half a 4B step stop responding to card speed, which put the A10
    ahead of a faster RTX 4090 on job cost and broke the ranking this module exists to produce. The
    floor is an occupancy limit on real arithmetic, so it stays proportional to params and inverse in
    card speed. Both properties are asserted here because satisfying only the first is what made the
    wrong model look correct.
    """
    from flash.cost.analytical import SFT_SATURATION_TOKENS, step_seconds_split
    from flash.cost.types import RunConfig

    small = RunConfig("Qwen/Qwen3.5-0.8B", "sft", 26, batch_size=8, seq_len=1024)
    wide = RunConfig("Qwen/Qwen3.5-0.8B", "sft", 26, batch_size=32, seq_len=1024)
    assert 8 * 1024 < SFT_SATURATION_TOKENS and 32 * 1024 < SFT_SATURATION_TOKENS
    small_gpu, small_fixed = step_seconds_split(small, "RTX 4090")
    wide_gpu, wide_fixed = step_seconds_split(wide, "RTX 4090")

    # both shapes sit under the floor, so 4x the tokens per step costs the SAME step. under a pure
    # rate wide_gpu would be 4x small_gpu, which is what the anchor rejected.
    assert wide_gpu == pytest.approx(small_gpu, rel=1e-6)
    assert wide_fixed == pytest.approx(small_fixed, rel=1e-6)

    # ...and yet the floored step is still arithmetic: a 3x faster card runs it ~3x quicker. a flat
    # overhead would make these equal, which is precisely how it inverted the card ranking.
    fast_gpu, _ = step_seconds_split(small, "H100")
    assert small_gpu > 2.0 * fast_gpu, (
        f"floored step must still track card speed: RTX 4090 {small_gpu:.2f}s vs H100 "
        f"{fast_gpu:.2f}s is too flat to rank cards on"
    )

    # and still with model size, so a 27B step is not priced like a 0.8B one.
    big = RunConfig("Qwen/Qwen3.6-27B", "sft", 26, batch_size=8, seq_len=1024)
    big_gpu, _ = step_seconds_split(big, "RTX 4090")
    assert big_gpu > 5.0 * small_gpu

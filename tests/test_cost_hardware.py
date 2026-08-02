"""Cost estimator: GPU compute table, pricing/VRAM lookups, cheapest-fit selection.

No network. The compute table and the selection rule must stay consistent with the
RunPod GPU registry in ``flash.providers.base``.
"""

from __future__ import annotations

import pytest

from flash.cost.facts import (
    GPU_COMPUTE_TFLOPS,
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


def test_step_floor_dominates_a_small_rollout_step():
    # regression: the model used to quote a grpo step as compute + reward wait and nothing else,
    # which ran ~0.48x of realized because it priced no per-step floor at all. a small 0.8B step is
    # ~0.3s of arithmetic against a realized 60-140s, so the floor -- not the flops -- must be what
    # the quote is made of. asserting the RATIO, so recalibrating any single constant cannot silently
    # return the model to a compute-only quote.
    from flash.cost.analytical import step_seconds_split
    from flash.cost.types import RunConfig

    config = RunConfig(
        "Qwen/Qwen3.5-0.8B", "grpo", 6, seq_len=512, completion_len=128, batch_size=8, group_size=4
    )
    gpu_bound, fixed = step_seconds_split(config, "H100")
    assert gpu_bound < 5.0  # the arithmetic really is negligible at this size
    assert fixed > 10 * gpu_bound  # ... so a quote that omits the floor is wrong by an order
    # and the floor specifically, not the reward wall, has to be in there: the reward term alone is
    # completions x reward_seconds, which this assert deliberately sits above.
    assert fixed > config.normalized().batch_size * config.normalized().group_size


def test_step_floor_is_charged_per_class_not_pooled():
    from flash.cost.facts import _DEFAULT_STEP_FLOOR_S, step_floor_seconds

    # measured classes carry their own floor; h200's is the outlier the pooled value would erase.
    assert step_floor_seconds("H200") > step_floor_seconds("H100")
    assert step_floor_seconds("RTX 5090") < step_floor_seconds("RTX 4090")
    # an unmeasured class falls back to the pooled median rather than to no floor at all.
    assert step_floor_seconds("some-unmeasured-class") == _DEFAULT_STEP_FLOOR_S
    assert _DEFAULT_STEP_FLOOR_S > 0.0


def test_step_floor_table_only_lists_real_classes():
    # same drift guard the TFLOPS table gets. a typo'd key here does not raise -- it silently falls
    # through to the pooled default, so the measured value would be quietly discarded and the class
    # would go on being mispriced with nothing failing.
    from flash.cost.facts import _STEP_FLOOR_S

    for name in _STEP_FLOOR_S:
        assert name in GPU_INFO, f"{name} is not a managed GPU class"


def test_step_floor_applies_to_rollout_methods_only():
    # the floor is vllm engine entry/exit + actor->rollout weight sync. sft runs neither, and the
    # floor was fitted on rollout arms only, so charging it to sft would invent ~45s per step.
    from flash.cost.analytical import step_seconds_split
    from flash.cost.facts import step_floor_seconds
    from flash.cost.types import RunConfig

    sft = RunConfig("Qwen/Qwen3.5-0.8B", "sft", 6, seq_len=512, batch_size=8)
    assert step_seconds_split(sft, "H100")[1] == pytest.approx(0.0)
    # rollout methods carry it, and by the card's own amount -- comparing two classes isolates the
    # floor from the reward/teacher wait, which is identical on both.
    for method in ("grpo", "opd"):
        config = RunConfig("Qwen/Qwen3.5-0.8B", method, 6, seq_len=512, completion_len=128)
        h100 = step_seconds_split(config, "H100")[1]
        h200 = step_seconds_split(config, "H200")[1]
        assert h200 - h100 == pytest.approx(step_floor_seconds("H200") - step_floor_seconds("H100"))


def test_step_floor_is_not_shardable_across_cards():
    # the allocator divides ONLY the first half by card count. engine init and weight sync do not
    # get faster on more ranks, so a floor that leaked into the shardable half would make wide
    # multi-card jobs look arbitrarily cheap.
    from flash.cost.analytical import step_seconds_split
    from flash.cost.facts import step_floor_seconds
    from flash.cost.types import RunConfig

    config = RunConfig(
        "Qwen/Qwen3.5-0.8B", "grpo", 6, seq_len=512, completion_len=128, batch_size=8, group_size=4
    )
    gpu_bound, fixed = step_seconds_split(config, "H100")
    assert fixed >= step_floor_seconds("H100")
    assert gpu_bound < step_floor_seconds("H100")


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
    from flash.cost.analytical import step_seconds_split
    from flash.cost.facts import ROLLOUT_SECONDS_PER_COMPLETION, step_floor_seconds
    from flash.cost.types import RunConfig

    shape = {"seq_len": 512, "completion_len": 128, "batch_size": 8, "group_size": 4}
    completions = shape["batch_size"] * shape["group_size"]

    measured = RunConfig(
        "Qwen/Qwen3.5-0.8B", "grpo", 6, reward_seconds_per_completion=0.0003, **shape
    )
    gpu_bound, fixed = step_seconds_split(measured, "H100")

    # h100 arms at this shape realized 56-100s/step with graders this fast. the broken model quoted
    # ~43s of constant and nothing per-completion, landing at ~0.5x realized; the rollout slope is
    # what closes that gap, so it must survive the caller supplying a true reward latency.
    assert fixed >= step_floor_seconds("H100") + completions * ROLLOUT_SECONDS_PER_COMPLETION
    assert gpu_bound + fixed > 60.0


def test_rollout_wall_scales_with_completion_count():
    # the wall is const_card + slope*completions. fitted on 16-256 completions/step, where the
    # per-card median rose ~3x (h100 77.8s at 32 -> 230.7s at 256). a constant-only model fitted on
    # the original campaign missed this because 30 of its 32 arms ran exactly 32 completions.
    from flash.cost.facts import ROLLOUT_SECONDS_PER_COMPLETION, step_floor_seconds

    base = step_floor_seconds("H100")
    assert step_floor_seconds("H100", 0) == pytest.approx(base)
    assert step_floor_seconds("H100", 256) == pytest.approx(
        base + 256 * ROLLOUT_SECONDS_PER_COMPLETION
    )
    # the slope is a rollout-path property, so it is the SAME on every card; only the constant differs.
    for card in ("H100", "H200", "RTX 5090"):
        delta = step_floor_seconds(card, 100) - step_floor_seconds(card, 0)
        assert delta == pytest.approx(100 * ROLLOUT_SECONDS_PER_COMPLETION)
    # a negative completion count cannot credit time back.
    assert step_floor_seconds("H100", -5) == pytest.approx(base)


def test_a100_sxm_40gb_inherits_its_siblings_wall():
    # same SMs and tensor cores as the 80GB board, less hbm only -- and this wall is engine-entry and
    # weight-sync bound, not capacity bound, so the 40GB variant cannot be ~2x cheaper per step than
    # the board it is a memory-trimmed version of. left out of the table it fell through to the
    # pooled default and under-priced the dominant per-step term on every lambda/vast quote in that
    # band. the TFLOPS table inherits across this same pair for the same reason.
    from flash.cost.facts import _DEFAULT_STEP_FLOOR_S, step_floor_seconds

    assert step_floor_seconds("A100 SXM 40GB") == step_floor_seconds("A100 SXM")
    assert step_floor_seconds("A100 SXM 40GB") != _DEFAULT_STEP_FLOOR_S


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


def test_resident_rollout_is_not_charged_the_engine_cycle():
    # the constant is vllm engine entry/exit + weight sync. a sleep_unsupported model pins its
    # rollout engine RESIDENT (catalog.py, backend_common.rollout_resident_overrides), so that cycle
    # never runs and charging it extrapolates outside the fit: all 56 fitted arms are sleep-capable
    # 0.8b/2b/4b models. the error is not marginal -- on qwen3.6-35b-a3b the h200 constant (126.4s)
    # alone exceeds the ~24s realized grpo step analytical.py documents for that model.
    from flash.cost.facts import rollout_is_resident, step_floor_seconds

    assert rollout_is_resident("Qwen/Qwen3.6-35B-A3B")
    assert not rollout_is_resident("Qwen/Qwen3.5-0.8B")

    for card in ("B200", "H200"):
        sleeping = step_floor_seconds(card, 0)
        resident = step_floor_seconds(card, 0, resident=True)
        assert resident == pytest.approx(0.0)
        # guards the mutation "resident flag ignored": that returns the sleeping value here.
        assert sleeping > 50.0


def test_resident_rollout_still_pays_the_per_completion_slope():
    # the opposite error, and the reason resident is not simply floor=0: a resident engine skips the
    # entry/exit CYCLE, not the per-sequence sampling/detokenize/dispatch work. zeroing the whole
    # wall would under-quote every resident step by slope*completions.
    from flash.cost.facts import ROLLOUT_SECONDS_PER_COMPLETION, step_floor_seconds

    assert step_floor_seconds("H200", 64, resident=True) == pytest.approx(
        64 * ROLLOUT_SECONDS_PER_COMPLETION
    )


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
    # step_floor_seconds honouring `resident` is worthless if step_seconds_split never passes it.
    # the helper-level tests above cannot see that wiring: they call the helper directly, so a
    # call site that drops the argument leaves them all green. this asserts through the real quote.
    from flash.cost.analytical import step_seconds_split
    from flash.cost.facts import ROLLOUT_SECONDS_PER_COMPLETION, step_floor_seconds
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
        _, fixed = step_seconds_split(cfg, "H200")
        # the h200 constant is 126.4s; if it were still charged `fixed` would clear it outright.
        assert fixed < step_floor_seconds("H200"), method
        # and the slope is still in there -- this is not a "wall deleted" pass.
        assert fixed >= slope, method


def test_hardware_choice_is_not_a_restatement_of_the_hourly_rate():
    """Selection must rank on what the JOB costs, which requires per-card step times to differ.

    This is the property the allocator's total-cost ranking rides on. Before the per-card wall, a
    step was quoted as compute plus reward wait, and compute is 0.3-9.3s on every class -- so all
    eight classes quoted within 0.6s of each other and dollars-per-step degenerated into a monotone
    function of the hourly rate. The ranking looked speed-aware and was arithmetically incapable of
    disagreeing with sorting on $/hr.

    Measured on the one campaign family that ran on all eight classes (Qwen3.5-0.8B, b8 g4, ctx512,
    cap128), RTX 5090 is the cheapest per job at $0.0123/step despite RTX 4090 being cheaper per
    hour ($0.69 vs $0.99): the 4090 takes 101.4s against the 5090's 44.6s. Ranking on rate alone
    buys the 4090 and pays 1.59x more per step.
    """
    from flash.cost.analytical import step_seconds_split
    from flash.cost.types import RunConfig

    cfg = RunConfig(
        "Qwen/Qwen3.5-0.8B", "grpo", 1, seq_len=512, completion_len=128, batch_size=8, group_size=4
    )
    rates = {"RTX 4090": 0.69, "RTX 5090": 0.99}
    seconds = {c: sum(step_seconds_split(cfg, c)) for c in rates}

    # the two cards must not be quoted as interchangeable; the 4090's wall is 4.5x the 5090's.
    assert seconds["RTX 4090"] > 2 * seconds["RTX 5090"]

    dollars = {c: seconds[c] * rates[c] / 3600.0 for c in rates}
    # cheaper per hour, dearer per job -- so the two orderings genuinely disagree here.
    assert rates["RTX 4090"] < rates["RTX 5090"]
    assert dollars["RTX 5090"] < dollars["RTX 4090"]


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
    from flash.cost.facts import step_floor_seconds

    assert step_floor_seconds("H100", 64) > step_floor_seconds("H100", 32)

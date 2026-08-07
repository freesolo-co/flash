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


def test_sft_shards_by_sequence_not_by_data(monkeypatch):
    """sft must not borrow the fsdp data-parallel constant.

    Flash pins ulysses_sp_size to the card count and verl derives dp_size = world_size //
    ulysses_sequence_parallel_size, so a multi-card sft run is dp_size == 1: pure sequence
    parallelism. grpo/opd run data-parallel. The two pay different collectives (fsdp moves MODEL
    bytes per layer, ulysses moves ACTIVATION bytes per layer, both directions) and cannot share a
    scaling constant.

    The shipped sp and dp constants are deliberately equal, so reading the returned value proves
    nothing about which one was consulted. Perturb the sp constant: sft must move with it and
    grpo/opd must not.
    """
    from flash.cost import analytical
    from flash.cost.types import RunConfig

    def cfg(method: str) -> RunConfig:
        return RunConfig(model_id="Qwen/Qwen3.5-9B", method=method, steps=8)

    # 0.95 is well clear of the non-decreasing clamp (which pins anything at or below 0.5 to 1.0x
    # at 2 cards), so a wrong reading cannot coincidentally land on the right number.
    monkeypatch.setattr(analytical, "MULTI_CARD_SCALING_SP_PCIE", 0.95)
    monkeypatch.setattr(analytical, "MULTI_CARD_SCALING_SP_NVLINK", 0.95)

    for card in ("RTX 4090", "A100 SXM"):
        sft = analytical.method_card_speedup(cfg("sft"), 2, card)
        assert sft == pytest.approx(2 * 0.95), f"sft on {card} ignored the sp constant: {sft}"
        for method in ("grpo", "opd"):
            other = analytical.method_card_speedup(cfg(method), 2, card)
            assert other == pytest.approx(analytical.multi_card_speedup(2, card)), (
                f"{method} on {card} was routed through the sequence-parallel constant"
            )


def test_sequence_parallel_speedup_keeps_the_multi_card_invariants():
    """The sp curve owes the same guarantees the dp curve does.

    A separate constant must not become a hole in the invariants: one card is one card, pcie never
    borrows nvlink scaling, no fabric delivers linear, and adding a card never models as slower.
    """
    from flash.cost.analytical import sequence_parallel_speedup

    for name in ("A100 SXM", "RTX 4090", "unknown"):
        assert sequence_parallel_speedup(1, name) == 1.0
    for n in (2, 3, 4):
        assert sequence_parallel_speedup(n, "RTX 4090") < sequence_parallel_speedup(n, "A100 SXM")
        assert sequence_parallel_speedup(n, "A100 SXM") < n
    for name in ("A100 SXM", "RTX 4090", "H100", "unlisted-class"):
        vals = [sequence_parallel_speedup(n, name) for n in range(1, 9)]
        assert vals == sorted(vals), f"{name} sp speedup decreases: {vals}"


def test_multi_card_sft_quote_moves_with_the_sequence_parallel_constant(monkeypatch):
    """The seam has to reach the shipped dollar figure, not just the helper.

    A 9B sft run pinned to 2x RTX 4090 is the reachable multi-card sft cell: one 4090 cannot hold
    it (needs 32 GB), two can, and sft has no step floor so the whole gpu-bound half is the token
    compute term. That makes the sp constant map straight to the price -- if the quote does not
    move when it changes, the fix never reached estimate_cost().
    """
    from flash.cost import analytical
    from flash.cost.types import RunConfig

    spec = RunConfig(
        model_id="Qwen/Qwen3.5-9B",
        method="sft",
        steps=64,
        seq_len=2048,
        batch_size=8,
        train_tokens=1_000_000,
        gpu_type="RTX 4090",
        gpu_count=2,
    )
    before = analytical.estimate_cost(spec)
    assert before.gpu_count == 2, "the reachable multi-card sft cell collapsed to one card"

    # degrade the realized scaling: the same work must take longer and cost more. 0.6 stays clear
    # of the non-decreasing clamp, so this exercises the constant rather than the floor under it.
    monkeypatch.setattr(analytical, "MULTI_CARD_SCALING_SP_PCIE", 0.6)
    after = analytical.estimate_cost(spec)
    assert after.train_seconds > before.train_seconds
    assert after.total_usd > before.total_usd

    # and the grpo constant must not be what is moving it.
    monkeypatch.setattr(analytical, "MULTI_CARD_SCALING_PCIE", 0.4)
    unchanged = analytical.estimate_cost(spec)
    assert unchanged.train_seconds == pytest.approx(after.train_seconds)

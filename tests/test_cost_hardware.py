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
    # No validation gate: every fitting class is eligible, ranked by static rate. 24 GB is the
    # floor (sub-24 GB classes dropped), so anything that fits <=24 GB lands on the cheapest 24 GB
    # card, L4 ($0.39).
    assert pick_gpu(12) == "L4"
    assert pick_gpu(24) == "L4"
    # > 24 GB needs the big-VRAM tier -> cheapest static >= 40 is the A40 ($0.44, 48 GB).
    assert pick_gpu(40) == "A40"


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


def test_pick_gpu_includes_unvalidated_classes():
    # No validation gate: the cheapest static-rate class wins regardless of validation status
    # (L4 is unvalidated yet is the cheapest fitting class).
    assert pick_gpu(12) == "L4"


def test_pick_gpu_impossible_raises():
    with pytest.raises(ValueError, match="no GPU class fits"):
        pick_gpu(100_000)


def test_pick_gpu_auto_matches_default():
    assert pick_gpu(24, provider="auto") == pick_gpu(24)
    assert pick_gpu(12) == "L4"

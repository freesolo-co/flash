"""Cost estimator: GPU compute table, pricing/VRAM lookups, cheapest-fit selection.

No network. The compute table and the selection rule must stay consistent with the
provider-agnostic GPU registry in ``flash.providers.base``.
"""

from __future__ import annotations

import pytest

from flash.cost.facts import (
    GPU_COMPUTE_TFLOPS,
    REALIZED_HOURLY_USD,
    gpu_hourly_usd,
    gpu_tflops,
    gpu_vram_gb,
    pick_gpu,
    realized_hourly_usd,
)
from flash.providers.base import GPU_INFO, providers_for


def test_realized_rate_never_exceeds_list_for_any_class():
    # The realized (spot/queue) rate is conservative: clamped to the on-demand list so the
    # estimator can never over-quote a class vs its list price. Holds for EVERY registry class,
    # by construction -- including the historically-over-list RTX A5000 ($0.304 raw vs $0.27 list).
    for name in GPU_INFO:
        assert realized_hourly_usd(name) <= gpu_hourly_usd(name), name
    assert realized_hourly_usd("RTX A5000") <= gpu_hourly_usd("RTX A5000")


def test_over_list_realized_entry_is_clamped_to_list():
    # The raw RTX A5000 table entry sits above its list price; realized_hourly_usd clamps it down
    # to list (and a genuinely-discounted entry like the RTX 5090 still reports its discount).
    assert REALIZED_HOURLY_USD["RTX A5000"] > gpu_hourly_usd("RTX A5000")  # raw entry is over list
    assert realized_hourly_usd("RTX A5000") == gpu_hourly_usd("RTX A5000")  # clamped to list
    assert realized_hourly_usd("RTX 5090") == REALIZED_HOURLY_USD["RTX 5090"]  # below list -> unchanged
    assert realized_hourly_usd("RTX 5090") < gpu_hourly_usd("RTX 5090")


def test_realized_rate_falls_back_to_list_when_unobserved():
    # A class without an observed realized rate reports its list price (no rate invented).
    assert "L40S" not in REALIZED_HOURLY_USD
    assert realized_hourly_usd("L40S") == gpu_hourly_usd("L40S")


def test_compute_table_only_lists_real_classes():
    # Every GPU we assign a TFLOPS figure to must be a real managed class (no drift).
    for name in GPU_COMPUTE_TFLOPS:
        assert name in GPU_INFO, f"{name} is not a managed GPU class"


def test_gpu_tflops_known_and_default():
    assert gpu_tflops("RTX 5090") == GPU_COMPUTE_TFLOPS["RTX 5090"]
    assert gpu_tflops("RTX 5090") > gpu_tflops("RTX A5000")  # newer/faster
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
    # No validation gate: every fitting class is eligible, ranked by the REALIZED rate it bills
    # at. The RTX 3090 ($0.239, 24 GB) is the cheapest realized card, so it wins anything <= 24 GB.
    assert pick_gpu(12) == "RTX 3090"
    assert pick_gpu(24) == "RTX 3090"
    # > 24 GB needs the big-VRAM tier -> cheapest realized >= 40 is the A40 ($0.44, 48 GB).
    assert pick_gpu(40) == "A40"


def test_pick_gpu_result_actually_fits_and_is_cheapest():
    for need in (8, 16, 24, 33, 48, 80):
        gpu = pick_gpu(need)
        assert gpu_vram_gb(gpu) >= need
        # No validation gate: nothing fitting is cheaper at the REALIZED (billed) rate.
        cheaper_fits = [
            g
            for g in GPU_INFO.values()
            if g.vram_gb >= need and realized_hourly_usd(g.name) < realized_hourly_usd(gpu)
        ]
        assert not cheaper_fits, f"{cheaper_fits} cheaper than {gpu} for {need} GB"


def test_pick_gpu_includes_unvalidated_classes():
    # No validation gate: the cheapest realized-rate class wins regardless of validation status.
    # The RTX 3090 ($0.239) is the cheapest realized card and wins at 12 GB.
    assert pick_gpu(12) == "RTX 3090"


def test_pick_gpu_impossible_raises():
    with pytest.raises(ValueError, match="no GPU class fits"):
        pick_gpu(100_000)


def test_pick_gpu_provider_pin_restricts_to_provisionable():
    # The only per-provider filter is PROVISIONABILITY (providers_for) -- there is no validation
    # gate. A provider pin only ever returns a class that provider can actually provision.
    for prov in ("runpod", "vast"):
        assert prov in providers_for(pick_gpu(24, provider=prov))


def test_pick_gpu_provider_filter_excludes_other_providers_only_class():
    # A class only the OTHER provider can provision must be excluded under a provider pin.
    # Provisionability is providers_for() (enum_member for runpod, vast_name for vast); pricing
    # a Vast-only class on a runpod pin would misquote the run.
    vast_only = [n for n, g in GPU_INFO.items() if g.vast_name and not g.enum_member]
    assert vast_only, "expected at least one Vast-only class in the registry"
    for need in (16, 24, 48, 80):
        try:
            runpod_pick = pick_gpu(need, provider="runpod")
        except ValueError:
            continue  # nothing runpod-provisionable fits this tier; fine
        assert "runpod" in providers_for(runpod_pick)
        assert runpod_pick not in vast_only


def test_pick_gpu_auto_spans_all_providers():
    # Without a provider pin, selection spans the whole registry: the cheapest (realized-rate)
    # fitting class overall, regardless of which provider(s) can run it or whether it's validated.
    assert pick_gpu(24, provider="auto") == "RTX 3090"
    assert pick_gpu(12) == "RTX 3090"

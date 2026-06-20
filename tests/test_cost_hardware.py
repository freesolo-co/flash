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
from flash.providers.base import GPU_INFO, UnsupportedGpuError, providers_for


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


def test_pick_gpu_cheapest_validated_fit():
    # Cheapest validated class with >= 12 GB is the RTX A5000 ($0.27, 24 GB).
    assert pick_gpu(12) == "RTX A5000"
    # 25 GB excludes the 24 GB cards -> cheapest validated >= 25 is the RTX 5090 (32 GB).
    assert pick_gpu(25) == "RTX 5090"
    # 40 GB needs the big-VRAM tier -> cheapest validated is the A100 PCIe ($1.39, 80 GB).
    assert pick_gpu(40) == "A100 PCIe"


def test_pick_gpu_result_actually_fits_and_is_cheapest():
    for need in (8, 16, 24, 33, 48, 80):
        gpu = pick_gpu(need)
        assert gpu_vram_gb(gpu) >= need
        cheaper_fits = [
            g
            for g in GPU_INFO.values()
            if g.validated and g.vram_gb >= need and g.hourly_usd < gpu_hourly_usd(gpu)
        ]
        assert not cheaper_fits, f"{cheaper_fits} cheaper than {gpu} for {need} GB"


def test_pick_gpu_pin_honored_only_when_it_fits():
    # A pin that fits is honored even if a cheaper class also fits.
    assert pick_gpu(12, pin="RTX 5090") == "RTX 5090"
    # A pin too small to fit escalates to the cheapest fitting class (allocator behavior).
    assert pick_gpu(40, pin="RTX 4090") == "A100 PCIe"


def test_pick_gpu_allow_unvalidated_widens_pool():
    # The unvalidated RTX 2000 Ada ($0.24, 16 GB) undercuts the validated A5000 ($0.27).
    assert pick_gpu(12, allow_unvalidated=True) == "RTX 2000 Ada"


def test_pick_gpu_impossible_raises():
    with pytest.raises(ValueError, match="no GPU class fits"):
        pick_gpu(100_000)


def test_pick_gpu_pin_canonicalizes_aliases():
    # An alias pin (the spelling the allocator accepts) resolves to the canonical class,
    # not silently to the cheapest auto pick. "5090"/"a5000" are short/alias spellings.
    assert pick_gpu(12, pin="5090") == "RTX 5090"
    assert pick_gpu(12, pin="a5000") == "RTX A5000"
    assert pick_gpu(12, pin="rtx 5090") == "RTX 5090"


def test_pick_gpu_unknown_pin_raises_not_silent_fallback():
    # An unknown pin must raise (mirror the allocator's canonicalization), rather than
    # silently dropping to the cheapest class and underquoting.
    with pytest.raises(UnsupportedGpuError):
        pick_gpu(12, pin="Tesla T4")


@pytest.mark.parametrize("policy_pin", ["auto", "cheapest", "AUTO", " cheapest ", "", None])
def test_pick_gpu_policy_pin_falls_through_to_auto_select(policy_pin):
    # gpu.type can be a POLICY sentinel ("auto"/"cheapest"/empty/None), not a GPU name.
    # canonical_gpu would raise on those, so they must fall through to cheapest-fit selection
    # (the allocator picks the class), NOT crash with UnsupportedGpuError. Cheapest validated
    # class with >= 12 GB is the RTX A5000.
    assert pick_gpu(12, pin=policy_pin) == "RTX A5000"


def test_pick_gpu_provider_pin_restricts_to_validated_on():
    # RTX A5000 (24 GB, runpod-only) is the cheapest auto pick at 24 GB, but a Vast pin
    # must exclude it -- pricing a runpod-only class @vast would underquote.
    assert "vast" not in GPU_INFO["RTX A5000"].validated_on
    chosen = pick_gpu(24, provider="vast")
    assert chosen != "RTX A5000"
    assert "vast" in GPU_INFO[chosen].validated_on
    # With provider auto (default), the runpod-only A5000 is back in the pool.
    assert pick_gpu(24, provider="auto") == "RTX A5000"


def test_pick_gpu_provider_pin_holds_under_allow_unvalidated():
    # allow_unvalidated relaxes the validation-status gate, NOT the per-provider
    # PROVISIONABILITY filter: a class the pinned provider can't *provision* must never be
    # quoted, even unvalidated. Mirrors allocator.allocate (it walks one provider's
    # gpu_classes()), so a runpod pin never prices a Vast-only class and vice versa.
    # Provisionability is providers_for() (enum_member / vast_name), NOT validated_on.
    runpod_pick = pick_gpu(24, provider="runpod", allow_unvalidated=True)
    assert "runpod" in providers_for(runpod_pick)
    vast_pick = pick_gpu(24, provider="vast", allow_unvalidated=True)
    assert "vast" in providers_for(vast_pick)
    # Without a provider pin, allow_unvalidated still widens across the whole registry.
    assert pick_gpu(12, allow_unvalidated=True) == "RTX 2000 Ada"


def test_pick_gpu_provider_filter_is_provisionability_not_validation():
    # The per-provider filter is PROVISIONABILITY (providers_for), not validation status,
    # matching the allocator (it walks gpu_classes(), gated separately by validated_on).
    # A Vast-only class (no enum_member) is never quoted on a runpod pin, even unvalidated.
    vast_only = [n for n, g in GPU_INFO.items() if g.vast_name and not g.enum_member]
    assert vast_only, "expected at least one Vast-only class in the registry"
    for need in (16, 24, 48, 80):
        try:
            runpod_pick = pick_gpu(need, provider="runpod", allow_unvalidated=True)
        except ValueError:
            continue  # nothing runpod-provisionable fits this tier; fine
        assert "runpod" in providers_for(runpod_pick)
        assert runpod_pick not in vast_only

    # And the inverse of the old bug: an unvalidated-but-provisionable class IS now in the
    # pinned provider's pool under allow_unvalidated. The RTX 3090 is runpod-provisionable
    # (enum_member set) yet validated only on Vast -- the allocator prices it on a runpod
    # pin with allow_unvalidated=True, so the estimator must consider it too. With the old
    # validated_on-based filter it would have been silently excluded.
    g3090 = GPU_INFO["RTX 3090"]
    assert g3090.enum_member  # runpod-provisionable
    assert "runpod" not in g3090.validated_on  # but not runpod-validated
    runpod_pool = [
        n
        for n, g in GPU_INFO.items()
        if g.vram_gb >= 24 and "runpod" in providers_for(n)
    ]
    assert "RTX 3090" in runpod_pool

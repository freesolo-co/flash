"""Cost estimator: the ``CostEstimate`` result type (breakdown, provider). No network."""

from __future__ import annotations

import dataclasses

import pytest

from flash.cost import RunConfig, estimate_cost
from flash.cost.types import CostEstimate


@pytest.fixture
def est() -> CostEstimate:
    return estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 150))


def test_is_frozen(est):
    with pytest.raises(dataclasses.FrozenInstanceError):
        est.total_usd = 0.0  # type: ignore[misc]


def test_wall_clock_hours_derivation(est):
    assert est.wall_clock_hours == pytest.approx(est.wall_clock_seconds / 3600.0)


def test_breakdown_lists_every_term(est):
    b = est.breakdown()
    for needle in ("GPU", "Setup", "Per step", "Train", "Wall clock", "TOTAL"):
        assert needle in b
    # GRPO estimate carries explanatory notes.
    assert "Notes" in b


def test_capped_estimate_flags_in_breakdown():
    capped = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 100_000))
    assert capped.wall_capped
    assert "CAPPED" in capped.breakdown()


def test_subhour_cap_note_renders_minutes_not_zero_hours():
    # A sub-hour wall cap (floored to 60s) must not print a confusing "0h" in the cap note.
    capped = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", 100_000, max_wall_seconds=60))
    assert capped.wall_capped
    cap_note = next(n for n in capped.notes if "wall cap" in n)
    assert "0h" not in cap_note
    assert "1m" in cap_note  # 60s -> "1m"


def test_fmt_duration_units():
    from flash.cost.analytical import _fmt_duration

    assert _fmt_duration(20) == "20s"  # sub-minute -> seconds, never "0m"
    assert _fmt_duration(59) == "59s"
    assert _fmt_duration(60) == "1m"  # sub-hour -> minutes, never "0h"
    assert _fmt_duration(1800) == "30m"
    assert _fmt_duration(24 * 3600) == "24h"  # whole hours stay clean
    assert _fmt_duration(int(1.5 * 3600)) == "1.5h"  # fractional multi-hour -> one decimal


def test_provider_is_normalized_and_validated():
    # Case/whitespace variants normalize to the canonical substrate; empty -> "auto".
    assert RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="RunPod").provider == "runpod"
    assert RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider=" vast ").provider == "vast"
    assert RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="").provider == "auto"
    assert RunConfig("Qwen/Qwen3.5-4B", "grpo", 10).provider == "auto"
    # An unknown substrate fails fast here (clear error) instead of as "no GPU class fits".
    with pytest.raises(ValueError, match="unknown provider"):
        RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="aws")


def test_estimate_reports_concrete_provider_for_single_substrate_pick():
    from flash.providers.base import providers_for

    # "H100 NVL" is a Vast-only class -- under the "auto" sentinel the estimate's chosen-hardware
    # provider should resolve to the concrete substrate, not stay "auto".
    est = estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, gpu="H100 NVL"))
    assert est.gpu == "H100 NVL"
    assert providers_for("H100 NVL") == ("vast",)
    assert est.provider == "vast"


def test_estimate_provider_invariant_holds_under_auto(est):
    # Whatever class auto-selection lands on: a single-substrate class is reported as that
    # substrate; a multi-substrate class stays the "auto" sentinel.
    from flash.providers.base import providers_for

    provs = providers_for(est.gpu)
    if len(provs) == 1:
        assert est.provider == provs[0]
    else:
        assert est.provider == "auto"


def test_estimate_keeps_explicit_provider_pin():
    # An explicit substrate pin is never overridden by single-substrate resolution.
    assert estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", 10, provider="runpod")).provider == "runpod"

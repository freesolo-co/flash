"""Cost estimator: the ``CostEstimate`` result type (breakdown, serialization).

No network.
"""

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


def test_to_dict_round_trip(est):
    d = est.to_dict()
    assert d["model_id"] == est.model_id
    assert d["total_usd"] == est.total_usd
    assert d["wall_clock_hours"] == pytest.approx(est.wall_clock_hours)
    # to_dict must be JSON-serializable (notes is a tuple -> list).
    import json

    json.loads(json.dumps(d))


def test_describe_contains_key_facts(est):
    line = est.describe()
    assert est.gpu in line
    assert f"${est.total_usd:.2f}" in line
    assert est.method.upper() in line


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

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

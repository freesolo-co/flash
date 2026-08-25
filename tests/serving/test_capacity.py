"""Live hosted-serving capacity snapshot semantics."""

from __future__ import annotations

import asyncio

import pytest

from flash.serving.src.traffic.capacity import (
    CapacitySnapshot,
    ConfiguredCapacityProvider,
    fixed_local_active_limit,
)

MODEL = "Qwen/Qwen3.5-9B"
IDENTITY = "modal-app-class:freesolo-lora-serving/LoraEngine_Qwen3_5_9B_deadbeef0000"


def _snapshot(**updates: object) -> CapacitySnapshot:
    values = {
        "model": MODEL,
        "deployment_identity": IDENTITY,
        "observed_at": 10.0,
        "total_runners": 1,
        "running_inputs": 10,
        "input_headroom": 54,
        "backlog": 0,
        "observed_local_active": 10,
        "local_active_limit": 64,
    }
    values.update(updates)
    return CapacitySnapshot(**values)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model": ""}, "model"),
        ({"model": f" {MODEL}"}, "model"),
        ({"deployment_identity": ""}, "deployment_identity"),
        ({"deployment_identity": " padded"}, "deployment_identity"),
        ({"deployment_identity": "x" * 257}, "deployment_identity"),
        ({"observed_at": -1.0}, "observed_at"),
        ({"running_inputs": True}, "counts"),
        ({"input_headroom": -1}, "counts"),
        ({"observed_local_active": -1}, "counts"),
        ({"local_active_limit": -1}, "counts"),
        ({"total_runners": 0}, "warm runner"),
        ({"unavailable": True}, "error indication"),
        ({"error": "unexpected"}, "cannot carry"),
    ],
)
def test_snapshot_rejects_malformed_fields(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _snapshot(**updates)


def test_snapshot_freshness_is_bounded_and_rejects_future_observations() -> None:
    snapshot = _snapshot()

    assert snapshot.is_fresh(10.0)
    assert snapshot.is_fresh(12.0)
    assert not snapshot.is_fresh(12.000001)
    assert not snapshot.is_fresh(9.999999)


def test_dispatchability_requires_exact_model_and_deployment_identity() -> None:
    snapshot = _snapshot()

    assert snapshot.is_dispatchable(11.0, model=MODEL, deployment_identity=IDENTITY)
    assert not snapshot.is_dispatchable(11.0, model="other-model", deployment_identity=IDENTITY)
    assert not snapshot.is_dispatchable(11.0, model=MODEL, deployment_identity="other-deployment")
    assert not snapshot.is_dispatchable(13.0, model=MODEL, deployment_identity=IDENTITY)


@pytest.mark.parametrize(
    ("observed_local_active", "running_inputs", "input_headroom", "expected"),
    [
        (0, 60, 4, 4),
        (2, 3, 4, 6),
        (2, 1, 4, 5),
        (64, 60, 4, 64),
        (0, 64, 0, 0),
    ],
)
def test_fixed_limit_accounts_for_reflected_and_unreflected_local_admissions(
    observed_local_active: int,
    running_inputs: int,
    input_headroom: int,
    expected: int,
) -> None:
    assert (
        fixed_local_active_limit(
            observed_local_active,
            running_inputs,
            input_headroom,
            hard_limit=128,
        )
        == expected
    )


def test_router_restart_uses_only_function_stats_headroom() -> None:
    assert fixed_local_active_limit(0, running_inputs=60, input_headroom=4, hard_limit=128) == 4


def test_backlog_is_observed_but_does_not_change_fixed_limit() -> None:
    without_backlog = _snapshot(backlog=0)
    with_backlog = _snapshot(backlog=999)

    assert without_backlog.local_active_limit == 64
    assert with_backlog.local_active_limit == 64


def test_fixed_limit_is_capped_by_hard_limit() -> None:
    assert fixed_local_active_limit(10, running_inputs=0, input_headroom=1000, hard_limit=64) == 64


@pytest.mark.parametrize(
    "values",
    [
        (True, 0, 1, 16),
        (-1, 0, 1, 16),
        (0, 1.5, 1, 16),
        (0, 0, "1", 16),
        (0, 0, 1, 0),
    ],
)
def test_fixed_limit_rejects_malformed_counts(
    values: tuple[object, object, object, object],
) -> None:
    with pytest.raises(ValueError, match=r"capacity observation counts|hard_limit"):
        fixed_local_active_limit(*values)


def test_configured_provider_is_explicitly_offline_test_only() -> None:
    provider = ConfiguredCapacityProvider(clock=lambda: 10.0)

    snapshot = asyncio.run(provider.capacity_snapshot(MODEL, observed_local_active=3))

    assert snapshot.deployment_identity == "offline/test-only:configured-capacity"
    assert snapshot.observed_local_active == 3
    assert snapshot.local_active_limit == 16
    assert provider.current_dispatch_capacity(MODEL) == 16

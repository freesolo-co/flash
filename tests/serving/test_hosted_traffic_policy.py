"""Hosted serving traffic policy configuration."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from flash.serving.src.engine.model_config import (
    _QWEN38_HOSTED_CANDIDATE,
    SERVING_MODELS,
    HostedTrafficPolicy,
    base_models,
    configured_warm_container_floor,
    hosted_traffic_policy_for,
)


def test_catalog_policy_preserves_explicit_current_values() -> None:
    for model in base_models():
        policy = hosted_traffic_policy_for(model)
        assert policy.max_num_seqs == 8
        assert policy.max_inputs == 8
        assert policy.target_inputs == 6
        assert policy.min_containers == 1
        assert policy.buffer_containers == 1

    assert all(model["traffic"] == {"max_inputs": 8, "target_inputs": 6} for model in SERVING_MODELS)
    assert _QWEN38_HOSTED_CANDIDATE["traffic"] == {"max_inputs": 8, "target_inputs": 6}


def test_request_concurrency_is_independent_of_engine_sequence_capacity() -> None:
    policy = HostedTrafficPolicy.from_config(
        {"max_num_seqs": 12},
        {"max_inputs": 8, "target_inputs": 6},
    )

    assert policy.max_num_seqs == 12
    assert policy.max_inputs == 8
    assert policy.target_inputs == 6


def test_catalog_warm_floor_is_aggregated() -> None:
    assert configured_warm_container_floor() == len(base_models())


def test_catalog_remains_json_serializable() -> None:
    json.dumps(SERVING_MODELS)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min_containers", 2, "min_containers"),
        ("buffer_containers", 0, "buffer_containers"),
        ("max_num_seqs", 0, "max_num_seqs"),
        ("max_inputs", 0, "max_inputs"),
        ("target_inputs", 0, "target_inputs"),
        ("target_inputs", 9, "target_inputs"),
    ],
)
def test_direct_policy_construction_rejects_values_outside_contract(
    field: str, value: int, message: str
) -> None:
    policy = HostedTrafficPolicy.from_config(
        {"max_num_seqs": 8},
        {"max_inputs": 8, "target_inputs": 6},
    )

    with pytest.raises(ValueError, match=message):
        replace(policy, **{field: value})


@pytest.mark.parametrize("value", [None, True, False, "8", 8.0, 0, -1])
def test_policy_rejects_malformed_max_num_seqs(value: object) -> None:
    engine = {} if value is None else {"max_num_seqs": value}

    with pytest.raises(ValueError, match="max_num_seqs"):
        HostedTrafficPolicy.from_config(
            engine,
            {"max_inputs": 8, "target_inputs": 6},
        )


@pytest.mark.parametrize("field", ["max_inputs", "target_inputs"])
@pytest.mark.parametrize("value", [None, True, False, "8", 8.0, 0, -1])
def test_policy_rejects_missing_malformed_or_nonpositive_traffic_fields(
    field: str, value: object
) -> None:
    traffic: dict[str, object] = {"max_inputs": 8, "target_inputs": 6}
    if value is None:
        traffic.pop(field)
    else:
        traffic[field] = value

    with pytest.raises(ValueError, match=field):
        HostedTrafficPolicy.from_config({"max_num_seqs": 8}, traffic)


def test_policy_rejects_target_inputs_above_max_inputs() -> None:
    with pytest.raises(ValueError, match="target_inputs"):
        HostedTrafficPolicy.from_config(
            {"max_num_seqs": 8},
            {"max_inputs": 7, "target_inputs": 8},
        )

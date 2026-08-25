"""Hosted serving traffic policy derivation and catalog totals."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from flash.serving.src.engine.model_config import (
    SERVING_MODELS,
    HostedTrafficPolicy,
    base_models,
    configured_hard_gpu_ceiling,
    configured_warm_container_floor,
    hosted_traffic_policy_for,
)


def test_catalog_policy_derives_8_max_and_6_target_inputs() -> None:
    for model in base_models():
        policy = hosted_traffic_policy_for(model)
        assert policy.max_num_seqs == 8
        assert policy.max_inputs == 8
        assert policy.target_inputs == 6
        assert policy.min_containers == 1
        assert policy.max_containers == 2
        assert policy.buffer_containers == 0
        assert policy.queue_capacity == 2
        assert policy.retry_after_seconds == 1


def test_catalog_warm_floor_and_hard_gpu_ceiling_are_aggregated() -> None:
    assert configured_warm_container_floor() == len(base_models())
    assert configured_hard_gpu_ceiling() == 2 * len(base_models())


def test_catalog_remains_json_serializable() -> None:
    json.dumps(SERVING_MODELS)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min_containers", 2, "container limits"),
        ("max_containers", 3, "container limits"),
        ("buffer_containers", 1, "buffer_containers"),
        ("queue_capacity", 9, "queue_capacity"),
        ("retry_after_seconds", 2, "retry_after_seconds"),
        ("max_num_seqs", 0, "max_num_seqs"),
        ("max_inputs", 7, "max_inputs"),
        ("target_inputs", 7, "target_inputs"),
    ],
)
def test_direct_policy_construction_rejects_values_outside_fixed_contract(
    field: str, value: int, message: str
) -> None:
    policy = HostedTrafficPolicy.from_engine({"max_num_seqs": 8})

    with pytest.raises(ValueError, match=message):
        replace(policy, **{field: value})


@pytest.mark.parametrize("value", [None, True, False, "8", 8.0, 0, -1])
def test_policy_rejects_malformed_max_num_seqs(value: object) -> None:
    engine = {} if value is None else {"max_num_seqs": value}

    with pytest.raises(ValueError, match="explicit positive max_num_seqs"):
        HostedTrafficPolicy.from_engine(engine)

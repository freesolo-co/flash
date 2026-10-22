"""JSON nesting failures must follow the broker's validation/error paths."""

import sys

import pytest

from flash.server.domain.teacher import broker, requests
from flash.server.domain.teacher.errors import TeacherBrokerError


@pytest.fixture
def deep_json():
    depth = max(10000, sys.getrecursionlimit() + 100)
    return b'{"nested":' + b"[" * depth + b"0" + b"]" * depth + b"}"


def test_deep_teacher_request_is_a_validation_error(deep_json):
    with pytest.raises(TeacherBrokerError) as raised:
        requests.parse_strict_json(deep_json)
    assert raised.value.status_code == 400


@pytest.mark.parametrize(
    "validator", [broker._validated_provider_response, broker._validated_chat_provider_response]
)
def test_deep_provider_response_is_a_contract_error(validator, deep_json):
    with pytest.raises(TeacherBrokerError) as raised:
        validator(deep_json, score_items=1, capability={"max_response_bytes": len(deep_json)})
    assert raised.value.code == "provider_contract_error"
    assert raised.value.status_code == 502

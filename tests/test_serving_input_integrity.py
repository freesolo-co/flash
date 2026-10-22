"""Malformed JSON tags must produce validation failures, not internal errors."""

import httpx
import pytest
from pydantic import ValidationError

from flash.serve.request.openai import OpenAIRequestError, parse_chat_request
from flash.serve.request.transport import retryable_smoke_unavailable
from flash.serve.request.validation import has_image_blocks
from flash.serving.src.io.openai_request import OpenAIGenerateRequest


@pytest.mark.parametrize("value", [[], {}, [{}], {"type": []}])
@pytest.mark.parametrize("field", ["role", "type"])
def test_chat_parser_rejects_non_string_message_tags(value, field):
    message = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    if field == "role":
        message["role"] = value
    else:
        message["content"][0]["type"] = value
    with pytest.raises(OpenAIRequestError, match=r"role|unsupported type"):
        parse_chat_request(
            {"model": "model", "messages": [message]},
            require_model=True,
            allow_managed_selectors=False,
        )


@pytest.mark.parametrize("value", [[], {}])
def test_image_detector_tolerates_unknown_type_shapes(value):
    assert not has_image_blocks(
        [{"role": "user", "content": [{"type": value}]}], sequence_types=list
    )


@pytest.mark.parametrize("value", [[], {}])
def test_hosted_request_rejects_non_string_tool_choice(value):
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }
    with pytest.raises(ValidationError, match="tool_choice must be auto or none"):
        OpenAIGenerateRequest(
            model_id="model",
            messages=[{"role": "user", "content": "hello"}],
            tools=[tool],
            tool_choice=value,
            parallel_tool_calls=True,
        )


@pytest.mark.parametrize("value", [[], {}])
def test_smoke_classifier_ignores_malformed_error_code(value):
    response = httpx.Response(
        503,
        json={
            "error": {
                "type": "adapter_unavailable",
                "retryable": True,
                "code": value,
                "requested_model": "run/final",
                "checkpoint_id": "run/final",
            }
        },
    )
    assert (
        retryable_smoke_unavailable(
            response, requested_model="run/final", expected_checkpoint_id="run/final"
        )
        is None
    )

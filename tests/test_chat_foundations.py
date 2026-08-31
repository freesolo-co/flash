from __future__ import annotations

import json
import subprocess
import sys
from types import MappingProxyType

import pytest

from flash._internal.openai_sse import (
    DeltaEvent,
    DoneEvent,
    OpenAISSEError,
    iter_openai_sse_events,
)
from flash.client.http import ClientError
from flash.serve.contract.provenance import (
    CheckpointProvenance,
    decode_flash_body,
    decode_flash_headers,
    decode_freesolo_body,
    decode_freesolo_headers,
    validate_header_provenance,
)
from flash.serve.request.openai import (
    DEFAULT_MAX_TOKENS,
    OpenAIRequestError,
    parse_chat_request,
    reject_tool_capability,
)
from flash.serve.request.streaming import _complete_sse_frames
from flash.serve.request.tool_calls import detached_template_messages
from flash.serve.request.transport import OpenAIStreamResponse
from flash.server.routes.serving_revisions import _authorized_chat_checkpoint


def test_canonical_request_parser_owns_defaults_and_strict_schema() -> None:
    request = parse_chat_request(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {
                "custom": 1,
                "enable_thinking": True,
                "return_tensors": "pt",
            },
        },
        require_model=False,
        allow_managed_selectors=True,
    )

    assert request.max_tokens == DEFAULT_MAX_TOKENS == 1024
    assert request.chat_template_kwargs == {"custom": 1, "enable_thinking": True}

    for strict in (False, "true", 1):
        with pytest.raises(OpenAIRequestError, match="strict"):
            parse_chat_request(
                {
                    "model": "run-1",
                    "messages": [{"role": "user", "content": "hello"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "schema": {"type": "object"},
                            "strict": strict,
                        },
                    },
                },
                require_model=True,
                allow_managed_selectors=False,
            )


def test_request_parser_rejects_retired_step_selector() -> None:
    with pytest.raises(OpenAIRequestError, match=r"unsupported chat request field.*step"):
        parse_chat_request(
            {
                "model": "run-1/final",
                "messages": [{"role": "user", "content": "hello"}],
                "step": 20,
            },
            require_model=True,
            allow_managed_selectors=True,
        )


@pytest.mark.parametrize("field", ["temperature", "top_p"])
def test_request_parser_rejects_numeric_overflow_as_a_controlled_error(field: str) -> None:
    with pytest.raises(OpenAIRequestError, match=f"{field} must be a finite number"):
        parse_chat_request(
            {
                "messages": [{"role": "user", "content": "hello"}],
                field: 10**400,
            },
            require_model=False,
            allow_managed_selectors=True,
        )


def test_thinking_import_does_not_load_deploy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import flash.serve.request.thinking; "
                "assert 'flash.serve.deployment.deploy' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_decoded_sse_parser_yields_typed_events_and_requires_done() -> None:
    events = list(
        iter_openai_sse_events(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"r"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
                "data: [DONE]\n\n",
            ]
        )
    )
    assert events == [DeltaEvent("r", None), DeltaEvent(None, "a"), DoneEvent()]

    with pytest.raises(OpenAISSEError, match=r"terminal \[DONE\]"):
        list(iter_openai_sse_events(['data: {"choices":[{"delta":{"content":"partial"}}]}\n\n']))


@pytest.mark.parametrize("payload", [{}, {"choices": []}])
def test_decoded_sse_parser_accepts_absent_or_empty_choices(payload: dict) -> None:
    events = list(iter_openai_sse_events([f"data: {json.dumps(payload)}\n\n", "data: [DONE]\n\n"]))

    assert events == [DoneEvent()]


@pytest.mark.parametrize("choices", [{}, "", 0, False, None])
def test_decoded_sse_parser_rejects_present_non_array_choices(choices: object) -> None:
    with pytest.raises(OpenAISSEError, match="choices must be an array"):
        list(iter_openai_sse_events([f'data: {{"choices":{json.dumps(choices)}}}\n\n']))


@pytest.mark.parametrize("choice", [None, "", 0, False, []])
def test_decoded_sse_parser_rejects_non_object_choice_entries(choice: object) -> None:
    with pytest.raises(OpenAISSEError, match="choice must be an object"):
        list(iter_openai_sse_events([f'data: {{"choices":[{json.dumps(choice)}]}}\n\n']))


@pytest.mark.parametrize("delta", [None, "", 0, False, []])
def test_decoded_sse_parser_rejects_present_non_object_delta(delta: object) -> None:
    with pytest.raises(OpenAISSEError, match="delta must be an object"):
        list(iter_openai_sse_events([f'data: {{"choices":[{{"delta":{json.dumps(delta)}}}]}}\n\n']))


@pytest.mark.parametrize("field", ["content", "reasoning_content"])
@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_decoded_sse_parser_rejects_present_non_string_text_fields(
    field: str, value: object
) -> None:
    payload = {"choices": [{"delta": {field: value}}]}

    with pytest.raises(OpenAISSEError, match=rf"{field} must be a string or null"):
        list(iter_openai_sse_events([f"data: {json.dumps(payload)}\n\n"]))


@pytest.mark.parametrize("field", ["content", "reasoning_content"])
def test_decoded_sse_parser_accepts_absent_or_null_text_fields(field: str) -> None:
    null_payload = {"choices": [{"delta": {field: None}}]}
    absent_payload = {"choices": [{"delta": {}}]}

    events = list(
        iter_openai_sse_events(
            [
                f"data: {json.dumps(null_payload)}\n\n",
                f"data: {json.dumps(absent_payload)}\n\n",
                "data: [DONE]\n\n",
            ]
        )
    )

    assert events == [DoneEvent()]


@pytest.mark.parametrize("error", [None, False, 0, "failure", []])
def test_decoded_sse_parser_rejects_present_non_object_error(error: object) -> None:
    with pytest.raises(OpenAISSEError, match="error must be an object"):
        list(iter_openai_sse_events([f'data: {{"error":{json.dumps(error)}}}\n\n']))


def test_decoded_sse_parser_accepts_cr_only_line_endings() -> None:
    payload = 'data: {"choices":[{"delta":{"content":"cr"}}]}\r\rdata: [DONE]\r\r'

    assert list(iter_openai_sse_events(payload)) == [DeltaEvent(None, "cr"), DoneEvent()]


def test_decoded_sse_parser_accepts_mixed_line_endings_across_chunks() -> None:
    chunks = iter(
        [
            'data: {"choices":[{"delta":{"content":"mixed"}}]}\r',
            "\n\r",
            "data: [DONE]\n",
            "\r",
            "\n",
        ]
    )

    assert list(iter_openai_sse_events(chunks)) == [DeltaEvent(None, "mixed"), DoneEvent()]


def test_decoded_sse_parser_strips_one_initial_bom() -> None:
    chunks = [
        '﻿data: {"choices":[{"delta":{"content":"accepted"}}]}\n\n',
        "data: [DONE]\n\n",
    ]

    assert list(iter_openai_sse_events(chunks)) == [DeltaEvent(None, "accepted"), DoneEvent()]


def test_decoded_sse_parser_strips_only_one_leading_bom() -> None:
    bom = chr(0xFEFF)

    with pytest.raises(OpenAISSEError, match=r"terminal \[DONE\]"):
        list(iter_openai_sse_events([f"{bom}{bom}data: [DONE]\n\n"]))


def test_decoded_sse_parser_does_not_strip_a_later_event_bom() -> None:
    chunks = [
        'data: {"choices":[{"delta":{"content":"first"}}]}\n\n',
        "﻿data: [DONE]\n\n",
    ]

    events = iter_openai_sse_events(chunks)
    assert next(events) == DeltaEvent(None, "first")
    with pytest.raises(OpenAISSEError, match=r"terminal \[DONE\]"):
        next(events)


def test_decoded_sse_parser_preserves_bom_in_payload_content() -> None:
    payload = {"choices": [{"delta": {"content": "﻿answer"}}]}

    assert list(
        iter_openai_sse_events([f"data: {json.dumps(payload)}\n\n", "data: [DONE]\n\n"])
    ) == [DeltaEvent(None, "﻿answer"), DoneEvent()]


def test_decoded_sse_parser_joins_data_lines_only_after_frame_delimiter() -> None:
    chunks = iter(
        [
            'data: {"choices":\n',
            'data: [{"delta":{"content":"joined"}}]}\n',
            "\n",
            "data: [DONE]\n\n",
        ]
    )

    events = iter_openai_sse_events(chunks)
    assert next(events) == DeltaEvent(None, "joined")
    assert next(events) == DoneEvent()


def test_decoded_sse_parser_rejects_complete_line_without_frame_delimiter() -> None:
    with pytest.raises(OpenAISSEError, match="incomplete server-sent event frame"):
        list(iter_openai_sse_events(['data: {"choices":[]}\n']))


class _Response:
    status_code = 200

    def __init__(self) -> None:
        self.headers = {"content-type": "text/plain"}

    def iter_bytes(self):
        yield b"one"


class _Context:
    def __init__(self) -> None:
        self.closed = False

    def __exit__(self, *_args):
        self.closed = True
        return False


@pytest.mark.parametrize(
    "payload",
    [
        b'data: {"choices":[]}\r\rdata: [DONE]\r\r',
        b'data: {"choices":[]}\r\n\rdata: [DONE]\n\r\n',
    ],
    ids=["cr-only", "mixed"],
)
def test_raw_sse_framing_preserves_valid_line_endings_across_byte_chunks(payload: bytes) -> None:
    chunks = (payload[index : index + 1] for index in range(len(payload)))

    assert b"".join(_complete_sse_frames(chunks)) == payload


def test_raw_sse_framing_preserves_an_initial_bom() -> None:
    payload = b"\xef\xbb\xbfdata: [DONE]\r\r"

    assert b"".join(_complete_sse_frames(iter([payload]))) == payload


def test_raw_sse_framing_strips_only_one_leading_bom_for_terminal_detection() -> None:
    payload = b"\xef\xbb\xbf\xef\xbb\xbfdata: [DONE]\r\r"
    frames = _complete_sse_frames(iter([payload]))

    assert next(frames) == payload
    with pytest.raises(ClientError, match=r"terminal \[DONE\]"):
        next(frames)


def test_raw_sse_framing_does_not_accept_a_later_bom_as_terminal() -> None:
    payload = b'data: {"choices":[]}\n\n\xef\xbb\xbfdata: [DONE]\n\n'
    frames = _complete_sse_frames(iter([payload]))

    assert next(frames) == b'data: {"choices":[]}\n\n'
    assert next(frames) == b"\xef\xbb\xbfdata: [DONE]\n\n"
    with pytest.raises(ClientError, match=r"terminal \[DONE\]"):
        next(frames)


def test_raw_stream_bytes_are_one_shot_and_owned() -> None:
    context = _Context()
    upstream = _Response()
    response = OpenAIStreamResponse(
        status_code=200,
        headers=dict(upstream.headers),
        context=context,
        response=upstream,
        frame_bytes=lambda chunks: chunks,
    )

    assert list(response.iter_bytes()) == [b"one"]
    assert context.closed
    with pytest.raises(RuntimeError, match="already been claimed"):
        response.iter_bytes()


def test_provenance_decoders_share_one_typed_value() -> None:
    checkpoint_id = "run-1/step-7"
    expected = CheckpointProvenance(checkpoint_id)

    assert decode_freesolo_body(expected.freesolo_body()) == expected
    assert decode_flash_body({"checkpoint_id": checkpoint_id, "deployment_id": "kept"}) == expected
    assert decode_freesolo_headers(expected.freesolo_headers()) == expected
    assert decode_flash_headers({"X-Flash-Checkpoint-Id": checkpoint_id}) == expected


def test_header_provenance_validates_every_present_family() -> None:
    expected = CheckpointProvenance("run-1/step-7")
    matching = {
        "X-Freesolo-Checkpoint": expected.checkpoint_id,
        "X-Flash-Checkpoint-Id": expected.checkpoint_id,
    }

    validate_header_provenance(matching, expected)

    with pytest.raises(ValueError, match="mismatched checkpoint provenance"):
        validate_header_provenance({**matching, "X-Flash-Checkpoint-Id": "run-1/step-8"}, expected)


def _function_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "look up weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def test_tool_controls_and_template_keys_are_strict() -> None:
    request = parse_chat_request(
        {
            "messages": [{"role": "user", "content": "weather"}],
            "tools": _function_tools(),
            "chat_template_kwargs": {"tools": ["bypass"], "tool_choice": "required"},
        },
        require_model=False,
        allow_managed_selectors=True,
    )
    assert request.tool_choice == "auto"
    assert request.parallel_tool_calls is True
    assert request.chat_template_kwargs == {}
    for update, match in (
        ({"tool_choice": "required"}, "auto or none"),
        ({"parallel_tool_calls": False}, "must be true"),
    ):
        with pytest.raises(OpenAIRequestError, match=match):
            parse_chat_request(
                {
                    "messages": [{"role": "user", "content": "weather"}],
                    "tools": _function_tools(),
                    **update,
                },
                require_model=False,
                allow_managed_selectors=True,
            )


def test_enum_member_complexity_is_a_request_error() -> None:
    tools = _function_tools()
    tools[0]["function"]["parameters"]["properties"]["days"]["enum"] = [
        json.loads("[" * 600 + "0" + "]" * 600)
    ]

    with pytest.raises(OpenAIRequestError, match="enum value complexity"):
        parse_chat_request(
            {"messages": [{"role": "user", "content": "weather"}], "tools": tools},
            require_model=False,
            allow_managed_selectors=True,
        )


def test_tool_names_and_schema_container_keywords_are_exact() -> None:
    valid = _function_tools()
    valid[0]["function"]["name"] = "9-weather_tool"
    request = parse_chat_request(
        {"messages": [{"role": "user", "content": "weather"}], "tools": valid},
        require_model=False,
        allow_managed_selectors=True,
    )
    assert request.tools is not None
    assert request.tools[0].name == "9-weather_tool"

    for name in ("weather.lookup", "x" * 65):
        invalid = _function_tools()
        invalid[0]["function"]["name"] = name
        with pytest.raises(OpenAIRequestError, match=r"function\.name is invalid"):
            parse_chat_request(
                {"messages": [{"role": "user", "content": "weather"}], "tools": invalid},
                require_model=False,
                allow_managed_selectors=True,
            )

    for name in ("city.name", "city>", "x" * 65):
        invalid = _function_tools()
        invalid[0]["function"]["parameters"]["properties"] = {name: {"type": "string"}}
        invalid[0]["function"]["parameters"]["required"] = [name]
        with pytest.raises(OpenAIRequestError, match=r"properties key is invalid"):
            parse_chat_request(
                {"messages": [{"role": "user", "content": "weather"}], "tools": invalid},
                require_model=False,
                allow_managed_selectors=True,
            )

    invalid_schema = _function_tools()
    invalid_schema[0]["function"]["parameters"]["items"] = {"type": "string"}
    with pytest.raises(OpenAIRequestError, match="object schema contains array-only keywords"):
        parse_chat_request(
            {
                "messages": [{"role": "user", "content": "weather"}],
                "tools": invalid_schema,
            },
            require_model=False,
            allow_managed_selectors=True,
        )


def test_tool_capability_rejection_uses_authoritative_thinking_and_parser() -> None:
    tools = parse_chat_request(
        {"messages": [{"role": "user", "content": "weather"}], "tools": _function_tools()},
        require_model=False,
        allow_managed_selectors=True,
    ).tools
    reject_tool_capability(
        tools=tools,
        tool_choice="auto",
        thinking=False,
        tool_parser="qwen3_coder",
    )
    reject_tool_capability(tools=tools, tool_choice="none", thinking=True, tool_parser=None)
    with pytest.raises(OpenAIRequestError, match="thinking-enabled"):
        reject_tool_capability(
            tools=tools,
            tool_choice="auto",
            thinking=True,
            tool_parser="qwen3_coder",
        )
    with pytest.raises(OpenAIRequestError, match="not qualified"):
        reject_tool_capability(
            tools=tools,
            tool_choice="auto",
            thinking=False,
            tool_parser=None,
        )


def test_message_copy_complexity_is_controlled_and_caller_values_stay_detached() -> None:
    nested: dict[str, object] = {}
    for _ in range(1500):
        nested = {"extra": nested}
    messages = [{"role": "user", "content": "hello", "metadata": nested}]

    with pytest.raises(OpenAIRequestError, match="messages exceed the supported complexity"):
        parse_chat_request(
            {"messages": messages},
            require_model=False,
            allow_managed_selectors=True,
        )

    assert messages[0]["metadata"] is nested
    metadata = {"nested": {"value": 1}}
    request = parse_chat_request(
        {"messages": [{"role": "user", "content": "hello", "metadata": metadata}]},
        require_model=False,
        allow_managed_selectors=True,
    )
    metadata["nested"]["value"] = 2
    assert request.messages[0]["metadata"] == {"nested": {"value": 1}}


def test_canonical_message_mapping_proxy_is_detached() -> None:
    metadata = {"value": 1}
    request = parse_chat_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                    "metadata": MappingProxyType(metadata),
                }
            ]
        },
        require_model=False,
        allow_managed_selectors=True,
    )

    metadata["value"] = 2
    assert request.messages[0]["metadata"] == {"value": 1}


@pytest.mark.parametrize(
    "content",
    ["bad\ud800", [{"type": "text", "text": "bad\ud800"}]],
    ids=["string", "text-block"],
)
def test_tool_result_content_rejects_unpaired_surrogates(content) -> None:
    messages = _historical_tool_messages("{}")
    messages[1]["content"] = content

    with pytest.raises(
        OpenAIRequestError, match="tool result content cannot contain an unpaired surrogate"
    ):
        parse_chat_request(
            {"messages": messages},
            require_model=False,
            allow_managed_selectors=True,
        )


def test_tool_result_content_accepts_non_bmp_text() -> None:
    for content in ("sunny ☀", [{"type": "text", "text": "sunny ☀"}]):
        request = parse_chat_request(
            {
                "messages": [
                    *_historical_tool_messages("{}")[:1],
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": content,
                    },
                ]
            },
            require_model=False,
            allow_managed_selectors=True,
        )
        assert request.messages[1]["content"] == content


def test_tool_history_is_strict_and_does_not_mutate_caller_messages() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "weather", "content": "sunny"},
        {"role": "user", "content": "thanks"},
    ]
    original = json.loads(json.dumps(messages))
    request = parse_chat_request(
        {"messages": messages}, require_model=False, allow_managed_selectors=True
    )
    assert request.messages == original
    assert messages == original
    with pytest.raises(OpenAIRequestError, match="before all preceding tool calls were resolved"):
        parse_chat_request(
            {"messages": [messages[0], messages[2]]},
            require_model=False,
            allow_managed_selectors=True,
        )

    for message, match in (
        (
            {
                "role": "user",
                "content": "not an assistant",
                "tool_calls": messages[0]["tool_calls"],
            },
            "tool_calls require the assistant role",
        ),
        (
            {"role": "assistant", "content": "not a tool", "tool_call_id": "call_1"},
            "tool_call_id requires the tool role",
        ),
    ):
        with pytest.raises(OpenAIRequestError, match=match):
            parse_chat_request(
                {"messages": [message]},
                require_model=False,
                allow_managed_selectors=True,
            )

    invalid_history = json.loads(json.dumps(messages[0]))
    invalid_history["tool_calls"][0]["function"]["name"] = "weather.lookup"
    with pytest.raises(OpenAIRequestError, match="function name is invalid"):
        parse_chat_request(
            {"messages": [invalid_history]},
            require_model=False,
            allow_managed_selectors=True,
        )

    text_parts = parse_chat_request(
        {
            "messages": [
                messages[0],
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "input_text", "text": "sun"},
                        {"type": "text", "text": "ny"},
                    ],
                },
            ]
        },
        require_model=False,
        allow_managed_selectors=True,
    )
    assert text_parts.messages[1]["content"] == [
        {"type": "input_text", "text": "sun"},
        {"type": "text", "text": "ny"},
    ]


def _historical_tool_messages(argument: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": argument},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]


@pytest.mark.parametrize(
    "argument",
    [
        '{"value":' + "[" * 600 + "0" + "]" * 600 + "}",
        json.dumps({"values": [0] * 511}),
    ],
    ids=["depth", "aggregate-nodes"],
)
def test_historical_tool_argument_complexity_is_a_request_error(argument: str) -> None:
    messages = _historical_tool_messages(argument)
    original = json.loads(json.dumps(messages))

    with pytest.raises(OpenAIRequestError, match="tool argument complexity"):
        parse_chat_request(
            {"messages": messages},
            require_model=False,
            allow_managed_selectors=True,
        )

    assert messages == original


@pytest.mark.parametrize(
    "argument",
    [
        '{"value":' + "[" * 7 + "0" + "]" * 7 + "}",
        json.dumps({"values": [0] * 510}),
    ],
    ids=["depth", "aggregate-nodes"],
)
def test_historical_tool_argument_complexity_boundary_succeeds(argument: str) -> None:
    messages = _historical_tool_messages(argument)
    original = json.loads(json.dumps(messages))

    request = parse_chat_request(
        {"messages": messages},
        require_model=False,
        allow_managed_selectors=True,
    )

    assert request.messages == original
    assert messages == original


@pytest.mark.parametrize(
    ("argument", "match"),
    [
        ('{"days":1e1000001}', "numeric exponent exceeds 1000000 magnitude limit"),
        ('{"days":1e-1000001}', "numeric exponent exceeds 1000000 magnitude limit"),
        ('{"days":1,"days":2}', "arguments must encode a JSON object"),
        ('{"nested":{"days":1,"days":2}}', "arguments must encode a JSON object"),
    ],
    ids=["positive-exponent", "negative-exponent", "duplicate-root", "duplicate-nested"],
)
def test_malformed_numeric_or_duplicate_key_tool_history_is_a_request_error(
    argument: str, match: str
) -> None:
    messages = _historical_tool_messages(argument)

    with pytest.raises(OpenAIRequestError, match=match):
        parse_chat_request(
            {"messages": messages},
            require_model=False,
            allow_managed_selectors=True,
        )


@pytest.mark.parametrize(
    "argument",
    [
        '{"direct":6.25e-1}',
        '{"nested":{"value":2.5e1}}',
        '{"values":[1e2,1.25e-2]}',
    ],
    ids=["direct", "nested", "list"],
)
def test_finite_exponent_tool_history_is_accepted(argument: str) -> None:
    messages = _historical_tool_messages(argument)

    request = parse_chat_request(
        {"messages": messages},
        require_model=False,
        allow_managed_selectors=True,
    )

    assert request.messages == messages


@pytest.mark.parametrize(
    "argument",
    [
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
    ],
    ids=["nan", "infinity", "negative-infinity"],
)
def test_nonfinite_tool_history_is_a_request_error(argument: str) -> None:
    """a non-finite constant is not JSON and has no faithful rendering at all."""
    with pytest.raises(OpenAIRequestError, match="arguments must encode a JSON object"):
        parse_chat_request(
            {"messages": _historical_tool_messages(argument)},
            require_model=False,
            allow_managed_selectors=True,
        )


@pytest.mark.parametrize(
    ("argument", "rendered"),
    [
        ('{"value":1.' + "0" * 309 + "1e309}", "1" + "0" * 309 + ".1"),
        ('{"value":1e-400}', "1e-400"),
        ('{"value":9007199254740993.1}', "9007199254740993.1"),
    ],
    ids=["overflow", "nonzero-underflow", "lossy-nonintegral"],
)
def test_finite_number_no_native_value_carries_renders_exactly(
    argument: str, rendered: str
) -> None:
    """a finite number keeps its exact text rather than rounding or being refused.

    each of these overflows, underflows, or loses digits when forced through a python
    float, which is why they used to be rejected. the template renders a scalar through
    ``string``, so the exact literal reaches the model unchanged and the prior call the
    model sees is the one it actually made.
    """
    normalized = parse_chat_request(
        {"messages": _historical_tool_messages(argument)},
        require_model=False,
        allow_managed_selectors=True,
    )

    detached = detached_template_messages(normalized.messages)
    values = detached[0]["tool_calls"][0]["function"]["arguments"]
    assert values["value"] == rendered


@pytest.mark.parametrize(
    "argument",
    [
        '{"direct":' + "9" * 1025 + "}",
        '{"nested":{"value":' + "9" * 1025 + "}}",
        '{"values":[' + "9" * 1025 + "]}",
    ],
    ids=["direct", "nested", "list"],
)
def test_oversized_integer_tool_history_is_a_request_error(argument: str) -> None:
    with pytest.raises(OpenAIRequestError, match="1024-digit limit"):
        parse_chat_request(
            {"messages": _historical_tool_messages(argument)},
            require_model=False,
            allow_managed_selectors=True,
        )


@pytest.mark.parametrize(
    ("argument", "rendered"),
    [
        ('{"direct":1e1024}', "1e+1024"),
        ('{"nested":{"value":1e1024}}', '{"value": 1e+1024}'),
        ('{"values":[1e1024,2]}', "[1e+1024, 2]"),
        ('{"pair":{"a":1e1024,"b":2}}', '{"a": 1e+1024, "b": 2}'),
        ('{"mixed":{"trailing":1.2300,"huge":1e1024}}', '{"trailing": 1.23, "huge": 1e+1024}'),
        ('{"signed":{"zero":-0.0,"huge":1e1024}}', '{"zero": -0.0, "huge": 1e+1024}'),
        ('{"enabled":true}', "true"),
        ('{"disabled":false}', "false"),
        ('{"value":null}', "null"),
    ],
    ids=[
        "direct",
        "nested",
        "list",
        "pair",
        "mixed-native-leaf",
        "mixed-signed-zero",
        "scalar-true",
        "scalar-false",
        "scalar-null",
    ],
)
def test_compact_exponent_tool_history_renders_without_expanding(
    argument: str, rendered: str
) -> None:
    """a compact exponent survives as history because the template never expands it.

    the grammar template sends a scalar through ``string`` and a container through
    ``tojson``, neither of which needs the fixed expansion. expanding here would turn a
    seven-character literal into a thousand-digit prompt and eventually trip python's own
    integer-to-string limit, so the exact compact text is what the model must see.
    """
    normalized = parse_chat_request(
        {"messages": _historical_tool_messages(argument)},
        require_model=False,
        allow_managed_selectors=True,
    )

    detached = detached_template_messages(normalized.messages)
    values = detached[0]["tool_calls"][0]["function"]["arguments"]
    assert next(iter(values.values())) == rendered


@pytest.mark.parametrize(
    ("argument", "rendered"),
    [
        ('{"wrapped":{"enabled":true,"value":null}}', '{"enabled": true, "value": null}'),
        ('{"listed":[true,false,null]}', "[true, false, null]"),
    ],
    ids=["nested", "listed"],
)
def test_contained_boolean_and_null_history_keeps_native_values(
    argument: str, rendered: str
) -> None:
    """a bool or null inside a container stays native, because ``tojson`` spells it right.

    only the scalar position needs pre-rendering: there the template uses ``string``, which
    would emit python's ``True`` and ``None``. pre-rendering a container as well would hand
    ``tojson`` a string and quote the whole structure, so the two positions differ.
    """
    normalized = parse_chat_request(
        {"messages": _historical_tool_messages(argument)},
        require_model=False,
        allow_managed_selectors=True,
    )

    detached = detached_template_messages(normalized.messages)
    values = detached[0]["tool_calls"][0]["function"]["arguments"]
    value = next(iter(values.values()))
    assert not isinstance(value, str)
    assert json.dumps(value, ensure_ascii=False) == rendered


@pytest.mark.parametrize(
    ("argument", "rendered"),
    [
        ('{"zero":-0.0}', "-0.0"),
        ('{"nested":{"zero":-0.0}}', '{"zero": -0.0}'),
        # the positive zero alongside it still collapses to an integer, which is what makes
        # this pair show that only the sign, not the decimal point, is what gets preserved.
        ('{"listed":[-0.0,0.0]}', "[-0.0, 0]"),
    ],
    ids=["scalar", "nested", "listed"],
)
def test_negative_zero_history_keeps_its_sign(argument: str, rendered: str) -> None:
    """a negative zero must replay as ``-0.0``, not as ``0``.

    ``-0.0`` is integral, so the decimal path used to hand it to ``int`` and drop the sign,
    showing the model a different prior call than the one flash emitted. ``float`` carries
    the sign and renders it back exactly, so signed zero takes that path instead. positive
    zero has no sign to lose and stays an integer.
    """
    normalized = parse_chat_request(
        {"messages": _historical_tool_messages(argument)},
        require_model=False,
        allow_managed_selectors=True,
    )

    detached = detached_template_messages(normalized.messages)
    values = detached[0]["tool_calls"][0]["function"]["arguments"]
    value = next(iter(values.values()))
    assert not isinstance(value, str)
    assert json.dumps(value, ensure_ascii=False) == rendered


def test_positive_zero_history_stays_an_integer() -> None:
    """the control for ``test_negative_zero_history_keeps_its_sign``.

    without this, widening the integral branch to send every zero through ``float`` would
    pass the signed-zero test while silently changing ``0`` into ``0.0`` for everyone else.
    """
    normalized = parse_chat_request(
        {"messages": _historical_tool_messages('{"zero":0.0}')},
        require_model=False,
        allow_managed_selectors=True,
    )

    detached = detached_template_messages(normalized.messages)
    value = next(iter(detached[0]["tool_calls"][0]["function"]["arguments"].values()))
    assert value == 0
    assert type(value) is int


def test_integer_negative_zero_history_keeps_its_exact_lexeme() -> None:
    normalized = parse_chat_request(
        {"messages": _historical_tool_messages('{"zero":-0}')},
        require_model=False,
        allow_managed_selectors=True,
    )

    detached = detached_template_messages(normalized.messages)
    value = next(iter(detached[0]["tool_calls"][0]["function"]["arguments"].values()))
    assert value == "-0"


@pytest.mark.parametrize("argument", ['{"text":"\\ud800"}', '{"text":"\\udc00"}'])
def test_unpaired_surrogate_in_tool_history_is_a_request_error(argument: str) -> None:
    messages = _historical_tool_messages(argument)

    with pytest.raises(OpenAIRequestError, match="arguments must encode a JSON object"):
        parse_chat_request(
            {"messages": messages},
            require_model=False,
            allow_managed_selectors=True,
        )


def test_valid_non_bmp_surrogate_pair_in_tool_history_is_accepted() -> None:
    messages = _historical_tool_messages('{"text":"\\ud83d\\ude00"}')

    request = parse_chat_request(
        {"messages": messages},
        require_model=False,
        allow_managed_selectors=True,
    )

    assert request.messages == messages


def test_authorized_checkpoint_requires_one_explicit_verified_target() -> None:
    run_id = "run-1"
    checkpoint_id = f"{run_id}/step-7"
    deployment = {
        "state": "ready",
        "checkpoint_id": checkpoint_id,
        "openai_model": checkpoint_id,
    }

    assert (
        _authorized_chat_checkpoint(
            run_id,
            deployment,
            checkpoint_id,
            {checkpoint_id},
        )
        == checkpoint_id
    )
    with pytest.raises(Exception, match="checkpoint_id must"):
        _authorized_chat_checkpoint(run_id, deployment, None, {checkpoint_id})

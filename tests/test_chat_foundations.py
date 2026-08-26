from __future__ import annotations

import json
import subprocess
import sys

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
)
from flash.serve.request.openai import DEFAULT_MAX_TOKENS, OpenAIRequestError, parse_chat_request
from flash.serve.request.streaming import _complete_sse_frames
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

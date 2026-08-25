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
from flash.serve._chat_transport import OpenAIStreamResponse
from flash.serve.openai_request import DEFAULT_MAX_TOKENS, OpenAIRequestError, parse_chat_request
from flash.serve.provenance import (
    ImmutableProvenance,
    decode_flash_body,
    decode_flash_headers,
    decode_freesolo_body,
    decode_freesolo_headers,
)
from flash.server.routes.serving_revisions import _authorized_chat_revision


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
    assert request.chat_template_kwargs == {"custom": 1}

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
                "import sys; import flash.serve.thinking; "
                "assert 'flash.serve.deploy' not in sys.modules"
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
    revision = "run-1@step-7." + "a" * 40
    expected = ImmutableProvenance.from_adapter_revision(revision)
    freesolo = expected.freesolo_body()
    flash = {
        "adapter_revision": expected.adapter_revision,
        "checkpoint": expected.checkpoint,
        "source_revision": expected.hf_revision,
        "deployment_id": "kept",
    }
    freesolo_headers = expected.freesolo_headers()
    flash_headers = {
        "X-Flash-Adapter-Revision": expected.adapter_revision,
        "X-Flash-Checkpoint": expected.checkpoint,
        "X-Flash-Source-Revision": expected.hf_revision,
    }

    assert decode_freesolo_body(freesolo) == expected
    assert decode_flash_body(flash) == expected
    assert decode_freesolo_headers(freesolo_headers) == expected
    assert decode_flash_headers(flash_headers) == expected


def test_authorized_revision_resolver_prefers_ready_revision_for_ambiguous_step() -> None:
    run_id = "run-1"
    older = f"{run_id}@step-7." + "a" * 40
    ready = f"{run_id}@step-7." + "b" * 40
    deployment = {"state": "ready", "adapter_revision": ready}

    assert (
        _authorized_chat_revision(
            run_id,
            deployment,
            None,
            7,
            {older, ready},
        )
        == ready
    )
    assert (
        _authorized_chat_revision(
            run_id,
            deployment,
            None,
            None,
            {older, ready},
        )
        == ready
    )

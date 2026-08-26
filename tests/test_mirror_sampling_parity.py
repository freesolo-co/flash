"""focused continuous-mirror sampling parity regressions."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from flash.serve.app.chat_stream import stream_chat_body
from flash.serve.app.openai import nonstream_response
from flash.serve.deployment import deploy
from flash.serve.request.openai import (
    OpenAIRequestError,
    parse_chat_request,
    reject_thinking_logprobs,
)
from flash.serve.runtime.errors import RuntimeNotReadyError
from flash.serve.runtime.sampling import complete_indexed_outputs, normalize_token_logprobs
from flash.serve.runtime.tool_calls import ParsedToolCall
from flash.serve.runtime.types import (
    GenerationChoice,
    GenerationResult,
    StreamChoiceFinished,
    StreamDelta,
    StreamFinished,
    StreamReady,
)
from flash.serving.src.accounting.usage_facts import usage_facts


def _payload(**updates):
    return {
        "model": "run@final." + "a" * 40,
        "messages": [{"role": "user", "content": "hello"}],
        **updates,
    }


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", True),
        ("n", 0),
        ("n", 5),
        ("seed", -1),
        ("seed", 2**63),
        ("frequency_penalty", float("nan")),
        ("frequency_penalty", 10**400),
        ("frequency_penalty", -2.01),
        ("presence_penalty", 10**400),
        ("presence_penalty", 2.01),
        ("logprobs", 1),
        ("top_logprobs", True),
        ("top_logprobs", 21),
    ],
)
def test_canonical_sampling_validation_is_strict(field, value):
    with pytest.raises(OpenAIRequestError):
        parse_chat_request(
            _payload(**{field: value}),
            require_model=True,
            allow_managed_selectors=False,
        )


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_canonical_chat_template_thinking_override_requires_boolean(value):
    with pytest.raises(OpenAIRequestError, match="enable_thinking must be a boolean"):
        parse_chat_request(
            _payload(chat_template_kwargs={"enable_thinking": value}),
            require_model=True,
            allow_managed_selectors=False,
        )


def test_canonical_sampling_controls_and_cross_field_rules():
    request = parse_chat_request(
        _payload(
            temperature=0.5,
            n=4,
            seed=-(2**63),
            frequency_penalty=-2,
            presence_penalty=2,
            logprobs=True,
            top_logprobs=20,
        ),
        require_model=True,
        allow_managed_selectors=False,
    )
    assert (
        request.n,
        request.seed,
        request.frequency_penalty,
        request.presence_penalty,
        request.logprobs,
        request.top_logprobs,
    ) == (4, -(2**63), -2.0, 2.0, True, 20)
    with pytest.raises(OpenAIRequestError, match="temperature greater than zero"):
        parse_chat_request(
            _payload(n=2, temperature=0),
            require_model=True,
            allow_managed_selectors=False,
        )
    with pytest.raises(OpenAIRequestError, match="requires logprobs=true"):
        parse_chat_request(
            _payload(top_logprobs=1),
            require_model=True,
            allow_managed_selectors=False,
        )


def test_thinking_logprobs_guard_is_post_resolution_policy():
    reject_thinking_logprobs(thinking=False, logprobs=True)
    with pytest.raises(OpenAIRequestError, match="thinking-enabled"):
        reject_thinking_logprobs(thinking=True, logprobs=True)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
@pytest.mark.parametrize(
    "stop", ["tool_call", "weather", "prefix</tool_call>", "answer<", ">suffix"]
)
def test_canonical_active_tools_reject_stop_sequences_that_overlap_qwen_markers(stream, stop):
    with pytest.raises(OpenAIRequestError, match=r"grammar markers.*tool_choice='auto'"):
        parse_chat_request(
            _payload(tools=_tools(), stop=stop, stream=stream),
            require_model=True,
            allow_managed_selectors=False,
        )


def test_canonical_tool_choice_none_allows_tool_marker_stop_sequences():
    request = parse_chat_request(
        _payload(tools=_tools(), tool_choice="none", stop="</tool_call>"),
        require_model=True,
        allow_managed_selectors=False,
    )
    assert request.stop == ("</tool_call>",)


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ({"logprobs": True, "top_logprobs": 1}, (True, None)),
        ({"response_format": {"type": "json_object"}}, (False, {"json_object": True})),
        (
            {"structured_outputs": {"choice": ["sunny", "rainy"]}},
            (False, {"choice": ["sunny", "rainy"]}),
        ),
    ],
    ids=["logprobs", "response-format", "structured-outputs"],
)
def test_canonical_tool_choice_none_allows_non_tool_features(feature, expected):
    request = parse_chat_request(
        _payload(tools=_tools(), tool_choice="none", **feature),
        require_model=True,
        allow_managed_selectors=False,
    )

    assert (request.logprobs, request.structured_outputs) == expected


@pytest.mark.parametrize(
    "outputs",
    [
        [SimpleNamespace(index=0), SimpleNamespace(index=0)],
        [SimpleNamespace(index=0)],
        [SimpleNamespace(index=-1), SimpleNamespace(index=1)],
        [SimpleNamespace(index=0), SimpleNamespace(index=2)],
    ],
)
def test_choice_normalization_rejects_malformed_complete_sets(outputs):
    with pytest.raises(RuntimeNotReadyError, match=r"choice index|duplicate|incomplete"):
        complete_indexed_outputs(SimpleNamespace(outputs=outputs), n=2)


@pytest.mark.parametrize(
    ("top_logprobs", "candidates", "expected_tokens"),
    [
        (2, [(12, -1.2, "å"), (10, -0.1, "a"), (13, -2.0, "c")], ["å", "a"]),
        (2, [(12, -1.2, "å"), (13, -2.0, "c"), (10, -0.1, "a")], ["å", "c"]),
        (0, [(10, -0.1, "a"), (12, -1.2, "å")], []),
    ],
)
def test_vllm_logprobs_match_openai_candidate_slicing(top_logprobs, candidates, expected_tokens):
    def candidate(value, token):
        return SimpleNamespace(logprob=value, decoded_token=token)

    normalized = normalize_token_logprobs(
        [10],
        [{token_id: candidate(value, token) for token_id, value, token in candidates}],
        top_logprobs=top_logprobs,
    )
    assert normalized is not None
    assert normalized[0]["token"] == "a"
    assert [record["token"] for record in normalized[0]["top_logprobs"]] == expected_tokens
    json.dumps(normalized, allow_nan=False)


def test_buffered_response_preserves_indexed_choices_and_aggregate_usage():
    choices = tuple(
        GenerationChoice(
            index=index, text=f"answer-{index}", finish_reason="stop", token_ids=(index,)
        )
        for index in range(4)
    )
    result = GenerationResult(
        request_id="request",
        adapter_id="adapter",
        incarnation="incarnation",
        choices=choices,
        prompt_tokens=5,
        completion_tokens=4,
        cached_tokens=2,
        cached_tokens_reported=True,
        thinking=False,
    )
    resolved = SimpleNamespace(
        requested_model="adapter",
        adapter=SimpleNamespace(
            adapter_revision="adapter",
            checkpoint="run",
            source_revision="a" * 40,
            source_subfolder=None,
            aggregate_sha256="incarnation",
        ),
    )
    manifest = SimpleNamespace(
        deployment_id="deployment",
        spec_id="spec",
        manifest_id="manifest",
        expected_oci_digest="sha256:image",
        logical_base_model="base",
        logical_base_revision="b" * 40,
        engine=SimpleNamespace(
            engine_id="engine",
            served_model="served",
            model_revision="c" * 40,
            tokenizer_model="tokenizer",
            tokenizer_revision="d" * 40,
        ),
    )
    response = nonstream_response(result, manifest, resolved)
    assert [choice["index"] for choice in response["choices"]] == [0, 1, 2, 3]
    assert response["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 4,
        "total_tokens": 9,
        "prompt_tokens_details": {"cached_tokens": 2},
    }


def test_packaged_sse_interleaves_choices_with_independent_reasoning():
    ready = StreamReady(
        request_id="request",
        runtime_id="runtime",
        adapter_id="adapter",
        incarnation="incarnation",
        thinking=True,
    )
    resolved = SimpleNamespace(
        requested_model="adapter",
        adapter=SimpleNamespace(adapter_revision="adapter", aggregate_sha256="incarnation"),
    )

    async def events():
        yield StreamDelta(index=0, text="reason-0</think>answer-0")
        yield StreamDelta(index=1, text="reason-1")
        yield StreamChoiceFinished(index=0, text="", finish_reason="stop", token_ids=(1,))
        yield StreamDelta(index=1, text="</think>answer-1")
        yield StreamChoiceFinished(index=1, text="", finish_reason="length", token_ids=(2, 3))
        yield StreamFinished(
            request_id="request",
            runtime_id="runtime",
            adapter_id="adapter",
            incarnation="incarnation",
            choices=(),
            prompt_tokens=4,
            completion_tokens=3,
            cached_tokens=1,
            cached_tokens_reported=True,
            thinking=True,
        )

    async def collect():
        return [
            chunk
            async for chunk in stream_chat_body(
                events(),
                ready,
                resolved,
                {},
                choice_count=2,
                include_usage=True,
            )
        ]

    chunks = asyncio.run(collect())
    assert chunks[-1] == b"data: [DONE]\n\n"
    payloads = [json.loads(chunk[6:-2]) for chunk in chunks[:-1]]
    terminals = [
        choice
        for payload in payloads
        for choice in payload.get("choices", [])
        if choice.get("finish_reason") is not None
    ]
    assert [(choice["index"], choice["finish_reason"]) for choice in terminals] == [
        (0, "stop"),
        (1, "length"),
    ]
    usage = next(payload["usage"] for payload in payloads if "usage" in payload)
    assert usage["prompt_tokens"] == 4
    assert usage["completion_tokens"] == 3


def test_packaged_sse_emits_empty_content_with_logprobs():
    ready = StreamReady(
        request_id="request",
        runtime_id="runtime",
        adapter_id="adapter",
        incarnation="incarnation",
        thinking=False,
    )
    resolved = SimpleNamespace(
        requested_model="adapter",
        adapter=SimpleNamespace(adapter_revision="adapter", aggregate_sha256="incarnation"),
    )
    token_logprobs = [{"token": "a", "logprob": -0.1, "bytes": [97], "top_logprobs": []}]

    async def events():
        yield StreamDelta(index=0, text="", logprobs=token_logprobs)
        yield StreamChoiceFinished(index=0, text="", finish_reason="stop", token_ids=(1,))
        yield StreamFinished(
            request_id="request",
            runtime_id="runtime",
            adapter_id="adapter",
            incarnation="incarnation",
            choices=(),
            prompt_tokens=1,
            completion_tokens=1,
            cached_tokens=0,
            cached_tokens_reported=True,
            thinking=False,
        )

    async def collect():
        return [
            chunk
            async for chunk in stream_chat_body(
                events(), ready, resolved, {}, choice_count=1, include_usage=False
            )
        ]

    payloads = [json.loads(chunk[6:-2]) for chunk in asyncio.run(collect())[:-1]]
    choice = next(
        payload["choices"][0]
        for payload in payloads
        if payload["choices"][0].get("logprobs") is not None
    )
    assert choice["delta"] == {"content": ""}
    assert choice["logprobs"] == {"content": token_logprobs}


def _response_context():
    resolved = SimpleNamespace(
        requested_model="adapter",
        adapter=SimpleNamespace(
            adapter_revision="adapter",
            checkpoint="run",
            source_revision="a" * 40,
            source_subfolder=None,
            aggregate_sha256="incarnation",
        ),
    )
    manifest = SimpleNamespace(
        deployment_id="deployment",
        spec_id="spec",
        manifest_id="manifest",
        expected_oci_digest="sha256:image",
        logical_base_model="base",
        logical_base_revision="b" * 40,
        engine=SimpleNamespace(
            engine_id="engine",
            served_model="served",
            model_revision="c" * 40,
            tokenizer_model="tokenizer",
            tokenizer_revision="d" * 40,
        ),
    )
    return manifest, resolved


def test_packaged_buffered_response_formats_structured_tool_calls():
    choice = GenerationChoice(
        index=0,
        text="",
        finish_reason="tool_calls",
        token_ids=(1, 2),
        tool_calls=(ParsedToolCall("call_1", "weather", '{"city":"Paris"}'),),
    )
    result = GenerationResult(
        request_id="request",
        adapter_id="revision",
        incarnation="digest",
        choices=(choice,),
        prompt_tokens=3,
        completion_tokens=2,
        cached_tokens=0,
        cached_tokens_reported=True,
        thinking=False,
    )
    manifest, resolved = _response_context()
    response = nonstream_response(result, manifest, resolved)
    wire_choice = response["choices"][0]
    assert wire_choice["message"]["content"] is None
    assert wire_choice["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
        }
    ]
    assert wire_choice["finish_reason"] == "tool_calls"


def test_packaged_sse_emits_complete_structured_tool_delta() -> None:
    ready = StreamReady(
        request_id="request",
        runtime_id="runtime",
        adapter_id="adapter",
        incarnation="incarnation",
        thinking=False,
    )
    _, resolved = _response_context()
    call = ParsedToolCall("call_1", "weather", '{"city":"Paris"}')

    async def events():
        yield StreamDelta(index=0, text="", tool_calls=(call,))
        yield StreamChoiceFinished(
            index=0,
            text="raw tags",
            finish_reason="tool_calls",
            token_ids=(1,),
            tool_calls=(call,),
        )
        yield StreamFinished(
            request_id="request",
            runtime_id="runtime",
            adapter_id="adapter",
            incarnation="incarnation",
            choices=(
                GenerationChoice(
                    index=0,
                    text="raw tags",
                    finish_reason="tool_calls",
                    token_ids=(1,),
                    tool_calls=(call,),
                ),
            ),
            prompt_tokens=2,
            completion_tokens=1,
            cached_tokens=0,
            cached_tokens_reported=True,
            thinking=False,
        )

    async def collect():
        return [
            chunk
            async for chunk in stream_chat_body(
                events(), ready, resolved, {}, choice_count=1, include_usage=False
            )
        ]

    payloads = [json.loads(chunk[6:-2]) for chunk in asyncio.run(collect())[:-1]]
    tool_delta = next(
        payload["choices"][0]["delta"]["tool_calls"]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    )
    assert tool_delta[0]["index"] == 0
    assert tool_delta[0]["id"] == "call_1"
    assert all("<tool_call>" not in chunk.decode() for chunk in asyncio.run(collect()))


def test_usage_facts_keeps_aggregate_completion_count():
    facts = usage_facts(
        {
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [9],
            "prompt_tokens": 2,
            "completion_tokens": 7,
            "cached_tokens": 0,
            "cached_tokens_reported": True,
            "thinking": False,
        }
    )
    assert facts.completion_tokens == 7


def test_text_only_stream_rejects_multi_choice_before_transport(monkeypatch):
    monkeypatch.setattr(
        deploy.transport,
        "request_chat_stream",
        lambda *_args, **_kwargs: pytest.fail("transport must not open"),
    )
    with pytest.raises(ValueError, match="requires n=1"):
        deploy.chat_stream("run", [{"role": "user", "content": "hi"}], n=2)
    with pytest.raises(ValueError, match="does not expose logprobs"):
        deploy.chat_stream("run", [{"role": "user", "content": "hi"}], logprobs=True)
    with pytest.raises(ValueError, match="does not support tools"):
        deploy.chat_stream("run", [{"role": "user", "content": "hi"}], tools=[])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n": True}, "n must be an integer"),
        ({"n": 1.0}, "n must be an integer"),
        ({"logprobs": 0}, "logprobs must be a boolean"),
        ({"top_logprobs": False}, "top_logprobs must be an integer"),
    ],
)
def test_text_only_stream_rejects_wrong_control_types_before_transport(
    monkeypatch, kwargs, message
):
    monkeypatch.setattr(
        deploy.transport,
        "request_chat_stream",
        lambda *_args, **_kwargs: pytest.fail("transport must not open"),
    )
    with pytest.raises(ValueError, match=message):
        deploy.chat_stream("run", [{"role": "user", "content": "hi"}], **kwargs)

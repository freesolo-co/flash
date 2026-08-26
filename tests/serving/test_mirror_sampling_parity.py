"""focused hosted sampling parity regressions."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from flash.serving.src.engine.lora_engine import _LoraEngineImpl
from flash.serving.src.engine.model_config import reasoning_parser_for, tool_parser_for
from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.io.openai_request import OpenAIGenerateRequest
from flash.serving.src.io.openai_stream import _produce_openai_chat_stream, openai_chat_stream
from flash.serving.src.io.schemas import AdapterRecord, GenerateRequest
from flash.serving.src.store.registry import AdapterRegistry

QWEN = "Qwen/Qwen3.5-9B"


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
        del messages, kwargs
        return [1, 2, 3]


class _Logprob:
    def __init__(self, value: float, token: str) -> None:
        self.logprob = value
        self.decoded_token = token


class _BufferedChoiceEngine:
    def __init__(self) -> None:
        self.sampling_params: Any = None

    async def generate(self, _prompt: Any, sampling: Any, _request_id: str, **_kwargs: Any):
        self.sampling_params = sampling
        outputs = []
        for index in range(sampling.n):
            token_id = 10 + index
            token = chr(ord("a") + index)
            outputs.append(
                SimpleNamespace(
                    index=index,
                    text=f"answer-{index}",
                    finish_reason="stop" if index != 3 else "length",
                    token_ids=[token_id],
                    logprobs=[
                        {
                            token_id: _Logprob(-0.1 - index, token),
                            100 + index: _Logprob(-1.0, f"alt-{index}"),
                        }
                    ],
                )
            )
        yield SimpleNamespace(
            outputs=outputs,
            prompt_token_ids=[1, 2],
            num_cached_tokens=1,
        )


class _ToolChoiceEngine:
    async def generate(self, _prompt: Any, _sampling: Any, _request_id: str, **_kwargs: Any):
        yield SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    text=(
                        "<tool_call>\n<function=weather>\n"
                        "<parameter=city>\nParis\n</parameter>\n"
                        "</function>\n</tool_call>  \n\t"
                    ),
                    finish_reason="stop",
                    token_ids=[10, 11],
                    logprobs=None,
                )
            ],
            prompt_token_ids=[1, 2],
            num_cached_tokens=0,
        )


class _StreamingToolChoiceEngine:
    async def generate(self, _prompt: Any, _sampling: Any, _request_id: str, **_kwargs: Any):
        for text, token_id, finish_reason in (
            ("<tool_call>\n<function=weather>\n", 10, None),
            (
                "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>  \n",
                11,
                "stop",
            ),
        ):
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        index=0,
                        text=text,
                        finish_reason=finish_reason,
                        token_ids=[token_id],
                        logprobs=None,
                    )
                ],
                prompt_token_ids=[1, 2],
                num_cached_tokens=0,
            )


class _InterleavedChoiceEngine:
    def __init__(self) -> None:
        self.sampling_params: Any = None

    async def generate(self, _prompt: Any, sampling: Any, _request_id: str, **_kwargs: Any):
        self.sampling_params = sampling
        for outputs in (
            [self._output(1, "b1", 21, None)],
            [self._output(0, "a0", 20, None)],
            [self._output(1, "b2", 22, "length")],
            [self._output(0, "a1", 23, "stop")],
        ):
            yield SimpleNamespace(
                outputs=outputs,
                prompt_token_ids=[1, 2, 3],
                num_cached_tokens=1,
            )

    @staticmethod
    def _output(index: int, text: str, token_id: int, finish_reason: str | None) -> Any:
        return SimpleNamespace(
            index=index,
            text=text,
            finish_reason=finish_reason,
            token_ids=[token_id],
            logprobs=[{token_id: _Logprob(-0.25, text)}],
        )


def _tool_payload() -> list[dict[str, Any]]:
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


def _historical_tool_messages(argument: str) -> list[dict[str, Any]]:
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


def _engine(
    vllm_engine: Any,
    *,
    thinking: bool = False,
    structured_outputs: dict[str, Any] | None = None,
) -> _LoraEngineImpl:
    engine = object.__new__(_LoraEngineImpl)
    engine.base_model = QWEN
    engine.reasoning_parser = reasoning_parser_for(QWEN)
    engine.tool_parser = tool_parser_for(QWEN)
    engine.tokenizer = _Tokenizer()
    engine.registry = AdapterRegistry()
    engine.registry.hydrate(
        [
            AdapterRecord(
                adapter_id="adapter",
                repo_id=QWEN,
                base_model=QWEN,
                serve_base_model=True,
                thinking=thinking,
                status="ready",
                structured_outputs=structured_outputs,
            )
        ]
    )
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()
    engine.engine = vllm_engine
    return engine


def test_hosted_base_model_engine_honors_boolean_thinking_override() -> None:
    engine = _engine(_BufferedChoiceEngine(), thinking=True)
    result = asyncio.run(
        engine._generate(
            {
                "adapter_id": "adapter",
                "messages": [{"role": "user", "content": "hi"}],
                "logprobs": True,
                "top_logprobs": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
    )
    assert result["thinking"] is False


def test_hosted_buffered_generation_parses_qualified_tool_calls() -> None:
    result = asyncio.run(
        _engine(_ToolChoiceEngine())._generate(
            {
                "adapter_id": "adapter",
                "messages": [{"role": "user", "content": "weather"}],
                "tools": _tool_payload(),
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            }
        )
    )
    choice = result["choices"][0]
    assert choice["text"] == ""
    assert choice["finish_reason"] == "tool_calls"
    assert choice["tool_calls"][0]["function"] == {
        "name": "weather",
        "arguments": '{"city":"Paris"}',
    }
    assert result["completion_tokens"] == 2


def test_hosted_stream_reports_hidden_tool_usage_before_structured_delta() -> None:
    engine = _engine(_StreamingToolChoiceEngine())

    async def collect() -> list[dict[str, Any]]:
        return [
            event
            async for event in engine._stream_generate(
                {
                    "adapter_id": "adapter",
                    "messages": [{"role": "user", "content": "weather"}],
                    "tools": _tool_payload(),
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                }
            )
        ]

    events = asyncio.run(collect())
    progress = [event for event in events if event["type"] == "usage_progress"]
    tool_delta = next(event for event in events if event.get("tool_calls"))
    assert [event["completion_tokens"] for event in progress] == [1, 2]
    assert tool_delta["completion_tokens"] == 2
    assert tool_delta["tool_calls"][0]["function"]["arguments"] == '{"city":"Paris"}'
    assert events[-1]["completion_tokens"] == 2
    assert events[-1]["choices"][0]["text"] == ""
    assert events[-1]["choices"][0]["tool_calls"] == tool_delta["tool_calls"]


@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
def test_hosted_tools_reject_effective_persisted_structured_default_after_resolution(
    streaming: bool,
) -> None:
    vllm_engine = _BufferedChoiceEngine()
    engine = _engine(vllm_engine, structured_outputs={"choice": ["sunny", "rainy"]})
    resolved: list[str] = []
    original_resolve = engine._lora_request

    async def resolve(adapter_id: str, record_dict: dict[str, Any] | None = None):
        result = await original_resolve(adapter_id, record_dict)
        resolved.append(result[1].adapter_id)
        return result

    engine._lora_request = resolve
    payload = {
        "adapter_id": "adapter",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": _tool_payload(),
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }

    async def exercise() -> None:
        if streaming:
            async for _event in engine._stream_generate(payload):
                pass
        else:
            await engine._generate(payload)

    with pytest.raises(ValueError, match=r"tools cannot be combined.*structured outputs"):
        asyncio.run(exercise())

    assert resolved == ["adapter"]
    assert vllm_engine.sampling_params is None


@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
def test_hosted_tool_choice_none_allows_effective_structured_default(streaming: bool) -> None:
    vllm_engine = _BufferedChoiceEngine()
    engine = _engine(vllm_engine, structured_outputs={"choice": ["sunny", "rainy"]})
    payload = {
        "adapter_id": "adapter",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": _tool_payload(),
        "tool_choice": "none",
        "parallel_tool_calls": True,
    }

    async def exercise() -> None:
        if streaming:
            async for _event in engine._stream_generate(payload):
                pass
        else:
            await engine._generate(payload)

    asyncio.run(exercise())

    assert vllm_engine.sampling_params.structured_outputs is not None


def _buffered_result(
    choice_count: int, *, top_logprobs: int = 3
) -> tuple[_BufferedChoiceEngine, dict[str, Any]]:
    vllm_engine = _BufferedChoiceEngine()
    engine = _engine(vllm_engine)
    result = asyncio.run(
        engine._generate(
            {
                "adapter_id": "adapter",
                "prompt": "hi",
                "temperature": 0.7,
                "n": choice_count,
                "seed": 42,
                "frequency_penalty": -0.5,
                "presence_penalty": 0.75,
                "logprobs": True,
                "top_logprobs": top_logprobs,
            }
        )
    )
    return vllm_engine, result


@pytest.mark.parametrize("choice_count", [1, 2, 4])
def test_hosted_buffered_sampling_params_and_choices_are_json_safe(choice_count: int) -> None:
    vllm_engine, result = _buffered_result(choice_count)
    sampling = vllm_engine.sampling_params
    assert (
        sampling.temperature,
        sampling.n,
        sampling.seed,
        sampling.frequency_penalty,
        sampling.presence_penalty,
        sampling.logprobs,
    ) == (0.7, choice_count, 42, -0.5, 0.75, 3)
    assert [choice["index"] for choice in result["choices"]] == list(range(choice_count))
    assert result["prompt_tokens"] == 2
    assert result["completion_tokens"] == choice_count
    assert result["cached_tokens"] == 1
    assert result["choices"][0]["logprobs"][0]["token"] == "a"
    assert [record["token"] for record in result["choices"][0]["logprobs"][0]["top_logprobs"]] == [
        "a",
        "alt-0",
    ]
    json.dumps(result, allow_nan=False)


def test_hosted_buffered_top_logprobs_zero_keeps_selected_record_only() -> None:
    _, result = _buffered_result(1, top_logprobs=0)
    token = result["choices"][0]["logprobs"][0]
    assert token["token"] == "a"
    assert token["top_logprobs"] == []


def test_hosted_engine_streams_interleaved_indexes_and_aggregate_usage() -> None:
    vllm_engine = _InterleavedChoiceEngine()
    engine = _engine(vllm_engine)

    async def collect() -> list[dict[str, Any]]:
        return [
            event
            async for event in engine._stream_generate(
                {
                    "adapter_id": "adapter",
                    "prompt": "hi",
                    "temperature": 0.5,
                    "n": 2,
                    "logprobs": True,
                    "top_logprobs": 2,
                }
            )
        ]

    events = asyncio.run(collect())
    deltas = [event for event in events if event["type"] == "delta"]
    terminals = [event for event in events if event["type"] == "choice_finished"]
    final = events[-1]
    assert [event["index"] for event in deltas] == [1, 0, 1, 0]
    assert [(event["index"], event["finish_reason"]) for event in terminals] == [
        (1, "length"),
        (0, "stop"),
    ]
    assert final["completion_tokens"] == 4
    assert final["prompt_tokens"] == 3
    assert final["choices"] == [
        {"index": 0, "text": "a0a1", "token_ids": [20, 23], "finish_reason": "stop"},
        {"index": 1, "text": "b1b2", "token_ids": [21, 22], "finish_reason": "length"},
    ]
    assert vllm_engine.sampling_params.n == 2
    assert vllm_engine.sampling_params.logprobs == 2
    json.dumps(events, allow_nan=False)


class _UsageSession:
    def __init__(self) -> None:
        self.finalized: list[dict[str, Any]] = []
        self.failed: list[tuple[dict[str, Any], str]] = []

    async def finalize(self, event: dict[str, Any]) -> None:
        self.finalized.append(event)

    async def fail(self, event: dict[str, Any], reason: str) -> None:
        self.failed.append((event, reason))

    async def capture(self, _event: dict[str, Any]) -> None:
        return None

    def relinquish(self) -> None:
        return None


@pytest.mark.parametrize(
    "argument",
    [
        '{"value":' + "[" * 600 + "0" + "]" * 600 + "}",
        json.dumps({"values": [0] * 511}),
    ],
    ids=["depth", "aggregate-nodes"],
)
def test_hosted_tool_free_history_rejects_excessive_argument_complexity(argument: str) -> None:
    messages = _historical_tool_messages(argument)
    original = json.loads(json.dumps(messages))

    with pytest.raises(ValidationError, match="tool argument complexity"):
        OpenAIGenerateRequest.model_validate({"adapter_id": "adapter", "messages": messages})

    assert messages == original


@pytest.mark.parametrize(
    "argument",
    [
        '{"value":' + "[" * 7 + "0" + "]" * 7 + "}",
        json.dumps({"values": [0] * 510}),
    ],
    ids=["depth", "aggregate-nodes"],
)
def test_hosted_tool_free_history_accepts_argument_complexity_boundary(argument: str) -> None:
    messages = _historical_tool_messages(argument)
    original = json.loads(json.dumps(messages))

    request = OpenAIGenerateRequest.model_validate({"adapter_id": "adapter", "messages": messages})

    assert request.messages == original
    assert messages == original


@pytest.mark.parametrize("argument", ['{"text":"\\ud800"}', '{"text":"\\udc00"}'])
def test_hosted_tool_history_rejects_unpaired_surrogates(argument: str) -> None:
    messages = _historical_tool_messages(argument)

    with pytest.raises(ValidationError, match="arguments must encode a JSON object"):
        OpenAIGenerateRequest.model_validate({"adapter_id": "adapter", "messages": messages})


def test_hosted_tool_history_accepts_valid_non_bmp_pair_and_serializes() -> None:
    messages = _historical_tool_messages('{"text":"\\ud83d\\ude00"}')

    request = OpenAIGenerateRequest.model_validate({"adapter_id": "adapter", "messages": messages})

    request.model_dump_json()


def test_hosted_message_validation_preserves_non_tool_and_active_tool_requests() -> None:
    plain_messages = [{"role": "user", "content": "weather"}]
    plain_request = OpenAIGenerateRequest.model_validate(
        {"adapter_id": "adapter", "messages": plain_messages}
    )
    tool_request = OpenAIGenerateRequest.model_validate(
        {
            "adapter_id": "adapter",
            "messages": plain_messages,
            "tools": _tool_payload(),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
    )

    assert plain_request.messages == plain_messages
    assert tool_request.messages == plain_messages
    assert tool_request.tools == _tool_payload()


def test_hosted_private_tool_envelope_rejects_active_stop_marker_collision() -> None:
    with pytest.raises(ValueError, match=r"grammar markers.*tool_choice='auto'"):
        OpenAIGenerateRequest.model_validate(
            {
                "adapter_id": "adapter",
                "messages": [{"role": "user", "content": "weather"}],
                "tools": _tool_payload(),
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "stop": "</tool_call>",
            }
        )
    request = OpenAIGenerateRequest.model_validate(
        {
            "adapter_id": "adapter",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": _tool_payload(),
            "tool_choice": "none",
            "parallel_tool_calls": True,
            "stop": "</tool_call>",
        }
    )
    assert request.stop == "</tool_call>"


@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
def test_hosted_generation_rejects_active_stop_marker_collision_before_dispatch(
    streaming: bool,
) -> None:
    vllm_engine = _BufferedChoiceEngine()
    engine = _engine(vllm_engine)
    payload = {
        "adapter_id": "adapter",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": _tool_payload(),
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stop": "<parameter=city>",
    }

    async def exercise() -> None:
        if streaming:
            async for _event in engine._stream_generate(payload):
                pass
        else:
            await engine._generate(payload)

    with pytest.raises(ValueError, match=r"grammar markers.*tool_choice='auto'"):
        asyncio.run(exercise())
    assert vllm_engine.sampling_params is None


def test_hosted_private_tool_envelope_requires_text_chat_messages() -> None:
    controls = {
        "adapter_id": "adapter",
        "tools": _tool_payload(),
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    with pytest.raises(ValueError, match="tools require chat messages"):
        OpenAIGenerateRequest.model_validate({**controls, "prompt": "weather"})
    with pytest.raises(ValueError, match="image messages"):
        OpenAIGenerateRequest.model_validate(
            {
                **controls,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": "data:image/png;base64,AA=="}],
                    }
                ],
            }
        )


def test_raw_generate_schema_forbids_openai_sampling_fields() -> None:
    base = {"adapter_id": "adapter", "prompt": "hi"}
    for field, value in (
        ("n", 2),
        ("seed", 7),
        ("frequency_penalty", 0.5),
        ("presence_penalty", -0.5),
        ("logprobs", True),
        ("top_logprobs", 2),
    ):
        with pytest.raises(ValueError, match="extra_forbidden"):
            GenerateRequest.model_validate({**base, field: value})

    request = OpenAIGenerateRequest.model_validate(
        {
            **base,
            "temperature": 0.5,
            "n": 2,
            "seed": 7,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
            "logprobs": True,
            "top_logprobs": 2,
        }
    )
    assert (
        request.n,
        request.seed,
        request.frequency_penalty,
        request.presence_penalty,
        request.logprobs,
        request.top_logprobs,
    ) == (2, 7, 0.5, -0.5, True, 2)


def _record() -> AdapterRecord:
    return AdapterRecord(
        adapter_id="adapter",
        repo_id=QWEN,
        base_model=QWEN,
        serve_base_model=True,
        thinking=True,
        status="ready",
    )


def test_hosted_sse_emits_empty_content_with_logprobs() -> None:
    session = _UsageSession()
    record = _record()
    token_logprobs = [{"token": "a", "logprob": -0.1, "bytes": [97], "top_logprobs": []}]

    async def events():
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0}
        yield {"type": "ready", "thinking": False, **usage}
        yield {
            "type": "delta",
            "index": 0,
            "text": "",
            "logprobs": token_logprobs,
            **usage,
        }
        yield {"type": "choice_finished", "index": 0, "finish_reason": "stop", **usage}
        yield {"type": "final", **usage}

    async def collect() -> list[bytes]:
        return [
            chunk
            async for chunk in openai_chat_stream(
                AdapterRouter([record]),
                record=record,
                events=events(),
                adapter_id="adapter",
                completion_id="chatcmpl-test",
                created=123,
                include_usage=False,
                usage_session=session,  # type: ignore[arg-type]
                thinking=False,
            )
        ]

    payloads = [json.loads(chunk[6:-2]) for chunk in asyncio.run(collect())[:-1]]
    choice = next(
        payload["choices"][0] for payload in payloads if "logprobs" in payload["choices"][0]
    )
    assert choice["delta"] == {"content": ""}
    assert choice["logprobs"] == {"content": token_logprobs}


def test_hosted_sse_uses_hidden_usage_progress_without_serializing_it() -> None:
    session = _UsageSession()
    record = _record()

    async def events():
        yield {"type": "ready", "thinking": False, "prompt_tokens": 2, "completion_tokens": 0}
        yield {"type": "usage_progress", "prompt_tokens": 2, "completion_tokens": 2}
        yield {"type": "error", "message": "engine failed", "code": 502}

    async def collect() -> list[bytes]:
        return [
            chunk
            async for chunk in openai_chat_stream(
                AdapterRouter([record]),
                record=record,
                events=events(),
                adapter_id="adapter",
                completion_id="chatcmpl-test",
                created=123,
                include_usage=False,
                usage_session=session,  # type: ignore[arg-type]
                thinking=False,
            )
        ]

    chunks = asyncio.run(collect())
    assert session.failed[0][0]["completion_tokens"] == 2
    assert session.failed[0][1] == "engine_failed"
    assert all(b"usage_progress" not in chunk for chunk in chunks)


def test_hosted_sse_has_independent_reasoning_and_post_settlement_terminals() -> None:
    session = _UsageSession()
    record = _record()

    async def events():
        usage = {
            "prompt_tokens": 3,
            "completion_tokens": 3,
            "cached_tokens": 1,
        }
        yield {"type": "ready", "thinking": True, **usage}
        yield {"type": "delta", "index": 0, "text": "reason-0</think>answer-0", **usage}
        yield {"type": "delta", "index": 1, "text": "reason-1", **usage}
        yield {"type": "choice_finished", "index": 0, "finish_reason": "stop", **usage}
        yield {"type": "delta", "index": 1, "text": "</think>answer-1", **usage}
        yield {"type": "choice_finished", "index": 1, "finish_reason": "length", **usage}
        yield {"type": "final", **usage}

    async def collect() -> list[bytes]:
        return [
            chunk
            async for chunk in openai_chat_stream(
                AdapterRouter([record]),
                record=record,
                events=events(),
                adapter_id="adapter",
                completion_id="chatcmpl-test",
                created=123,
                include_usage=True,
                usage_session=session,  # type: ignore[arg-type]
                thinking=True,
                choice_count=2,
            )
        ]

    chunks = asyncio.run(collect())
    payloads = [json.loads(chunk[6:-2]) for chunk in chunks[:-1]]
    terminals = [
        choice
        for payload in payloads
        for choice in payload.get("choices", [])
        if choice.get("finish_reason") is not None
    ]
    assert session.finalized
    assert session.failed == []
    assert [(choice["index"], choice["finish_reason"]) for choice in terminals] == [
        (0, "stop"),
        (1, "length"),
    ]
    reasoning = "".join(
        choice.get("delta", {}).get("reasoning_content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
        if choice["index"] == 1
    )
    assert reasoning == "reason-1"
    terminal_payload = next(payload for payload in payloads if "usage" in payload)
    assert terminal_payload["usage"]["completion_tokens"] == 3
    assert chunks[-1] == b"data: [DONE]\n\n"


@pytest.mark.parametrize(
    "late_event",
    [
        {"type": "delta", "index": 0, "text": "late"},
        {"type": "choice_finished", "index": 0, "finish_reason": "stop"},
        {"type": "usage_progress"},
        {"type": "error", "message": "late error", "code": 502},
        {"type": "final"},
    ],
    ids=["delta", "choice-finished", "usage-progress", "error", "second-final"],
)
def test_hosted_sse_rejects_every_event_after_final_before_output_or_usage(
    late_event: dict[str, Any],
) -> None:
    session = _UsageSession()
    record = _record()
    chunks: list[bytes] = []

    async def events():
        yield {"type": "ready", "thinking": False, "prompt_tokens": 2, "completion_tokens": 0}
        yield {
            "type": "choice_finished",
            "index": 0,
            "finish_reason": "stop",
            "prompt_tokens": 2,
            "completion_tokens": 1,
        }
        yield {"type": "final", "prompt_tokens": 2, "completion_tokens": 1}
        yield {**late_event, "prompt_tokens": 99, "completion_tokens": 99}

    async def collect() -> None:
        async for chunk in openai_chat_stream(
            AdapterRouter([record]),
            record=record,
            events=events(),
            adapter_id="adapter",
            completion_id="chatcmpl-test",
            created=123,
            include_usage=True,
            usage_session=session,  # type: ignore[arg-type]
            thinking=False,
        ):
            chunks.append(chunk)  # noqa: PERF401

    with pytest.raises(RuntimeError, match="followed request terminal"):
        asyncio.run(collect())

    assert session.finalized == []
    assert session.failed == [
        ({"type": "final", "prompt_tokens": 2, "completion_tokens": 1}, "stream_failed")
    ]
    assert all(b"late" not in chunk and b'"completion_tokens":99' not in chunk for chunk in chunks)
    assert all(b'"usage"' not in chunk and chunk != b"data: [DONE]\n\n" for chunk in chunks)


def test_disconnect_after_final_does_not_hide_a_delayed_late_event() -> None:
    session = _UsageSession()
    record = _record()
    output: asyncio.Queue[tuple[bytes | None, Exception | None]] = asyncio.Queue()
    disconnected = asyncio.Event()
    final_observed = asyncio.Event()
    release_late = asyncio.Event()

    async def events():
        yield {"type": "ready", "thinking": False, "prompt_tokens": 2, "completion_tokens": 0}
        yield {
            "type": "choice_finished",
            "index": 0,
            "finish_reason": "stop",
            "prompt_tokens": 2,
            "completion_tokens": 1,
        }
        yield {"type": "final", "prompt_tokens": 2, "completion_tokens": 1}
        final_observed.set()
        await release_late.wait()
        yield {"type": "delta", "index": 0, "text": "late"}

    async def exercise() -> list[tuple[bytes | None, Exception | None]]:
        producer = asyncio.create_task(
            _produce_openai_chat_stream(
                AdapterRouter([record]),
                output,
                disconnected,
                record=record,
                events=events(),
                adapter_id="adapter",
                completion_id="chatcmpl-test",
                created=123,
                include_usage=True,
                usage_session=session,  # type: ignore[arg-type]
                thinking=False,
            )
        )
        await final_observed.wait()
        disconnected.set()
        release_late.set()
        await producer
        queued = []
        while not output.empty():
            queued.append(output.get_nowait())
        return queued

    queued = asyncio.run(exercise())
    chunks = [chunk for chunk, error in queued if chunk is not None and error is None]
    assert session.finalized == []
    assert session.failed == [
        ({"type": "final", "prompt_tokens": 2, "completion_tokens": 1}, "stream_failed")
    ]
    assert all(b"late" not in chunk for chunk in chunks)
    assert all(b'"usage"' not in chunk and chunk != b"data: [DONE]\n\n" for chunk in chunks)

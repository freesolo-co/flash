"""focused hosted sampling parity regressions."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from flash.serving.src.engine.lora_engine import _LoraEngineImpl
from flash.serving.src.engine.model_config import reasoning_parser_for
from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.io.openai_request import OpenAIGenerateRequest
from flash.serving.src.io.openai_stream import openai_chat_stream
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


def _engine(vllm_engine: Any, *, thinking: bool = False) -> _LoraEngineImpl:
    engine = object.__new__(_LoraEngineImpl)
    engine.base_model = QWEN
    engine.reasoning_parser = reasoning_parser_for(QWEN)
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

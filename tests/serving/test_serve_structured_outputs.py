"""Engine-side structured outputs (guided decoding): the effective spec reaches vLLM.

``_LoraEngineImpl._generate``/``_stream_generate`` must build ``SamplingParams`` with a
``StructuredOutputsParams`` for the EFFECTIVE spec: the per-call spec wins, an unspecified request
falls back to the adapter's registered default, and an explicit ``{}`` disables the default for
that call (structured_outputs=None). Runs offline against the conftest vLLM stub (whose
StructuredOutputsParams enforces the real exactly-one-constraint ValueError) and a fake engine
that captures the SamplingParams it was handed.

modal_app imports the ``modal`` SDK at module top (decorators run at import), which isn't installed
in the offline test env, so we stub it just enough to import the module + reach the engine class.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import types
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from flash.serve.contract.provenance import engine_adapter_name
from flash.serve.request.openai import parse_chat_request
from flash.serving.src.engine import generation as engine_generation
from flash.serving.src.engine.model_config import reasoning_parser_for
from flash.serving.src.engine.support import _require_reasoning_api_compatibility
from flash.serving.src.io.responses import openai_generate_fields
from flash.serving.src.io.schemas import AdapterRecord, GenerateRequest, internal_adapter_payload
from flash.serving.src.io.streaming import openai_chat_stream
from flash.serving.src.store.registry import AdapterRegistry
from tests.serving.checkpoint_fixtures import checkpoint_record

QWEN = "Qwen/Qwen3.5-9B"
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}


def _passthrough_decorator(*_a: Any, **_k: Any):
    def deco(obj: Any) -> Any:
        return obj

    return deco


@pytest.fixture(scope="module")
def modal_app_module(load_modal_app_under_stub):
    modal_stub = MagicMock(name="modal")
    modal_stub.concurrent.side_effect = _passthrough_decorator
    modal_stub.method.side_effect = _passthrough_decorator
    modal_stub.enter.side_effect = _passthrough_decorator
    modal_stub.asgi_app.side_effect = _passthrough_decorator
    modal_stub.parameter.return_value = None
    app_mock = MagicMock(name="app")
    app_mock.cls.side_effect = _passthrough_decorator
    app_mock.function.side_effect = _passthrough_decorator
    app_mock.local_entrypoint.side_effect = _passthrough_decorator
    modal_stub.App.return_value = app_mock
    modal_stub.Period.return_value = MagicMock()
    return load_modal_app_under_stub(modal_stub)


class _Tok:
    """Prompt-path tokenizer stand-in (one int per codepoint)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs) -> list[int]:
        return [1, 2, 3]


class _CaptureEngine:
    """Fake AsyncLLMEngine that records the SamplingParams each generate() was handed."""

    def __init__(self) -> None:
        self.sampling_params: list[Any] = []
        self.reasoning_ended: list[bool | None] = []
        self.reasoning_parser_kwargs: list[dict[str, Any] | None] = []

    async def generate(
        self,
        prompt_input,
        sampling_params,
        request_id,
        lora_request=None,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
    ):
        self.sampling_params.append(sampling_params)
        self.reasoning_ended.append(reasoning_ended)
        self.reasoning_parser_kwargs.append(reasoning_parser_kwargs)
        yield types.SimpleNamespace(
            outputs=[
                types.SimpleNamespace(index=0, text="ok", finish_reason="stop", token_ids=[1, 2])
            ],
            prompt_token_ids=[1],
            num_cached_tokens=0,
        )


class _RepeatedDeltaEngine(_CaptureEngine):
    async def generate(
        self,
        prompt_input,
        sampling_params,
        request_id,
        lora_request=None,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
    ):
        self.sampling_params.append(sampling_params)
        self.reasoning_ended.append(reasoning_ended)
        self.reasoning_parser_kwargs.append(reasoning_parser_kwargs)
        for text, token_id, finish_reason in (
            ("\n", 10, None),
            ("\n", 10, None),
            ("done", 11, "stop"),
        ):
            yield types.SimpleNamespace(
                outputs=[
                    types.SimpleNamespace(
                        index=0,
                        text=text,
                        finish_reason=finish_reason,
                        token_ids=[token_id],
                    )
                ],
                prompt_token_ids=[1, 2],
                num_cached_tokens=0,
            )


class _FirstAdvanceErrorEngine(_CaptureEngine):
    async def generate(
        self,
        prompt_input,
        sampling_params,
        request_id,
        lora_request=None,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
    ):
        self.sampling_params.append(sampling_params)
        self.reasoning_ended.append(reasoning_ended)
        self.reasoning_parser_kwargs.append(reasoning_parser_kwargs)
        if sampling_params is not None:
            raise ValueError("json schema semantic validation failed")
        yield None


class _CleanupEngine(_CaptureEngine):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_ran = False
        self.output_stream: Any = None

    def generate(
        self,
        prompt_input,
        sampling_params,
        request_id,
        lora_request=None,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
    ):
        self.sampling_params.append(sampling_params)
        self.reasoning_ended.append(reasoning_ended)
        self.reasoning_parser_kwargs.append(reasoning_parser_kwargs)

        async def output_stream():
            try:
                yield types.SimpleNamespace(
                    outputs=[
                        types.SimpleNamespace(index=0, text="ok", finish_reason=None, token_ids=[1])
                    ],
                    prompt_token_ids=[1],
                    num_cached_tokens=0,
                )
                await asyncio.Event().wait()
            finally:
                self.cleanup_ran = True

        self.output_stream = output_stream()
        return self.output_stream


class _ConcurrentCapacityEngine(_CaptureEngine):
    def __init__(self) -> None:
        super().__init__()
        self.entered = 0
        self.both_entered = asyncio.Event()

    async def generate(
        self,
        prompt_input,
        sampling_params,
        request_id,
        lora_request=None,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
    ):
        del prompt_input, request_id, lora_request
        self.sampling_params.append(sampling_params)
        self.reasoning_ended.append(reasoning_ended)
        self.reasoning_parser_kwargs.append(reasoning_parser_kwargs)
        self.entered += 1
        if self.entered == 2:
            self.both_entered.set()
        await self.both_entered.wait()
        yield types.SimpleNamespace(
            outputs=[
                types.SimpleNamespace(index=0, text="ok", finish_reason="stop", token_ids=[1])
            ],
            prompt_token_ids=[1],
            num_cached_tokens=0,
            metrics=types.SimpleNamespace(time_in_queue=0.125),
        )


def _engine(
    modal_app_module: Any,
    *,
    default: dict | None = None,
    base_model: str = QWEN,
    thinking: bool = False,
) -> Any:
    """A minimal engine impl around a base-model record with an optional constraint default."""
    eng = object.__new__(modal_app_module._LoraEngineImpl)
    eng.base_model = base_model
    eng.reasoning_parser = reasoning_parser_for(base_model)
    eng.tokenizer = _Tok()
    eng.registry = AdapterRegistry()
    eng.registry.hydrate(
        [
            AdapterRecord(
                adapter_id="r1",
                repo_id=base_model,
                base_model=base_model,
                serve_base_model=True,
                thinking=thinking,
                status="ready",
                structured_outputs=default,
            )
        ]
    )
    eng._adapter_locks = {}
    eng._adapter_locks_guard = asyncio.Lock()
    eng.engine = _CaptureEngine()
    return eng


def _forwarded_record(eng: Any, adapter_id: str) -> dict[str, Any]:
    record = eng.registry.get(None, adapter_id)
    assert record is not None
    return internal_adapter_payload(record)


def _generate(eng: Any, **payload_extra: Any) -> Any:
    """Run _generate and return the SamplingParams the (fake) vLLM engine received."""
    payload = {"adapter_id": "r1"}
    if "messages" not in payload_extra:
        payload["prompt"] = "hi"
    payload.update(payload_extra)
    record = eng.registry.get(None, payload["adapter_id"])
    assert record is not None
    result = asyncio.run(eng._generate(payload, internal_adapter_payload(record)))
    assert result["ok"] is True
    return eng.engine.sampling_params[-1]


def test_nonstream_generation_requires_a_real_finish_reason(modal_app_module) -> None:
    class _NonterminalEngine:
        async def generate(self, *_args, **_kwargs):
            yield types.SimpleNamespace(
                outputs=[
                    types.SimpleNamespace(
                        index=0, text="partial", finish_reason=None, token_ids=[1]
                    )
                ],
                prompt_token_ids=[1],
                num_cached_tokens=0,
            )

    eng = _engine(modal_app_module)
    eng.engine = _NonterminalEngine()

    with pytest.raises(RuntimeError, match="ended without a finish reason"):
        asyncio.run(
            eng._generate(
                {"adapter_id": "r1", "prompt": "hi"},
                _forwarded_record(eng, "r1"),
            )
        )


def test_reasoning_api_compatibility_check_fails_closed():
    @dataclasses.dataclass
    class NoParserArgs:
        model: str | None = None

    async def no_reasoning_generate(prompt, sampling_params, request_id):
        return None

    with pytest.raises(RuntimeError, match="reasoning_parser"):
        _require_reasoning_api_compatibility(NoParserArgs, no_reasoning_generate, "qwen3")


def test_reasoning_api_compatibility_check_accepts_current_shape():
    @dataclasses.dataclass
    class ParserArgs:
        reasoning_parser: str = ""

    async def generate(
        prompt,
        sampling_params,
        request_id,
        *,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
    ):
        return None

    _require_reasoning_api_compatibility(ParserArgs, generate, "qwen3")


def test_per_call_spec_reaches_sampling_params(modal_app_module):
    sp = _generate(_engine(modal_app_module), structured_outputs={"json": SCHEMA})
    assert sp.structured_outputs.json == SCHEMA


def test_per_call_bare_schema_is_normalized_engine_side(modal_app_module):
    """The engine re-validates the RPC dict, so even a bare JSON schema sent straight to an engine
    method lands as the canonical {"json": schema} constraint."""
    sp = _generate(
        _engine(modal_app_module),
        structured_outputs=SCHEMA,
    )
    assert sp.structured_outputs.json == SCHEMA


def test_no_spec_anywhere_is_unconstrained(modal_app_module):
    sp = _generate(_engine(modal_app_module))
    assert sp.structured_outputs is None


def test_adapter_default_applies_when_request_has_none(modal_app_module):
    sp = _generate(_engine(modal_app_module, default={"choice": ["a", "b"]}))
    assert sp.structured_outputs.choice == ["a", "b"]


def test_per_call_empty_spec_disables_adapter_default(modal_app_module):
    """{} = explicitly unconstrained: the adapter default must NOT apply to this call."""
    sp = _generate(_engine(modal_app_module, default={"choice": ["a", "b"]}), structured_outputs={})
    assert sp.structured_outputs is None


def test_per_call_spec_overrides_adapter_default(modal_app_module):
    sp = _generate(
        _engine(modal_app_module, default={"choice": ["a", "b"]}),
        structured_outputs={"regex": r"\d+"},
    )
    assert sp.structured_outputs.regex == r"\d+"
    assert sp.structured_outputs.choice is None


def test_reasoning_state_matches_effective_thinking_mode(modal_app_module):
    unconstrained = _engine(modal_app_module)
    _generate(unconstrained)
    assert unconstrained.engine.reasoning_ended[-1] is None
    assert unconstrained.engine.reasoning_parser_kwargs[-1] is None

    non_thinking = _engine(modal_app_module)
    _generate(non_thinking, structured_outputs={"json": SCHEMA})
    assert non_thinking.engine.reasoning_ended[-1] is True
    assert non_thinking.engine.reasoning_parser_kwargs[-1] == {
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}
    }

    thinking = _engine(modal_app_module, thinking=False, default={"json": SCHEMA})
    _generate(
        thinking,
        messages=[{"role": "user", "content": "hi"}],
        chat_template_kwargs={"enable_thinking": True, "tools": ["search"]},
    )
    assert thinking.engine.reasoning_ended[-1] is False
    assert thinking.engine.reasoning_parser_kwargs[-1] == {
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": False,
        }
    }


def test_request_precedence_and_explicit_off_control_reasoning_state(modal_app_module):
    eng = _engine(modal_app_module, thinking=True, default={"choice": ["adapter"]})
    params = _generate(
        eng,
        messages=[{"role": "user", "content": "hi"}],
        structured_outputs={"choice": ["request"]},
    )
    assert params.structured_outputs.choice == ["request"]
    assert eng.engine.reasoning_ended[-1] is False

    params = _generate(eng, structured_outputs={})
    assert params.structured_outputs is None
    assert eng.engine.reasoning_ended[-1] is None
    assert eng.engine.reasoning_parser_kwargs[-1] is None


def test_each_request_gets_fresh_structured_outputs_params(modal_app_module):
    eng = _engine(modal_app_module, default={"json": SCHEMA})
    first = _generate(eng).structured_outputs
    second = _generate(eng).structured_outputs
    assert first is not second


def test_concurrent_requests_do_not_leak_reasoning_or_grammar_state(modal_app_module):
    eng = _engine(modal_app_module, thinking=True, default={"json": SCHEMA})
    eng.registry.upsert(
        AdapterRecord(
            adapter_id="non-thinking",
            repo_id=QWEN,
            base_model=QWEN,
            serve_base_model=True,
            thinking=False,
            structured_outputs={"choice": ["plain"]},
        )
    )

    async def run_both() -> None:
        await asyncio.gather(
            eng._generate(
                {
                    "adapter_id": "r1",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                _forwarded_record(eng, "r1"),
            ),
            eng._generate(
                {"adapter_id": "non-thinking", "prompt": "hi"},
                _forwarded_record(eng, "non-thinking"),
            ),
        )

    asyncio.run(run_both())
    assert sorted(eng.engine.reasoning_ended, key=str) == [False, True]
    assert len({id(params.structured_outputs) for params in eng.engine.sampling_params}) == 2
    assert {
        tuple(params.structured_outputs.choice or []) for params in eng.engine.sampling_params
    } == {
        (),
        ("plain",),
    }


def test_thinking_constraint_requires_messages(modal_app_module):
    eng = _engine(modal_app_module, thinking=True, default={"json": SCHEMA})
    with pytest.raises(ValueError, match="require messages"):
        _generate(eng)
    assert eng.engine.sampling_params == []


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("prompt", ["raw prompt", None])
def test_thinking_constraint_rejects_empty_messages_before_vllm(
    modal_app_module, streaming, prompt
):
    eng = _engine(modal_app_module, thinking=True, default={"json": SCHEMA})
    payload = {"adapter_id": "r1", "messages": [], "prompt": prompt}

    async def run_request():
        if streaming:
            return await anext(
                eng._stream_generate(
                    payload,
                    _forwarded_record(eng, payload["adapter_id"]),
                )
            )
        return await eng._generate(
            payload,
            _forwarded_record(eng, payload["adapter_id"]),
        )

    with pytest.raises(ValueError, match="exactly one nonempty prompt or messages"):
        asyncio.run(run_request())
    assert eng.engine.sampling_params == []


def test_thinking_constraint_requires_configured_parser(modal_app_module):
    eng = _engine(modal_app_module, thinking=True, default={"json": SCHEMA})
    eng.reasoning_parser = None
    with pytest.raises(ValueError, match="parser-enabled base model"):
        _generate(eng, messages=[{"role": "user", "content": "hi"}])
    assert eng.engine.sampling_params == []


def test_stream_generate_attests_the_resolved_checkpoint_before_deltas(modal_app_module):
    eng = _engine(modal_app_module)
    revision = checkpoint_record("run-1", QWEN)
    checkpoint_id = revision.adapter_id

    async def resolved_lora(_adapter_id, _record_dict=None):
        return (
            types.SimpleNamespace(lora_name=engine_adapter_name(revision.org_id, checkpoint_id)),
            revision,
        )

    eng._lora_request = resolved_lora

    async def first_event():
        stream = eng._stream_generate(
            {"adapter_id": checkpoint_id, "prompt": "hi"},
            internal_adapter_payload(revision),
            expected_checkpoint=checkpoint_id,
        )
        try:
            return await anext(stream)
        finally:
            await stream.aclose()

    ready = asyncio.run(first_event())

    assert ready["type"] == "ready"
    assert ready["lora_request_adapter"] == checkpoint_id
    assert ready["checkpoint"] == checkpoint_id


def test_queue_wait_prefers_real_metrics_and_omits_unavailable() -> None:
    direct = types.SimpleNamespace(metrics=types.SimpleNamespace(time_in_queue=0.125))
    derived = types.SimpleNamespace(
        metrics=types.SimpleNamespace(
            time_in_queue=None,
            arrival_time=20.0,
            first_scheduled_time=20.375,
        )
    )

    assert engine_generation._queue_wait_seconds(direct) == 0.125
    assert engine_generation._queue_wait_seconds(derived) == 0.375
    assert engine_generation._queue_wait_seconds(types.SimpleNamespace()) is None


def test_generation_reports_capacity_at_admission_and_releases_counter(modal_app_module):
    from vllm.sampling_params import RequestOutputKind

    eng = _engine(modal_app_module)
    eng.engine = _ConcurrentCapacityEngine()
    eng._replica_in_flight_requests = 0
    eng._replica_first_request_pending = True
    eng._replica_boot_duration_seconds = 12.5

    async def run_both():
        payload = {"adapter_id": "r1", "prompt": "hi"}
        record = _forwarded_record(eng, "r1")
        return await asyncio.gather(eng._generate(payload, record), eng._generate(payload, record))

    results = asyncio.run(run_both())

    assert [result["replica_in_flight_requests_at_admission"] for result in results] == [1, 2]
    assert [result["replica_freshly_booted"] for result in results] == [True, False]
    assert all(result["replica_boot_duration_seconds"] == 12.5 for result in results)
    assert all(result["queue_wait_seconds"] == 0.125 for result in results)
    # the buffered path reports no time-to-first-token: FINAL_ONLY yields once, at completion,
    # so any such value would be the completion interval wearing a first-token name.
    assert all("time_to_first_token_seconds" not in result for result in results)
    # and it must still run FINAL_ONLY -- telemetry may not change how generation executes.
    assert all(
        params.output_kind == RequestOutputKind.FINAL_ONLY for params in eng.engine.sampling_params
    )
    assert eng._replica_in_flight_requests == 0


def test_stream_generate_carries_structured_outputs(modal_app_module):
    eng = _engine(modal_app_module)

    async def _drain(agen):
        return [event async for event in agen]

    events = asyncio.run(
        _drain(
            eng._stream_generate(
                {"adapter_id": "r1", "prompt": "hi", "structured_outputs": SCHEMA},
                _forwarded_record(eng, "r1"),
            )
        )
    )
    assert [event["type"] for event in events] == [
        "ready",
        "delta",
        "choice_finished",
        "final",
    ]
    ready = events[0].copy()
    assert ready.pop("inference_time_seconds") >= 0
    assert ready.pop("time_to_first_token_seconds") >= 0
    # the two ids are uuid4-derived, so they cannot be spelled literally. pin their shape here and
    # their VALUE on the delta below, against this event -- writing `ready["request_id"]` into this
    # dict would compare the value to itself and pin nothing at all.
    request_id = ready.pop("request_id")
    replica_id = ready.pop("engine_replica_id")
    assert uuid.UUID(hex=request_id.removeprefix("fsgen-")).version == 4
    assert uuid.UUID(hex=replica_id).version == 4
    assert ready == {
        "type": "ready",
        "thinking": False,
        "prompt_token_ids": [1],
        "completion_token_ids": [1, 2],
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "cached_tokens": 0,
        "cached_tokens_reported": True,
        "reasoning_tokens": 0,
        "checkpoint": "",
    }
    delta = events[1].copy()
    assert delta.pop("inference_time_seconds") >= 0
    assert delta.pop("time_to_first_token_seconds") >= 0
    assert delta == {
        "type": "delta",
        "index": 0,
        "text": "ok",
        "logprobs": None,
        "thinking": False,
        "prompt_token_ids": [1],
        "completion_token_ids": [1, 2],
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "cached_tokens": 0,
        "cached_tokens_reported": True,
        "reasoning_tokens": 0,
        "request_id": request_id,
        "engine_replica_id": replica_id,
        "checkpoint": "",
    }
    assert events[-2]["finish_reason"] == "stop"
    assert events[-1]["prompt_tokens"] == 1
    assert events[-1]["completion_tokens"] == 2
    sp = eng.engine.sampling_params[-1]
    assert sp.structured_outputs.json == SCHEMA  # raw schema normalized to a json constraint
    assert eng.engine.reasoning_ended[-1] is True


def _repeated_delta_events(modal_app_module):
    eng = _engine(modal_app_module)
    eng.engine = _RepeatedDeltaEngine()

    async def drain():
        return [
            event
            async for event in eng._stream_generate(
                {"adapter_id": "r1", "prompt": "hi"},
                _forwarded_record(eng, "r1"),
            )
        ]

    return eng, asyncio.run(drain())


def test_stream_generate_extends_repeated_delta_tokens(modal_app_module):
    from vllm.sampling_params import RequestOutputKind

    eng, events = _repeated_delta_events(modal_app_module)

    assert eng.engine.sampling_params[-1].output_kind == RequestOutputKind.DELTA
    assert [event["completion_tokens"] for event in events if event["type"] == "delta"] == [
        1,
        2,
        3,
    ]


def test_stream_generate_reports_repeated_delta_usage_totals(modal_app_module):
    _, events = _repeated_delta_events(modal_app_module)

    assert events[0]["completion_tokens"] == 1
    assert events[-1]["prompt_tokens"] == 2
    assert events[-1]["completion_tokens"] == 3


def test_openai_sse_keeps_repeated_delta_text(modal_app_module):
    _, events = _repeated_delta_events(modal_app_module)
    record = AdapterRecord(
        adapter_id="r1",
        repo_id=QWEN,
        base_model=QWEN,
        serve_base_model=True,
        thinking=False,
        status="ready",
    )

    async def event_stream():
        for event in events:
            yield event

    class UsageSession:
        async def finalize(self, _result):
            return None

        async def capture(self, _result):
            return None

        async def fail(self, _result, _code):
            return None

    async def drain():
        return [
            chunk
            async for chunk in openai_chat_stream(
                MagicMock(),
                record=record,
                events=event_stream(),
                adapter_id="r1",
                completion_id="completion-1",
                created=1,
                include_usage=True,
                usage_session=UsageSession(),
            )
        ]

    chunks = asyncio.run(drain())
    payloads = [
        json.loads(chunk.removeprefix(b"data: ").removesuffix(b"\n\n"))
        for chunk in chunks
        if chunk != b"data: [DONE]\n\n"
    ]
    emitted_text = "".join(
        payload["choices"][0]["delta"].get("content", "") for payload in payloads
    )

    assert emitted_text == "\n\ndone"
    assert payloads[-1]["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }


def test_stream_close_after_ready_closes_inner_generator(modal_app_module):
    eng = _engine(modal_app_module)
    eng.engine = _CleanupEngine()

    async def prime_and_close():
        stream = eng._stream_generate(
            {"adapter_id": "r1", "prompt": "hi"},
            _forwarded_record(eng, "r1"),
        )
        ready = (await anext(stream)).copy()
        assert ready.pop("inference_time_seconds") >= 0
        assert ready.pop("time_to_first_token_seconds") >= 0
        assert ready == {
            "type": "ready",
            "thinking": False,
            "prompt_token_ids": [1],
            "completion_token_ids": [1],
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
            "cached_tokens_reported": True,
            "reasoning_tokens": 0,
            "request_id": ready["request_id"],
            "engine_replica_id": ready["engine_replica_id"],
            "checkpoint": "",
        }
        await stream.aclose()
        assert eng.engine.cleanup_ran is True

    asyncio.run(prime_and_close())


def test_streaming_and_non_streaming_reasoning_state_match(modal_app_module):
    eng = _engine(modal_app_module, thinking=True, default={"json": SCHEMA})
    messages = [{"role": "user", "content": "hi"}]
    _generate(eng, messages=messages)

    async def _drain(agen):
        return [event async for event in agen]

    events = asyncio.run(
        _drain(
            eng._stream_generate(
                {"adapter_id": "r1", "messages": messages},
                _forwarded_record(eng, "r1"),
            )
        )
    )
    assert events[0]["type"] == "ready"
    assert events[-1]["type"] == "final"
    assert eng.engine.reasoning_ended == [False, False]
    assert eng.engine.reasoning_parser_kwargs == [
        {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": False,
            }
        },
        {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": False,
            }
        },
    ]
    assert (
        eng.engine.sampling_params[0].structured_outputs
        is not eng.engine.sampling_params[1].structured_outputs
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_semantically_invalid_schema_raises_on_first_engine_advance(modal_app_module, streaming):
    eng = _engine(modal_app_module)
    eng.engine = _FirstAdvanceErrorEngine()
    eng._self_heal_if_dead = MagicMock()
    payload = {"adapter_id": "r1", "prompt": "hi", "structured_outputs": {"json": SCHEMA}}

    async def run_request():
        if streaming:
            return await anext(
                eng._stream_generate(
                    payload,
                    _forwarded_record(eng, payload["adapter_id"]),
                )
            )
        return await eng._generate(
            payload,
            _forwarded_record(eng, payload["adapter_id"]),
        )

    with pytest.raises(ValueError, match="semantic validation failed"):
        asyncio.run(run_request())
    assert len(eng.engine.sampling_params) == 1
    assert eng.engine.sampling_params[0].structured_outputs.json == SCHEMA
    eng._self_heal_if_dead.assert_called_once_with("stream_generate" if streaming else "generate")


def test_invalid_stored_spec_raises_value_error_before_ready(modal_app_module):
    """Defensive path: a spec that somehow bypassed normalization (here: model_construct) must
    surface as a ValueError — which the router maps to a 400 — and on the streaming path it must
    raise BEFORE the first ("ready") event so _prepare_stream's first-anext guard catches it."""
    eng = _engine(modal_app_module)
    bad = AdapterRecord.model_construct(
        adapter_id="r1",
        repo_id=QWEN,
        base_model=QWEN,
        serve_base_model=True,
        thinking=False,
        status="ready",
        structured_outputs={"json": SCHEMA, "regex": r"\d+"},  # two constraints
    )
    eng.registry.hydrate([bad])

    with pytest.raises(ValueError, match="structured outputs spec must set exactly one constraint"):
        asyncio.run(
            eng._generate(
                {"adapter_id": "r1", "prompt": "hi"},
                _forwarded_record(eng, "r1"),
            )
        )

    async def _first(agen):
        return await anext(agen)

    with pytest.raises(ValueError, match="structured outputs spec must set exactly one constraint"):
        asyncio.run(
            _first(
                eng._stream_generate(
                    {"adapter_id": "r1", "prompt": "hi"},
                    _forwarded_record(eng, "r1"),
                )
            )
        )
    assert eng.engine.sampling_params == []  # never reached vLLM


# --- stop sequences ---------------------------------------------------------------------------
# `docs/serving-contract.md` documents `stop` as an accepted field and requires stop handling to be
# preserved. Pydantic drops keys a model does not declare, so an undeclared `stop` was accepted with
# a 200 and then ignored: generation ran past the caller's stop string and billed the extra tokens.


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("STOP", "STOP"),
        (["</s>", "\n\n"], ["</s>", "\n\n"]),
        (None, None),
        ([], None),  # no sequences supplied, so no constraint
    ],
)
def test_stop_reaches_sampling_params(modal_app_module, value, expected):
    assert _generate(_engine(modal_app_module), stop=value).stop == expected


def test_an_empty_stop_string_is_rejected_rather_than_silently_dropped(modal_app_module):
    """An authored sequence that can never terminate generation must not become "unconstrained".

    Accepting `""` dispatched a run with no stop constraint while the caller believed they had set
    one. The same value is already refused inside a list, and the flash-owned runtime validator
    refuses it too, so the bare form was the inconsistent case. `None` remains the one spelling of
    "no constraint".
    """
    with pytest.raises(ValidationError, match="must not be an empty string"):
        _generate(_engine(modal_app_module), stop="")


def test_stop_reaches_sampling_params_when_streaming(modal_app_module):
    """The streaming path builds its own SamplingParams and must carry stop too."""
    eng = _engine(modal_app_module)

    async def _drain(agen):
        return [event async for event in agen]

    events = asyncio.run(
        _drain(
            eng._stream_generate(
                {"adapter_id": "r1", "prompt": "hi", "stop": ["</s>"]},
                _forwarded_record(eng, "r1"),
            )
        )
    )
    assert [event["type"] for event in events] == [
        "ready",
        "delta",
        "choice_finished",
        "final",
    ]
    assert eng.engine.sampling_params[-1].stop == ["</s>"]


def test_openai_body_forwards_stop():
    """The OpenAI chat-completions translation must not drop the standard `stop` field."""
    with_stop = parse_chat_request(
        {"model": "r1", "messages": [{"role": "user", "content": "hi"}], "stop": ["</s>"]},
        require_model=True,
        allow_managed_selectors=False,
    )
    without_stop = parse_chat_request(
        {"model": "r1", "messages": [{"role": "user", "content": "hi"}]},
        require_model=True,
        allow_managed_selectors=False,
    )
    assert openai_generate_fields(with_stop, "r1")["stop"] == ["</s>"]
    assert openai_generate_fields(without_stop, "r1")["stop"] is None


@pytest.mark.parametrize("bad", [123, {"a": 1}, ["ok", ""], ["ok", 5]])
def test_malformed_stop_is_rejected_rather_than_ignored(bad):
    """A bad `stop` is a 422, not a silently discarded field."""
    with pytest.raises(ValidationError):
        GenerateRequest(adapter_id="r1", prompt="hi", stop=bad)

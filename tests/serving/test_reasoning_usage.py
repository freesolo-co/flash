"""Native thinking accounting, including multi-choice and partial streaming output."""

import asyncio
from types import SimpleNamespace

import pytest

from flash.serving.src.accounting.usage_facts import usage_facts
from flash.serving.src.accounting.usage_outbox import UsageOutboxError
from flash.serving.src.engine.reasoning_usage import ReasoningTokenCounter, reasoning_token_ids
from tests.serving import test_serve_structured_outputs as structured_helpers

modal_app_module = structured_helpers.modal_app_module

START = 151667
END = 151668


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ([10, 11, END, 12], 2),
        ([START, 10, END, 12], 1),
        ([10, 11], 2),
        ([END, 12], 0),
        ([START, END], 0),
        ([], 0),
        ([10, END, START, 12, END], 1),
        ([START, 10, START, 11, END], 3),
    ],
)
def test_reasoning_count_is_independent_of_chunk_boundaries(tokens, expected):
    for boundary in range(len(tokens) + 1):
        counter = ReasoningTokenCounter((START, END))
        counter.consume(0, tokens[:boundary])
        counter.consume(0, tokens[boundary:])
        assert counter.total == expected


@pytest.mark.parametrize("vocab", [{}, {"<think>": START}, {"<think>": 1, "</think>": 1}])
def test_missing_native_boundaries_fail_closed(vocab):
    with pytest.raises(ValueError, match="native think token IDs"):
        reasoning_token_ids("qwen3", SimpleNamespace(convert_tokens_to_ids=vocab.get))


def test_unqualified_parser_fails_closed():
    with pytest.raises(ValueError, match="qwen3"):
        reasoning_token_ids("unknown", None)


def test_non_thinking_does_not_require_parser_or_tokenizer():
    counter = ReasoningTokenCounter(None)
    counter.consume(0, [START, 1, END])
    assert counter.total == 0


def _output(index, tokens, finish=None):
    return SimpleNamespace(index=index, text="", token_ids=tokens, finish_reason=finish)


def _batch(*outputs):
    return SimpleNamespace(outputs=outputs, prompt_token_ids=[START, 8], num_cached_tokens=0)


class _NativeEngine:
    def __init__(self, batches):
        self.batches = batches
        self.closed = False

    async def generate(self, *_args, **_kwargs):
        try:
            for batch in self.batches:
                yield batch
        finally:
            self.closed = True


@pytest.mark.parametrize("thinking", [True, False])
def test_buffered_and_streamed_native_counts_agree_for_multiple_choices(modal_app_module, thinking):
    eng = structured_helpers._engine(modal_app_module, thinking=thinking)
    payload = {"model_id": "r1", "prompt": "hi", "n": 2, "temperature": 0.7}
    record = structured_helpers._forwarded_record(eng, "r1")
    eng.engine = _NativeEngine(
        [_batch(_output(1, [START, 20, 21], "length"), _output(0, [10, END, 11, 12], "stop"))]
    )
    buffered = asyncio.run(eng._generate(payload, record))
    eng.engine = _NativeEngine(
        [
            _batch(_output(1, [START, 20]), _output(0, [10, END, 11])),
            _batch(_output(0, [12], "stop")),
            _batch(_output(1, [21], "length")),
        ]
    )

    async def collect():
        return [event async for event in eng._stream_generate(payload, record)]

    events = asyncio.run(collect())
    assert events[0]["type"] == "ready"
    assert events[-1]["type"] == "final"
    assert events[0]["completion_tokens"] == 5
    assert events[-1]["completion_tokens"] == buffered["completion_tokens"] == 7
    assert events[-1]["reasoning_tokens"] == buffered["reasoning_tokens"] == (3 if thinking else 0)
    assert events[0]["reasoning_tokens"] == (2 if thinking else 0)
    for field in ("completion_tokens", "reasoning_tokens", "inference_time_seconds"):
        counts = [event[field] for event in events]
        assert counts == sorted(counts)
    for event in events:
        assert usage_facts(event).reasoning_tokens <= event["completion_tokens"]
    assert eng.engine.closed


def test_closing_after_ready_preserves_partial_reasoning_facts(modal_app_module):
    eng = structured_helpers._engine(modal_app_module, thinking=True)
    eng.engine = _NativeEngine([_batch(_output(0, [10, 11]))])

    async def disconnect():
        stream = eng._stream_generate(
            {"model_id": "r1", "prompt": "hi"}, structured_helpers._forwarded_record(eng, "r1")
        )
        ready = await anext(stream)
        await stream.aclose()
        return usage_facts(ready)

    facts = asyncio.run(disconnect())
    assert facts.reasoning_tokens == facts.completion_tokens == 2
    assert eng.engine.closed


@pytest.mark.parametrize("reasoning", [None, True, -1, 3])
def test_invalid_thinking_counts_cannot_settle(reasoning):
    with pytest.raises(UsageOutboxError):
        usage_facts(
            {
                "thinking": True,
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "reasoning_tokens": reasoning,
            }
        )


def test_boundary_lookup_is_cached_per_engine_without_loading_vocab(modal_app_module):
    calls = []

    class Tokenizer(structured_helpers._Tok):
        def get_vocab(self):
            raise AssertionError("must not copy the vocabulary on the serving loop")

        def convert_tokens_to_ids(self, token):
            calls.append(token)
            return super().convert_tokens_to_ids(token)

    eng = structured_helpers._engine(modal_app_module, thinking=True)
    eng.tokenizer = Tokenizer()
    eng.engine = _NativeEngine([_batch(_output(0, [10, END], "stop"))])
    payload = {"model_id": "r1", "prompt": "hi"}
    record = structured_helpers._forwarded_record(eng, "r1")
    first = asyncio.run(eng._generate(payload, record))
    second = asyncio.run(eng._generate(payload, record))

    async def stream():
        return [event async for event in eng._stream_generate(payload, record)]

    events = asyncio.run(stream())
    assert (
        first["reasoning_tokens"]
        == second["reasoning_tokens"]
        == events[-1]["reasoning_tokens"]
        == 1
    )
    assert calls == ["<think>", "</think>"]


def test_missing_boundary_must_not_resolve_to_unknown_token():
    tokenizer = SimpleNamespace(
        unk_token_id=0, convert_tokens_to_ids=lambda token: START if token == "<think>" else 0
    )
    with pytest.raises(ValueError, match="native think token IDs"):
        reasoning_token_ids("qwen3", tokenizer)

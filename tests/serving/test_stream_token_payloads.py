"""Bounded streaming payloads and the real engine/channel interface."""

import asyncio
import time
from types import SimpleNamespace

import orjson
import pytest

from flash.serving.src.accounting.usage_facts import usage_facts
from flash.serving.src.engine.lora_engine import _LoraEngineImpl
from flash.serving.src.stream_channel.engine import stream_generate_call
from flash.serving.src.stream_channel.protocol import DATA_PARTITION, ChannelErrorCode
from tests.serving.test_inference_duration import _Owner, _revision
from tests.serving.test_stream_channel import _control, _FakeQueue, _patch_modal


class _StreamingEngine:
    def __init__(self, length, *, choices=1, hidden=False):
        self.length = length
        self.choices = choices
        self.hidden = hidden
        self.started = False
        self.closed = False

    async def generate(self, *_args, **_kwargs):
        self.started = True
        try:
            for position in range(self.length):
                yield SimpleNamespace(
                    prompt_token_ids=[11] * 256,
                    num_cached_tokens=128,
                    outputs=[
                        SimpleNamespace(
                            index=index,
                            text="" if self.hidden else "x",
                            token_ids=[21 + index],
                            logprobs=None,
                            finish_reason="stop" if position == self.length - 1 else None,
                        )
                        for index in range(self.choices)
                    ],
                )
        finally:
            self.closed = True


class _ServingOwner(_Owner, _LoraEngineImpl):
    """Keep the real serving wrapper and generation, substituting GPU/model I/O."""


def _owner(length=4, *, choices=1, hidden=False):
    owner = _ServingOwner(_revision())
    owner.engine = _StreamingEngine(length, choices=choices, hidden=hidden)
    return owner


def _payload(owner, *, choices=1):
    return {"model_id": owner.record.model_id, "prompt": "hi", "n": choices, "temperature": 0.7}


@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("choices", [1, 2])
def test_progress_has_only_token_deltas_and_settleable_native_counters(hidden, choices):
    owner = _owner(choices=choices, hidden=hidden)

    async def collect():
        return [event async for event in owner._stream_generate(_payload(owner, choices=choices))]

    events = asyncio.run(collect())
    ready, final = events[0], events[-1]
    assert ready["type"] == "ready"
    assert final["type"] == "final"
    assert len(ready["prompt_token_ids"]) == len(final["prompt_token_ids"]) == 256
    assert len(ready["completion_token_ids"]) == choices
    assert len(final["completion_token_ids"]) == 4 * choices
    deltas = {index: [] for index in range(choices)}
    for event in events[1:-1]:
        assert "prompt_token_ids" not in event
        assert "completion_token_ids" not in event
        facts = usage_facts(event)
        assert facts.prompt_tokens == 256
        assert facts.cached_tokens == 128
        assert facts.completion_tokens == event["completion_tokens"]
        if "token_ids" in event:
            deltas[event["index"]].extend(event["token_ids"])
    for choice in final["choices"]:
        assert deltas[choice["index"]] == choice["token_ids"]
    assert owner.engine.closed


def test_stream_transport_bytes_scale_linearly_with_output_length():
    async def wire_bytes(length):
        owner = _owner(length)
        return sum(
            [len(orjson.dumps(event)) async for event in owner._stream_generate(_payload(owner))]
        )

    small = asyncio.run(wire_bytes(128))
    large = asyncio.run(wire_bytes(512))
    # Four times as many tokens should stay near four times the bytes, even with
    # full histories at ready/final. Repeating growing histories fails this bound.
    assert large < small * 4.2


@pytest.mark.parametrize("cancel_after_preparation", [False, True])
def test_channel_drives_real_engine_and_rechecks_lease_before_gpu_work(
    monkeypatch, cancel_after_preparation
):
    async def scenario():
        queue = _FakeQueue()
        _patch_modal(monkeypatch, queue)
        await queue.control(_control(0, call_id=None))
        await queue.control(_control(1))
        owner = _owner()
        if cancel_after_preparation:
            prepare = owner._prepare_prompt_input

            async def prepare_then_cancel(*args):
                prompt = await prepare(*args)
                await queue.control(_control(2, kind="cancel"))
                return prompt

            owner._prepare_prompt_input = prepare_then_cancel
        manifest = await stream_generate_call(
            owner,
            _payload(owner),
            None,
            None,
            "generation-1",
            time.time() + 5,
            queue.object_id,
            "nonce-1",
        )
        envelopes = [value for partition, value in queue.put_values if partition == DATA_PARTITION]
        return owner, manifest, envelopes

    owner, manifest, envelopes = asyncio.run(scenario())
    if cancel_after_preparation:
        assert not owner.engine.started
        assert manifest["terminal_kind"] == "error"
        assert envelopes[-1]["error_code"] == ChannelErrorCode.CANCELLED
    else:
        assert owner.engine.started
        assert owner.engine.closed
        assert manifest["terminal_kind"] == "event"
        assert envelopes[0]["event"]["type"] == "ready"
        assert envelopes[-1]["event"]["type"] == "final"
        assert envelopes[-1]["event"]["completion_tokens"] == 4

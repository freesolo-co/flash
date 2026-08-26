"""Bounded pre-header dispatch expiry for queued Modal engine inputs."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from flash.serving.src.engine import dispatch, generation
from flash.serving.src.engine.dispatch import (
    PRE_HEADER_DISPATCH_TIMEOUT_SECONDS,
    PreHeaderDispatchExpired,
    new_pre_header_dispatch_deadline,
)
from flash.serving.src.http.context import ServingContext
from flash.serving.src.io.schemas import GenerateRequest


def test_dispatch_deadline_is_absolute_bounded_and_not_public() -> None:
    payload = GenerateRequest(adapter_id="run-a", prompt="hi")

    ServingContext.set_pre_header_dispatch_deadline(payload)

    remaining = payload._pre_header_dispatch_deadline - __import__("time").time()
    assert 0 < remaining <= PRE_HEADER_DISPATCH_TIMEOUT_SECONDS
    assert PRE_HEADER_DISPATCH_TIMEOUT_SECONDS < 150
    assert "pre_header_dispatch_deadline" not in payload.model_dump()
    routed = payload.model_copy(update={"adapter_id": "run-a@final.sha"})
    assert routed._pre_header_dispatch_deadline == payload._pre_header_dispatch_deadline
    with pytest.raises(ValueError, match="extra_forbidden"):
        GenerateRequest.model_validate(
            {
                "adapter_id": "run-a",
                "prompt": "hi",
                "pre_header_dispatch_deadline": 1,
            }
        )


def test_absolute_deadline_uses_wall_clock() -> None:
    assert new_pre_header_dispatch_deadline(clock=lambda: 1000.0) == (
        1000.0 + PRE_HEADER_DISPATCH_TIMEOUT_SECONDS
    )


@pytest.mark.parametrize("stream", [False, True])
def test_expired_dispatch_never_hydrates_adapter_or_starts_generation(stream: bool) -> None:
    class Owner:
        async def _lora_request(self, *_args):
            raise AssertionError("expired dispatch reached adapter hydration")

    async def scenario() -> None:
        if stream:
            events = generation.stream_generate(
                Owner(),
                {"adapter_id": "run-a", "prompt": "hi"},
                pre_header_dispatch_deadline=1.0,
            )
            await anext(events)
        else:
            await generation.generate(
                Owner(),
                {"adapter_id": "run-a", "prompt": "hi"},
                pre_header_dispatch_deadline=1.0,
            )

    with pytest.raises(PreHeaderDispatchExpired):
        asyncio.run(scenario())


def test_deadline_is_rechecked_immediately_before_generation(monkeypatch) -> None:
    sampling_params = types.ModuleType("vllm.sampling_params")
    sampling_params.RequestOutputKind = types.SimpleNamespace(FINAL_ONLY="final")
    vllm = types.ModuleType("vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)
    now = iter((0.0, 0.0, 10.0))
    monkeypatch.setattr(dispatch.time, "time", lambda: next(now))

    payload = types.SimpleNamespace(
        adapter_id="run-a",
        logprobs=False,
        generation_id="request-1",
        n=1,
        top_logprobs=0,
    )
    monkeypatch.setattr(generation, "_payload", lambda _payload: (payload, False))
    monkeypatch.setattr(generation, "_sampling_params", lambda *_args: object())

    class Engine:
        def generate(self, *_args, **_kwargs):
            raise AssertionError("expired dispatch started gpu generation")

    class Owner:
        engine = Engine()

        async def _lora_request(self, *_args):
            return None, object()

        def _lora_request_attestation(self, *_args):
            return None

        def _enforce_expected_checkpoint(self, *_args):
            return "run-a"

        def _thinking_default(self, *_args):
            return False

        def _structured_outputs_state(self, *_args):
            return None, True, None

        async def _prepare_prompt_input(self, *_args):
            return {"prompt_token_ids": [1]}

        def _close_prompt_images(self, *_args):
            return None

    with pytest.raises(PreHeaderDispatchExpired):
        asyncio.run(
            generation.generate(
                Owner(),
                {"adapter_id": "run-a", "prompt": "hi"},
                pre_header_dispatch_deadline=5.0,
            )
        )

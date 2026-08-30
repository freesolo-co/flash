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
from flash.serving.src.stream_channel.protocol import ChannelErrorCode, StreamChannelError

pytest.importorskip("fastapi")


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


def test_channel_pre_generate_check_replaces_local_stream_deadline_fallback(
    monkeypatch,
) -> None:
    sampling_params = types.ModuleType("vllm.sampling_params")
    sampling_params.RequestOutputKind = types.SimpleNamespace(DELTA="delta")
    vllm = types.ModuleType("vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

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
            raise AssertionError("expired channel work started gpu generation")

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

        def _replica_identifier(self):
            return "replica-1"

        def _structured_outputs_state(self, *_args):
            return None, True, None

        async def _prepare_prompt_input(self, *_args):
            return {"prompt_token_ids": [1]}

        def _close_prompt_images(self, *_args):
            return None

    checked = False

    async def pre_generate_check() -> None:
        nonlocal checked
        checked = True
        raise StreamChannelError(
            ChannelErrorCode.DISPATCH_DEADLINE,
            "dispatch deadline expired before generation",
        )

    async def scenario() -> None:
        events = generation.stream_generate(
            Owner(),
            {"adapter_id": "run-a", "prompt": "hi"},
            pre_generate_check=pre_generate_check,
        )
        await anext(events)

    with pytest.raises(StreamChannelError) as exc_info:
        asyncio.run(scenario())
    assert checked
    assert exc_info.value.code == ChannelErrorCode.DISPATCH_DEADLINE


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


@pytest.mark.parametrize("stream", [False, True])
def test_every_inference_route_stamps_the_deadline_before_dispatch(stream: bool) -> None:
    """The front door, not the engine, authors the deadline.

    Asserted on what the pool RECEIVES rather than on the helper in isolation: a route that never
    calls the helper would leave the engine with no deadline at all, and every bounded-dispatch
    guard downstream reads as satisfied because there is nothing to compare against.
    """
    from fastapi.testclient import TestClient
    from test_router import QWEN, FakePool, _allow, _router_for

    from flash.serving.src.http.router import build_offline_serving_app

    seen: list[object] = []

    class _RecordingPool(FakePool):
        async def generate(self, base_model, payload, record, *, expected_checkpoint=None):
            seen.append(getattr(payload, "_pre_header_dispatch_deadline", None))
            return await super().generate(
                base_model, payload, record, expected_checkpoint=expected_checkpoint
            )

        def stream_generate(self, base_model, payload, record, *, expected_checkpoint=None):
            seen.append(getattr(payload, "_pre_header_dispatch_deadline", None))
            return super().stream_generate(
                base_model, payload, record, expected_checkpoint=expected_checkpoint
            )

    app = build_offline_serving_app(
        _RecordingPool(), _router_for("run-a", QWEN), chat_authorizer=_allow
    )
    client = TestClient(app, headers={"Authorization": "Bearer t"})
    body = {"model": "run-a/final", "messages": [{"role": "user", "content": "hi"}]}
    if stream:
        body["stream"] = True

    resp = client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200
    assert len(seen) == 1
    deadline = seen[0]
    assert isinstance(deadline, float)
    remaining = deadline - __import__("time").time()
    assert 0 < remaining <= PRE_HEADER_DISPATCH_TIMEOUT_SECONDS

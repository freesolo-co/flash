from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

import httpx
import pytest

from flash.serving.loadtest.protocol import (
    ProtocolError,
    get_health,
    resolve_discovered_models,
    stream_chat,
)
from flash.serving.loadtest.schedule import FakeClock
from flash.serving.loadtest.schema import (
    AdapterTarget,
    BaseTarget,
    DeploymentExpectation,
    RequestProfile,
)


def _profile() -> RequestProfile:
    return RequestProfile(
        name="short",
        messages=[{"role": "user", "content": "never persist me"}],
        max_tokens=8,
    )


def _adapter() -> AdapterTarget:
    return AdapterTarget(
        name="adapter",
        kind="adapter",
        model="run@final." + "a" * 40,
        base_model="model-a",
        checkpoint="run",
        adapter_revision="run@final." + "a" * 40,
        hf_revision="a" * 40,
    )


def _sse(*values: Any, done: bool = True) -> bytes:
    chunks = [f"data: {json.dumps(value)}\n\n" for value in values]
    if done:
        chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode()


def _success_headers() -> dict[str, str]:
    target = _adapter()
    return {
        "content-type": "text/event-stream",
        "X-Freesolo-Checkpoint": target.checkpoint,
        "X-Freesolo-Adapter-Revision": target.adapter_revision,
        "X-Freesolo-HF-Revision": target.hf_revision,
    }


def _success_body(*, usage: bool = True, done: bool = True) -> bytes:
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "answer"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    if usage:
        chunks[-1]["usage"] = {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 1},
        }
    return _sse(*chunks, done=done)


def _run_chat(handler, target=None):
    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await stream_chat(
                client,
                "https://example.invalid",
                "credential-secret",
                target or _adapter(),
                _profile(),
                FakeClock(),
            )

    return asyncio.run(run())


def test_health_identity_capabilities_and_two_of_three_discovery() -> None:
    async def run():
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "accounting_ok": True,
                    "deployment_sha": "94210a3",
                    "deployment_id": "deploy-1",
                    "capabilities": ["permanent_checkpoint_identity", "other"],
                    "base_models": ["model-a", "model-b", "model-c"],
                    "gpus": 3,
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await get_health(
                client,
                "https://example.invalid",
                DeploymentExpectation(sha="94210a3", deployment_id="deploy-1"),
                ["permanent_checkpoint_identity"],
            )

    health = asyncio.run(run())
    assert resolve_discovered_models(
        health.base_models,
        ["model-a", "model-b", "model-c"],
        ["model-b"],
        ["model-a"],
    ) == ["model-a", "model-c"]
    with pytest.raises(ProtocolError, match="required base models"):
        resolve_discovered_models(health.base_models, [], [], ["model-z"])


def _health_body(**overrides) -> dict:
    body = {
        "ok": True,
        "deployment_sha": "94210a3",
        "deployment_id": "deploy-1",
        "capabilities": ["permanent_checkpoint_identity"],
        "base_models": ["model-a"],
    }
    body.update(overrides)
    return body


def _get_health(body: dict):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=body))
        ) as client:
            return await get_health(
                client,
                "https://example.invalid",
                DeploymentExpectation(sha="94210a3", deployment_id="deploy-1"),
                ["permanent_checkpoint_identity"],
            )

    return asyncio.run(run())


def test_absent_accounting_field_is_not_read_as_an_accounting_failure() -> None:
    """dev's health body omits accounting_ok; a missing field is not a reported failure.

    treating absence as False would make every dev deployment permanently unusable, so the
    harness distinguishes not-reported from reported-unhealthy.
    """
    health = _get_health(_health_body())
    assert health.accounting_ok is None
    assert health.ok is True


def test_explicitly_unhealthy_accounting_still_stops_the_run() -> None:
    with pytest.raises(ProtocolError, match="accounting"):
        _get_health(_health_body(accounting_ok=False))


def test_undeclared_provenance_headers_are_not_asserted() -> None:
    """a target that declares no adapter revision verifies checkpoint identity alone."""
    target = AdapterTarget(
        name="adapter",
        kind="adapter",
        model="adapter-a@run",
        base_model="model-a",
        checkpoint="run",
    )
    observation = _run_chat(
        lambda _request: httpx.Response(
            200,
            headers={"X-Freesolo-Checkpoint": "run"},
            content=_success_body(),
        ),
        target=target,
    )
    assert observation.outcome == "success"


def test_declared_provenance_header_missing_from_response_is_a_mismatch() -> None:
    """declaring a header asserts the deployment emits it, so an omission must fail."""
    target = AdapterTarget(
        name="adapter",
        kind="adapter",
        model="adapter-a@run",
        base_model="model-a",
        checkpoint="run",
        hf_revision="a" * 40,
    )
    observation = _run_chat(
        lambda _request: httpx.Response(
            200,
            headers={"X-Freesolo-Checkpoint": "run"},
            content=_success_body(),
        ),
        target=target,
    )
    assert observation.outcome == "protocol_error"
    assert "hf-revision" in observation.error_detail


def test_server_error_prose_is_never_persisted_in_the_observation() -> None:
    """an error body can echo the prompt, so only a machine code shape survives."""
    observation = _run_chat(
        lambda _request: httpx.Response(
            503,
            headers={"Retry-After": "1"},
            json={"error": {"code": "secret prompt leaked here", "message": "secret prompt"}},
        )
    )
    assert "secret prompt" not in (observation.error_detail or "")
    assert "unrecognized_code" in (observation.error_detail or "")


def test_strict_sse_tracks_reasoning_and_visible_content_separately() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["X-Freesolo-Expected-Checkpoint"] == "run"
        payload = json.loads(request.content)
        assert payload["stream_options"] == {"include_usage": True}
        return httpx.Response(200, headers=_success_headers(), content=_success_body())

    observation = _run_chat(handler)
    assert observation.outcome == "success"
    assert observation.first_generated_ns is not None
    assert observation.first_visible_ns is not None
    assert observation.finish_reasons == ["stop"]
    assert observation.done_count == 1
    assert observation.prompt_tokens == 4
    assert observation.completion_tokens == 2
    assert len(requests) == 1


def test_role_only_opening_chunk_does_not_count_as_the_first_generated_token() -> None:
    """an empty content string is not a token, so it must not stamp ttft.

    openai-compatible servers open a stream with a role delta carrying ``content: ""``. keying
    ttft on the presence of the key rather than on generated text would record the first token at
    header time and report a time-to-first-token the server never achieved.
    """

    class TickingClock(FakeClock):
        def monotonic_ns(self) -> int:
            self.advance_ns(1_000_000)
            return super().monotonic_ns()

    def handler(_request: httpx.Request) -> httpx.Response:
        body = _sse(
            {"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
            {"choices": [{"delta": {"reasoning_content": ""}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 1},
                },
            },
        )
        return httpx.Response(200, headers=_success_headers(), content=body)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await stream_chat(
                client,
                "https://example.invalid",
                "credential-secret",
                _adapter(),
                _profile(),
                TickingClock(),
            )

    observation = asyncio.run(run())
    assert observation.outcome == "success"
    # the empty deltas arrive before the real one, so a key-presence check would stamp earlier
    assert observation.first_generated_ns == observation.first_visible_ns


def test_missing_usage_makes_token_throughput_unavailable_without_estimate() -> None:
    observation = _run_chat(
        lambda _request: httpx.Response(
            200, headers=_success_headers(), content=_success_body(usage=False)
        )
    )
    assert observation.outcome == "success"
    assert observation.prompt_tokens is None
    assert observation.completion_tokens is None


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (_success_body(done=False), "exactly one"),
        (b"data: not-json\n\n", "valid json"),
        (
            _sse({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}),
            "finish reasons",
        ),
        (_success_body() + b"data: {}\n\n", "after [done]"),
    ],
)
def test_strict_sse_rejects_malformed_terminal_contract(content: bytes, match: str) -> None:
    observation = _run_chat(
        lambda _request: httpx.Response(200, headers=_success_headers(), content=content)
    )
    assert observation.error_class == "protocol_error"
    assert match in observation.error_detail


def test_identity_and_immutable_provenance_mismatch_fail_closed() -> None:
    headers = _success_headers()
    headers["X-Freesolo-HF-Revision"] = "b" * 40
    observation = _run_chat(
        lambda _request: httpx.Response(200, headers=headers, content=_success_body())
    )
    assert observation.error_class == "protocol_error"
    assert "hf-revision" in observation.error_detail


def test_exact_capacity_other_503_and_429_are_separate_and_never_retried() -> None:
    cases = [
        (
            503,
            {"Retry-After": "1"},
            {"error": {"code": "serving_capacity_unavailable", "message": "busy"}},
            "exact_capacity_503",
        ),
        (503, {"Retry-After": "1"}, {"error": {"code": "accounting", "message": "x"}}, "other_503"),
        (429, {}, {"error": {"code": "rate_limit", "message": "x"}}, "http_429"),
    ]
    for status, headers, body, expected in cases:
        calls = 0

        def handler(
            _request: httpx.Request,
            response_status: int = status,
            response_headers: dict[str, str] = headers,
            response_body: dict = body,
        ) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(response_status, headers=response_headers, json=response_body)

        observation = _run_chat(handler, BaseTarget(name="base", model="model-a"))
        assert observation.error_class == expected
        assert observation.retry_count == 0
        assert calls == 1


def test_observation_excludes_secrets_prompts_generated_text_and_raw_bodies() -> None:
    observation = _run_chat(
        lambda _request: httpx.Response(
            503,
            headers={"Authorization": "leak", "Retry-After": "1"},
            json={
                "error": {
                    "code": "serving_capacity_unavailable",
                    "message": "bounded detail",
                }
            },
        ),
        BaseTarget(name="base", model="model-a"),
    )
    serialized = json.dumps(asdict(observation))
    assert "credential-secret" not in serialized
    assert "never persist me" not in serialized
    assert "answer" not in serialized
    assert "authorization" not in serialized.lower()
    assert observation.response_headers == {
        "content-type": "application/json",
        "retry-after": "1",
    }

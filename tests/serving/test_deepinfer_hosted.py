"""Exercise the real router, private HTTP bridge, and accounting with a fake native engine."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from flash.serving.deepinfer.app.client import CONNECTION_KEYS, HostedDeepInfer, configured
from flash.serving.deepinfer.app.http import build_app
from flash.serving.deepinfer.contract.config import MODEL_ID
from flash.serving.deepinfer.contract.errors import DeepInferError
from flash.serving.src.engine.model_config import base_models, hosted_base_models
from flash.serving.src.http.rate_limit import allow_offline_inference
from flash.serving.src.http.router import AdapterRouter, build_serving_app
from flash.serving.src.io.openai_request import OpenAIGenerateRequest
from flash.serving.src.io.schemas import AdapterRecord
from tests.serving.conftest import RecordingUsageStore
from tests.serving.test_deepinfer import FakeRuntime, body


@pytest.fixture
def connection(monkeypatch):
    for key, value in zip(CONNECTION_KEYS, ("key", "secret"), strict=True):
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        "flash.serving.deepinfer.app.client.endpoint_url", lambda: "https://deepinfer.modal.run"
    )


def record():
    return AdapterRecord(
        model_id=MODEL_ID,
        repo_id=MODEL_ID,
        base_model=MODEL_ID,
        serve_base_model=True,
        thinking=False,
    )


def app(runtime):
    transport = httpx.ASGITransport(app=build_app(runtime, manage_lifecycle=False))
    pool = HostedDeepInfer(transport=transport)
    store = RecordingUsageStore()

    async def authorize(*args):
        return "org-deepseek"

    return build_serving_app(
        pool,
        AdapterRouter([record()]),
        usage_store=store,
        chat_authorizer=authorize,
        rate_limiter=allow_offline_inference,
    ), store


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
def test_hosted_chat_bills_native_usage_at_requested_prices(connection, stream, thinking):
    events = [{"tokens": [4, 11, 2, 1] if thinking else [2, 3, 1], "finish_reason": "stop"}]
    runtime = FakeRuntime(events)
    router, store = app(runtime)
    payload = body(stream=stream, chat_template_kwargs={"enable_thinking": thinking})
    if stream:
        payload["stream_options"] = {"include_usage": True}
    with TestClient(router) as client:
        response = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer test"}, json=payload
        )
    assert response.status_code == 200, response.text
    if stream:
        assert response.text.endswith("data: [DONE]\n\n")
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        result_usage = chunks[-1]["usage"]
        content = "".join(
            chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk["choices"]
        )
    else:
        result_usage = response.json()["usage"]
        content = response.json()["choices"][0]["message"]["content"]
    assert content == ("Hi" if thinking else "Hi there")
    assert result_usage["completion_tokens"] == (4 if thinking else 3)
    settled = store.finalized[-1]
    assert settled.principal.orgId == "org-deepseek"
    assert settled.facts.prompt_tokens == 2
    assert settled.facts.reasoning_tokens == int(thinking)
    assert settled.facts.cached_tokens == 0
    assert settled.facts.cached_tokens_reported is False
    assert settled.price.snapshot == {
        "prompt_token_usd": "0.00000015",
        "completion_token_usd": "0.0000006",
        "cached_prompt_token_usd": "0.000000003",
    }
    assert settled.price.version == "2026-10-23.deepseek-1017.1"
    assert runtime.closed


@pytest.mark.parametrize("stream", [False, True])
def test_hosted_chat_preserves_split_utf8_across_native_events(connection, stream):
    runtime = FakeRuntime(
        [
            {"tokens": [5], "finish_reason": None},
            {"tokens": [6, 1], "finish_reason": "stop"},
        ]
    )
    router, store = app(runtime)
    with TestClient(router) as client:
        response = client.post(
            "/v1/chat/completions",
            json=body(stream=stream),
            headers={"Authorization": "Bearer test"},
        )
    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        content = "".join(
            chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk["choices"]
        )
    else:
        content = response.json()["choices"][0]["message"]["content"]
    assert content == "€"
    assert store.finalized[-1].facts.completion_tokens == 3
    assert runtime.closed


@pytest.mark.parametrize("stream", [False, True])
def test_hosted_capacity_error_precedes_response_and_settlement(connection, stream):
    runtime = FakeRuntime([OverflowError()])
    router, store = app(runtime)
    with TestClient(router) as client:
        response = client.post(
            "/v1/chat/completions",
            json=body(stream=stream),
            headers={"Authorization": "Bearer test"},
        )
    assert response.status_code == 429
    assert not store.finalized
    assert runtime.closed


def test_auth_and_unsupported_controls_fail_before_native_work(connection):
    runtime = FakeRuntime()
    router, store = app(runtime)
    with TestClient(router) as client:
        assert client.post("/v1/chat/completions", json=body()).status_code == 401
        response = client.post(
            "/v1/chat/completions",
            json=body(stop=["halt"]),
            headers={"Authorization": "Bearer test"},
        )
        assert response.status_code == 400
    assert runtime.calls == 0
    assert not store.finalized


def test_native_failure_never_settles_success(connection):
    runtime = FakeRuntime([{"tokens": [2], "finish_reason": None}, RuntimeError("rank died")])
    router, store = app(runtime)
    with TestClient(router) as client:
        response = client.post(
            "/v1/chat/completions", json=body(stream=True), headers={"Authorization": "Bearer test"}
        )
    chunks = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")
    ]
    assert chunks[-1]["choices"][0]["finish_reason"] == "error"
    assert store.failed
    assert not store.finalized
    assert runtime.closed


def test_catalog_requires_complete_authenticated_connection(monkeypatch):
    for key in CONNECTION_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert hosted_base_models() == base_models()
    monkeypatch.setenv("DEEPINFER_PROXY_KEY", "key")
    with pytest.raises(ValueError, match="both Modal proxy credentials"):
        configured()
    monkeypatch.setenv("DEEPINFER_PROXY_KEY", "key")
    monkeypatch.setenv("DEEPINFER_PROXY_SECRET", "secret")
    assert MODEL_ID in hosted_base_models()


@pytest.mark.parametrize("fault", ["truncated", "wrong-id", "error"])
def test_bridge_rejects_incomplete_or_misattributed_results(connection, fault):
    async def handler(request):
        assert request.headers["Modal-Key"] == "key"
        assert request.headers["Modal-Secret"] == "secret"
        event = {"type": "ready", "request_id": "fsgen-test"}
        if fault == "wrong-id":
            event = {"type": "final", "request_id": "wrong"}
        if fault == "error":
            event = {"type": "error"}
        return httpx.Response(200, text=json.dumps(event) + "\n")

    pool = HostedDeepInfer(transport=httpx.MockTransport(handler))
    payload = OpenAIGenerateRequest(
        model_id=MODEL_ID, messages=body()["messages"], generation_id="fsgen-test"
    )
    with pytest.raises(DeepInferError) as caught:
        asyncio.run(pool.generate(MODEL_ID, payload, record()))
    assert caught.value.status_code == 502


@pytest.mark.parametrize("cross_origin", [False, True])
def test_cold_start_redirect_polls_result_without_resubmission_or_credential_leak(
    connection, cross_origin
):
    calls = []

    async def handler(request):
        calls.append(request)
        if request.method == "POST":
            location = (
                "https://other.example/result" if cross_origin else "/internal/generate?result=one"
            )
            return httpx.Response(303, headers={"location": location})
        assert request.method == "GET"
        return httpx.Response(
            200, text=json.dumps({"type": "final", "request_id": "fsgen-test"}) + "\n"
        )

    pool = HostedDeepInfer(transport=httpx.MockTransport(handler))
    payload = OpenAIGenerateRequest(
        model_id=MODEL_ID, messages=body()["messages"], generation_id="fsgen-test"
    )
    if cross_origin:
        with pytest.raises(DeepInferError, match="unsafe result redirect"):
            asyncio.run(pool.generate(MODEL_ID, payload, record()))
        assert len(calls) == 1
    else:
        assert asyncio.run(pool.generate(MODEL_ID, payload, record()))["type"] == "final"
        assert len(calls) == 2


@pytest.mark.parametrize("stream", [False, True])
def test_native_prefix_usage_is_billed_and_namespace_is_authoritative(connection, stream):
    runtime = FakeRuntime(
        [
            {
                "tokens": [2, 1],
                "finish_reason": "stop",
                "cached_tokens": 1,
                "cached_tokens_reported": True,
            }
        ]
    )
    seen = []
    original = runtime.prepare_tokens

    async def prepare(request):
        seen.append(request.cache_namespace)
        return await original(request)

    runtime.prepare_tokens = prepare
    router, store = app(runtime)
    with TestClient(router) as client:
        response = client.post(
            "/v1/chat/completions",
            json=body(stream=stream),
            headers={"Authorization": "Bearer test"},
        )
    assert response.status_code == 200
    assert len(seen[0]) == 64
    assert store.finalized[-1].facts.cached_tokens == 1
    assert store.finalized[-1].facts.cached_tokens_reported is True


@pytest.mark.parametrize(
    "endpoint", ["https://backend.modal.run", None, "http://backend", "https://user:pass@backend"]
)
def test_endpoint_is_discovered_from_fixed_same_environment_backend(monkeypatch, endpoint):
    from types import SimpleNamespace

    import modal

    from flash.serving.deepinfer.app.client import endpoint_url

    calls = []

    def lookup(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(get_web_url=lambda: endpoint)

    endpoint_url.cache_clear()
    monkeypatch.setattr(modal.Function, "from_name", lookup)
    try:
        if endpoint == "https://backend.modal.run":
            assert endpoint_url() == endpoint
            assert endpoint_url() == endpoint
        else:
            with pytest.raises(DeepInferError, match="valid HTTPS endpoint"):
                endpoint_url()
        assert calls == [(("freesolo-deepseek-serving", "serve"), {})]
    finally:
        endpoint_url.cache_clear()


def test_bridge_maps_endpoint_discovery_failure_to_unavailable(connection, monkeypatch):
    def fail():
        raise RuntimeError("Modal lookup failed")

    monkeypatch.setattr("flash.serving.deepinfer.app.client.endpoint_url", fail)
    payload = OpenAIGenerateRequest(
        model_id=MODEL_ID, messages=body()["messages"], generation_id="fsgen-test"
    )
    with pytest.raises(DeepInferError) as caught:
        asyncio.run(HostedDeepInfer().generate(MODEL_ID, payload, record()))
    assert caught.value.status_code == 503

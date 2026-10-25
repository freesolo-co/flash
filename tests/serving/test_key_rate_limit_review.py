"""Validation feedback and image admission ordering for keys with and without TPM overrides."""

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from flash.serving.deepinfer.app.client import HostedDeepInfer
from flash.serving.deepinfer.app.http import build_app
from flash.serving.src.accounting.usage import AuthorizedTraffic, principal_for_external_org
from flash.serving.src.engine.token_budget import token_budget
from flash.serving.src.http.router import AdapterRouter, build_offline_serving_app
from flash.serving.src.io.schemas import internal_adapter_payload
from tests.serving import test_key_rate_limit as key_tests
from tests.serving.checkpoint_fixtures import checkpoint_record
from tests.serving.test_deepinfer import FakeRuntime
from tests.serving.test_deepinfer_hosted import record
from tests.serving.test_key_rate_limit import ALLOWED, DENIED, KEY, PREFLIGHT
from tests.serving.test_rate_limit import BASE, ORG, REQUESTS
from tests.serving.test_router import FakePool


@pytest.fixture
def limiter_factory(monkeypatch):
    # Keep the quota transport separate from the real private token-counting transport.
    def build(handler):
        with monkeypatch.context() as scoped:
            return key_tests.limiter_factory.__wrapped__(scoped)(handler)

    return build


@pytest.fixture
def connection(monkeypatch):
    return key_tests.connection.__wrapped__(monkeypatch)


@pytest.mark.parametrize("mode", ["org", "key_without_override", "key_with_override"])
@pytest.mark.parametrize(("path", "base_payload"), REQUESTS)
def test_exhausted_quota_prevents_image_decoding(
    mode, path, base_payload, limiter_factory, monkeypatch
):
    def reject_decode(*args, **kwargs):
        pytest.fail("over-limit request reached image decoding")

    monkeypatch.setattr("flash.serving.src.io.multimodal._decode_images", reject_decode)
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        if mode == "key_with_override":
            result = PREFLIGHT if len(seen) == 1 else DENIED
        else:
            result = dict(DENIED, rpm_limit=600, remaining=0, tpm_limit=None, tokens_remaining=None)
        return httpx.Response(200, json=[result])

    class Pool(FakePool):
        async def token_budget(self, base, payload, target):
            assert mode == "key_with_override"
            owner = SimpleNamespace(base_model=base, _thinking_default=lambda *args: False)
            return await token_budget(owner, payload.model_dump(), internal_adapter_payload(target))

    async def authorize(*args):
        return AuthorizedTraffic(
            principal_for_external_org(ORG), "generation-a", None if mode == "org" else KEY
        )

    pool = Pool()
    app = build_offline_serving_app(
        pool,
        AdapterRouter([checkpoint_record("qa", BASE, org_id=ORG, thinking=False)]),
        chat_authorizer=authorize,
        rate_limiter=limiter_factory(handle),
    )
    payload = {k: v for k, v in base_payload.items() if k != "prompt"}
    payload["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW52YWxpZA=="}}
            ],
        }
    ]
    with TestClient(app) as client:
        response = client.post(path, json=payload, headers={"Authorization": "Bearer valid"})
    assert response.status_code == 429
    assert len(seen) == (2 if mode == "key_with_override" else 1)
    assert pool.generated == []


@pytest.mark.parametrize(
    "failure",
    [
        "context_length",
        "unsupported_control",
        "timeout",
        "malformed_budget",
        "malformed_validation",
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_token_counting_preserves_validation_but_fails_closed_on_infrastructure(
    failure,
    stream,
    limiter_factory,
    connection,
):
    class Runtime(FakeRuntime):
        async def prepare_tokens(self, request):
            raise ValueError("prompt plus output exceeds the serving context")

    runtime = Runtime()
    if failure in {"context_length", "unsupported_control"}:
        transport = httpx.ASGITransport(app=build_app(runtime, manage_lifecycle=False))
    else:

        def handle(request):
            if failure == "timeout":
                raise httpx.ReadTimeout("backend unavailable", request=request)
            if failure == "malformed_validation":
                return httpx.Response(422, json={"unexpected": "body"})
            return httpx.Response(200, content=b"not json")

        transport = httpx.MockTransport(handle)
    deepinfer = HostedDeepInfer(transport=transport)
    seen = []

    def quota(request):
        seen.append(request)
        return httpx.Response(200, json=[PREFLIGHT])

    class Pool(FakePool):
        async def token_budget(self, base, payload, target):
            return await deepinfer.token_budget(payload)

    async def authorize(*args):
        return AuthorizedTraffic(principal_for_external_org(ORG), api_key_id=KEY)

    pool = Pool()
    app = build_offline_serving_app(
        pool,
        AdapterRouter([record()]),
        chat_authorizer=authorize,
        rate_limiter=limiter_factory(quota),
    )
    body = {
        "model": record().model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    }
    if failure == "unsupported_control":
        body["stop"] = "END"
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer valid"}
        )
    expected = {
        "context_length": (422, "prompt plus output exceeds the serving context"),
        "unsupported_control": (400, "DeepInfer does not support stop"),
    }
    if failure in expected:
        code, detail = expected[failure]
        assert response.status_code == code
        assert response.json()["detail"] == detail
        assert "retry-after" not in response.headers
    else:
        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"
    assert len(seen) == 1
    assert runtime.calls == 0
    assert pool.generated == []


def test_invalid_images_consume_admission_before_validation(limiter_factory, monkeypatch):
    events = []

    def quota(request):
        events.append("admit")
        return httpx.Response(200, json=[dict(ALLOWED, tpm_limit=None, tokens_remaining=None)])

    def decode(*args, **kwargs):
        from flash.serving.src.io.multimodal import MultimodalRequestError

        events.append("decode")
        raise MultimodalRequestError("invalid image")

    monkeypatch.setattr("flash.serving.src.io.multimodal._decode_images", decode)

    async def authorize(*args):
        return AuthorizedTraffic(principal_for_external_org(ORG), "generation-a", KEY)

    app = build_offline_serving_app(
        FakePool(),
        AdapterRouter([checkpoint_record("qa", BASE, org_id=ORG, thinking=False)]),
        chat_authorizer=authorize,
        rate_limiter=limiter_factory(quota),
    )
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "model_id": "qa/final",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,aW52YWxpZA=="},
                            }
                        ],
                    }
                ],
            },
            headers={"Authorization": "Bearer valid"},
        )
    assert response.status_code == 400
    assert events == ["admit", "decode"]


def test_text_is_normalized_before_token_counting(limiter_factory):
    seen = []

    def quota(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=[PREFLIGHT if len(seen) == 1 else ALLOWED])

    class Pool(FakePool):
        async def token_budget(self, base, payload, target):
            assert payload.messages == [{"role": "system", "content": "Be concise"}]
            return 300

    async def authorize(*args):
        return AuthorizedTraffic(principal_for_external_org(ORG), "generation-a", KEY)

    app = build_offline_serving_app(
        Pool(),
        AdapterRouter([checkpoint_record("qa", BASE, org_id=ORG, thinking=False)]),
        chat_authorizer=authorize,
        rate_limiter=limiter_factory(quota),
    )
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "model_id": "qa/final",
                "messages": [{"role": "developer", "content": "Be concise"}],
            },
            headers={"Authorization": "Bearer valid"},
        )
    assert response.status_code == 200
    assert seen[-1]["p_tokens"] == 300

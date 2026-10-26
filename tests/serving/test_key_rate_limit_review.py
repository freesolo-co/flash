"""Validation feedback and image admission ordering for keys with and without TPM overrides."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from flash.serving.src.accounting.usage import AuthorizedTraffic, principal_for_external_org
from flash.serving.src.http.router import AdapterRouter, build_offline_serving_app
from tests.serving import test_key_rate_limit as key_tests
from tests.serving.checkpoint_fixtures import checkpoint_record
from tests.serving.test_key_rate_limit import ALLOWED, DENIED, KEY
from tests.serving.test_rate_limit import BASE, ORG, REQUESTS
from tests.serving.test_router import FakePool


@pytest.fixture
def limiter_factory(monkeypatch):
    def build(handler):
        with monkeypatch.context() as scoped:
            return key_tests.limiter_factory.__wrapped__(scoped)(handler)

    return build


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
            result = DENIED
        else:
            result = dict(DENIED, rpm_limit=600, remaining=0, tpm_limit=None, tokens_remaining=None)
        return httpx.Response(200, json=[result])

    class Pool(FakePool):
        async def token_budget(self, *args):
            pytest.fail("admission must not tokenize or decode images")

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
    assert len(seen) == 1
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


def test_text_generation_does_not_require_token_counting(limiter_factory):
    seen = []

    def quota(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=[ALLOWED])

    class Pool(FakePool):
        async def token_budget(self, *args):
            pytest.fail("admission must not tokenize")

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
    assert seen == [{"p_org_id": ORG, "p_api_key_id": KEY}]

"""Key identity and admission against recorded usage without token reservations."""

import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from flash.serving.app.modal_app import _authorize_response
from flash.serving.src.accounting.usage import AuthorizedTraffic, principal_for_external_org
from flash.serving.src.http.router import AdapterRouter, build_offline_serving_app
from tests.serving import test_deepinfer_hosted as deepinfer_tests
from tests.serving import test_rate_limit as rate_tests
from tests.serving.checkpoint_fixtures import checkpoint_record
from tests.serving.test_rate_limit import BASE, ORG, REQUESTS
from tests.serving.test_router import FakePool


@pytest.fixture
def limiter_factory(monkeypatch):
    return rate_tests.limiter_factory.__wrapped__(monkeypatch)


@pytest.fixture
def connection(monkeypatch):
    return deepinfer_tests.connection.__wrapped__(monkeypatch)


KEY = "20000000-0000-4000-8000-000000000001"
PREFLIGHT = {
    "allowed": False,
    "rpm_limit": 10000,
    "remaining": 10000,
    "retry_after": 1,
    "tpm_limit": 30000000,
    "tokens_remaining": 30000000,
    "requires_tokens": True,
}
ALLOWED = dict(
    PREFLIGHT, allowed=True, remaining=9999, tokens_remaining=29999700, requires_tokens=False
)
DENIED = dict(ALLOWED, allowed=False, tokens_remaining=0)


@pytest.mark.parametrize(("path", "payload"), REQUESTS)
def test_key_tpm_rejects_before_generation_on_all_routes(path, payload, limiter_factory):
    seen = []

    class Pool(FakePool):
        async def token_budget(self, *args):
            pytest.fail("admission must not tokenize")

    pool = Pool()

    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=[DENIED])

    async def authorize(*args):
        return AuthorizedTraffic(principal_for_external_org(ORG), "generation-a", KEY)

    app = build_offline_serving_app(
        pool,
        AdapterRouter([checkpoint_record("qa", BASE, org_id=ORG, thinking=False)]),
        chat_authorizer=authorize,
        rate_limiter=limiter_factory(handle),
    )
    with TestClient(app) as client:
        response = client.post(
            path,
            json=payload,
            headers={
                "Authorization": "Bearer valid",
                "X-Freesolo-Api-Key-Id": "spoofed",
            },
        )
    assert response.status_code == 429
    assert response.headers["x-ratelimit-limit-tokens"] == "30000000"
    assert response.headers["x-ratelimit-remaining"] == "9999"
    assert seen == [
        {"p_org_id": ORG, "p_api_key_id": KEY},
    ]
    assert pool.generated == []


def test_key_without_override_does_not_tokenize(limiter_factory):
    async def run():
        limiter = limiter_factory(
            lambda _: httpx.Response(
                200,
                json=[
                    {
                        "allowed": True,
                        "rpm_limit": 600,
                        "remaining": 599,
                        "retry_after": 17,
                        "tpm_limit": None,
                        "tokens_remaining": None,
                        "requires_tokens": False,
                    }
                ],
            )
        )
        try:
            await limiter(ORG, api_key_id=KEY)
        finally:
            await limiter.aclose()

    asyncio.run(run())


def test_old_reservation_rpc_fails_closed_without_tokenizing(limiter_factory):
    async def run():
        limiter = limiter_factory(lambda _: httpx.Response(200, json=[PREFLIGHT]))
        try:
            with pytest.raises(HTTPException) as exc:
                await limiter(ORG, api_key_id=KEY)
            assert exc.value.status_code == 503
        finally:
            await limiter.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {**PREFLIGHT, "allowed": True},
        {**ALLOWED, "tpm_limit": True},
        {**ALLOWED, "tokens_remaining": -1},
        {**ALLOWED, "requires_tokens": "false"},
        {**ALLOWED, "remaining": 10000},
    ],
)
def test_malformed_key_rpc_fails_closed(bad, limiter_factory):
    async def run():
        limiter = limiter_factory(lambda _: httpx.Response(200, json=[bad]))
        try:
            with pytest.raises(HTTPException) as exc:
                await limiter(ORG, api_key_id=KEY)
            assert exc.value.status_code == 503
        finally:
            await limiter.aclose()

    asyncio.run(run())


def test_ambiguous_admission_is_not_retried(limiter_factory):
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("lost response", request=request)

    async def run():
        limiter = limiter_factory(handle)
        try:
            with pytest.raises(HTTPException) as exc:
                await limiter(ORG, api_key_id=KEY)
            assert exc.value.status_code == 503
            assert len(calls) == 1
        finally:
            await limiter.aclose()

    asyncio.run(run())


def test_authentication_supplies_validated_key_id():
    response = httpx.Response(200, json={"orgId": ORG, "apiKeyId": KEY})
    assert _authorize_response(response, BASE, require_api_key_id=True).api_key_id == KEY
    for body in ({"orgId": ORG}, {"orgId": ORG, "apiKeyId": "spoofed"}):
        with pytest.raises(HTTPException) as exc:
            _authorize_response(httpx.Response(200, json=body), BASE, require_api_key_id=True)
        assert exc.value.status_code == 503

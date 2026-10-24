"""Admission rejects requests before generation, across every hosted inference route."""

import asyncio
import json
from collections import Counter

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from flash.serving.src.accounting.usage import AuthorizedTraffic, principal_for_external_org
from flash.serving.src.http.rate_limit import SupabaseRateLimiter
from flash.serving.src.http.router import AdapterRouter, build_offline_serving_app
from flash.serving.src.store.settings import Settings
from tests.serving.checkpoint_fixtures import checkpoint_record
from tests.serving.test_router import FakePool

ORG = "10000000-0000-4000-8000-000000000001"
OTHER_ORG = "10000000-0000-4000-8000-000000000002"
MODEL = "qa/final"
BASE = "Qwen/Qwen3.5-9B"
ALLOWED = {"allowed": True, "rpm_limit": 600, "remaining": 599, "retry_after": 17}
DENIED = {"allowed": False, "rpm_limit": 600, "remaining": 0, "retry_after": 17}


@pytest.fixture
def limiter_factory(monkeypatch):
    client_class = httpx.AsyncClient

    def build(handler):
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs),
        )
        return SupabaseRateLimiter(
            Settings(
                supabase_url="https://rate-limit.test", supabase_service_role_key="sb_secret_test"
            )
        )

    return build


def test_rpc_contract_and_denial_headers(limiter_factory):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=[ALLOWED if len(seen) == 1 else DENIED])

    async def run():
        limiter = limiter_factory(handle)
        try:
            await limiter(ORG)
            with pytest.raises(HTTPException) as caught:
                await limiter(ORG)
            assert caught.value.status_code == 429
            assert caught.value.headers == {
                "Retry-After": "17",
                "X-RateLimit-Limit": "600",
                "X-RateLimit-Remaining": "0",
            }
            assert caught.value.detail["code"] == "rate_limit_exceeded"
            assert len(seen) == 2
            for request in seen:
                assert request.url.path == "/rest/v1/rpc/check_serving_rate_limit"
                assert json.loads(request.content) == {"p_org_id": ORG}
                assert request.headers["Content-Profile"] == "public"
                assert request.headers["apikey"] == "sb_secret_test"
        finally:
            await limiter.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        [],
        [ALLOWED, ALLOWED],
        [None],
        [{}],
        [{**ALLOWED, "allowed": "true"}],
        [{**ALLOWED, "rpm_limit": True}],
        [{**ALLOWED, "remaining": -1}],
        [{**ALLOWED, "retry_after": 0}],
        [{**ALLOWED, "retry_after": 61}],
        [{**DENIED, "remaining": 1}],
    ],
)
def test_malformed_rpc_fails_closed(body, limiter_factory):
    async def run():
        limiter = limiter_factory(lambda _: httpx.Response(200, json=body))
        try:
            with pytest.raises(HTTPException) as caught:
                await limiter(ORG)
            assert caught.value.status_code == 503
            assert caught.value.headers == {"Retry-After": "5"}
        finally:
            await limiter.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["timeout", "http", "json"])
def test_backend_failure_is_not_retried(failure, limiter_factory):
    calls = []

    def handle(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("unavailable", request=request)
        if failure == "http":
            return httpx.Response(500)
        return httpx.Response(200, content=b"not JSON")

    async def run():
        limiter = limiter_factory(handle)
        try:
            with pytest.raises(HTTPException) as caught:
                await limiter(ORG)
            assert caught.value.status_code == 503
            assert len(calls) == 1
        finally:
            await limiter.aclose()

    asyncio.run(run())


async def _authorize(token, model_id, scope):
    if token == "invalid":
        raise HTTPException(403, "denied")
    org = OTHER_ORG if token == "other-org" else ORG
    return AuthorizedTraffic(principal_for_external_org(org), "generation-a")


def _app(limiter, pool):
    return build_offline_serving_app(
        pool,
        AdapterRouter([checkpoint_record("qa", BASE, org_id=ORG, thinking=False)]),
        chat_authorizer=_authorize,
        internal_key="internal",
        rate_limiter=limiter,
    )


REQUESTS = [
    ("/generate", {"model_id": MODEL, "prompt": "hi"}),
    (f"/adapters/{MODEL}/generate", {"prompt": "hi"}),
    ("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
    (
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ),
]


@pytest.mark.parametrize(("path", "payload"), REQUESTS)
def test_every_endpoint_rejects_before_gpu_dispatch(path, payload, limiter_factory):
    limiter = limiter_factory(lambda _: httpx.Response(200, json=[DENIED]))
    pool = FakePool()
    with TestClient(_app(limiter, pool)) as client:
        response = client.post(path, json=payload, headers={"Authorization": "Bearer valid"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "17"
    assert "text/event-stream" not in response.headers["content-type"]
    assert pool.generated == []
    assert limiter._client.is_closed


def test_endpoints_keys_and_replicas_share_org_allowance(limiter_factory):
    counts = Counter()

    def handle(request):
        org = json.loads(request.content)["p_org_id"]
        counts[org] += 1
        return httpx.Response(200, json=[ALLOWED if counts[org] <= 2 else DENIED])

    pool = FakePool()
    with (
        TestClient(_app(limiter_factory(handle), pool)) as first,
        TestClient(_app(limiter_factory(handle), pool)) as second,
    ):
        path, body = REQUESTS[0]
        assert (
            first.post(path, json=body, headers={"Authorization": "Bearer key-one"}).status_code
            == 200
        )
        path, body = REQUESTS[1]
        assert (
            second.post(path, json=body, headers={"Authorization": "Bearer key-two"}).status_code
            == 200
        )
        for path, body in REQUESTS[2:]:
            response = first.post(
                path,
                json=body,
                headers={
                    "Authorization": "Bearer key-three",
                    "X-Freesolo-Org-Id": OTHER_ORG,
                },
            )
            assert response.status_code == 429
        # Trusted product proxies must consume the same org allowance.
        response = second.post(
            REQUESTS[0][0],
            json=REQUESTS[0][1],
            headers={
                "X-Freesolo-Internal-Key": "internal",
                "X-Freesolo-Org-Id": ORG,
            },
        )
        assert response.status_code == 429
        assert first.get("/healthz").status_code == 200
    assert set(counts) == {ORG}
    assert len(pool.generated) == 2


def test_auth_denials_do_not_consume_quota(limiter_factory):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=[ALLOWED])

    with TestClient(_app(limiter_factory(handle), FakePool())) as client:
        path, body = REQUESTS[0]
        assert client.post(path, json=body).status_code == 401
        assert (
            client.post(path, json=body, headers={"Authorization": "Bearer invalid"}).status_code
            == 403
        )
    assert seen == []

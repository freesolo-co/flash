"""A healthy candidate can coexist with an older router still accepting chat inputs."""

import asyncio

import httpx
import pytest

from flash.serving.promotion import canary
from flash.serving.promotion.evidence import verify_stream
from flash.serving.src.http.router import build_offline_serving_app
from tests.serving.test_promotion_canary import _request
from tests.serving.test_router import QWEN, FakePool, _router_for


def _app(sha, deployment_id):
    return build_offline_serving_app(
        FakePool(),
        _router_for("qa", QWEN),
        internal_key=_request().api_key,
        deployment_sha=sha,
        deployment_id=deployment_id,
    )


@pytest.mark.parametrize(
    "old_identity", [None, ("old-sha", "old-run"), ("candidate-sha", "old-run")]
)
def test_canary_drains_old_router_then_proves_candidate_stream(old_identity, monkeypatch):
    monkeypatch.setattr(canary, "_RELEASE_RETRY_SECONDS", 0)
    calls, drained = [], []
    old = httpx.ASGITransport(app=_app(*(old_identity or ("old-sha", "old-run"))))
    current = httpx.ASGITransport(app=_app("candidate-sha", "12345-1"))

    async def respond(request):
        calls.append(request)
        response = await (old if len(calls) == 1 else current).handle_async_request(request)
        body = await response.aread()
        # Actual router SSE, including usage and finalization. Reproduce the first rollout from
        # a predecessor that does not yet stamp identity headers.
        if len(calls) == 1 and old_identity is None:
            del response.headers["x-freesolo-serving-release"]
            del response.headers["x-freesolo-serving-deployment-id"]
        drained.append(b"data: [DONE]" in body)
        return response

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await canary.run_stream_canary(
                _request(model="qa/final", org_id="org-1"), client=client
            )

    evidence = asyncio.run(run())
    assert verify_stream(evidence).ok
    assert drained == [True, True]
    assert len(calls) == 2
    assert {request.headers["x-correlation-id"] for request in calls} == {"fspromo-12345-1"}
    assert all(request.headers["connection"] == "close" for request in calls)


def test_permanently_stale_router_fails_with_bounded_attempts(monkeypatch):
    monkeypatch.setattr(canary, "_RELEASE_RETRY_SECONDS", 0)
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text="data: [DONE]\n\n"
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await canary.run_stream_canary(_request(), client=client)

    with pytest.raises(canary.CanaryError, match=f"^{canary.CANARY_RELEASE_MISMATCH}$"):
        asyncio.run(run())
    assert len(calls) == canary._MAX_RELEASE_ATTEMPTS


@pytest.mark.wallclock
def test_handoff_retry_uses_the_original_deadline(monkeypatch):
    # A long retry sleep must still be interrupted by the original overall timeout.
    monkeypatch.setattr(canary, "_RELEASE_RETRY_SECONDS", 60)
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text="data: [DONE]\n\n"
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await canary.run_stream_canary(_request(timeout_seconds=0.1), client=client)

    with pytest.raises(canary.CanaryError, match=f"^{canary.CANARY_TIMEOUT}$"):
        asyncio.run(run())
    assert len(calls) == 1

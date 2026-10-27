"""Pre-admission failover must never duplicate accepted or streamed work."""

import asyncio
import json

import httpx
import pytest

from flash.serving.deepinfer.app.client import HostedDeepInfer
from flash.serving.deepinfer.contract.config import MODEL_ID
from flash.serving.deepinfer.contract.errors import DeepInferError
from flash.serving.src.io.openai_request import OpenAIGenerateRequest
from tests.serving.test_deepinfer_hosted import record


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("DEEPINFER_PROXY_KEY", "key")
    monkeypatch.setenv("DEEPINFER_PROXY_SECRET", "secret")


def payload():
    return OpenAIGenerateRequest(
        model_id=MODEL_ID, messages=[{"role": "user", "content": "Hi"}], generation_id="fsgen-retry"
    )


def success():
    return httpx.Response(
        200, text=json.dumps({"type": "final", "request_id": "fsgen-retry"}) + "\n"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["connect", "capacity", "draining"])
async def test_failure_before_admission_uses_another_ready_replica(failure):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            if failure == "connect":
                raise httpx.ConnectError("not connected")
            return httpx.Response(
                429 if failure == "capacity" else 503,
                headers={"X-DeepInfer-Retryable": "not-started"},
            )
        return success()

    pool = HostedDeepInfer(
        transport=httpx.MockTransport(handler),
        targets=lambda: ["https://one.modal.direct", "https://two.modal.direct"],
    )
    try:
        await pool.generate(MODEL_ID, payload(), record())
    finally:
        await pool.aclose()
    assert [r.url.host for r in calls] == ["one.modal.direct", "two.modal.direct"]
    assert calls[0].content == calls[1].content  # Same authorized generation and tenant namespace.


@pytest.mark.anyio
async def test_single_replica_retries_are_bounded():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("not connected")

    pool = HostedDeepInfer(
        transport=httpx.MockTransport(handler), targets=lambda: ["https://one.modal.direct"]
    )
    try:
        with pytest.raises(DeepInferError, match="transport failed"):
            await pool.generate(MODEL_ID, payload(), record())
    finally:
        await pool.aclose()
    assert len(calls) == 3


@pytest.mark.anyio
async def test_connect_failure_fetching_accepted_result_never_replays_post():
    methods = []

    def handler(request):
        methods.append(request.method)
        if request.method == "POST":
            return httpx.Response(303, headers={"Location": "/result"})
        raise httpx.ConnectError("result connection failed after POST acceptance")

    pool = HostedDeepInfer(
        transport=httpx.MockTransport(handler),
        targets=lambda: ["https://one.modal.direct", "https://two.modal.direct"],
    )
    try:
        with pytest.raises(DeepInferError, match="transport failed"):
            await pool.generate(MODEL_ID, payload(), record())
    finally:
        await pool.aclose()
    assert methods == ["POST", "GET"]


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["read", "write", "503", "429", "partial"])
async def test_ambiguous_failure_or_started_stream_is_not_replayed(failure):
    calls = []

    def handler(request):
        calls.append(request)
        if failure == "read":
            raise httpx.ReadTimeout("request might have been accepted")
        if failure == "write":
            raise httpx.WriteError("request might have been accepted")
        if failure in {"503", "429"}:
            return httpx.Response(int(failure))
        return httpx.Response(200, text='data: {"type":"ready","request_id":"fsgen-retry"}\n\n')

    pool = HostedDeepInfer(
        transport=httpx.MockTransport(handler),
        targets=lambda: ["https://one.modal.direct", "https://two.modal.direct"],
    )
    try:
        with pytest.raises(DeepInferError):
            await pool.generate(MODEL_ID, payload(), record())
    finally:
        await pool.aclose()
    assert len(calls) == 1


@pytest.mark.anyio
async def test_transient_directory_failure_retries_before_sending_work():
    lookups, requests = [], []

    def targets():
        lookups.append(True)
        if len(lookups) == 1:
            raise RuntimeError("control plane temporarily unavailable")
        return ["https://one.modal.direct"]

    def handler(request):
        requests.append(request)
        return success()

    pool = HostedDeepInfer(transport=httpx.MockTransport(handler), targets=targets)
    try:
        await pool.generate(MODEL_ID, payload(), record())
    finally:
        await pool.aclose()
    assert len(lookups) == 2
    assert len(requests) == 1


@pytest.mark.anyio
async def test_cancellation_during_retry_does_not_start_another_generation():
    called = asyncio.Event()
    calls = []

    def handler(request):
        calls.append(request)
        called.set()
        return httpx.Response(503, headers={"X-DeepInfer-Retryable": "not-started"})

    pool = HostedDeepInfer(
        transport=httpx.MockTransport(handler), targets=lambda: ["https://one.modal.direct"]
    )
    task = asyncio.create_task(pool.generate(MODEL_ID, payload(), record()))
    await called.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await pool.aclose()
    assert len(calls) == 1

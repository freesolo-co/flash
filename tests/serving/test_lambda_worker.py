"""Private worker protocol, cancellation, and bootstrap credential boundaries."""

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from flash.serving.lambda_cloud.bootstrap import identity, user_data
from flash.serving.lambda_cloud.transport import WorkerError, WorkerTransport
from flash.serving.lambda_cloud.worker import create_app

MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def tls():
    return identity()


def fake_engine():
    engine = SimpleNamespace(
        base_model=MODEL,
        _load=AsyncMock(),
        _shutdown=AsyncMock(),
        _generate=AsyncMock(return_value={"text": "ok"}),
        _register=AsyncMock(return_value={"ok": True}),
        _unregister=AsyncMock(return_value={"ok": True}),
        _health=lambda: {"ok": True, "base_model": MODEL},
        engine=SimpleNamespace(abort=AsyncMock()),
    )

    async def stream(*args):
        yield {"type": "delta", "text": "ok"}
        yield {"type": "final", "usage": {"completion_tokens": 1}}

    engine._stream_generate = stream
    return engine


def test_worker_auth_lifecycle_and_exact_payload_forwarding():
    engine = fake_engine()
    body = {
        "payload": {"model_id": MODEL, "prompt": "test"},
        "record": {"model_id": MODEL},
        "expected_checkpoint": "checkpoint",
        "generation_id": "g1",
    }
    with TestClient(create_app(engine, "private")) as client:
        assert client.get("/health").status_code == 401
        headers = {"Authorization": "Bearer private"}
        assert client.get("/health", headers=headers).json()["base_model"] == MODEL
        assert client.post("/generate", json=body, headers=headers).json() == {"text": "ok"}
        engine._generate.assert_awaited_once_with(
            body["payload"], body["record"], "checkpoint", "g1"
        )
        response = client.post("/stream", json=body, headers=headers)
        assert [json.loads(line)["type"] for line in response.text.splitlines()] == [
            "delta",
            "final",
        ]
    engine._load.assert_awaited_once()
    engine._shutdown.assert_awaited_once()


def test_bootstrap_forwards_only_gpu_credentials_and_pins_source(tls):
    config = user_data(
        MODEL,
        "a" * 40,
        {
            "HF_TOKEN": "upstream",
            "LAMBDA_API_KEY": "provider-secret",
            "SUPABASE_SERVICE_ROLE_KEY": "database-secret",
            "FREESOLO_INTERNAL_KEY": "router-secret",
        },
        tls,
    )
    payload = json.loads(config.split("\n", 1)[1])
    files = {
        item["path"]: base64.b64decode(item["content"]).decode() for item in payload["write_files"]
    }
    content = "\n".join(files.values())
    assert "upstream" in content
    for secret in ("provider-secret", "database-secret", "router-secret"):
        assert secret not in content
    assert all(item["permissions"] == "0600" for item in payload["write_files"])
    assert "mountpoint -q /vol/hosting-cache" in files["/opt/flash/setup.sh"]
    assert "@sha256:" in content
    assert "a" * 40 in content
    assert "python -m flash.serving.lambda_cloud.worker" in content
    with pytest.raises(ValueError, match="exact source"):
        user_data(MODEL, "main", {}, tls)


@pytest.mark.anyio
async def test_transport_rejects_truncated_stream_and_preserves_validation(tls):
    async def handler(request):
        assert request.headers["authorization"] == f"Bearer {tls['token']}"
        assert request.extensions["sni_hostname"] == "lambda-worker.internal"
        if request.url.path == "/stream":
            return httpx.Response(200, text=json.dumps({"type": "delta"}) + "\n")
        return httpx.Response(
            400, json={"kind": "value_error", "message": "checkpoint mismatch: stale"}
        )

    transport = WorkerTransport("8.8.8.8", tls, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ValueError, match="checkpoint mismatch"):
            await transport.call("/generate", {})
        with pytest.raises(WorkerError, match="without completion"):
            _ = [event async for event in transport.stream({})]
    finally:
        await transport.aclose()


@pytest.mark.anyio
async def test_worker_cancellation_closes_generation():
    engine = fake_engine()
    closed = asyncio.Event()

    async def stream(*args):
        try:
            yield {"type": "delta", "text": "first"}
            await asyncio.Event().wait()
        finally:
            closed.set()

    engine._stream_generate = stream
    app = create_app(engine, "private")
    # Drive ASGI directly so disconnect occurs while the response is still open.
    body = json.dumps({"generation_id": "cancel-me", "payload": {}}).encode()
    first = True
    sent = asyncio.Event()

    async def receive():
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": body, "more_body": False}
        await sent.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            sent.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "method": "POST",
        "path": "/stream",
        "raw_path": b"/stream",
        "query_string": b"",
        "scheme": "https",
        "http_version": "1.1",
        "server": ("test", 443),
        "client": ("test", 1),
        "headers": [(b"authorization", b"Bearer private"), (b"content-type", b"application/json")],
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=3)
    assert closed.is_set()

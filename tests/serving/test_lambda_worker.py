"""Private worker protocol, cancellation, and bootstrap credential boundaries."""

import asyncio
import base64
import contextlib
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

    async def stream(*args, **kwargs):
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
        health = client.get("/health", headers=headers).json()
        assert health["base_model"] == MODEL
        assert health["canary_lane"] is True
        client.post("/lease", json={"generation_id": "g1"}, headers=headers)
        assert client.post("/generate", json=body, headers=headers).json() == {"text": "ok"}
        engine._generate.assert_awaited_once_with(
            body["payload"], body["record"], "checkpoint", "g1"
        )
        client.post("/lease", json={"generation_id": "g1"}, headers=headers)
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
    assert "mountpoint -q /data/hosting-cache" in files["/opt/flash/setup.sh"]
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
        if request.url.path == "/lease":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/stream":
            return httpx.Response(200, text=json.dumps({"type": "delta"}) + "\n")
        return httpx.Response(
            400, json={"kind": "value_error", "message": "checkpoint mismatch: stale"}
        )

    transport = WorkerTransport("8.8.8.8", tls, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ValueError, match="checkpoint mismatch"):
            await transport.call("/generate", {"generation_id": "test"})
        with pytest.raises(WorkerError, match="without completion"):
            _ = [event async for event in transport.stream({"generation_id": "test"})]
    finally:
        await transport.aclose()


@pytest.mark.anyio
async def test_worker_cancellation_closes_generation():
    engine = fake_engine()
    closed = asyncio.Event()

    async def stream(*args, **kwargs):
        try:
            yield {"type": "delta", "text": "first"}
            await asyncio.Event().wait()
        finally:
            closed.set()

    engine._stream_generate = stream
    app = create_app(engine, "private")
    app.state.generation_leases.renew("cancel-me")
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


@pytest.mark.anyio
async def test_real_tls_checks_certificate_and_worker_hostname(tls, tmp_path):
    import ssl
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

        def log_message(self, *args):
            pass

    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert.write_text(tls["cert"])
    key.write_text(tls["key"])
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    worker = WorkerTransport("8.8.8.8", tls)
    worker.client.base_url = f"https://127.0.0.1:{server.server_port}"
    try:
        assert await worker.call("/health") == {"ok": True}
        # The certificate is for the worker DNS identity, not its transport IP.
        wrong_name = worker.client.build_request("GET", "/health")
        with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
            await worker.client.send(wrong_name)
    finally:
        await worker.aclose()
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.anyio
async def test_generation_lease_expiry_aborts_gpu_before_cancelling_owner():
    from unittest.mock import Mock

    from flash.serving.lambda_cloud.leases import GenerationLeases
    from flash.serving.src.stream_channel.protocol import LEASE_SECONDS

    now = 100
    leases = GenerationLeases(2, clock=lambda: now)
    leases.renew("generation")
    await leases.check("generation")
    now += LEASE_SECONDS + 0.1
    with pytest.raises(TimeoutError, match="lease expired"):
        await leases.check("generation")
    owner = Mock()
    abort = AsyncMock()
    await leases.watch("generation", owner, abort)
    abort.assert_awaited_once_with("generation")
    owner.cancel.assert_called_once()


def test_runtime_pin_allows_version_only_bumps_but_rejects_dependency_drift(tmp_path):
    from flash.serving.lambda_cloud.runtime_image import verify

    base = '[[package]]\nname="freesolo-flash"\nversion="1.4.25"\n[[package]]\nname="vllm"\nversion="0.23.0"\n'
    image, source = tmp_path / "image.lock", tmp_path / "source.lock"
    image.write_text(base)
    source.write_text(base.replace("1.4.25", "1.4.26"))
    verify(image, source)
    source.write_text(base.replace("0.23.0", "0.24.0"))
    with pytest.raises(RuntimeError, match="dependency lock mismatch"):
        verify(image, source)


@pytest.mark.anyio
async def test_canary_lane_answers_while_regular_admission_is_full(monkeypatch):
    import flash.serving.lambda_cloud.worker as worker

    monkeypatch.setattr(worker, "engine_overrides_for", lambda model: {"max_num_seqs": 2})
    engine = fake_engine()
    release = asyncio.Event()

    async def generate(payload, record, expected, generation_id):
        if not record.get("serve_base_model"):
            await release.wait()
        return {"request_id": generation_id}

    engine._generate = generate
    app = create_app(engine, "private")
    headers = {"Authorization": "Bearer private"}

    def body(generation_id, *, canary=False):
        return {
            "payload": {"model_id": MODEL, "prompt": "1", "max_tokens": 1},
            "record": {"model_id": MODEL, "serve_base_model": canary},
            "generation_id": generation_id,
            **({"canary": True} if canary else {}),
        }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://worker"
    ) as client:
        for generation_id in ("busy", "queued", "canary"):
            app.state.generation_leases.renew(generation_id)
        # max_num_seqs - 1 = 1 regular slot: "busy" holds it and "queued" waits behind it.
        busy = asyncio.create_task(client.post("/generate", json=body("busy"), headers=headers))
        queued = asyncio.create_task(client.post("/generate", json=body("queued"), headers=headers))
        await asyncio.sleep(0.1)
        canary = await asyncio.wait_for(
            client.post("/generate", json=body("canary", canary=True), headers=headers), 2
        )
        assert canary.json() == {"request_id": "canary"}
        assert not queued.done()
        release.set()
        assert (await busy).status_code == 200
        assert (await queued).status_code == 200


def test_canary_flag_requires_a_one_token_base_model_generation():
    engine = fake_engine()
    body = {
        "payload": {"model_id": MODEL, "prompt": "long", "max_tokens": 512},
        "record": {"model_id": MODEL, "serve_base_model": True},
        "generation_id": "jump-the-queue",
        "canary": True,
    }
    with TestClient(create_app(engine, "private")) as client:
        headers = {"Authorization": "Bearer private"}
        client.post("/lease", json={"generation_id": "jump-the-queue"}, headers=headers)
        response = client.post("/generate", json=body, headers=headers)
    assert response.status_code == 400
    engine._generate.assert_not_awaited()


@pytest.mark.parametrize(
    ("response", "status", "detail"),
    [
        (httpx.Response(500, text="Internal Server Error"), 502, "(HTTP 500)"),
        (httpx.Response(504, json={"kind": "timeout"}), 504, "(HTTP 504, kind=timeout)"),
        (httpx.Response(401, json={"error": "unauthorized"}), 502, "(HTTP 401, kind=unauthorized)"),
        (httpx.Response(503, json={"kind": "token=secret"}), 502, "(HTTP 503)"),
    ],
)
def test_worker_failure_names_status_and_fixed_kind_only(response, status, detail):
    from flash.serving.lambda_cloud.transport import check_response

    with pytest.raises(WorkerError) as excinfo:
        check_response(response)
    assert str(excinfo.value) == f"Lambda worker request failed {detail}"
    assert excinfo.value.status_code == status


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("code", "body", "status", "detail"),
    [
        (401, {"error": "unauthorized"}, 502, "(HTTP 401, kind=unauthorized)"),
        (504, {"kind": "timeout"}, 504, "(HTTP 504, kind=timeout)"),
    ],
)
async def test_stream_failure_reads_error_body_before_naming_kind(tls, code, body, status, detail):
    async def chunks():
        yield json.dumps(body).encode()

    async def handler(request):
        if request.url.path == "/lease":
            return httpx.Response(200, json={"ok": True})
        # A truly streamed body stays unread under send(stream=True), like a real socket.
        return httpx.Response(code, content=chunks())

    transport = WorkerTransport("8.8.8.8", tls, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(WorkerError) as excinfo:
            _ = [event async for event in transport.stream({"generation_id": "test"})]
    finally:
        await transport.aclose()
    assert str(excinfo.value) == f"Lambda worker request failed {detail}"
    assert excinfo.value.status_code == status


@pytest.mark.anyio
async def test_slow_lease_renewal_still_renews_on_its_own_pool(tls):
    """A renewal slower than one heartbeat (but inside the lease) must not be abandoned."""
    renewed = []

    async def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/lease":
            await asyncio.sleep(1.2 if renewed else 0)
            renewed.append(body["generation_id"])
            return httpx.Response(200, json={"ok": True})
        await asyncio.sleep(4.0)
        return httpx.Response(200, json={"request_id": body["generation_id"]})

    transport = WorkerTransport("8.8.8.8", tls, transport=httpx.MockTransport(handler))
    try:
        assert transport._client_for("/lease") is transport.lease_client
        assert transport._client_for("/generate") is transport.client
        assert transport.lease_client is not transport.client
        result = await transport.call("/generate", {"generation_id": "g-slow"})
        assert result["request_id"] == "g-slow"
        # the initial lease plus at least one slow (1.2 s) renewal that completed
        assert len(renewed) >= 2
    finally:
        await transport.aclose()
    assert transport.lease_client.is_closed
    assert transport.client.is_closed


@pytest.mark.anyio
async def test_stalled_lease_renewal_is_retried_before_the_lease_lapses(tls):
    """One stalled renewal must leave time for a retry inside LEASE_SECONDS."""
    from flash.serving.src.stream_channel.protocol import LEASE_SECONDS

    loop = asyncio.get_running_loop()
    leases = []
    retried = asyncio.Event()

    async def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/lease":
            leases.append(loop.time())
            if len(leases) == 2:
                await asyncio.sleep(60)  # first renewal stalls before reaching the worker
            if len(leases) == 3:
                retried.set()
            return httpx.Response(200, json={"ok": True})
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(LEASE_SECONDS + 2):
                await retried.wait()
        return httpx.Response(200, json={"request_id": body["generation_id"]})

    transport = WorkerTransport("8.8.8.8", tls, transport=httpx.MockTransport(handler))
    try:
        await transport.call("/generate", {"generation_id": "g-stall"})
    finally:
        await transport.aclose()
    assert len(leases) >= 3
    # the retry reached the worker while the lease from the initial renewal was still live
    assert leases[2] - leases[0] < LEASE_SECONDS

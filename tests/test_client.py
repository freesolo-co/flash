"""ApiClient: auth headers, error mapping, log paging (stdlib stub server, CPU-only)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from flash.client import ApiClient, ApiError, ClientError


@pytest.fixture
def stub():
    seen: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen["auth"] = self.headers.get("Authorization")
            seen["path"] = self.path
            if self.path.startswith("/v1/runs/authfail"):
                self._send(401, {"detail": "invalid or missing API key"})
            elif self.path.startswith("/v1/runs/missing"):
                self._send(404, {"detail": "unknown run_id: missing"})
            elif self.path.startswith("/v1/runs/r1/logs"):
                self._send(200, {"run_id": "r1", "logs": "hi\n", "offset": 3, "state": "running"})
            else:
                self._send(200, {"runs": []})

        def do_POST(self):
            seen["auth"] = self.headers.get("Authorization")
            seen["path"] = self.path
            n = int(self.headers.get("Content-Length") or 0)
            seen["body"] = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/v1/envs":
                self._send(200, {"id": "freesolo-co/e"})
                return
            if self.path == "/v1/runs/json-chat/chat":
                self._send(200, {"choices": [{"message": {"content": "json reply"}}]})
                return
            if self.path == "/v1/runs/r1/chat":
                body = "héllo".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                for byte in body:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                return
            self._send(200, {"run_id": "r1", "state": "queued"})

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def test_bearer_header_and_payload(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    out = client.create_run({"model": "m"})
    assert out["run_id"] == "r1"
    assert seen["auth"] == "Bearer fslo-user-test"
    assert seen["body"] == {"spec": {"model": "m"}}


def test_create_run_sends_runtime_secrets_outside_spec(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    client.create_run({"model": "m"}, runtime_secrets={"WANDB_API_KEY": "wb-user"})
    assert seen["body"] == {
        "spec": {"model": "m"},
        "runtime_secrets": {"WANDB_API_KEY": "wb-user"},
    }


def test_api_error_carries_server_detail(stub):
    url, _ = stub
    client = ApiClient(url, "fslo-user-test")
    with pytest.raises(ApiError) as excinfo:
        client.get_run("missing")
    assert excinfo.value.status == 404
    assert "unknown run_id: missing" in str(excinfo.value)


def test_get_worker_output_tolerates_missing_route(monkeypatch):
    """A managed server predating the /worker route returns a bare 404 'Not Found' -> get_worker_output
    yields {} so a CLI upgraded ahead of the rollout doesn't hard-fail. Real 404s (unknown run_id) and
    other errors still propagate."""
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test")

    def _raise(status, msg):
        def _req(method, path, **kw):
            raise ApiError(status, msg)

        return _req

    monkeypatch.setattr(client, "_request", _raise(404, "Not Found"))
    assert client.get_worker_output("r1") == {}  # route absent -> graceful empty
    monkeypatch.setattr(client, "_request", _raise(404, "unknown run_id: r1"))
    with pytest.raises(ApiError):  # a real 404 still surfaces
        client.get_worker_output("r1")
    monkeypatch.setattr(client, "_request", _raise(500, "boom"))
    with pytest.raises(ApiError):  # non-404 propagates
        client.get_worker_output("r1")


def test_api_error_mentions_env_override(stub):
    url, _ = stub
    client = ApiClient(url, "fslo-user-test", key_source="FREESOLO_API_KEY")
    with pytest.raises(ApiError) as excinfo:
        client.get_run("authfail")
    assert excinfo.value.status == 401
    assert "invalid or missing API key" in str(excinfo.value)
    assert "FREESOLO_API_KEY is set and overrides" in str(excinfo.value)


def test_logs_offset_in_query(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    page = client.get_logs("r1", offset=3)
    assert page["offset"] == 3
    assert page["logs"] == "hi\n"
    assert seen["path"].endswith("/v1/runs/r1/logs?offset=3")


def test_chat_stream_sends_stream_request_and_yields_text(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    chunks = list(
        client.chat_stream("r1", [{"role": "user", "content": "hi"}], temperature=0.2, max_tokens=7)
    )
    assert "".join(chunks) == "héllo"
    assert seen["path"] == "/v1/runs/r1/chat"
    assert seen["auth"] == "Bearer fslo-user-test"
    assert seen["body"] == {
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "max_tokens": 7,
        "stream": True,
    }


def test_chat_stream_accepts_json_fallback(stub):
    url, _ = stub
    client = ApiClient(url, "fslo-user-test")
    chunks = list(client.chat_stream("json-chat", [{"role": "user", "content": "hi"}]))
    assert chunks == ["json reply"]


def test_publish_env_plain_without_progress(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")
    out = client.publish_env(name="e", package_b64="QQ==")
    assert out["id"] == "freesolo-co/e"
    assert seen["path"] == "/v1/envs"
    assert seen["body"] == {"name": "e", "package_b64": "QQ=="}


def test_publish_env_streams_body_and_reports_progress(stub, monkeypatch):
    import flash.client.http as http_mod

    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    # spy on the streaming reader so we prove _post_with_progress (not the plain _request
    # path) ran — a refactor that faked progress around a one-shot send must fail this test.
    wrapped: list[int] = []
    real_reader = http_mod._ProgressReader

    class _SpyReader(real_reader):
        def __init__(self, data, progress):
            wrapped.append(len(data))
            super().__init__(data, progress)

    monkeypatch.setattr(http_mod, "_ProgressReader", _SpyReader)

    # a payload large enough to span several 8192-byte http.client send chunks, so the
    # callback fires repeatedly with a growing count instead of one all-at-once call.
    big = "A" * 30000
    body = {"name": "e", "package_b64": big}
    calls: list[tuple[int, int]] = []
    out = client.publish_env(
        name="e", package_b64=big, progress=lambda sent, total: calls.append((sent, total))
    )
    assert out["id"] == "freesolo-co/e"
    # the server reads exactly Content-Length bytes, so a correct multi-chunk stream
    # round-trips the full 30 KB body byte-for-byte across the chunk boundaries.
    assert seen["body"] == body
    expected_total = len(json.dumps(body).encode())
    assert wrapped == [expected_total]  # the streaming reader wrapped the full payload
    assert len(calls) > 1  # multiple chunks => multiple progress updates
    assert all(sent <= total for sent, total in calls)
    assert calls[0][0] < calls[-1][0]  # the byte count grew across chunks
    assert calls[-1] == (expected_total, expected_total)  # reached 100% of the real payload


def test_publish_env_progress_errors_do_not_abort_upload(stub):
    url, seen = stub
    client = ApiClient(url, "fslo-user-test")

    def boom(sent, total):
        raise RuntimeError("render failed")

    # a raising progress widget must never abort an in-flight upload (contextlib.suppress).
    out = client.publish_env(name="e", package_b64="QQ==", progress=boom)
    assert out["id"] == "freesolo-co/e"
    assert seen["body"] == {"name": "e", "package_b64": "QQ=="}


def test_unreachable_server_is_actionable():
    client = ApiClient("http://127.0.0.1:1", "fslo-user-test", timeout=2)
    with pytest.raises(ClientError, match="FLASH_API_URL"):
        client.health()

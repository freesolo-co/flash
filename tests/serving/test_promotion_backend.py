"""A stale backend must be rejected without generating or billing a canary."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from flash.serving.promotion import backend, gate
from tests.serving.test_promotion_gate import _gate_step_env


@pytest.fixture
def server():
    replies = []
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, dict(self.headers)))
            status, headers, body = replies.pop(0)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", replies, calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def schema():
    return {"paths": {path: {"post": {}} for path in backend.REQUIRED_POST_ROUTES}}


def test_read_only_contract_probe(server):
    url, replies, calls = server
    replies.append((200, {}, json.dumps(schema()).encode()))
    assert backend.check_backend(url) == ""
    assert [path for path, _ in calls] == ["/openapi.json"]
    assert all("Authorization" not in headers for _, headers in calls)


@pytest.mark.parametrize("body", [None, [], {}, {"paths": []}, {"paths": {}}])
def test_missing_or_malformed_routes_fail_closed(body):
    assert not backend.compatible_schema(body)


@pytest.mark.parametrize("path", backend.REQUIRED_POST_ROUTES)
def test_get_route_does_not_prove_post_route(path):
    payload = schema()
    payload["paths"][path] = {"get": {}}
    assert not backend.compatible_schema(payload)


@pytest.mark.parametrize("status", [301, 303, 401, 403, 404, 500])
def test_http_errors_and_redirects_are_not_followed_or_logged(server, monkeypatch, capsys, status):
    url, replies, calls = server
    replies.append((status, {"Location": url + "/credential-canary"}, b"secret response"))
    monkeypatch.setenv("PLATFORM_BACKEND_URL", url)
    assert backend.main() == 1
    assert len(calls) == 1
    output = capsys.readouterr()
    assert output.err == f"backend preflight failed: {backend.BACKEND_UNAVAILABLE}\n"
    assert output.out == ""


def test_stale_backend_stops_promotion_before_gpu_or_database_requests(server, monkeypatch):
    url, replies, calls = server
    replies.append((200, {}, json.dumps({"paths": {}}).encode()))
    env = dict(_gate_step_env(), PLATFORM_BACKEND_URL=url)
    plan = gate._resolve("https://serve.example", env)

    def unexpected_client(*args, **kwargs):
        pytest.fail("promotion must stop before the serving/database clients start")

    monkeypatch.setattr("httpx.AsyncClient", unexpected_client)
    verdict = asyncio.run(gate._run(plan))
    assert verdict.reason == backend.BACKEND_INCOMPATIBLE
    assert len(calls) == 1


def test_backend_preflight_precedes_live_deployment():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/deploy-modal.yml").read_text())
    steps = workflow["jobs"]["deploy"]["steps"]
    preflight = next(
        i
        for i, step in enumerate(steps)
        if step.get("run") == "python -m flash.serving.promotion.backend"
    )
    deploy = next(i for i, step in enumerate(steps) if step.get("id") == "deploy_serving")
    assert preflight < deploy
    assert steps[preflight]["if"] == steps[deploy]["if"]

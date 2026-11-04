"""The checkpoint proxy must not hide an upstream 503 or replay generation."""

import httpx
import pytest

import flash.serve.deployment.deploy as serving_deploy
from flash.server.routes.serving_errors import upstream_chat_error
from tests._helpers.managed_chat import _deployed_chat_run
from tests.test_server_api import _bearer

pytest_plugins = ("tests._helpers.server_api_plugin",)


def test_checkpoint_proxy_preserves_unavailable_cause_without_retry(api, monkeypatch, caplog):
    key, run_id = _deployed_chat_run(api)
    calls = []

    def unavailable(**kwargs):
        calls.append(kwargs)
        httpx.Response(
            503,
            json={"error": {"code": "serving_capacity_unavailable", "message": "No serving slot"}},
            headers={"Retry-After": "3", "X-Internal-Secret": "must-not-forward"},
            request=httpx.Request("POST", "https://serving.example/v1/chat/completions"),
        ).raise_for_status()

    monkeypatch.setattr(serving_deploy, "chat", unavailable)
    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "checkpoint_id": f"{run_id}/final",
            "messages": [{"role": "user", "content": "private-prompt-not-for-logs"}],
        },
        headers=_bearer(key),
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "No serving slot"
    assert response.headers["Retry-After"] == "3"
    assert response.headers["X-Flash-Upstream-Code"] == "serving_capacity_unavailable"
    assert "X-Internal-Secret" not in response.headers
    assert len(calls) == 1
    assert "upstream_status=503 code=serving_capacity_unavailable" in caplog.text
    assert "private-prompt-not-for-logs" not in caplog.text


@pytest.mark.parametrize("body", [None, [], "unexpected", {"error": []}])
def test_malformed_unavailable_body_remains_a_503(body):
    response = httpx.Response(503, json=body, request=httpx.Request("POST", "https://example.test"))
    error = httpx.HTTPStatusError("unavailable", request=response.request, response=response)
    result = upstream_chat_error(error, checkpoint_id="run/final")
    assert result.status_code == 503
    assert result.detail == "Serving rejected the inference request (HTTP 503)."


def test_error_headers_and_logs_reject_untrusted_codes(caplog):
    response = httpx.Response(
        503,
        json={"error": {"code": "token=secret-value", "message": "temporarily unavailable"}},
        headers={"Retry-After": "token=secret-value"},
        request=httpx.Request("POST", "https://example.test"),
    )
    error = httpx.HTTPStatusError("unavailable", request=response.request, response=response)
    result = upstream_chat_error(error, checkpoint_id="run/final")
    assert result.headers is None
    assert "secret-value" not in caplog.text


@pytest.mark.parametrize(("request_id", "expected"), [("a" * 32, True), ("malformed", False)])
def test_proxy_forwards_only_valid_pre_generation_authorization_marker(request_id, expected):
    response = httpx.Response(
        503,
        json={"detail": "serving auth backend unreachable"},
        headers={"X-Freesolo-Auth-Request-Id": request_id, "X-Freesolo-Inference-Started": "false"},
        request=httpx.Request("POST", "https://example.test"),
    )
    error = httpx.HTTPStatusError("unavailable", request=response.request, response=response)
    result = upstream_chat_error(error, checkpoint_id="run/final")
    assert ((result.headers or {}).get("X-Freesolo-Inference-Started") == "false") == expected

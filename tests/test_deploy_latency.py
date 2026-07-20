from __future__ import annotations

import json
import sys
import threading
import types

import pytest


def test_control_http_client_is_reused_and_all_clients_close(monkeypatch):
    import flash.serve.deploy as deploy

    created = []

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self.requests = []
            created.append(self)

        def request(self, method, url, **kwargs):
            self.requests.append((method, url, kwargs))
            return _Response()

        def close(self):
            self.closed = True

    deploy._close_http_client()
    monkeypatch.setattr(deploy.httpx, "Client", _Client)

    deploy._serving_request("GET", "https://serve.example/healthz")
    deploy._serving_request("POST", "https://serve.example/adapters", json={"id": "r1"})

    assert len(created) == 1
    assert created[0] is deploy._http_client()
    assert [request[:2] for request in created[0].requests] == [
        ("GET", "https://serve.example/healthz"),
        ("POST", "https://serve.example/adapters"),
    ]
    assert created[0].kwargs == {"follow_redirects": True, "max_redirects": 100}

    deploy._close_http_client()

    assert created[0].closed is True
    assert deploy._HTTP_CLIENT is None
    assert deploy._STREAM_HTTP_CLIENT is None


def test_streaming_pool_cannot_starve_control_requests(monkeypatch):
    import flash.serve.deploy as deploy

    created = []

    class _UndeployResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "run_id": "run-1",
                "disabled_aliases": ["run-1"],
                "disabled_revisions": [],
            }

    class _StreamResponse:
        def __init__(self, client, request):
            self.client = client
            self.request = request
            self.headers = {"content-type": "text/event-stream"}

        def __enter__(self):
            if not self.client.pool.acquire(blocking=False):
                raise deploy.httpx.PoolTimeout("pool exhausted", request=self.request)
            return self

        def __exit__(self, *_args):
            self.client.pool.release()
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"held"}}]}'
            yield "data: [DONE]"

    class _PoolLimitedClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.pool = threading.BoundedSemaphore(1)
            self.closed = False
            created.append(self)

        def stream(self, method, url, **_kwargs):
            return _StreamResponse(self, deploy.httpx.Request(method, url))

        def request(self, method, url, **_kwargs):
            request = deploy.httpx.Request(method, url)
            if not self.pool.acquire(blocking=False):
                raise deploy.httpx.PoolTimeout("pool exhausted", request=request)
            try:
                return _UndeployResponse()
            finally:
                self.pool.release()

        def close(self):
            self.closed = True

    deploy._close_http_client()
    monkeypatch.setattr(deploy.httpx, "Client", _PoolLimitedClient)

    stream = deploy.chat_stream("run-1", [{"role": "user", "content": "hello"}])
    assert next(stream) == "held"
    try:
        result = deploy.undeploy_adapter("run-1")
    finally:
        stream.close()

    assert result["serving_deregistered"] is True
    assert len(created) == 2
    assert created[0] is deploy._STREAM_HTTP_CLIENT
    assert created[1] is deploy._HTTP_CLIENT

    deploy._close_http_client()
    assert all(client.closed for client in created)


def test_readiness_backoff_honors_retry_after_and_cap(monkeypatch):
    import flash.serve.deploy as deploy

    revision = "run-1@final." + "a" * 40
    subfolder = "sft/run-1/seed0/adapter"
    loading = {
        "adapter_id": revision,
        "subfolder": subfolder,
        "metadata": {"lifecycle_state": "loading"},
    }
    ready = {
        "adapter_id": revision,
        "subfolder": subfolder,
        "metadata": {"lifecycle_state": "ready"},
    }
    outcomes = [
        deploy.ServingError("temporary 503", status_code=503, retry_after="1.25"),
        (loading, types.SimpleNamespace(headers={"Retry-After": "9"})),
        (ready, types.SimpleNamespace(headers={})),
    ]
    sleeps = []

    def registered_adapter_response(adapter_id, *, timeout_s=None):
        assert adapter_id == revision
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(deploy, "READBACK_DELAY_SECONDS", 0.5)
    monkeypatch.setattr(deploy, "READBACK_MAX_DELAY_SECONDS", 2.0)
    monkeypatch.setattr(
        deploy,
        "_registered_adapter_response",
        registered_adapter_response,
    )
    monkeypatch.setattr(deploy.time, "sleep", sleeps.append)

    assert deploy._wait_revision_ready(revision, subfolder, budget_s=30.0) == ready
    assert sleeps == [1.25, 2.0]


def test_adapter_preflight_validates_config_before_listing_tensors(monkeypatch, tmp_path):
    import flash.serve.deploy as deploy

    config_path = tmp_path / "adapter_config.json"
    config_path.write_text(json.dumps({"r": 32}), encoding="utf-8")
    calls = []

    def hf_hub_download(**_kwargs):
        calls.append("config")
        return str(config_path)

    class _HfApi:
        def list_repo_tree(self, **kwargs):
            calls.append("tensors")
            subfolder = kwargs["path_in_repo"]
            return [
                types.SimpleNamespace(
                    path=f"{subfolder}/adapter_model.safetensors",
                    size=123,
                )
            ]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download, HfApi=_HfApi),
    )

    assert (
        deploy.adapter_artifact_lora_rank(
            "org/repo",
            "sft/run-1/seed0/adapter",
            hf_revision="a" * 40,
        )
        == 32
    )
    assert calls == ["config", "tensors"]


def test_adapter_preflight_config_failure_does_not_start_tensor_listing(monkeypatch):
    import flash.serve.deploy as deploy

    tensor_started = False

    def hf_hub_download(**_kwargs):
        raise RuntimeError("config failed")

    class _HfApi:
        def list_repo_tree(self, **_kwargs):
            nonlocal tensor_started
            tensor_started = True
            return []

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download, HfApi=_HfApi),
    )

    with pytest.raises(deploy.ServingError, match="failed to read org/repo"):
        deploy.adapter_artifact_lora_rank(
            "org/repo",
            "sft/run-1/seed0/adapter",
            hf_revision="a" * 40,
        )

    assert tensor_started is False


def test_zero_retry_after_uses_positive_readiness_backoff(monkeypatch):
    import flash.serve.deploy as deploy

    monkeypatch.setattr(deploy, "READBACK_DELAY_SECONDS", 0.5)
    assert deploy._readback_delay(0, "0") == 0.5


def test_activation_reconciliation_keeps_reliability_delay(monkeypatch):
    import flash.serve.deploy as deploy

    revision = "run-1@final." + "a" * 40
    previous = "run-1@final." + "b" * 40
    aliases = [
        {"metadata": {"alias_of": previous}},
        {"metadata": {"alias_of": previous}},
        {"metadata": {"alias_of": revision}, "updated_at": "2026-07-20T00:00:00Z"},
    ]
    sleeps = []

    def fail_activation(*_args, **_kwargs):
        raise deploy.ServingError("activation response lost")

    monkeypatch.setattr(deploy, "_serving_request", fail_activation)
    monkeypatch.setattr(deploy, "_registered_adapter", lambda _run_id: aliases.pop(0))
    monkeypatch.setattr(deploy.time, "sleep", sleeps.append)

    result = deploy._activate_revision(
        "run-1",
        revision,
        "run-1/step-10",
        expected_adapter_revision=previous,
    )

    assert result["target_adapter_revision"] == revision
    assert sleeps == [
        deploy.ACTIVATION_READBACK_DELAY_SECONDS,
        deploy.ACTIVATION_READBACK_DELAY_SECONDS,
    ]


def test_bounded_smoke_chat_uses_isolated_client(monkeypatch):
    import flash.serve.deploy as deploy

    created = []

    class _Response:
        status_code = 200

        def __init__(self):
            self.headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    class _Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            created.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True
            return False

        def post(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(deploy.httpx, "Client", _Client)
    monkeypatch.setattr(
        deploy,
        "_http_client",
        lambda: (_ for _ in ()).throw(AssertionError("smoke must not use shared client")),
    )

    assert deploy.chat(
        "run-1@final." + "a" * 40,
        [{"role": "user", "content": "hello"}],
        timeout_s=1.0,
        retry_unavailable=True,
    ) == {"choices": []}
    assert len(created) == 1
    assert created[0].closed is True

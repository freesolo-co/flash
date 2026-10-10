from __future__ import annotations

import json
import sys
import threading
import types

import httpx
import pytest

import flash.serve.contract.errors as serving_errors
import flash.serve.deployment.adapter_check as adapter_check
import flash.serve.deployment.deploy as serving_deploy
import flash.serve.request.transport as serving_transport


def test_control_http_client_is_reused_and_all_clients_close(monkeypatch):

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

    serving_transport._close_http_client()
    monkeypatch.setattr(httpx, "Client", _Client)

    serving_transport.serving_request("GET", "https://serve.example/healthz")
    serving_transport.serving_request("POST", "https://serve.example/adapters", json={"id": "r1"})

    assert len(created) == 1
    assert created[0] is serving_transport._http_client()
    assert [request[:2] for request in created[0].requests] == [
        ("GET", "https://serve.example/healthz"),
        ("POST", "https://serve.example/adapters"),
    ]
    assert created[0].kwargs == {
        "follow_redirects": True,
        "max_redirects": serving_transport._MAX_REDIRECTS,
        "event_hooks": {"request": [serving_transport._strip_internal_key_off_origin]},
    }

    serving_transport._close_http_client()

    assert created[0].closed is True
    assert serving_transport._HTTP_CLIENT is None
    assert serving_transport._CHAT_HTTP_CLIENT is None


def test_streaming_pool_cannot_starve_control_requests(monkeypatch):
    import flash.serve.deployment.deploy as deploy

    created = []

    class _UndeployResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "checkpoint_id": "run-1/final",
                "disabled_checkpoints": ["run-1/final"],
            }

    class _StreamResponse:
        def __init__(self, client, request):
            self.client = client
            self.request = request
            self.headers = {"content-type": "text/event-stream"}

        def __enter__(self):
            if not self.client.pool.acquire(blocking=False):
                raise httpx.PoolTimeout("pool exhausted", request=self.request)
            return self

        def __exit__(self, *_args):
            self.client.pool.release()
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"held"}}]}'
            yield ""
            yield "data: [DONE]"
            yield ""

    class _PoolLimitedClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.pool = threading.BoundedSemaphore(1)
            self.closed = False
            created.append(self)

        def stream(self, method, url, **_kwargs):
            return _StreamResponse(self, httpx.Request(method, url))

        def request(self, method, url, **_kwargs):
            request = httpx.Request(method, url)
            if not self.pool.acquire(blocking=False):
                raise httpx.PoolTimeout("pool exhausted", request=request)
            try:
                return _UndeployResponse()
            finally:
                self.pool.release()

        def close(self):
            self.closed = True

    serving_transport._close_http_client()
    monkeypatch.setattr(httpx, "Client", _PoolLimitedClient)

    stream = serving_deploy.chat_stream(
        "run-1/final",
        [{"role": "user", "content": "hello"}],
        org_id="org-1",
    )
    assert next(stream) == "held"
    try:
        result = deploy.undeploy_adapter("run-1/final", org_id="org-1")
    finally:
        stream.close()

    assert result["serving_deregistered"] is True
    assert len(created) == 2
    assert created[0] is serving_transport._CHAT_HTTP_CLIENT
    assert created[1] is serving_transport._HTTP_CLIENT
    assert created[0].kwargs["limits"].max_connections is None
    assert created[0].kwargs["limits"].max_keepalive_connections == 100

    serving_transport._close_http_client()
    assert all(client.closed for client in created)


def test_normal_chat_pool_cannot_starve_control_requests(monkeypatch):
    import flash.serve.deployment.deploy as deploy

    created = []
    chat_started = threading.Event()
    release_chat = threading.Event()
    chat_result = []

    class _Response:
        status_code = 200

        def __init__(self):
            self.headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    class _UndeployResponse(_Response):
        def json(self):
            return {
                "checkpoint_id": "run-1/final",
                "disabled_checkpoints": ["run-1/final"],
            }

    class _PoolLimitedClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.pool = threading.BoundedSemaphore(1)
            self.closed = False
            created.append(self)

        def post(self, *_args, **_kwargs):
            assert self.pool.acquire(blocking=False)
            chat_started.set()
            assert release_chat.wait(timeout=2.0)
            self.pool.release()
            return _Response()

        def request(self, *_args, **_kwargs):
            assert self.pool.acquire(blocking=False)
            try:
                return _UndeployResponse()
            finally:
                self.pool.release()

        def close(self):
            self.closed = True

    serving_transport._close_http_client()
    monkeypatch.setattr(httpx, "Client", _PoolLimitedClient)

    thread = threading.Thread(
        target=lambda: chat_result.append(
            deploy.chat(
                "run-1/final",
                [{"role": "user", "content": "hello"}],
                org_id="org-1",
            )
        )
    )
    thread.start()
    assert chat_started.wait(timeout=1.0)
    try:
        result = deploy.undeploy_adapter("run-1/final", org_id="org-1")
    finally:
        release_chat.set()
        thread.join(timeout=2.0)

    assert result["serving_deregistered"] is True
    assert chat_result == [{"choices": []}]
    assert created[0] is serving_transport._CHAT_HTTP_CLIENT
    assert created[1] is serving_transport._HTTP_CLIENT
    assert created[0].kwargs["limits"].max_connections is None
    assert created[0].kwargs["limits"].max_keepalive_connections == 100

    serving_transport._close_http_client()
    assert all(client.closed for client in created)


def test_readiness_backoff_honors_retry_after_and_cap(monkeypatch):
    import flash.serve.deployment.deploy as deploy

    revision = "run-1/final"
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
        serving_errors.ServingError("temporary 503", status_code=503, retry_after="1.25"),
        (loading, types.SimpleNamespace(headers={"Retry-After": "9"})),
        (ready, types.SimpleNamespace(headers={})),
    ]
    sleeps = []

    def registered_adapter_response(org_id, adapter_id, *, timeout_s=None):
        assert org_id == "org-1"
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

    assert deploy._wait_checkpoint_ready("org-1", revision, subfolder, budget_s=30.0) == ready
    assert sleeps == [1.25, 2.0]


def test_adapter_preflight_validates_config_before_listing_tensors(monkeypatch, tmp_path):

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
        adapter_check.adapter_artifact_metadata(
            "org/repo",
            "sft/run-1/seed0/adapter",
            artifact_revision="a" * 40,
        ).lora_rank
        == 32
    )
    assert calls == ["config", "tensors"]


def test_adapter_preflight_config_failure_does_not_start_tensor_listing(monkeypatch):

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

    with pytest.raises(serving_errors.ServingError, match="failed to read org/repo"):
        adapter_check.adapter_artifact_metadata(
            "org/repo",
            "sft/run-1/seed0/adapter",
            artifact_revision="a" * 40,
        )

    assert tensor_started is False


def test_zero_retry_after_uses_positive_readiness_backoff(monkeypatch):
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(deploy, "READBACK_DELAY_SECONDS", 0.5)
    assert deploy._readback_delay(0, "0") == 0.5


def test_bounded_smoke_chat_uses_isolated_client(monkeypatch):
    import flash.serve.deployment.deploy as deploy

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

    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setattr(
        serving_transport,
        "_http_client",
        lambda: (_ for _ in ()).throw(AssertionError("smoke must not use shared client")),
    )

    assert deploy.chat(
        "run-1/final",
        [{"role": "user", "content": "hello"}],
        org_id="org-1",
        timeout_s=1.0,
        retry_unavailable=True,
    ) == {"choices": []}
    assert len(created) == 1
    assert created[0].closed is True

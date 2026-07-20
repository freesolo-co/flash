from __future__ import annotations

import json
import sys
import threading
import types


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


def test_adapter_preflight_fetches_config_and_tensors_concurrently(monkeypatch, tmp_path):
    import flash.serve.deploy as deploy

    config_path = tmp_path / "adapter_config.json"
    config_path.write_text(json.dumps({"r": 32}), encoding="utf-8")
    barrier = threading.Barrier(2, timeout=2.0)
    started = []

    def hf_hub_download(**kwargs):
        started.append("config")
        barrier.wait()
        return str(config_path)

    class _HfApi:
        def list_repo_tree(self, **kwargs):
            started.append("tensors")
            barrier.wait()
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
    assert set(started) == {"config", "tensors"}


def test_adapter_preflight_config_failure_does_not_wait_for_tensors(monkeypatch):
    import flash.serve.deploy as deploy

    tensor_started = threading.Event()
    release_tensor = threading.Event()
    result_ready = threading.Event()
    background_exception_consumed = threading.Event()
    result = {}

    def hf_hub_download(**_kwargs):
        assert tensor_started.wait(1.0)
        raise RuntimeError("config failed")

    class _HfApi:
        def list_repo_tree(self, **_kwargs):
            tensor_started.set()
            assert release_tensor.wait(2.0)
            raise RuntimeError("tensor failed")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download, HfApi=_HfApi),
    )
    original_consume = deploy._consume_future_exception

    def consume_future_exception(future):
        original_consume(future)
        background_exception_consumed.set()

    monkeypatch.setattr(deploy, "_consume_future_exception", consume_future_exception)

    def run_preflight():
        try:
            deploy.adapter_artifact_lora_rank(
                "org/repo",
                "sft/run-1/seed0/adapter",
                hf_revision="a" * 40,
            )
        except Exception as exc:
            result["error"] = exc
        finally:
            result_ready.set()

    caller = threading.Thread(target=run_preflight)
    caller.start()
    completed_before_tensor_release = result_ready.wait(0.5)
    release_tensor.set()
    caller.join(timeout=2.0)

    assert completed_before_tensor_release
    assert isinstance(result["error"], deploy.ServingError)
    assert "failed to read org/repo" in str(result["error"])
    assert background_exception_consumed.wait(1.0)

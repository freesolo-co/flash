"""Serving modes: deploy kwargs, per-run endpoints, undeploy, proxy shaping (CPU-only)."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from autoslm.serve.deploy import (
    MODES,
    Deployment,
    deploy_adapter,
    resolve_serve_deps,
    serve_endpoint_name,
    undeploy_adapter,
)


def test_serve_endpoint_name_is_per_run():
    a = serve_endpoint_name("RTX 5090", "autoslm-123-abcd1234")
    b = serve_endpoint_name("RTX 5090", "autoslm-456-ffff0000")
    assert a != b
    assert a.startswith("autoslm-serve-5090-")


def test_deploy_dry_run_modes():
    dev = deploy_adapter("r1", "Qwen/Qwen3-0.6B", "repo", "rl/r1/seed0", "RTX 4090", dry_run=True)
    assert dev.mode == "dev"
    assert dev.est_idle_cost_usd_per_day == 0.0
    warm = deploy_adapter(
        "r1",
        "Qwen/Qwen3-0.6B",
        "repo",
        "rl/r1/seed0",
        "RTX 4090",
        mode="always-on",
        dry_run=True,
    )
    assert warm.mode == "always-on"
    from autoslm.flash.pricing import hourly_rate

    assert warm.est_idle_cost_usd_per_day == pytest.approx(hourly_rate("RTX 4090") * 24, rel=0.01)


def test_deploy_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        deploy_adapter("r1", "m", "repo", "p", "RTX 4090", mode="warmish", dry_run=True)
    assert set(MODES) == {"dev", "always-on"}


def test_resolve_serve_deps(monkeypatch):
    monkeypatch.delenv("AUTOSLM_SERVE_DEPS", raising=False)
    assert any("vllm==0.19" in d for d in resolve_serve_deps())  # the single pinned stack
    monkeypatch.setenv("AUTOSLM_SERVE_DEPS", "vllm==9.9")  # explicit override wins
    assert resolve_serve_deps() == ["vllm==9.9"]


def test_resolve_serve_deps_preserves_comma_ranges(monkeypatch):
    # A PEP 440 range like transformers>=5.6,<5.11 must NOT be comma-split into two
    # pip args. Whitespace-separated string and JSON list both keep it intact.
    monkeypatch.setenv("AUTOSLM_SERVE_DEPS", "torch==2.10.0 transformers>=5.6,<5.11 vllm==0.19.1")
    assert resolve_serve_deps() == ["torch==2.10.0", "transformers>=5.6,<5.11", "vllm==0.19.1"]
    monkeypatch.setenv("AUTOSLM_SERVE_DEPS", '["transformers>=5.6,<5.11", "vllm==0.19.1"]')
    assert resolve_serve_deps() == ["transformers>=5.6,<5.11", "vllm==0.19.1"]


def test_undeploy_uses_rest(monkeypatch):
    from autoslm.flash import runpod_api
    from autoslm.serve import deploy as deploy_mod

    monkeypatch.setattr(
        runpod_api,
        "find_endpoints_by_name",
        lambda name: [{"id": "ep1", "name": name}],
    )
    deleted_ids = []
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda eid: deleted_ids.append(eid) or True)
    # A stale cache entry for this run+gpu must be dropped on undeploy so a later
    # redeploy builds a fresh endpoint instead of reusing the deleted one.
    name = deploy_mod.serve_endpoint_name("RTX 5090", "autoslm-1-abc")
    deploy_mod._ENDPOINT_CACHE[f"{name}:dev:300"] = object()
    out = undeploy_adapter("autoslm-1-abc", gpu_name="RTX 5090")
    assert deleted_ids == ["ep1"]
    assert len(out) == 1
    assert not [k for k in deploy_mod._ENDPOINT_CACHE if k.startswith(f"{name}:")]


def test_deployment_roundtrip_dict():
    d = Deployment(
        run_id="r",
        model="m",
        adapter_hf_prefix="p",
        gpu="RTX 4090",
        openai_model="autoslm-r",
        endpoint_name="e",
        mode="dev",
    )
    assert d.to_dict()["mode"] == "dev"


# ---------------------------------------------------------------------------
# Proxy: /v1 shaping with a mocked upstream chat
# ---------------------------------------------------------------------------
def test_proxy_chat_completions(monkeypatch):
    import autoslm.serve.proxy as proxy_mod
    from autoslm.client import ApiClient

    class FakeClient(ApiClient):
        def chat(self, run_id, messages, temperature=0.0, max_tokens=512):
            assert run_id == "r1"
            assert messages == [{"role": "user", "content": "hi"}]
            return {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "autoslm-r1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
            }

    # run the proxy on an ephemeral port in a thread
    from http.server import ThreadingHTTPServer

    server_holder = {}
    orig_init = ThreadingHTTPServer.__init__

    def capture_init(self, *a, **k):
        orig_init(self, *a, **k)
        server_holder["server"] = self

    monkeypatch.setattr(ThreadingHTTPServer, "__init__", capture_init)

    t = threading.Thread(
        target=proxy_mod.run_proxy,
        kwargs={
            "client": FakeClient("http://unused", "sk-autoslm-test"),
            "run_id": "r1",
            "host": "127.0.0.1",
            "port": 0,
        },
        daemon=True,
    )
    t.start()
    import time

    for _ in range(50):
        if "server" in server_holder:
            break
        time.sleep(0.05)
    server = server_holder["server"]
    port = server.server_address[1]

    # /v1/models
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models") as r:
        models = json.load(r)
    assert models["data"][0]["id"] == "autoslm-r1"

    # /v1/chat/completions
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        out = json.load(r)
    assert out["choices"][0]["message"]["content"] == "hello"

    server.shutdown()

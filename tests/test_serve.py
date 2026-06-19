"""Tests for the Flash serving wiring (no GPU/network).

Serving is delegated to the freesolo platform's multi-LoRA serving app; flash is a thin
client. These assert the deploy/undeploy/chat HTTP calls (httpx is monkeypatched) and the
dry-run Deployment shaping — there is no flash-owned vLLM endpoint to provision anymore.
"""

from __future__ import annotations

import pytest


def test_deploy_dry_run():
    from flash.serve.deploy import deploy_adapter

    dep = deploy_adapter(
        run_id="r1",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/r1/seed0",
        gpu_name="RTX 4090",
        dry_run=True,
    )
    d = dep.to_dict()
    assert d["state"] == "dry_run"
    assert d["gpu"] == "RTX 4090"
    # The adapter is addressed by its run_id on the freesolo serving app.
    assert d["openai_model"] == "r1"
    assert d["adapter_hf_prefix"] == "sft/r1/seed0/adapter"
    # freesolo serving scales to zero per base model — no flash-side idle billing.
    assert d["est_idle_cost_usd_per_day"] == 0.0


def test_deploy_qlora_dry_run_is_not_rejected():
    """A QLoRA tier (9B) is deployable: freesolo serving folds the bf16 LoRA delta into
    the bf16 base just like a bf16 tier, instead of being rejected up front."""
    from flash.serve.deploy import deploy_adapter

    dep = deploy_adapter(
        run_id="q1",
        model="Qwen/Qwen3.5-9B",
        hf_repo="org/repo",
        adapter_prefix="sft/q1/seed0",
        gpu_name="RTX 5090",
        dry_run=True,
    )
    assert dep.to_dict()["state"] == "dry_run"


def test_deploy_rejects_unsupported_gpu():
    from flash.providers.base import UnsupportedGpuError
    from flash.serve.deploy import deploy_adapter

    with pytest.raises(UnsupportedGpuError):
        deploy_adapter(
            run_id="r1",
            model="Qwen/Qwen3.5-0.8B",
            hf_repo="org/repo",
            adapter_prefix="sft/r1/seed0",
            gpu_name="TPU v5",  # junk still rejects
            dry_run=True,
        )


def test_deploy_registers_with_freesolo_serving(monkeypatch):
    """A non-dry-run deploy POSTs the adapter to {FREESOLO_SERVING_URL}/adapters with the
    right body and the internal-key auth header."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")

    seen = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["url"] = url
        seen["json"] = json
        seen["headers"] = headers
        return _Resp()

    monkeypatch.setattr(d.httpx, "post", fake_post)

    dep = d.deploy_adapter(
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        gpu_name="RTX 5090",
    )
    assert seen["url"] == "https://serve.example/adapters"
    assert seen["json"] == {
        "adapterId": "flash-7-abcd",
        "repoId": "org/repo",
        "baseModel": "Qwen/Qwen3.5-0.8B",
        "subfolder": "sft/flash-7-abcd/seed0/adapter",
        "status": "ready",
    }
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    assert dep.openai_model == "flash-7-abcd"
    assert dep.endpoint_name == "https://serve.example"
    assert dep.state == "ready"


def test_deploy_propagates_serving_error(monkeypatch):
    """A non-2xx from the serving app surfaces (the server maps it to a 5xx)."""
    import flash.serve.deploy as d

    class _Resp:
        status_code = 500

        def raise_for_status(self):
            raise d.httpx.HTTPStatusError("boom", request=None, response=None)

    monkeypatch.setattr(d.httpx, "post", lambda *a, **k: _Resp())
    with pytest.raises(d.httpx.HTTPStatusError):
        d.deploy_adapter("r1", "Qwen/Qwen3.5-0.8B", "org/repo", "sft/r1/seed0", "RTX 5090")


def test_undeploy_deletes_on_freesolo_serving(monkeypatch):
    """undeploy DELETEs {FREESOLO_SERVING_URL}/adapters/{run_id} with the auth header and
    returns [run_id] on success, [] on 404."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")

    seen = {}

    class _Resp:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            return None

    def fake_delete(url, headers=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        return _Resp(200)

    monkeypatch.setattr(d.httpx, "delete", fake_delete)
    out = d.undeploy_adapter("flash-7-abcd", gpu_name="RTX 5090")
    assert out == ["flash-7-abcd"]
    assert seen["url"] == "https://serve.example/adapters/flash-7-abcd"
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"

    # A 404 (already gone) returns an empty list, not an error.
    monkeypatch.setattr(d.httpx, "delete", lambda *a, **k: _Resp(404))
    assert d.undeploy_adapter("flash-7-abcd", gpu_name="RTX 5090") == []


def test_chat_posts_to_freesolo_serving(monkeypatch):
    """chat POSTs to {FREESOLO_SERVING_URL}/v1/chat/completions addressing the adapter by
    run_id, and returns the parsed OpenAI response dict."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")

    seen = {}
    completion = {
        "object": "chat.completion",
        "model": "flash-7-abcd",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi there"}}],
    }

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return completion

    def fake_post(url, json=None, timeout=None):
        seen["url"] = url
        seen["json"] = json
        return _Resp()

    monkeypatch.setattr(d.httpx, "post", fake_post)
    out = d.chat(
        run_id="flash-7-abcd",
        messages=[{"role": "user", "content": "2+2?"}],
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        temperature=0.0,
        max_tokens=8,
    )
    assert seen["url"] == "https://serve.example/v1/chat/completions"
    assert seen["json"]["model"] == "flash-7-abcd"
    assert seen["json"]["max_tokens"] == 8
    assert seen["json"]["messages"] == [{"role": "user", "content": "2+2?"}]
    # The OpenAI shape is preserved so resp["choices"][0]["message"]["content"] works.
    assert out["choices"][0]["message"]["content"] == "hi there"

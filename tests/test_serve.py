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
    assert "mode" not in d
    assert "est_idle_cost_usd_per_day" not in d


def test_deploy_9b_dry_run_is_not_rejected():
    """The 9B (bf16 LoRA) tier is deployable: freesolo serving folds the bf16 LoRA delta
    into the bf16 base, instead of being rejected up front."""
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

    def fake_post(url, json=None, headers=None, timeout=None, follow_redirects=None):
        seen["url"] = url
        seen["json"] = json
        seen["headers"] = headers
        seen["follow_redirects"] = follow_redirects
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
        # flash always uploads adapters to HF *dataset* repos, so serving must be told to
        # pull from the dataset namespace (else snapshot_download 404s on the model namespace).
        "repoType": "dataset",
        "status": "ready",
    }
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    # Modal 303-redirects slow requests to an async-result poll URL, so registration follows them.
    assert seen["follow_redirects"] is True
    assert dep.openai_model == "flash-7-abcd"
    assert dep.endpoint_name == "https://serve.example"
    assert dep.state == "ready"


def test_deploy_includes_org_id_when_provided(monkeypatch):
    """When the deploying org is known, registration carries `orgId` so serving can persist
    hosted_lora_adapters.org_id and later authorize external chat by org. Omitted when unknown."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")

    seen = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_post(url, json=None, headers=None, timeout=None, follow_redirects=None):
        seen["json"] = json
        return _Resp()

    monkeypatch.setattr(d.httpx, "post", fake_post)

    d.deploy_adapter(
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        gpu_name="RTX 5090",
        org_id="org-xyz",
    )
    assert seen["json"]["orgId"] == "org-xyz"

    # No org -> the key is omitted entirely (registration shape unchanged for older callers).
    d.deploy_adapter(
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        gpu_name="RTX 5090",
    )
    assert "orgId" not in seen["json"]


def test_deploy_propagates_serving_error(monkeypatch):
    """A non-2xx from the serving app surfaces as a ServingError (the server maps it to a 502)
    instead of swallowing it or letting a raw httpx error escape as an unhandled 500."""
    import flash.serve.deploy as d

    class _Resp:
        status_code = 500

        def raise_for_status(self):
            raise d.httpx.HTTPStatusError("boom", request=None, response=None)

    monkeypatch.setattr(d.httpx, "post", lambda *a, **k: _Resp())
    with pytest.raises(d.ServingError):
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

    def fake_delete(url, headers=None, timeout=None, follow_redirects=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["follow_redirects"] = follow_redirects
        return _Resp(200)

    monkeypatch.setattr(d.httpx, "delete", fake_delete)
    out = d.undeploy_adapter("flash-7-abcd")
    assert out == ["flash-7-abcd"]
    assert seen["url"] == "https://serve.example/adapters/flash-7-abcd"
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    # Modal 303-redirects slow requests to an async-result poll URL, so undeploy follows them too.
    assert seen["follow_redirects"] is True

    # A 404 (already gone) returns an empty list, not an error.
    monkeypatch.setattr(d.httpx, "delete", lambda *a, **k: _Resp(404))
    assert d.undeploy_adapter("flash-7-abcd") == []


def test_undeploy_propagates_serving_error(monkeypatch):
    """A non-404 failure from the serving app surfaces as a ServingError (carrying the upstream
    status, so the server maps it to a 502) — exactly like deploy — instead of letting a raw
    httpx error escape as an unhandled 500. A 404 still no-ops (already-gone is success)."""
    import flash.serve.deploy as d

    class _Resp:
        def __init__(self, code):
            self.status_code = code
            self.text = "kaboom"

        def raise_for_status(self):
            raise d.httpx.HTTPStatusError("boom", request=None, response=self)

    # Non-404 (500) → ServingError carrying the upstream status, not a raw httpx error.
    monkeypatch.setattr(d.httpx, "delete", lambda *a, **k: _Resp(500))
    with pytest.raises(d.ServingError) as ei:
        d.undeploy_adapter("flash-7-abcd")
    assert ei.value.status_code == 500

    # A transport error (never reached the backend) is also translated into a ServingError.
    # httpx.RequestError must carry the originating request (httpx>=0.27); building it with only a
    # message can raise TypeError before undeploy_adapter() can translate it, so mirror the real
    # undeploy call (DELETE {serving}/adapters/{run_id}).
    def _boom_delete(*a, **k):
        raise d.httpx.RequestError(
            "no route to host",
            request=d.httpx.Request("DELETE", "https://serve.example/adapters/flash-7-abcd"),
        )

    monkeypatch.setattr(d.httpx, "delete", _boom_delete)
    with pytest.raises(d.ServingError):
        d.undeploy_adapter("flash-7-abcd")

    # A 404 short-circuits before raise_for_status(), so it stays a no-op success (not a ServingError).
    monkeypatch.setattr(d.httpx, "delete", lambda *a, **k: _Resp(404))
    assert d.undeploy_adapter("flash-7-abcd") == []


def test_chat_posts_to_freesolo_serving(monkeypatch):
    """chat POSTs to {FREESOLO_SERVING_URL}/v1/chat/completions addressing the adapter by
    run_id, and returns the parsed OpenAI response dict."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")

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

    class _FakeClient:
        # chat() uses an explicit httpx.Client (context manager) so it can follow Modal's 303
        # async-result redirects; the fake records the call and the client kwargs.
        def __init__(self, *args, **kwargs):
            seen["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json=None, headers=None):
            seen["url"] = url
            seen["json"] = json
            seen["headers"] = headers or {}
            return _Resp()

    monkeypatch.setattr(d.httpx, "Client", _FakeClient)
    out = d.chat(
        run_id="flash-7-abcd",
        messages=[{"role": "user", "content": "2+2?"}],
        temperature=0.0,
        max_tokens=8,
        thinking=True,
    )
    assert seen["url"] == "https://serve.example/v1/chat/completions"
    # Modal 303-redirects slow ASGI requests to an async-result poll URL, so the chat client
    # MUST follow redirects (else httpx raises on the 303 mid cold-start).
    assert seen["client_kwargs"]["follow_redirects"] is True
    assert seen["json"]["model"] == "flash-7-abcd"
    assert seen["json"]["max_tokens"] == 8
    assert seen["json"]["messages"] == [{"role": "user", "content": "2+2?"}]
    # Per-run thinking parity: the thinking flag is forwarded to the chat template so a
    # thinking-trained adapter serves with thinking (not silently dropped).
    assert seen["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    # The OpenAI shape is preserved so resp["choices"][0]["message"]["content"] works.
    assert out["choices"][0]["message"]["content"] == "hi there"
    # The control plane is a trusted serving caller, so it presents the internal key — this is
    # what lets `flash chat` keep working when the serving app enforces external chat auth.
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"


def test_with_system_prompt_restores_when_absent_and_preserves_caller():
    """_with_system_prompt prepends the env's training system prompt only when the caller
    sent none (parity restore), and is a no-op when the prompt is empty or already has one."""
    import flash.serve.deploy as d

    user = [{"role": "user", "content": "hi"}]
    # No env prompt available → unchanged.
    assert d._with_system_prompt(user, None) == user
    # Restored when the caller supplied no system message.
    assert d._with_system_prompt(user, "SYS") == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]
    # Caller's own system message wins (never double-prepended / overridden).
    caller_sys = [{"role": "system", "content": "A"}, {"role": "user", "content": "hi"}]
    assert d._with_system_prompt(caller_sys, "SYS") == caller_sys


def test_chat_injects_env_system_prompt_for_parity(monkeypatch):
    """chat() restores the env's training system prompt when the caller sent none, so the
    served model gets the task spec it was trained with (deploy/chat parity)."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    seen = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "{}"}}]}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json=None, headers=None):
            seen["json"] = json
            return _Resp()

    monkeypatch.setattr(d.httpx, "Client", _FakeClient)
    d.chat(
        run_id="flash-7-abcd",
        messages=[{"role": "user", "content": "find founders"}],
        system_prompt="TRAINED-SYS",
    )
    assert seen["json"]["messages"][0] == {"role": "system", "content": "TRAINED-SYS"}
    assert seen["json"]["messages"][1] == {"role": "user", "content": "find founders"}
    # Default token budget is generous (not the old 512) so a reasoning model isn't truncated.
    assert seen["json"]["max_tokens"] == d.DEFAULT_MAX_TOKENS == 2048


def test_chat_stream_yields_openai_sse_content(monkeypatch):
    """chat_stream requests OpenAI streaming and yields assistant content deltas only."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")
    seen = {}

    class _StreamResp:
        def __init__(self):
            self.headers = {"content-type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(
                [
                    'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
                    'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}',
                    'data: {"choices":[{"delta":{"content":" there"},"finish_reason":null}]}',
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            seen["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, method, url, json=None, headers=None):
            seen["method"] = method
            seen["url"] = url
            seen["json"] = json
            seen["headers"] = headers or {}
            return _StreamResp()

    monkeypatch.setattr(d.httpx, "Client", _FakeClient)

    chunks = list(
        d.chat_stream(
            run_id="flash-7-abcd",
            messages=[{"role": "user", "content": "2+2?"}],
            temperature=0.0,
            max_tokens=8,
            thinking=True,
        )
    )

    assert chunks == ["hi", " there"]
    assert seen["client_kwargs"]["follow_redirects"] is True
    assert seen["method"] == "POST"
    # Trusted-caller bypass: chat_stream presents the internal key, like the non-streaming chat.
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    assert seen["url"] == "https://serve.example/v1/chat/completions"
    assert seen["json"]["stream"] is True
    assert seen["json"]["model"] == "flash-7-abcd"
    assert seen["json"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_chat_stream_accepts_json_fallback(monkeypatch):
    """A new Flash server can still talk to an older serving app that ignores stream=true."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")

    class _JsonResp:
        def __init__(self):
            self.headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "full reply"}}]}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, method, url, json=None, headers=None):
            return _JsonResp()

    monkeypatch.setattr(d.httpx, "Client", _FakeClient)

    assert list(d.chat_stream("flash-7-abcd", [{"role": "user", "content": "hi"}])) == [
        "full reply"
    ]

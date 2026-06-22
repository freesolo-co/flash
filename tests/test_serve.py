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


class _ChatResp:
    """A minimal httpx.Response stand-in for the chat client fakes below.

    ``status`` 200 -> a completed completion; a 3xx with a Modal ``__modal_function_call_id=``
    Location -> the async-result redirect that the helper must poll itself.
    """

    def __init__(self, status, *, location=None, body=None, url="https://serve.example/x"):
        self.status_code = status
        self.headers = {"location": location} if location else {}
        self._body = body if body is not None else {"choices": [{"message": {"content": "ok"}}]}
        # resp.url.join(location) is how chat() resolves the next poll URL.
        self.url = _httpx().URL(url)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _httpx().HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._body


def _httpx():
    import httpx

    return httpx


def _install_fake_chat_client(monkeypatch, *, post_resp, poll_resps=()):
    """Patch httpx.Client so chat() drives a scripted POST + sequence of poll GETs.

    Records the client kwargs, the POST url/json, and every poll URL into ``seen``.
    """
    import flash.serve.deploy as d

    seen = {"polls": []}
    polls = list(poll_resps)

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            seen["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json=None, headers=None):
            seen["url"] = url
            seen["json"] = json
            seen["post_headers"] = headers
            return post_resp

        def get(self, url, headers=None):
            seen["polls"].append(url)
            return polls.pop(0)

    monkeypatch.setattr(d.httpx, "Client", _FakeClient)
    return seen


def test_chat_posts_to_freesolo_serving(monkeypatch):
    """chat POSTs to {FREESOLO_SERVING_URL}/v1/chat/completions addressing the adapter by
    run_id, and returns the parsed OpenAI response dict (warm path: a direct 200)."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")

    completion = {
        "object": "chat.completion",
        "model": "flash-7-abcd",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi there"}}],
    }
    seen = _install_fake_chat_client(monkeypatch, post_resp=_ChatResp(200, body=completion))

    out = d.chat(
        run_id="flash-7-abcd",
        messages=[{"role": "user", "content": "2+2?"}],
        temperature=0.0,
        max_tokens=8,
        thinking=True,
    )
    assert seen["url"] == "https://serve.example/v1/chat/completions"
    # The chat client must NOT let httpx chase Modal's async-result redirect chain (it has no
    # global deadline and the poll URL long-polls — that was the >2min hang). We poll ourselves.
    assert seen["client_kwargs"]["follow_redirects"] is False
    # And the per-request socket timeout is bounded (not the old 30-minute-per-hop value).
    assert seen["client_kwargs"]["timeout"] == d.CHAT_HTTP_TIMEOUT_S
    # The internal key rides the chat POST so it reaches the (auth'd) serving backend.
    assert seen["post_headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    assert seen["json"]["model"] == "flash-7-abcd"
    assert seen["json"]["max_tokens"] == 8
    assert seen["json"]["messages"] == [{"role": "user", "content": "2+2?"}]
    # Per-run thinking parity: the thinking flag is forwarded to the chat template so a
    # thinking-trained adapter serves with thinking (not silently dropped).
    assert seen["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    # Warm path performs no async-result polling.
    assert seen["polls"] == []
    # The OpenAI shape is preserved so resp["choices"][0]["message"]["content"] works.
    assert out["choices"][0]["message"]["content"] == "hi there"


def test_chat_polls_modal_async_result_redirect(monkeypatch):
    """On a cold start Modal redirects to a ?__modal_function_call_id= poll URL; chat() must poll
    that URL itself (with our own backoff) and return the eventual completion — instead of
    letting httpx chase the long-polling redirect chain (the old >2min hang)."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(d.time, "sleep", lambda *_a: None)  # don't actually wait between polls

    poll_url = "/v1/chat/completions?__modal_function_call_id=fc-XYZ"
    done = {"choices": [{"message": {"content": "cold-done"}}]}
    seen = _install_fake_chat_client(
        monkeypatch,
        # First the POST 303s to the async-result URL...
        post_resp=_ChatResp(303, location=poll_url),
        # ...then the poll redirects to itself once more, then delivers the result.
        poll_resps=[_ChatResp(302, location=poll_url), _ChatResp(200, body=done)],
    )

    out = d.chat(run_id="flash-7-abcd", messages=[{"role": "user", "content": "hi"}])
    assert out["choices"][0]["message"]["content"] == "cold-done"
    # Polled the resolved async-result URL (twice: pending, then ready).
    assert len(seen["polls"]) == 2
    assert all("__modal_function_call_id=fc-XYZ" in u for u in seen["polls"])


def test_chat_raises_serving_error_on_deadline(monkeypatch):
    """If the async-result poll never resolves before the wall-clock deadline, chat() raises a
    ServingError (carrying a clear timeout message) — it can NEVER hang indefinitely."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_CHAT_TIMEOUT_S", "5")  # tiny budget
    monkeypatch.setattr(d.time, "sleep", lambda *_a: None)
    # A monotonic clock that jumps past the deadline after the first poll.
    ticks = iter([0.0, 100.0, 200.0, 300.0])
    monkeypatch.setattr(d.time, "monotonic", lambda: next(ticks))

    poll_url = "/v1/chat/completions?__modal_function_call_id=fc-NEVER"
    _install_fake_chat_client(
        monkeypatch,
        post_resp=_ChatResp(303, location=poll_url),
        # The poll keeps redirecting (never ready); the deadline must cut it off.
        poll_resps=[_ChatResp(303, location=poll_url) for _ in range(10)],
    )

    with pytest.raises(d.ServingError) as exc:
        d.chat(run_id="flash-7-abcd", messages=[{"role": "user", "content": "hi"}])
    assert "timed out" in str(exc.value)


def test_chat_translates_transport_error_to_serving_error(monkeypatch):
    """A transport failure during chat is surfaced as a ServingError, not a raw httpx error
    escaping (so the API maps it to a clean 502, like deploy/undeploy)."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")

    class _BoomClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *a, **k):
            raise d.httpx.ConnectError(
                "no route", request=d.httpx.Request("POST", "https://serve.example/x")
            )

    monkeypatch.setattr(d.httpx, "Client", _BoomClient)
    with pytest.raises(d.ServingError) as exc:
        d.chat(run_id="flash-7-abcd", messages=[{"role": "user", "content": "hi"}])
    assert "could not reach the serving backend" in str(exc.value)

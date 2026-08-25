from __future__ import annotations

import asyncio
import json

import pytest

import flash.serve.request.transport as serving_transport
from tests._helpers.chat_provenance import (
    managed_chat_result as _managed_chat_result,
)
from tests._helpers.chat_provenance import (
    managed_stream_headers as _managed_stream_headers,
)
from tests._helpers.managed_chat import _deployed_chat_run, _RawManagedChatResponse
from tests.test_server_api import SPEC, _bearer, _login

pytest_plugins = ("tests._helpers.server_api_plugin",)


def test_managed_stream_response_closes_upstream_before_first_body_byte() -> None:
    from flash.server.routes.serving import _UpstreamStreamingResponse

    upstream = _RawManagedChatResponse([b"data: [DONE]\n\n"])
    response = _UpstreamStreamingResponse(
        upstream.iter_bytes(),
        upstream=upstream,
        media_type="text/event-stream",
    )

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        assert message["type"] == "http.response.start"
        raise RuntimeError("downstream disconnected before the first body byte")

    with pytest.raises(RuntimeError, match="before the first body byte"):
        asyncio.run(
            response(
                {"type": "http", "asgi": {"spec_version": "2.4"}, "method": "POST"},
                receive,
                send,
            )
        )

    assert upstream.closed


def test_managed_stream_response_closes_upstream_after_midstream_disconnect() -> None:
    from flash.server.routes.serving import _UpstreamStreamingResponse

    upstream = _RawManagedChatResponse([b"first", b"second"])
    response = _UpstreamStreamingResponse(
        upstream.iter_bytes(),
        upstream=upstream,
        media_type="text/event-stream",
    )
    body_sends = 0

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        nonlocal body_sends
        if message["type"] != "http.response.body":
            return
        body_sends += 1
        raise RuntimeError("downstream disconnected midstream")

    with pytest.raises(RuntimeError, match="disconnected midstream"):
        asyncio.run(
            response(
                {"type": "http", "asgi": {"spec_version": "2.4"}, "method": "POST"},
                receive,
                send,
            )
        )

    assert body_sends == 1
    assert upstream.closed


def test_chat_streams_deployed_run(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    revision = f"{run_id}@final." + "a" * 40
    runner_transitions.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner_verified_revisions.verified_adapter_revision_generation(
            run_id
        ),
    )

    seen = {}

    def fake_stream(**kwargs):
        seen.update(kwargs)
        return _RawManagedChatResponse(
            [
                b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n',
                b'data: {"choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}\n\n',
                b"data: [DONE]\n\n",
            ],
            headers=_managed_stream_headers(revision),
        )

    monkeypatch.setattr(app_mod, "serve_chat_sse", fake_stream)

    with api.stream(
        "POST",
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        headers=_bearer(key),
    ) as resp:
        text = resp.read().decode()

    assert resp.status_code == 200, text
    assert text == (
        'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}\n\n'
        "data: [DONE]\n\n"
    )
    assert seen["run_id"] == revision
    assert seen["messages"] == [{"role": "user", "content": "hello"}]


def test_chat_streams_verified_immutable_revision_unchanged(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    revision = f"{run_id}@final." + "a" * 40
    status = runner_status.get_status(run_id)
    status.state = "deployed"
    status.deployment = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "adapter_revision": revision,
    }
    runner_state._save_status(status)
    generation = runner_verified_revisions.verified_adapter_revision_generation(run_id)
    assert runner_verified_revisions.add_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=generation,
    )
    seen = {}

    def fake_stream(**kwargs):
        seen.update(kwargs)
        return _RawManagedChatResponse(
            [
                b'data: {"choices":[{"index":0,"delta":{"content":"verified"}}]}\n\n',
                b"data: [DONE]\n\n",
            ],
            headers=_managed_stream_headers(revision),
        )

    monkeypatch.setattr(app_mod, "serve_chat_sse", fake_stream)

    with api.stream(
        "POST",
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "adapter_revision": revision,
            "stream": True,
        },
        headers=_bearer(key),
    ) as response:
        text = response.read().decode()

    assert response.status_code == 200, text
    assert text == (
        'data: {"choices":[{"index":0,"delta":{"content":"verified"}}]}\n\ndata: [DONE]\n\n'
    )
    assert seen["run_id"] == revision


def test_chat_forwards_supported_openai_fields_and_enforces_run_contract(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions
    import flash.server.asgi.app as app_mod

    key = _login()
    spec = json.loads(json.dumps(SPEC))
    spec["train"] = {**spec["train"], "stop_sequences": ["</answer>", "shared"]}
    run_id = api.post(
        "/v1/runs", json={"spec": spec, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    revision = f"{run_id}@final." + "a" * 40
    runner_transitions.mark_deployed(
        run_id,
        {"state": "ready", "endpoint_name": "https://serve.example", "adapter_revision": revision},
        verification_generation=runner_verified_revisions.verified_adapter_revision_generation(
            run_id
        ),
    )
    seen = {}

    def fake_chat(**kwargs):
        seen.update(kwargs)
        return {
            "id": "chatcmpl-native",
            "object": "chat.completion",
            "model": "backend-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            "flash_provenance": {
                "adapter_revision": revision,
                "checkpoint": run_id,
                "source_revision": "a" * 40,
                "native": True,
            },
        }

    monkeypatch.setattr(app_mod, "serve_chat", fake_chat)
    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "model": "caller-must-not-select-this",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
            "max_tokens": 17,
            "top_p": 0.8,
            "stop": ["shared", "caller-stop"],
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {
                "enable_thinking": True,
                "custom_template_flag": "kept",
                "return_tensors": "must-be-dropped",
            },
        },
        headers=_bearer(key),
    )

    assert response.status_code == 200, response.text
    deployed_revision = runner_status.get_status(run_id).deployment["adapter_revision"]
    assert seen == {
        "run_id": deployed_revision,
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "max_tokens": 17,
        "thinking": False,
        "top_p": 0.8,
        "stop": ["</answer>", "shared", "caller-stop"],
        "chat_template_kwargs": {
            "custom_template_flag": "kept",
            "enable_thinking": False,
        },
        "structured_outputs": {"json_object": True},
    }
    payload = response.json()
    assert payload["id"] == "chatcmpl-native"
    assert payload["model"] == "backend-model"
    assert payload["flash_provenance"]["native"] is True
    assert payload["freesolo"] == {
        "adapter_revision": deployed_revision,
        "checkpoint": run_id,
        "hf_revision": "a" * 40,
    }


def test_chat_rejects_success_without_backend_provenance(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **_kwargs: {"choices": [{"message": {"content": "ok"}}]},
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers=_bearer(key),
    )

    assert response.status_code == 502
    assert "omitted immutable provenance" in response.json()["detail"]


def test_chat_rejects_mismatched_backend_provenance(api, monkeypatch):
    import flash.runner.lifecycle.status as runner_status
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    revision = runner_status.get_status(run_id).deployment["adapter_revision"]
    payload = _managed_chat_result(revision)
    payload["flash_provenance"]["source_revision"] = "b" * 40
    monkeypatch.setattr(app_mod, "serve_chat", lambda **_kwargs: payload)

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers=_bearer(key),
    )

    assert response.status_code == 502
    assert "mismatched immutable source revision" in response.json()["detail"]


@pytest.mark.parametrize(
    "unsupported",
    [
        "tools",
        "tool_choice",
        "n",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "top_logprobs",
        "service_tier",
        "modalities",
        "enable_thinking",
        "unknown_field",
    ],
)
def test_chat_rejects_unsupported_top_level_fields(api, monkeypatch, unsupported):
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **_kwargs: pytest.fail("unsupported fields must not reach serving"),
    )
    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            unsupported: [] if unsupported in {"tools", "modalities"} else True,
        },
        headers=_bearer(key),
    )

    assert response.status_code == 400
    assert unsupported in response.json()["detail"]


def test_chat_rejects_conflicting_structured_forms_and_invalid_stop(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **_kwargs: pytest.fail("invalid chat contracts must not reach serving"),
    )
    cases = [
        {
            "structured_outputs": {"json_object": True},
            "response_format": {"type": "text"},
        },
        {"stop": ""},
        {"stop": ["ok", ""]},
        {"stop": 3},
        {"chat_template_kwargs": []},
        {"stream_options": {"include_usage": True}},
    ]
    for extra in cases:
        response = api.post(
            f"/v1/runs/{run_id}/chat",
            json={"messages": [{"role": "user", "content": "hello"}], **extra},
            headers=_bearer(key),
        )
        assert response.status_code == 400, (extra, response.text)

    raw = '{"messages":[{"role":"user","content":"hello"}],"chat_template_kwargs":{"bad":NaN}}'
    response = api.post(
        f"/v1/runs/{run_id}/chat",
        content=raw,
        headers={**_bearer(key), "content-type": "application/json"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("strict", [None, True])
def test_chat_accepts_strict_json_schema_response_format(api, monkeypatch, strict):
    import flash.runner.lifecycle.status as runner_status
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    revision = runner_status.get_status(run_id).deployment["adapter_revision"]
    seen = {}

    def fake_chat(**kwargs):
        seen.update(kwargs)
        return _managed_chat_result(kwargs["run_id"])

    monkeypatch.setattr(app_mod, "serve_chat", fake_chat)
    declaration = {
        "name": "answer",
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
    }
    if strict is not None:
        declaration["strict"] = strict
    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "json_schema", "json_schema": declaration},
        },
        headers=_bearer(key),
    )

    assert response.status_code == 200, response.text
    assert seen["run_id"] == revision
    assert seen["structured_outputs"] == {"json": declaration["schema"]}


def test_chat_rejects_non_strict_json_schema_response_format(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **_kwargs: pytest.fail("strict=false must not reach serving"),
    )
    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": False,
                },
            },
        },
        headers=_bearer(key),
    )

    assert response.status_code == 400
    assert "strict=false is not supported" in response.json()["detail"]


def test_chat_stream_preserves_raw_openai_sse_and_provenance_headers(api, monkeypatch):
    import flash.runner.lifecycle.status as runner_status
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    revision = runner_status.get_status(run_id).deployment["adapter_revision"]
    frames = [
        b'data: {"id":"chatcmpl-1","choices":[{"index":2,"delta":{"content":"hi"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-1","choices":[{"index":2,"delta":{},"finish_reason":"length"}]}\n\n',
        b'data: {"id":"chatcmpl-1","choices":[],"usage":{"total_tokens":9}}\n\n',
        b"data: [DONE]\n\n",
    ]
    seen = {}
    closed = []

    class RawResponse:
        status_code = 200

        def __init__(self):
            self.headers = {
                **_managed_stream_headers(revision),
                "x-backend-native": "preserved",
            }

        def iter_bytes(self):
            try:
                yield from frames
            finally:
                self.close()

        def close(self):
            if not closed:
                closed.append(True)

    def fake_sse(**kwargs):
        seen.update(kwargs)
        return RawResponse()

    monkeypatch.setattr(app_mod, "serve_chat_sse", fake_sse)
    with api.stream(
        "POST",
        f"/v1/runs/{run_id}/chat",
        json={
            "model": "ignored",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "top_p": 0.7,
            "stream_options": {"include_usage": True},
        },
        headers=_bearer(key),
    ) as response:
        body = response.read()

    revision = runner_status.get_status(run_id).deployment["adapter_revision"]
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-backend-native"] == "preserved"
    assert response.headers["x-freesolo-adapter-revision"] == revision
    assert response.headers["x-freesolo-checkpoint"] == run_id
    assert response.headers["x-freesolo-hf-revision"] == "a" * 40
    assert "x-freesolo-lora-request-adapter" not in response.headers
    assert body == b"".join(frames)
    assert closed == [True]
    assert seen["run_id"] == revision
    assert seen["top_p"] == 0.7
    assert seen["stream_options"] == {"include_usage": True}


def test_chat_stream_strips_hop_by_hop_and_connection_extension_headers(api, monkeypatch):
    import flash.runner.lifecycle.status as runner_status
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    revision = runner_status.get_status(run_id).deployment["adapter_revision"]
    upstream = _RawManagedChatResponse(
        [b"data: [DONE]\n\n"],
        headers={
            **_managed_stream_headers(revision),
            "connection": "keep-alive, X-Remove-Me",
            "keep-alive": "timeout=5",
            "proxy-authenticate": "Basic realm=test",
            "proxy-authorization": "secret",
            "te": "trailers",
            "trailer": "x-checksum",
            "transfer-encoding": "chunked",
            "upgrade": "websocket",
            "x-freesolo-lora-request-adapter": revision,
            "x-remove-me": "connection extension",
            "x-preserved": "yes",
        },
    )
    monkeypatch.setattr(app_mod, "serve_chat_sse", lambda **_kwargs: upstream)

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        headers=_bearer(key),
    )

    assert response.status_code == 200, response.text
    assert response.headers["x-preserved"] == "yes"
    for header in (
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-freesolo-lora-request-adapter",
        "x-remove-me",
    ):
        assert header not in response.headers


@pytest.mark.parametrize("case", ["missing", "mismatch"])
def test_chat_stream_rejects_missing_or_mismatched_adapter_attestation(api, monkeypatch, case):
    import flash.runner.lifecycle.status as runner_status
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    revision = runner_status.get_status(run_id).deployment["adapter_revision"]
    headers = _managed_stream_headers(revision)
    if case == "missing":
        del headers["x-freesolo-lora-request-adapter"]
    else:
        headers["x-freesolo-lora-request-adapter"] = f"{run_id}@final." + "b" * 40
    upstream = _RawManagedChatResponse([b"data: [DONE]\n\n"], headers=headers)
    monkeypatch.setattr(app_mod, "serve_chat_sse", lambda **_kwargs: upstream)

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        headers=_bearer(key),
    )

    assert response.status_code == 502
    expected = "omitted" if case == "missing" else "mismatched"
    assert f"{expected} LoRA request adapter attestation" in response.json()["detail"]
    assert upstream.closed


@pytest.mark.parametrize("case", ["missing", "mismatch"])
def test_chat_stream_rejects_incomplete_backend_provenance(api, monkeypatch, case):
    import flash.runner.lifecycle.status as runner_status
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    revision = runner_status.get_status(run_id).deployment["adapter_revision"]
    headers = _managed_stream_headers(revision)
    if case == "missing":
        del headers["x-flash-source-revision"]
    else:
        headers["x-flash-source-revision"] = "b" * 40
    upstream = _RawManagedChatResponse([b"data: [DONE]\n\n"], headers=headers)
    monkeypatch.setattr(app_mod, "serve_chat_sse", lambda **_kwargs: upstream)

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        headers=_bearer(key),
    )

    assert response.status_code == 502
    expected = (
        "omitted hf_revision provenance"
        if case == "missing"
        else "mismatched hf_revision provenance"
    )
    assert expected in response.json()["detail"]
    assert upstream.closed


def test_chat_stream_rejects_embedded_event_stream_media_type(api, monkeypatch):
    import flash.runner.lifecycle.status as runner_status
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    revision = runner_status.get_status(run_id).deployment["adapter_revision"]
    upstream = _RawManagedChatResponse(
        [b"data: [DONE]\n\n"],
        headers={
            **_managed_stream_headers(revision),
            "content-type": "application/x-text/event-stream",
        },
    )
    monkeypatch.setattr(app_mod, "serve_chat_sse", lambda **_kwargs: upstream)

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        headers=_bearer(key),
    )

    assert response.status_code == 502
    assert "non-sse streaming response" in response.json()["detail"]
    assert upstream.closed


@pytest.mark.parametrize(
    ("status_code", "headers", "body"),
    [
        (302, {"content-type": "application/json", "location": "/moved"}, {"moved": True}),
        (
            429,
            {"content-type": "application/json", "retry-after": "3"},
            {"error": {"message": "rate limited"}},
        ),
    ],
)
def test_chat_stream_preserves_upstream_non_success_status_headers_and_body(
    api, monkeypatch, status_code, headers, body
):
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    upstream = _RawManagedChatResponse(
        [json.dumps(body).encode()],
        status_code=status_code,
        headers=headers,
    )
    monkeypatch.setattr(app_mod, "serve_chat_sse", lambda **_kwargs: upstream)
    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        headers=_bearer(key),
        follow_redirects=False,
    )

    assert response.status_code == status_code
    assert response.headers["content-type"].startswith("application/json")
    for name, value in headers.items():
        assert response.headers[name] == value
    assert "x-freesolo-adapter-revision" not in response.headers
    assert "x-freesolo-checkpoint" not in response.headers
    assert "x-freesolo-hf-revision" not in response.headers
    assert response.json() == body


def test_chat_internal_key_requires_matching_org_and_project_scope(api, monkeypatch):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    import flash.runner.results.verified_revisions as runner_verified_revisions
    import flash.runner.supervise.transitions as runner_transitions
    import flash.server.asgi.app as app_mod

    internal = _bearer("fslo-internal-test")
    project = "33333333-3333-4333-8333-333333333333"
    run_id = api.post(
        "/v1/runs",
        json={"spec": {**SPEC, "project": project}, "dry_run": True},
        headers=internal,
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    status.platform_context = {"org_id": "org-chat"}
    status.billing_context = None
    runner_state._save_status(status)
    revision = f"{run_id}@final." + "a" * 40
    runner_transitions.mark_deployed(
        run_id,
        {"state": "ready", "endpoint_name": "https://serve.example", "adapter_revision": revision},
        verification_generation=runner_verified_revisions.verified_adapter_revision_generation(
            run_id
        ),
    )
    calls = []
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: calls.append(kwargs) or _managed_chat_result(kwargs["run_id"]),
    )
    body = {"messages": [{"role": "user", "content": "hello"}]}
    for headers in (
        internal,
        {**internal, "X-Freesolo-Org-Id": "org-wrong", "X-Freesolo-Project-Id": project},
        {**internal, "X-Freesolo-Org-Id": "org-chat"},
        {
            **internal,
            "X-Freesolo-Org-Id": "org-chat",
            "X-Freesolo-Project-Id": "project-wrong",
        },
    ):
        response = api.post(f"/v1/runs/{run_id}/chat", json=body, headers=headers)
        assert response.status_code == 404
        assert response.json() == {"detail": f"unknown run_id: {run_id}"}
    assert calls == []

    matched = {
        **internal,
        "X-Freesolo-Org-Id": "org-chat",
        "X-Freesolo-Project-Id": project,
    }
    response = api.post(f"/v1/runs/{run_id}/chat", json=body, headers=matched)
    assert response.status_code == 200, response.text
    assert len(calls) == 1


def test_chat_stream_upstream_error_before_first_byte_is_502(api, monkeypatch):
    """The streaming branch returns a real 502 when the upstream request fails at start.

    Drives the REAL chat_stream (only the httpx seams are stubbed): a lazy generator whose
    request and raise_for_status run only once Starlette iterates the body does so after the
    200 has been flushed, so the route's except can never fire and an upstream 502 arrives as
    an empty success."""
    import flash.serve.deployment.deploy as deploy

    key, run_id = _deployed_chat_run(api)

    class _ErrorResp:
        def __init__(self):
            self.headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            request = deploy.httpx.Request("POST", "https://serve.example/v1/chat/completions")
            response = deploy.httpx.Response(502, request=request)
            raise deploy.httpx.HTTPStatusError("bad gateway", request=request, response=response)

    class _FakeClient:
        def stream(self, method, url, **kwargs):
            return _ErrorResp()

    monkeypatch.setattr(serving_transport, "_chat_http_client", lambda: _FakeClient())
    monkeypatch.setattr(
        serving_transport, "serving_openai_base_url", lambda: "https://serve.example/v1"
    )

    resp = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        headers=_bearer(key),
    )
    assert resp.status_code == 502, resp.text
    assert "inference failure" in resp.json()["detail"]


def test_chat_stream_midstream_failure_aborts_response(api, monkeypatch):
    """A failure after the first streamed chunk aborts the response, not a clean eof.

    Once bytes are flowing the status is committed, so the only legible signal is an aborted
    body: the exception must propagate out of the response iterator (uvicorn then drops the
    connection without the terminating chunk) rather than being swallowed into an eof the
    client would read as a finished answer."""
    import flash.runner.lifecycle.status as runner_status
    import flash.server.asgi.app as app_mod

    key, run_id = _deployed_chat_run(api)
    revision = runner_status.get_status(run_id).deployment["adapter_revision"]

    def chunks():
        yield b'data: {"choices":[{"index":0,"delta":{"content":"partial "}}]}\n\n'
        raise RuntimeError("upstream failed mid-generation")

    monkeypatch.setattr(
        app_mod,
        "serve_chat_sse",
        lambda **_kwargs: _RawManagedChatResponse(
            chunks(),
            headers=_managed_stream_headers(revision),
        ),
    )

    with (
        pytest.raises(RuntimeError, match="upstream failed mid-generation"),
        api.stream(
            "POST",
            f"/v1/runs/{run_id}/chat",
            json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
            headers=_bearer(key),
        ) as resp,
    ):
        resp.read()

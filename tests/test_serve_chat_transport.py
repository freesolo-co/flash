from __future__ import annotations

import pytest

import flash.serve.request.transport as transport
from flash.serve.contract.errors import RetryableServingUnavailable
from tests._helpers.chat_transport import StreamClient, StreamContext, StreamResponse


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }
    ]


@pytest.fixture(autouse=True)
def _reset_chat_clients(monkeypatch):
    monkeypatch.setattr(transport, "_CHAT_HTTP_CLIENT", None)


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("text/event-stream", True),
        ("Text/Event-Stream; Charset=UTF-8", True),
        ("application/x-text/event-stream", False),
        ("text/event-stream-extra", False),
        ("", False),
    ],
)
def test_event_stream_content_type_requires_exact_media_type(content_type, expected):
    assert transport.is_event_stream_content_type(content_type) is expected


def test_chat_classifies_retryable_alias_smoke_503_for_the_expected_revision(monkeypatch):
    import flash.serve.deployment.deploy as d

    run_id = "run-1"
    revision = f"{run_id}@final." + "a" * 40

    class Response:
        status_code = 503

        def __init__(self):
            self.headers = {"Retry-After": "1.5"}

        def json(self):
            return {
                "error": {
                    "type": "adapter_unavailable",
                    "code": "adapter_loading",
                    "message": "adapter revision is loading",
                    "retryable": True,
                    "requested_model": run_id,
                    "adapter_revision": revision,
                    "retry_after_seconds": 2,
                }
            }

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(d.httpx, "Client", Client)

    with pytest.raises(RetryableServingUnavailable) as exc_info:
        d.chat(
            run_id,
            [{"role": "user", "content": "hello"}],
            expected_adapter_revision=revision,
            timeout_s=5.0,
            retry_unavailable=True,
        )

    assert exc_info.value.code == "adapter_loading"
    assert exc_info.value.retry_after_seconds == 1.5


@pytest.mark.parametrize(
    "error",
    [
        {
            "type": "adapter_unavailable",
            "code": "adapter_load_failed",
            "retryable": True,
        },
        {
            "type": "adapter_unavailable",
            "code": "adapter_loading",
            "retryable": False,
        },
    ],
)
def test_chat_fails_closed_for_unrecognized_smoke_503(monkeypatch, error):
    import flash.serve.deployment.deploy as d

    revision = "run-1@final." + "a" * 40

    class Response:
        status_code = 503

        def __init__(self):
            self.headers = {"Retry-After": "1"}
            self.request = d.httpx.Request("POST", "https://serve.example/v1/chat/completions")

        def json(self):
            return {
                "error": {
                    **error,
                    "requested_model": revision,
                    "adapter_revision": revision,
                }
            }

        def raise_for_status(self):
            raise d.httpx.HTTPStatusError("unavailable", request=self.request, response=self)

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(d.httpx, "Client", Client)

    with pytest.raises(d.httpx.HTTPStatusError):
        d.chat(
            revision,
            [{"role": "user", "content": "hello"}],
            timeout_s=5.0,
            retry_unavailable=True,
        )


def test_chat_posts_to_freesolo_serving(monkeypatch):
    """A terminal /v1 override produces one OpenAI path for direct chat."""
    import flash.serve.deployment.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/v1/")
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
        # chat() uses an explicit httpx client context manager so it can follow modal's 303
        # async-result redirects; the fake records the call and the client kwargs.
        def __init__(self, *args, **kwargs):
            seen["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json=None, headers=None, timeout=None):
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
    # modal 303-redirects slow asgi requests to an async-result poll url, so the chat client
    # must follow redirects (else httpx raises on the 303 mid cold-start).
    assert seen["client_kwargs"]["follow_redirects"] is True
    assert seen["json"]["model"] == "flash-7-abcd"
    assert seen["json"]["max_tokens"] == 8
    assert seen["json"]["messages"] == [{"role": "user", "content": "2+2?"}]
    # per-run thinking parity: the thinking flag is forwarded to the chat template so a
    # thinking-trained adapter serves with thinking (not silently dropped).
    assert seen["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    # the openai shape is preserved so resp["choices"][0]["message"]["content"] works.
    assert out["choices"][0]["message"]["content"] == "hi there"
    # the control plane is a trusted serving caller, so it presents the internal key; this is
    # what lets `flash chat` keep working when the serving app enforces external chat auth.
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"


@pytest.mark.parametrize("helper_name", ["chat", "chat_sse"])
@pytest.mark.parametrize(
    "controls",
    [
        {"tool_choice": "none"},
        {"parallel_tool_calls": True},
    ],
    ids=["tool-choice", "parallel-tool-calls"],
)
def test_public_chat_helpers_reject_tool_controls_without_tools(
    monkeypatch, helper_name: str, controls: dict[str, object]
) -> None:
    import flash.serve.deployment.deploy as d

    class Client:
        def post(self, *_args, **_kwargs):
            pytest.fail("invalid tool controls must not reach buffered transport")

        def stream(self, *_args, **_kwargs):
            pytest.fail("invalid tool controls must not reach streaming transport")

    monkeypatch.setattr(transport, "_chat_http_client", Client)
    helper = getattr(d, helper_name)

    with pytest.raises(ValueError, match="tool controls require tools"):
        helper("run-1", [{"role": "user", "content": "hi"}], **controls)


def test_chat_preserves_explicit_empty_structured_override_and_omits_none(monkeypatch):
    import flash.serve.deployment.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")
    requests = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class Client:
        def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return Response()

    monkeypatch.setattr(transport, "_chat_http_client", Client)

    messages = [{"role": "user", "content": "hello"}]
    d.chat("run-1", messages, structured_outputs={})
    d.chat("run-1", messages)
    d.chat("run-1", messages, tools=_tools())

    first_url, first = requests[0]
    second_url, second = requests[1]
    third_url, third = requests[2]
    assert first_url == second_url == third_url == "https://serve.example/v1/chat/completions"
    assert first["json"]["structured_outputs"] == {}
    assert "structured_outputs" not in second["json"]
    assert third["json"]["tool_choice"] == "auto"
    assert third["json"]["parallel_tool_calls"] is True
    assert first["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    assert second["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"


def test_chat_sse_preserves_pre_tool_positional_order(monkeypatch):
    import flash.serve.deployment.deploy as d

    seen = {}
    exits = []

    upstream = StreamResponse(
        byte_chunks=(
            b'data: {"id":"one","choices":[{"index":3,"delta":{"content":"a"}}]}\n',
            b'\ndata: {"id":"one","choices":[{"index":3,"finish_reason":"length"}]}\n\n',
            b"data: [DONE]\n\n",
        ),
        headers={
            "content-type": "text/event-stream",
            "x-freesolo-adapter-revision": "run-1@final." + "a" * 40,
        },
    )
    client = StreamClient(StreamContext(upstream, exits), seen)
    monkeypatch.setattr(transport, "_chat_http_client", lambda: client)
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")
    response = d.chat_sse(
        "run-1",
        [{"role": "user", "content": "hi"}],
        0.2,
        9,
        False,
        0.7,
        ["end"],
        {"enable_thinking": False, "custom": 1},
        {"json_object": True},
        {"include_usage": True},
        2,
        42,
        -0.5,
        0.75,
        True,
        3,
    )

    frames = list(response.iter_bytes())

    assert frames == [
        b'data: {"id":"one","choices":[{"index":3,"delta":{"content":"a"}}]}\n\n',
        b'data: {"id":"one","choices":[{"index":3,"finish_reason":"length"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    assert response.status_code == 200
    assert seen["json"] == {
        "model": "run-1",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 9,
        "temperature": 0.2,
        "top_p": 0.7,
        "n": 2,
        "seed": 42,
        "frequency_penalty": -0.5,
        "presence_penalty": 0.75,
        "logprobs": True,
        "top_logprobs": 3,
        "chat_template_kwargs": {"enable_thinking": False, "custom": 1},
        "stream": True,
        "stop": ["end"],
        "structured_outputs": {"json_object": True},
        "stream_options": {"include_usage": True},
    }
    assert len(exits) == 1


def test_chat_sse_defaults_tool_controls(monkeypatch):
    import flash.serve.deployment.deploy as d

    seen = {}
    exits = []
    upstream = StreamResponse(byte_chunks=(b"data: [DONE]\n\n",))
    client = StreamClient(StreamContext(upstream, exits), seen)
    monkeypatch.setattr(transport, "_chat_http_client", lambda: client)
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")

    response = d.chat_sse(
        "run-1",
        [{"role": "user", "content": "weather"}],
        tools=_tools(),
    )

    assert list(response.iter_bytes()) == [b"data: [DONE]\n\n"]
    assert seen["json"]["tool_choice"] == "auto"
    assert seen["json"]["parallel_tool_calls"] is True
    assert len(exits) == 1


def test_chat_sse_stops_at_done_and_ignores_trailing_incomplete_bytes(monkeypatch):
    import flash.serve.deployment.deploy as d

    terminal = b"data: [DONE]\r\n\r\n"
    exits = []
    upstream = StreamResponse(byte_chunks=(terminal + b"trailing incomplete bytes",))
    client = StreamClient(StreamContext(upstream, exits))
    monkeypatch.setattr(transport, "_chat_http_client", lambda: client)
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")

    frames = list(d.chat_sse("run-1", [{"role": "user", "content": "hi"}]).iter_bytes())

    assert frames == [terminal]
    assert len(exits) == 1


def test_chat_sse_stops_at_error_without_reading_trailing_bytes(monkeypatch):
    import flash.serve.deployment.deploy as d

    terminal = b'data: {"error":{"message":"engine failed"}}\n\n'
    trailing_reads = []
    exits = []

    def chunks():
        yield terminal
        trailing_reads.append(True)
        yield b"trailing garbage"

    upstream = StreamResponse(byte_chunks=chunks())
    client = StreamClient(StreamContext(upstream, exits))
    monkeypatch.setattr(transport, "_chat_http_client", lambda: client)
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")

    frames = list(d.chat_sse("run-1", [{"role": "user", "content": "hi"}]).iter_bytes())

    assert frames == [terminal]
    assert trailing_reads == []
    assert len(exits) == 1


def test_chat_sse_close_before_first_byte_releases_upstream(monkeypatch):
    import flash.serve.deployment.deploy as d

    exits = []

    upstream = StreamResponse(byte_chunks=(b"data: [DONE]\n\n",))
    client = StreamClient(StreamContext(upstream, exits))
    monkeypatch.setattr(transport, "_chat_http_client", lambda: client)
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")
    response = d.chat_sse("run-1", [{"role": "user", "content": "hi"}])
    body = response.iter_bytes()

    body.close()

    assert len(exits) == 1


def test_chat_sse_close_without_exhaustion_releases_upstream(monkeypatch):
    import flash.serve.deployment.deploy as d

    exits = []

    upstream = StreamResponse(byte_chunks=(b'data: {"choices":[]}\n\n', b"data: [DONE]\n\n"))
    client = StreamClient(StreamContext(upstream, exits))
    monkeypatch.setattr(transport, "_chat_http_client", lambda: client)
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")
    response = d.chat_sse("run-1", [{"role": "user", "content": "hi"}])
    body = response.iter_bytes()

    assert next(body) == b'data: {"choices":[]}\n\n'
    body.close()
    assert len(exits) == 1


def test_chat_sse_complete_frame_without_done_raises_after_preserving_bytes(monkeypatch):
    import flash.serve.deployment.deploy as d
    from flash.client.http import ClientError

    frame = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    upstream = StreamResponse(byte_chunks=(frame,))
    client = StreamClient(StreamContext(upstream))
    monkeypatch.setattr(transport, "_chat_http_client", lambda: client)
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")
    stream = d.chat_sse("run-1", [{"role": "user", "content": "hi"}]).iter_bytes()

    assert next(stream) == frame
    with pytest.raises(ClientError, match=r"terminal \[DONE\]"):
        next(stream)


def test_chat_preserves_pre_managed_parity_positional_order(monkeypatch):
    import flash.serve.deployment.deploy as d

    seen = {}

    class Response:
        def __init__(self):
            self.headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class Client:
        def post(self, url, **kwargs):
            seen.update({"url": url, **kwargs})
            return Response()

    monkeypatch.setattr(transport, "_chat_http_client", lambda: Client())
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")
    d.chat(
        "run-1",
        [{"role": "user", "content": "hi"}],
        0.2,
        17,
        True,
        None,
        None,
        9.0,
        False,
        ["end"],
        {"regex": "[ab]+"},
        0.6,
        {"custom": 1},
        2,
        42,
        -0.5,
        0.75,
        True,
        3,
    )

    assert seen["timeout"] == 9.0
    assert seen["json"] == {
        "model": "run-1",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 17,
        "temperature": 0.2,
        "top_p": 0.6,
        "n": 2,
        "seed": 42,
        "frequency_penalty": -0.5,
        "presence_penalty": 0.75,
        "logprobs": True,
        "top_logprobs": 3,
        "chat_template_kwargs": {"custom": 1},
        "stop": ["end"],
        "structured_outputs": {"regex": "[ab]+"},
    }


def test_chat_posts_supported_managed_fields(monkeypatch):
    import flash.serve.deployment.deploy as d

    seen = {}

    class _Response:
        def __init__(self):
            self.headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _Client:
        def post(self, url, **kwargs):
            seen.update({"url": url, **kwargs})
            return _Response()

    monkeypatch.setattr(transport, "_chat_http_client", lambda: _Client())
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")
    d.chat(
        "run-1",
        [{"role": "user", "content": "hi"}],
        top_p=0.6,
        stop=["end"],
        chat_template_kwargs={"enable_thinking": True, "custom": 2},
        structured_outputs={"regex": "[ab]+"},
    )

    assert seen["json"] == {
        "model": "run-1",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 0.6,
        "n": 1,
        "seed": None,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logprobs": False,
        "top_logprobs": 0,
        "chat_template_kwargs": {"enable_thinking": True, "custom": 2},
        "stop": ["end"],
        "structured_outputs": {"regex": "[ab]+"},
    }


def test_chat_stream_yields_openai_sse_content(monkeypatch):
    """A terminal /v1 override produces one OpenAI path for streaming chat."""
    import flash.serve.deployment.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/v1/")
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
                    "",
                    'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}',
                    "",
                    'data: {"choices":[{"delta":{"content":" there"},"finish_reason":null}]}',
                    "",
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                    "",
                    "data: [DONE]",
                    "",
                ]
            )

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            seen["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, method, url, json=None, headers=None, timeout=None):
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
    # trusted-caller bypass: chat_stream presents the internal key, like the non-streaming chat.
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    assert seen["url"] == "https://serve.example/v1/chat/completions"
    assert seen["json"]["stream"] is True
    assert seen["json"]["model"] == "flash-7-abcd"
    assert seen["json"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_chat_stream_uses_the_thinking_delimiter_patch_seam(monkeypatch):
    import flash.serve.deployment.deploy as d
    import flash.serve.request.thinking as thinking

    calls = []
    real_find_delimiter = thinking._find_delimiter

    def counting_find(buffer: str, start: int) -> int:
        calls.append((buffer, start))
        return real_find_delimiter(buffer, start)

    upstream = StreamResponse(
        line_chunks=(
            'data: {"choices":[{"delta":{"reasoning_content":"reason"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"reason"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"</think>"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"answer"}}]}',
            "",
            "data: [DONE]",
            "",
        )
    )
    client = StreamClient(StreamContext(upstream))
    monkeypatch.setattr(transport, "_chat_http_client", lambda: client)
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")
    monkeypatch.setattr(thinking, "_find_delimiter", counting_find)

    assert (
        "".join(d.chat_stream("run-1", [{"role": "user", "content": "hi"}], thinking=True))
        == "<think>reason</think>answer"
    )
    assert calls


def test_chat_stream_rejects_sse_eof_without_done(monkeypatch):
    import flash.serve.deployment.deploy as d
    from flash.client.http import ClientError

    class _StreamResp:
        def __init__(self):
            self.headers = {"content-type": "text/event-stream"}

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}'
            yield ""

    class _Context:
        def __enter__(self):
            return _StreamResp()

        def __exit__(self, *_exc):
            return False

    _erroring_stream_seams(monkeypatch, _Context())
    stream = d.chat_stream("run-1", [{"role": "user", "content": "hi"}])

    assert next(stream) == "partial"
    with pytest.raises(ClientError, match=r"terminal \[DONE\]"):
        next(stream)


@pytest.mark.parametrize(
    "content_type",
    ["application/json", "Application/JSON; Charset=UTF-8"],
)
def test_chat_stream_accepts_json_fallback(monkeypatch, content_type):
    """A new Flash server can still talk to an older serving app that ignores stream=true.

    Drives a REAL httpx streaming response (MockTransport) so the read-before-.json() contract is
    actually exercised — a stub with a bare .json() would mask the ResponseNotRead bug.
    """
    import httpx

    import flash.serve.deployment.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            json={"choices": [{"message": {"content": "full reply"}}]},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(d.httpx, "Client", _client)

    assert list(d.chat_stream("flash-7-abcd", [{"role": "user", "content": "hi"}])) == [
        "full reply"
    ]


def test_chat_stream_does_not_treat_a_non_json_media_type_as_json(monkeypatch):
    import flash.serve.deployment.deploy as d

    upstream = StreamResponse(
        headers={"content-type": "text/application/json"},
        line_chunks=(
            'data: {"choices":[{"delta":{"content":"streamed"}}]}',
            "",
            "data: [DONE]",
            "",
        ),
    )
    client = StreamClient(StreamContext(upstream))
    monkeypatch.setattr(transport, "_chat_http_client", lambda: client)
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")

    assert list(d.chat_stream("run-1", [{"role": "user", "content": "hi"}])) == ["streamed"]


def _erroring_stream_seams(monkeypatch, resp):
    """Point the chat_stream seams at a fake client whose stream() yields ``resp``."""

    class _FakeClient:
        def stream(self, method, url, **kwargs):
            return resp

    monkeypatch.setattr(transport, "_chat_http_client", lambda: _FakeClient())
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "https://serve.example/v1")


def test_chat_stream_upstream_error_raises_before_first_chunk(monkeypatch):
    """An upstream 4xx/5xx raises at chat_stream() call time, not during iteration.

    The serving route wraps only the serve_chat_stream CALL in its try/except; by the time
    the body iterates, the 200 and headers are already flushed. The request and
    raise_for_status therefore must run inside chat_stream itself, and the upstream response
    must be closed on the way out.
    """
    import httpx

    import flash.serve.deployment.deploy as d

    exits = []

    class _ErrorResp:
        def __init__(self):
            self.headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            exits.append(exc)
            return False

        def raise_for_status(self):
            request = httpx.Request("POST", "https://serve.example/v1/chat/completions")
            response = httpx.Response(502, request=request)
            raise httpx.HTTPStatusError("bad gateway", request=request, response=response)

    _erroring_stream_seams(monkeypatch, _ErrorResp())

    with pytest.raises(httpx.HTTPStatusError):
        d.chat_stream("flash-7-abcd", [{"role": "user", "content": "hi"}])
    assert len(exits) == 1


class _MidstreamFailureResp:
    def __init__(self):
        self.exits = []
        self.headers = {"content-type": "text/event-stream"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exits.append(exc)
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}'
        yield ""
        raise RuntimeError("upstream connection lost")


def test_chat_stream_midstream_failure_raises_and_closes_upstream(monkeypatch):
    """A failure after the first chunk propagates out of the iterator and closes upstream.

    The propagating exception is what makes the serving route abort the chunked body, so the
    client sees a truncated transfer rather than a clean eof."""
    import flash.serve.deployment.deploy as d

    resp = _MidstreamFailureResp()
    _erroring_stream_seams(monkeypatch, resp)

    stream = d.chat_stream("flash-7-abcd", [{"role": "user", "content": "hi"}])
    assert next(stream) == "hi"
    with pytest.raises(RuntimeError, match="upstream connection lost"):
        next(stream)
    assert len(resp.exits) == 1


def test_chat_stream_close_without_iterating_closes_upstream(monkeypatch):
    """Closing the returned iterator before reading any chunk still releases the response.

    chat_stream opens the upstream connection eagerly, so the returned generator must already
    be running: close() on a never-started generator skips the finally that exits the httpx
    stream context."""
    import flash.serve.deployment.deploy as d

    resp = _MidstreamFailureResp()
    _erroring_stream_seams(monkeypatch, resp)

    stream = d.chat_stream("flash-7-abcd", [{"role": "user", "content": "hi"}])
    stream.close()
    assert len(resp.exits) == 1

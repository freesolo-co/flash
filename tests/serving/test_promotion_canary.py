"""The canary's request shape, its SSE parsing, and its silence about credentials."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from flash.serve.request.openai import parse_chat_request
from flash.serving.promotion.canary import (
    CANARY_TIMEOUT,
    CANARY_TRANSPORT_FAILURE,
    CanaryError,
    CanaryRequest,
    _payload,
    correlation_id_for,
    run_stream_canary,
)

API_KEY = "sk-promotion-canary-secret-value"


def _request(**overrides) -> CanaryRequest:
    fields = {
        "base_url": "https://serve.freesolo.co",
        "model": "Qwen/Qwen3.5-9B",
        "api_key": API_KEY,
        "correlation_id": correlation_id_for("12345", "1"),
        "timeout_seconds": 5.0,
        "max_tokens": 32,
    }
    fields.update(overrides)
    return CanaryRequest(**fields)


class _FakeResponse:
    def __init__(self, lines: list[str], *, content_type: str, status_code: int = 200) -> None:
        self._lines = lines
        self.headers = {"content-type": content_type}
        self.status_code = status_code

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStream:
    def __init__(self, response: _FakeResponse, recorder: dict) -> None:
        self._response = response
        self._recorder = recorder

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc_info) -> bool:
        return False


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: dict = {}

    def stream(self, method, url, *, headers, json, timeout):
        self.calls = {
            "method": method,
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        }
        return _FakeStream(self._response, self.calls)


class _ExplodingClient:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def stream(self, *args, **kwargs):
        raise self._exc


def _sse(*chunks: str) -> list[str]:
    return [f"data: {chunk}" for chunk in chunks]


def _delta(content: str) -> str:
    return f'{{"choices": [{{"delta": {{"content": "{content}"}}}}]}}'


_TERMINAL = '{"choices": [{"delta": {}, "finish_reason": "stop"}]}'
_USAGE = '{"choices": [], "usage": {"completion_tokens": 7}}'


def _run(client, request=None):
    return asyncio.run(run_stream_canary(request or _request(), client=client))


def test_the_canary_payload_survives_the_real_router_parser():
    """The gate's own request must not be the thing that fails the gate.

    Asserting the payload's keys against a hand-written list is what let `max_completion_tokens`
    ship: it is the correct OpenAI spelling, it looks right, and no test in this file could tell
    that the hosted router rejects it. `parse_chat_request` uses a STRICT allowlist and raises on
    any unknown top-level key, so an unrecognized field 422s before a single token is generated.

    The gate reads that 422 as a non-SSE response and fails -- and the rollback step keys off
    `if: failure()`, so the predecessor gets redeployed over a perfectly healthy release, on every
    deploy, deterministically. Running the payload through the real parser is the only assertion
    that cannot drift from the router's actual contract.
    """
    normalized = parse_chat_request(
        _payload(_request()), require_model=True, allow_managed_selectors=False
    )
    assert normalized.max_tokens == 32
    assert normalized.stream is True


def test_the_canary_key_cannot_reach_a_build_log_through_a_repr():
    """`CanaryRequest` holds `FREESOLO_INTERNAL_KEY`, and build logs are public and permanent.

    A default dataclass repr renders every field verbatim, so any future f-string, debug print, or
    exception with the request in scope would publish the key. `CanaryError`'s docstring already
    warns against exactly this; the field is set `repr=False` so the warning cannot be violated by
    accident.
    """
    assert API_KEY not in repr(_request())
    assert API_KEY not in str(_request())


def test_the_canary_identifies_its_own_generation_for_the_accounting_readback():
    """The correlation id is the only link between this request and its durable usage row.

    Without it the gate could only see that SOME usage settled, not that the canary's own
    generation did.
    """
    client = _FakeClient(
        _FakeResponse(_sse(_delta("ok"), "[DONE]"), content_type="text/event-stream")
    )
    _run(client)
    assert client.calls["headers"]["X-Correlation-ID"] == "fspromo-12345-1"
    assert client.calls["url"] == "https://serve.freesolo.co/v1/chat/completions"
    assert client.calls["json"]["stream"] is True
    # without include_usage the terminal chunk carries no token counts at all.
    assert client.calls["json"]["stream_options"] == {"include_usage": True}


def test_the_canary_authenticates_as_trusted_infra_against_the_real_authorizer():
    """The canary's own headers must satisfy the server's actual internal-key check.

    Asserting a header NAME here would prove nothing -- it would pass just as happily on the wrong
    one. So this feeds the canary's real headers to `is_trusted_internal`, the same function
    `authorize_inference` calls, and requires it to accept them.

    The canary previously sent `Authorization: Bearer <FREESOLO_INTERNAL_KEY>`. That is not an
    internal credential: `authorize_inference` recognizes the internal key only via
    `X-Freesolo-Internal-Key`, and a bearer token instead falls through to `chat_authorizer`, which
    resolves CUSTOMER api keys. The internal key is not one, so every promotion 401'd at the stream
    stage -- and the rollback step keys off `failure()`, so a healthy release would be redeployed
    back to its predecessor on every single deploy.
    """
    from flash.serving.promotion.canary import _headers
    from flash.serving.src.http.headers import is_trusted_internal

    request = SimpleNamespace(headers=_headers(_request()))

    assert is_trusted_internal(request, (API_KEY,)) is True
    # and a DIFFERENT key must still be rejected, so the assertion above is not vacuous.
    assert is_trusted_internal(request, ("some-other-key",)) is False


def test_a_transport_failure_never_puts_the_key_in_the_error():
    """CanaryError messages are printed into a public build log."""
    client = _ExplodingClient(RuntimeError(f"connection refused for key {API_KEY}"))
    with pytest.raises(CanaryError) as excinfo:
        _run(client)
    assert str(excinfo.value) == CANARY_TRANSPORT_FAILURE
    assert API_KEY not in str(excinfo.value)


def test_a_timeout_is_reported_as_a_timeout_not_a_hang():
    """A stream that opens and then goes silent must not hold the deploy job open."""

    class _SilentResponse(_FakeResponse):
        async def aiter_lines(self):
            await asyncio.sleep(10)
            yield "data: [DONE]"

    client = _FakeClient(_SilentResponse([], content_type="text/event-stream"))
    with pytest.raises(CanaryError) as excinfo:
        _run(client, _request(timeout_seconds=0.05))
    assert str(excinfo.value) == CANARY_TIMEOUT


def test_a_non_sse_response_is_reported_without_echoing_its_body():
    client = _FakeClient(
        _FakeResponse(['data: {"error": "boom"}'], content_type="application/json", status_code=500)
    )
    evidence = _run(client)
    assert evidence.content_type_ok is False
    assert evidence.content_delta_count == 0


def test_an_empty_string_delta_is_not_generated_content():
    """A stream that opens the assistant turn and produces nothing emits exactly this."""
    client = _FakeClient(
        _FakeResponse(
            _sse(_delta(""), _TERMINAL, _USAGE, "[DONE]"), content_type="text/event-stream"
        )
    )
    evidence = _run(client)
    assert evidence.content_delta_count == 0
    assert evidence.saw_done_sentinel is True


def test_one_unparseable_frame_does_not_abort_the_stream():
    """A malformed keepalive must not discard the evidence in the frames around it."""
    client = _FakeClient(
        _FakeResponse(
            _sse(_delta("re"), "{not json", _delta("ady"), _TERMINAL, _USAGE, "[DONE]"),
            content_type="text/event-stream",
        )
    )
    evidence = _run(client)
    assert evidence.content_delta_count == 2
    assert evidence.finish_reason == "stop"
    assert evidence.completion_tokens == 7


def test_a_complete_stream_reports_every_field_the_verdict_needs():
    client = _FakeClient(
        _FakeResponse(
            _sse(_delta("ready"), _TERMINAL, _USAGE, "[DONE]"),
            content_type="text/event-stream; charset=utf-8",
        )
    )
    evidence = _run(client)
    assert evidence.content_type_ok is True
    assert evidence.content_delta_count == 1
    assert evidence.finish_reason == "stop"
    assert evidence.completion_tokens == 7
    assert evidence.saw_done_sentinel is True

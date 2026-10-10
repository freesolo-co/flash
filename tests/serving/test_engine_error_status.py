"""Engine dispatch failures must reach the caller as a meaningful HTTP status.

Engine work crosses a Modal RPC boundary. vLLM raises ``VLLMValidationError`` (a ``ValueError``)
for an oversized prompt, but Modal's own infrastructure failures -- a per-input timeout, an
undeserializable remote exception -- subclass ``modal.exception.Error`` instead. The router used
to catch only ``ValueError``, so those escaped as unhandled 500s, and a failure that arrived after
the stream's 200 had been sent truncated the SSE body with no error and no ``[DONE]``.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from modal.exception import ExecutionError, FunctionTimeoutError
from test_router import QWEN, _allow, _router_for

from flash.client.http import ClientError
from flash.serve.request.streaming import _openai_stream_content
from flash.serving.src.http.router import build_offline_serving_app as build_serving_app


class VLLMValidationError(ValueError):
    """Mirrors vllm 0.23.0 ``vllm/exceptions.py``: a ValueError carrying parameter/value.

    Defined here rather than imported because vllm is a GPU-side dependency the router tests
    deliberately run without.
    """

    def __init__(self, message: str, *, parameter: str | None = None, value: object = None) -> None:
        super().__init__(message)
        self.parameter = parameter
        self.value = value

    def __str__(self) -> str:
        base = super().__str__()
        extras = []
        if self.parameter is not None:
            extras.append(f"parameter={self.parameter}")
        if self.value is not None:
            extras.append(f"value={self.value}")
        return f"{base} ({', '.join(extras)})" if extras else base


# the verbatim message observed in production on 2026-07-29 (32806 tokens against a 32768 window)
CONTEXT_OVERFLOW = VLLMValidationError(
    "This model's maximum context length is 32768 tokens. However, you requested 0 output "
    "tokens and your prompt contains 32806 input tokens, for a total of 32806 tokens. Please "
    "reduce the length of the input prompt or the number of requested output tokens.",
    parameter="input_tokens",
    value=32806,
)


class _FailingPool:
    """Engine pool whose dispatch raises ``exc`` -- before the stream starts, or partway through."""

    def __init__(self, exc: Exception, *, mid_stream: bool = False) -> None:
        self._exc = exc
        self._mid_stream = mid_stream

    async def generate(self, base_model, payload, record, *, expected_checkpoint=None):
        raise self._exc

    async def stream_generate(self, base_model, payload, record, *, expected_checkpoint=None):
        if self._mid_stream:
            yield {
                "type": "ready",
                "checkpoint": "run-a/final",
                "thinking": False,
                "lora_request_adapter": record.adapter_id,
            }
            yield {"type": "delta", "text": "Hello"}
        raise self._exc

    async def register(self, base_model, record):
        return None

    async def unregister(self, base_model, adapter_id, expected_generation=None):
        return None


class _CleanlyTruncatedPool(_FailingPool):
    async def stream_generate(self, base_model, payload, record, *, expected_checkpoint=None):
        yield {
            "type": "ready",
            "checkpoint": "run-a/final",
            "thinking": False,
            "lora_request_adapter": record.adapter_id,
        }
        yield {"type": "delta", "text": "partial"}


class _SyncRaisingStreamPool(_FailingPool):
    """A conforming pool whose ``stream_generate`` raises while BUILDING the iterator.

    ``EnginePool`` declares ``stream_generate`` as an ordinary method returning an AsyncIterator.
    The Modal pool happens to be an async generator, deferring its body to the first ``anext``, but
    the protocol does not require that -- so dispatch failure must map the same way for both.
    """

    def stream_generate(self, base_model, payload, record, *, expected_checkpoint=None):
        raise self._exc


def _client(exc: Exception, *, mid_stream: bool = False, pool_cls=_FailingPool) -> TestClient:
    app = build_serving_app(
        pool_cls(exc, mid_stream=mid_stream),
        _router_for("run-a", QWEN),
        chat_authorizer=_allow,
    )
    return TestClient(app, raise_server_exceptions=False, headers={"Authorization": "Bearer t"})


def _chat(client: TestClient, *, stream: bool = False):
    body = {"model": "run-a/final", "messages": [{"role": "user", "content": "hi"}]}
    if stream:
        body["stream"] = True
    return client.post("/v1/chat/completions", json=body)


@pytest.mark.parametrize("stream", [False, True])
def test_context_overflow_is_400_with_the_engine_message(stream: bool) -> None:
    """An oversized prompt is a caller error: 400, carrying vLLM's actionable message."""
    resp = _chat(_client(CONTEXT_OVERFLOW), stream=stream)
    assert resp.status_code == 400
    assert "maximum context length is 32768 tokens" in resp.text
    assert "32806 input tokens" in resp.text


@pytest.mark.parametrize("stream", [False, True])
def test_engine_timeout_is_504_not_500(stream: bool) -> None:
    """A per-input timeout is a gateway timeout, not an unhandled server crash."""
    exc = FunctionTimeoutError("Task's current input hit its timeout of 600s")
    resp = _chat(_client(exc), stream=stream)
    assert resp.status_code == 504
    assert "timeout" in resp.text.lower()


@pytest.mark.parametrize("stream", [False, True])
def test_undeserializable_remote_error_is_502_not_500(stream: bool) -> None:
    """A remote exception the router cannot reconstruct is an upstream failure: 502."""
    exc = ExecutionError("Could not deserialize remote exception due to local error")
    resp = _chat(_client(exc), stream=stream)
    assert resp.status_code == 502


def test_stream_dispatch_failure_is_mapped_even_when_raised_synchronously() -> None:
    """A pool that raises while constructing the iterator maps like one that raises on advance."""
    exc = FunctionTimeoutError("Task's current input hit its timeout of 600s")
    resp = _chat(_client(exc, pool_cls=_SyncRaisingStreamPool), stream=True)
    assert resp.status_code == 504


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("router-side bug"),
        TypeError("router-side bug"),
        HTTPException(403, "forbidden"),
    ],
    ids=["runtime", "type", "http"],
)
def test_unrelated_failures_are_not_rewritten_as_engine_errors(
    exc: Exception, stream: bool
) -> None:
    """Only identifiable engine failures are translated.

    A blanket ``Exception -> 502`` would hide router-side defects behind an upstream-failure status
    and would strip a deliberate ``HTTPException`` status (such as the authorizer's 403) down to a
    generic 502. Neither is an engine failure, so neither may be mapped.
    """
    resp = _chat(_client(exc), stream=stream)
    expected = exc.status_code if isinstance(exc, HTTPException) else 500
    assert resp.status_code == expected
    assert resp.status_code != 502


def test_midstream_router_side_bug_is_not_reported_as_an_engine_failure() -> None:
    """A defect after the stream opens must not be dressed up as an upstream engine error.

    The guard covers advancing the engine iterator only. If it covered event rendering too, a
    router bug would reach the caller as a well-formed ``engine_error`` event, blaming the engine
    and hiding the traceback.

    An unidentified failure keeps the pre-existing behaviour instead: it propagates, so the
    traceback is logged and the connection drops. That is a worse caller experience than the clean
    termination an engine failure gets, and deliberately so -- a truncated stream is diagnosable,
    whereas a fabricated ``engine_error`` points every investigation at the wrong subsystem.
    """
    resp = _chat(_client(RuntimeError("router-side bug"), mid_stream=True), stream=True)
    assert "engine_error" not in resp.text
    assert "[DONE]" not in resp.text  # no fabricated clean termination for a router-side defect


def test_clean_iterator_end_without_final_is_a_consumer_visible_error() -> None:
    response = _chat(
        _client(RuntimeError("unused"), pool_cls=_CleanlyTruncatedPool),
        stream=True,
    )
    stream = _openai_stream_content(iter(response.text.splitlines()), thinking=False)

    assert response.status_code == 200
    assert next(stream) == "partial"
    with pytest.raises(ClientError, match="ended without a terminal event"):
        next(stream)


def test_midstream_failure_terminates_the_stream_with_an_error_event() -> None:
    """A failure after the 200 cannot change the status, so it must be visible in the body.

    Without this the caller gets a well-formed but silently truncated stream -- no error, no
    ``[DONE]`` -- which is indistinguishable from a short completion.
    """
    exc = FunctionTimeoutError("Task's current input hit its timeout of 600s")
    resp = _chat(_client(exc, mid_stream=True), stream=True)

    assert resp.status_code == 200  # headers already sent; the status cannot carry the failure
    assert "Hello" in resp.text  # deltas emitted before the failure are preserved
    assert '"finish_reason":"error"' in resp.text.replace(" ", "")
    assert "engine_error" in resp.text
    # the status the request would have gotten travels in the body, since the header cannot carry it
    assert '"code":504' in resp.text.replace(" ", "")
    assert resp.text.rstrip().endswith("data: [DONE]")  # protocol terminated, not truncated

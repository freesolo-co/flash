"""The one real request a promotion has to survive.

Sends an authenticated streaming chat completion to the freshly deployed router and reduces the SSE
body to `StreamEvidence`. It supplies its own `X-Correlation-ID` so the durable usage row this
generation produces is identifiable afterwards; the router stamps that id onto the usage event
alongside its own release identity (`flash/serving/src/accounting/usage.py`).

Engine-side proof cannot come from the response: `flash/serving/src/io/responses.py` deliberately
strips `engine_replica_id` from public bodies, and the engines only receive `HF_TOKEN`, so they
cannot report the release sha either. The response proves generation happened; accounting proves
which release it happened on.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from flash.serving.promotion.evidence import StreamEvidence

if TYPE_CHECKING:
    import httpx

_DATA_PREFIX = "data: "
_DONE = "[DONE]"

CANARY_TIMEOUT = "canary_timeout"
CANARY_TRANSPORT_FAILURE = "canary_transport_failure"


class CanaryError(Exception):
    """Raised with a stable reason code, never with a credential or a response body.

    The message reaches a public build log. Formatting the offending request or response into it is
    how a key ends up permanently readable in CI output.
    """


@dataclass(frozen=True)
class CanaryRequest:
    base_url: str
    model: str
    # `repr=False` because this dataclass holds `FREESOLO_INTERNAL_KEY`. The default repr would
    # render it verbatim into any f-string, debug print, or exception that has the request in
    # scope -- and those land in a PUBLIC github actions log, permanently readable. This is the
    # exact hazard `CanaryError`'s docstring names; the field must not be able to cause it.
    api_key: str = field(repr=False)
    correlation_id: str
    timeout_seconds: float
    # `max_tokens`, not `max_completion_tokens`. The hosted router parses with
    # `flash.serve.request.openai.parse_chat_request`, whose `_ALLOWED_REQUEST_KEYS` is a strict
    # allowlist that raises on any unknown top-level key -- and the newer OpenAI spelling is not in
    # it. Sending it 422s before generation, which reads here as a non-SSE response and fails the
    # gate, so `if: failure()` would redeploy the PREDECESSOR over a healthy release on every
    # deploy. The name is pinned to the wire key so the two cannot drift apart silently.
    max_tokens: int


def correlation_id_for(run_id: str, run_attempt: str) -> str:
    return f"fspromo-{run_id}-{run_attempt}"


def _payload(request: CanaryRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "messages": [{"role": "user", "content": "Reply with the single word: ready."}],
        "stream": True,
        # without this the terminal chunk carries no usage, and "did this generate any tokens?"
        # becomes unanswerable from the stream alone.
        "stream_options": {"include_usage": True},
        "max_tokens": request.max_tokens,
        "temperature": 0,
    }


def _headers(request: CanaryRequest) -> dict[str, str]:
    """The canary authenticates as trusted infra, which is a HEADER, not a bearer token.

    `ServingContext.authorize_inference` recognizes `FREESOLO_INTERNAL_KEY` only through
    `X-Freesolo-Internal-Key`. The same value sent as `Authorization: Bearer` is not an internal
    credential at all: it falls through to `chat_authorizer`, which resolves CUSTOMER api keys, and
    the internal key is not one -- so every promotion would 401 at the stream stage and roll a
    healthy release back to its predecessor.
    """
    return {
        "X-Freesolo-Internal-Key": request.api_key,
        "X-Correlation-ID": request.correlation_id,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }


class _StreamReader:
    """Accumulates evidence across SSE frames.

    Only a non-empty string delta counts as content. An empty-string delta is what a stream emits
    when it opens the assistant turn and then produces nothing, so counting it would let a
    zero-token generation pass as a successful one.
    """

    def __init__(self) -> None:
        self.content_delta_count = 0
        self.finish_reason: str | None = None
        self.completion_tokens: int | None = None
        self.saw_done_sentinel = False

    def feed(self, line: str) -> bool:
        """Consume one SSE line. Returns False once the stream is finished."""
        stripped = line.strip()
        if not stripped.startswith(_DATA_PREFIX):
            return True
        body = stripped[len(_DATA_PREFIX) :].strip()
        if body == _DONE:
            self.saw_done_sentinel = True
            return False
        try:
            chunk = json.loads(body)
        except ValueError:
            # a single unparseable frame is not fatal, but it also never counts as evidence.
            return True
        if isinstance(chunk, dict):
            self._read_chunk(chunk)
        return True

    def _read_chunk(self, chunk: dict[str, Any]) -> None:
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                delta = first.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        self.content_delta_count += 1
                reason = first.get("finish_reason")
                if isinstance(reason, str) and reason:
                    self.finish_reason = reason
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            tokens = usage.get("completion_tokens")
            if not isinstance(tokens, bool) and isinstance(tokens, int):
                self.completion_tokens = tokens

    def evidence(self, *, content_type_ok: bool) -> StreamEvidence:
        return StreamEvidence(
            content_type_ok=content_type_ok,
            content_delta_count=self.content_delta_count,
            finish_reason=self.finish_reason,
            completion_tokens=self.completion_tokens,
            saw_done_sentinel=self.saw_done_sentinel,
        )


async def run_stream_canary(request: CanaryRequest, *, client: httpx.AsyncClient) -> StreamEvidence:
    """Stream one completion and report what it proved.

    The whole read loop is bounded, not just the connect: a stream that opens and then never sends
    another byte would otherwise hold the deploy job until the runner's own timeout, turning a
    failed promotion into a hung one.
    """
    try:
        async with asyncio.timeout(request.timeout_seconds):
            return await _read_stream(request, client=client)
    except TimeoutError as exc:
        raise CanaryError(CANARY_TIMEOUT) from exc
    except CanaryError:
        raise
    # every transport failure collapses to one opaque reason code.
    except Exception as exc:
        raise CanaryError(CANARY_TRANSPORT_FAILURE) from exc


async def _read_stream(request: CanaryRequest, *, client: httpx.AsyncClient) -> StreamEvidence:
    url = f"{request.base_url.rstrip('/')}/v1/chat/completions"
    reader = _StreamReader()
    async with client.stream(
        "POST",
        url,
        headers=_headers(request),
        json=_payload(request),
        timeout=request.timeout_seconds,
    ) as response:
        content_type = str(response.headers.get("content-type", ""))
        content_type_ok = content_type.startswith("text/event-stream")
        if response.status_code != 200 or not content_type_ok:
            # a non-200 or non-SSE response has no frames worth parsing; report what it was, not
            # what it said, so an error body cannot reach the log.
            return reader.evidence(content_type_ok=False)
        async for line in response.aiter_lines():
            if not reader.feed(line):
                break
    return reader.evidence(content_type_ok=content_type_ok)

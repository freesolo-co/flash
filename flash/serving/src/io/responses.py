"""Response shaping for the serving front door: reasoning-block splitting (streaming and
non-streaming) and OpenAI response_format -> structured-outputs translation.

Split out of router.py, which is a single large FastAPI app builder; these helpers are pure
functions over strings and dicts, with no engine or app state, so they are unit-testable on their
own and keep the router focused on routing.

Do NOT add `from __future__ import annotations`: router.py re-exports these next to FastAPI
handlers whose closure-local body models are used as annotations, and the future import turns
those into unresolvable strings -> silent 422.
"""

from typing import Any

from fastapi.responses import JSONResponse

from flash.serve.app.openai import split_reasoning
from flash.serve.request.openai import NormalizedChatRequest
from flash.serving.src.io.provenance import _checkpoint_provenance, _provenance_headers
from flash.serving.src.io.schemas import AdapterRecord

_THINK_CLOSE = "</think>"


def _inference_json_response(result: dict[str, Any], target: AdapterRecord) -> JSONResponse:
    # attach revision provenance while keeping engine-process attribution internal to metering.
    active_checkpoint = result.get("checkpoint")
    provenance = _checkpoint_provenance(target, active_checkpoint)
    internal_fields = {
        "cached_tokens_reported",
        "completion_token_ids",
        "engine_replica_id",
        "lora_request_adapter",
        "queue_wait_seconds",
        "replica_boot_duration_seconds",
        "replica_freshly_booted",
        "replica_in_flight_requests_at_admission",
        "prompt_token_ids",
    }
    public_result = {key: value for key, value in result.items() if key not in internal_fields}
    body = {**public_result, "freesolo": provenance} if provenance is not None else public_result
    return JSONResponse(body, headers=_provenance_headers(provenance, active_checkpoint))


def _split_reasoning(text: str, thinking: bool) -> tuple[str | None, str]:
    """Split a completion into ``(reasoning_content, content)`` for the OpenAI surface.

    The chat template opens the reasoning block in the PROMPT, so a thinking completion begins
    inside the block and its text is ``<reasoning></think><answer>`` -- a closing tag with no
    opener. Returned verbatim, the caller sees a stray ``</think>`` glued to the answer and an
    empty ``reasoning_content``, and every downstream parser that keys off the opening tag fails.

    ``thinking`` is the mode the generation was RENDERED with, reported by the engine. It is
    required rather than inferred: a non-thinking completion that merely quotes ``</think>`` must
    not be silently torn in half.
    """
    if not thinking:
        return None, text
    head, sep, tail = text.partition(_THINK_CLOSE)
    if not sep:
        # thinking mode, but the block never closed (hit max_tokens mid-reasoning). all of it is
        # reasoning; there is no answer yet. an empty content is the honest report.
        return head, ""
    return head, tail


def _partial_tag_suffix(text: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of the closing tag."""
    for size in range(min(len(text), len(_THINK_CLOSE) - 1), 0, -1):
        if _THINK_CLOSE.startswith(text[-size:]):
            return size
    return 0


class _ReasoningStreamSplitter:
    """Routes streamed deltas to ``reasoning_content`` until the reasoning block closes.

    Streaming cannot reuse :func:`_split_reasoning`: the closing tag arrives token by token and
    can straddle a chunk boundary, so a trailing partial match is held back rather than emitted
    as reasoning text it might turn out not to be.
    """

    def __init__(self, thinking: bool) -> None:
        self._closed = not thinking
        self._pending = ""

    def feed(self, text: str) -> tuple[str, str]:
        """Split one delta into ``(reasoning_delta, content_delta)``."""
        if self._closed:
            return "", text
        buffer = self._pending + text
        head, sep, tail = buffer.partition(_THINK_CLOSE)
        if sep:
            self._closed = True
            self._pending = ""
            return head, tail
        hold = _partial_tag_suffix(buffer)
        split = len(buffer) - hold
        self._pending = buffer[split:]
        return buffer[:split], ""

    def flush(self) -> str:
        """Held-back text at end of stream.

        Reaching here means the block never closed (the generation stopped mid-reasoning), so the
        remainder is reasoning with no answer -- the same call :func:`_split_reasoning` makes.
        """
        pending, self._pending = self._pending, ""
        return pending


def _usage_block(prompt_tokens: int, completion_tokens: int, cached_tokens: Any) -> dict[str, Any]:
    """OpenAI-style ``usage`` object for a completion response.

    Mirrors the OpenAI schema, including ``prompt_tokens_details.cached_tokens`` (the
    prefix-cached subset of the prompt) so a client can see how many prompt tokens were served
    from cache — the same count the backend bills at a discount. ``cached_tokens`` is clamped to
    ``prompt_tokens`` and the details block is omitted entirely when there were no cached tokens.
    """
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    try:
        cached = max(0, min(int(cached_tokens or 0), prompt_tokens))
    except (TypeError, ValueError):
        cached = 0
    if cached > 0:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


def openai_chat_completion(
    *,
    completion_id: str,
    created: int,
    adapter_id: str,
    generation: dict[str, Any],
    provenance: dict[str, str] | None,
) -> dict[str, Any]:
    """Assemble the non-streaming OpenAI chat-completion body from an engine result."""
    raw_choices = generation.get("choices")
    if not isinstance(raw_choices, list):
        raw_choices = [
            {
                "index": 0,
                "text": generation["text"],
                "finish_reason": generation.get("finish_reason"),
                "logprobs": None,
            }
        ]
    choices = []
    for choice in raw_choices:
        reasoning, content = split_reasoning(
            str(choice["text"]), thinking=bool(generation.get("thinking"))
        )
        tool_calls = choice.get("tool_calls") or []
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content if content else None if tool_calls else content,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        choice_logprobs = choice.get("logprobs")
        choices.append(
            {
                "index": choice["index"],
                "message": message,
                "finish_reason": choice.get("finish_reason"),
                "logprobs": {"content": choice_logprobs} if choice_logprobs is not None else None,
            }
        )
    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": adapter_id,
        "choices": choices,
    }
    if provenance is not None:
        response["freesolo"] = provenance
    prompt_tokens = generation.get("prompt_tokens")
    completion_tokens = generation.get("completion_tokens")
    if prompt_tokens is not None and completion_tokens is not None:
        response["usage"] = _usage_block(
            int(prompt_tokens), int(completion_tokens), generation.get("cached_tokens")
        )
    return response


def openai_generate_fields(request: NormalizedChatRequest, adapter_id: str) -> dict[str, Any]:
    """bind one canonical OpenAI request to the hosted internal envelope."""

    return {
        "adapter_id": adapter_id,
        "messages": request.messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "n": request.n,
        "seed": request.seed,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
        "logprobs": request.logprobs,
        "top_logprobs": request.top_logprobs,
        "chat_template_kwargs": request.chat_template_kwargs,
        "stop": list(request.stop) or None,
        "structured_outputs": request.structured_outputs,
        "tools": None if request.tools is None else [tool.wire() for tool in request.tools],
        "tool_choice": request.tool_choice,
        "parallel_tool_calls": request.parallel_tool_calls,
    }

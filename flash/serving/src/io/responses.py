"""Public JSON responses and shared OpenAI completion shaping for hosted serving."""

from typing import Any

from fastapi.responses import JSONResponse

from flash.serve.app.openai import split_reasoning
from flash.serve.request.openai import NormalizedChatRequest
from flash.serving.src.io.provenance import _checkpoint_provenance, _provenance_headers
from flash.serving.src.io.schemas import AdapterRecord


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


def _usage_block(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: Any,
    reasoning_tokens: int | None = None,
) -> dict[str, Any]:
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
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return usage


def openai_chat_completion(
    *,
    completion_id: str,
    created: int,
    model_id: str,
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
        "model": model_id,
        "choices": choices,
    }
    if provenance is not None:
        response["freesolo"] = provenance
    prompt_tokens = generation.get("prompt_tokens")
    completion_tokens = generation.get("completion_tokens")
    if prompt_tokens is not None and completion_tokens is not None:
        response["usage"] = _usage_block(
            int(prompt_tokens),
            int(completion_tokens),
            generation.get("cached_tokens"),
            generation.get("reasoning_tokens"),
        )
    return response


def openai_generate_fields(request: NormalizedChatRequest, model_id: str) -> dict[str, Any]:
    """bind one canonical OpenAI request to the hosted internal envelope."""

    return {
        "model_id": model_id,
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

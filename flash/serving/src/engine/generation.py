"""choice-aware buffered and streaming generation for the hosted engine."""

from __future__ import annotations

import inspect
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, TypedDict

from flash.serve.runtime.sampling import (
    complete_indexed_outputs,
    indexed_outputs,
    normalize_token_logprobs,
)
from flash.serving.src.engine.support import (
    _cached_tokens_reported,
    _num_cached_tokens,
    _num_prompt_tokens,
)
from flash.serving.src.io.openai_request import OpenAIGenerateRequest
from flash.serving.src.io.schemas import GenerateRequest


class _ChoiceState(TypedDict):
    text: str
    token_ids: list[int]
    finish_reason: str | None


_OPENAI_FIELDS = frozenset(
    {
        "n",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "top_logprobs",
    }
)


def _payload(payload_dict: dict[str, Any]) -> tuple[OpenAIGenerateRequest, bool]:
    is_openai = bool(_OPENAI_FIELDS & payload_dict.keys())
    if is_openai:
        return OpenAIGenerateRequest.model_validate(payload_dict), True
    raw = GenerateRequest.model_validate(payload_dict)
    return OpenAIGenerateRequest.model_validate(raw.model_dump()), False


def _sampling_params(payload: OpenAIGenerateRequest, structured: Any, output_kind: Any) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        top_p=payload.top_p,
        n=payload.n,
        seed=payload.seed,
        frequency_penalty=payload.frequency_penalty,
        presence_penalty=payload.presence_penalty,
        logprobs=payload.top_logprobs if payload.logprobs else None,
        output_kind=output_kind,
        structured_outputs=structured,
        stop=payload.stop,
    )


def _choice(index: int, output: Any, *, top_logprobs: int) -> dict[str, Any]:
    token_ids = [int(value) for value in (getattr(output, "token_ids", None) or [])]
    finish_reason = getattr(output, "finish_reason", None)
    if not isinstance(finish_reason, str) or not finish_reason:
        raise RuntimeError("vLLM generation ended without a finish reason")
    return {
        "index": index,
        "text": str(getattr(output, "text", "") or ""),
        "finish_reason": finish_reason,
        "token_ids": token_ids,
        "logprobs": normalize_token_logprobs(
            token_ids,
            getattr(output, "logprobs", None),
            top_logprobs=top_logprobs,
        ),
    }


async def _close_output_stream(output_stream: Any) -> None:
    if output_stream is None:
        return
    close = getattr(output_stream, "aclose", None)
    if close is None:
        return
    active_exception = sys.exc_info()[0] is not None
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception:
        if not active_exception:
            raise


async def _confirm_lora_consumed(owner: Any, record: Any, lora_request: Any) -> None:
    confirm = getattr(owner, "_mark_lora_consumed", None)
    if confirm is not None:
        await confirm(record, lora_request)


@asynccontextmanager
async def _in_flight_lease(owner: Any, record: Any, lora_request: Any) -> AsyncIterator[None]:
    factory = getattr(owner, "_lora_request_in_flight", None)
    if factory is None:
        yield
        return
    async with factory(record, lora_request):
        yield


@asynccontextmanager
async def _source_lease(owner: Any, record: Any, lora_request: Any) -> AsyncIterator[None]:
    factory = getattr(owner, "_source_generation_lease", None)
    if factory is None:
        yield
        return
    async with factory(record, lora_request):
        yield


def _usage_fields(
    request_output: Any,
    completion_token_ids: list[int],
    *,
    start: float,
    request_id: str,
    engine_replica_id: str,
    checkpoint: str,
    thinking: bool,
) -> dict[str, Any]:
    prompt_token_ids = [
        int(value) for value in (getattr(request_output, "prompt_token_ids", []) or [])
    ]
    fields = {
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": list(completion_token_ids),
        "prompt_tokens": _num_prompt_tokens(request_output),
        "completion_tokens": len(completion_token_ids),
        "cached_tokens": _num_cached_tokens(request_output),
        "cached_tokens_reported": _cached_tokens_reported(request_output),
        "inference_time_seconds": time.time() - start,
        "request_id": request_id,
        "engine_replica_id": engine_replica_id,
        "checkpoint": checkpoint,
        "thinking": thinking,
    }
    if not thinking:
        fields["reasoning_tokens"] = 0
    return fields


async def generate(
    owner: Any,
    payload_dict: dict[str, Any],
    record_dict: dict[str, Any] | None = None,
    expected_checkpoint: str | None = None,
    generation_id: str | None = None,
) -> dict[str, Any]:
    from vllm.sampling_params import RequestOutputKind

    payload, is_openai = _payload(payload_dict)
    lora_request, record = await owner._lora_request(payload.adapter_id, record_dict)
    attestation = owner._lora_request_attestation(record, lora_request)
    active_checkpoint = owner._enforce_expected_checkpoint(record, expected_checkpoint)
    thinking = owner._thinking_default(record, payload)
    if thinking and payload.logprobs:
        raise ValueError("logprobs are not supported for thinking-enabled generation")
    structured, reasoning_ended, parser_kwargs = owner._structured_outputs_state(
        payload, record, thinking
    )
    sampling = _sampling_params(payload, structured, RequestOutputKind.FINAL_ONLY)
    request_id = generation_id or payload.generation_id or f"fsgen-{uuid.uuid4().hex}"
    start = time.time()
    final_output = None
    output_stream = None
    in_flight_lease = None
    prompt_input = await owner._prepare_prompt_input(payload, thinking)
    # the source lease pins this adapter's materialized directory for the whole generation, so a
    # concurrent cache reclamation cannot delete weights out from under a running request.
    source_lease = _source_lease(owner, record, lora_request)
    await source_lease.__aenter__()
    try:
        try:
            in_flight_lease = _in_flight_lease(owner, record, lora_request)
            await in_flight_lease.__aenter__()
            output_stream = owner.engine.generate(
                prompt_input,
                sampling,
                request_id,
                lora_request=lora_request,
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs=parser_kwargs,
            )
            async for output in output_stream:
                await _confirm_lora_consumed(owner, record, lora_request)
                final_output = output
        except Exception:
            owner._self_heal_if_dead("generate")
            raise
    finally:
        try:
            await _close_output_stream(output_stream)
        finally:
            try:
                owner._close_prompt_images(prompt_input)
            finally:
                try:
                    await source_lease.__aexit__(None, None, None)
                finally:
                    if in_flight_lease is not None:
                        await in_flight_lease.__aexit__(None, None, None)
    if final_output is None:
        raise RuntimeError("vLLM returned no output")
    choices = [
        _choice(index, output, top_logprobs=payload.top_logprobs)
        for index, output in sorted(complete_indexed_outputs(final_output, n=payload.n).items())
    ]
    completion_ids = [token for choice in choices for token in choice["token_ids"]]
    first = choices[0]
    return {
        "ok": True,
        "adapter_id": payload.adapter_id,
        **({"lora_request_adapter": attestation} if attestation is not None else {}),
        "text": first["text"],
        "finish_reason": first["finish_reason"],
        "token_ids": first["token_ids"],
        **({"choices": choices} if is_openai else {}),
        **_usage_fields(
            final_output,
            completion_ids,
            start=start,
            request_id=request_id,
            engine_replica_id=owner._replica_identifier(),
            checkpoint=active_checkpoint,
            thinking=thinking,
        ),
    }


async def stream_generate(
    owner: Any,
    payload_dict: dict[str, Any],
    record_dict: dict[str, Any] | None = None,
    expected_checkpoint: str | None = None,
    generation_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    from vllm.sampling_params import RequestOutputKind

    payload, _ = _payload(payload_dict)
    lora_request, record = await owner._lora_request(payload.adapter_id, record_dict)
    attestation = owner._lora_request_attestation(record, lora_request)
    active_checkpoint = owner._enforce_expected_checkpoint(record, expected_checkpoint)
    thinking = owner._thinking_default(record, payload)
    if thinking and payload.logprobs:
        raise ValueError("logprobs are not supported for thinking-enabled generation")
    structured, reasoning_ended, parser_kwargs = owner._structured_outputs_state(
        payload, record, thinking
    )
    sampling = _sampling_params(payload, structured, RequestOutputKind.DELTA)
    request_id = generation_id or payload.generation_id or f"fsgen-{uuid.uuid4().hex}"
    start = time.time()
    prompt_input = await owner._prepare_prompt_input(payload, thinking)
    source_lease = _source_lease(owner, record, lora_request)
    await source_lease.__aenter__()
    output_stream = None
    in_flight_lease = None
    completion_ids: list[int] = []
    choice_state: dict[int, _ChoiceState] = {
        index: {"text": "", "token_ids": [], "finish_reason": None} for index in range(payload.n)
    }
    usage_kwargs = {
        "start": start,
        "request_id": request_id,
        "engine_replica_id": owner._replica_identifier(),
        "checkpoint": active_checkpoint,
        "thinking": thinking,
    }
    try:
        try:
            in_flight_lease = _in_flight_lease(owner, record, lora_request)
            await in_flight_lease.__aenter__()
            output_stream = owner.engine.generate(
                prompt_input,
                sampling,
                request_id,
                lora_request=lora_request,
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs=parser_kwargs,
            )
            first_output = await anext(output_stream)
            await _confirm_lora_consumed(owner, record, lora_request)
        except StopAsyncIteration as exc:
            raise RuntimeError("vLLM returned no output") from exc
        except Exception:
            owner._self_heal_if_dead("stream_generate")
            raise
        first_ids = [
            int(token)
            for output in indexed_outputs(first_output, n=payload.n).values()
            for token in (getattr(output, "token_ids", None) or [])
        ]
        yield {
            "type": "ready",
            "thinking": thinking,
            **({"lora_request_adapter": attestation} if attestation is not None else {}),
            **_usage_fields(first_output, first_ids, **usage_kwargs),
        }
        current = first_output
        last_output = first_output
        try:
            while True:
                last_output = current
                for index, output in sorted(indexed_outputs(current, n=payload.n).items()):
                    state = choice_state[index]
                    if state["finish_reason"] is not None:
                        raise RuntimeError("vLLM emitted data after a choice terminal")
                    token_ids = [int(value) for value in (getattr(output, "token_ids", None) or [])]
                    logprobs = normalize_token_logprobs(
                        token_ids,
                        getattr(output, "logprobs", None),
                        top_logprobs=payload.top_logprobs,
                    )
                    text = str(getattr(output, "text", "") or "")
                    state["text"] += text
                    state["token_ids"].extend(token_ids)
                    completion_ids.extend(token_ids)
                    if text or logprobs is not None:
                        yield {
                            "type": "delta",
                            "index": index,
                            "text": text,
                            "logprobs": logprobs,
                            **_usage_fields(current, completion_ids, **usage_kwargs),
                        }
                    finish_reason = getattr(output, "finish_reason", None)
                    if finish_reason is not None:
                        if not isinstance(finish_reason, str) or not finish_reason:
                            raise RuntimeError("vLLM returned an invalid finish reason")
                        state["finish_reason"] = finish_reason
                        yield {
                            "type": "choice_finished",
                            "index": index,
                            "finish_reason": finish_reason,
                            **_usage_fields(current, completion_ids, **usage_kwargs),
                        }
                try:
                    current = await anext(output_stream)
                except StopAsyncIteration:
                    break
        except Exception:
            owner._self_heal_if_dead("stream_generate")
            raise
        if any(state["finish_reason"] is None for state in choice_state.values()):
            raise RuntimeError("vLLM ended with unterminated output choices")
        yield {
            "type": "final",
            "ok": True,
            "adapter_id": payload.adapter_id,
            "choices": [{"index": index, **state} for index, state in sorted(choice_state.items())],
            **_usage_fields(last_output, completion_ids, **usage_kwargs),
        }
    finally:
        try:
            await _close_output_stream(output_stream)
        finally:
            try:
                owner._close_prompt_images(prompt_input)
            finally:
                try:
                    await source_lease.__aexit__(None, None, None)
                finally:
                    if in_flight_lease is not None:
                        await in_flight_lease.__aexit__(None, None, None)

"""managed chat resolution and forwarding for immutable deployed runs."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator
from typing import Any

import anyio
import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import flash.serve.deployment.deploy as serving_deploy
from flash.runner.lifecycle.status import effective_spec_from_status
from flash.serve.contract.provenance import (
    CheckpointProvenance,
    validate_body_provenance,
    validate_header_provenance,
)
from flash.serve.request.openai import (
    OpenAIRequestError,
    merge_stop_sequences,
    parse_chat_request,
    reject_thinking_logprobs,
    reject_tool_capability,
)
from flash.serve.request.tool_calls import qualified_tool_parser, validate_tool_stop_sequences
from flash.serve.request.transport import RawChatStream, is_event_stream_content_type
from flash.server.platform.deps import require_manageable_run
from flash.server.platform.internal_client import run_serving_org_id
from flash.server.routes.serving_errors import upstream_chat_error
from flash.server.routes.serving_revisions import (
    _DEPLOYMENT_BUSY_STATES,
    _authorized_chat_checkpoint,
    _managed_chat_messages,
)

logger = logging.getLogger(__name__)
# Forwarding budget for managed chat. A saturated budget sheds load with 429/503 + Retry-After
# instead of queueing thousands of sockets behind minute-long generations.
_CHAT_CONCURRENCY_ENV = "FLASH_CHAT_FORWARDING_CONCURRENCY"
_CHAT_QUEUE_LIMIT_ENV = "FLASH_CHAT_QUEUE_LIMIT"
_CHAT_QUEUE_WAIT_ENV = "FLASH_CHAT_QUEUE_WAIT_SECONDS"
_CHAT_UPSTREAM_TIMEOUT_ENV = "FLASH_CHAT_UPSTREAM_TIMEOUT_SECONDS"
# Final evaluations (a verifier scoring a checkpoint) must not be shed behind research traffic on
# the same control plane: one 429 on a 180-case evaluation used to fail the whole attempt. A request
# that names this traffic class takes its own, smaller bounded lane with the same 429/503 shedding,
# so it never competes with research chats and can never take more than its own slots.
_TRAFFIC_CLASS_HEADER = "x-flash-traffic-class"
_EVALUATION_TRAFFIC_CLASS = "evaluation"
_CHAT_EVALUATION_CONCURRENCY_ENV = "FLASH_CHAT_EVALUATION_CONCURRENCY"
_CHAT_RETRY_AFTER_SECONDS = "5"
_STREAM_LIMITER: anyio.CapacityLimiter | None = None
_UPSTREAM_RESPONSE_HEADER_EXCLUSIONS = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-freesolo-adapter-revision",
        "x-freesolo-checkpoint",
        "x-freesolo-hf-revision",
        "x-freesolo-lora-request-adapter",
    }
)


def _env_number(name: str, default: float, *, minimum: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    if not math.isfinite(value) or value < minimum:
        logger.warning("chat_admission_config_invalid name=%s value=%r", name, raw)
        return default
    return value


def chat_forwarding_concurrency() -> int:
    return int(_env_number(_CHAT_CONCURRENCY_ENV, 128, minimum=1))


def chat_upstream_timeout_s() -> float:
    """bound one forwarded generation; keep it below callers' own request timeouts."""

    return _env_number(_CHAT_UPSTREAM_TIMEOUT_ENV, 600.0, minimum=1.0)


def chat_evaluation_concurrency() -> int:
    return int(_env_number(_CHAT_EVALUATION_CONCURRENCY_ENV, 32, minimum=1))


def _state_limiter(state: Any, name: str, capacity: int | None = None) -> anyio.CapacityLimiter:
    limiter = getattr(state, name, None)
    if limiter is None:
        limiter = anyio.CapacityLimiter(capacity or chat_forwarding_concurrency())
        setattr(state, name, limiter)
    return limiter


def _chat_lane(request: Any) -> tuple[str, str, int | None]:
    """(forwarding limiter, thread limiter, capacity) for this request's traffic class."""

    traffic_class = (request.headers.get(_TRAFFIC_CLASS_HEADER) or "").strip().lower()
    if traffic_class == _EVALUATION_TRAFFIC_CLASS:
        capacity = chat_evaluation_concurrency()
        return "chat_evaluation_limiter", "chat_evaluation_thread_limiter", capacity
    # any other value (or none) is research traffic on the shared lane.
    return "chat_forwarding_limiter", "chat_thread_limiter", None


def _chat_overloaded(
    status_code: int, reason: str, limiter: anyio.CapacityLimiter
) -> HTTPException:
    stats = limiter.statistics()
    logger.warning(
        "chat_admission_rejected reason=%s in_flight=%d waiting=%d capacity=%s",
        reason,
        stats.borrowed_tokens,
        stats.tasks_waiting,
        limiter.total_tokens,
    )
    return HTTPException(
        status_code=status_code,
        detail=f"checkpoint chat is at capacity ({reason}); retry after backoff",
        headers={"Retry-After": _CHAT_RETRY_AFTER_SECONDS},
    )


async def admitted_managed_chat(
    request: Any,
    run_id: str,
    payload: dict[str, Any],
    key: dict[str, Any],
    x_freesolo_org_id: str | None,
    x_freesolo_project_id: str | None,
) -> Any:
    """forward one managed chat inside a bounded slot, shedding load when saturated."""

    state = request.app.state
    limiter_name, thread_limiter_name, capacity = _chat_lane(request)
    limiter = _state_limiter(state, limiter_name, capacity)
    lane = "evaluation " if capacity is not None else ""
    stats = limiter.statistics()
    queue_limit = int(_env_number(_CHAT_QUEUE_LIMIT_ENV, 128, minimum=0))
    if stats.borrowed_tokens >= limiter.total_tokens and stats.tasks_waiting >= queue_limit:
        raise _chat_overloaded(429, f"{lane}queue_full", limiter)
    slot = object()
    try:
        # Take an idle slot without a checkpoint, so a zero queue wait still admits.
        limiter.acquire_on_behalf_of_nowait(slot)
    except anyio.WouldBlock:
        with anyio.move_on_after(_env_number(_CHAT_QUEUE_WAIT_ENV, 30.0, minimum=0.0)) as wait:
            await limiter.acquire_on_behalf_of(slot)
        if wait.cancelled_caught:
            raise _chat_overloaded(503, f"{lane}queue_timeout", limiter) from None
    try:
        if await request.is_disconnected():
            # The caller gave up while queued; do not spend a generation on nobody.
            raise HTTPException(status_code=499, detail="client disconnected before forwarding")
        # The thread budget mirrors the admission budget, so it never queues itself and
        # never borrows the default AnyIO workers used by authentication and status routes.
        return await anyio.to_thread.run_sync(
            managed_chat,
            run_id,
            payload,
            key,
            x_freesolo_org_id,
            x_freesolo_project_id,
            limiter=_state_limiter(state, thread_limiter_name, capacity),
        )
    finally:
        limiter.release_on_behalf_of(slot)


def _stream_limiter() -> anyio.CapacityLimiter:
    global _STREAM_LIMITER
    if _STREAM_LIMITER is None:
        _STREAM_LIMITER = anyio.CapacityLimiter(chat_forwarding_concurrency())
    return _STREAM_LIMITER


async def _iterate_on_chat_workers(chunks: Iterator[bytes]):
    """read a blocking upstream stream without borrowing the default AnyIO workers."""

    end = object()
    limiter = _stream_limiter()
    while (chunk := await anyio.to_thread.run_sync(next, chunks, end, limiter=limiter)) is not end:
        yield chunk


def _upstream_response_headers(headers: dict[str, str]) -> dict[str, str]:
    connection_tokens = {
        token.strip().lower()
        for key, value in headers.items()
        if key.lower() == "connection"
        for token in value.split(",")
        if token.strip()
    }
    excluded = _UPSTREAM_RESPONSE_HEADER_EXCLUSIONS | connection_tokens
    return {key: value for key, value in headers.items() if key.lower() not in excluded}


class _UpstreamStreamingResponse(StreamingResponse):
    def __init__(self, *args: Any, upstream: RawChatStream, **kwargs: Any) -> None:
        self._upstream = upstream
        super().__init__(*args, **kwargs)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Closing a synchronous HTTP stream can wait on transport locks or I/O.
            # Keep health checks responsive, and finish cleanup even on disconnect.
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(self._upstream.close, limiter=_stream_limiter())


def _resolve_chat_request(
    run_id: str,
    payload: dict[str, Any],
    key: dict[str, Any],
    x_freesolo_org_id: str | None,
    x_freesolo_project_id: str | None,
) -> tuple[Any, list[dict[str, Any]], Any, str, str]:
    status = require_manageable_run(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
    try:
        request = parse_chat_request(
            payload,
            require_model=False,
            allow_managed_selectors=True,
        )
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    messages = _managed_chat_messages(request.messages)
    target = payload.get("checkpoint_id")
    deployment = status.deployments.get(target, {}) if isinstance(target, str) else {}
    authorized_checkpoint = _authorized_chat_checkpoint(
        run_id,
        deployment,
        payload.get("checkpoint_id"),
    )
    try:
        effective_spec = effective_spec_from_status(status)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    authorized_checkpoint = _require_active_deployment(
        run_id, status.state, deployment, authorized_checkpoint
    )
    if not effective_spec.train.hf_repo:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} has no [train].hf_repo; its adapter cannot be served",
        )
    try:
        reject_thinking_logprobs(
            thinking=effective_spec.thinking,
            logprobs=request.logprobs,
        )
        reject_tool_capability(
            tools=request.tools,
            tool_choice=request.tool_choice,
            thinking=effective_spec.thinking,
            tool_parser=qualified_tool_parser(effective_spec.model),
        )
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    org_id = run_serving_org_id(status)
    if not org_id:
        raise HTTPException(status_code=409, detail=f"run {run_id} has no organization scope")
    return request, messages, effective_spec, authorized_checkpoint, org_id


def _require_active_deployment(
    run_id: str,
    run_state: str,
    deployment: dict[str, Any],
    authorized_checkpoint: str | None,
) -> str:
    if authorized_checkpoint is not None:
        return authorized_checkpoint
    deployment_state = deployment.get("state")
    if deployment_state in _DEPLOYMENT_BUSY_STATES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"run {run_id} deployment is {deployment_state}; run "
                "`flash models deployments` to check progress"
            ),
        )
    if deployment_state == "failed":
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} deployment failed: {deployment.get('error') or 'unknown error'}",
        )
    if run_state == "cancelled":
        raise HTTPException(
            status_code=409,
            detail=(
                f"run {run_id} was cancelled; deploy a checkpoint with "
                f"`flash models deploy {run_id}/step-<N>` first"
            ),
        )
    raise HTTPException(
        status_code=409,
        detail=(
            f"run {run_id} has no active deployment; deploy an exact checkpoint ID "
            "with `flash models deploy <checkpoint-id>` first"
        ),
    )


def _tool_forward_fields(request: Any) -> dict[str, Any]:
    if request.tools is None:
        return {}
    return {
        "tools": request.tools,
        "tool_choice": request.tool_choice,
        "parallel_tool_calls": request.parallel_tool_calls,
    }


def _logprob_forward_fields(request: Any) -> dict[str, Any]:
    if request.logprob_token_ids is None:
        return {}
    return {"logprob_token_ids": list(request.logprob_token_ids)}


def _forward_stream(
    *,
    run_id: str,
    request: Any,
    messages: list[dict[str, Any]],
    thinking: bool,
    stop_sequences: list[str] | None,
    chat_template_kwargs: dict[str, Any],
    provenance: CheckpointProvenance,
    org_id: str,
) -> _UpstreamStreamingResponse:
    upstream: RawChatStream = serving_deploy.chat_sse(
        run_id=run_id,
        messages=messages,
        org_id=org_id,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        thinking=thinking,
        top_p=request.top_p,
        n=request.n,
        seed=request.seed,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        logprobs=request.logprobs,
        top_logprobs=request.top_logprobs,
        stop=stop_sequences,
        chat_template_kwargs=chat_template_kwargs,
        structured_outputs=request.structured_outputs,
        stream_options=request.stream_options,
        **_tool_forward_fields(request),
        **_logprob_forward_fields(request),
    )
    try:
        content_type = upstream.headers.get("content-type", "")
        headers = _upstream_response_headers(upstream.headers)
        if 200 <= upstream.status_code < 300:
            if not is_event_stream_content_type(content_type):
                raise ValueError("serving backend returned a non-sse streaming response")
            normalized_headers = {key.lower(): value for key, value in upstream.headers.items()}
            attested_adapter = normalized_headers.get("x-freesolo-lora-request-adapter")
            if not attested_adapter:
                raise ValueError("serving backend omitted LoRA request adapter attestation")
            if attested_adapter != provenance.checkpoint_id:
                raise ValueError(
                    "serving backend returned mismatched LoRA request adapter attestation"
                )
            validate_header_provenance(upstream.headers, provenance)
            headers.update(provenance.freesolo_headers())
        return _UpstreamStreamingResponse(
            _iterate_on_chat_workers(iter(upstream.iter_bytes())),
            status_code=upstream.status_code,
            headers=headers,
            upstream=upstream,
        )
    except BaseException:
        upstream.close()
        raise


def managed_chat(
    run_id: str,
    payload: dict[str, Any],
    key: dict[str, Any],
    x_freesolo_org_id: str | None,
    x_freesolo_project_id: str | None,
    *,
    timeout_s: float | None = None,
    serving_provider: str | None = None,
) -> Any:
    request, messages, effective_spec, authorized_checkpoint, org_id = _resolve_chat_request(
        run_id,
        payload,
        key,
        x_freesolo_org_id,
        x_freesolo_project_id,
    )
    mandatory_stops = tuple(getattr(effective_spec.train, "stop_sequences", ()) or ())
    stop_sequences = merge_stop_sequences(mandatory_stops, request.stop)
    try:
        validate_tool_stop_sequences(
            stop_sequences or (),
            tools=request.tools,
            tool_choice=request.tool_choice,
            error_type=OpenAIRequestError,
        )
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    chat_template_kwargs = {
        **request.chat_template_kwargs,
        "enable_thinking": effective_spec.thinking,
    }
    provenance = CheckpointProvenance(authorized_checkpoint)
    try:
        if request.stream:
            return _forward_stream(
                run_id=authorized_checkpoint,
                request=request,
                messages=messages,
                thinking=effective_spec.thinking,
                stop_sequences=stop_sequences,
                chat_template_kwargs=chat_template_kwargs,
                provenance=provenance,
                org_id=org_id,
            )
        response = serving_deploy.chat(
            run_id=authorized_checkpoint,
            messages=messages,
            org_id=org_id,
            timeout_s=chat_upstream_timeout_s() if timeout_s is None else timeout_s,
            serving_provider=serving_provider,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            thinking=effective_spec.thinking,
            top_p=request.top_p,
            n=request.n,
            seed=request.seed,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            logprobs=request.logprobs,
            top_logprobs=request.top_logprobs,
            stop=stop_sequences,
            chat_template_kwargs=chat_template_kwargs,
            structured_outputs=request.structured_outputs,
            **_tool_forward_fields(request),
            **_logprob_forward_fields(request),
        )
        if not isinstance(response, dict):
            raise ValueError("serving backend returned a non-object chat response")
        return validate_body_provenance(response, provenance)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        raise upstream_chat_error(exc, checkpoint_id=authorized_checkpoint) from exc
    except Exception as exc:
        # Exception text can carry upstream detail; the type is enough to classify the failure.
        logger.warning(
            "serving_failure checkpoint=%s error=%s", authorized_checkpoint, type(exc).__name__
        )
        raise HTTPException(status_code=502, detail=f"inference failure: {exc}") from exc

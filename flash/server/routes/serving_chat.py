"""managed chat resolution and forwarding for immutable deployed runs."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

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
from flash.server.asgi import app as _app
from flash.server.platform.deps import manageable_run
from flash.server.platform.internal_client import run_serving_org_id
from flash.server.routes.serving_revisions import (
    _DEPLOYMENT_BUSY_STATES,
    _authorized_chat_checkpoint,
    _managed_chat_messages,
    _verified_checkpoints,
)

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
            self._upstream.close()


def _resolve_chat_request(
    run_id: str,
    payload: dict[str, Any],
    key: dict[str, Any],
    x_freesolo_org_id: str | None,
    x_freesolo_project_id: str | None,
) -> tuple[Any, list[dict[str, Any]], Any, str]:
    status = manageable_run(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
    try:
        request = parse_chat_request(
            payload,
            require_model=False,
            allow_managed_selectors=True,
        )
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    messages = _managed_chat_messages(request.messages)
    deployment = status.deployment or {}
    authorized_checkpoint = _authorized_chat_checkpoint(
        run_id,
        deployment,
        payload.get("checkpoint_id"),
        _verified_checkpoints(status),
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
        detail=f"run {run_id} has no active deployment; `flash models deploy {run_id}` first",
    )


def _tool_forward_fields(request: Any) -> dict[str, Any]:
    if request.tools is None:
        return {}
    return {
        "tools": request.tools,
        "tool_choice": request.tool_choice,
        "parallel_tool_calls": request.parallel_tool_calls,
    }


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
    upstream: RawChatStream = _app.serve_chat_sse(
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
            upstream.iter_bytes(),
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
        response = _app.serve_chat(
            run_id=authorized_checkpoint,
            messages=messages,
            org_id=org_id,
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
        )
        if not isinstance(response, dict):
            raise ValueError("serving backend returned a non-object chat response")
        return validate_body_provenance(response, provenance)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"inference failure: {exc}") from exc

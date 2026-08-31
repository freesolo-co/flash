"""Caller-facing inference routes: /generate, /adapters/{id}/generate, /v1/chat/completions.

Split out of router.py's app builder, where these were nested handlers closing over a dozen app
variables. They reach that state through ``ServingContext.of(request)`` instead.

Every route here is gated by ``ServingContext.authorize_inference``, which requires a Freesolo API
key unless the caller presents the shared internal key.
"""

import asyncio
import time
import uuid
from collections.abc import Awaitable
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from flash.serve.request.openai import (
    OpenAIRequestError,
    parse_chat_request,
    reject_thinking_logprobs,
    reject_tool_capability,
)
from flash.serving.src.accounting.usage import (
    AuthorizedTraffic,
    InferenceAuthorization,
    captured_now,
    new_request_identity,
)
from flash.serving.src.accounting.usage_outbox import UsageOutboxError
from flash.serving.src.engine.model_config import tool_parser_for
from flash.serving.src.http.context import ServingContext, require_attributed_traffic
from flash.serving.src.io.multimodal import _prepare_generate_request
from flash.serving.src.io.provenance import _checkpoint_provenance, _provenance_headers
from flash.serving.src.io.requests import (
    _expected_checkpoint,
    _parse_generate,
    _parse_openai_generate,
)
from flash.serving.src.io.responses import (
    _inference_json_response,
    openai_chat_completion,
    openai_generate_fields,
)
from flash.serving.src.io.schemas import GenerateRequest
from flash.serving.src.io.streaming import _close_async_iterator

inference_router = APIRouter()


@inference_router.post("/generate", tags=["inference"])
async def generate(payload: GenerateRequest, request: Request) -> JSONResponse:
    context = ServingContext.of(request)
    authorization = await context.authorize_inference(request, payload.adapter_id)
    requested, target = await context.lookup.resolve(
        payload.adapter_id, org_id=_authorization_org_id(authorization)
    )
    traffic = require_attributed_traffic(authorization, target)
    context.reject_unsettleable_thinking(payload, target)
    await _prepare_generate_request(payload, target)
    identity = new_request_identity(request)
    admitted_at = captured_now()
    try:
        result = await _await_until_disconnect(
            request,
            context.generate(
                payload,
                requested,
                target,
                identity=identity,
                traffic=traffic,
                captured_at=admitted_at,
                expected_checkpoint=_expected_checkpoint(request),
            ),
        )
    except UsageOutboxError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "durable serving accounting unavailable"
        ) from exc
    return _inference_json_response(result, target)


@inference_router.post("/adapters/{adapter_id:path}/generate", tags=["inference"])
async def generate_for_adapter(
    adapter_id: str, payload: dict[str, Any], request: Request
) -> JSONResponse:
    context = ServingContext.of(request)
    normalized_adapter_id = _path_adapter_id(adapter_id)
    authorization = await context.authorize_inference(request, normalized_adapter_id)
    req = _parse_generate({**payload, "adapter_id": normalized_adapter_id})
    requested, target = await context.lookup.resolve(
        req.adapter_id, org_id=_authorization_org_id(authorization)
    )
    traffic = require_attributed_traffic(authorization, target)
    context.reject_unsettleable_thinking(req, target)
    await _prepare_generate_request(req, target)
    identity = new_request_identity(request)
    admitted_at = captured_now()
    try:
        result = await _await_until_disconnect(
            request,
            context.generate(
                req,
                requested,
                target,
                identity=identity,
                traffic=traffic,
                captured_at=admitted_at,
                expected_checkpoint=_expected_checkpoint(request),
            ),
        )
    except UsageOutboxError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "durable serving accounting unavailable"
        ) from exc
    return _inference_json_response(result, target)


def _authorization_org_id(authorization: InferenceAuthorization) -> str | None:
    # tenant scope for the lookup itself. an external key is already bound to its org; a trusted
    # internal caller scopes by the org header when it supplied one.
    if isinstance(authorization, AuthorizedTraffic):
        return authorization.principal.orgId
    return authorization.org_id


def _path_adapter_id(adapter_id: str) -> str:
    """return the stripped path adapter id, rejecting a blank one before the authorizer runs."""
    stripped = adapter_id.strip()
    if not stripped:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "adapter_id must not be empty")
    return stripped


def _openai_adapter_id(payload: dict[str, Any]) -> str:
    """return the stripped adapter id required to authorize an openai-shaped request."""
    adapter_id = payload.get("model")
    if not isinstance(adapter_id, str) or not adapter_id.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model must be the adapter id")
    return adapter_id.strip()


@inference_router.post("/v1/chat/completions", tags=["openai"])
async def chat_completions(payload: dict[str, Any], request: Request) -> Any:
    context = ServingContext.of(request)
    adapter_id = _openai_adapter_id(payload)
    authorization = await context.authorize_inference(request, adapter_id)
    try:
        normalized = parse_chat_request(
            payload,
            require_model=True,
            allow_managed_selectors=False,
        )
    except OpenAIRequestError as exc:
        request_status = (
            status.HTTP_400_BAD_REQUEST
            if str(exc).startswith(("message ", "stream must"))
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(request_status, str(exc)) from exc
    requested, target = await context.lookup.resolve(
        adapter_id, org_id=_authorization_org_id(authorization)
    )
    traffic = require_attributed_traffic(authorization, target)
    effective_thinking = target.thinking
    if target.serve_base_model:
        override = normalized.chat_template_kwargs.get("enable_thinking")
        if type(override) is bool:
            effective_thinking = override
    try:
        reject_thinking_logprobs(thinking=effective_thinking, logprobs=normalized.logprobs)
        reject_tool_capability(
            tools=normalized.tools,
            tool_choice=normalized.tool_choice,
            thinking=effective_thinking,
            tool_parser=tool_parser_for(target.base_model),
        )
    except OpenAIRequestError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    generate_fields = openai_generate_fields(normalized, adapter_id)
    generate_fields["chat_template_kwargs"] = {
        **normalized.chat_template_kwargs,
        "enable_thinking": effective_thinking,
    }
    req = _parse_openai_generate(generate_fields)
    context.reject_unsettleable_thinking(req, target)
    stream = normalized.stream
    include_usage = normalized.include_usage
    await _prepare_generate_request(req, target)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    identity = new_request_identity(request, openai_completion_id=completion_id)
    admitted_at = captured_now()
    created = int(time.time())
    if stream:
        return await _stream_chat_completion(
            context,
            request,
            req,
            requested,
            target,
            adapter_id=adapter_id,
            completion_id=completion_id,
            created=created,
            include_usage=include_usage,
            identity=identity,
            traffic=traffic,
            admitted_at=admitted_at,
        )

    try:
        generation = await _await_until_disconnect(
            request,
            context.generate(
                req,
                requested,
                target,
                identity=identity,
                traffic=traffic,
                captured_at=admitted_at,
                expected_checkpoint=_expected_checkpoint(request),
            ),
        )
    except UsageOutboxError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "durable serving accounting unavailable"
        ) from exc
    # already attested in `generate_once`, before usage was metered; this only strips the field
    # off the body so it does not leak into the OpenAI-shaped response.
    lora_request_adapter = generation.pop("lora_request_adapter", None)
    active_checkpoint = generation.get("checkpoint")
    provenance = _checkpoint_provenance(target, active_checkpoint)
    response = openai_chat_completion(
        completion_id=completion_id,
        created=created,
        adapter_id=adapter_id,
        generation=generation,
        provenance=provenance,
    )
    response_headers = _provenance_headers(provenance, active_checkpoint)
    if target.is_checkpoint:
        response_headers["X-Freesolo-LoRA-Request-Adapter"] = lora_request_adapter
    return JSONResponse(response, headers=response_headers)


async def _await_until_disconnect(request: Request, awaitable: Awaitable[Any]) -> Any:
    operation = asyncio.ensure_future(awaitable)
    disconnect = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait({operation, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        # `done` is a SET: if generation finished and the peer left in the same tick, both are in
        # it. deciding on the disconnect first discards a completed, durably finalized result.
        # resolve the tie toward the
        # operation, matching the packaged helper in flash/serve/app/http.py.
        if operation in done:
            return operation.result()
        disconnect.result()
        raise asyncio.CancelledError
    finally:
        disconnect.cancel()
        if not operation.done():
            operation.cancel()
        await asyncio.gather(operation, disconnect, return_exceptions=True)


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def _discard_prepared_stream(usage_session: Any, events: Any) -> None:
    try:
        first = await anext(events)
        if first.get("prompt_tokens") is not None and first.get("completion_tokens") is not None:
            await usage_session.fail(first, "client_disconnected")
    except StopAsyncIteration:
        pass
    finally:
        await _close_async_iterator(events)


async def _stream_chat_completion(
    context: ServingContext,
    request: Request,
    req: Any,
    requested: Any,
    target: Any,
    *,
    adapter_id: str,
    completion_id: str,
    created: int,
    include_usage: bool,
    identity: Any,
    traffic: Any,
    admitted_at: Any,
) -> StreamingResponse:
    preparation = asyncio.create_task(
        context.prepare_stream(
            req,
            requested,
            target,
            generation_id=identity.request_id,
            expected_checkpoint=_expected_checkpoint(request),
        )
    )
    disconnect = asyncio.create_task(_wait_for_disconnect(request))
    prepared = None
    usage_session = None
    transferred = False
    try:
        done, _ = await asyncio.wait({preparation, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        if preparation not in done:
            disconnect.result()
            raise asyncio.CancelledError
        prepared = preparation.result()
        events, checkpoint_headers, thinking, first = prepared
        usage_session = context.usage_session(
            identity, traffic, requested, target, first, admitted_at
        )
        try:
            await usage_session.capture(first)
        except UsageOutboxError as exc:
            await _close_async_iterator(events)
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "durable serving accounting unavailable",
            ) from exc
        response = StreamingResponse(
            context.chat_stream(
                record=requested,
                events=events,
                adapter_id=adapter_id,
                completion_id=completion_id,
                created=created,
                include_usage=include_usage,
                usage_session=usage_session,
                thinking=thinking,
                choice_count=getattr(req, "n", 1),
            ),
            media_type="text/event-stream",
            # Disable proxy and CDN buffering so each SSE chunk reaches the client immediately.
            # Without X-Accel-Buffering, Nginx accumulates tokens until its output buffer fills,
            # adding 100+ ms of hidden TTFT for small completions.
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                **checkpoint_headers,
            },
        )
        transferred = True
        return response
    finally:
        disconnect.cancel()
        if not preparation.done():
            preparation.cancel()
        preparation_result, _ = await asyncio.gather(
            preparation, disconnect, return_exceptions=True
        )
        if prepared is None and isinstance(preparation_result, tuple):
            prepared = preparation_result
        if prepared is not None and not transferred:
            if usage_session is None:
                first = prepared[3]
                usage_session = context.usage_session(
                    identity, traffic, requested, target, first, admitted_at
                )
            await _discard_prepared_stream(usage_session, prepared[0])

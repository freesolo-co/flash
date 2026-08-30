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
)
from flash.serving.src.accounting.usage import captured_now, new_request_identity
from flash.serving.src.accounting.usage_outbox import CapturedPrice, UsageOutboxError
from flash.serving.src.http.context import ServingContext
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
    traffic = await context.authorize_inference(request, payload.adapter_id)
    requested, target = await context.lookup.resolve(
        payload.adapter_id, org_id=context.traffic_org_id(traffic)
    )
    context.reject_unsettleable_thinking(payload, target)
    price = _capture_price(context, traffic, target)
    identity = new_request_identity(request, traffic=traffic)
    admitted_at = captured_now()
    usage_session = await _admit_usage(
        context, identity, traffic, requested, target, price, admitted_at
    )
    await _prepare_admitted_request(context, usage_session, payload, target)
    try:
        result = await _await_until_disconnect(
            request,
            context.generate(
                payload,
                requested,
                target,
                usage_session=usage_session,
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
    traffic = await context.authorize_inference(request, normalized_adapter_id)
    req = _parse_generate({**payload, "adapter_id": normalized_adapter_id})
    requested, target = await context.lookup.resolve(
        req.adapter_id, org_id=context.traffic_org_id(traffic)
    )
    context.reject_unsettleable_thinking(req, target)
    price = _capture_price(context, traffic, target)
    identity = new_request_identity(request, traffic=traffic)
    admitted_at = captured_now()
    usage_session = await _admit_usage(
        context, identity, traffic, requested, target, price, admitted_at
    )
    await _prepare_admitted_request(context, usage_session, req, target)
    try:
        result = await _await_until_disconnect(
            request,
            context.generate(
                req,
                requested,
                target,
                usage_session=usage_session,
                expected_checkpoint=_expected_checkpoint(request),
            ),
        )
    except UsageOutboxError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "durable serving accounting unavailable"
        ) from exc
    return _inference_json_response(result, target)


def _capture_price(context: ServingContext, traffic: Any, target: Any) -> CapturedPrice:
    try:
        return context.capture_price(traffic, target)
    except UsageOutboxError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "durable serving accounting unavailable"
        ) from exc


async def _admit_usage(
    context: ServingContext,
    identity: Any,
    traffic: Any,
    requested: Any,
    target: Any,
    price: CapturedPrice,
    admitted_at: Any,
) -> Any:
    try:
        return await context.admit_usage(identity, traffic, requested, target, price, admitted_at)
    except UsageOutboxError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "durable serving accounting unavailable"
        ) from exc


async def _prepare_admitted_request(
    context: ServingContext, usage_session: Any, payload: Any, target: Any
) -> None:
    try:
        await _prepare_generate_request(payload, target)
    except BaseException:
        await context.fail_usage(usage_session, "request_preparation_failed")
        raise


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
    traffic = await context.authorize_inference(request, adapter_id)
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
        adapter_id, org_id=context.traffic_org_id(traffic)
    )
    effective_thinking = target.thinking
    if target.serve_base_model:
        override = normalized.chat_template_kwargs.get("enable_thinking")
        if type(override) is bool:
            effective_thinking = override
    try:
        reject_thinking_logprobs(thinking=effective_thinking, logprobs=normalized.logprobs)
    except OpenAIRequestError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    generate_fields = openai_generate_fields(normalized, adapter_id)
    generate_fields["chat_template_kwargs"] = {
        **normalized.chat_template_kwargs,
        "enable_thinking": effective_thinking,
    }
    req = _parse_openai_generate(generate_fields)
    context.reject_unsettleable_thinking(req, target)
    price = _capture_price(context, traffic, target)
    stream = normalized.stream
    include_usage = normalized.include_usage
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    identity = new_request_identity(request, openai_completion_id=completion_id, traffic=traffic)
    admitted_at = captured_now()
    usage_session = await _admit_usage(
        context, identity, traffic, requested, target, price, admitted_at
    )
    await _prepare_admitted_request(context, usage_session, req, target)
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
            usage_session=usage_session,
        )

    try:
        generation = await _await_until_disconnect(
            request,
            context.generate(
                req,
                requested,
                target,
                usage_session=usage_session,
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
        else:
            await usage_session.fail_admission("client_disconnected")
    except StopAsyncIteration:
        pass
    finally:
        await _close_async_iterator(events)


def _streaming_response(
    context: ServingContext,
    req: Any,
    requested: Any,
    prepared: tuple[Any, dict[str, str], bool, dict[str, Any]],
    usage_session: Any,
    *,
    adapter_id: str,
    completion_id: str,
    created: int,
    include_usage: bool,
) -> StreamingResponse:
    events, checkpoint_headers, thinking, _ = prepared
    return StreamingResponse(
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
        # disable proxy and cdn buffering so each sse chunk reaches the client immediately.
        # without x-accel-buffering, nginx accumulates tokens until its output buffer fills,
        # adding 100+ ms of hidden ttft for small completions.
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            **checkpoint_headers,
        },
    )


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
    usage_session: Any,
) -> StreamingResponse:
    preparation = asyncio.create_task(
        context.prepare_stream(
            req,
            requested,
            target,
            generation_id=usage_session.identity.request_id,
            expected_checkpoint=_expected_checkpoint(request),
        )
    )
    disconnect = asyncio.create_task(_wait_for_disconnect(request))
    prepared = None
    transferred = False
    failure_code = "generation_failed"
    try:
        done, _ = await asyncio.wait({preparation, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        if preparation not in done:
            disconnect.result()
            failure_code = "client_disconnected"
            raise asyncio.CancelledError
        prepared = preparation.result()
        usage_session = usage_session.with_attestation(prepared[3])
        response = _streaming_response(
            context,
            req,
            requested,
            prepared,
            usage_session,
            adapter_id=adapter_id,
            completion_id=completion_id,
            created=created,
            include_usage=include_usage,
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
            usage_session = usage_session.with_attestation(prepared[3])
        if prepared is not None and not transferred:
            await _discard_prepared_stream(usage_session, prepared[0])
        elif prepared is None:
            await context.fail_usage(usage_session, failure_code)

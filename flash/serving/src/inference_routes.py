"""Caller-facing inference routes: /generate, /adapters/{id}/generate, /v1/chat/completions.

Split out of router.py's app builder, where these were nested handlers closing over a dozen app
variables. They reach that state through ``ServingContext.of(request)`` instead.

Every route here is gated by ``ServingContext.authorize_inference``, which requires a Freesolo API
key unless the caller presents the shared internal key.
"""

import asyncio
import re
import time
import uuid
from collections.abc import Awaitable
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from flash.serving.src.context import ServingContext
from flash.serving.src.responses import (
    openai_chat_completion,
    openai_generate_fields,
    openai_include_usage,
)
from flash.serving.src.schemas import GenerateRequest
from flash.serving.src.serving_io import (
    _expected_checkpoint,
    _inference_json_response,
    _parse_generate,
    _prepare_generate_request,
    _provenance_headers,
    _revision_provenance,
)
from flash.serving.src.streaming import _close_async_iterator
from flash.serving.src.structured_outputs import StructuredOutputsError

_FLASH_CHECKPOINT_MODEL_RE = re.compile(
    r"(?P<run_id>flash-[0-9]{1,20}-[0-9a-f]{8})/step-[0-9]{1,18}"
)

inference_router = APIRouter()


@inference_router.post("/generate", tags=["inference"])
async def generate(payload: GenerateRequest, request: Request) -> JSONResponse:
    context = ServingContext.of(request)
    caller_org = await context.authorize_inference(request, payload.adapter_id)
    requested, target = await context.lookup.resolve(payload.adapter_id)
    await _prepare_generate_request(payload, target)
    result = await _await_until_disconnect(
        request,
        context.generate(
            payload,
            requested,
            target,
            expected_checkpoint=_expected_checkpoint(request),
            caller_org=caller_org,
        ),
    )
    return _inference_json_response(result, target)


@inference_router.post("/adapters/{adapter_id}/generate", tags=["inference"])
async def generate_for_adapter(
    adapter_id: str, payload: dict[str, Any], request: Request
) -> JSONResponse:
    # Parse first so the GenerateRequest validator normalizes (strips) the adapter id, then
    # authorize and route against that same normalized value (not the raw path parameter).
    context = ServingContext.of(request)
    req = _parse_generate({**payload, "adapter_id": adapter_id})
    caller_org = await context.authorize_inference(request, req.adapter_id)
    requested, target = await context.lookup.resolve(req.adapter_id)
    await _prepare_generate_request(req, target)
    result = await _await_until_disconnect(
        request,
        context.generate(
            req,
            requested,
            target,
            expected_checkpoint=_expected_checkpoint(request),
            caller_org=caller_org,
        ),
    )
    return _inference_json_response(result, target)


def _openai_model_id(payload: dict[str, Any]) -> str:
    """The adapter id an OpenAI-shaped request is asking for, stripped and validated.

    Rejecting a checkpoint identifier here is the difference between a clear 400 and a confusing
    404: `flash-<run>/step-<n>` is a real thing the caller has, just not a servable model id.
    """
    adapter_id = payload.get("model")
    if not isinstance(adapter_id, str) or not adapter_id.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model must be the adapter id")
    # use the stripped id consistently for validation, auth, routing, and the echoed response
    # model, so a caller that sends "  qa  " is authorized against and routed to "qa".
    adapter_id = adapter_id.strip()
    match = _FLASH_CHECKPOINT_MODEL_RE.fullmatch(adapter_id)
    if match is not None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "This is a checkpoint identifier, not a serving model identifier. "
            f"Deploy it first or use model {match.group('run_id')}.",
        )
    return adapter_id


@inference_router.post("/v1/chat/completions", tags=["openai"])
async def chat_completions(payload: dict[str, Any], request: Request) -> Any:
    context = ServingContext.of(request)
    adapter_id = _openai_model_id(payload)
    caller_org = await context.authorize_inference(request, adapter_id)
    requested, target = await context.lookup.resolve(adapter_id)
    try:
        fields = openai_generate_fields(payload, adapter_id)
    except StructuredOutputsError as exc:
        # Malformed OpenAI response_format (json_schema with no schema, or an unknown type) ->
        # 422, not 500.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    req = _parse_generate(fields)
    await _prepare_generate_request(req, target)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    include_usage = openai_include_usage(payload)
    if payload.get("stream") is True:
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
            caller_org=caller_org,
        )

    generation = await _await_until_disconnect(
        request,
        context.generate(
            req,
            requested,
            target,
            expected_checkpoint=_expected_checkpoint(request),
            caller_org=caller_org,
        ),
    )
    # already attested in `generate_once`, before usage was metered; this only strips the field
    # off the body so it does not leak into the OpenAI-shaped response.
    lora_request_adapter = generation.pop("lora_request_adapter", None)
    active_checkpoint = generation.get("checkpoint")
    provenance = _revision_provenance(target, active_checkpoint)
    response = openai_chat_completion(
        completion_id=completion_id,
        created=created,
        adapter_id=adapter_id,
        generation=generation,
        provenance=provenance,
    )
    response_headers = _provenance_headers(provenance, active_checkpoint)
    if target.is_revision:
        response_headers["X-Freesolo-LoRA-Request-Adapter"] = lora_request_adapter
    return JSONResponse(response, headers=response_headers)


async def _await_until_disconnect(request: Request, awaitable: Awaitable[Any]) -> Any:
    operation = asyncio.ensure_future(awaitable)
    disconnect = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait({operation, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        if disconnect in done:
            disconnect.result()
            raise asyncio.CancelledError
        return operation.result()
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


async def _discard_prepared_stream(
    context: ServingContext,
    requested: Any,
    caller_org: str | None,
    events: Any,
) -> None:
    try:
        # prepare_stream always returns a replay iterator whose first advance is already-resolved.
        # advancing it activates the wrapper's finally block and preserves billing for first-output
        # work that won the race with a disconnect, without waiting for another engine event.
        first = await anext(events)
        if first.get("prompt_tokens") is not None and first.get("completion_tokens") is not None:
            context.schedule_usage(requested, first, caller_org)
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
    caller_org: str | None,
) -> StreamingResponse:
    preparation = asyncio.create_task(
        context.prepare_stream(
            req,
            requested,
            target,
            expected_checkpoint=_expected_checkpoint(request),
        )
    )
    disconnect = asyncio.create_task(_wait_for_disconnect(request))
    prepared = None
    transferred = False
    try:
        done, _ = await asyncio.wait({preparation, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        if disconnect in done:
            disconnect.result()
            raise asyncio.CancelledError
        prepared = preparation.result()
        events, checkpoint_headers, thinking = prepared
        response = StreamingResponse(
            context.chat_stream(
                record=requested,
                events=events,
                adapter_id=adapter_id,
                completion_id=completion_id,
                created=created,
                include_usage=include_usage,
                caller_org=caller_org,
                thinking=thinking,
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
            await _discard_prepared_stream(context, requested, caller_org, prepared[0])

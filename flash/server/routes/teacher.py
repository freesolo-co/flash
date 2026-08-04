"""Operation-specific managed-teacher broker routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from flash.server.teacher_broker import (
    MAX_REQUEST_BODY_BYTES,
    TeacherBrokerError,
    authenticate_teacher_capability,
    complete_fireworks_request,
)

router = APIRouter()
MAX_CONCURRENT_BODY_READERS = 8
_BODY_INGRESS_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_BODY_READERS)


async def _bounded_body(request: Request) -> bytearray:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise TeacherBrokerError("invalid_content_length", status_code=400) from exc
        if declared < 0 or declared > MAX_REQUEST_BODY_BYTES:
            raise TeacherBrokerError("request_too_large", status_code=413)
    body = bytearray()
    async for chunk in request.stream():
        if len(chunk) > MAX_REQUEST_BODY_BYTES - len(body):
            raise TeacherBrokerError("request_too_large", status_code=413)
        body.extend(chunk)
    return body


def _bearer(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token or " " in token:
        raise TeacherBrokerError("invalid_capability", status_code=401)
    return token


@router.post("/v1/teacher/completions")
async def teacher_completions(request: Request):
    request_id = request.headers.get("x-flash-teacher-request-id", "")
    try:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise TeacherBrokerError("unsupported_content_type", status_code=415)
        capability = _bearer(request)
        await asyncio.to_thread(
            authenticate_teacher_capability,
            capability_token=capability,
            request_id=request_id,
        )
        async with _BODY_INGRESS_SEMAPHORE:
            body = await _bounded_body(request)
        response = await asyncio.to_thread(
            complete_fireworks_request,
            capability_token=capability,
            request_id=request_id,
            raw_body=body,
        )
        return JSONResponse(response)
    except TeacherBrokerError as exc:
        return JSONResponse(exc.payload(), status_code=exc.status_code)

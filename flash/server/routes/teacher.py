"""Operation-specific managed-teacher broker routes."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import threading
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from flash.server.domain.teacher.broker import (
    MAX_REQUEST_BODY_BYTES,
    TeacherBrokerError,
    authenticate_teacher_capability,
    complete_teacher_chat_request,
    complete_teacher_request,
)

router = APIRouter()
MAX_CONCURRENT_BODY_READERS = 8
MAX_CONCURRENT_BODY_READERS_PER_CAPABILITY = 2
MAX_BODY_INGRESS_SECONDS = 30.0
_BODY_INGRESS_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_BODY_READERS)
_CAPABILITY_INGRESS_LOCK = threading.Lock()
_CAPABILITY_INGRESS: dict[str, tuple[asyncio.Semaphore, int]] = {}


@contextlib.asynccontextmanager
async def _body_ingress_slot(capability_token: str, *, max_seconds: float):
    key = hashlib.sha256(capability_token.encode()).hexdigest()
    with _CAPABILITY_INGRESS_LOCK:
        semaphore, references = _CAPABILITY_INGRESS.get(
            key,
            (asyncio.Semaphore(MAX_CONCURRENT_BODY_READERS_PER_CAPABILITY), 0),
        )
        _CAPABILITY_INGRESS[key] = (semaphore, references + 1)
    capability_acquired = False
    global_acquired = False
    try:
        async with asyncio.timeout(max_seconds):
            await semaphore.acquire()
            capability_acquired = True
            await _BODY_INGRESS_SEMAPHORE.acquire()
            global_acquired = True
            yield
    except TimeoutError as exc:
        raise TeacherBrokerError(
            "body_ingress_timeout",
            status_code=408,
            retryable=True,
        ) from exc
    finally:
        if global_acquired:
            _BODY_INGRESS_SEMAPHORE.release()
        if capability_acquired:
            semaphore.release()
        with _CAPABILITY_INGRESS_LOCK:
            current = _CAPABILITY_INGRESS.get(key)
            if current is not None:
                current_semaphore, current_references = current
                if current_references <= 1:
                    _CAPABILITY_INGRESS.pop(key, None)
                else:
                    _CAPABILITY_INGRESS[key] = (
                        current_semaphore,
                        current_references - 1,
                    )


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
    try:
        async for chunk in request.stream():
            if len(chunk) > MAX_REQUEST_BODY_BYTES - len(body):
                raise TeacherBrokerError("request_too_large", status_code=413)
            body.extend(chunk)
    except (TeacherBrokerError, asyncio.CancelledError):
        raise
    except Exception as exc:
        raise TeacherBrokerError("body_ingress_failed", status_code=400) from exc
    return body


def _bearer(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token or " " in token:
        raise TeacherBrokerError("invalid_capability", status_code=401)
    return token


async def _teacher_request(request: Request, complete):
    request_id = request.headers.get("x-flash-teacher-request-id", "")
    try:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise TeacherBrokerError("unsupported_content_type", status_code=415)
        capability = _bearer(request)
        binding = await asyncio.to_thread(
            authenticate_teacher_capability,
            capability_token=capability,
            request_id=request_id,
        )
        remaining = float(binding[2]["expires_at"]) - time.time()
        if remaining <= 0:
            raise TeacherBrokerError("expired_capability", status_code=401)
        ingress_timeout = min(MAX_BODY_INGRESS_SECONDS, remaining)
        async with _body_ingress_slot(capability, max_seconds=ingress_timeout):
            body = await _bounded_body(request)
        response = await asyncio.to_thread(
            complete,
            capability_token=capability,
            request_id=request_id,
            raw_body=body,
        )
        return JSONResponse(response)
    except TeacherBrokerError as exc:
        return JSONResponse(exc.payload(), status_code=exc.status_code)


@router.post("/v1/teacher/completions")
async def teacher_completions(request: Request):
    return await _teacher_request(request, complete_teacher_request)


@router.post("/v1/teacher/chat_completions")
async def teacher_chat_completions(request: Request):
    return await _teacher_request(request, complete_teacher_chat_request)

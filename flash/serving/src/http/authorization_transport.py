"""Bounded retries for read-only authorization lookups, without logging credentials."""

import asyncio
import logging
import math
import random
import time
import uuid

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)
AUTHORIZATION_BUDGET_SECONDS = 20.0
# Enough attempts to span the deadline even when a restarting proxy rejects immediately.
# Two attempts exhausted in ~300ms, well before a healthy replacement could bind its port.
_ATTEMPTS = 20
_TRANSIENT_STATUS = {500, 502, 503, 504}
_TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


class TransientAuthorizationError(HTTPException):
    """The backend gave no authorization decision because its transport was unavailable."""

    def __init__(self, detail: str, request_id: str | None = None) -> None:
        headers = {"Retry-After": "1", "X-Freesolo-Inference-Started": "false"}
        if request_id:
            headers["X-Freesolo-Auth-Request-Id"] = request_id
        super().__init__(503, detail, headers=headers)


async def post_authorization(client, url, *, api_key, model_id, scope):
    request_id = uuid.uuid4().hex
    headers = {**(scope or {}), "X-Freesolo-Auth-Request-Id": request_id}
    started = time.monotonic()
    detail = "serving auth backend unreachable"
    try:
        async with asyncio.timeout(AUTHORIZATION_BUDGET_SECONDS):
            for attempt in range(1, _ATTEMPTS + 1):
                retry_after = 0.0
                try:
                    response = await client.post(
                        url, json={"apiKey": api_key, "modelId": model_id}, headers=headers
                    )
                except _TRANSIENT_ERRORS as error:
                    # Exception messages/URLs can contain credentials. Log only the type.
                    kind = type(error).__name__
                    detail = "serving auth backend unreachable"
                except Exception as error:
                    logger.warning(
                        "serving_auth_transport_error request_id=%s kind=%s",
                        request_id,
                        type(error).__name__,
                    )
                    raise HTTPException(503, "serving auth backend unreachable") from error
                else:
                    if response.status_code not in _TRANSIENT_STATUS:
                        logger.info(
                            "serving_auth_response request_id=%s attempt=%d status=%d elapsed_ms=%d",
                            request_id,
                            attempt,
                            response.status_code,
                            int((time.monotonic() - started) * 1000),
                        )
                        return response
                    kind = f"http_{response.status_code}"
                    detail = "serving auth backend unavailable"
                    try:
                        hint = float(response.headers.get("Retry-After", ""))
                    except ValueError:
                        pass
                    else:
                        if math.isfinite(hint) and hint >= 0:
                            retry_after = hint
                logger.warning(
                    "serving_auth_transient_failure request_id=%s kind=%s attempt=%d elapsed_ms=%d",
                    request_id,
                    kind,
                    attempt,
                    int((time.monotonic() - started) * 1000),
                )
                if attempt < _ATTEMPTS:
                    delay = min(0.25 * 2 ** (attempt - 1), 2.0)
                    # Backpressure is a minimum wait, not a jitter ceiling. The enclosing
                    # deadline cancels long hints without issuing another backend request.
                    await asyncio.sleep(max(retry_after, delay * random.uniform(0.8, 1.2)))
    except TimeoutError:
        logger.warning(
            "serving_auth_deadline_exceeded request_id=%s budget_seconds=%s",
            request_id,
            AUTHORIZATION_BUDGET_SECONDS,
        )
    raise TransientAuthorizationError(detail, request_id)

"""Bounded retries for read-only authorization lookups, without logging credentials."""

import asyncio
import logging
import random
import time

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)
AUTHORIZATION_BUDGET_SECONDS = 20.0
_ATTEMPTS = 2
_TRANSIENT_STATUS = {500, 502, 503, 504}
_TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


class TransientAuthorizationError(HTTPException):
    """The backend gave no authorization decision because its transport was unavailable."""

    def __init__(self, detail: str) -> None:
        super().__init__(503, detail, headers={"Retry-After": "1"})


async def post_authorization(client, url, *, api_key, model_id, scope):
    started = time.monotonic()
    detail = "serving auth backend unreachable"
    try:
        async with asyncio.timeout(AUTHORIZATION_BUDGET_SECONDS):
            for attempt in range(1, _ATTEMPTS + 1):
                try:
                    response = await client.post(
                        url, json={"apiKey": api_key, "modelId": model_id}, headers=scope or None
                    )
                except _TRANSIENT_ERRORS as error:
                    # Exception messages/URLs can contain credentials. Log only the type.
                    kind = type(error).__name__
                    detail = "serving auth backend unreachable"
                except Exception as error:
                    logger.warning("serving_auth_transport_error kind=%s", type(error).__name__)
                    raise HTTPException(503, "serving auth backend unreachable") from error
                else:
                    if response.status_code not in _TRANSIENT_STATUS:
                        return response
                    kind = f"http_{response.status_code}"
                    detail = "serving auth backend unavailable"
                logger.warning(
                    "serving_auth_transient_failure kind=%s attempt=%d elapsed_ms=%d",
                    kind,
                    attempt,
                    int((time.monotonic() - started) * 1000),
                )
                if attempt < _ATTEMPTS:
                    await asyncio.sleep(random.uniform(0.2, 0.4))
    except TimeoutError:
        logger.warning(
            "serving_auth_deadline_exceeded budget_seconds=%s", AUTHORIZATION_BUDGET_SECONDS
        )
    raise TransientAuthorizationError(detail)

"""Keep serving rejection details intact across the managed checkpoint proxy."""

import logging
import re

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)
_FORWARDED_STATUSES = {400, 422, 429, 503}
_SAFE_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")


def upstream_chat_error(error: httpx.HTTPStatusError, *, checkpoint_id: str) -> HTTPException:
    response = error.response
    status = response.status_code
    if status not in _FORWARDED_STATUSES:
        return HTTPException(status_code=502, detail=f"inference failure: {error}")
    detail = f"Serving rejected the inference request (HTTP {status})."
    code = None
    try:
        body = response.json()
        message = body.get("detail")
        envelope = body.get("error")
        if isinstance(envelope, dict):
            if not isinstance(message, str):
                message = envelope.get("message")
            raw_code = envelope.get("code")
            if isinstance(raw_code, str) and _SAFE_ERROR_CODE.fullmatch(raw_code):
                code = raw_code
        if isinstance(message, str) and message.strip():
            detail = message[:4096]
    except (ValueError, AttributeError):
        pass
    headers = {}
    if status in {429, 503}:
        # Retain a bounded numeric hint without accepting arbitrary upstream headers.
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isascii() and retry_after.isdecimal() and len(retry_after) <= 5:
            headers["Retry-After"] = retry_after
        if code is not None:
            headers["X-Flash-Upstream-Code"] = code
        # Do not log request bodies, authorization material, or arbitrary error text.
        logger.warning(
            "serving_unavailable checkpoint=%s upstream_status=%d code=%s",
            checkpoint_id,
            status,
            code or "unspecified",
        )
    return HTTPException(status_code=status, detail=detail, headers=headers or None)

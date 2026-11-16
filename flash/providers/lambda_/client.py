"""Bounded Lambda Cloud REST calls; never log headers, user_data or response bodies.

Two hard-won rules: Cloudflare in front of the API 403s Python's default urllib User-Agent, so a
real one is always sent; and ``POST /instance-operations/launch`` is NOT idempotent, so it is never
retried here (a blind retry can rent and bill a second VM). Callers reconcile an ambiguous launch by
listing instances by name instead.
"""

import random
import re
import time

from flash.providers.lambda_.config import settings

URL = "https://cloud.lambda.ai/api/v1"
USER_AGENT = "flash-training/1.0 (+https://freesolo.co)"
LAUNCH_PATH = "/instance-operations/launch"
# Statuses worth another try for a request that is safe to repeat.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_CODE = re.compile(r"[A-Za-z0-9_./-]{1,120}")


class LambdaApiError(RuntimeError):
    def __init__(self, message="Lambda operation failed", *, status=0, code=""):
        super().__init__(message)
        self.status = status
        # Lambda's machine-readable error code (``instance-operations/launch/insufficient-capacity``),
        # never its free-text message, which can echo request data.
        self.code = code


def _safe_code(response) -> str:
    try:
        data = response.json()
    except ValueError:
        return ""
    error = data.get("error") if isinstance(data, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and _CODE.fullmatch(code) else ""


def _once(method, path, *, body, deadline_at):
    import httpx

    budget = min(30, deadline_at - time.time()) if deadline_at is not None else 30
    if budget <= 0:
        raise TimeoutError("Lambda operation exceeded its deadline")
    try:
        with httpx.Client(timeout=budget, follow_redirects=False) as client:
            response = client.request(
                method,
                URL + path,
                headers={
                    "Authorization": f"Bearer {settings()['api_key']}",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
                json=body,
            )
            if not 200 <= response.status_code < 300:
                raise LambdaApiError(
                    f"Lambda {method} {path} returned {response.status_code}",
                    status=response.status_code,
                    code=_safe_code(response),
                )
            data = response.json()
    except (httpx.HTTPError, ValueError):
        raise LambdaApiError("Lambda operation failed or timed out") from None
    if not isinstance(data, dict) or "data" not in data:
        raise LambdaApiError("Lambda returned an invalid response envelope")
    return data["data"]


def call(method, path, *, body=None, deadline_at=None, retries=None):
    """One API call, unwrapped from Lambda's ``{"data": ...}`` envelope.

    Reads and terminate requests retry transport failures, 429 and 5xx with jittered backoff.
    Launch never retries, whatever the caller asks for.
    """
    if retries is None:
        retries = 0 if method != "GET" and path != "/instance-operations/terminate" else 3
    if path == LAUNCH_PATH:
        retries = 0
    for attempt in range(retries + 1):
        try:
            return _once(method, path, body=body, deadline_at=deadline_at)
        except LambdaApiError as error:
            if attempt >= retries or (error.status and error.status not in _RETRY_STATUSES):
                raise
        delay = min(8.0, 1.0 * 2**attempt) * (0.5 + random.random() / 2)
        if deadline_at is not None:
            delay = min(delay, max(0.0, deadline_at - time.time()))
        time.sleep(delay)
    raise LambdaApiError("Lambda retries exhausted")  # unreachable: the last attempt raises

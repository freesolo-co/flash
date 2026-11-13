"""Bounded Compute API requests; never log response bodies or worker credentials."""

import json
import re
import time

from flash.providers.mithril.config import settings

URL = "https://api.mithril.ai/v2"


class MithrilApiError(RuntimeError):
    def __init__(self, message="Mithril operation failed", *, status=0, reason=""):
        super().__init__(message)
        self.status = status
        # Mithril's short error message, redacted against the request (see _safe_reason).
        self.reason = reason


_REASON_KEYS = ("detail", "message", "error", "title")
_REASON_LIMIT = 200
# Tokens that could be (part of) a credential: 8+ characters with a digit or symbol.
_TOKEN = re.compile(r"[\w+/=.-]{8,}")


def _safe_reason(response, *, body=None, params=None) -> str:
    """Mithril's short error message, or "" when it might echo request data.

    Only a top-level message field (or a validation list's ``msg`` entries) is read, never the raw
    body. It is dropped if it contains the API key or any non-alphabetic 8+ character token that also
    appears anywhere in the request (startup scripts carry worker credentials), then bounded.
    """
    try:
        data = response.json()
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    reason = ""
    for key in _REASON_KEYS:
        value = data.get(key)
        if isinstance(value, str):
            reason = value
            break
        if isinstance(value, list):
            reason = "; ".join(
                item["msg"]
                for item in value
                if isinstance(item, dict) and isinstance(item.get("msg"), str)
            )
            break
    reason = " ".join(reason.split())
    request = json.dumps([body, params], default=str)
    key = settings()["api_key"]
    if not reason or (key and key in reason):
        return ""
    # Checked on the whole message before bounding it, so a secret crossing the limit is caught.
    if any(not t.isalpha() and t in request for t in _TOKEN.findall(reason)):
        return ""
    return reason[:_REASON_LIMIT]


def call(method, path, *, body=None, params=None, deadline_at=None):
    import httpx

    budget = min(30, deadline_at - time.time()) if deadline_at is not None else 30
    if budget <= 0:
        raise TimeoutError("Mithril operation exceeded run deadline")
    try:
        with httpx.Client(timeout=budget, follow_redirects=False) as client:
            response = client.request(
                method,
                URL + path,
                headers={"Authorization": f"Bearer {settings()['api_key']}"},
                params=params,
                json=body,
            )
            if not 200 <= response.status_code < 300:
                # Never the raw body: Mithril can echo request data (including secrets) in error
                # bodies. Only a short message that echoes no request value is kept.
                raise MithrilApiError(
                    f"Mithril {method} {path} returned {response.status_code}",
                    status=response.status_code,
                    reason=_safe_reason(response, body=body, params=params),
                )
            return None if response.status_code == 204 else response.json()
    except (httpx.HTTPError, ValueError):
        raise MithrilApiError("Mithril operation failed or timed out") from None


def pages(path, *, params=None, deadline_at=None):
    query = {**(params or {}), "limit": 100}
    seen = set()
    for _ in range(100):
        data = call("GET", path, params=query, deadline_at=deadline_at)
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise MithrilApiError("Mithril returned an invalid page")
        if any(not isinstance(row, dict) for row in data["data"]):
            raise MithrilApiError("Mithril returned an invalid page row")
        yield from data["data"]
        cursor = data.get("next_cursor")
        if not cursor:
            return
        if not isinstance(cursor, str) or cursor in seen:
            raise MithrilApiError("Mithril returned a repeated or invalid cursor")
        seen.add(cursor)
        query["next_cursor"] = cursor
    raise MithrilApiError("Mithril pagination exceeded its bound")

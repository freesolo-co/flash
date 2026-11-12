"""Bounded Compute API requests; never log response bodies or worker credentials."""

import time

from flash.providers.mithril.config import settings

URL = "https://api.mithril.ai/v2"


class MithrilApiError(RuntimeError):
    def __init__(self, message="Mithril operation failed", *, status=0):
        super().__init__(message)
        self.status = status


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
                raise MithrilApiError(status=response.status_code)
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

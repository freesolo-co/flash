"""Organization-wide admission through an atomic, database-clock Supabase counter."""

from collections.abc import Awaitable, Callable

import httpx
from fastapi import HTTPException, status

from flash.serving.src.store.settings import Settings
from flash.serving.src.store.supabase_rest import supabase_headers, supabase_table_url

RateLimiter = Callable[[str], Awaitable[None]]


async def allow_offline_inference(_org_id: str) -> None:
    """Explicit bypass for hermetic tests and offline app construction only."""


class SupabaseRateLimiter:
    def __init__(self, settings: Settings) -> None:
        if not settings.supabase_url:
            raise RuntimeError("Missing SUPABASE_URL for serving rate limits")
        self._url = supabase_table_url(settings, "rpc/check_serving_rate_limit")
        self._client = httpx.AsyncClient(headers=supabase_headers(settings, "public"), timeout=5.0)

    async def __call__(self, org_id: str) -> None:
        # Never cache or single-flight admission, even when authorization itself is cached.
        # Do not retry ambiguous RPC failures: the database may already have counted the request.
        try:
            response = await self._client.post(self._url, json={"p_org_id": org_id})
            response.raise_for_status()
            rows = response.json()
            if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
                raise ValueError("invalid rate-limit result")
            row = rows[0]
            allowed = row.get("allowed")
            limit = row.get("rpm_limit")
            remaining = row.get("remaining")
            retry_after = row.get("retry_after")
            if (
                type(allowed) is not bool
                or type(limit) is not int
                or limit <= 0
                or type(remaining) is not int
                or not 0 <= remaining < limit
                or type(retry_after) is not int
                or not 1 <= retry_after <= 60
                or (not allowed and remaining != 0)
            ):
                raise ValueError("invalid rate-limit counters")
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "serving rate limit unavailable",
                headers={"Retry-After": "5"},
            ) from exc
        if not allowed:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                {
                    "code": "rate_limit_exceeded",
                    "message": "Organization serving rate limit exceeded",
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

    async def aclose(self) -> None:
        await self._client.aclose()

"""Atomic admission against RPM and already-recorded actual token usage."""

import httpx
from fastapi import HTTPException


def _validate(rows: object) -> dict:
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("invalid key rate-limit response")
    row = rows[0]
    for name in ("allowed", "requires_tokens"):
        if type(row.get(name)) is not bool:
            raise ValueError("invalid key rate-limit flag")
    for name in ("rpm_limit", "remaining", "retry_after"):
        if type(row.get(name)) is not int:
            raise ValueError("invalid key rate-limit counter")
    if not (
        row["rpm_limit"] > 0
        and 0 <= row["remaining"] <= row["rpm_limit"]
        and 1 <= row["retry_after"] <= 60
    ):
        raise ValueError("invalid key rate-limit range")
    limit, remaining = row.get("tpm_limit"), row.get("tokens_remaining")
    if limit is None:
        if remaining is not None or row["requires_tokens"]:
            raise ValueError("invalid org fallback")
    elif (
        type(limit) is not int
        or limit <= 0
        or type(remaining) is not int
        or not 0 <= remaining <= limit
    ):
        raise ValueError("invalid token allowance")
    if row["requires_tokens"] and row["allowed"]:
        raise ValueError("token preflight must not admit a request")
    if row["allowed"] and row["remaining"] >= row["rpm_limit"]:
        raise ValueError("admitted request was not counted")
    return row


async def admit_key(
    client: httpx.AsyncClient,
    org_url: str,
    org_id: str,
    api_key_id: str,
) -> None:
    url = org_url.rsplit("/", 1)[0] + "/check_serving_api_key_rate_limit"
    body: dict[str, str | int] = {"p_org_id": org_id, "p_api_key_id": api_key_id}

    async def check():
        try:
            response = await client.post(url, json=body)
            response.raise_for_status()
            return _validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            # Never retry an ambiguous admission: the database may have counted it.
            raise HTTPException(
                503, "serving rate limit unavailable", headers={"Retry-After": "5"}
            ) from exc

    row = await check()
    if row["requires_tokens"]:
        raise HTTPException(
            503, "Actual-usage rate limits are not deployed", headers={"Retry-After": "5"}
        )
    if not row["allowed"]:
        headers = {
            "Retry-After": str(row["retry_after"]),
            "X-RateLimit-Limit": str(row["rpm_limit"]),
            "X-RateLimit-Remaining": str(row["remaining"]),
        }
        if row["tpm_limit"] is not None:
            headers["X-RateLimit-Limit-Tokens"] = str(row["tpm_limit"])
            headers["X-RateLimit-Remaining-Tokens"] = str(row["tokens_remaining"])
        raise HTTPException(
            429,
            {
                "code": "rate_limit_exceeded",
                "message": "Serving rate limit exceeded",
            },
            headers=headers,
        )

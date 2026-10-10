from __future__ import annotations

import httpx

from flash.serving.src.store.settings import Settings


def supabase_table_url(settings: Settings, table: str) -> str:
    return f"{settings.supabase_url.rstrip('/')}/rest/v1/{table}"


def supabase_headers(settings: Settings, schema: str) -> dict[str, str]:
    if not settings.supabase_service_role_key:
        raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY")
    if not settings.supabase_service_role_key.startswith("sb_secret_"):
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY must use the sb_secret_ format")
    # postgrest routes by profile header: Accept-Profile for reads,
    # Content-Profile for writes. sending both keeps every verb on the
    # requested schema.
    return {
        "apikey": settings.supabase_service_role_key,
        "Content-Type": "application/json",
        "Accept-Profile": schema,
        "Content-Profile": schema,
    }


def postgrest_error(response: httpx.Response) -> tuple[str, str]:
    """The PostgreSQL ``SQLSTATE`` and human detail PostgREST reported, or ``("", "")``.

    PostgREST collapses several distinct SQLSTATEs onto one HTTP status: a unique violation
    (``23505``) and a foreign key violation (``23503``) are BOTH 409. The status alone therefore
    cannot tell "this row already exists" from "this row points at something that does not", which
    are opposite diagnoses -- one resolves by reading the winner, the other never resolves at all.
    The code in the body is the only field that separates them.
    """
    try:
        body = response.json()
    except ValueError:
        return "", ""
    if not isinstance(body, dict):
        return "", ""
    detail = body.get("details") or body.get("message") or ""
    return str(body.get("code") or ""), str(detail)[:300]


def raise_for_supabase(response: httpx.Response, action: str) -> None:
    if not response.is_error:
        return
    raise RuntimeError(f"Failed to {action}: {response.status_code} {response.text[:1000]}")

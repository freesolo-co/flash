"""Read-only accounting smoke-test evidence for a candidate serving deployment."""

from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from flash.serving.promotion.evidence import parse_accounting

# The canary's own outbox rows, read through idx_serving_usage_outbox_canary. Mirrors the
# `canary_settled` predicate of serving_promotion_accounting_snapshot() exactly.
_CANARY_COLUMNS = "state,completion_tokens,terminal_at,delivered_at,billing_usage_id"
_CANARY_ROW_LIMIT = 100


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("accounting timestamp missing")
    try:
        result = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("accounting timestamp malformed") from None
    if result.tzinfo is None:
        raise ValueError("accounting timestamp lacks timezone")
    return result


async def _oldest_timestamp(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    scope: dict[str, str],
    field: str,
    filters: dict[str, str],
) -> datetime | None:
    response = await client.get(
        url,
        headers=headers,
        params={**scope, **filters, "select": field, "order": f"{field}.asc", "limit": "1"},
        timeout=15,
    )
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or len(rows) > 1:
        raise ValueError("accounting rows malformed")
    if not rows:
        return None
    if not isinstance(rows[0], dict):
        raise ValueError("accounting row malformed")
    return _timestamp(rows[0].get(field))


async def _global_snapshot(
    client: httpx.AsyncClient,
    rpc_url: str,
    headers: dict[str, str],
    canary_scope: dict[str, str],
) -> dict[str, Any]:
    response = await client.post(rpc_url, headers=headers, json=canary_scope, timeout=15)
    response.raise_for_status()
    snapshot = response.json()
    if isinstance(snapshot, list) and len(snapshot) == 1:
        snapshot = snapshot[0]
    if parse_accounting(snapshot) is None:
        raise ValueError("accounting snapshot malformed")
    return snapshot


def _unavailable_reason(error: httpx.HTTPError) -> str:
    """Status and type only: the response body is database text, not gate evidence."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


def _canary_settled(rows: Any) -> bool:
    """`canary_settled` of serving_promotion_accounting_snapshot(), computed from the rows."""
    if not isinstance(rows, list) or len(rows) >= _CANARY_ROW_LIMIT:
        raise ValueError("accounting canary rows malformed")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("accounting canary row malformed")
    return bool(rows) and all(
        row.get("state") == "delivered"
        and type(row.get("completion_tokens")) is int
        and row["completion_tokens"] > 0
        and row.get("terminal_at") is not None
        and row.get("delivered_at") is not None
        and row.get("billing_usage_id") is not None
        for row in rows
    )


async def _canary_snapshot(
    client: httpx.AsyncClient,
    table_url: str,
    table_headers: dict[str, str],
    canary_scope: dict[str, str],
) -> dict[str, Any]:
    """Canary proof without the global RPC; database time comes from the response's Date."""
    response = await client.get(
        table_url,
        headers=table_headers,
        params={
            "org_id": f"eq.{canary_scope['p_org_id']}",
            "correlation_id": f"eq.{canary_scope['p_correlation_id']}",
            "serving_deployment_id": f"eq.{canary_scope['p_serving_deployment_id']}",
            "serving_release": f"eq.{canary_scope['p_serving_release']}",
            "public_model_id": f"eq.{canary_scope['p_public_model_id']}",
            "select": _CANARY_COLUMNS,
            "limit": str(_CANARY_ROW_LIMIT),
        },
        timeout=15,
    )
    response.raise_for_status()
    settled = _canary_settled(response.json())
    try:
        captured = parsedate_to_datetime(response.headers["date"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("accounting timestamp missing") from None
    if captured.tzinfo is None:
        raise ValueError("accounting timestamp lacks timezone")
    snapshot = {
        "captured_at": captured.isoformat(),
        # an empty object, not a fabricated count: the global backlog was not read.
        "states": {},
        "canary_settled": settled,
        "expired_leases": 0,
        "oldest_undelivered_age_seconds": None,
    }
    if parse_accounting(snapshot) is None:
        raise ValueError("accounting snapshot malformed")
    return snapshot


async def read_accounting(
    client: httpx.AsyncClient,
    rpc_url: str,
    headers: dict[str, str],
    canary_scope: dict[str, str],
) -> dict[str, Any]:
    """Read billing settlement and backlog evidence for the deployment smoke test.

    The existing RPC verifies the exact canary's billing evidence, but its stall counters cover
    all releases. A historical delivery quarantine needs reconciliation, not an endless rollback
    of healthy deployments. Keep the global snapshot for diagnostics and read this deployment's
    oldest undelivered row and lease separately. No usage or billing records are changed.
    """
    table_url = rpc_url.rsplit("/rpc/", 1)[0] + "/serving_usage_outbox"
    table_headers = {**headers, "Accept-Profile": "internal", "Content-Profile": "internal"}
    try:
        snapshot = await _global_snapshot(client, rpc_url, headers, canary_scope)
        global_backlog: dict[str, Any] = snapshot
    except (httpx.HTTPStatusError, httpx.TimeoutException) as error:
        # The RPC also aggregates the GLOBAL backlog, whose cost grows with every undelivered row
        # in the table; under load it exceeds the API role's statement timeout (SQLSTATE 57014,
        # HTTP 500) and used to roll back healthy releases. Its stall counters are not used here
        # (see below), so on a server error or timeout prove the canary from its own indexed rows
        # instead. 4xx responses (auth, schema drift) still fail closed.
        if isinstance(error, httpx.HTTPStatusError) and error.response.status_code < 500:
            raise
        snapshot = await _canary_snapshot(client, table_url, table_headers, canary_scope)
        global_backlog = {"unavailable": _unavailable_reason(error)}
    now = _timestamp(snapshot.get("captured_at"))
    scope = {
        "serving_deployment_id": f"eq.{canary_scope['p_serving_deployment_id']}",
        "serving_release": f"eq.{canary_scope['p_serving_release']}",
    }
    oldest = await _oldest_timestamp(
        client,
        table_url,
        table_headers,
        scope,
        "captured_at",
        {
            "state": "in.(pending,leased,quarantined,disputed)",
            "or": "(quarantine_stage.is.null,quarantine_stage.neq.generation)",
        },
    )
    lease = await _oldest_timestamp(
        client, table_url, table_headers, scope, "lease_expires_at", {"state": "eq.leased"}
    )
    return {
        **snapshot,
        "global_backlog": global_backlog,
        # The gate needs presence, not an unbounded scan/count of expired leases.
        "expired_leases": int(lease is not None and lease <= now),
        "oldest_undelivered_age_seconds": (
            max(0.0, (now - oldest).total_seconds()) if oldest is not None else None
        ),
    }

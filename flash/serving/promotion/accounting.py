"""Read promotion accounting without making historical quarantines a release veto."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from flash.serving.promotion.evidence import parse_accounting

if TYPE_CHECKING:
    import httpx


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


async def read_accounting(
    client: httpx.AsyncClient,
    rpc_url: str,
    headers: dict[str, str],
    canary_scope: dict[str, str],
) -> dict[str, Any]:
    """Require durable canary settlement and inspect stalls across this whole deployment.

    The existing RPC verifies the exact canary's billing evidence, but its stall counters cover
    all releases. A historical delivery quarantine needs reconciliation, not an endless rollback
    of healthy deployments. Keep the global snapshot for diagnostics and read this deployment's
    oldest undelivered row and lease separately. No usage or billing records are changed.
    """
    response = await client.post(rpc_url, headers=headers, json=canary_scope, timeout=15)
    response.raise_for_status()
    snapshot = response.json()
    if isinstance(snapshot, list) and len(snapshot) == 1:
        snapshot = snapshot[0]
    if parse_accounting(snapshot) is None:
        raise ValueError("accounting snapshot malformed")
    now = _timestamp(snapshot.get("captured_at"))
    scope = {
        "serving_deployment_id": f"eq.{canary_scope['p_serving_deployment_id']}",
        "serving_release": f"eq.{canary_scope['p_serving_release']}",
    }
    table_url = rpc_url.rsplit("/rpc/", 1)[0] + "/serving_usage_outbox"
    table_headers = {**headers, "Accept-Profile": "internal", "Content-Profile": "internal"}
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
        "global_backlog": snapshot,
        # The gate needs presence, not an unbounded scan/count of expired leases.
        "expired_leases": int(lease is not None and lease <= now),
        "oldest_undelivered_age_seconds": (
            max(0.0, (now - oldest).total_seconds()) if oldest is not None else None
        ),
    }

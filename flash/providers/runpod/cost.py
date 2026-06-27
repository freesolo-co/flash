"""Shape RunPod realized billing rows into a RealizedCost. Pure shaping (offline-testable);
the HTTP call is isolated in ``api.billing_endpoints``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from flash.providers.realized import RealizedCost


def _get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), UTC).isoformat()


def shape_endpoint_cost(rows: list[Any], *, endpoint_id: str | None) -> RealizedCost:
    """Sum realized USD across billing rows; ``amount`` already includes disk."""
    total = 0.0
    billed_ms = 0
    for row in rows:
        if endpoint_id is not None:
            row_eid = _get(row, "endpointId") or _get(row, "endpoint_id")
            if row_eid is not None and str(row_eid) != str(endpoint_id):
                continue
        total += float(_get(row, "amount") or 0)
        billed_ms += int(_get(row, "timeBilledMs") or 0)
    total = round(total, 6)
    return RealizedCost(
        provider="runpod",
        realized_usd=total,
        by_resource={"gpu": total},
        wall_seconds=(billed_ms / 1000.0) if billed_ms else None,
        source={"endpoint_id": endpoint_id},
    )


def realized_cost(endpoint_id: str | None, *, start: float, end: float) -> RealizedCost | None:
    """Pull + shape this run's realized RunPod cost; None when there's no endpoint to query."""
    if not endpoint_id:
        return None
    from flash.providers.runpod import api

    rows = api.billing_endpoints(
        start_time=_iso(start), end_time=_iso(end), endpoint_id=endpoint_id
    )
    return shape_endpoint_cost(rows, endpoint_id=endpoint_id)

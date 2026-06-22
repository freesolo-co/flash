"""Shape Vast realized charge entries into a RealizedCost. Pure shaping (offline-testable);
the HTTP call is isolated in ``api.get_charges``."""

from __future__ import annotations

from typing import Any

from flash.providers.realized import RealizedCost


def _get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def _matches_instance(entry: Any, instance_id: int) -> bool:
    """True if a charge entry belongs to ``instance_id`` (by ``source`` "instance-<id>" or a
    metadata/field instance id). When an entry carries no instance marker at all we keep it
    (the query was already filtered to the window), erring toward not dropping real cost."""
    source = _get(entry, "source")
    if source is not None:
        return str(source) == f"instance-{instance_id}"
    meta = _get(entry, "metadata") or {}
    meta_id = meta.get("instance_id") if isinstance(meta, dict) else None
    explicit = _get(entry, "instance_id")
    if meta_id is not None or explicit is not None:
        return int(meta_id if meta_id is not None else explicit) == instance_id
    return True


def shape_instance_cost(rows: list[Any], *, instance_id: int) -> RealizedCost:
    """Sum realized USD for ``instance_id``, itemized by resource (gpu/disk/bwd/bwu).

    Each entry has a total ``amount`` and an ``items[]`` breakdown; we sum the total and, when
    present, accumulate the per-resource items so storage/bandwidth (which a $/hr estimate
    misses) is visible. Falls back to attributing the whole total to ``gpu`` when unitemized.
    """
    total = 0.0
    by_resource: dict[str, float] = {}
    for entry in rows:
        if not _matches_instance(entry, instance_id):
            continue
        total += float(_get(entry, "amount") or 0)
        for item in _get(entry, "items") or []:
            kind = str(_get(item, "type") or "other")
            by_resource[kind] = round(
                by_resource.get(kind, 0.0) + float(_get(item, "amount") or 0), 6
            )
    total = round(total, 6)
    if not by_resource:
        by_resource = {"gpu": total}
    return RealizedCost(
        provider="vast",
        realized_usd=total,
        by_resource=by_resource,
        source={"instance_id": instance_id},
    )


def realized_cost(instance_id: int | None, *, start: float, end: float) -> RealizedCost | None:
    """Pull + shape this run's realized Vast cost; None when there's no instance to query."""
    if not instance_id:
        return None
    from flash.providers.vast import api

    rows = api.get_charges(start_ts=start, end_ts=end)
    return shape_instance_cost(rows, instance_id=int(instance_id))

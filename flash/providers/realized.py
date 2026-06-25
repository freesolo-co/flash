"""Realized provider cost (COGS) for a finished run -- the cost side of estimator accuracy.

RunPod's billing API gives the dollars it ACTUALLY charged, which the reconciliation job
compares against the run's charged pre-flight estimate. This module owns the ``RealizedCost``
shape and dispatches to the RunPod shaper by the run's persisted handle
(``RunStatus.remote['provider']``). The HTTP calls live in the provider's ``api.py``; the pure
shaping lives in its ``cost.py`` so it stays offline-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RealizedCost:
    provider: str
    realized_usd: float
    by_resource: dict[str, float] = field(
        default_factory=dict
    )  # {"gpu": .., "disk": .., "bwd": ..}
    wall_seconds: float | None = None
    source: dict = field(default_factory=dict)  # audit: resource ids / raw refs


def realized_cost_for_remote(
    remote: dict | None, *, start: float, end: float
) -> RealizedCost | None:
    """Pull realized cost for a run from its persisted provider handle, or None if unattributable.

    ``remote`` is ``RunStatus.remote`` (the last/successful attempt's handle dict). ``start``/
    ``end`` bound the provider billing query (unix seconds). Returns None when there is no handle,
    no resource id, or an unknown provider -- the run then stays unreconciled (and is retried).

    RunPod exposes a billing API, so its realized cost is the dollars it actually charged. The
    instance providers (Lambda/Hyperstack) have no per-run billing endpoint: an instance bills at a
    flat $/hr from launch to teardown, so the realized COGS is wall x rate computed from the handle's
    launch timestamp and $/hr (the same quantities ``poll`` already stamps into ``metrics.cost_usd``).
    """
    if not remote:
        return None
    provider = remote.get("provider") or "runpod"
    if provider == "runpod":
        from flash.providers.runpod.cost import realized_cost as runpod_realized

        return runpod_realized(remote.get("endpoint_id"), start=start, end=end)
    if provider in ("lambda", "hyperstack"):
        return _instance_realized_cost(remote, start=start, end=end)
    return None


def _instance_realized_cost(
    remote: dict, *, start: float, end: float
) -> RealizedCost | None:
    """Realized COGS for an instance-billed provider: wall-clock x the instance's flat $/hr.

    The instance billed from its launch (``started_ts`` on the handle) until teardown (~``end``).
    Unattributable -> None (no rate persisted) so the run stays unreconciled rather than booking $0.
    """
    rate = remote.get("hourly_usd")
    if not rate:
        return None
    launch = remote.get("started_ts") or start
    wall = max(0.0, float(end) - float(launch))
    usd = round(wall / 3600.0 * float(rate), 6)
    rid = remote.get("instance_id") or remote.get("vm_id")
    return RealizedCost(
        provider=str(remote.get("provider")),
        realized_usd=usd,
        by_resource={"gpu": usd},
        wall_seconds=wall,
        source={"resource_id": rid, "hourly_usd": float(rate), "started_ts": float(launch)},
    )

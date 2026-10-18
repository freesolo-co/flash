"""Realized provider COGS for a finished run; dispatches by ``RunStatus.remote['provider']``."""

from __future__ import annotations

import math
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


def realized_cost_for_remote(remote, *, start, end, run_end=None):
    """Reserved-resource estimate; Modal storage and network are not attributable here."""
    if not remote or remote.get("provider") != "modal":
        return None
    result = _instance_realized_cost(remote, end=run_end if run_end is not None else end)
    if result is not None:
        result.source["basis"] = "reserved_resource_estimate"
    return result


def _instance_realized_cost(remote: dict, *, end: float) -> RealizedCost | None:
    """Realized COGS for an instance-billed provider: wall-clock x flat $/hr."""
    rate = remote.get("hourly_usd")
    rid = remote.get("instance_id")
    if not rate or not rid:
        return None
    launch = remote.get("started_ts")
    if isinstance(launch, bool) or not isinstance(launch, (int, float)):
        return None
    launch = float(launch)
    if not math.isfinite(launch) or launch <= 0:
        return None
    wall = max(0.0, float(end) - launch)
    usd = round(wall / 3600.0 * float(rate), 6)
    return RealizedCost(
        provider=str(remote.get("provider")),
        realized_usd=usd,
        by_resource={"gpu": usd},
        wall_seconds=wall,
        source={"resource_id": str(rid), "hourly_usd": float(rate), "started_ts": float(launch)},
    )

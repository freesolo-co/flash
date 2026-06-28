"""Realized provider COGS for a finished run; dispatches by ``RunStatus.remote['provider']``."""

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
    remote: dict | None, *, start: float, end: float, run_end: float | None = None
) -> RealizedCost | None:
    """Return realized cost from the persisted provider handle, or None if unattributable.

    ``end`` is the settle-padded billing-query bound for RunPod; ``run_end`` is the true teardown
    time for instance providers (Lambda) — using ``end`` there would overbill by the settle padding.
    """
    if not remote:
        return None
    provider = remote.get("provider")
    if provider == "runpod":
        from flash.providers.runpod.cost import realized_cost as runpod_realized

        return runpod_realized(remote.get("endpoint_id"), start=start, end=end)
    if provider == "lambda":
        return _instance_realized_cost(remote, start=start, end=run_end if run_end is not None else end)
    return None


def _instance_realized_cost(
    remote: dict, *, start: float, end: float
) -> RealizedCost | None:
    """Realized COGS for an instance-billed provider: wall-clock x flat $/hr."""
    rate = remote.get("hourly_usd")
    rid = remote.get("instance_id")
    if not rate or not rid:
        return None
    launch = remote.get("started_ts") or start
    wall = max(0.0, float(end) - float(launch))
    usd = round(wall / 3600.0 * float(rate), 6)
    return RealizedCost(
        provider=str(remote.get("provider")),
        realized_usd=usd,
        by_resource={"gpu": usd},
        wall_seconds=wall,
        source={"resource_id": str(rid), "hourly_usd": float(rate), "started_ts": float(launch)},
    )

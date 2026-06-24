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
    """
    if not remote:
        return None
    provider = remote.get("provider") or "runpod"
    if provider == "runpod":
        from flash.providers.runpod.cost import realized_cost as runpod_realized

        return runpod_realized(remote.get("endpoint_id"), start=start, end=end)
    return None

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
    remote: dict | None, *, start: float, end: float, run_end: float | None = None
) -> RealizedCost | None:
    """Pull realized cost for a run from its persisted provider handle, or None if unattributable.

    ``remote`` is ``RunStatus.remote`` (the last/successful attempt's handle dict). Returns None
    when there is no handle, no resource id, or an unknown provider -- the run then stays
    unreconciled (and is retried).

    Two distinct time bounds, because the two cost sources are different:
      * ``start``/``end`` bound the RunPod BILLING-API query window. The caller pads ``end`` past the
        run's terminal time so the settled invoice row is in range (see reconcile ``_SETTLE_SECONDS``).
      * ``run_end`` is the run's ACTUAL terminal time (~teardown). The instance provider
        (Lambda) has no billing endpoint: an instance bills at a flat $/hr from launch to
        teardown, so its realized COGS is wall x rate over ``started_ts -> run_end`` — it must NOT
        use the settle-padded ``end`` or it would over-bill by the padding (up to an hour). Defaults
        to ``end`` for back-compat when the caller doesn't distinguish.
    """
    if not remote:
        return None
    provider = remote.get("provider") or "runpod"
    if provider == "runpod":
        from flash.providers.runpod.cost import realized_cost as runpod_realized

        return runpod_realized(remote.get("endpoint_id"), start=start, end=end)
    from flash.providers import INSTANCE_PROVIDERS

    if provider in INSTANCE_PROVIDERS:
        # Both instance-billed providers: realized COGS == wall-clock x the instance's flat $/hr
        # (Vast's $/hr is the offer's live rate stamped on the handle as ``hourly_usd``).
        return _instance_realized_cost(remote, start=start, end=run_end if run_end is not None else end)
    return None


def _instance_realized_cost(
    remote: dict, *, start: float, end: float
) -> RealizedCost | None:
    """Realized COGS for an instance-billed provider: wall-clock x the instance's flat $/hr.

    The instance billed from its launch (``started_ts`` on the handle) until teardown (``end``, the
    run's true terminal time — NOT a settle-padded billing-query bound). Unattributable -> None (no
    rate persisted) so the run stays unreconciled rather than booking $0.
    """
    rate = remote.get("hourly_usd")
    rid = remote.get("instance_id")
    # Honor the module contract: no rate OR no auditable resource id -> unattributable (None), so the
    # run stays unreconciled rather than booking instance cost we can't tie to a resource.
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

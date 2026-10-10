"""Readiness budgets for immutable adapter deployment."""

from __future__ import annotations

# how long to wait for serving to report a newly registered revision ready. the wait covers a cold
# engine: serving pulls the base model, starts the engine, then loads the adapter, and none of that
# is proportional to the adapter, which is megabytes. so the budget scales with the base model.
#
# the floor is not the old 5 minutes: a 4b deploy was observed timing out on a cold engine and then
# succeeding in ~4.6 minutes against the now-warm one, so 5 minutes did not even cover the warm case
# with margin. the floor is doubled to 10 and the per-b term covers a bigger base on top.
#
# the cap is the real constraint, and readiness is only one leg of the attempt. the same deploy also
# spends time before this wait (resolving the hub revision, downloading the adapter config to read
# its rank, the capability check, registration) and after it (600s of immutable smoke, activation,
# then 600s of alias smoke). the whole attempt must finish inside both the plane's 2400-second
# in-flight deployment stale threshold and the cli's 2400s default `--wait`, or a deploy that is
# still progressing is reaped or reported as failed.
#
# so the cap leaves real slack rather than just clearing smoke: 900 + 600 + 600 = 2100, keeping 300s
# for the surrounding hub reads, registration, activation, and poll latency, none of which share a
# wall-clock bound with this one.
#
# a longer budget costs little: an adapter serving rejects raises as soon as the revision reports
# `failed`, so only a revision that is genuinely still loading waits out the clock.
REVISION_READY_MIN_BUDGET_SECONDS = 10 * 60.0
REVISION_READY_MAX_BUDGET_SECONDS = 15 * 60.0
REVISION_READY_SECONDS_PER_PARAM_B = 20.0


def revision_ready_budget_seconds(model: str) -> float:
    """Readiness budget for a cold serving engine holding ``model``, scaled by base-model size.

    An unknown model keeps the floor: a fork can add a catalog entry, and a revision-pinned id need
    not be a catalog key, so a lookup miss must not fail a deploy that would otherwise succeed.
    """
    from flash.core.catalog import MODELS

    info = MODELS.get(str(model or "").strip())
    if info is None:
        return REVISION_READY_MIN_BUDGET_SECONDS
    # total params, not active: an moe loads every expert into vram even though a token routes
    # through few, so the cold-start cost tracks the full checkpoint.
    scaled = REVISION_READY_MIN_BUDGET_SECONDS + REVISION_READY_SECONDS_PER_PARAM_B * max(
        0.0, float(info.params_b)
    )
    return min(scaled, REVISION_READY_MAX_BUDGET_SECONDS)

"""Shared deadlines for hosted engine startup and checkpoint deployment."""

from __future__ import annotations

# The slowest measured B200 cold boot was 1821s. Compilation depends on model
# architecture, not just parameter count: the dense 27B boots slower than the 35B MoE.
ENGINE_STARTUP_TIMEOUT_SECONDS = 45 * 60
ENGINE_REQUEST_TIMEOUT_SECONDS = 10 * 60
REVISION_READY_MAX_BUDGET_SECONDS = ENGINE_STARTUP_TIMEOUT_SECONDS + ENGINE_REQUEST_TIMEOUT_SECONDS
DEPLOYMENT_SMOKE_TIMEOUT_SECONDS = 10 * 60
DEPLOYMENT_OVERHEAD_SECONDS = 5 * 60
DEPLOYMENT_TIMEOUT_SECONDS = (
    REVISION_READY_MAX_BUDGET_SECONDS
    + DEPLOYMENT_SMOKE_TIMEOUT_SECONDS
    + DEPLOYMENT_OVERHEAD_SECONDS
)


def revision_ready_budget_seconds(model: str) -> float:
    """Allow the engine's full startup and adapter-load deadlines for every model.

    All hosted catalog entries use the same bounded engine lifecycle. Unknown or
    revision-pinned model IDs need the same allowance; a size lookup must not shorten it.
    Rejected adapters still fail immediately when their persisted state becomes failed.
    """
    return float(REVISION_READY_MAX_BUDGET_SECONDS)

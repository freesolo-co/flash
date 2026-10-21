"""pure deterministic planning for customer-owned modal deployments."""

from __future__ import annotations

from ._serialization import canonical_adapter_sort_key
from .types import (
    DeploymentRequest,
    DeploymentSpec,
    validate_deployment_request,
)


class PlanningError(ValueError):
    """invalid immutable deployment input."""


def plan_deployment(request: DeploymentRequest) -> DeploymentSpec:
    """validate and return one deterministic credential-free deployment spec."""

    try:
        validate_deployment_request(request)
    except (TypeError, ValueError) as exc:
        raise PlanningError(str(exc)) from exc
    ordered = tuple(sorted(request.adapters, key=canonical_adapter_sort_key))
    try:
        return DeploymentSpec(
            deployment_id=request.deployment_id,
            generation=request.generation,
            provider=request.provider,
            placement=request.placement,
            engine=request.engine,
            adapters=ordered,
        )
    except (TypeError, ValueError) as exc:
        raise PlanningError(str(exc)) from exc

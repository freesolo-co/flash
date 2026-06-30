"""Flash training-cost estimator: a deterministic, equation-based pre-flight estimate
(``estimate_cost``) of cost = training-only GPU hours x market $/hr. No output multiplier."""

from __future__ import annotations

from .analytical import estimate_cost
from .spec import estimate_for_spec, runconfig_from_spec
from .types import CostEstimate, RunConfig

__all__ = [
    "CostEstimate",
    "RunConfig",
    "estimate_cost",
    "estimate_for_spec",
    "runconfig_from_spec",
]

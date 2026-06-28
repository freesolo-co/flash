"""Flash training-cost estimator: deterministic pre-flight estimate of wall-clock hours x $/hr."""

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

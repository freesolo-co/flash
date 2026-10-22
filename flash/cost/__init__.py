"""Flash training-cost estimator: a deterministic, equation-based pre-flight estimate
(``estimate_cost``) of cost = training-only GPU hours x market $/hr. No output multiplier."""

from __future__ import annotations

from flash.cost.analytical import estimate_cost
from flash.cost.spec import estimate_for_spec, run_config_from_spec
from flash.cost.types import CostEstimate, RunConfig

# Preserve the existing public import while using the clearer name internally.
runconfig_from_spec = run_config_from_spec

__all__ = [
    "CostEstimate",
    "RunConfig",
    "estimate_cost",
    "estimate_for_spec",
    "run_config_from_spec",
    "runconfig_from_spec",
]

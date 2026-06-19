"""Flash training-cost estimator.

A deterministic, **fully equation-based** model (``estimate_cost``): cost = wall-clock
hours x market $/hr, every term a real physical/economic quantity (FLOPs, MFU, cold-start
seconds, concurrent reward grading, the spot/queue rate the provider bills). There is **no
output multiplier** -- nothing scales the dollar figure to hit a target. Its physical
constants are calibrated against measured RunPod/Vast runs via ``calibration`` (price +
throughput calibration, not a correction on the result), and ``verify_accuracy`` grades
the raw equation against that measured cost.
"""

from __future__ import annotations

from .analytical import estimate_cost, seconds_per_step, select_gpu, setup_seconds
from .calibration import (
    environment_cost_sweep,
    fit_constants,
    verify_accuracy,
)
from .config import RunConfig
from .estimate import CostEstimate

__all__ = [
    "CostEstimate",
    "RunConfig",
    "environment_cost_sweep",
    "estimate_cost",
    "fit_constants",
    "seconds_per_step",
    "select_gpu",
    "setup_seconds",
    "verify_accuracy",
]

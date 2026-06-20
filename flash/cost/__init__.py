"""Flash training-cost estimator.

A deterministic, **fully equation-based** pre-flight estimate (``estimate_cost``): cost =
wall-clock hours x market $/hr, every term a real physical/economic quantity (FLOPs, MFU,
cold-start seconds, concurrent reward grading, the spot/queue rate the provider bills).
There is **no output multiplier** -- nothing scales the dollar figure to hit a target.
"""

from __future__ import annotations

from .analytical import estimate_cost
from .types import CostEstimate, RunConfig

__all__ = ["CostEstimate", "RunConfig", "estimate_cost"]

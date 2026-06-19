"""Flash training-cost estimator.

Two estimators over the same ``RunConfig``:

* ``estimate_cost`` -- a deterministic, first-principles analytical model built on
  Flash's own pricing / VRAM / allocator primitives. This is the ground truth.
* ``LLMCostEstimator`` -- a Claude-backed estimator whose system prompt can be loaded
  with progressively more Flash framework knowledge (``prompts.py``). The experiment
  in ``experiment.py`` measures how its accuracy converges on the analytical reference
  as the prompt is enriched.
"""

from __future__ import annotations

from .analytical import estimate_cost, seconds_per_step, select_gpu, setup_seconds
from .config import RunConfig
from .estimate import CostEstimate

__all__ = [
    "CostEstimate",
    "RunConfig",
    "estimate_cost",
    "seconds_per_step",
    "select_gpu",
    "setup_seconds",
]

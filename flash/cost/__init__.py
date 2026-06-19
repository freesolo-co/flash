"""Flash training-cost estimator.

A deterministic, first-principles **analytical model** (``estimate_cost``) prices an
SFT/GRPO run from Flash's own pricing / VRAM / allocator primitives: cost = wall-clock
hours x GPU $/hr. On its own it runs ~1.4x high versus measured cost, so the
**break-even calibration** (``breakeven_estimate``) scales it per method so the summed
quote equals what real runs actually cost -- the figure to quote. The calibration
factors are derived (and refreshed) from measured RunPod/Vast runs via ``measured``.
"""

from __future__ import annotations

from .analytical import estimate_cost, seconds_per_step, select_gpu, setup_seconds
from .calibration import (
    BREAKEVEN_FACTORS,
    breakeven_estimate,
    breakeven_factor,
    breakeven_factor_from_real_runs,
    environment_cost_sweep,
    verify_centering,
)
from .config import RunConfig
from .estimate import CostEstimate

__all__ = [
    "BREAKEVEN_FACTORS",
    "CostEstimate",
    "RunConfig",
    "breakeven_estimate",
    "breakeven_factor",
    "breakeven_factor_from_real_runs",
    "environment_cost_sweep",
    "estimate_cost",
    "seconds_per_step",
    "select_gpu",
    "setup_seconds",
    "verify_centering",
]

"""The cost-estimate result type: a fully itemized, serializable breakdown.

``CostEstimate`` is what the analytical model emits and what the LLM estimator is
graded against. It carries every term that goes into the dollar figure (chosen GPU
class + rate, the cold-start overhead, the per-step wall time, the step count, and
the resulting wall-clock hours) so the number is auditable rather than a black box.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CostEstimate:
    """A pre-flight training-cost estimate for one run.

    All seconds are wall-clock; ``total_usd`` is the headline figure
    (``wall_clock_hours * gpu_hourly_usd``).
    """

    # --- the run this estimate is for (echoed for traceability) ---
    model_id: str
    method: str  # "sft" | "grpo"
    steps: int

    # --- chosen hardware ---
    gpu: str
    provider: str
    gpu_vram_gb: int
    required_vram_gb: int
    gpu_hourly_usd: float

    # --- timing breakdown (seconds) ---
    setup_seconds: float  # cold start: boot + deps + model download (+ vLLM init for GRPO)
    seconds_per_step: float  # steady-state per optimizer step
    train_seconds: float  # steps * seconds_per_step (post wall-clock cap)
    wall_clock_seconds: float  # setup + train
    wall_capped: bool  # True if train_seconds was clamped to the run's wall-clock cap

    # --- money ---
    total_usd: float  # the headline figure (calibrated, if a factor was applied)

    # --- free-form notes (assumptions, GRPO rollout split, MoE active params, …) ---
    notes: tuple[str, ...] = ()

    # --- break-even calibration multiplier applied to ``total_usd`` ---
    # 1.0 = the raw first-principles analytical reference (what ``estimate_cost`` emits).
    # < 1.0 = a quote scaled to break even against MEASURED real-run cost (the analytical
    # model runs ~1.4x high vs billed cost); see ``calibration.breakeven_estimate``. The
    # un-calibrated figure is always recoverable as ``total_usd / calibration_factor``.
    calibration_factor: float = 1.0

    @property
    def wall_clock_hours(self) -> float:
        return self.wall_clock_seconds / 3600.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["wall_clock_hours"] = self.wall_clock_hours
        return d

    def describe(self) -> str:
        """One-line human summary."""
        cap = " (wall-capped)" if self.wall_capped else ""
        return (
            f"{self.model_id} {self.method.upper()} x{self.steps} steps -> "
            f"${self.total_usd:.2f} on {self.gpu}@{self.provider} "
            f"(${self.gpu_hourly_usd:.2f}/hr x {self.wall_clock_hours:.2f}h{cap})"
        )

    def breakdown(self) -> str:
        """Multi-line itemized breakdown, suitable for CLI output."""
        lines = [
            f"Run        : {self.model_id}  [{self.method.upper()}, {self.steps} steps]",
            f"GPU        : {self.gpu} on {self.provider} "
            f"({self.gpu_vram_gb} GB; run needs >= {self.required_vram_gb} GB) "
            f"@ ${self.gpu_hourly_usd:.2f}/hr",
            f"Setup      : {self.setup_seconds / 60:.1f} min (cold start: boot + deps + download)",
            f"Per step   : {self.seconds_per_step:.2f} s",
            f"Train      : {self.train_seconds / 60:.1f} min"
            + ("  [CAPPED at the wall-clock limit]" if self.wall_capped else ""),
            f"Wall clock : {self.wall_clock_hours:.2f} h",
        ]
        if self.calibration_factor != 1.0:
            raw = self.total_usd / self.calibration_factor
            lines.append(f"Raw model  : ${raw:.2f}  (first-principles analytical reference)")
            lines.append(
                f"Break-even : x{self.calibration_factor:.3f}  "
                "(calibrated to measured real-run cost)"
            )
        lines.append(f"TOTAL      : ${self.total_usd:.2f}")
        if self.notes:
            lines.append("Notes      :")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)

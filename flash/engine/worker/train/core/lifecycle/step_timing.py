"""RL step pace derived from Verl's native per-step duration metric."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

_RETAINED_DURATION_SAMPLES = 64


@dataclass
class StepTiming:
    """Retain recent steady-state RL durations for heartbeat projections."""

    _warmup_seen: bool = False
    _durations: list[float] = field(default_factory=list)

    def record_duration(self, duration: float | None) -> None:
        """record one positive native duration from a parsed Verl training step."""
        if duration is None or duration <= 0:
            return
        if not self._warmup_seen:
            self._warmup_seen = True
            return
        self._durations = [*self._durations, duration][-_RETAINED_DURATION_SAMPLES:]

    def heartbeat_fields(
        self,
        *,
        current_step: int,
        total_steps: int,
        remaining_wall_s: float | None,
    ) -> dict[str, float | bool]:
        """Return measured pace and a best-effort remaining-training projection."""
        durations = self._durations
        if not durations:
            return {}

        remaining_steps = max(0, int(total_steps) - int(current_step))
        try:
            median_duration_s = statistics.median(durations)
            mean_duration_s = statistics.fmean(durations)
            projected_remaining_s = mean_duration_s * remaining_steps
        except OverflowError:
            return {}
        if not all(
            math.isfinite(value)
            for value in (median_duration_s, mean_duration_s, projected_remaining_s)
        ):
            return {}
        fields: dict[str, float | bool] = {
            "step_duration_s": median_duration_s,
            "projected_remaining_s": projected_remaining_s,
        }
        if (
            remaining_wall_s is not None
            and remaining_wall_s >= 0
            and projected_remaining_s > remaining_wall_s
        ):
            fields["wall_deadline_at_risk"] = True
        return fields

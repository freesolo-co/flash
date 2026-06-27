"""Result records: ``GateReport`` (eligibility) and ``BenchResult`` (a scored run)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuleResult:
    """One gate rule's verdict."""

    name: str
    passed: bool
    detail: str = ""
    # "ok" | "fail" | "skipped" — skipped rules (e.g. a dataset probe deferred offline) do not
    # by themselves make a case ineligible, but are surfaced so the gap is visible.
    status: str = "ok"


@dataclass
class GateReport:
    """The outcome of running a ``PaperCase`` through the eligibility gate."""

    case_id: str
    eligible: bool
    flash_model: str
    rules: list[RuleResult] = field(default_factory=list)
    # Recorded when the gate substitutes a Flash catalog model for the paper's base model.
    model_substituted: bool = False
    estimated_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        head = f"{'ELIGIBLE' if self.eligible else 'INELIGIBLE'}  {self.case_id}  -> {self.flash_model}"
        lines = [head]
        for r in self.rules:
            mark = {"ok": "  ok ", "fail": " FAIL", "skipped": " skip"}.get(r.status, "  ? ")
            lines.append(f"  [{mark}] {r.name}: {r.detail}")
        if self.estimated_usd is not None:
            lines.append(f"  preflight cost estimate: ${self.estimated_usd:.2f}")
        return "\n".join(lines)


@dataclass
class BenchResult:
    """A scored replication attempt for one ``PaperCase``."""

    case_id: str
    state: str  # "scored" | "errored" | "invalid" | "ineligible" | "dry_run"
    flash_model: str
    run_id: str | None = None
    env_id: str | None = None
    # Metric numbers (all on the paper's eval split, with the paper's metric).
    paper_metric: float | None = None
    base_metric: float | None = None
    agent_metric: float | None = None
    # Improvement-normalized achievement in [0, 1]; None when undefined (e.g. paper==base).
    achievement: float | None = None
    ratio: float | None = None
    noise_band: float | None = None
    cost_usd: float | None = None
    gate: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    def summary(self) -> str:
        lines = [f"{self.state.upper()}  {self.case_id}  ({self.flash_model})"]
        if self.run_id:
            lines.append(f"  run: {self.run_id}")
        if self.agent_metric is not None:
            lines.append(
                f"  base={_fmt(self.base_metric)}  agent={_fmt(self.agent_metric)}  "
                f"paper={_fmt(self.paper_metric)}"
            )
        if self.achievement is not None:
            band = f" (noise +/-{self.noise_band:.3f})" if self.noise_band is not None else ""
            lines.append(f"  achievement: {self.achievement:.3f}{band}")
        if self.cost_usd is not None:
            lines.append(f"  cost: ${self.cost_usd:.2f}")
        for key, val in self.diagnostics.items():
            lines.append(f"  ! {key}: {val}")
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"

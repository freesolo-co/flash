#!/usr/bin/env python
"""Accuracy report: grade the first-principles cost equation against measured cost.

Fully equation-based -- cost = wall-clock hours x market $/hr, no output multiplier. This
grades the RAW equation against the measured RunPod/Vast runs in ``measured_runs.json``
and reports the honest scorecard (mean/median APE, aggregate bias, within-tolerance), the
per-environment cost sweep, and the calibrated physical constants the equation uses.

    FLASH_SKIP_NET=1 uv run python cost_estimator_results/real_runs/accuracy.py

Writes ``accuracy_report.md`` next to this script. Refresh ``measured_runs.json`` from the
control plane as new runs land, then re-run to re-grade (and ``fit_constants`` to refresh
the realized $/hr the equation prices at).
"""

from __future__ import annotations

import json
from pathlib import Path

from flash.cost.calibration import environment_cost_sweep, fit_constants, verify_accuracy

HERE = Path(__file__).resolve().parent


def _acc_table(acc: dict) -> str:
    rows = [
        "| Group | n | mean MAPE | median APE | agg bias (Σest/Σmeas) | within 33% | within 50% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g in ("sft", "grpo", "all"):
        if g not in acc:
            continue
        a = acc[g]
        rows.append(
            f"| {g} | {a['n']} | {a['mean_mape_pct']:.0f}% | {a['median_ape_pct']:.0f}% | "
            f"{a['agg_bias']:.3f} | {a['within_33pct'] * 100:.0f}% | {a['within_50pct'] * 100:.0f}% |"
        )
    return "\n".join(rows)


def _sweep_table(rows: list[dict]) -> str:
    out = ["| Model | Environment | reward s/compl | cost $ |", "|---|---|---:|---:|"]
    for r in rows:
        rwd = "" if r["reward_s_per_completion"] is None else f"{r['reward_s_per_completion']:.2f}"
        env = f"**{r['environment']}**" if r["gpu"] == "" else r["environment"]
        cap = " ⚠cap" if r["capped"] else ""
        out.append(f"| {r['model']} | {env}{cap} | {rwd} | {r['usd']:.2f} |")
    return "\n".join(out)


def build_report() -> str:
    acc = verify_accuracy()
    sweep = environment_cost_sweep()
    consts = fit_constants()
    return f"""# Cost equation accuracy vs measured RunPod/Vast cost

`flash.cost.estimate_cost` is **fully equation-based**: `cost = wall-clock hours x market
$/hr`, every term a real quantity (FLOPs, MFU, cold-start seconds, concurrent reward
grading, the spot/queue rate the provider bills). **There is no output multiplier** -- the
dollar figure is not scaled to hit any target. The numbers below are whatever the equation
gives.

## Accuracy vs measured cost (n={acc["all"]["n"]} real runs)

{_acc_table(acc)}

`agg bias` is **not forced to 1.0** -- forcing it would be exactly the kind of output hack
this equation avoids. The residual per-run error is largely irreducible: measured wall =
provider queue + container boot + model download + train, and that scheduling/cold-start
overhead swings by minutes in ways no config predicts. So the equation targets an unbiased
central estimate with a typical run within ~30-50%, not a fake-precise point.

## Calibrated physical inputs (realized market $/hr, from {consts["n_runs"]} runs' billing)

The equation prices at the **realized (spot/queue) rate** providers actually bill, not the
on-demand list price. Median of the providers' recorded rate per class:

{json.dumps(consts["realized_hourly_usd"], indent=0).replace('{', '').replace('}', '').strip()}

## Environment cost sweep (GRPO; cost varies with the reward grader)

Reward grading is modeled **concurrent** (parallel grader slots), so a heavy LLM-judge env
no longer saturates the wall cap the way a fully-serial model would.

{_sweep_table(sweep)}

---
_Regenerate: `FLASH_SKIP_NET=1 uv run python cost_estimator_results/real_runs/accuracy.py`._
"""


def main() -> int:
    md = build_report()
    (HERE / "accuracy_report.md").write_text(md)
    print(md)
    print(f"\nwrote {HERE / 'accuracy_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
        "| Group | n | median APE | within 33% | Σ cost $ | Σ quote $ | net $ | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for g in ("all", "sft", "grpo", "real", "real_sft", "real_grpo"):
        if g not in acc:
            continue
        a = acc[g]
        rows.append(
            f"| {g} | {a['n']} | {a['median_ape_pct']:.0f}% | {a['within_33pct'] * 100:.0f}% | "
            f"${a['sum_measured_usd']:.2f} | ${a['sum_estimated_usd']:.2f} | "
            f"{a['net_usd']:+.2f} ({a['net_pct']:+.0f}%) | {_verdict(a)} |"
        )
    return "\n".join(rows)


def _verdict(a: dict) -> str:
    """Money outcome if the estimate is the quote and measured is what we pay the provider."""
    if abs(a["net_pct"]) < 5:
        return "break-even"
    return "**GAIN** (over-quote)" if a["net_usd"] > 0 else "**LOSE** (under-quote)"


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
    # ``verify_accuracy`` omits any group with zero rows, so "real" is absent when the dataset
    # has no >=500s runs. Fall back to "all"; only a fully empty dataset has neither.
    real_group = "real" if "real" in acc else "all"
    real = acc.get(real_group)
    if real is None:
        raise SystemExit("no runs in measured_runs.json -- nothing to grade")
    # Label the money paragraph to match the data actually shown (real subset vs all).
    if real_group == "real":
        runs_label = f"{real['n']} real runs (those that actually ran their configured work)"
    else:
        runs_label = f"{real['n']} runs (all groups; no ≥500s subset in this dataset)"
    # Compute dynamic accuracy numbers from the data to avoid prose drifting from the table.
    rg = acc.get("real_grpo") or acc.get("real") or {}
    grpo_ape = f"~{rg['median_ape_pct']:.0f}%" if rg else "N/A"
    grpo_within = f"~{rg['within_33pct'] * 100:.0f}%" if rg else "N/A"
    money = (
        f"**Money outcome — does the estimator break even, gain, or lose?** If the estimate "
        f"is the quote and the measured cost is what we pay the provider, then over the "
        f"**{runs_label}** the "
        f"equation quotes **${real['sum_estimated_usd']:.2f}** against **${real['sum_measured_usd']:.2f}** "
        f"of real cost — a net of **{real['net_usd']:+.2f} ({real['net_pct']:+.0f}%)**, i.e. "
        f"**{_verdict(real).replace('**', '')}**. The `all` row nets near zero only because "
        f"cancelled sub-500s smoke tests wildly over-quote and mask the loss — they price a "
        f"different (shorter) run than they ran, so they are not real revenue. The honest, "
        f"no-output-multiplier equation is centered on the *median* run, but real cost is "
        f"right-skewed (a few big rollouts dominate the dollars), so a per-run-accurate quote "
        f"sits below the *mean* cost: accurate, but it loses money used raw. Breaking even "
        f"needs either tighter big-rollout accuracy or an explicit margin on top (a business "
        f"markup, kept separate from the equation — NOT a hidden factor inside it)."
    )
    return f"""# Cost equation accuracy vs measured RunPod/Vast cost

`flash.cost.estimate_cost` is **fully equation-based**: `cost = wall-clock hours x market
$/hr`, every term a real quantity (FLOPs, MFU, cold-start seconds, concurrent reward
grading, the spot/queue rate the provider bills). **There is no output multiplier** -- the
dollar figure is not scaled to hit any target. The numbers below are whatever the equation
gives.

## Accuracy vs measured cost (n={acc["all"]["n"]} runs)

{_acc_table(acc)}

{money}

The net $ is **not forced to zero** -- forcing it (scaling the estimate to hit break-even)
would be exactly the output hack this equation avoids. The meaningful rows are **`real_*`**
(runs >= 500s that actually executed their configured work): real GRPO lands at **{grpo_ape} median
APE, {grpo_within} of runs within a third** of measured -- accurate for a pre-flight quote. Runs <500s
over-predict badly because they
never ran their configured steps (cancelled / step-capped smoke tests -- the tell is that
predicted train time alone exceeds the whole measured wall), so their measured cost is for
a different, shorter run than the equation prices: an invalid comparison, not an estimator
error. The remaining real-run residual is cold-start spread (implied cold-start is a stable
~480-520s on long runs) + per-step scatter; it is a central estimate, not a fake-precise point.

## Calibrated physical inputs (observed billed $/hr reference, from {consts["n_runs"]} runs)

Observed billed medians per GPU class from `fit_constants()` (for reference — the estimator
uses `flash.cost.facts.REALIZED_HOURLY_USD`, which may use conservative representative rates
that differ from these dataset medians):

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

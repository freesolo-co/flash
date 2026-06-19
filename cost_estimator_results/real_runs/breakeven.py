#!/usr/bin/env python
"""Break-even calibration report: center the equation on measured cost + sweep envs.

Two artifacts, both offline (analytical model + the committed ``summary_real.json``):

  1. *Centering* -- apply the per-method break-even factors to every measured run's
     analytical quote and show that Σ calibrated quote == Σ measured cost (ratio ~1.0),
     i.e. across the workload the platform breaks even.
  2. *Environment sweep* -- price a GRPO run across the reward-latency tiers (a free
     exact-match grader through an LLM-judge) and report the average cost.

Writes ``breakeven_report.md`` + ``breakeven.json`` next to this script.

    FLASH_SKIP_NET=1 uv run python cost_estimator_results/real_runs/breakeven.py

The factors come from ``flash.cost.calibration`` (regenerable via
``breakeven_factor_from_real_runs``); refresh them there as new runs are measured.
"""

from __future__ import annotations

import json
from pathlib import Path

from flash.cost.calibration import (
    BREAKEVEN_FACTORS,
    breakeven_factor_from_real_runs,
    environment_cost_sweep,
    verify_centering,
)

HERE = Path(__file__).resolve().parent


def _centering_table(cen: dict) -> str:
    rows = [
        "| Group | n | Σ measured $ | Σ break-even quote $ | ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in ("sft", "grpo", "all"):
        c = cen[g]
        rows.append(
            f"| {g} | {int(c['n'])} | {c['sum_measured']:.4f} | "
            f"{c['sum_calibrated']:.4f} | {c['ratio']:.3f} |"
        )
    return "\n".join(rows)


def _sweep_table(rows: list[dict]) -> str:
    out = [
        "| Model | Environment | reward s/compl | raw $ | break-even $ |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        rwd = "" if r["reward_s_per_completion"] is None else f"{r['reward_s_per_completion']:.2f}"
        raw = "" if r["raw_usd"] is None else f"{r['raw_usd']:.2f}"
        cap = " ⚠cap" if r.get("capped") else ""
        env = f"**{r['environment']}**" if r["raw_usd"] is None else r["environment"]
        out.append(f"| {r['model']} | {env}{cap} | {rwd} | {raw} | {r['breakeven_usd']:.2f} |")
    return "\n".join(out)


def build_report() -> tuple[str, dict]:
    recomputed = breakeven_factor_from_real_runs()
    cen = verify_centering()
    sweep = environment_cost_sweep()
    grand = sweep[-1]["breakeven_usd"]

    md = f"""# Break-even calibration: center the equation on measured cost

The first-principles analytical model (`flash.cost.estimate_cost`) is the experiment's
*reference*, but graded against MEASURED control-plane cost it runs **~1.42x high**
(SFT 1.29x, GRPO 1.90x). For pricing we want quotes that, summed over a workload,
equal what the runs actually cost -- the platform **breaks even**. The unbiased
estimator for that is the ratio of totals, applied per method:

> break-even factor = Σ measured cost / Σ analytical (static) cost

Published constants (`flash.cost.calibration.BREAKEVEN_FACTORS`), regenerated from the
committed `summary_real.json`:

| Method | factor (published) | factor (recomputed) |
|---|---:|---:|
| SFT | {BREAKEVEN_FACTORS["sft"]:.3f} | {recomputed["sft"]:.4f} |
| GRPO | {BREAKEVEN_FACTORS["grpo"]:.3f} | {recomputed["grpo"]:.4f} |
| global | 0.703 | {recomputed["global"]:.4f} |

## 1. Centering — does the calibrated equation break even?

Apply the per-method factor to every measured run's analytical quote; the calibrated
total should equal the measured total (ratio 1.0).

{_centering_table(cen)}

**Σ calibrated quote = Σ measured cost (ratio ≈ 1.00): the equation now sits right
down the middle.** Per-run spread remains (this centers the *total*, not each run);
tightening the per-run error is the job of the environment-testing loop below.

## 2. Environment cost sweep — average GRPO cost across reward tiers

Environment enters cost only through the GRPO reward grader (`rewards.py`): a free
exact-match check vs. a seconds-per-completion LLM-judge. Same run otherwise.

{_sweep_table(sweep)}

**Grand average GRPO break-even cost ≈ ${grand:.2f}.** Rows marked ⚠cap saturate the
24h wall cap because reward latency is modeled as **fully serial** across all 512
completions (3.0 s/compl x 512 ~ 25 min/step). Real verifiers envs grade concurrently,
so this over-estimates heavy-reward GRPO — the **#1 fix** for the env-testing loop
(add a reward-concurrency factor, calibrated against real heavy-env runs).

---
_Regenerate: `FLASH_SKIP_NET=1 uv run python cost_estimator_results/real_runs/breakeven.py`._
"""
    data = {
        "published_factors": BREAKEVEN_FACTORS,
        "recomputed_factors": recomputed,
        "centering": cen,
        "environment_sweep": sweep,
        "grand_average_breakeven_usd": grand,
    }
    return md, data


def main() -> int:
    md, data = build_report()
    (HERE / "breakeven_report.md").write_text(md)
    (HERE / "breakeven.json").write_text(json.dumps(data, indent=2))
    print(md)
    print(f"\nwrote {HERE / 'breakeven_report.md'} and {HERE / 'breakeven.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

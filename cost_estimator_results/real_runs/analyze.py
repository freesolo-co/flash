#!/usr/bin/env python
"""Grade the cost estimators against MEASURED cost from real training runs.

Reads the launched-run manifest, fetches each run's measured cost/wall/rate/spec from
the control plane, then:
  1. analytical estimate vs measured cost (raw + rate-adjusted to the actual $/hr),
  2. the LLM estimator across prompt versions 1..6 vs measured cost (MAPE),
and writes report_real.md, results_real.json, mape_real.png next to this script.

    ANTHROPIC_API_KEY=... uv run python cost_estimator_results/real_runs/analyze.py
    # offline (analytical + measured only, skip the LLM sweep):
    uv run python cost_estimator_results/real_runs/analyze.py --no-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flash.client.http import client_from_config
from flash.cost import estimate_cost
from flash.cost.experiment import ExperimentResult, live_estimator_factory, run_experiment
from flash.cost.measured import MeasuredRun, measured_from_status, measured_ground_truth_fn
from flash.cost.report import ascii_chart, save_json, save_png

HERE = Path(__file__).resolve().parent


def load_measured(manifest: Path) -> list[MeasuredRun]:
    client = client_from_config()
    runs: list[MeasuredRun] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        # Manifest is `<run_id>\t<label>`. Split once so a label may contain tabs; skip (don't
        # crash the whole script on) a line missing its tab.
        parts = line.split("\t", 1)
        if len(parts) != 2:
            print(f"  ! skipping malformed manifest line (need '<run_id>\\t<label>'): {line!r}")
            continue
        run_id, label = parts[0].strip(), parts[1].strip()
        try:
            status = client.get_run(run_id)
        except Exception as exc:
            print(f"  ! {label}: fetch failed: {exc}")
            continue
        m = measured_from_status(status, label=label)
        runs.append(m)
        print(
            f"  {label:<30} {m.state:<10} ${m.cost_usd:<8.4f} {m.gpu}@{m.provider} {m.wall_seconds:.0f}s"
        )
    return runs


def analytical_table(runs: list[MeasuredRun]) -> str:
    rows = [
        "| Run | Measured $ | Analytical $ (static) | Analytical $ (actual rate) | Meas wall | Est wall | GPU meas/est |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for m in runs:
        e = estimate_cost(m.config)
        rate_adj = e.wall_clock_hours * (m.hourly_usd or e.gpu_hourly_usd)
        rows.append(
            f"| {m.config.label} | {m.cost_usd:.4f} | {e.total_usd:.4f} | {rate_adj:.4f} | "
            f"{m.wall_seconds:.0f}s | {e.wall_clock_seconds:.0f}s | {m.gpu}/{e.gpu} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "launched.tsv"))
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM sweep (analytical only)")
    ap.add_argument("--effort", default="low")
    args = ap.parse_args(argv)

    print("Fetching measured runs...")
    runs = load_measured(Path(args.manifest))
    done = [m for m in runs if m.ok]
    print(f"\n{len(done)}/{len(runs)} runs done with measured cost.\n")
    if not done:
        print("No completed runs yet -- re-run once they finish.")
        return 1

    out = [
        "# Cost estimator vs MEASURED real-run cost",
        "",
        f"{len(done)} real training runs on the `freesolo-co/flash-bench` reward env "
        "(SFT + GRPO, RTX 5090 / A100 PCIe, varied group/steps/model). Ground truth is "
        "the actual control-plane `cost_usd`, not the analytical equation.",
        "",
        "## Analytical equation vs measured",
        "",
        "`static` uses the registry's fallback $/hr; `actual rate` re-prices the same "
        "estimated wall-clock at the rate the run was actually billed (isolates the "
        "wall-clock model from the spot-vs-static pricing gap).",
        "",
        analytical_table(done),
        "",
    ]

    # MAPE of analytical vs measured (raw + rate-adjusted + calibrated).
    def mape(values: list[float]) -> float:
        return sum(values) / len(values)

    def median(values: list[float]) -> float:
        s = sorted(values)
        return s[len(s) // 2]

    raw_apes, adj_apes = [], []
    for m in done:
        e = estimate_cost(m.config)
        raw_apes.append(abs(e.total_usd - m.cost_usd) / m.cost_usd * 100)
        adj = e.wall_clock_hours * (m.hourly_usd or e.gpu_hourly_usd)
        adj_apes.append(abs(adj - m.cost_usd) / m.cost_usd * 100)

    # Calibration: fit the cold-start constants to the (warm-image) real runs and
    # reprice at the actual rate. Shows how much of the residual is uncalibrated
    # cold-start vs irreducible spot-pricing noise. Overrides module globals that
    # estimate_cost reads at call time, then restores them.
    import flash.cost.analytical as A

    saved = (A.WORKER_BOOT_S, A.DEPS_INSTALL_S, A.VLLM_INIT_S, A.DOWNLOAD_RATE_GBPS)
    A.WORKER_BOOT_S, A.DEPS_INSTALL_S, A.VLLM_INIT_S, A.DOWNLOAD_RATE_GBPS = 120.0, 45.0, 45.0, 2.5
    cal_apes = []
    try:
        for m in done:
            e = estimate_cost(m.config)
            cal = e.wall_clock_hours * (m.hourly_usd or e.gpu_hourly_usd)
            cal_apes.append(abs(cal - m.cost_usd) / m.cost_usd * 100)
    finally:
        # Restore even if estimate_cost raises mid-loop, or later estimate_cost calls in this
        # process keep using the temporary cold-start constants.
        A.WORKER_BOOT_S, A.DEPS_INSTALL_S, A.VLLM_INIT_S, A.DOWNLOAD_RATE_GBPS = saved

    out.append(
        f"Analytical MAPE vs measured: **{mape(raw_apes):.0f}%** (static rate), "
        f"**{mape(adj_apes):.0f}%** (actual rate), "
        f"**{mape(cal_apes):.0f}%** (actual rate + cold-start calibrated to these runs). "
        f"Median APE: {median(raw_apes):.0f}% / {median(adj_apes):.0f}% / {median(cal_apes):.0f}%.\n"
    )

    result: ExperimentResult | None = None
    if not args.no_llm:
        print("Running LLM sweep (versions 1-6) vs MEASURED cost...")
        gt = measured_ground_truth_fn(done)
        result = run_experiment(
            grid=[m.config for m in done],
            estimator_factory=live_estimator_factory(effort=args.effort),
            ground_truth_fn=gt,
            max_workers=8,
            progress=lambda line: print(line, flush=True),
        )
        out += [
            "## LLM estimator vs measured cost (by prompt version)",
            "",
            "Median APE is the robust headline (mean MAPE is inflated by v4, whose "
            "SFT-timing-without-rollout estimate explodes on tiny-cost GRPO runs).",
            "",
            "| Version | MAPE | Median APE | SFT | GRPO | GPU-pick acc |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for s in result.summaries:
            ga = f"{s.gpu_accuracy * 100:.0f}%" if s.gpu_accuracy is not None else "n/a"
            sft = f"{s.mape_sft:.0f}%" if s.mape_sft is not None else "n/a"
            grpo = f"{s.mape_grpo:.0f}%" if s.mape_grpo is not None else "n/a"
            med = f"{s.median_ape:.0f}%" if s.median_ape is not None else "n/a"
            out.append(f"| {s.label} | {s.mape:.0f}% | {med} | {sft} | {grpo} | {ga} |")
        out += ["", "```", ascii_chart(result), "```", ""]

    (HERE / "report_real.md").write_text("\n".join(out))
    payload = {
        "measured": [
            {
                "label": m.config.label,
                "cost_usd": m.cost_usd,
                "wall_seconds": m.wall_seconds,
                "gpu": m.gpu,
                "provider": m.provider,
                "hourly_usd": m.hourly_usd,
                "analytical_static_usd": estimate_cost(m.config).total_usd,
            }
            for m in done
        ],
        "analytical_mape_static": mape(raw_apes),
        "analytical_mape_actual_rate": mape(adj_apes),
        # The report headlines the cold-start-calibrated MAPE (actual rate + fitted
        # cold-start constants) and the median APEs alongside the mean MAPEs; mirror
        # them here so the machine-readable summary carries the same figures.
        "analytical_mape_calibrated": mape(cal_apes),
        "analytical_median_ape_static": median(raw_apes),
        "analytical_median_ape_actual_rate": median(adj_apes),
        "analytical_median_ape_calibrated": median(cal_apes),
    }
    if result is not None:
        save_json(result, HERE / "results_real.json")
        save_png(result, HERE / "mape_real.png")
        payload["llm_summaries"] = [s.to_dict() for s in result.summaries]
    (HERE / "summary_real.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {HERE / 'report_real.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

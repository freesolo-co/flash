"""CLI for the Flash training-cost estimator.

    python -m flash.cost estimate --model Qwen/Qwen3.5-4B --method grpo --steps 150
    python -m flash.cost verify

``estimate`` prints the first-principles cost breakdown for one run (wall-clock hours x
market $/hr; no output adjustment). ``verify`` grades the raw equation against measured
RunPod/Vast cost and sweeps the per-environment cost.
"""

from __future__ import annotations

import argparse
import sys

from .analytical import estimate_cost
from .calibration import environment_cost_sweep, verify_accuracy
from .types import RunConfig


def _cmd_estimate(args: argparse.Namespace) -> int:
    cfg = RunConfig(
        args.model,
        args.method,
        args.steps,
        seq_len=args.seq_len,
        thinking=args.thinking,
        gpu=args.gpu,
        environment=args.environment,
    )
    print(estimate_cost(cfg).breakdown())
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    acc = verify_accuracy()
    print("Equation vs MEASURED cost  (raw first-principles, no output factor):")
    print(
        f"  {'group':10s} {'n':>3s} {'medAPE':>7s} {'<=33%':>6s} "
        f"{'Σcost$':>8s} {'Σquote$':>8s} {'net$':>9s}  verdict"
    )
    for g in ("all", "sft", "grpo", "real", "real_sft", "real_grpo"):
        if g not in acc:
            continue
        a = acc[g]
        verdict = (
            "break-even"
            if abs(a["net_pct"]) < 5
            else ("GAIN (over-quote)" if a["net_usd"] > 0 else "LOSE (under-quote)")
        )
        print(
            f"  {g:10s} {a['n']:3d} {a['median_ape_pct']:6.0f}% {a['within_33pct'] * 100:5.0f}% "
            f"{a['sum_measured_usd']:8.2f} {a['sum_estimated_usd']:8.2f} "
            f"{a['net_usd']:+8.2f}({a['net_pct']:+.0f}%)  {verdict}"
        )
    print("  real_* = runs >=500s (ran their work) = the meaningful pricing accuracy.")
    print("  net$ = Σquote - Σcost: +gain (over-quote), -loss (under-quote). NOT forced to 0.")
    print("\nEnvironment cost sweep  (GRPO; cost varies with reward grader):")
    print(f"  {'model':18s} {'environment':30s} {'rwd_s':>6s} {'usd$':>8s}")
    for r in environment_cost_sweep(steps=args.steps):
        rwd = "" if r["reward_s_per_completion"] is None else f"{r['reward_s_per_completion']:.2f}"
        cap = " cap" if r["capped"] else ""
        print(f"  {r['model']:18s} {r['environment']:30s} {rwd:>6s} {r['usd']:8.2f}{cap}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flash.cost", description="Flash training-cost estimator")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="first-principles cost breakdown for one run")
    e.add_argument("--model", required=True)
    e.add_argument("--method", choices=("sft", "grpo"), required=True)
    e.add_argument("--steps", type=int, required=True)
    e.add_argument("--seq-len", type=int, default=None, dest="seq_len")
    e.add_argument("--gpu", default=None, help="pin a GPU class (else cheapest fit)")
    e.add_argument("--environment", default=None, help="verifiers env slug (GRPO reward tier)")
    e.add_argument("--thinking", action="store_true")
    e.set_defaults(func=_cmd_estimate)

    v = sub.add_parser("verify", help="grade the equation vs measured cost + env sweep")
    v.add_argument("--steps", type=int, default=100, help="steps for the environment sweep")
    v.set_defaults(func=_cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

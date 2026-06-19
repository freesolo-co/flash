"""CLI for the Flash training-cost estimator.

    python -m flash.cost estimate --model Qwen/Qwen3.5-4B --method grpo --steps 150
    python -m flash.cost estimate --model Qwen/Qwen3.5-4B --method grpo --steps 150 --raw
    python -m flash.cost breakeven

``estimate`` prints the cost breakdown for one run -- the break-even quote (calibrated
to measured real-run cost) by default, or the raw first-principles analytical reference
with ``--raw``. ``breakeven`` proves the calibration centers on measured cost and sweeps
the per-environment cost.
"""

from __future__ import annotations

import argparse
import sys

from .analytical import estimate_cost
from .calibration import breakeven_estimate, environment_cost_sweep, verify_centering
from .config import RunConfig


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
    # Default to the break-even quote (calibrated to measured cost); --raw shows the
    # un-calibrated first-principles analytical reference.
    est = estimate_cost(cfg) if args.raw else breakeven_estimate(cfg)
    print(est.breakdown())
    return 0


def _cmd_breakeven(args: argparse.Namespace) -> int:
    cen = verify_centering()
    print("Break-even centering  (sum calibrated quote vs sum measured cost over real runs):")
    print(f"  {'group':5s} {'n':>3s} {'sum meas':>11s} {'sum quote':>10s} {'ratio':>7s}")
    for g in ("sft", "grpo", "all"):
        c = cen[g]
        print(
            f"  {g:5s} {int(c['n']):3d} {c['sum_measured']:11.4f} "
            f"{c['sum_calibrated']:10.4f} {c['ratio']:7.3f}"
        )
    print("\nEnvironment cost sweep  (GRPO; break-even quote varies with reward grader):")
    print(f"  {'model':18s} {'environment':30s} {'rwd_s':>6s} {'raw$':>8s} {'quote$':>8s}")
    for r in environment_cost_sweep(steps=args.steps):
        rwd = "" if r["reward_s_per_completion"] is None else f"{r['reward_s_per_completion']:.2f}"
        raw = "" if r["raw_usd"] is None else f"{r['raw_usd']:.2f}"
        print(
            f"  {r['model']:18s} {r['environment']:30s} {rwd:>6s} {raw:>8s} "
            f"{r['breakeven_usd']:8.2f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flash.cost", description="Flash training-cost estimator")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="cost breakdown for one run (break-even quote)")
    e.add_argument("--model", required=True)
    e.add_argument("--method", choices=("sft", "grpo"), required=True)
    e.add_argument("--steps", type=int, required=True)
    e.add_argument("--seq-len", type=int, default=None, dest="seq_len")
    e.add_argument("--gpu", default=None, help="pin a GPU class (else cheapest fit)")
    e.add_argument("--environment", default=None, help="verifiers env slug (GRPO reward tier)")
    e.add_argument("--thinking", action="store_true")
    e.add_argument(
        "--raw",
        action="store_true",
        help="show the un-calibrated first-principles analytical reference",
    )
    e.set_defaults(func=_cmd_estimate)

    b = sub.add_parser(
        "breakeven", help="prove break-even centering + sweep cost across environments"
    )
    b.add_argument("--steps", type=int, default=300, help="steps for the environment sweep")
    b.set_defaults(func=_cmd_breakeven)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

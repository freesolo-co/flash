"""CLI for the Flash cost estimator.

    python -m flash.cost estimate --model Qwen/Qwen3.5-4B --method grpo --steps 150
    python -m flash.cost experiment --effort low --out runs/cost
    python -m flash.cost experiment --offline --out runs/cost   # no API, demo/CI

``estimate`` prints the analytical breakdown for one run. ``experiment`` runs the
prompt-convergence sweep and writes ``report.md`` + ``mape.png`` + ``results.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analytical import estimate_cost
from .config import RunConfig
from .experiment import (
    decaying_stub_factory,
    default_grid,
    diverse_grid,
    live_estimator_factory,
    run_experiment,
)
from .prompts import NUM_VERSIONS
from .report import ascii_chart, markdown_report, save_json, save_png


def _cmd_estimate(args: argparse.Namespace) -> int:
    cfg = RunConfig(
        args.model,
        args.method,
        args.steps,
        seq_len=args.seq_len,
        thinking=args.thinking,
        gpu=args.gpu,
    )
    print(estimate_cost(cfg).breakdown())
    return 0


def _parse_versions(spec: str) -> list[int]:
    """Parse a ``--versions`` spec into a list of ints.

    Supports comma lists, ``a-b`` ranges, and any mix of the two: ``"1,3-6"`` ->
    ``[1, 3, 4, 5, 6]``. Each comma-separated token is expanded independently, so a
    range inside a list no longer swallows the whole string. Overlapping tokens are
    de-duplicated and the result is sorted (``"1,1-3"`` -> ``[1, 2, 3]``), so a repeated
    version doesn't grade the same prompt twice and skew the experiment.

    Raises ``ValueError`` for malformed input -- a non-integer token (``"x"``), a
    reversed range (``"6-1"``), or a spec that expands to nothing (``","``) -- so the
    caller can surface a clean CLI error instead of a later traceback.
    """
    seen: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token:
                lo_s, hi_s = token.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if lo > hi:
                    raise ValueError(f"reversed range {token!r} (low > high)")
                seen.update(range(lo, hi + 1))
            else:
                seen.add(int(token))
        except ValueError as exc:
            raise ValueError(f"invalid version token {token!r}: {exc}") from exc
    if not seen:
        raise ValueError(f"no versions parsed from {spec!r}")
    return sorted(seen)


def _cmd_experiment(args: argparse.Namespace) -> int:
    try:
        versions = _parse_versions(args.versions)
        out_of_range = [v for v in versions if not 1 <= v <= NUM_VERSIONS]
        if out_of_range:
            raise ValueError(
                f"version(s) {out_of_range} out of range 1..{NUM_VERSIONS}"
            )
    except ValueError as exc:
        print(f"error: --versions {args.versions!r}: {exc}", file=sys.stderr)
        return 2
    grid = diverse_grid() if args.grid == "diverse" else default_grid()
    if args.offline:
        factory = decaying_stub_factory()
        print(f"[offline stub] sweeping versions {versions} over {len(grid)} {args.grid} runs\n")
    else:
        factory = live_estimator_factory(effort=args.effort)
        print(
            f"[live: claude-opus-4-8, effort={args.effort}] sweeping versions {versions} "
            f"over {len(grid)} {args.grid} runs ({len(versions) * len(grid)} calls)\n"
        )

    result = run_experiment(
        grid=grid,
        versions=versions,
        estimator_factory=factory,
        max_workers=args.max_workers,
        progress=lambda line: print(line, flush=True),
    )

    print("\n" + ascii_chart(result) + "\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(markdown_report(result))
    save_json(result, out / "results.json")
    print(f"wrote {out / 'report.md'} and {out / 'results.json'}")
    try:
        save_png(result, out / "mape.png")
        print(f"wrote {out / 'mape.png'}")
    except Exception as exc:  # matplotlib optional
        print(f"(skipped PNG: {exc})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flash.cost", description="Flash training-cost estimator")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="analytical cost breakdown for one run")
    e.add_argument("--model", required=True)
    e.add_argument("--method", choices=("sft", "grpo"), required=True)
    e.add_argument("--steps", type=int, required=True)
    e.add_argument("--seq-len", type=int, default=None, dest="seq_len")
    e.add_argument("--gpu", default=None, help="pin a GPU class (else cheapest fit)")
    e.add_argument("--thinking", action="store_true")
    e.set_defaults(func=_cmd_estimate)

    x = sub.add_parser("experiment", help="prompt-convergence sweep")
    x.add_argument("--versions", default=f"1-{NUM_VERSIONS}", help="e.g. 1-6 or 1,3,6")
    x.add_argument(
        "--grid",
        default="default",
        choices=("default", "diverse"),
        help="default = 24-run model x method x steps; diverse = varied GPU/settings/env",
    )
    x.add_argument("--effort", default="low", choices=("low", "medium", "high"))
    x.add_argument("--offline", action="store_true", help="deterministic stub (no API)")
    x.add_argument("--max-workers", type=int, default=8, dest="max_workers")
    x.add_argument("--out", default="runs/cost")
    x.set_defaults(func=_cmd_experiment)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

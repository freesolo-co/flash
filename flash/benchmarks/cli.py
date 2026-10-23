"""Run and replay quality/economics suites without adding customer CLI dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flash.benchmarks.runner import replay, run_suite
from flash.benchmarks.schema import Suite


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m flash.benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("suite", type=Path)
    run = commands.add_parser("run")
    run.add_argument("suite", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--source", type=Path, default=Path.cwd())
    run.add_argument(
        "--smoke", action="store_true", help="synthetic offline evidence; never production results"
    )
    summarize = commands.add_parser("summarize")
    summarize.add_argument("output", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--allow-change", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            from flash.benchmarks.comparison import compare_runs

            result = compare_runs(
                args.baseline, args.candidate, allow_changes=tuple(args.allow_change)
            )
        elif args.command == "summarize":
            result = replay(args.output)
        else:
            suite = Suite.model_validate_json(args.suite.read_text())
            result = (
                suite.model_dump(mode="json")
                if args.command == "validate"
                else run_suite(
                    suite,
                    args.output,
                    args.source,
                    smoke=args.smoke,
                )
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get("all_trials_valid", True) else 1
    except (OSError, ValueError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("benchmark interrupted; retained partial evidence has no seal", file=sys.stderr)
        return 130

"""``autoenv`` CLI — the paper-replication benchmark harness.

A separate console script (not a ``flash`` subcommand): Flash's parser is part of its
agent-CLI contract and zero-dependency client surface, so the harness stays out of it. The
dispatch mirrors ``flash/cli/__init__.py``: argparse subcommands, each with a ``func`` handler.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autoenv import __version__
from autoenv.manifest import ManifestError, PaperCase

_USER_ERRORS = (ManifestError, FileNotFoundError, ValueError)


def _load_case(path: str) -> PaperCase:
    return PaperCase.load(path)


def cmd_gate(args) -> int:
    from autoenv.gate import gate_case

    case = _load_case(args.case)
    report = gate_case(case, probe_dataset=not args.offline)
    print(report.summary())
    return 0 if report.eligible else 1


def _train_rows(case: PaperCase) -> list[dict]:
    from autoenv.ingest import fetch_rows

    return fetch_rows(
        case.resolved_train(),
        split=case.dataset.train_split,
        input_field=case.dataset.input_field,
        output_field=case.dataset.output_field,
    )


def cmd_drive(args) -> int:
    from autoenv.drive import drive
    from autoenv.drive.agent_backend import get_backend
    from autoenv.gate.model_match import resolve_flash_model

    case = _load_case(args.case)
    match = resolve_flash_model(case.flash_model, case.base_model_paper, case.algorithm)
    dest = args.dest or f"autoenv-runs/{case.id}"
    result = drive(
        case,
        backend=get_backend(args.backend),
        model_id=match.model_id,
        train_rows=_train_rows(case),
        dest=dest,
        dry_run=not args.real,
    )
    print(f"[{result.state}] {case.id} -> {match.model_id}")
    print(f"  workspace: {result.workspace}")
    if result.estimated_usd is not None:
        print(f"  preflight cost: ${result.estimated_usd:.2f} (budget ${case.max_usd:.2f})")
    if result.run_id:
        print(f"  run: {result.run_id}")
    if result.notes:
        print(f"  {result.notes}")
    return 0 if result.state in ("dry_run", "done") else 1


def cmd_run(args) -> int:
    from autoenv.gate import gate_case

    case = _load_case(args.case)
    report = gate_case(case, probe_dataset=not args.offline)
    print(report.summary())
    if not report.eligible:
        print("\nineligible — not driving.")
        return 1
    print()
    return cmd_drive(args)


def _not_yet(stage: str) -> int:
    print(
        f"`autoenv {stage}` is not available in this build yet; see autoenv/README.md "
        "for the roadmap.",
        file=sys.stderr,
    )
    return 2


def cmd_eval(args) -> int:
    """Benchmark a completed run: deploy → infer over the eval split → improvement-normalized score."""
    from autoenv.eval.score import EvalConfig, evaluate_run
    from autoenv.ingest import fetch_rows
    from flash.client import client_from_config

    case = _load_case(args.case)
    eval_rows = fetch_rows(
        case.resolved_eval(),
        split=case.dataset.eval_split,
        input_field=case.dataset.input_field,
        output_field=case.dataset.output_field,
    )
    # Mirror the rows the run trained on for the leakage guard (best-effort; same cap as drive).
    train_rows = fetch_rows(
        case.resolved_train(),
        split=case.dataset.train_split,
        input_field=case.dataset.input_field,
        output_field=case.dataset.output_field,
        limit=case.max_train_examples or None,
    )
    result = evaluate_run(
        case,
        args.run_id,
        client=client_from_config(),
        eval_rows=eval_rows,
        train_rows=train_rows,
        config=EvalConfig(max_eval=args.max_eval, max_tokens=args.max_tokens),
    )
    print(result.summary())
    out = args.out or f"autoenv-runs/{case.id}-result.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    result.write_json(out)
    print(f"  wrote {out}")
    return 0 if result.state == "scored" else 1


def cmd_report(args) -> int:
    return _not_yet("report")


def _add_case_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("case", help="path to a PaperCase TOML manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoenv", description="Paper-replication benchmark on Flash"
    )
    parser.add_argument("-V", "--version", action="version", version=f"autoenv {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gate = sub.add_parser("gate", help="check whether a paper case is replicable on Flash")
    _add_case_arg(gate)
    gate.add_argument("--offline", action="store_true", help="skip the dataset-availability probe")
    gate.set_defaults(func=cmd_gate)

    drive = sub.add_parser("drive", help="scaffold + drive a case (dry-run by default)")
    _add_case_arg(drive)
    drive.add_argument("--backend", default="scripted", help="agent backend (default: scripted)")
    drive.add_argument(
        "--dest", default=None, help="workspace dir (default: autoenv-runs/<case-id>)"
    )
    drive.add_argument(
        "--real", action="store_true", help="submit a real run (needs credentials; M2)"
    )
    drive.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)
    drive.set_defaults(func=cmd_drive)

    run = sub.add_parser("run", help="gate then drive a case end-to-end")
    _add_case_arg(run)
    run.add_argument("--backend", default="scripted", help="agent backend (default: scripted)")
    run.add_argument("--dest", default=None, help="workspace dir (default: autoenv-runs/<case-id>)")
    run.add_argument(
        "--real", action="store_true", help="submit a real run (needs credentials; M2)"
    )
    run.add_argument("--offline", action="store_true", help="skip the dataset-availability probe")
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("eval", help="benchmark a completed run on the paper's eval split")
    _add_case_arg(ev)
    ev.add_argument(
        "--run-id", dest="run_id", required=True, help="the completed Flash run id to benchmark"
    )
    ev.add_argument("--max-eval", type=int, default=40, help="eval subsample size (default: 40)")
    ev.add_argument(
        "--max-tokens", type=int, default=512, help="generation budget per row (default: 512)"
    )
    ev.add_argument("--out", default=None, help="where to write the BenchResult JSON")
    ev.set_defaults(func=cmd_eval)

    report = sub.add_parser("report", help="aggregate BenchResults across cases (milestone M4)")
    _add_case_arg(report)
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except _USER_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("aborted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

"""operator-only module cli for hosted inference load testing."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from flash.serving.loadtest.artifacts import (
    ArtifactError,
    load_events,
    verify_result_directory,
)
from flash.serving.loadtest.metrics import summarize_events
from flash.serving.loadtest.runner import discover_scenario, run_scenario
from flash.serving.loadtest.schema import Scenario, capacity_expectations, public_scenario_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m flash.serving.loadtest")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "discover"):
        command = commands.add_parser(name)
        command.add_argument("scenario", type=Path)
    run = commands.add_parser("run")
    run.add_argument("scenario", type=Path)
    run.add_argument("result_dir", type=Path)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("result_dir", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("result_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            scenario = load_scenario(args.scenario)
            _print_json(public_scenario_dict(scenario))
            return 0
        if args.command == "discover":
            scenario = load_scenario(args.scenario)
            credential = _credential(scenario)
            resolved = asyncio.run(discover_scenario(scenario, credential))
            value = resolved.model_dump(mode="json")
            value["authored"] = public_scenario_dict(scenario)
            _print_json(value)
            return 0
        if args.command == "run":
            scenario = load_scenario(args.scenario)
            credential = _credential(scenario)
            summary = asyncio.run(run_scenario(scenario, credential, args.result_dir))
            _print_json(summary)
            return 0
        if args.command == "summarize":
            authored = load_scenario(args.result_dir / "scenario.authored.json")
            events = load_events(args.result_dir / "events.jsonl")
            _print_json(
                summarize_events(
                    events,
                    fake=authored.fake,
                    capacity_expectations=capacity_expectations(authored),
                )
            )
            return 0
        completion = verify_result_directory(args.result_dir)
        _print_json(completion)
        return 0
    except (ArtifactError, OSError, ValueError, ValidationError) as exc:
        print(f"loadtest error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("loadtest interrupted; no completion marker was written", file=sys.stderr)
        return 130


def load_scenario(path: Path) -> Scenario:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"scenario is not valid json: {path}") from exc
    return Scenario.model_validate(value)


def _credential(scenario: Scenario) -> str:
    value = os.environ.get(scenario.credential_env)
    if not value:
        raise ValueError(f"credential environment variable is unset: {scenario.credential_env}")
    return value


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))

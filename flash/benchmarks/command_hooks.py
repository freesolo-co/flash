"""Bridge existing environment evaluators and infrastructure collectors via JSON files.

Commands are argv arrays, executed without a shell. Each receives one context.json path and
writes its result to the specified result_path. Credential values remain in the environment.
"""

from __future__ import annotations

import json
import subprocess

from flash.benchmarks.evaluation import evaluate_targets, validate_fallback
from flash.benchmarks.evidence import write_json
from flash.benchmarks.schema import EVENT, Complete


def validate(context):
    names = ["prepare", "cleanup", "collect"]
    if context.case.workload == "long_multi_turn" or "evaluate_command" in context.config:
        names.append("evaluate")
    for name in names:
        command = context.config.get(f"{name}_command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(arg, str) or not arg for arg in command)
        ):
            raise ValueError(f"{name}_command must be a nonempty argv array")
    if "evaluate" not in names:
        validate_fallback(context.case)


def _invoke(context, name, *, targets=None):
    # Separate evaluations are retained rather than overwritten by the next checkpoint.
    directory = context.directory / f"{name}-{len(list(context.directory.glob(name + '-*')))}"
    directory.mkdir()
    result_path = directory / "result.json"
    write_json(
        directory / "context.json",
        {
            "request": context.request,
            "run_id": context.run_id,
            "trial_directory": str(context.directory),
            "result_path": str(result_path),
            "targets": [
                {"adapter": adapter, "customer": customer}
                for _, adapter, customer in (targets or [])
            ],
        },
    )
    command = context.config[f"{name}_command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) for arg in command)
    ):
        raise ValueError(f"{name}_command must be a nonempty argv array")
    with (
        (directory / "stdout.log").open("wb") as stdout,
        (directory / "stderr.log").open("wb") as stderr,
    ):
        subprocess.run(
            [*command, str(directory / "context.json")], stdout=stdout, stderr=stderr, check=True
        )
    result = json.loads(result_path.read_text())
    for raw in result.get("events", []):
        event = EVENT.validate_python(raw)
        if isinstance(event, Complete):
            raise ValueError("only the managed driver may emit completion")
        context.recorder.emit(event)
    return result


def prepare(context):
    _invoke(context, "prepare")


def cleanup(context):
    _invoke(context, "cleanup")


def collect(context):
    return _invoke(context, "collect")["complete_ledgers"]


def evaluate(context, targets):
    if "evaluate_command" in context.config:
        result = _invoke(context, "evaluate", targets=targets)
        return result["value"], result["samples"]
    if context.case.workload == "long_multi_turn":
        raise ValueError("long multi-turn agents require evaluate_command")
    if context.case.target.metric != "exact_match":
        raise ValueError("the default evaluator only measures exact_match")
    return evaluate_targets(
        targets,
        context.config["episodes"],
        context.recorder,
        max_tokens=context.config.get("max_tokens", 128),
    )

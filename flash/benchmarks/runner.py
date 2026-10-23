"""Execute isolated cold/warm trials and retain sufficient evidence to replay their reports."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import get_args

from flash.benchmarks.evidence import digest, load_events, seal, verify, write_json
from flash.benchmarks.report import summarize_case
from flash.benchmarks.schema import Suite, Workload


def _source_snapshot(source: Path, output: Path, *, smoke: bool) -> None:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])

    revision = git("rev-parse", "HEAD").decode().strip()
    dirty = bool(git("status", "--porcelain"))
    if dirty and not smoke:
        raise ValueError("live benchmarks require a clean source checkout; commit the driver first")
    write_json(
        output / "harness.json",
        {
            "code_revision": revision,
            "dirty": dirty,
            "synthetic": smoke,
            "python": sys.version,
            "platform": sys.platform,
        },
    )
    with (output / "source.tar").open("wb") as stream:
        subprocess.run(["git", "-C", str(source), "archive", revision], stdout=stream, check=True)


def _execute(
    command: list[str], trial: Path, source: Path, timeout: float, cleanup_timeout: float
) -> dict:
    """Give the driver a cleanup grace period, then kill the entire local process group.

    A driver owning remote resources must handle SIGTERM and release those resources. Its
    resource identifiers and cleanup evidence belong in the trial directory before paid work.
    """
    started = time.monotonic()
    timed_out = False
    with (trial / "stdout.log").open("wb") as stdout, (trial / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=source,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=cleanup_timeout)
            except BaseException:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=cleanup_timeout)
                raise
        finally:
            # Reap grandchildren even when the driver exits before its children.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "elapsed_s": time.monotonic() - started,
    }


def _trial(
    case, seed: int, cache: str, previous: Path | None, root: Path, source: Path, smoke: bool
):
    trial = root / f"{case.name}-{seed}-{cache}"
    trial.mkdir()
    configuration = {"parameters": case.configuration, "seed": seed, "cache": cache}
    request = {
        "schema_version": 1,
        "case": case.model_dump(mode="json"),
        "seed": seed,
        "cache": cache,
        "configuration": configuration,
        "configuration_digest": digest(configuration),
        "previous_trial": str(previous) if previous else None,
        "origin_monotonic_s": time.monotonic(),
        "synthetic": smoke,
    }
    write_json(trial / "request.json", request)
    command = [sys.executable, "-m", "flash.benchmarks.smoke"] if smoke else case.command
    try:
        execution = _execute(
            [*command, str(trial / "request.json")],
            trial,
            source,
            case.timeout_s,
            case.cleanup_timeout_s,
        )
    except OSError as exc:
        execution = {
            "returncode": -1,
            "timed_out": False,
            "elapsed_s": time.monotonic() - request["origin_monotonic_s"],
            "launch_error": type(exc).__name__,
        }
    execution["elapsed_s"] = time.monotonic() - request["origin_monotonic_s"]
    write_json(trial / "execution.json", execution)
    return _read_trial(trial, case, smoke)


def _read_trial(trial: Path, case, smoke: bool) -> dict:
    request = json.loads((trial / "request.json").read_text())
    execution = json.loads((trial / "execution.json").read_text())
    try:
        if (
            request.get("case") != case.model_dump(mode="json")
            or request.get("configuration", {}).get("parameters") != case.configuration
        ):
            raise ValueError("trial request does not match retained suite")
        events = load_events(trial / "events.jsonl")
        if not smoke and any(
            getattr(event, "end_s", getattr(event, "at_s", 0.0)) > execution["elapsed_s"]
            for event in events
        ):
            raise ValueError(
                "event timestamp exceeds observed driver lifetime; check clock translation"
            )
        summary = summarize_case(
            case,
            request,
            events,
            process_ok=execution["returncode"] == 0 and not execution["timed_out"],
            observed_end_s=execution["elapsed_s"],
        )
        summary["driver_elapsed_s"] = execution["elapsed_s"]
    except (ValueError, OSError) as exc:
        summary = {
            "case": case.name,
            "seed": request["seed"],
            "cache_requested": request["cache"],
            "valid": False,
            "issues": [f"invalid_evidence: {exc}"],
            "target_reached": False,
        }
    summary["synthetic"] = smoke
    summary["trial"] = trial.name
    return summary


def suite_summary(suite: Suite, trials: list[dict], *, smoke: bool) -> dict:
    groups = []
    for case in suite.cases:
        for cache in ("cold", "warm"):
            selected = [
                row
                for row in trials
                if row["case"] == case.name and row["cache_requested"] == cache
            ]
            valid = [row for row in selected if row["valid"]]
            reached = [row for row in valid if row["target_reached"]]
            # Missing targets remain failures in the denominator; never impute their time as 0.
            groups.append(
                {
                    "case": case.name,
                    "cache": cache,
                    "trials": len(selected),
                    "valid_trials": len(valid),
                    "targets_reached": len(reached),
                    "target_attainment_fraction": len(reached) / len(selected)
                    if selected
                    else None,
                    "median_time_to_target_s_among_reached": statistics.median(
                        row["time_to_target_s"] for row in reached
                    )
                    if reached
                    else None,
                }
            )
    missing = sorted(set(get_args(Workload)) - {case.workload for case in suite.cases})
    return {
        "schema_version": 1,
        "suite": suite.name,
        "synthetic": smoke,
        "all_trials_valid": all(row["valid"] for row in trials),
        "missing_workloads": missing,
        "production_claims_eligible": not smoke
        and not missing
        and all(row["valid"] and row.get("accounting_complete", False) for row in trials),
        "groups": groups,
        "trials": trials,
        "accounting": {
            "phase_time": "phase busy times overlap; exclusive + overlap + unattributed equals wall",
            "phase_cost": "uniform accrual per receipt interval, equal split across overlapping phases; attribution estimate only",
            "target_cost": "bounds include receipts straddling first observed held-out target; allocation is estimated",
            "provenance": "driver-attested runtime identities; hashes detect artifact changes, not malicious drivers",
        },
    }


def run_suite(suite: Suite, output: Path, source: Path, *, smoke: bool = False) -> dict:
    output, source = output.resolve(), source.resolve()
    # Validate cleanliness before creating artifacts, including when output is inside the checkout.
    output.mkdir(parents=True, exist_ok=False)
    _source_snapshot(source, output, smoke=smoke)
    write_json(output / "suite.json", suite.model_dump(mode="json"))
    trials = []
    for case in suite.cases:
        for seed in suite.seeds:
            previous = None
            for cache in ("cold", "warm"):
                row = _trial(case, seed, cache, previous, output, source, smoke)
                trials.append(row)
                previous = output / row["trial"]
    summary = suite_summary(suite, trials, smoke=smoke)
    write_json(output / "summary.json", summary)
    seal(output)
    return summary


def replay(output: Path) -> dict:
    verify(output)
    suite = Suite.model_validate_json((output / "suite.json").read_text())
    smoke = json.loads((output / "harness.json").read_text())["synthetic"]
    trials = [
        _read_trial(output / f"{case.name}-{seed}-{cache}", case, smoke)
        for case in suite.cases
        for seed in suite.seeds
        for cache in ("cold", "warm")
    ]
    return suite_summary(suite, trials, smoke=smoke)

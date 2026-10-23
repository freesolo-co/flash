"""Pair sealed benchmark trials without hiding quality failures or missing costs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from flash.benchmarks.runner import replay
from flash.benchmarks.schema import LEDGERS


def _differences(left, right, prefix="") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(_differences(left[key], right[key], path))
        return result
    return [] if left == right else [prefix]


def _ratio(baseline, candidate):
    return (
        candidate / baseline
        if baseline is not None and baseline > 0 and candidate is not None
        else None
    )


def _last_observed_quality(trial):
    points = trial.get("quality_curve", [])
    return points[-1]["value"] if points else None


def _pair(baseline, candidate):
    valid = baseline["valid"] and candidate["valid"]
    both_reached = valid and baseline["target_reached"] and candidate["target_reached"]
    left_quality, right_quality = (
        _last_observed_quality(baseline),
        _last_observed_quality(candidate),
    )
    money = {}
    for ledger in LEDGERS:
        left = baseline.get("money", {}).get(ledger, {})
        right = candidate.get("money", {}).get(ledger, {})
        settled = valid and left.get("complete", False) and right.get("complete", False)
        money[ledger] = {
            "baseline_known_usd": left.get("known_usd"),
            "candidate_known_usd": right.get("known_usd"),
            "both_complete": bool(settled),
            "total_cost_ratio": _ratio(left.get("total_usd"), right.get("total_usd"))
            if settled
            else None,
            # Preserve receipt bounds; a uniform interval allocation is not an actual bill.
            "baseline_to_target": left.get("to_target") if both_reached and settled else None,
            "candidate_to_target": right.get("to_target") if both_reached and settled else None,
        }
    return {
        "case": baseline["case"],
        "seed": baseline["seed"],
        "cache": baseline["cache_requested"],
        "both_valid": valid,
        "baseline_target_reached": baseline["valid"] and baseline["target_reached"],
        "candidate_target_reached": candidate["valid"] and candidate["target_reached"],
        "time_to_target_ratio": _ratio(
            baseline.get("time_to_target_s"), candidate.get("time_to_target_s")
        )
        if both_reached
        else None,
        "horizon_wall_ratio": _ratio(
            baseline.get("timing", {}).get("wall_s"), candidate.get("timing", {}).get("wall_s")
        )
        if valid
        else None,
        "baseline_last_observed_quality": left_quality,
        "candidate_last_observed_quality": right_quality,
        "last_observed_quality_delta": right_quality - left_quality
        if valid and left_quality is not None and right_quality is not None
        else None,
        "money": money,
    }


def compare_runs(baseline: Path, candidate: Path, *, allow_changes: tuple[str, ...] = ()) -> dict:
    """Verify both seals and require explicit acknowledgement of changed experimental inputs.

    Model, dataset, environment, evaluator, target, workload, seeds, and case identity must
    match. Only code/image/adapter revisions and configuration paths can be allowed to differ.
    The caller remains responsible for identifying confounding operational configuration.
    """
    for path in allow_changes:
        if path not in {
            "revisions.code",
            "revisions.image",
            "revisions.adapters",
        } and not path.startswith("configuration."):
            raise ValueError(f"cannot allow a change to comparison invariant: {path}")
    reports = [replay(root) for root in (baseline, candidate)]
    suites = [json.loads((root / "suite.json").read_text()) for root in (baseline, candidate)]
    if suites[0]["seeds"] != suites[1]["seeds"]:
        raise ValueError("comparison requires identical seeds")
    cases = [{case["name"]: case for case in suite["cases"]} for suite in suites]
    if cases[0].keys() != cases[1].keys():
        raise ValueError("comparison requires identical case names")
    changes = {}
    for name, left in cases[0].items():
        right = cases[1][name]
        keys = ("workload", "target", "revisions", "configuration")
        changed = _differences({key: left[key] for key in keys}, {key: right[key] for key in keys})
        unexpected = [
            path
            for path in changed
            if not any(
                path == allowed or path.startswith(allowed + ".") for allowed in allow_changes
            )
        ]
        if unexpected:
            raise ValueError(f"unmatched inputs for {name}: {', '.join(unexpected)}")
        changes[name] = changed
    indexes = []
    for report in reports:
        index = {}
        for row in report["trials"]:
            key = (row["case"], row["seed"], row["cache_requested"])
            if key in index:
                raise ValueError("duplicate trial pairing key")
            index[key] = row
        indexes.append(index)
    if indexes[0].keys() != indexes[1].keys():
        raise ValueError("comparison requires the same trial matrix")
    pairs = [_pair(left, indexes[1][key]) for key, left in sorted(indexes[0].items())]
    groups = []
    for name in sorted(cases[0]):
        for cache in ("cold", "warm"):
            rows = [row for row in pairs if row["case"] == name and row["cache"] == cache]
            ratios = [
                row["time_to_target_ratio"]
                for row in rows
                if row["time_to_target_ratio"] is not None
            ]
            groups.append(
                {
                    "case": name,
                    "cache": cache,
                    "paired_trials": len(rows),
                    "baseline_targets_reached": sum(row["baseline_target_reached"] for row in rows),
                    "candidate_targets_reached": sum(
                        row["candidate_target_reached"] for row in rows
                    ),
                    "both_reached_pairs": len(ratios),
                    "median_time_to_target_ratio_among_both_reached": statistics.median(ratios)
                    if ratios
                    else None,
                }
            )
    return {
        "schema_version": 1,
        "baseline_suite": reports[0]["suite"],
        "candidate_suite": reports[1]["suite"],
        "changed_inputs": changes,
        "all_trials_valid": all(pair["both_valid"] for pair in pairs),
        "production_claims_eligible": all(
            report["production_claims_eligible"] for report in reports
        ),
        "synthetic": any(report["synthetic"] for report in reports),
        "groups": groups,
        "pairs": pairs,
        "interpretation": "Ratios are candidate / baseline. Fixed-horizon wall time is not time to equal quality. Unreached targets and incomplete costs stay null; attainment counts retain every pair. Repetitions are descriptive, not a statistical superiority test.",
    }

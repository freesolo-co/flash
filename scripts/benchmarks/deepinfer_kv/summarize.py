"""Summarize paired trials without treating token count changes as speedups."""

import argparse
import json
import statistics
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def request_rows(result):
    return [row for rank in result["ranks"] for row in rank["result"].get("requests", [])]


def describe(result, extra_seconds=0):
    rows = request_rows(result)
    wall = result["wall_seconds"] + extra_seconds
    return {
        "requests": len(rows),
        "output_tokens": sum(len(row["tokens"]) for row in rows),
        "wall_seconds": wall,
        "output_tps": sum(len(row["tokens"]) for row in rows) / wall,
        "ttft_p50_seconds": percentile([row["ttft_seconds"] + extra_seconds for row in rows], 0.5),
        "ttft_p95_seconds": percentile([row["ttft_seconds"] + extra_seconds for row in rows], 0.95),
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "cached_tokens": sum(row["cached_tokens"] for row in rows),
        "retrieval_checks": sum(row.get("expected") is not None for row in rows),
        "retrieval_passed": sum(
            row["expected"] in row["text"] for row in rows if row.get("expected")
        ),
    }


def paired(rows, group_keys, generation_key=None, extra_key=None):
    groups = {}
    for trial in rows:
        key = tuple(trial[k] for k in group_keys)
        groups.setdefault(key, {}).setdefault(trial["repeat"], {})[trial["arm"]] = trial
    summaries = []
    for key, repeats in groups.items():
        trials = []
        for repeat, arms in repeats.items():
            if set(arms) != {"A", "B"}:
                raise ValueError("Incomplete A/B pair")
            metrics = {}
            signatures = {}
            for arm, trial in arms.items():
                result = trial[generation_key] if generation_key else trial
                extra = trial.get(extra_key, 0) - result["wall_seconds"] if extra_key else 0
                metrics[arm] = describe(result, extra)
                signatures[arm] = {r["case"]: r["token_sha256"] for r in request_rows(result)}
            exact = signatures["A"] == signatures["B"]
            trials.append(
                {
                    "repeat": repeat,
                    **metrics,
                    "exact_output_match": exact,
                    "exact_requests": sum(
                        signatures["A"].get(k) == v for k, v in signatures["B"].items()
                    ),
                    "compared_requests": len(signatures["B"]),
                    "throughput_ratio": metrics["B"]["output_tps"] / metrics["A"]["output_tps"],
                    "latency_ratio": metrics["B"]["wall_seconds"] / metrics["A"]["wall_seconds"],
                }
            )
        ratios = [t["throughput_ratio"] for t in trials]
        summaries.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "trials": trials,
                "median_throughput_ratio": statistics.median(ratios),
                "throughput_ratio_range": [min(ratios), max(ratios)],
                "all_exact_output_match": all(t["exact_output_match"] for t in trials),
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind",
        choices=[
            "routing",
            "tiering",
            "compression",
            "sharing",
            "disaggregation",
            "blending",
            "retention",
        ],
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.path.read_text())
    if args.kind == "routing":
        summary = paired(rows, ["length", "distribution"])
    elif args.kind == "retention":
        normalized = []
        for trial in rows:
            ranks = []
            for index, step in enumerate(trial["steps"]):
                for rank in step["generation"]["ranks"]:
                    requests = []
                    for row in rank["result"]["requests"]:
                        row = dict(row)
                        row["case"] = f"{index}-" + row["case"]
                        row["ttft_seconds"] += (
                            step["warmup"]["wall_seconds"] if step["warmup"] else 0
                        )
                        requests.append(row)
                    ranks.append({"result": {"requests": requests}})
            normalized.append(
                dict(
                    trial,
                    ranks=ranks,
                    wall_seconds=sum(step["wall_seconds"] for step in trial["steps"]),
                )
            )
        summary = paired(normalized, ["distribution"])
        for group in summary:
            group["hit_fraction_by_arm"] = {
                arm: statistics.mean(
                    sum(step["hit"] for step in t["steps"]) / len(t["steps"])
                    for t in rows
                    if t["distribution"] == group["distribution"] and t["arm"] == arm
                )
                for arm in ["A", "B"]
            }
    else:
        summary = paired(
            rows,
            ["length"],
            generation_key="generation",
            extra_key="request_wall_seconds",
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

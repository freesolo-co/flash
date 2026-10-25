"""Summarize paired arms without dropping unfavorable repeats."""

import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def summarize(root):
    grouped = defaultdict(lambda: defaultdict(list))
    quality = {}
    health = {}
    output_hashes = defaultdict(lambda: defaultdict(set))
    for label in ("fp4-first", "fp8-first", "fp8-second", "fp4-second"):
        arm = root / label
        mode = label.split("-")[0]
        health[label] = json.loads((arm / "health.json").read_text())
        quality[label] = json.loads((arm / "quality.json").read_text())
        for line in (arm / "measurements.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert not row["errors"], row
            assert len(row["requests"]) == row["concurrency"]
            key = f"{row['workload']}/{row['cache']}/c{row['concurrency']}"
            grouped[key][mode].append(row)
            for request in row["requests"]:
                output_hashes[key][mode].add(request["text_sha256"])
        log = (arm / "server.log").read_text()
        assert ("fp8xfp4" if mode == "fp4" else "fp8xfp8") in log, (
            f"No actual kernel proof: {label}"
        )
    comparisons = []
    for key, modes in sorted(grouped.items()):
        result = {"workload": key, "arms": {}}
        for mode, rows in modes.items():
            assert len(rows) == 6, (key, mode, len(rows))
            result["arms"][mode] = {
                field: statistics.median(row[field] for row in rows)
                for field in (
                    "aggregate_output_tps",
                    "ttft_s_p50",
                    "ttft_s_p95",
                    "latency_s_p95",
                    "cached_input_fraction",
                )
            }
            result["arms"][mode]["throughput_repeats"] = [r["aggregate_output_tps"] for r in rows]
        a, b = result["arms"]["fp4"], result["arms"]["fp8"]
        result["fp4_over_fp8_throughput"] = a["aggregate_output_tps"] / b["aggregate_output_tps"]
        result["fp4_over_fp8_p95_ttft"] = a["ttft_s_p95"] / b["ttft_s_p95"]
        result["fp4_over_fp8_p95_latency"] = a["latency_s_p95"] / b["latency_s_p95"]
        result["within_performance_thresholds"] = (
            result["fp4_over_fp8_throughput"] >= 0.95
            and result["fp4_over_fp8_p95_ttft"] <= 1.10
            and result["fp4_over_fp8_p95_latency"] <= 1.10
        )
        result["identical_output_sets"] = output_hashes[key]["fp4"] == output_hashes[key]["fp8"]
        comparisons.append(result)
    quality_summary = {
        label: {"correct": sum(r["correct"] for r in rows), "total": len(rows)}
        for label, rows in quality.items()
    }
    failures = {
        label: [
            {
                "prompt_sha256": hashlib.sha256(r["prompt"].encode()).hexdigest(),
                "expected": r["expected"],
                "text": r["text"],
            }
            for r in rows
            if not r["correct"]
        ]
        for label, rows in quality.items()
    }
    quality_no_regression = all(
        all(
            not other["correct"] or row["correct"]
            for other in quality["fp8-first"] + quality["fp8-second"]
            if other["prompt"] == row["prompt"]
        )
        for label in ("fp4-first", "fp4-second")
        for row in quality[label]
    )
    memory = [json.loads(line) for line in (root / "gpu-memory.jsonl").read_text().splitlines()]
    resources = {}
    for label in health:
        start = json.loads((root / label / "started.json").read_text())["time"]
        end = json.loads((root / label / "stopped.json").read_text())["time"]
        samples = [r for r in memory if start <= r["time"] <= end]
        peaks = [
            max(float(gpu.split(",")[2].strip().split()[0]) for gpu in r["gpus"]) for r in samples
        ]
        log = (root / label / "server.log").read_text()
        allocated = re.findall(
            r"Memory allocated: builder=[0-9.]+ GiB, deepinfer=([0-9.]+) GiB", log
        )
        blocks = re.findall(r"init block_manager.*=\((\d+), (\d+),", log)
        resources[label] = {
            "startup_seconds": health[label]["startup_seconds"],
            "peak_sampled_gpu_mib": max(peaks),
            "model_allocated_gib_per_rank": [float(v) for v in allocated],
            "kv_blocks_per_rank": [int(v[0]) for v in blocks],
        }
    return {
        "comparisons": comparisons,
        "quality": quality_summary,
        "quality_failures": failures,
        "quality_no_regression": quality_no_regression,
        "resources": resources,
        "fp4_meets_all_thresholds": quality_no_regression
        and all(r["within_performance_thresholds"] for r in comparisons),
    }


if __name__ == "__main__":
    root = Path(sys.argv[1])
    result = summarize(root)
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

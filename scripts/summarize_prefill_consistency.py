"""Summarize matched prefill waves; paired ranges are descriptive, not confidence intervals."""

import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def summarize(rows):
    matched = {}
    for row in rows:
        key = row["case"]["name"], row["block"], row["repeat"]
        pair = matched.setdefault(key, {})
        if row["budget"] in pair:
            raise ValueError(f"Duplicate wave: {key}, {row['budget']}")
        pair[row["budget"]] = row
    by_case = {}
    for (name, block, repeat), pair in matched.items():
        if set(pair) != {4096, 8192}:
            continue
        before, after = pair[4096], pair[8192]
        assert [r["prompt_hash"] for r in before["requests"]] == [
            r["prompt_hash"] for r in after["requests"]
        ], f"Inputs differ: {name}/{block}/{repeat}"
        assert before["gpu"].split(",")[0] == after["gpu"].split(",")[0], "GPU changed"
        by_case.setdefault(name, []).append((before, after))
    summaries = []
    for name, pairs in sorted(by_case.items()):
        result = {"case": name, "paired_waves": len(pairs)}
        for metric in (
            "aggregate_tps",
            "p50_stream_tps",
            "p05_stream_tps",
            "p95_ttft",
            "p95_latency",
        ):
            ratios = [b[metric] / a[metric] for a, b in pairs]
            result[f"{metric}_4096"] = statistics.mean(a[metric] for a, _ in pairs)
            result[f"{metric}_8192"] = statistics.mean(b[metric] for _, b in pairs)
            result[f"{metric}_change_pct"] = 100 * (geometric_mean(ratios) - 1)
            result[f"{metric}_min_change_pct"] = 100 * (min(ratios) - 1)
            result[f"{metric}_max_change_pct"] = 100 * (max(ratios) - 1)
            for block in (0, 1):
                values = [b[metric] / a[metric] for a, b in pairs if a["block"] == block]
                if values:
                    result[f"{metric}_block{block}_change_pct"] = 100 * (geometric_mean(values) - 1)
        result["output_hash_matches"] = sum(
            x["output_hash"] == y["output_hash"]
            for a, b in pairs
            for x, y in zip(a["requests"], b["requests"], strict=True)
        )
        result["paired_requests"] = sum(len(a["requests"]) for a, _ in pairs)
        summaries.append(result)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.path.suffix == ".log":
        rows = [
            json.loads(line.split("PREFILL_WAVE ", 1)[1])
            for line in args.path.read_text().splitlines()
            if "PREFILL_WAVE {" in line
        ]
    else:
        content = (
            gzip.decompress(args.path.read_bytes()).decode()
            if args.path.suffix == ".gz"
            else args.path.read_text()
        )
        phases = json.loads(content)
        rows = [r for phase in phases for r in phase["measurements"]]
    summaries = summarize(rows)
    output = args.path.parent / "summary.json"
    output.write_text(json.dumps(summaries, indent=2) + "\n")
    if summaries:
        with (args.path.parent / "summary.csv").open("w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=sorted(set().union(*(s.keys() for s in summaries))),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summaries)
    print(
        f"{len(rows)} waves; {sum(len(r['requests']) for r in rows)} requests; "
        f"{len(summaries)} matched cases"
    )
    for row in summaries:
        print(
            f"{row['case']:36} n={row['paired_waves']} "
            f"aggregate {row['aggregate_tps_change_pct']:+6.1f}% "
            f"stream {row['p50_stream_tps_change_pct']:+6.1f}% "
            f"TTFT {row['p95_ttft_change_pct']:+6.1f}%"
        )


if __name__ == "__main__":
    main()

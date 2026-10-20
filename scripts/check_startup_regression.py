"""Compare matched worker timing reports; a missing stage is not a performance improvement."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def regressions(baseline: dict, candidate: dict, *, max_ratio: float) -> list[str]:
    if not math.isfinite(max_ratio) or max_ratio < 1:
        raise ValueError("max_ratio must be finite and at least one")
    if not baseline.get("configuration") or baseline["configuration"] != candidate.get(
        "configuration"
    ):
        raise ValueError("reports must identify the same model/revision, GPU, image and workload")
    previous = baseline.get("stage_timings", {})
    current = candidate.get("stage_timings", {})
    if not previous:
        raise ValueError("baseline has no stage timings")
    failures = []
    for stage, item in previous.items():
        old = item.get("seconds")
        new = current.get(stage, {}).get("seconds")
        if (
            isinstance(old, bool)
            or not isinstance(old, (int, float))
            or not math.isfinite(old)
            or old <= 0
            or isinstance(new, bool)
            or not isinstance(new, (int, float))
            or not math.isfinite(new)
            or new < 0
        ):
            raise ValueError(f"missing or invalid duration for {stage}")
        if item.get("count") != current[stage].get("count"):
            raise ValueError(f"different sample count for {stage}")
        if item.get("failures", 0) or current[stage].get("failures", 0):
            raise ValueError(f"failed work cannot establish a timing comparison: {stage}")
        if new > old * max_ratio:
            failures.append(f"{stage}: {old:.2f}s -> {new:.2f}s")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-ratio", type=float, default=1.2)
    args = parser.parse_args()
    failures = regressions(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        max_ratio=args.max_ratio,
    )
    for message in failures:
        print(message)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

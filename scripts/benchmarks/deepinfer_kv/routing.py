"""Matched native A/B: least-active vs cache-aware routing, uniform and skewed traffic."""

import json
import random

from cache_routing import RankCandidate, choose_rank
from driver import ROOT, all_ranks, fixed_batch, phase, warm


def assign(cases, aware):
    assignments = [[] for _ in range(4)]
    work = [0.0] * 4
    for case in cases:
        candidates = [
            RankCandidate(
                rank,
                case["cut"] if case["group"] == rank else 0,
                work[rank],
                len(assignments[rank]),
                24,
            )
            for rank in range(4)
        ]
        rank = (
            choose_rank(candidates, len(case["tokens"]), case["output"] * 32)
            if aware
            else min(range(4), key=lambda r: (len(assignments[r]), r))
        )
        request = dict(case, key=case["key"] if case["group"] == rank else None)
        assignments[rank].append(request)
        work[rank] += (
            len(case["tokens"])
            - (case["cut"] if case["group"] == rank else 0)
            + case["output"] * 32
        )
    return [{"op": "batch", "requests": requests} for requests in assignments]


def run():
    results = []
    for length in [2048, 8192, 16384]:
        cases = json.loads((ROOT / f"workload-{length}.json").read_text())
        all_ranks("routing-clear", {"op": "clear"})
        fixed_batch("routing-prewarm-cold", cases, False)
        warm(cases)
        fixed_batch("routing-prewarm-cached", cases, True)
        for distribution in ["uniform", "skewed"]:
            for repeat in range(3):
                rng = random.Random(900 + repeat)
                selected = [
                    dict(
                        rng.choice(
                            cases if distribution == "uniform" or rng.random() > 0.75 else cases[:4]
                        ),
                        case=f"{distribution}-{repeat}-{i}",
                    )
                    for i in range(32)
                ]
                # Same arrivals and prompts in each arm. Alternate order to reduce drift.
                for aware in [False, True] if repeat % 2 == 0 else [True, False]:
                    arm = "B" if aware else "A"
                    result = phase(
                        f"routing-{length}-{distribution}-{repeat}-{arm}", assign(selected, aware)
                    )
                    result.update(length=length, distribution=distribution, repeat=repeat, arm=arm)
                    results.append(result)
                    (ROOT / "routing-results.json").write_text(json.dumps(results))
    print("ROUTING_COMPLETE", len(results), flush=True)


if __name__ == "__main__":
    run()

"""Two-replica disaggregation A/B experiment; run on the controller host."""

import json
from pathlib import Path

from multi_driver import Replica, batch, clear_relay, parallel, transfer


def disaggregation(source, destination, root):
    results = []
    for length in [2048, 8192, 16384]:
        cases = json.loads(source.sandbox.filesystem.read_text(f"/results/workload-{length}.json"))
        for repeat in range(3):
            for arm in ["A", "B"] if repeat % 2 == 0 else ["B", "A"]:
                parallel(
                    [
                        lambda: source.all("pd-source-reset", {"op": "clear"}),
                        lambda: destination.all("pd-destination-reset", {"op": "clear"}),
                    ]
                )
                clear_relay(root)
                if arm == "A":
                    outputs = parallel(
                        [
                            lambda cases=cases: source.phase(
                                "pd-colocated-source", batch(cases, range(2))
                            ),
                            lambda cases=cases: destination.phase(
                                "pd-colocated-peer", batch(cases, range(2, 4))
                            ),
                        ]
                    )
                    generation = {
                        "ranks": outputs[0]["ranks"] + outputs[1]["ranks"],
                        "wall_seconds": max(x["wall_seconds"] for x in outputs),
                    }
                    result = {
                        "generation": generation,
                        "request_wall_seconds": generation["wall_seconds"],
                    }
                else:
                    prefill = source.phase(
                        "pd-prefill",
                        [
                            {
                                "op": "extension",
                                "module": "pd_actions",
                                "requests": [
                                    dict(cases[r * 4 + q], key=cases[r * 4 + q]["case"])
                                    for q in range(4)
                                ],
                            }
                            for r in range(4)
                        ],
                    )
                    push, pull = transfer(
                        source,
                        destination,
                        f"pd-{length}-{repeat}",
                        [[cases[r * 4 + q]["case"] for q in range(4)] for r in range(4)],
                    )
                    generation = destination.phase("pd-decode", batch(cases, cached=True))
                    result = {
                        "prefill": prefill,
                        "push": push,
                        "pull": pull,
                        "generation": generation,
                        "request_wall_seconds": sum(
                            x["wall_seconds"] for x in [prefill, push, pull, generation]
                        ),
                    }
                result.update(length=length, repeat=repeat, arm=arm)
                results.append(result)
                (root / "disaggregation-results.json").write_text(json.dumps(results))
                print("PD", length, repeat, arm, result["request_wall_seconds"], flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    entries = json.loads((args.root / "ledger.json").read_text())["allocations"]
    source = Replica(next(x for x in entries if x["label"] == "primary2"))
    destination = Replica(next(x for x in entries if x["label"] == "peer2"))
    disaggregation(source, destination, args.root)


if __name__ == "__main__":
    main()

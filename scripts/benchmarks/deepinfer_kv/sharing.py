"""Two-replica sharing A/B experiment; run on the controller host."""

import json
from pathlib import Path

from multi_driver import Replica, batch, clear_relay, transfer


def sharing(source, destination, root):
    results = []
    for length in [2048, 8192, 16384]:
        cases = json.loads(source.sandbox.filesystem.read_text(f"/results/workload-{length}.json"))
        source.all("sharing-source-reset", {"op": "clear"})
        source.phase(
            "sharing-warm",
            [
                {
                    "op": "warm",
                    "key": f"p{r}",
                    "tokens": cases[r * 4]["tokens"],
                    "cut": cases[r * 4]["cut"],
                }
                for r in range(4)
            ],
        )
        # Compile both uncached and restored paths before measured trials.
        destination.phase("sharing-prewarm-cold", batch(cases))
        for repeat in range(3):
            for arm in ["A", "B"] if repeat % 2 == 0 else ["B", "A"]:
                clear_relay(root)
                destination.all("sharing-destination-reset", {"op": "clear"})
                push = pull = None
                if arm == "B":
                    push, pull = transfer(
                        source,
                        destination,
                        f"sharing-{length}-{repeat}",
                        [[f"p{r}"] for r in range(4)],
                    )
                commands = [
                    {
                        "op": "batch",
                        "requests": [
                            dict(cases[r * 4 + q], key=f"p{r}" if arm == "B" else None)
                            for q in range(4)
                        ],
                    }
                    for r in range(4)
                ]
                generation = destination.phase(f"sharing-{length}-{repeat}-{arm}", commands)
                result = {
                    "length": length,
                    "repeat": repeat,
                    "arm": arm,
                    "generation": generation,
                    "push": push,
                    "pull": pull,
                    "request_wall_seconds": generation["wall_seconds"]
                    + (push["wall_seconds"] + pull["wall_seconds"] if push else 0),
                }
                results.append(result)
                (root / "sharing-results.json").write_text(json.dumps(results))
                print("SHARING", length, repeat, arm, result["request_wall_seconds"], flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    entries = json.loads((args.root / "ledger.json").read_text())["allocations"]
    source = Replica(next(x for x in entries if x["label"] == "primary2"))
    destination = Replica(next(x for x in entries if x["label"] == "peer2"))
    sharing(source, destination, args.root)


if __name__ == "__main__":
    main()

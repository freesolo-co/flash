"""A/B host/storage compression. Does not claim active GPU KV-memory reduction."""

import json

from driver import ROOT, all_ranks, correctness, fixed_batch, phase, warm


def run():
    results = []
    for length in [2048, 8192, 16384]:
        cases = json.loads((ROOT / f"workload-{length}.json").read_text())
        for repeat in range(3):
            all_ranks("compression-reset", {"op": "clear"})
            warm(cases)
            phase(
                "compression-snapshot",
                [
                    {"op": "snapshot", "key": f"p{r}", "module": "compression_actions"}
                    for r in range(4)
                ],
            )
            pair = {}
            for arm in ["A", "B"] if repeat % 2 == 0 else ["B", "A"]:
                restore = phase(
                    f"compression-restore-{arm}",
                    [
                        {
                            "op": "restore" if arm == "A" else "extension",
                            "key": f"p{r}",
                            "module": "compression_actions",
                        }
                        for r in range(4)
                    ],
                )
                generation = fixed_batch(f"compression-{length}-{repeat}-{arm}", cases, True)
                pair[arm] = generation
                results.append(
                    {
                        "length": length,
                        "repeat": repeat,
                        "arm": arm,
                        "restore": restore,
                        "generation": generation,
                        "request_wall_seconds": restore["wall_seconds"]
                        + generation["wall_seconds"],
                    }
                )
            results[-1]["correctness"] = correctness(pair["A"], pair["B"])
            (ROOT / "compression-results.json").write_text(json.dumps(results))
    print("COMPRESSION_COMPLETE", flush=True)


if __name__ == "__main__":
    run()

"""File-based command client. Timings are native-engine, not public HTTP latency."""

import json
import time
from pathlib import Path

ROOT = Path("/results")


def phase(name, ranks):
    assert len(ranks) == 4
    sequence = len(list((ROOT / "commands").glob("*.json")))
    path = ROOT / "commands" / f"{sequence:05d}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"name": name, "ranks": ranks}))
    started = time.perf_counter()
    temporary.replace(path)
    response = ROOT / "responses" / path.name
    while not response.exists():
        assert time.perf_counter() - started < 600, f"Phase timeout {name}"
        time.sleep(0.05)
    result = json.loads(response.read_text())
    print("RESULT", name, round(result["wall_seconds"], 3), flush=True)
    return result


def all_ranks(name, command):
    return phase(name, [command] * 4)


def warm(cases):
    return phase(
        "warm-prefixes",
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


def fixed_batch(name, cases, cached):
    return phase(
        name,
        [
            {
                "op": "batch",
                "requests": [
                    dict(cases[r * 4 + q], key=f"p{r}" if cached else None) for q in range(4)
                ],
            }
            for r in range(4)
        ],
    )


def rows(result):
    return [request for rank in result["ranks"] for request in rank["result"].get("requests", [])]


def correctness(reference, candidate):
    expected = {r["case"]: r for r in rows(reference)}
    checks = []
    for row in rows(candidate):
        other = expected[row["case"]]
        checks.append(
            {
                "case": row["case"],
                "exact_tokens": other["tokens"] == row["tokens"],
                "retrieval_correct": row["expected"] in row["text"]
                if row.get("expected")
                else None,
                "first_difference": next(
                    (
                        i
                        for i, (x, y) in enumerate(
                            zip(other["tokens"], row["tokens"], strict=False)
                        )
                        if x != y
                    ),
                    None,
                ),
            }
        )
    return checks


def smoke():
    cases = json.loads((ROOT / "workload-8192.json").read_text())
    all_ranks("reset", {"op": "clear"})
    reference = fixed_batch("uncached-reference", cases, False)
    warm(cases)
    cached = fixed_batch("retained-reference", cases, True)
    export = phase("snapshot", [{"op": "snapshot", "key": f"p{r}"} for r in range(4)])
    phase("drop", [{"op": "drop", "key": f"p{r}"} for r in range(4)])
    restore = phase("restore", [{"op": "restore", "key": f"p{r}"} for r in range(4)])
    restored = fixed_batch("restored-candidate", cases, True)
    result = {
        "cached": correctness(reference, cached),
        "restored": correctness(reference, restored),
        "restore_vs_retained": correctness(cached, restored),
        "export": export,
        "restore": restore,
    }
    (ROOT / "smoke.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    smoke()

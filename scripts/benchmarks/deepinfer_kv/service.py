"""Four-rank native benchmark daemon, no public endpoint or customer traffic."""

import importlib
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def worker(rank, incoming, outgoing):
    try:
        from serving_init import initialize

        args = SimpleNamespace(
            spec=3,
            confidence_temperature=1.0,
            engram_storage="cpu",
            yoco=1,
            export_history=0,
            extra_experts=0,
            batch_invariant=1,
            graph_interval=8,
            dense_short_context=0,
            offline_state_pool=1,
            text_only=1,
            eplb_interval=0,
            eplb_window=3,
            eplb_after_prefill=0,
            local_transport=True,
            gpus=4,
            loader_allocator_memory_fraction=0.96,
            allocator_memory_fraction=0.96,
            max_model_len=32768,
            kv_gib=8,
            batches=[24],
            chunk=4096,
            memory=1.0,
            results="/results",
        )
        llm, tokenizer, engine = initialize(rank, args)
        from native import Native

        native = Native(llm, tokenizer, engine, rank)
        outgoing.put({"rank": rank, "ready": True})
        import actions

        while True:
            command = incoming.get()
            if command["op"] == "stop":
                break
            actions = importlib.reload(actions)
            result = actions.execute(native, command)
            outgoing.put({"rank": rank, "result": result})
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        outgoing.put({"rank": rank, "error": error})
        raise


def collect(outgoing, processes, deadline=600):
    rows = []
    until = time.monotonic() + deadline
    while len(rows) < 4:
        try:
            row = outgoing.get(timeout=1)
            rows.append(row)
            if "error" in row:
                raise RuntimeError(row["error"])
        except queue.Empty:
            assert all(p.is_alive() for p in processes), "Rank process exited"
            assert time.monotonic() < until, "Rank response timeout"
    return sorted(rows, key=lambda row: row["rank"])


def main():
    root = Path("/results")
    (root / "failure.json").unlink(missing_ok=True)
    commands = root / "commands"
    results = root / "responses"
    commands.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    context = mp.get_context("spawn")
    outgoing = context.Queue()
    incoming = [context.Queue() for _ in range(4)]
    processes = [
        context.Process(target=worker, args=(rank, incoming[rank], outgoing)) for rank in range(4)
    ]
    for p in processes:
        p.start()
    try:
        ready = collect(outgoing, processes, 1800)
        (root / "ready.json").write_text(json.dumps(ready))
        while time.time() < float(os.environ["BENCH_DEADLINE"]) - 120:
            available = [
                p for p in sorted(commands.glob("*.json")) if not (results / p.name).exists()
            ]
            if not available:
                time.sleep(0.1)
                continue
            path = available[0]
            command = json.loads(path.read_text())
            started = time.perf_counter()
            for rank, q in enumerate(incoming):
                q.put(command["ranks"][rank])
            result = {
                "name": command["name"],
                "ranks": collect(outgoing, processes),
                "wall_seconds": time.perf_counter() - started,
                "finished_epoch": time.time(),
            }
            tmp = (results / path.name).with_suffix(".tmp")
            tmp.write_text(json.dumps(result))
            tmp.replace(results / path.name)
            print("PHASE_DONE", command["name"], result["wall_seconds"], flush=True)
    except BaseException:
        (root / "failure.json").write_text(json.dumps({"error": traceback.format_exc()}))
        raise
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(10)


if __name__ == "__main__":
    main()

"""Paired ABBA CUDA-graph fetch benchmarks on four simultaneously active GPUs.

Each replay changes the row indices. Tables exceed GPU cache, and A/B use the same
table values, index sequence, GPU topology, process group, and CUDA graph mode.
This measures Engram transport, not full-model serving throughput.

Run in the pinned four-GPU image after qualification:
    python scripts/benchmark_deepinfer_engram.py --baseline /tmp/baseline.py \
        --candidate flash/serving/deepinfer/native/engram_transport.py \
        --replays 10 --output /tmp/engram-ab.json
"""

import argparse
import gc
import json
import os
import statistics
from datetime import timedelta
from pathlib import Path

from qualify_deepinfer_engram import load


def worker(rank, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29882",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=180),
    )
    records = []
    hidden, entries = 256, 24
    rows = [args.rows, args.rows + 128]
    classes = {"A": load(args.baseline), "B": load(args.candidate)}
    for storage in ("cpu", "cuda"):
        for order, variant in enumerate("ABBA"):
            weights, scales = [], []
            for layer, size in enumerate(rows):
                # Fill in CUDA, then stage CPU mode once outside measurement.
                weight = torch.empty((size, hidden), device="cuda", dtype=torch.uint8)
                weight.fill_(30 + rank + layer)
                weights.append(weight.view(torch.float8_e4m3fn).to(storage))
                scales.append(torch.full((size, hidden // 32), 2.0**rank, device=storage))
            buf = classes[variant](dist.group.WORLD, rows, hidden, 4096, entries)
            buf.write(weights, scales)
            for tokens in (1, 8, 32, 128, 512, 4096):
                gen = torch.Generator().manual_seed(712 + rank + tokens)
                ids = (
                    torch.stack(
                        [
                            torch.randint(0, size * 4, (tokens, entries), generator=gen)
                            for size in rows
                        ]
                    )
                    .int()
                    .cuda()
                )
                limits = torch.tensor(rows, dtype=torch.int32, device="cuda")[:, None, None] * 4

                def step(ids=ids, limits=limits, buf=buf):
                    ids.add_(104729).remainder_(limits)
                    return [hook() for hook in buf.fetch(ids)]

                for _ in range(3):
                    step()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.unroll):
                        output = step()
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                times = []
                for _sample in range(7):
                    dist.barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(args.replays):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end) * 1000 / (args.replays * args.unroll))
                records.append(
                    {
                        "rank": rank,
                        "storage": storage,
                        "order": order,
                        "variant": variant,
                        "tokens": tokens,
                        "microseconds": times,
                        "median_us": statistics.median(times),
                    }
                )
                if rank == 0:
                    print(json.dumps(records[-1]), flush=True)
                del graph, output, ids, limits
            buf.destroy()
            del buf, weights, scales, weight
            gc.collect()
            dist.barrier()
            torch.cuda.empty_cache()
    gathered = [None] * 4
    dist.all_gather_object(gathered, records)
    if rank == 0:
        Path(args.output).write_text(
            json.dumps(
                {
                    "rows": rows,
                    "hidden": hidden,
                    "entries": entries,
                    "replays": args.replays,
                    "unroll": args.unroll,
                    "records": [record for rank_records in gathered for record in rank_records],
                },
                indent=2,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=1048576)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--unroll", type=int, default=10)
    args = parser.parse_args()
    import torch.multiprocessing as mp

    mp.spawn(worker, args=(args,), nprocs=4)

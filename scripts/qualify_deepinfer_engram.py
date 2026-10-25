"""Run in the pinned native image on four GPUs; no model weights required.

Checks CPU/GPU shards, exact FP8 bytes/scales, both scale layouts, graph replay
with changed indices, invalid rows, partial kernel blocks, and input rejection.
"""

import argparse
import importlib.util
import json
import os
import time
from datetime import timedelta
from pathlib import Path


def load(path):
    spec = importlib.util.spec_from_file_location("tested_engram", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LocalEngramBuffer


def must_reject(operation, *args, exception=ValueError):
    try:
        operation(*args)
    except exception:
        return
    raise AssertionError("Invalid input was accepted")


def worker(rank, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29881",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    cls = load(args.module)
    records = []
    for storage in ("cpu", "cuda"):
        for hidden in (128, 256, 384, 512):
            rows = [37, 53]
            entries = 3
            weights, scales = [], []
            for layer, size in enumerate(rows):
                row = torch.arange(size, device=storage)[:, None] + rank * size
                col = torch.arange(hidden, device=storage)[None, :]
                weights.append(
                    ((row * 3 + col + layer * 7) % 120 + 1)
                    .to(torch.uint8)
                    .view(torch.float8_e4m3fn)
                )
                block = torch.arange(hidden // 32, device=storage)[None, :]
                scales.append(torch.exp2(((row + block + layer) % 11 - 5).float()).contiguous())
            buf = cls(dist.group.WORLD, rows, hidden, 4096, entries)

            must_reject(buf.write, weights, None)
            must_reject(buf.write, [w.float() for w in weights], scales)
            must_reject(buf.write, weights, [s[:, :-1] for s in scales])
            must_reject(
                buf.fetch,
                torch.zeros((2, 1, entries), dtype=torch.int32, device="cuda"),
                exception=RuntimeError,
            )
            torch.set_default_device("cuda")
            buf.write(weights, scales)
            torch.set_default_device("cpu")

            def verify(
                hooks,
                ids,
                col_major=True,
                rows=rows,
                hidden=hidden,
                entries=entries,
                storage=storage,
            ):
                for layer in reversed(range(len(rows))):
                    out, sf = hooks[layer]()
                    token_count = ids.shape[1]
                    r = ids[layer].cpu().long().flatten()
                    valid = (r >= 0) & (r < rows[layer] * 4)
                    cols = torch.arange(hidden)[None, :]
                    expected = ((r[:, None] * 3 + cols + layer * 7) % 120 + 1).to(torch.uint8)
                    expected[~valid] = 0
                    blocks = torch.arange(hidden // 32)[None, :]
                    packed = (127 + (r[:, None] + blocks + layer) % 11 - 5).to(torch.uint8)
                    packed[~valid] = 127
                    expected_sf = packed.view(torch.int32).reshape(
                        token_count, entries * hidden // 128
                    )
                    assert torch.equal(
                        out.cpu().view(torch.uint8), expected.reshape(token_count, entries * hidden)
                    ), (storage, hidden, layer, "values")
                    assert torch.equal(sf.cpu(), expected_sf), (storage, hidden, layer, "scales")
                    if token_count > 1:
                        if col_major:
                            assert sf.stride() == (1, ((token_count + 3) // 4) * 4)
                        else:
                            assert sf.is_contiguous(), "row-major request ignored"

            for count in (0, 1, 5, 17, 129, 4096):
                gen = torch.Generator().manual_seed(1000 + rank + count)
                ids = (
                    torch.stack(
                        [
                            torch.randint(0, size * 4, (count, entries), generator=gen)
                            for size in rows
                        ]
                    )
                    .int()
                    .cuda()
                )
                if count:
                    ids[:, 0, 0] = -1
                    for layer, size in enumerate(rows):
                        ids[layer, -1, -1] = size * 4
                verify(buf.fetch(ids), ids)
                verify(buf.fetch(ids, use_tma_aligned_col_major_sf=False), ids, False)
                if count:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        hooks = buf.fetch(ids)
                    ids.fill_(0)
                    graph.replay()
                    verify(hooks, ids)
                    del graph, hooks
                records.append(
                    {"storage": storage, "hidden": hidden, "tokens": count, "exact": True}
                )

            # Both original contract failures must now be fixed, not merely logged.
            ids = torch.zeros((2, 5, entries), dtype=torch.int32, device="cuda")
            try:
                verify(buf.fetch(ids, use_tma_aligned_col_major_sf=False), ids, False)
                layout = "pass"
            except AssertionError as error:
                layout = str(error)
            noncontiguous = torch.arange(
                2 * 5 * entries * 2, dtype=torch.int32, device="cuda"
            ).reshape(2, 5, entries * 2)[:, :, ::2]
            try:
                verify(buf.fetch(noncontiguous), noncontiguous)
                noncontiguous_result = "supported"
            except (ValueError, AssertionError) as error:
                noncontiguous_result = "rejected" if "contiguous" in str(error) else "wrong values"
            assert layout == "pass", layout
            assert noncontiguous_result == "rejected", noncontiguous_result
            must_reject(buf.fetch, ids.long())
            must_reject(buf.fetch, ids.cpu())
            must_reject(buf.fetch, ids[0])
            must_reject(buf.fetch, ids[:, :, :2].contiguous())
            must_reject(
                buf.fetch, torch.zeros((2, 4097, entries), dtype=torch.int32, device="cuda")
            )
            must_reject(buf.write, weights, scales, exception=RuntimeError)
            # Retaining one fetch while issuing another must not overwrite its result.
            first = buf.fetch(ids)
            second_ids = ids + 1
            second = buf.fetch(second_ids)
            verify(first, ids)
            verify(second, second_ids)
            del first, second
            records.append(
                {
                    "storage": storage,
                    "hidden": hidden,
                    "row_major": layout,
                    "noncontiguous": noncontiguous_result,
                }
            )
            buf.destroy()
            del buf, weights, scales
            torch.cuda.empty_cache()
    gathered = [None] * 4
    dist.all_gather_object(gathered, {"rank": rank, "checks": records})
    if rank == 0:
        Path(args.output).write_text(json.dumps(gathered, indent=2))
        print(
            json.dumps(
                {
                    "output": args.output,
                    "checks_per_rank": len(records),
                    "contracts": [r for r in records if "row_major" in r],
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        default=str(
            Path(__file__).resolve().parents[1]
            / "flash/serving/deepinfer/native/engram_transport.py"
        ),
    )
    parser.add_argument("--output", default="/tmp/engram-qualification.json")
    args = parser.parse_args()
    import torch.multiprocessing as mp

    context = mp.spawn(worker, args=(args,), nprocs=4, join=False)
    deadline = time.monotonic() + 300
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("Engram qualification exceeded five minutes")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join(5)

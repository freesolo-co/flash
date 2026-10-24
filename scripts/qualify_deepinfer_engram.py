"""Run inside the native image on four GPUs; check host Engram transport and graph replay."""

import json
import multiprocessing
import os
import time
from datetime import timedelta
from pathlib import Path


def rank_main(rank):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch
    import torch.distributed as dist

    from flash.serving.deepinfer.native.engram_transport import LocalEngramBuffer

    torch.cuda.set_device(0)
    torch.set_default_device("cuda")
    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29977",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    rows = torch.arange(8, device="cpu")[:, None] + rank * 8
    cols = torch.arange(256, device="cpu")[None, :]
    weight = ((rows * 3 + cols) % 120 + 1).to(torch.uint8).view(torch.float8_e4m3fn)
    scales = torch.tensor(
        [[2.0 ** ((rank + row + block) % 5) for block in range(8)] for row in range(8)],
        dtype=torch.float32,
        device="cpu",
    )
    buf = LocalEngramBuffer(dist.group.WORLD, [8], 256, 5, 2)
    buf.write([weight], [scales])
    indices = torch.tensor(
        [0, 7, 8, 15, 16, 23, 24, 31, -1, 32], dtype=torch.int32, device="cuda"
    ).reshape(1, 5, 2)

    def verify(result, values):
        out, sf = result[0]()
        expected = []
        packed = []
        for row in values:
            owner = row // 8
            if 0 <= row < 32:
                expected.append(
                    ((row * 3 + torch.arange(256, device="cpu")) % 120 + 1)
                    .to(torch.uint8)
                    .view(torch.float8_e4m3fn)
                )
                for start in (0, 4):
                    exponents = bytes(
                        127 + (owner + row % 8 + block) % 5 for block in range(start, start + 4)
                    )
                    packed.append(int.from_bytes(exponents, "little", signed=True))
            else:
                expected.append(torch.zeros(256, dtype=torch.float8_e4m3fn, device="cpu"))
                packed.extend([2139062143] * 2)
        assert torch.equal(
            out.cpu().view(torch.uint8), torch.stack(expected).reshape(5, 512).view(torch.uint8)
        )
        assert torch.equal(
            sf.cpu(), torch.tensor(packed, dtype=torch.int32, device="cpu").reshape(5, 4)
        )

    values = indices.cpu().flatten().tolist()
    verify(buf.fetch(indices), values)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = buf.fetch(indices)
    changed = [31, 24, 23, 16, 15, 8, 7, 0, 32, -1]
    indices.copy_(torch.tensor(changed, dtype=torch.int32, device="cuda").reshape_as(indices))
    graph.replay()
    torch.cuda.synchronize()
    verify(result, changed)
    Path(f"/tmp/engram-proof-{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "exact_embedding_bytes": True,
                "exact_packed_scales": True,
                "cuda_graph_changed_indices": True,
                "invalid_indices_zeroed": True,
            }
        )
    )
    buf.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=rank_main, args=(r,)) for r in range(4)]
    for p in processes:
        p.start()
    deadline = time.monotonic() + 150
    for p in processes:
        p.join(max(0, deadline - time.monotonic()))
    for p in processes:
        if p.is_alive():
            p.kill()
            p.join(5)
    assert all(p.exitcode == 0 for p in processes), [p.exitcode for p in processes]
    print(
        json.dumps(
            {
                "checks": [
                    json.loads(Path(f"/tmp/engram-proof-{r}.json").read_text()) for r in range(4)
                ]
            }
        ),
        flush=True,
    )

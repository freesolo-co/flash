"""End-to-end demo of the shared rollout pool (no GPU needed — fake vLLM backends).

Run from the repo root:  python examples/rollout_pool_demo.py

It boots a router + a small GPU fleet of fake vLLM servers + reward workers, then runs TWO GRPO
training runs CONCURRENTLY against the shared pool and prints:

  * which adapters landed on which GPU (multiple runs sharing one GPU = multi-LoRA),
  * how generation spread across the fleet (the nginx load-balancing),
  * the pool $/hr (rented once, shared by every run),
  * a serial-vs-pipelined wall-clock comparison showing reward latency overlapped away.

The fake backends speak vLLM's real dynamic-LoRA + OpenAI surface, so this is the same code path
the system uses against real vLLM.
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash.engine.pool_trainer import GRPOPoolLoop
from flash.pool.client import RolloutPoolClient
from tests._helpers.pool_harness import build_harness


def main() -> None:
    # A fleet: 2 GPUs serving base "org/Qwen-A", 1 GPU serving "org/Qwen-B". Small decode + reward
    # latency so the overlap is visible.
    h = build_harness(
        [
            {"id": "gpuA0", "base_model": "org/Qwen-A", "latency": 0.04, "cost_per_hour": 0.88},
            {"id": "gpuA1", "base_model": "org/Qwen-A", "latency": 0.04, "cost_per_hour": 0.88},
            {"id": "gpuB0", "base_model": "org/Qwen-B", "latency": 0.04, "cost_per_hour": 1.20},
        ],
        reward_workers=2,
        reward_latency=0.15,  # a deliberately SLOW reward
    )
    try:
        runs = [
            ("run-alpha", "org/Qwen-A"),
            ("run-beta", "org/Qwen-A"),  # second run on the SAME base -> shares the A GPUs
            ("run-gamma", "org/Qwen-B"),
        ]
        print(f"pool: 3 GPUs, ${h.status()['pool']['summary']['cost_per_hour']}/hr, "
              f"{len(runs)} concurrent runs sharing them\n")

        def train_one(name: str, base: str, train_time: float, steps: int) -> float:
            client = RolloutPoolClient(h.router_url, adapter=name, base_model=base)
            loop = GRPOPoolLoop(client, group_size=4, prefetch=2)
            loop.register(uri=f"/lora/{name}/v0", replicas=2 if base == "org/Qwen-A" else 1)
            batches = [[f"{name}-p{j}-{k}" for k in range(3)] for j in range(steps)]
            t0 = time.monotonic()
            loop.run(batches, lambda exp, adv, n=name: f"/lora/{n}/v{exp.step + 1}",
                     on_step=lambda r: time.sleep(train_time))
            client.close()
            return time.monotonic() - t0

        # Run all three concurrently (the realistic multi-tenant case).
        threads = [threading.Thread(target=train_one, args=(n, b, 0.10, 5)) for n, b in runs]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.monotonic() - t0

        print(f"3 runs x 5 steps finished in {wall:.1f}s (concurrent, sharing 3 GPUs)\n")
        print("per-GPU placement + traffic:")
        for bid in ("gpuA0", "gpuA1", "gpuB0"):
            rec = h.record(bid)
            print(f"  {bid:7s} base={rec.base_model:12s} adapters={sorted(rec.loaded)} "
                  f"chat_requests={len(rec.chat_calls)}")
        print("\n  -> runs alpha+beta share gpuA0/gpuA1 (multiple models on one GPU);")
        print("     generation load-balanced across the A GPUs; reward ran on 2 off-GPU workers.")

        # Serial vs pipelined for one run (shows reward latency overlapped away).
        print("\nreward-latency cost (one run, 5 steps, 0.15s reward, 0.10s 'train'):")
        client = RolloutPoolClient(h.router_url, adapter="run-alpha", base_model="org/Qwen-A")
        loop = GRPOPoolLoop(client, group_size=4, prefetch=2)
        batches = [[f"x{j}"] for j in range(5)]
        t0 = time.monotonic()
        loop.run(batches, lambda e, a: None, on_step=lambda r: time.sleep(0.10))
        pipelined = time.monotonic() - t0
        # serial baseline
        t0 = time.monotonic()
        for b in batches:
            comps = client.generate(b[0], n=4)
            client.score([b[0]] * len(comps), comps)
            time.sleep(0.10)
        serial = time.monotonic() - t0
        client.close()
        print(f"  serial (reward on critical path): {serial:.2f}s")
        print(f"  pipelined (reward overlapped):    {pipelined:.2f}s  "
              f"({(1 - pipelined / serial) * 100:.0f}% faster)")
    finally:
        h.stop()


if __name__ == "__main__":
    main()

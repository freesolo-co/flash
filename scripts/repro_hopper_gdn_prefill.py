"""Bounded GPU reproducer for the Hopper GDN prefill stall.

Run inside Flash's pinned H100 worker image with the verl interpreter:
    PYTHONPATH=. /opt/verl-venv/bin/python scripts/repro_hopper_gdn_prefill.py

Each trial uses synthetic tensors with the sequence lengths recovered from a
failed Qwen3.5-9B TP2 rollout's scheduler dump: 35 decodes and eight prefills.
This isolates the prefill implementation; it does not replay the entire trainer.
Compilation is warmed separately. Trial timeouts only count as kernel stalls
after warmup succeeds. Both backends run in fresh processes with identical seeds.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SEQUENCE_LENGTHS = [1] * 35 + [417, 301, 297, 301, 297, 101, 301, 301]


def trial(args):
    import torch
    import triton
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (
        fi_chunk_gated_delta_rule,
        fla_chunk_gated_delta_rule,
    )

    from flash.engine.worker.train.core.child.flashinfer_cache import activate_current

    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This reproducer requires a Hopper SM90 GPU")
    activate_current()
    triton.set_allocator(
        lambda size, _alignment, _stream: torch.empty(size, dtype=torch.int8, device="cuda")
    )
    torch.manual_seed(args.seed)
    tokens = sum(SEQUENCE_LENGTHS)
    cu = torch.tensor([0, *SEQUENCE_LENGTHS], device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    q, k = [torch.randn(1, tokens, 8, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    v = torch.randn(1, tokens, 16, 128, device="cuda", dtype=torch.bfloat16)
    gate = -torch.randn(1, tokens, 16, device="cuda", dtype=torch.float32).abs()
    beta = torch.rand(1, tokens, 16, device="cuda", dtype=torch.bfloat16).sigmoid()
    state = torch.randn(len(SEQUENCE_LENGTHS), 16, 128, 128, device="cuda") * 0.01
    fn = fi_chunk_gated_delta_rule if args.backend == "flashinfer" else fla_chunk_gated_delta_rule

    def invoke():
        return fn(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
        )

    invoke()
    torch.cuda.synchronize()
    print("WARMED", flush=True)
    if args.warmup:
        return
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        output, final_state = invoke()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    print("CAPTURED", flush=True)
    started = time.monotonic()
    for _ in range(args.replays):
        q.normal_()
        k.normal_()
        v.normal_()
        beta.uniform_().sigmoid_()
        state.normal_().mul_(0.01)
        graph.replay()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all() or not torch.isfinite(final_state).all():
        raise RuntimeError("Non-finite GDN output")
    print(json.dumps({"replays": args.replays, "seconds": time.monotonic() - started}), flush=True)


def run_child(args, backend, seed, *, warmup=False):
    label = f"{backend}-{seed}" + ("-warmup" if warmup else "")
    path = args.output / f"{label}.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--seed",
        str(seed),
        "--replays",
        str(args.replays),
    ]
    if warmup:
        command.append("--warmup")
    status = None
    with path.open("w") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        deadline = time.monotonic() + (900 if warmup else 180)
        ready_at = None
        try:
            while child.poll() is None:
                body = path.read_text()
                if not warmup and "WARMED\n" in body and ready_at is None:
                    ready_at = time.monotonic()
                if ready_at is not None and time.monotonic() - ready_at > args.timeout:
                    status = "kernel_timeout"
                    child.send_signal(signal.SIGUSR1)
                    time.sleep(1)
                    break
                if ready_at is None and time.monotonic() > deadline:
                    status = "setup_timeout"
                    break
                time.sleep(0.1)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    result = {
        "backend": backend,
        "seed": seed,
        "warmup": warmup,
        "status": status or ("passed" if child.returncode == 0 else "error"),
        "returncode": child.returncode,
        "log": str(path),
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["both", "flashinfer", "triton"], default="both")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, default=Path("results/hopper-gdn-repro"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--warmup", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.trials, args.replays, args.timeout) <= 0:
        parser.error("trials, replays, and timeout must be positive")
    if args.worker:
        if args.backend == "both":
            parser.error("a worker needs one backend")
        trial(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    backends = ["flashinfer", "triton"] if args.backend == "both" else [args.backend]
    for backend in backends:
        warm = run_child(args, backend, args.seed, warmup=True)
        results.append(warm)
        if warm["status"] == "passed":
            for seed in range(args.seed, args.seed + args.trials):
                results.append(run_child(args, backend, seed))
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(result["status"] != "passed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

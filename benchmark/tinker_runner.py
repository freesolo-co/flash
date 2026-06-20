"""Tinker GRPO runner for the benchmark.

Must be invoked with a Python that has `verifiers` installed, e.g.:
    /path/to/venv-with-verifiers/bin/python benchmark/tinker_runner.py

Runs Qwen3.5-4B GRPO on the GSM8K verifiers env via tinker_cookbook,
matched to the flash side: 30 steps, groups_per_batch=4, group_size=4,
max_tokens=512.

Writes a JSON result to --output (default /tmp/tinker_bench_result.json)
so the bench.py orchestrator can read it after the subprocess exits.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--groups-per-batch", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--env-id", default="gsm8k")
    p.add_argument("--log-path", default="/tmp/tinker-bench-gsm8k")
    p.add_argument("--output", default="/tmp/tinker_bench_result.json")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> dict:
    """Run Tinker GRPO training and return a metrics dict."""
    from tinker_cookbook.recipes.verifiers_rl.train import CLIConfig, cli_main

    cfg = CLIConfig(
        model_name=args.model,
        lora_rank=args.lora_rank,
        vf_env_id=args.env_id,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        max_steps=args.steps,
        # eval every 10 steps (disable with 0 to keep parity with flash eval cadence)
        eval_every=10,
        save_every=args.steps,
        log_path=args.log_path,
        behavior_if_log_dir_exists="delete",
        wandb_project=None,
    )

    t0 = time.monotonic()
    await cli_main(cfg, None)
    wall = time.monotonic() - t0
    return {"wall_s": wall}


def _read_metrics_jsonl(log_path: str) -> list[dict]:
    """Read Tinker's metrics.jsonl file from log_path."""
    import glob
    # Tinker writes metrics.jsonl directly in log_path
    candidates = glob.glob(os.path.join(log_path, "**", "metrics.jsonl"), recursive=True)
    candidates += [os.path.join(log_path, "metrics.jsonl")]
    for path in candidates:
        if os.path.exists(path):
            records = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            if records:
                return records
    return []


def main() -> None:
    args = parse_args()

    # Run the async training
    timing = asyncio.run(_run(args))

    # Extract reward trajectory from metrics.jsonl
    records = _read_metrics_jsonl(args.log_path)
    reward_history = [r.get("reward/total", 0.0) for r in records if "reward/total" in r]

    result = {
        "platform": "tinker",
        "status": "done",
        "model": args.model,
        "env_id": args.env_id,
        "steps": args.steps,
        "wall_s": timing["wall_s"],
        # Tinker does not expose billing via API; compute from time × list price.
        # H100 SXM at ~$3.50/hr is a representative Tinker training GPU.
        "cost_usd_estimated": round(timing["wall_s"] / 3600 * 3.50, 4),
        "cost_note": "estimated (Tinker does not expose cost via API; check your dashboard)",
        "first_train_reward": reward_history[0] if reward_history else None,
        "final_train_reward": reward_history[-1] if reward_history else None,
        "reward_history": reward_history,
        "log_path": args.log_path,
    }

    out = pathlib.Path(args.output)
    out.write_text(json.dumps(result, indent=2))
    print(f"[tinker] done — result written to {out}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

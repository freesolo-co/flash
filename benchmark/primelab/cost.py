#!/usr/bin/env python3
"""Prime Lab (Prime Intellect Hosted Training) cost-of-training estimator.

Prime Lab bills RL **per token** on three meters (confirmed live from
`prime train models` for Qwen/Qwen3.5-4B, June 2026):

    inference input   $0.10 / 1M tokens
    inference output  $0.30 / 1M tokens
    training          $0.30 / 1M tokens

That is the SAME billing model the launch banner prints:
    "Pricing (per 1M tokens, charged on actual usage)
       Training: $0.3   Inference Input: $0.1   Inference Output: $0.3"

Unlike Tinker (whose per-run $ is not API-exposed -> the benchmark uses a
GPU-time proxy), Prime Lab's bill IS exposed: after a run,
`prime train usage <run_id>` returns the exact token counts and price. We were
billing-gated (wallet $0), so this script estimates the cost analytically from
the GRPO config + Prime Lab's confirmed rates. Run the configs in ../primelab/
on a funded account and read the real number off `prime train usage`.

Token model (per run), GRPO with the benchmark matrix:
    rollouts R   = max_steps * batch_size * group_size      (= 30*4*4 = 480)
    input_tok    = prompt_len     * R     -> billed @ $0.10/Mtok (inference in)
    output_tok   = completion_len * R     -> billed @ $0.30/Mtok (inference out)
    train_tok    = completion_len * R     -> billed @ $0.30/Mtok (training)
                   (GRPO computes the loss on completion tokens -- the tokens
                    that "move the model"; prompt tokens are masked. If Prime
                    Lab's training meter instead counts the full prompt+
                    completion sequence, the training line is ~30% higher; see
                    --train-full.)

completion_len anchor: Flash's own trainer reports a throughput basis of
368,640 trained tokens for this exact config (= 480 * 768), i.e. ~768
completion tokens/rollout (= 0.75 * max_tokens=1024). gsm8k/hendrycks-math CoT
runs near that cap; reverse-text generations are short (the task just reverses a
string), so it gets a much smaller completion_len.
"""
from __future__ import annotations

import argparse
import json

# Confirmed-live Prime Lab rates for Qwen/Qwen3.5-4B ($ per 1,000,000 tokens).
RATE_IN = 0.10
RATE_OUT = 0.30
RATE_TRAIN = 0.30

# Benchmark matrix (identical to the Flash/Tinker runs).
MAX_STEPS = 30
BATCH_SIZE = 4          # prompts per step
GROUP_SIZE = 4          # rollouts_per_example (GRPO group)
MAX_TOKENS = 1024       # sampling cap

# Per-task token-length assumptions (documented in primelab_pricing.md).
#   prompt_len     : system + question tokens
#   completion_len : avg generated tokens/rollout (central estimate)
TASKS = {
    "gsm8k":          {"prompt_len": 256, "completion_len": 768},
    "hendrycks-math": {"prompt_len": 320, "completion_len": 768},
    "reverse-text":   {"prompt_len": 96,  "completion_len": 96},
}

# Measured Flash (RunPod billed) and Tinker (active-compute proxy) $ for context.
REFERENCE = {
    "gsm8k":          {"flash": 0.1636, "tinker": 0.8411},
    "reverse-text":   {"flash": 0.0651, "tinker": 0.8405},
    "hendrycks-math": {"flash": 0.2137, "tinker": 0.8272},
}


def rollouts() -> int:
    return MAX_STEPS * BATCH_SIZE * GROUP_SIZE


def estimate(prompt_len: int, completion_len: int, train_full: bool = False) -> dict:
    r = rollouts()
    input_tok = prompt_len * r
    output_tok = completion_len * r
    train_tok = (prompt_len + completion_len) * r if train_full else completion_len * r
    cost_in = input_tok / 1e6 * RATE_IN
    cost_out = output_tok / 1e6 * RATE_OUT
    cost_train = train_tok / 1e6 * RATE_TRAIN
    return {
        "input_tok": input_tok,
        "output_tok": output_tok,
        "train_tok": train_tok,
        "cost_in": cost_in,
        "cost_out": cost_out,
        "cost_train": cost_train,
        "cost_total": cost_in + cost_out + cost_train,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-full", action="store_true",
                    help="bill training on the full prompt+completion sequence "
                         "(upper bound) instead of completion-only")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    out = {}
    print(f"Prime Lab cost-of-training estimate  (Qwen3.5-4B, {MAX_STEPS} steps, "
          f"batch {BATCH_SIZE} x group {GROUP_SIZE} = {rollouts()} rollouts)")
    print(f"rates per 1M tok:  in ${RATE_IN}  out ${RATE_OUT}  train ${RATE_TRAIN}"
          f"   (train meter: {'full-seq' if args.train_full else 'completion-only'})\n")
    hdr = f"{'task':<16}{'central $':>11}{'@cap $':>10}{'Flash $':>10}{'Tinker $':>10}{'PL/Flash':>10}{'Tinker/PL':>11}"
    print(hdr)
    print("-" * len(hdr))
    for task, t in TASKS.items():
        central = estimate(t["prompt_len"], t["completion_len"], args.train_full)
        at_cap = estimate(t["prompt_len"], MAX_TOKENS, args.train_full)
        flash = REFERENCE[task]["flash"]
        tinker = REFERENCE[task]["tinker"]
        pl = central["cost_total"]
        print(f"{task:<16}{pl:>11.4f}{at_cap['cost_total']:>10.4f}"
              f"{flash:>10.4f}{tinker:>10.4f}{pl/flash:>9.2f}x{tinker/pl:>10.2f}x")
        out[task] = {
            "prime_lab_central_usd": round(pl, 4),
            "prime_lab_at_cap_usd": round(at_cap["cost_total"], 4),
            "flash_measured_usd": flash,
            "tinker_proxy_usd": tinker,
            "breakdown_central": {k: round(v, 5) if "cost" in k else v
                                   for k, v in central.items()},
            "assumptions": t,
        }
    if args.json:
        print("\n" + json.dumps({
            "model": "Qwen/Qwen3.5-4B",
            "rates_per_mtok": {"input": RATE_IN, "output": RATE_OUT, "training": RATE_TRAIN},
            "config": {"max_steps": MAX_STEPS, "batch_size": BATCH_SIZE,
                       "group_size": GROUP_SIZE, "max_tokens": MAX_TOKENS,
                       "rollouts": rollouts()},
            "train_meter": "full-seq" if args.train_full else "completion-only",
            "tasks": out,
        }, indent=2))


if __name__ == "__main__":
    main()

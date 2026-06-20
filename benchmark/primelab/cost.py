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

# Per-token rates for Qwen/Qwen3.5-4B ($ per 1,000,000 tokens).
# Prime Lab: confirmed live from `prime train models` (June 2026).
# Tinker:    published rate card at https://thinkingmachines.ai/tinker (prefill /
#            sample / train) -- Tinker ALSO bills per token, so the apples-to-
#            apples comparison is per-token vs per-token (NOT vs the GPU-time
#            proxy the original Flash-vs-Tinker run used). Tinker's card is a flat
#            ~2.2x Prime Lab across all three meters, so Tinker/PrimeLab ~= 2.2x
#            on every task regardless of the token-length assumptions below.
RATE_IN = 0.10
RATE_OUT = 0.30
RATE_TRAIN = 0.30

TINKER_IN = 0.22
TINKER_OUT = 0.67
TINKER_TRAIN = 0.67

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

# Flash = measured (RunPod billed). tinker_proxy = the original active-compute x
# $2/hr GPU-TIME proxy (kept only to show how much it overstated Tinker vs its
# real per-token card).
REFERENCE = {
    "gsm8k":          {"flash": 0.1636, "tinker_proxy": 0.8411},
    "reverse-text":   {"flash": 0.0651, "tinker_proxy": 0.8405},
    "hendrycks-math": {"flash": 0.2137, "tinker_proxy": 0.8272},
}


def rollouts() -> int:
    return MAX_STEPS * BATCH_SIZE * GROUP_SIZE


def estimate(prompt_len: int, completion_len: int,
             r_in: float, r_out: float, r_train: float,
             train_full: bool = False) -> dict:
    r = rollouts()
    input_tok = prompt_len * r
    output_tok = completion_len * r
    train_tok = (prompt_len + completion_len) * r if train_full else completion_len * r
    cost_in = input_tok / 1e6 * r_in
    cost_out = output_tok / 1e6 * r_out
    cost_train = train_tok / 1e6 * r_train
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
    meter = "full-seq" if args.train_full else "completion-only"
    print(f"Cost of training  (Qwen3.5-4B, {MAX_STEPS} steps, "
          f"batch {BATCH_SIZE} x group {GROUP_SIZE} = {rollouts()} rollouts, "
          f"train meter: {meter})")
    print(f"per-token rates /1M:  Tinker {TINKER_IN}/{TINKER_OUT}/{TINKER_TRAIN}"
          f"   PrimeLab {RATE_IN}/{RATE_OUT}/{RATE_TRAIN}   (in/out/train)\n")
    print("Apples-to-apples: both Tinker and Prime Lab bill PER TOKEN. Flash is a")
    print("measured GPU-rental bill (different model). tinker_proxy below is the OLD")
    print("GPU-time proxy ($2/hr) -- shown only to flag how much it overstated Tinker.\n")
    hdr = (f"{'task':<16}{'Flash(meas)':>12}{'Tinker/tok':>12}{'PrimeLab/tok':>13}"
           f"{'Tink/PL':>9}{'[tink_proxy]':>13}")
    print(hdr)
    print("-" * len(hdr))
    for task, t in TASKS.items():
        pl = estimate(t["prompt_len"], t["completion_len"],
                      RATE_IN, RATE_OUT, RATE_TRAIN, args.train_full)["cost_total"]
        tk = estimate(t["prompt_len"], t["completion_len"],
                      TINKER_IN, TINKER_OUT, TINKER_TRAIN, args.train_full)["cost_total"]
        flash = REFERENCE[task]["flash"]
        proxy = REFERENCE[task]["tinker_proxy"]
        print(f"{task:<16}{flash:>12.3f}{tk:>12.3f}{pl:>13.3f}"
              f"{tk/pl:>8.2f}x{proxy:>13.3f}")
        out[task] = {
            "flash_measured_usd": flash,
            "tinker_per_token_usd": round(tk, 4),
            "prime_lab_per_token_usd": round(pl, 4),
            "tinker_over_primelab": round(tk / pl, 2),
            "tinker_gpu_time_proxy_usd": proxy,
            "assumptions": t,
        }
    if args.json:
        print("\n" + json.dumps({
            "model": "Qwen/Qwen3.5-4B",
            "rates_per_mtok": {
                "tinker": {"input": TINKER_IN, "output": TINKER_OUT, "training": TINKER_TRAIN},
                "prime_lab": {"input": RATE_IN, "output": RATE_OUT, "training": RATE_TRAIN},
            },
            "config": {"max_steps": MAX_STEPS, "batch_size": BATCH_SIZE,
                       "group_size": GROUP_SIZE, "max_tokens": MAX_TOKENS,
                       "rollouts": rollouts()},
            "train_meter": meter,
            "tasks": out,
        }, indent=2))


if __name__ == "__main__":
    main()

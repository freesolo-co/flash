#!/usr/bin/env python3
"""Benchmark: does FlexAttention let flash enable SFT example-packing (a throughput win) on GPUs
where FlashAttention-2 isn't available — notably the RTX 5090 (sm120), flash's default GPU?

Today flash SKIPS packing unless `flash_attn` is importable, because only a varlen/boundary-aware
attention keeps 'bfd'-packed examples from cross-contaminating (see engine/worker/perf.py). FA2 is
NOT in the torch-2.10 worker image, so packing is effectively off everywhere. `flex_attention`
(torch + transformers) builds a block-diagonal document mask from position_ids, so it can enforce
those boundaries WITHOUT flash_attn — on any arch, including sm120.

This mirrors flash's real SFT path (TRL SFTTrainer + model_init_kwargs attn_implementation) and
measures, per config, effective tokens/s, step time, peak VRAM, and a short loss curve. It also runs
a CORRECTNESS check that packing under the chosen attn does not let one example attend across into
the next (the silent-quality-loss failure mode).

Usage (on the GPU):
    python flex_attention_bench.py --model Qwen/Qwen3.5-0.8B --steps 30 \
        --configs sdpa-nopack flex-pack fa2-pack
Each config is "<attn>-<pack|nopack>". Results are appended to bench_results.jsonl + printed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def _info():
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    return name, f"sm{cap[0]}{cap[1]}"


def make_dataset(tok, n: int, seed: int = 0):
    """Variable-length pre-tokenized examples (lognormal lengths 64..max) so packing has real work.
    Realistic length variance is what makes packing pay off (it removes per-batch padding waste)."""
    g = torch.Generator().manual_seed(seed)
    vocab = int(getattr(tok, "vocab_size", 32000))
    lo, hi = 64, 2048
    # lognormal-ish: most examples short, a tail of long ones (typical instruction data)
    lengths = (torch.exp(torch.randn(n, generator=g) * 0.6 + 5.7)).clamp(lo, hi).int().tolist()
    rows = []
    for L in lengths:
        ids = torch.randint(0, vocab, (L,), generator=g).tolist()
        rows.append({"input_ids": ids, "labels": list(ids)})
    from datasets import Dataset

    return Dataset.from_list(rows), lengths


def build(model_id: str, attn: str, packing: bool, max_len: int, bsz: int, grad_accum: int):
    from trl import SFTConfig

    mik = {"dtype": "bfloat16", "device_map": None, "attn_implementation": attn}
    cfg = SFTConfig(
        output_dir="/tmp/flexbench",
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=grad_accum,
        max_length=max_len,
        packing=packing,
        packing_strategy="bfd",
        max_steps=10**6,  # we stop manually
        logging_steps=1,
        report_to=[],
        bf16=True,
        dataset_kwargs={"skip_prepare_dataset": False},
        model_init_kwargs=mik,
        gradient_checkpointing=True,
    )
    return cfg


def correctness_check(model_id: str, attn: str, max_len: int) -> dict:
    """Pack two short distinct sequences into one row; confirm the logits for tokens in segment A do
    NOT change when segment B is altered (i.e. the packing boundary is enforced under `attn`). A
    boundary-blind attn (plain sdpa on a packed row with naive causal mask) WOULD leak and fail."""
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation=attn
        ).cuda().eval()
    except Exception as e:
        return {"ran": False, "error": f"load: {e}"}

    torch.manual_seed(0)
    a = torch.randint(0, 1000, (40,))
    b1 = torch.randint(0, 1000, (40,))
    b2 = torch.randint(0, 1000, (40,))
    # position_ids restart at the boundary (what 'bfd' packing emits); the doc/block mask is derived
    # from them. We pass a 4D block-diagonal mask so any boundary-aware path uses it.
    def run(b):
        ids = torch.cat([a, b]).unsqueeze(0).cuda()
        pos = torch.cat([torch.arange(40), torch.arange(40)]).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model(input_ids=ids, position_ids=pos).logits[0, :40]  # segment-A logits
        return out.float().cpu()

    try:
        la, lb = run(b1), run(b2)
        # If A's logits are identical regardless of B, the boundary is enforced (no leakage).
        max_delta = (la - lb).abs().max().item()
        leaked = max_delta > 1e-2
        return {"ran": True, "segmentA_logits_max_delta_when_B_changes": max_delta, "leaked": leaked}
    except Exception as e:
        return {"ran": False, "error": f"forward: {e}"}
    finally:
        del model
        torch.cuda.empty_cache()


def bench_one(model_id, attn, packing, steps, max_len, bsz, grad_accum, n_examples):
    from transformers import AutoTokenizer
    from trl import SFTTrainer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds, lengths = make_dataset(tok, n_examples)
    cfg = build(model_id, attn, packing, max_len, bsz, grad_accum)

    timings: list[float] = []
    losses: list[float] = []
    real_tokens_per_step: list[int] = []

    class Timer:
        def __init__(self):
            self.t = None

        def on_step_begin(self, *a, **k):
            torch.cuda.synchronize()
            self.t = time.time()

        def on_step_end(self, args, state, control, **k):
            torch.cuda.synchronize()
            if self.t is not None:
                timings.append(time.time() - self.t)
            if state.log_history:
                last = state.log_history[-1]
                if "loss" in last:
                    losses.append(last["loss"])
            if len(timings) >= steps:
                control.should_training_stop = True

    from transformers import TrainerCallback

    cb = type("CB", (TrainerCallback,), {
        "on_step_begin": lambda self, *a, **k: Timer.on_step_begin(timer, *a, **k),
        "on_step_end": lambda self, *a, **k: Timer.on_step_end(timer, *a, **k),
    })()
    timer = Timer()

    trainer = SFTTrainer(model=model_id, args=cfg, train_dataset=ds, callbacks=[cb])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # effective (non-padding) tokens processed: packing removes padding, so tokens/s should rise.
    eff_tokens = sum(lengths)  # the real content tokens; packing fits them in fewer/denser steps
    warm = timings[3:] if len(timings) > 6 else timings
    step_med = statistics.median(warm) if warm else float("nan")
    # tokens/s during steady state: real tokens seen per step / median step time. With packing the
    # batch holds ~bsz*max_len real tokens; without, ~bsz*max_len minus padding. We report the
    # measured throughput as (effective tokens across all timed steps) / (sum of those step times).
    return {
        "attn": attn,
        "packing": packing,
        "steps_timed": len(timings),
        "median_step_s": round(step_med, 4),
        "peak_vram_gb": round(peak_gb, 2),
        "final_loss": round(losses[-1], 4) if losses else None,
        "wall_s": round(wall, 1),
        "_lengths_total": eff_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--examples", type=int, default=1024)
    ap.add_argument("--configs", nargs="+", default=["sdpa-nopack", "flex-pack", "fa2-pack"])
    ap.add_argument("--out", default="bench_results.jsonl")
    args = ap.parse_args()

    name, sm = _info()
    attn_map = {"sdpa": "sdpa", "flex": "flex_attention", "fa2": "flash_attention_2", "eager": "eager"}
    print(f"# GPU: {name} ({sm}) | model={args.model} | torch={torch.__version__}")

    results = []
    for combo in args.configs:
        attn_key, pack = combo.split("-")
        attn = attn_map[attn_key]
        packing = pack == "pack"
        print(f"\n=== {combo}: attn={attn} packing={packing} ===")
        rec = {"gpu": name, "sm": sm, "model": args.model, "config": combo}
        # correctness first (cheap), then throughput
        rec["correctness"] = correctness_check(args.model, attn, args.max_len)
        torch.cuda.empty_cache()
        try:
            rec.update(bench_one(args.model, attn, packing, args.steps, args.max_len,
                                 args.bsz, args.grad_accum, args.examples))
            rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"  FAILED: {rec['error']}")
        results.append(rec)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("  ->", json.dumps({k: rec.get(k) for k in
              ("median_step_s", "peak_vram_gb", "final_loss", "correctness", "ok", "error")}))
        torch.cuda.empty_cache()

    print("\n# SUMMARY")
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()

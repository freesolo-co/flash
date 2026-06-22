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


def build(model_id: str, attn: str, packing: bool, max_len: int, bsz: int, grad_accum: int,
          steps: int):
    from trl import SFTConfig

    mik = {"dtype": "bfloat16", "device_map": None, "attn_implementation": attn}
    return SFTConfig(
        # Per-config output_dir so reusing the bench across configs in one run can't collide
        # ("output directory exists and is not empty"). (overwrite_output_dir was dropped from
        # SFTConfig in the worker's TRL/transformers-5 stack, so a unique dir is the portable fix.)
        output_dir=f"/tmp/flexbench/{attn}-{'pack' if packing else 'nopack'}",
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=grad_accum,
        max_length=max_len,
        packing=packing,
        packing_strategy="bfd",
        # `--steps` bounds the run: max_steps overrides num_train_epochs in the HF Trainer, so each
        # config trains EXACTLY `steps` optimizer steps (cheap + directly comparable). <=0 means
        # "unbounded" -> fall back to one full epoch over the dataset.
        num_train_epochs=1,
        max_steps=steps if steps > 0 else -1,
        logging_steps=5,
        report_to=[],
        bf16=True,
        dataset_kwargs={"skip_prepare_dataset": False},
        model_init_kwargs=mik,
        gradient_checkpointing=True,
    )


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
    # Cap token ids to the model's real vocab so tiny/toy models don't hit an embedding IndexError.
    vocab = min(1000, getattr(model.config, "vocab_size", 1000) or 1000)
    a = torch.randint(0, vocab, (40,))
    b1 = torch.randint(0, vocab, (40,))
    b2 = torch.randint(0, vocab, (40,))
    # position_ids restart at the boundary (what 'bfd' packing emits) — this is the ONLY boundary
    # signal we pass. A flex/varlen path derives its block-diagonal document mask from these restarts;
    # we deliberately do NOT pass an explicit 4D mask, mirroring flash's real packed SFT call.
    def run(m, b):
        ids = torch.cat([a, b]).unsqueeze(0).cuda()
        pos = torch.cat([torch.arange(40), torch.arange(40)]).unsqueeze(0).cuda()
        with torch.no_grad():
            out = m(input_ids=ids, position_ids=pos).logits[0, :40]  # segment-A logits
        return out.float().cpu()

    try:
        la, lb = run(model, b1), run(model, b2)
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
    ds, _ = make_dataset(tok, n_examples)
    cfg = build(model_id, attn, packing, max_len, bsz, grad_accum, steps)

    timings: list[float] = []
    step_tokens: list[int] = []  # ground-truth real (supervised) tokens per optimizer step
    losses: list[float] = []

    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        def on_step_begin(self, *a, **k):
            torch.cuda.synchronize()
            self._t = time.time()

        def on_step_end(self, args, state, control, **k):
            torch.cuda.synchronize()
            timings.append(time.time() - getattr(self, "_t", time.time()))
            if state.log_history and "loss" in state.log_history[-1]:
                losses.append(state.log_history[-1]["loss"])

    # Count REAL tokens per step from the COLLATED batch — the ground truth for BOTH modes: padded
    # / prompt positions are -100, packed positions are all-real. Counting in training_step (once
    # per optimizer step, BEFORE backward) means gradient-checkpointing's forward recompute can't
    # double-count, and packing's denser batches get credited correctly. The old metric estimated
    # real_tokens as mean_len * (n_steps*bsz), i.e. it counted batch ROWS as examples — right for
    # no-packing, but a packed row holds MANY examples, so it undercounted packed tokens ~Nx and
    # made packing look slower than it is. This counts what the model actually trained on.
    class CountingTrainer(SFTTrainer):
        def training_step(self, model, inputs, *targs, **tkw):
            labels = inputs.get("labels")
            step_tokens.append(int((labels != -100).sum().item()) if labels is not None else 0)
            return super().training_step(model, inputs, *targs, **tkw)

    # Mirror flash: LoRA adapters (not full fine-tune) keep optimizer memory tiny. (Attention impl
    # and packing — what we're measuring — are orthogonal to LoRA.)
    from peft import LoraConfig

    lora = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = CountingTrainer(
        model=model_id, args=cfg, train_dataset=ds, callbacks=[StepTimer()], peft_config=lora
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # THE decision metric: real (non-padding) supervised tokens / sec, measured on a STEADY-STATE
    # window. Drop the first few steps: flex_attention compiles its block-mask kernel on the first
    # call (torch.compile), so the opening steps are dramatically slower and would otherwise tax
    # flex-pack for a one-time cost the real (multi-thousand-step) SFT run amortizes to nothing.
    # real_tokens is the ground-truth sum of label!=-100 tokens over exactly those timed steps, so
    # packed (dense) and unpacked (padded) batches are compared on the SAME basis. Higher = better.
    n_steps = len(timings)
    warm = 5 if n_steps > 8 else (2 if n_steps > 4 else 0)
    st = timings[warm:] or timings
    tk = step_tokens[warm:warm + len(st)]
    real_tokens = sum(tk)
    train_s = sum(st)
    step_med = statistics.median(st) if st else float("nan")
    return {
        "attn": attn,
        "packing": packing,
        "steps": n_steps,
        "steady_steps": len(st),
        "real_tokens": real_tokens,
        "train_s": round(train_s, 2),
        "real_tokens_per_s": round(real_tokens / train_s) if (train_s and real_tokens) else None,
        "median_step_s": round(step_med, 4),
        "peak_vram_gb": round(peak_gb, 2),
        "final_loss": round(losses[-1], 4) if losses else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bsz", type=int, default=8)  # >1 so no-packing pays the intra-batch padding tax
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--examples", type=int, default=768)
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
              ("real_tokens_per_s", "train_s", "steps", "median_step_s", "peak_vram_gb",
               "final_loss", "ok", "error")}))
        torch.cuda.empty_cache()

    print("\n# SUMMARY")
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()

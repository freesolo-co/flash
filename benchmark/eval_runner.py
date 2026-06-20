"""Unified held-out eval for the Tinker side (base vs trained), GSM8K.

The in-training GRPO reward is NOT comparable across stacks (different verifiers
versions + task presentation; e.g. matched max_tokens truncates Tinker's verbose
answers before the boxed result). This script removes those confounds for the
Tinker models: greedy decoding, generous max_tokens, and ONE version-independent
exact-match scorer applied identically to base and trained.

It answers: did Tinker GRPO actually improve held-out GSM8K accuracy?

  base Qwen3.5-4B        -> Tinker sampling (base_model)
  Tinker-trained (final) -> Tinker sampling (sampler_path)
  score: extract \\boxed{} / last number, exact-match the gold answer.

Usage (needs TINKER_API_KEY + an interpreter with tinker + verifiers):
    /usr/bin/python3 benchmark/eval_runner.py --env-id gsm8k --n 50 \
        --sampler "tinker://.../sampler_weights/final" \
        --out benchmark/results/eval_tinker_gsm8k.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

BASE_MODEL = "Qwen/Qwen3.5-4B"


def _normalize(s: str) -> str:
    """Canonicalize a numeric answer the SAME way for both prediction and gold.

    Drops thousands separators, currency/percent symbols, surrounding whitespace, and a
    trailing period, so exact-match isn't defeated by cosmetic differences (e.g. the gold
    keeping a trailing "." or a "%" the prediction stripped).
    """
    return s.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")


def extract_answer(text: str) -> str:
    """Version-independent answer extraction: last \\boxed{...} else last number."""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    cand = boxed[-1] if boxed else ""
    if not cand:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        cand = nums[-1] if nums else ""
    return _normalize(cand)


def gold_answer(row: dict) -> str:
    a = str(row.get("answer", ""))
    if "####" in a:
        a = a.split("####")[-1]
    return _normalize(a)


def eval_one(env_rows: list, *, base_model=None, sampler_path=None, n=50, max_tokens=1024) -> dict:
    import tinker
    from tinker import types
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    sc = tinker.ServiceClient()
    if sampler_path:
        sampling = sc.create_sampling_client(model_path=sampler_path)
    else:
        sampling = sc.create_sampling_client(base_model=base_model)

    tokenizer = get_tokenizer(BASE_MODEL)
    rname = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(rname, tokenizer)
    params = types.SamplingParams(
        max_tokens=max_tokens, temperature=0.0,
        stop=list(renderer.get_stop_sequences()),
    )

    correct = 0
    truncated = 0
    rows = env_rows[:n]
    for i, row in enumerate(rows):
        messages = row["prompt"]  # chat messages (system + user)
        model_input = renderer.build_generation_prompt(messages)
        result = sampling.sample(
            prompt=model_input, num_samples=1, sampling_params=params
        ).result(timeout=600)
        toks = list(result.sequences[0].tokens)
        text = tokenizer.decode(toks)
        if len(toks) >= max_tokens:
            truncated += 1
        pred = extract_answer(text)
        if pred and pred == gold_answer(row):
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(rows)} acc={correct/(i+1):.3f}", flush=True)
    return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "truncated_frac": truncated / len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="gsm8k")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--sampler", required=True, help="tinker:// sampler_path of the trained model")
    ap.add_argument("--out", default="benchmark/results/eval_tinker_gsm8k.json")
    args = ap.parse_args()

    import verifiers as vf
    env = vf.load_environment(args.env_id)
    ds = env.eval_dataset if getattr(env, "eval_dataset", None) is not None else env.dataset
    rows = [ds[i] for i in range(min(args.n, len(ds)))]

    print(f"[eval] {args.env_id}: base vs trained on {len(rows)} held-out examples "
          f"(greedy, max_tokens={args.max_tokens})")
    print("[eval] base Qwen3.5-4B ...")
    base = eval_one(rows, base_model=BASE_MODEL, n=args.n, max_tokens=args.max_tokens)
    print(f"[eval]   base accuracy: {base['accuracy']:.3f}")
    print("[eval] Tinker-trained ...")
    trained = eval_one(rows, sampler_path=args.sampler, n=args.n, max_tokens=args.max_tokens)
    print(f"[eval]   trained accuracy: {trained['accuracy']:.3f}")

    out = {
        "env_id": args.env_id, "model": BASE_MODEL, "max_tokens": args.max_tokens,
        "scorer": "exact-match on \\boxed{}/last-number (version-independent)",
        "base": base, "trained": trained,
        "delta": round(trained["accuracy"] - base["accuracy"], 4),
    }
    p = pathlib.Path(args.out)
    p.write_text(json.dumps(out, indent=2))
    print(f"[eval] wrote {p}: base={base['accuracy']:.3f} trained={trained['accuracy']:.3f} "
          f"delta={out['delta']:+.3f}")


if __name__ == "__main__":
    main()

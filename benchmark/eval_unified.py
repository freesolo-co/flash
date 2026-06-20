"""Unified three-way held-out eval: base vs Flash-trained vs Tinker-trained, ONE scorer.

This is the only valid cross-stack PERFORMANCE comparison. All three models generate on the
SAME held-out examples with identical decoding (greedy, max_tokens), and the SAME
version-independent exact-match scorer (from eval_runner) grades every completion:

  base Qwen3.5-4B   -> Tinker sampling (base_model)
  Tinker-trained    -> Tinker sampling (sampler_path)
  Flash-trained     -> Flash's Modal LoRA serving (POST /v1/chat/completions)

The Flash adapter must be deployed first (``slm deploy <run_id>`` registers it with the serving);
pass its served model id + serve URL here. Base/Tinker need TINKER_API_KEY + an interpreter with
tinker + verifiers; Flash needs only HTTP to the serving.

Usage:
    /usr/bin/python3 benchmark/eval_unified.py --env-id gsm8k --n 50 \
        --tinker-sampler "tinker://.../sampler_weights/final" \
        --flash-serve-url https://clado-ai--freesolo-lora-serving.modal.run \
        --flash-model <adapter_id> \
        --out benchmark/results/eval_unified_gsm8k.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from eval_runner import BASE_MODEL, eval_one, extract_answer, gold_answer


def _load_rows(env_id: str, n: int) -> list[dict]:
    import verifiers as vf
    env = vf.load_environment(env_id)
    ds = env.eval_dataset if getattr(env, "eval_dataset", None) is not None else env.dataset
    return [ds[i] for i in range(min(n, len(ds)))]


def _renderer_stop_sequences() -> list[str]:
    """The SAME stop sequences eval_runner.eval_one passes to the Tinker sampler.

    eval_one builds the BASE_MODEL chat renderer and decodes with
    ``stop=list(renderer.get_stop_sequences())`` (e.g. the chat turn/EOS delimiters). The Flash
    serving must terminate on the identical contract, otherwise its generations run past the
    answer while the Tinker side stops — biasing the cross-stack comparison. We rebuild the same
    renderer here so both paths share one source of truth for where a generation ends. The
    tinker_cookbook import is function-local so the (mocked) HTTP test never needs it.
    """
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(BASE_MODEL)
    rname = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(rname, tokenizer)
    return list(renderer.get_stop_sequences())


def eval_via_serving(
    rows: list[dict], serve_url: str, model: str, max_tokens: int,
    stop: list[str] | None = None,
) -> dict:
    """Generate from a deployed Flash LoRA via the serving's OpenAI chat endpoint, then score.

    ``stop`` mirrors the renderer stop sequences eval_one uses for base/Tinker, so the Flash
    generations terminate under the SAME contract (identical greedy decoding AND stop set). It is
    forwarded as the OpenAI ``stop`` param only when non-empty (an empty list would be a no-op /
    rejected by some servers).
    """
    url = serve_url.rstrip("/") + "/v1/chat/completions"
    correct = truncated = errors = 0
    for i, row in enumerate(rows):
        messages = row["prompt"]  # [system, user] — same prompt the env feeds every stack
        body = {
            "model": model, "messages": messages,
            "temperature": 0.0, "max_tokens": max_tokens,
        }
        if stop:
            body["stop"] = stop  # same generation-termination contract as the Tinker eval
        payload = json.dumps(body).encode()
        text = ""
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    url, data=payload, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = json.loads(resp.read())
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""
                if choice.get("finish_reason") == "length":
                    truncated += 1
                break
            except Exception as exc:
                if attempt == 3:
                    errors += 1
                    print(f"    serving error ex {i}: {repr(exc)[:80]}", flush=True)
                else:
                    time.sleep(5.0 * (attempt + 1))
        if text and extract_answer(text) == gold_answer(row):
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"    flash {i+1}/{len(rows)} acc={correct/(i+1):.3f}", flush=True)
    n = len(rows)
    return {"n": n, "correct": correct, "accuracy": correct / n,
            "truncated_frac": truncated / n, "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="gsm8k")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tinker-sampler", default=None, help="tinker:// sampler_path of the Tinker-trained model")
    ap.add_argument("--flash-serve-url", default=None)
    ap.add_argument("--flash-model", default=None, help="served adapter/model id on the Flash serving")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--out", default="benchmark/results/eval_unified_gsm8k.json")
    args = ap.parse_args()

    rows = _load_rows(args.env_id, args.n)
    print(f"[unified-eval] {args.env_id}: {len(rows)} held-out, greedy, max_tokens={args.max_tokens}, "
          f"one exact-match scorer")
    out: dict = {"env_id": args.env_id, "model": BASE_MODEL, "n": len(rows),
                 "max_tokens": args.max_tokens,
                 "scorer": "exact-match on \\boxed{}/last-number (version-independent)"}

    if not args.skip_base:
        print("[unified-eval] base (Tinker sampling) ...")
        out["base"] = eval_one(rows, base_model=BASE_MODEL, n=args.n, max_tokens=args.max_tokens)
        print(f"  base: {out['base']['accuracy']:.3f}")

    if args.tinker_sampler:
        print("[unified-eval] Tinker-trained (Tinker sampling) ...")
        out["tinker_trained"] = eval_one(
            rows, sampler_path=args.tinker_sampler, n=args.n, max_tokens=args.max_tokens
        )
        print(f"  tinker-trained: {out['tinker_trained']['accuracy']:.3f}")

    if args.flash_serve_url and args.flash_model:
        print("[unified-eval] Flash-trained (Modal serving) ...")
        # Forward the SAME stop sequences the base/Tinker eval uses, so all three paths share one
        # generation-termination contract (identical greedy decoding AND stop set).
        stop = _renderer_stop_sequences()
        out["stop_sequences"] = stop
        out["flash_trained"] = eval_via_serving(
            rows, args.flash_serve_url, args.flash_model, args.max_tokens, stop=stop
        )
        print(f"  flash-trained: {out['flash_trained']['accuracy']:.3f}")

    base_acc = out.get("base", {}).get("accuracy")
    for key in ("flash_trained", "tinker_trained"):
        if key in out and base_acc is not None:
            out[key]["delta_vs_base"] = round(out[key]["accuracy"] - base_acc, 4)

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[unified-eval] wrote {args.out}")
    for k in ("base", "flash_trained", "tinker_trained"):
        if k in out:
            d = out[k]
            print(f"  {k:16} acc={d['accuracy']:.3f}" + (f" (Δ{d['delta_vs_base']:+.3f})"
                  if "delta_vs_base" in d else ""))


if __name__ == "__main__":
    main()

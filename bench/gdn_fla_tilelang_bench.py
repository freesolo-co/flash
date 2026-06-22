#!/usr/bin/env python3
"""A/B benchmark for PR#32: does fla's tilelang GatedDeltaNet backend actually beat the
pure-PyTorch delta-rule fallback on Hopper (sm90)?

Context: on Hopper with Triton>=3.4 fla's gated chunk_bwd Triton kernel is miscomputed and
HARD-RAISES (fla #640). The worker USED to drop fla -> transformers' pure-PyTorch delta rule
(correct, slow). PR#32 instead installs fla's **tilelang** backend (correct on Triton>=3.4) and
keeps fla. transformers gates GDN on is_fla_available(), so:

  * mode 'off'      -> fla physically removed -> pure-PyTorch delta rule  (the OLD fallback)
  * mode 'tilelang' -> fla + pinned tilelang present -> fla tilelang GDN  (the NEW fast path)

Run BOTH as separate processes (fla import state is sticky), then compare ms/step + peak VRAM:
    FLA_MODE=off      python gdn_fla_tilelang_bench.py
    FLA_MODE=tilelang python gdn_fla_tilelang_bench.py

Higher speedup (off_ms / tilelang_ms) and lower peak mem => PR#32's win is real on this host.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

TILELANG_PIN = "0.1.11"
TVM_FFI_PIN = "0.1.11"
FLA_GIT = "git+https://github.com/fla-org/flash-linear-attention.git@f0e213dbd8b5fb90c3c7eca869ac1706d5377139"


def _run(*a: str) -> int:
    return subprocess.run([sys.executable, "-m", "pip", *a], check=False).returncode


def _remove_fla() -> None:
    """Physically remove fla so is_fla_available() is False -> transformers native torch delta."""
    import importlib
    import importlib.util
    import shutil

    _run("uninstall", "-y", "-q", "flash-linear-attention")
    for _ in range(6):
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("fla")
        if spec is None:
            break
        locs = list(getattr(spec, "submodule_search_locations", []) or [])
        if spec.origin:
            locs.append(os.path.dirname(spec.origin))
        gone = False
        for loc in locs:
            if loc and os.path.isdir(loc) and os.path.basename(loc.rstrip("/")) == "fla":
                shutil.rmtree(loc, ignore_errors=True)
                gone = True
        if not gone:
            break
    importlib.invalidate_caches()


def _ensure_tilelang() -> None:
    import importlib
    import importlib.metadata as md
    import importlib.util  # find_spec() below lives in importlib.util, not bare importlib

    def ver(d):
        try:
            return md.version(d)
        except Exception:
            return None

    if ver("tilelang") != TILELANG_PIN:
        _run("install", "-q", f"tilelang=={TILELANG_PIN}")
    if ver("apache-tvm-ffi") != TVM_FFI_PIN:
        _run("install", "-q", f"apache-tvm-ffi=={TVM_FFI_PIN}")
    if importlib.util.find_spec("fla") is None or importlib.util.find_spec("fla.modules") is None:
        _run("install", "-q", "--no-deps", FLA_GIT)
    importlib.invalidate_caches()


def main() -> None:
    mode = os.environ.get("FLA_MODE", "tilelang")
    model_id = os.environ.get("MODEL", "Qwen/Qwen3.5-0.8B")
    seqs = [int(s) for s in os.environ.get("SEQS", "4096,8192,16384").split(",")]
    warmup, iters = 2, 6

    if mode == "off":
        _remove_fla()
    else:
        _ensure_tilelang()

    import torch

    # Report the resolved backend state so the comparison is auditable.
    try:
        from transformers.utils.import_utils import is_fla_available

        fla_ok = bool(is_fla_available())
    except Exception:
        import importlib.util as _u

        fla_ok = _u.find_spec("fla") is not None
    tl = None
    try:
        import importlib.metadata as md

        import tilelang  # noqa: F401

        tl = md.version("tilelang")
    except Exception:
        tl = None

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(
        f"[gdn-bench] mode={mode} gpu={name} sm{cap[0]}{cap[1]} "
        f"is_fla_available={fla_ok} tilelang={tl} torch={torch.__version__}",
        flush=True,
    )

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    model.train()
    model.gradient_checkpointing_enable()
    vocab = model.config.vocab_size

    results = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for L in seqs:
        rec = {"seq": L}
        try:
            ids = torch.randint(0, vocab, (1, L), generator=g).cuda()
            torch.cuda.reset_peak_memory_stats()

            def step(ids=ids):
                out = model(input_ids=ids, labels=ids)
                out.loss.backward()
                model.zero_grad(set_to_none=True)
                return float(out.loss.detach())

            loss = 0.0
            for _ in range(warmup):
                loss = step()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                loss = step()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / iters * 1000.0
            rec.update(
                ms_per_step=round(ms, 1),
                peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                loss=round(loss, 5),
            )
            print(f"[gdn-bench] seq={L}: {rec['ms_per_step']} ms/step  "
                  f"peak={rec['peak_gb']} GB  loss={rec['loss']}", flush=True)
        except Exception as e:  # capture (e.g. fla #640 hard-raise) instead of aborting the sweep
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[gdn-bench] seq={L}: ERROR {rec['error']}", flush=True)
            torch.cuda.empty_cache()
        results.append(rec)

    print("RESULT_JSON " + json.dumps({"mode": mode, "model": model_id, "gpu": name,
                                       "sm": f"{cap[0]}{cap[1]}", "is_fla_available": fla_ok,
                                       "tilelang": tl, "results": results}), flush=True)


if __name__ == "__main__":
    main()

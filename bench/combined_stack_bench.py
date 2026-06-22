#!/usr/bin/env python3
"""Stacked A/B: the FULL old worker stack vs the FULL new worker stack on one Qwen3.5 SFT step.

The combined PR changes two things that both touch a Qwen3.5 training step on Hopper:
  * GDN backend:  drop-fla -> pure-PyTorch delta   (OLD)   ->   fla + tilelang   (NEW, PR#32)
  * fused kernels: Liger (rms/swiglu/FLCE)          (OLD)   ->   chalk standalone (NEW, PR#60)
This measures them ADDED TOGETHER — one fwd+bwd step on a real Qwen3.5 (real dims H=4096,
V=248320, a hybrid linear+full layer mix) under each complete stack:

  STACK=old  -> fla physically removed (pure-PyTorch GDN) + chalk apply liger=True  (Liger kernels)
  STACK=new  -> fla+tilelang ensured (fast GDN)           + chalk apply liger=False (chalk-only)

Run BOTH as separate processes (fla state + kernel class-patches are global/sticky), then compare:
    STACK=old python combined_stack_bench.py
    STACK=new python combined_stack_bench.py
old_ms / new_ms = the total worker speedup the combined PR delivers for a flagship Qwen3.5 step.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time

TILELANG_PIN = "0.1.11"
TVM_FFI_PIN = "0.1.11"
FLA_GIT = "git+https://github.com/fla-org/flash-linear-attention.git@f0e213dbd8b5fb90c3c7eca869ac1706d5377139"


def _pip(*args: str) -> int:
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", *args], check=False
    ).returncode


def _remove_fla() -> None:
    import shutil

    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "--break-system-packages",
         "flash-linear-attention"], check=False
    )
    for _ in range(6):
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("fla")
        if spec is None:
            break
        locs = list(getattr(spec, "submodule_search_locations", []) or [])
        if spec.origin:
            locs.append(os.path.dirname(spec.origin))
        if not any(os.path.isdir(p) and os.path.basename(p.rstrip("/")) == "fla" for p in locs):
            break
        for loc in locs:
            if loc and os.path.isdir(loc) and os.path.basename(loc.rstrip("/")) == "fla":
                shutil.rmtree(loc, ignore_errors=True)
    importlib.invalidate_caches()


def _ensure_tilelang() -> None:
    import importlib.metadata as md

    def ver(d):
        try:
            return md.version(d)
        except Exception:
            return None

    if ver("tilelang") != TILELANG_PIN:
        _pip(f"tilelang=={TILELANG_PIN}")
    if ver("apache-tvm-ffi") != TVM_FFI_PIN:
        _pip(f"apache-tvm-ffi=={TVM_FFI_PIN}")
    if importlib.util.find_spec("fla") is None or importlib.util.find_spec("fla.modules") is None:
        _pip("--no-deps", FLA_GIT)
    importlib.invalidate_caches()


def main() -> None:
    stack = os.environ.get("STACK", "new")
    layers = int(os.environ.get("LAYERS", "4"))
    seq = int(os.environ.get("SEQ", "2048"))
    warmup, iters = 4, 8

    if stack == "old":
        _remove_fla()  # -> transformers pure-PyTorch GDN delta rule
    else:
        _ensure_tilelang()  # -> fla tilelang GDN fast path

    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    try:
        from transformers.utils.import_utils import is_fla_available

        fla_ok = bool(is_fla_available())
    except Exception:
        fla_ok = importlib.util.find_spec("fla") is not None

    base_pattern = ["linear_attention"] * 3 + ["full_attention"]
    cfg = Qwen3_5TextConfig(
        hidden_size=4096, intermediate_size=12288, vocab_size=248320, num_hidden_layers=layers,
        layer_types=[base_pattern[i % 4] for i in range(layers)],
        num_attention_heads=16, num_key_value_heads=4, head_dim=256,
    )
    cfg._attn_implementation = "sdpa"
    model = Qwen3_5ForCausalLM(cfg).cuda().to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)

    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, target_modules=["gate_proj", "up_proj", "down_proj"],
                   bias="none", task_type="CAUSAL_LM"),
    )
    inner = model
    while hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
        inner = inner.base_model.model
        break

    # Apply the fused-kernel stack: Liger-composed (old) or chalk standalone (new).
    from chalk.transformers import apply_chalk_kernel_to_qwen35

    report = apply_chalk_kernel_to_qwen35(inner, liger=(stack == "old"))

    name = torch.cuda.get_device_name(0)
    print(f"[combined] STACK={stack} gpu={name} is_fla_available={fla_ok} "
          f"kernels={ {k: v for k, v in report.items() if k != 'liger'} }", flush=True)

    ids = torch.randint(0, cfg.vocab_size, (1, seq), device="cuda")

    def step():
        out = model(input_ids=ids, labels=ids, use_cache=False)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        return float(out.loss.detach())

    loss = 0.0
    for _ in range(warmup):
        loss = step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    print("RESULT_JSON " + json.dumps({
        "stack": stack, "seq": seq, "layers": layers, "gpu": name, "is_fla_available": fla_ok,
        "ms_per_step": round(statistics.median(times), 1),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "loss": round(loss, 5),
    }), flush=True)


if __name__ == "__main__":
    main()

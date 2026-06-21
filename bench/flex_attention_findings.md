# FlexAttention for flash SFT packing — benchmark + decision

**Question:** would adding `attn_implementation="flex_attention"` help flash? **Answer: conditionally yes** — it enables SFT example-packing (a measured ~1.7× throughput win) on standard-decoder models, but **not** on flash's flagship Qwen3.5/3.6 tier (transformers doesn't support flex for that arch yet).

## Why it could help

flash SFT enables TRL `packing` (strategy `bfd`) **only when `flash_attn` is importable** — because packed examples cross-contaminate unless the attention honors example boundaries (FA2 varlen *or* flex_attention's block-diagonal doc mask). FA2 is **not** in the torch-2.10 worker image and **doesn't exist for sm120** (the RTX 5090 — flash's *default* GPU). So **packing is effectively off everywhere today.** FlexAttention builds the boundary mask without flash-attn, so it could turn packing back on.

## Measured (real GPU)

Rented an **RTX 5090 (sm120)** on Vast; flash's stack (torch 2.10 cu128 + transformers 5.10.4 + trl 1.6), LoRA + gradient-checkpointing, one epoch over 768 variable-length examples, bsz=4, max_len=1024. Metric = **real (non-padding) content tokens/sec** (same examples per config; only wall time differs).

### Qwen2.5-0.5B (standard `Qwen2ForCausalLM`)

| config | real tok/s | vs baseline | steps | peak VRAM |
|---|---|---|---|---|
| `sdpa`, no packing — *flash today* | 6,785 | 1.00× | 192 | 9.02 GB |
| **`flex_attention`, packing** | **11,553** | **1.70×** | 69 | 8.78 GB |
| `sdpa`, packing — *leaky control* | 14,439 | 2.13× | 69 | 8.80 GB |

**FlexAttention packing = 1.70× throughput at equal/lower VRAM.** Packing collapses 192 padded steps → 69 dense ones. The `sdpa`-packing control (2.13×) is the boundary-*incorrect* upper bound; flex's per-call overhead (incl. first-call `torch.compile`) costs the gap, but a large win remains.

### Qwen3.5-0.8B (flash flagship, hybrid-GDN `Qwen3_5ForConditionalGeneration`)

```
ValueError: Qwen3_5ForConditionalGeneration does not support an attention
implementation through torch's flex_attention   (HF transformers #34809)
```

**flex_attention is NOT supported for the Qwen3.5/3.6 architecture** in transformers 5.x. So the win does not reach flash's flagship/default tier — and FA2 isn't available there either, so packing stays off for Qwen3.5 regardless until upstream support lands.

### Per-arch support (probed via `_supports_flex_attn`)

| flash catalog model | arch | flex? |
|---|---|---|
| `openbmb/MiniCPM5-1B` | LlamaForCausalLM | ✅ |
| `Qwen/Qwen2.5-*` (ref) | Qwen2ForCausalLM | ✅ |
| `Qwen/Qwen3.5-*`, `Qwen3.6-*` | Qwen3_5ForConditionalGeneration | ❌ |

## Decision & change

Worth adding as a **gated** option: enable `flex_attention` + packing **only for arches that support it** (`model_supports_flex_attn`). Net effect:
- **MiniCPM5-1B (Llama) tier → packing on → ~1.7× SFT.**
- **Qwen3.5/3.6 tier → unchanged** (flex unsupported → packing stays off, as today).
- **Forward-compatible:** when transformers adds Qwen3.5 flex support (#34809), flash gets the win automatically — no further change.

Implemented in `engine/worker/perf.py::model_supports_flex_attn` + the SFT packing gate in `engine/worker/__init__.py` (CPU-tested in `tests/test_worker_stack.py`). Cost of this investigation: ~$0.40 of 5090 time.

## Reproduce

`bench/flex_attention_bench.py` on any GPU with the flash stack:
```
python flex_attention_bench.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --bsz 4 --max-len 1024 --examples 768 --configs sdpa-nopack flex-pack sdpa-pack
```

# Pool trainer — optimization parity with origin/dev, + 35B and the model→GPU matrix

The pool's training server (`flash/engine/pool_policy.py`) applies the **same optimization stack as
origin/dev's worker**, by calling **dev's own helpers** (so there's one implementation, not a fork):

| optimization | how the pool applies it | dev source it reuses |
|---|---|---|
| **Liger** kernels (fused RMSNorm/RoPE/SwiGLU) | `_apply_liger_kernel_to_instance(model)` after build | dev uses TRL `use_liger_kernel`; gated by `perf.liger_on` defaults |
| **Chalk** kernels (Qwen3.5/3.6 fused MLP/RoPE/embedding/LoRA-delta) | `install_chalk_kernels(model)` | `flash.engine.chalk_kernels.install_chalk_kernels` (identical call) |
| **FLA-drop** (Qwen3.5/3.6 GDN on Hopper) | `_drop_fla_on_hopper()` before load | `flash.engine.worker.perf._drop_fla_on_hopper` |
| **QLoRA** (4-bit NF4 base) | `BitsAndBytesConfig` + `prepare_model_for_kbit_training` | tier from `lora.model_quant` (catalog 4-bit) |
| **8-bit paged AdamW** | `loraplus_optimizer_cls(fused_optim_name())` → `PagedAdamW8bit` | `flash.engine.worker.perf` (identical) |

Each is **resolved per model** by `resolve_opt_config(model_id)` with dev's defaults (Liger/Chalk/
8-bit on; FLA-drop on for Qwen3.5/3.6; QLoRA on for the catalog's 4-bit tiers — 9B and 35B-A3B), and
every flag is overridable. Each is a no-op off-GPU / where inapplicable (Chalk no-ops on non-Qwen3.5,
FLA-drop on non-Qwen3.5, QLoRA off-GPU), exactly like dev.

## Does it work with 35B? Yes — and *better* than dev's colocate

`Qwen/Qwen3.6-35B-A3B` is the catalog's 4-bit-QLoRA MoE. dev runs it **colocated** (trainer + vLLM on
one card) → **~103 GB → H200-only**. The pool **disaggregates**, so the trainer carries no vLLM:

- **trainer** = QLoRA 4-bit base (~20 GB) → **fits a single A40 (48 GB, ~$0.44/hr)**, not an H200.
- **rollout** = 35B vLLM (expert-parallel) on the shared inference pool — the only piece that needs a
  big card, and it's shared across runs.

So the pool makes 35B trainable on ordinary hardware where dev's colocate cannot. `pool_gpu_plan()`
returns the (trainer, inference) GPU pick per model:

```
model                 qlora  trainer GPU (vram)     inference GPU (vram)
Qwen3.5-0.8B          no     small card (~7 GB)     small card
Qwen3.5-2B            no     small card (~10 GB)    small card
Qwen3.5-4B            no     ~14 GB                 ~18 GB
Qwen3.5-9B            yes    ~11 GB (4-bit)         ~17 GB
Qwen3.6-35B-A3B       yes    A40 ~29 GB (4-bit)     big card + expert-parallel
MiniCPM5-1B           no     small card             small card
```
(VRAM is an estimate; the live allocator calibrates. The point: every model maps to a GPU and 35B's
trainer fits an ordinary card.)

## Verification

- **CPU-tested** (`tests/test_pool_optimizations.py`): the same optimization SET as dev is selected
  per model (Liger/Chalk/FLA-drop/QLoRA/8-bit), overrides honored, every catalog model maps to a GPU,
  and the 35B trainer fits a 48 GB card via QLoRA.
- **Kernel application is GPU-side** (inside `build_lora_policy_update`) — covered by the live run.
- The Chalk path requires Qwen3.5/3.6 + the patched vLLM (the `flash-worker` image); Liger / QLoRA /
  8-bit AdamW / FLA-drop are stack-agnostic and run on any model.

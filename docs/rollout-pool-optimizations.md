# Pool trainer — optimization parity with origin/dev, + 35B and the model→GPU matrix

The pool's training server (`flash/engine/pool_policy.py`) applies the **same optimization stack as
origin/dev's worker**, by calling **dev's own helpers** (so there's one implementation, not a fork):

| optimization | how the pool applies it | dev source it reuses |
|---|---|---|
| **Chalk** kernels (standalone fused RMSNorm/SwiGLU/FLCE + RoPE/LoRA-delta/embedding) | `install_chalk_kernels(model)` after build | `flash.engine.chalk_kernels.install_chalk_kernels` (identical call; runs `liger=False`) |
| **fla fast-path** (Qwen3.5/3.6 GDN on Hopper) | `_ensure_fla_fastpath_on_hopper()` before load | `flash.engine.worker.perf._ensure_fla_fastpath_on_hopper` |
| **8-bit paged AdamW** | `loraplus_optimizer_cls(fused_optim_name())` → `PagedAdamW8bit` | `flash.engine.worker.perf` (identical) |

Each is **resolved per model** by `resolve_opt_config(model_id)` with dev's defaults (chalk + 8-bit
on; the Hopper fla fast-path on for Qwen3.5/3.6), and every flag is overridable. Chalk is **standalone**
— it installs its own RMSNorm/SwiGLU/fused-CE (Liger faded out, flash#66), so the policy passes
`liger=False`. The bases load in **bf16** (QLoRA was removed in #74: the GRPO vLLM rollout merges the
LoRA into a 4-bit base and that rounding collapses the importance-sampling ratio → no learning, so the
catalog is bf16 everywhere). Each optimization is a no-op off-GPU / where inapplicable (chalk and the
fla fast-path no-op on non-Qwen3.5), exactly like dev.

## Does it work with 35B? Yes — and *better* than dev's colocate

`Qwen/Qwen3.6-35B-A3B` is the catalog's MoE tier — **bf16** (not QLoRA), `requires_disaggregated`,
`single_trainer_only`, with a 300 GB disk floor (`flash/catalog.py`). dev runs GRPO **colocated**
(trainer + vLLM rollout on one card), which OOMs for the 35B, so its GRPO is rejected for the colocate
path → it needs a dedicated inference GPU. The pool **disaggregates** by construction, so the trainer
carries no vLLM:

- **trainer** = a bf16 LoRA-only trainer (no colocated vLLM, no second weight copy + KV pool).
- **rollout** = the 35B vLLM (MoE, expert-parallel, all experts resident) on the shared inference
  pool — the only piece that needs a big card, and it's shared across runs.

Removing the colocated vLLM from the trainer is the whole disaggregation win: the LoRA-only trainer
fits a smaller, cheaper card than dev's colocate. Submit-time validation enforces the constraints
(`[train].inference_gpus>0`, single-trainer 1:N only — see
`engine.rollout_bench.validate_disaggregated_requirement`). For current sizing read `pool_gpu_plan()`
rather than trusting any number embedded here; the table below is the plan's output at the time of
writing, and the live allocator calibrates:

```
model                 trainer GPU (vram)            inference GPU (vram)
Qwen3.5-0.8B          RTX 2000 Ada (~7 GB)          RTX 2000 Ada (~9 GB)
MiniCPM5-1B           RTX 2000 Ada (~7 GB)          RTX 2000 Ada (~9 GB)
Qwen3.5-2B            RTX 2000 Ada (~10 GB)         RTX 2000 Ada (~12 GB)
Qwen3.5-4B            RTX 2000 Ada (~14 GB)         RTX A4500 (~16 GB)
Qwen3.5-9B            A40 (~26 GB)                  A40 (~28 GB)
Qwen3.6-35B-A3B       RTX Pro 6000 WK (~86 GB)      RTX Pro 6000 WK (~89 GB), expert-parallel
```
(VRAM and GPU pick come straight from `pool_gpu_plan(model_id)` — `flash/engine/pool_policy.py` —
which sizes a bf16 base + LoRA + 8-bit optimizer state under grad checkpointing for the trainer and the
served base + KV for inference, then picks the cheapest `flash.providers.base.GPU_INFO` class that fits.
All entries are `qlora=False`. The point: every model maps to a GPU, and the disaggregated 35B trainer
fits a single card instead of dev's H200 colocate.)

## Verification

- **CPU-tested** (`tests/test_pool_optimizations.py`): the same optimization SET as dev is selected
  per model (chalk + the Hopper fla fast-path for Qwen3.5/3.6 + 8-bit AdamW, `liger=False`), overrides
  are honored, `qlora` is `False` for every model (the catalog is bf16), every catalog model maps to a
  trainer + inference GPU, and the 35B trainer is bf16 with a trainer cost far below an H200 colocate.
- **Live multi-GPU run (done)** — the pool ran end-to-end on a real **4× RTX 3090** box: two real
  `vllm serve --enable-lora` servers behind the router + two real concurrent LoRA trainers
  (`GRPOPoolLoop` + `pool_policy`, real transformers/peft forward/backward) pushing real adapters
  through it (generation load-balanced 6/6 across both GPUs, both LoRAs co-resident on both GPUs,
  per-step weight-sync hot-swap to vLLM, version 1→3). That run used a **public Qwen2.5** model so the
  stack is fully reproducible — which means it exercised the pool plumbing and the **model-agnostic**
  optimizers (bf16 LoRA + 8-bit paged AdamW), not the Qwen3.5/3.6-only paths (chalk and the fla
  fast-path correctly no-op on Qwen2.5). Full report + logs in
  [`rollout-pool-live-run.md`](rollout-pool-live-run.md).

## Live coverage of the Qwen3.5/3.6-specific paths (chalk, fla fast-path, 35B)

Chalk + the Hopper fla fast-path + the 35B MoE only apply to **Qwen3.5/3.6**, whose `qwen3_5` arch is
served only by the **patched vLLM in the `flash-worker` image** — public vLLM 0.23 does *not* register
`qwen3_5` (verified: a Qwen2.5 rollout boots on the public image, a Qwen3.5 rollout does not). On Vast,
the `flash-worker:cu128` image (~15–20 GB) pulls **unreliably** (instances repeatedly vanished
mid-pull), which blocks a live Qwen3.5/35B pool run here. That is an **infrastructure** limit, not a
code one:

- the chalk / fla-fast-path calls are **dev's own already-validated helpers** (identical code);
- the per-model selection is **CPU-tested** and the GPU matrix is built;
- 35B already trains live on H100 via the **same disaggregated approach** (the verl one-step-off path,
  freesolo PR #215) — the pool generalizes that fleet-wide. Run a live Qwen3.5/35B pool E2E on a host
  that has the `flash-worker` image cached (or a pre-pulled/registry-mirrored worker image).

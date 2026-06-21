# Is the pool cheaper than `origin/dev`'s one-GPU-per-job? Yes — especially at scale

`origin/dev` runs GRPO **colocated**: every training run rents **one GPU** that does the whole step
— vLLM rollout, then reward grading, then the optimizer update — sequentially. (Confirmed: dev's
worker has **zero** disaggregation; `[train].inference_gpus` and the pool don't exist there.)

The model below uses **dev's own cost constants** (`flash/cost/analytical.py` + `facts.py`): a GRPO
step is `gen_s` (vLLM decode, MFU 0.12) + `reward_s` (grading) + `update_s` (train, MFU 0.35), priced
at the real `$/hr` from `flash/providers/base.py`. Reproduce: `python scripts/pool_cost_model.py`.

## Where dev's colocate wastes money

For each run, on one **expensive training-class GPU** (A100, ~$1.00/hr):

1. **`reward_s` is pure GPU idle** — grading is CPU / LLM-judge / sandbox work; the GPU sits idle
   waiting (4s/step at the 1.0s default, 12s/step for an LLM-judge).
2. **`gen_s` runs on the expensive card** — but decode is *memory-bandwidth-bound*, so it doesn't
   need A100 compute; you're paying training-class $/hr for bandwidth work.
3. **No sharing** — N runs ⇒ N expensive GPUs, each reloading the base model, each under-batched.

## What the pool changes

- **trainer GPU does only `update_s`** — rollout + reward are prefetched/overlapped off it (proven
  live: the pipelined producer + per-step weight-sync).
- **rollout on a shared pool of cheaper, bandwidth-class cards** (A40, ~$0.44/hr) holding the base
  once + many LoRA adapters (**multi-LoRA**, proven live: both runs' adapters co-resident on both
  GPUs); continuous batching across concurrent runs raises decode efficiency at scale.
- **reward on cheap CPU workers** (~$0.10/hr), not the GPU.

## Result (Qwen-4B GRPO, batch 8 × group 8, 512 tok, 30 steps)

savings = `1 − pool$ / dev$`; **+ = pool cheaper**:

| concurrent runs | light reward (0.1s) | default reward (1.0s) | heavy reward / LLM-judge (3.0s) |
|---|---|---|---|
| 1   | −12% (colocate wins) | +8%  | +34% |
| 4   | +17% | +32% | +51% |
| 8   | +21% | +35% | +53% |
| 32  | +23% | +37% | +54% |
| 64  | +24% | +37% | **+55%** |

Concretely at **N=32, default reward**: dev = **$5.49** (32× A100) vs pool = **$3.47** (32× A100
trainers + 25× A40 inference + CPU reward) — **37% cheaper**. With an LLM-judge reward it's **~54%**.

## Honest reading

- **At scale (N ≥ 4) the pool is always cheaper**, and the win **grows with the reward weight** —
  reward-bearing GRPO (LLM-judge, code-exec, multi-turn tools) is exactly where dev's colocate GPU
  idles most, and where the pool wins biggest (up to ~55%).
- **For a single light-reward run, colocate can be marginally cheaper** (−12%): disaggregation adds
  a second card, which only pays off once reward is non-trivial or runs share the inference pool.
  That's the expected crossover, not a flaw — small one-off jobs should still colocate.
- The model's levers are the mechanisms the **[live run](rollout-pool-live-run.md) already proved**:
  reward genuinely off-GPU, multi-LoRA packing (many adapters per inference GPU), rollout
  distributed across the fleet, and per-step weight-sync. The one projected (not yet measured at
  high N) lever is the continuous-batching decode bonus — modeled conservatively (≤2×).

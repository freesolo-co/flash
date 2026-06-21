# Is the pool cheaper? Yes — vs both `origin/dev` (colocate) and PR #4 (disaggregated), especially at scale

Two baselines:

- **`origin/dev` — colocate, one GPU per run.** Each run rents **one GPU** that does the whole step
  (rollout → reward → update) sequentially. (Confirmed: dev's worker has **zero** disaggregation.)
- **[PR #4](https://github.com/freesolo-co/flash/pull/4) — disaggregated, per run.** Each run rents
  a **multi-GPU box**: a dedicated `trl vllm-serve` rollout server on inference GPU(s) + the trainer.
  TRL server mode is **synchronous** (the trainer blocks on each generation batch) and reward runs
  **inline on the trainer** — so the dedicated inference GPU idles during reward+update, the trainer
  idles during rollout, and both are billed the whole run. It buys single-run **speed** (TP/DDP), not
  cost. The **pool is the same disaggregation idea, but shared across the fleet** instead of
  dedicated per-run, with prefetch overlap and off-GPU reward.

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

Pool savings (**+ = pool cheaper**), default 1.0 s/completion reward:

| concurrent runs | total cost (dev / PR#4 / pool) | pool vs dev | pool vs PR #4 |
|---|---|---|---|
| 1   | $0.17 / $0.34 / $0.16 | +8%  | +53% |
| 4   | $0.69 / $1.35 / $0.47 | +32% | +65% |
| 8   | $1.37 / $2.70 / $0.89 | +35% | +67% |
| 32  | $5.49 / $10.82 / $3.47 | +37% | +68% |
| 64  | $10.99 / $21.63 / $6.91 | +37% | +68% |

vs **PR #4** specifically, across reward regimes (pool is cheaper at *every* scale, including N=1,
because PR #4 dedicates two synchronous GPUs per run):

| concurrent runs | light reward | default reward | LLM-judge reward |
|---|---|---|---|
| 1  | +46% | +53% | +63% |
| 8  | +62% | +67% | +74% |
| 64 | +63% | +68% | **+75%** |

vs **dev colocate** across reward regimes:

| concurrent runs | light reward | default reward | LLM-judge reward |
|---|---|---|---|
| 1  | −12% (colocate wins) | +8%  | +34% |
| 8  | +21% | +35% | +53% |
| 64 | +24% | +37% | **+55%** |

Concretely at **N=32, default reward**: dev **$5.49** (32× A100) / PR #4 **$10.82** (32×[A100+A40]) /
pool **$3.47** (32× A100 trainers + 25× A40 inference + CPU reward) — **37% under dev, 68% under PR #4**.

## Honest reading

- **At scale (N ≥ 4) the pool is always cheaper**, and the win **grows with the reward weight** —
  reward-bearing GRPO (LLM-judge, code-exec, multi-turn tools) is exactly where dev's colocate GPU
  idles most, and where the pool wins biggest (up to ~55%).
- **For a single light-reward run, colocate can be marginally cheaper** (−12%): disaggregation adds
  a second card, which only pays off once reward is non-trivial or runs share the inference pool.
  That's the expected crossover, not a flaw — small one-off jobs should still colocate.
- **PR #4 is a *latency* play, not a *cost* play — and is complementary, not a competitor.** It makes
  a *single* run finish faster by throwing dedicated TP/DDP GPUs at it (measured ~2× on 4 GPUs, up to
  5.2× on 8); that's strictly *more* GPU-hours than colocate, so it costs more per run (the table
  models PR #4's cheapest 1:1 config; higher ratios are faster but pricier still). The pool keeps
  PR #4's disaggregation *benefits* (off-trainer rollout, overlap) but **shares** the inference GPUs
  across the whole fleet and offloads reward, so it optimizes **$/run at scale** instead of single-run
  wall-clock. Use PR #4 when one run must finish ASAP; use the pool to train many runs cheaply.
- The model's levers are the mechanisms the **[live run](rollout-pool-live-run.md) already proved**:
  reward genuinely off-GPU, multi-LoRA packing (many adapters per inference GPU), rollout
  distributed across the fleet, and per-step weight-sync. The one projected (not yet measured at
  high N) lever is the continuous-batching decode bonus — modeled conservatively (≤2×).

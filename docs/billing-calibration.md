# Training billing and estimator calibration

## What a training run is charged

SFT and GRPO runs are charged for **metered training GPU time**:

```
charge = billable_train_seconds / 3600 × rate-card $/hr for the rented card × rented cards
```

- **Meter start:** the start of the attempt's first optimizer update (`FLASH_TRAINING_PHASE
  first_update started`, observed by the worker parent).
- **Meter stop:** the moment the trainer child exits (SFT child; GRPO verl), before the checkpoint
  watcher or required-save validation drains any pending adapter publication or upload.
- **Never billed:** queueing and provisioning, container boot, model download, data loading,
  capability checks, trainer/model/vLLM initialization, and final adapter export/publication.
- **Billed:** every optimizer update, including the first one (graph capture and warm-up happen
  inside it), plus checkpoint saves taken during training.
- **Before the first update:** the worker reports an explicit `billable_train_s = 0`, so a run
  cancelled during trainer/model initialization bills $0 rather than looking like a legacy worker.
- **Retries and Spot resumes:** `record_heartbeat` keeps one record per attempt,
  `billable_by_attempt = {attempt: {"s": seconds, "shape": "provider|gpu|cards"}}`:
  - `s` is the attempt's **highest** observed meter. Heartbeats can publish out of order, so a stale
    smaller sample never lowers it.
  - `shape` is stamped only while the live provider handle belongs to that same attempt, so a late
    sample from a replaced attempt never borrows the replacement's shape.
  - The final attempt is raised to its frozen terminal meter. Every heartbeat counts, including the
    first one observed.
- **Attempts on different shapes:** each attempt bills its seconds at its own shape's rate and card
  count. An OOM retry on a larger card or count, or a Spot resume on another class, never reprices
  earlier attempts at the final rental. An attempt whose shape was never observed bills at the
  final rental.
- **Cancellation:** the same per-attempt record, with the latest heartbeat's meter for the live
  attempt.
- **Spot:** dev's fixed class rates (H100 $2.00/hr, H200 $2.50/hr).
- **Legacy workers** that report no meter keep the previous quote-based pricing.
- **Custom runs** are unchanged. They were already metered on session time, which starts after
  the backend loads the model and ends at session close or cancel. Session time includes time the
  GPU sits reserved while the client script prepares data or waits for rewards.

The quote shown by `flash train --cost` is the **estimate** of that charge. It is no longer the
amount billed.

## Why

An audit of 1,469 single-attempt, on-demand, completed production runs compared the charged
quote with measured training-only time. Charges were **2.27×** the training-only cost ($7,596
charged vs $3,343 of measured training). Individual runs were charged anywhere from 0.6× to 4×
their real training cost.

## Estimator calibration

The time model (`flash/cost/analytical.py`) is fit to the metered window. Constants come from
two sources: historical production runs (relative least squares, 20% holdout) and live
calibration runs with exact first-update → loop-exit timing. Wherever the two disagree, the live
runs decide, because historical "actual" values include unobservable queue time.

| term | before | after |
|---|---|---|
| SFT effective MFU (vs `effective_train_tflops`) | 0.25 | 0.85 |
| SFT attention term (× mean example tokens / 32768) | — | 2.0 |
| SFT per-step floor | 0.8 s | 0.10 s |
| SFT first update (one-time) | — | 250 s + 5 s × active params (B) |
| MoE SFT MFU (total params) | 0.10 | 0.45 |
| Image SFT vision encoding (multimodal profiles only) | — | 0.41 s/example |
| GRPO step floor | 62.7 s + 0.805 s/completion | 3.5 s + 0.466 s/completion |
| GRPO decode / update MFU | 0.12 / 0.35 | 0.25 / 0.72 |
| GRPO first update (one-time) | — | 60 s + 6 s × active params (B) |

### Historical runs (1,469 completed, on-demand, single attempt)

| estimate ÷ actual | median | mean abs error | within ±20% |
|---|---|---|---|
| before | 2.28 | 124% | 12% |
| after | ~1.0 | 22% | 58% |

### Live calibration runs (exact timing)

Live runs cover Qwen3.5-0.8B, 4B, 9B, 9B-Base and Qwen3.6-35B-A3B on 1× H100, 1× H200 and
2× H100. They include 16-, 64- and 256-completion GRPO rollouts and 300–800-step SFT.

| | n | median | mean abs error | within ±20% |
|---|---|---|---|---|
| SFT | 29 | 0.97 | 11% | 83% |
| GRPO | 21 | 0.99 | 11% | 81% |

What the quote still cannot know:
- **GRPO realized completion length.** The quote prices with `max_completion_tokens`. A 2,048-token
  cap trained faster than a 512-token cap when the model stopped early.
- **Per-architecture efficiency.** Qwen3.5-4B runs ~0.8–0.9× its estimate, which suggests it is
  less efficient per FLOP than 0.8B or 9B.

- **A100 SXM** measured ~0.68× its spec-peak estimate for both SFT (0.69) and GRPO (0.67). Its
  training throughput is capped at the measured **162 TFLOPS** (catalog 264), which brings A100 SFT to
  1.00. Three live A100 GRPO arms (16, 64 and 128 completions) ran a uniform ~0.78 of estimate, so
  A100 also carries a measured whole-step GRPO scale of **1.28** (`GRPO_CARD_STEP_SCALE`). That
  brings them to 0.90 / 0.93 / 1.03. The cap also changes dollars-per-step placement: unconstrained 9B GRPO now lands
  on **RTX Pro 6000**, which live runs confirm is cheaper per job ($0.214 vs $0.324 on A100 for the
  same run; RTX Pro measured 0.94 for SFT and 1.21 for GRPO).
- **L40S** is **not** capped. It measured slow for SFT (0.59; its catalog 362 TFLOPS is the sparse
  figure) but fast for GRPO (1.47, faster than H100), which a single throughput number cannot
  represent.

- **B200** measured 1.12–1.24, faster than estimated. The existing policy deliberately caps B200 at
  H200-class training throughput ("never cheaper or faster than H200"), so this over-estimate is
  intentional and left in place.
- **Covered with no extra term needed:** thinking-mode SFT (1.00), 12k context (0.97–1.05), 9B-Base
  vs instruct (within 5% of each other for both SFT and GRPO), 2× H100 for SFT and GRPO, and H200.
- **LoRA rank** has no measurable effect on this workload. One live rank-128 run took 1.64× its
  rank-16 twin, but its repeat (0.97) and a rank-64 run (1.00) matched rank 16. The slow run was a
  one-off outlier, so no rank term was added.
- **Image SFT** (MiniWoB screenshots; gated on the profile's `architecture_mode == "multimodal"`,
  not on exact-unpacked packing, which text-only models that cannot pack also use) was
  under-estimated at 0.63 (n=46). A
  per-example vision-encoding term (0.41 s/example) brings it to ~1.00. Three live 600-step MiniWoB
  image runs measured 0.97 / 1.20 / 1.14. Identical repeats spread 816–1,006 s. Each spent
  900–1,100 s in setup (image loading and model download), which the meter excludes.
- **Run-to-run noise.** Identical repeats differed by 9% (GRPO 0.8B and 9B, SFT 9B) and 23% (short
  SFT 0.8B). No estimate can beat that.

The charge is metered, so these limits affect only the estimate, never the amount billed.

### What metering changes in practice

Per-run charged vs. actual training cost before this change (live runs):

| run | charged (quote) | actual training | ratio |
|---|---|---|---|
| SFT 9B 800 steps | $5.40 | $2.03 | 2.66× |
| SFT 35B-A3B (MoE) | $11.17 | $3.28 | 3.41× |
| GRPO 0.8B | $0.81 | $0.24 | 3.41× |
| SFT 0.8B | $0.30 | $0.42 | 0.72× (under) |
| GRPO 9B on 2× H100 | $3.34 | $3.97 | 0.84× (under) |
| SFT 9B cancelled at step 150/1500 | $1.02 | $0.66 | 1.55× |

## Re-running the calibration

1. Collect `flash runs status --json` for completed runs.
2. For each run, actual training seconds = final heartbeat ts − (worker start +
   `worker_to_first_update_s` − `longest_phase_s.first_update`) − `longest_phase_s.publication`.
   For a metered worker, read `training_timing.billable_train_s` directly.
3. Rebuild each run's `RunConfig` from its spec and `workload_profile`, then compare
   `estimate_cost(cfg).train_seconds` with the actual.

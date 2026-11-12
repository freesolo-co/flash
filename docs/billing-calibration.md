# Training billing and estimator calibration

## What a training run is charged

SFT and GRPO runs are charged for **metered training GPU time**:

```
charge = billable_train_seconds / 3600 × rate-card $/hr for the rented card × rented cards
```

- **Meter start:** the start of the attempt's first optimizer update (`FLASH_TRAINING_PHASE
  first_update started`, observed by the worker parent).
- **Meter stop:** the training loop exits (SFT child exit; GRPO verl exit).
- **Never billed:** queueing and provisioning, container boot, model download, data loading,
  capability checks, trainer/model/vLLM initialization, and final adapter export/publication.
- **Billed:** every optimizer update, including the first one (graph capture and warm-up happen
  inside it), plus checkpoint saves taken during training.
- **Retries and Spot resumes:** each attempt meters its own window. `record_heartbeat` carries the
  earlier attempts' metered seconds as `billable_train_s_prior`, so the charge covers every attempt.
- **Cancellation:** the latest heartbeat's meter (`training_timing.billable_train_s`), plus earlier
  attempts.
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

The time model (`flash/cost/analytical.py`) was refit to the metered window by relative least
squares, with a 20% holdout:

| term | before | after |
|---|---|---|
| SFT effective MFU (vs `effective_train_tflops`) | 0.25 | 0.85 |
| SFT attention term (× mean example tokens / 32768) | — | 0.47 |
| SFT per-step floor | 0.8 s | 0.26 s |
| SFT first update (one-time) | — | 205 s |
| GRPO step floor | 62.7 s + 0.805 s/completion | 14.2 s + 0.048 s/completion |
| GRPO decode / update MFU | 0.12 / 0.35 | 0.18 / 0.52 |
| GRPO first update (one-time) | — | 135 s |

Estimate ÷ actual on historical runs:

| | median | mean abs error | within ±20% |
|---|---|---|---|
| before | 2.28 | 124% | 12% |
| after | 0.92 | 22% | 59% |

**Known gaps:**
- A100 runs are under-predicted (median 0.63, n=17).
- Historical "actual" values include provisioning queue time for runs submitted in large parallel
  campaigns, which biases them upward.
- Live calibration rounds with exact log timing are tracked in the PR that introduced this file.

## Re-running the calibration

1. Collect `flash runs status --json` for completed runs.
2. For each run, actual training seconds = final heartbeat ts − (worker start +
   `worker_to_first_update_s` − `longest_phase_s.first_update`) − `longest_phase_s.publication`.
   For a metered worker, read `training_timing.billable_train_s` directly.
3. Rebuild each run's `RunConfig` from its spec and `workload_profile`, then compare
   `estimate_cost(cfg).train_seconds` with the actual.

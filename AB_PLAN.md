# SFT activation-shaping A/B plan

## Objective

Measure whether sizing from realized sequence lengths, unpacked length grouping, cadence-only quality metrics, corrected token accounting, and non-blocking host-to-device copies improve SFT throughput without changing the training objective or effective-batch contract.

This document is a preregistered plan only. Do not allocate a paid GPU as part of this branch task.

## Arms

- control: `origin/dev` at `8fdfc8580b9058d8c40289d4fa1a2e6e3b7bced2`
- treatment: the final commit on `perf/sft-activation-shaping`

Run both arms through the same Flash control plane on separate, matched RunPod GPUs with the same exact GPU type, worker image, CUDA stack, model revision, environment revision, seed, and launch configuration. Do not reuse a warmed worker across arms.

## Frozen workload

- model: `Qwen/Qwen3.5-0.8B`, pinned to one immutable Hugging Face revision
- algorithm: SFT
- provider: RunPod
- hardware: the same exact RTX 4090 SKU for both arms
- seed: `42`
- environment: one existing real Freesolo hub environment revision with at least 1,024 labeled SFT rows and no empty completion targets; record the immutable environment id before launch and use it unchanged in both arms
- row-length gate: profile the rendered and tokenized rows before launch; require maximum realized length at most 512 tokens
- `max_context_tokens = 32768`, intentionally much larger than the gated realized maximum and above the dense-mask packing limit so both arms remain unpacked
- `batch_size = 8`
- same learning rate, LoRA rank, epochs or `max_steps`, save schedule, input prefix, data revision, and all other fields
- bounded duration: 100 optimizer updates after a 10-update warmup window

The treatment is expected to size at the safe realized maximum and reach a larger microbatch than the control, while gradient accumulation preserves the requested effective batch of at least eight examples.

## Measurements

Collect from worker metadata and trainer logs:

1. steady-state training tokens per second over updates 11 through 100, using exact realized input-token counts and training wall time for that window
2. `per_device_train_batch_size`, `gradient_accumulation_steps`, configured maximum length, and realized maximum length
3. cumulative `num_tokens` at every logging event, checked against an independent sum of realized attention-mask lengths from the same batches
4. step-aligned logged loss, normalized loss-curve area, and final logged loss
5. peak allocated GPU memory, optimizer-step count, examples consumed, and any OOM, NaN, retry, or under-run

Exclude model download, environment download, tokenizer preprocessing, checkpoint upload, and trainer initialization from the throughput window.

## Preregistered thresholds

The treatment passes only if all correctness gates and the throughput gate pass.

### Correctness gates

- no OOM, NaN, worker retry, or under-run in either arm
- treatment cumulative token count equals the independent realized-token count exactly at every logging event; direction: zero error
- both arms complete the same optimizer-step count and consume the same requested effective examples per update; direction: no reduction
- treatment normalized loss-curve area differs from control by at most 3%; direction: no material increase
- treatment final logged loss differs from control by at most `max(0.02, 3% of the control final loss)`; direction: no material increase
- treatment peak GPU memory remains within the card limit with at least 5% headroom; direction: no OOM-risk regression

### Throughput gate

- treatment steady-state tokens per second must be at least 15% higher than control; direction: increase
- treatment realized per-device microbatch must be larger than control, expected `4` versus `1`; direction: increase
- if throughput improves by less than 15%, or loss parity fails, report the result as not validated and do not generalize the optimization from this A/B

## Reporting

Report the immutable control and treatment commits, environment and model revisions, exact RunPod GPU identity, full frozen configs, raw logging rows, token-count reconciliation, microbatch and accumulation values, throughput ratio, loss-curve comparison, memory peaks, and a pass or fail decision against every threshold. Do not tune thresholds or workload after observing results.

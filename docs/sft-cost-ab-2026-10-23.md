# Short SFT cost A/B — 2026-10-23

Three paired seeds compare dev `60ce2683` with the production code in `8b3b2de0`.
Both sides use automatic GPU selection; all six selected one L40S on the first attempt.
Each run starts with a fresh, isolated weight-cache volume. No training GPU type,
count, provider, or region was pinned.

| Mean across seeds 17, 29, 43 | Before | After |
| --- | ---: | ---: |
| Estimated compute cost per run, including CPU stages | $0.521 | $0.264 |
| End-to-end completion | 10.11 min | 7.94 min |
| GPU worker lifetime | 9.45 min | 6.10 min |
| Trainer wall time | 5.91 min | 5.16 min |
| Steady update time, steps 129–512 | 0.518 s | 0.400 s |
| Held-out accuracy | 57.29% | 57.29% |
| Held-out target-token NLL | 0.4928 | 0.4803 |
| Peak GPU memory | 23.51 GiB | 26.74–26.76 GiB |

Estimated compute cost fell **49.3%**, end-to-end time **21.4%**, and steady update
latency **22.7%**. Every paired seed was cheaper after the change. The remaining cost
is still about 2.1× the earlier matched Tinker token-price estimate ($0.125/run).
This does not establish price parity with Tinker.

The paired seed-and-question bootstrap (20,000 resamples) gives an accuracy difference
of 0.00 percentage points with a 95% interval of **[−3.91, +3.91] points**, inside the
predeclared ±5-point equivalence margin. This is a small, single-task study, not proof
of identical numerics or universal quality equivalence. Loss trajectories and individual
predictions differ; the change does not promise bitwise identity.

## What changed in the measured candidate

- Stage the exact pinned public SFT base weights on bounded CPU compute before GPU launch.
  Candidate GPU-side downloads measured 0 seconds; controls measured 98.7–238.3 seconds.
- Reduce training-worker minimum reservations from 4 CPU cores/32 GiB to 2 cores/8 GiB
  per GPU. Bursting remains enabled; estimates use the same reservation constants as launch.
- Avoid gradient-checkpoint recomputation for the qualified single-GPU text workload:
  Qwen3.5-9B revision below, SM89 with 47–50 decimal GB usable memory, microbatch 1,
  rank ≤8, realized sequence length ≤512, fused loss and known model geometry. An
  additional 8 GB estimated margin is required. Other checkpointing gates remain unchanged.

The reservation change applies to training generally; paid performance validation here
covers only 9B SFT. Larger models and GRPO were not rebenchmarked in this PR. The code
retains burst capacity rather than adding hard limits. Warm multi-job workers, Modal
Functions migration, larger batches, and an allocator trained on complete job costs are
outside this change.

## Matched recipe and evidence

- Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, BF16.
- Published MMLU-Pro “other” category, custom 512-train/256-held-out split; not an official
  MMLU-Pro score. Same preserved split and tokenized examples as the previous comparison.
- 512 optimizer updates, batch 1, MLP-only LoRA rank 8/alpha 32/dropout 0; exact same
  initial adapter tensors for each paired seed.
- Adam betas 0.9/0.999, epsilon 1e-8, weight decay 0, gradient clip 1; LR 1e-4,
  15 warmup updates followed by linear decay. Same completion-only loss masks and order.
- Live worker parquet hashes verify every pair's token IDs, masks and example order.
  All workers attested the expected immutable source archive and completed 512 updates.
- All six adapters use the same pinned H100 evaluation image and Transformers BF16
  flash_attention_2 evaluator, greedy max 16 tokens, batch 8. The fresh base evaluation
  differs from the earlier study by one held-out answer; both A/B arms use this fresh
  common evaluation. No new Tinker training was purchased.

## Cost accounting and limitations

The primary cost estimate uses the complete observed GPU-worker lifetime at its minimum
resource rate, plus separate **upper estimates** for CPU staging and final checkpoint
publication. It excludes storage, egress, platform startup before the worker marker,
and unobserved CPU/RAM bursts; it is not a final invoice.

Live resource meters corroborate the estimate, but reset when billing data flushes.
The report sums observed windows rather than treating the last partial window as a
complete run. Those observed totals plus CPU upper estimates average $0.514 before
and $0.257 after. Small gaps between samples remain unmetered.

Modal's current provider report totals **$1.6149 for the three baseline jobs**, versus
**$1.1499 for the three candidate jobs plus an extra pilot**. These totals include
reported CPU, RAM, GPU and network egress, but can lag and are not finalized invoices.
The pilot did not activate the initial nominal-48-GB checkpointing gate on an L40S
exposing 47.696 GB. It is excluded from the paired comparison and included in the
experiment budget. No failed trial was silently substituted into the comparison.

Physical hosts and regions were assigned by Modal. Cold-download and hardware variance
are visible in the per-seed results; a longer or already-warm job may save less.
The benchmark also imposed identical 8-core/128-GiB burst ceilings on GPU workers for
budget control; production retains unbounded bursts above its minimum reservations.

Machine-readable per-run metrics, source attestations, token hashes, bootstrap interval,
and provider billing records are in [the result JSON](sft-cost-ab-2026-10-23.json).
The measured production-source archive matches candidate `8b3b2de0`. Its validation included
Python 3.11 and 3.12 CI, lint, size/layout gates, offline tests, and serial wall-clock tests.

## Subsequent memory-policy change

The PR now uses the shared memory estimate for text SFT with fused loss instead of the
model/revision/GPU allowlist above. Checkpointing is disabled only when estimated weights,
LoRA/optimizer state and activations fit with headroom: 8 GB for dense models and 18 GB for
MoE. Actual row length, microbatch, rank and model geometry drive the estimate. Missing or
invalid sizing inputs keep checkpointing enabled. Multimodal training and GRPO retain their
existing defaults; the estimate does not model vision activations. Multi-GPU text SFT uses
the conservative full-model estimate per card without assuming any sharding savings.

The benchmark configuration still disables checkpointing under the new rule. CPU regression
tests cover other model identities, revisions, geometries, memory capacities and workload
sizes, as well as the existing measured OOM boundaries. The six GPU runs above tested the
earlier policy and do not validate every newly eligible workload. The live checks below
separately exercise the revised policy.


## Automatic GPU selection checks on the revised policy

Two fresh-cache SFT runs tested commit `14e93b92` through the production `allocate()` and
Modal `submit_attempt()` paths. GPU type, count and provider were all omitted from the
input; the allocator selected the first candidate without retries or manual overrides.
Both workers attested source archive
`0b6d629cda40b64a082afc31dccb3ae58475dee6a8b0f763ad681f1b3fa7d24b`.

| Workload | Selected GPU | Actual max tokens | Checkpointing | Updates | Peak GPU memory | Completion |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Qwen3.5-9B | 1× L40S | 304 | Off | 32/32 | 24.67 GiB | 4.74 min |
| Qwen3.8-27B | 1× H100 | 1,294 | On | 32/32 | 58.49 GiB | 7.97 min |

Both completed with finite training losses, no out-of-memory errors, and successful final
checkpoint publication. Live GPU probes matched the automatically selected hardware.
The short 9B workload fit without checkpointing; the longer 27B workload retained it.

Each run used 32 synthetic text examples, batch 1, LoRA rank 8, seed 17, and a 20-minute
deadline. CPU/RAM reservations remained production defaults; benchmark burst ceilings were
8 cores/128 GiB per GPU. Estimated compute including CPU staging and publication was
$0.135 and $0.379, respectively ($0.515 total), against a $5.786 conservative compute
ceiling. These estimates exclude storage, network and unobserved bursts and are not invoices.

This validates allocation and training completion for these two single-GPU configurations;
it is not a model-quality comparison or validation of all architectures, long contexts,
or multi-GPU jobs. It exercises production allocation/worker code directly, not hosted API
routing. Checkpoints and raw evidence were preserved locally before all six owned GPU/CPU
sandboxes, the dedicated app, cache/handoff volumes and temporary private artifact repository
were cleaned up. [Machine-readable results and allocation evidence](sft-memory-autogpu-2026-10-23.json).

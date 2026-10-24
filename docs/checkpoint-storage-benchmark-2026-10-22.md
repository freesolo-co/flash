# Checkpoint storage overlap: GPU measurement

Measured on October 22, 2026 (America/Los_Angeles). With checkpoint overlap,
median GPU-worker time decreased **17.8%**, from 248.7 s to 204.5 s, in four
same-region SFT runs. Median training time decreased **20.0%**. This is evidence
for a short workload that saves every step, with two runs per variant; it is not
a production-wide throughput estimate.

## Workload and controls

- One NVIDIA H100 80 GB HBM3 per run, Modal `us-central`, driver 610.57.04.
- Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- SFT, LoRA rank 8 / alpha 16, seed 42, batch size 1, maximum context 256.
- Four short arithmetic examples, six optimizer steps, saves at steps 1–6.
  Actual sequences were 24 tokens; 144 training tokens total. This deliberately
  emphasizes checkpoint costs and does not measure compute-saturated throughput.
- Cached model weights, same worker image, CPU allocation 4–8 cores, memory
  allocation 32–64 GiB. Independent sandboxes ran sequentially in A–B–B–A order.
- Baseline commit `0f97fdd2070be50db0b2a90f606095d8778ac706`; candidate was the
  checkpoint-overlap worktree snapshot. Identical observational timer wrappers
  were added to both. Normal runs had no artificial upload delay.
- Worker image digest:
  `c794abb352bb99c8a90e58f4f1f8f4b56012a1c00fcfe607dd8a037d0f119fb8`.
- Frozen source archive SHA-256:
  baseline `9b7bf27a7aacff775522351951c575c9c84ef9fef2000eb8b9c2016c737e4e1a`;
  candidate `3c2c02e98ad09afac52322488d985e123cd535e71f642a1a7772efc5d4c1e89e`.

## Results

All durations below are seconds. Medians use two successful runs per variant.

| Metric | Baseline | Overlap | Change |
|---|---:|---:|---:|
| GPU-worker elapsed time | 248.7 | 204.5 | 17.8% lower |
| Training elapsed time, including final watcher drain | 194.4 | 155.5 | 20.0% lower |
| Checkpoint window | 100.3 | 64.6 | 35.6% lower |
| Sum of trainer retention waits | 76.7 | 40.5 | 47.2% lower |
| Final watcher drain | 9.5 | 11.2 | 17.9% higher |
| Reported peak GPU memory, GB | 21.325 | 21.325 | unchanged |

The ratio of median training times is **1.25×** for the same amount of work.
The GPU-worker interval runs from the worker start marker to its terminal GPU
marker. It includes worker preparation and training, but excludes provider
allocation, teardown and the subsequent CPU-only final checkpoint upload. It is
not an invoiced GPU-duration measurement. Training elapsed time includes child
model initialization and first-step compilation as well as training and drain.
The checkpoint window runs from the first adapter export start through final
watcher drain; it also includes subsequent training and captures.

| Run, in execution order | GPU-worker time | Training time | Checkpoint window | Retention waits | Final drain |
|---|---:|---:|---:|---:|---:|
| baseline-normal-3 | 268.1 | 212.4 | 108.8 | 83.4 | 8.2 |
| candidate-normal-3 | 204.4 | 152.7 | 72.1 | 44.6 | 16.7 |
| candidate-normal-4 | 204.6 | 158.2 | 57.1 | 36.5 | 5.8 |
| baseline-normal-4 | 229.3 | 176.4 | 91.8 | 70.0 | 10.9 |

Storage and host timing varied substantially. The first matched candidate spent
20.1 s applying queue backpressure; the second recorded none. Moving work off
the publication path can leave more work for final drain, which is included in
the reported totals. Overlapping phase durations must not be added together as
elapsed time. These results show reduced waiting, not faster GPU kernels. Sparse
heartbeat samples do not establish average SM utilization or memory bandwidth.

## Artifact checks

For every matched run, GPU and final CPU completion markers attest to the expected
source archive. All six required adapter configurations and final native checkpoint
files are present. All 496 final adapter tensors are finite and byte-identical
across the four matched runs, with identical shapes and dtypes. Their logged loss
curves also match: `[6.0356, 4.8291, 2.938, 2.5656, 1.0039, 0.7678]`.

This verifies the measured final adapters, not full optimizer-state equivalence,
resume correctness or learning quality. No resume experiment was run here.

## Exclusions and incomplete measurement

Two preliminary runs (`baseline-normal-1`, `candidate-normal-1`) were excluded
before the controlled campaign because placement was uncontrolled: the baseline
ran in `us-central` and the candidate in `eu-south`. Their training times were
192.2 s and 228.9 s. The candidate pilot's first-step loss and final tensors
differed from the baseline, before the checkpoint change could affect training.
The controlled same-region runs above did reproduce identical tensors.

The planned slow-storage pair injected eight seconds before each intermediate
native upload. Its baseline (`baseline-slow-2`) reached step 6, but Hugging Face
rejected terminal publication with HTTP 429: the repository exceeded 320 commits
per hour. The controller reported `job_preempted` because it could not observe a
successful terminal marker, although the sandbox exited with code 0. This run
is not counted as a success or performance sample. The paired candidate was not
launched, and no slow-storage speedup is claimed.

Longer sequences, representative save intervals, GRPO, OPD, multiple GPUs,
disk high-water usage, time to a target learning metric and persisted-checkpoint
resume equivalence remain unmeasured.

## Evidence and cleanup

Raw logs, specifications, frozen source trees, scripts, terminal markers and
per-run results are retained in the sibling local directory
`../flash-checkpoint-storage-benchmark-20261022/` (relative to the worktree root).
Key files are `summary.json`, `artifact-verification.json`, `provenance.json`,
`matched-campaign.json`, `created.jsonl`, and `cleanup.json`. `summarize.py` rebuilds
the timing table; `verify_artifacts.py` checks the remote artifacts and hashes.

Private remote artifacts remain in the dataset repository
`DavidBShan/flash-checkpoint-storage-bench-20261022`, under
`sft/storage-bench-<run-name>/`. No production serving deployment was changed.
All 14 owned sandboxes (seven GPU runs, six CPU final uploaders and one CPU cache
prefetch) were confirmed stopped. All seven per-attempt handoff volumes were
already absent, and the dedicated model-cache volume was deleted.

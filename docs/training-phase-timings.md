# Training phase timings

The worker now reports measured phases independently of the existing heartbeat
stage and billing step. A short `parent_preparation_seconds` value at child launch
is not a claim that the model is initialized or that the first update is ready.
The legacy `setup_seconds` metadata field remains compatible, with an explicit
scope in `notes.setup_seconds_scope`.

`FLASH_TRAINING_PHASE` JSON records contain start/completion/failure events,
process and span identity, and monotonic elapsed seconds. SFT, GRPO and OPD report
model downloads, capability checks, trainer initialization, model construction,
first and subsequent updates, checkpoint capture, and folder publication.
vLLM reports its compilation, kernel warmup and graph-capture operation as
`compilation_and_warmup`. Lazy training-kernel compilation remains inside the
first update; it is not estimated or mislabeled as a separately measured span.

The first update is the first executed update in this attempt, including after a
resume. For SFT it covers the training batch call; for GRPO/OPD it covers the
trainer's step call, including rollout/reward work and optimizer work. Native
checkpoint saving and subsequent sampler weight synchronization are outside
that interval. Existing native full-step pace metrics retain their own scope.

Child records are consumed by the parent and included in existing heartbeats as
`training_timing`, and retained in `train_meta.json` / metrics notes:

- `active_phase_elapsed_s`: elapsed since the parent observed each active phase.
- `longest_phase_s`: longest completed invocation per phase, across ranks. These
  are maxima, not totals or steady-state medians.
- `worker_to_first_update_s`: parent-worker elapsed time until it observes the
  first completed update. Includes preparation and child startup, excludes
  pre-worker allocation/bootstrap, and can include stdout delivery delay.

Nested and concurrent phases overlap; never add these fields to derive wall
time. Failure events remove active spans but do not count as completed samples.
An abruptly killed child may leave an unfinished span. Diagnostic events cannot
advance optimizer-step accounting. The recorder is bounded and attempt-local.
The status panel labels active durations as observations at the last heartbeat;
the existing heartbeat upload throttle still applies.

Modal separately logs allocation/bootstrap time from submission to its attested
worker-start timestamp and stores `notes.allocation_to_worker_start_s`. This is
not a pure GPU-allocation timer. Existing CPU publication duration and cost notes
remain separate; GPU termination and Volume commit precede that publisher.

## Checkpoint work already in the baseline

[Compact LoRA recovery](compact-checkpoints.md) retains trainable tensors,
buffers, optimizer, scheduler, RNG and data-position state with a pinned base.
The existing live Qwen3.5-9B SFT evidence reports 19.15 GB full versus 329 MB
compact checkpoints, about 31 seconds versus 2 seconds for native saves, and
exact resumed continuation. Those are prior measurements, not results of this
change. [CPU publication](cpu-checkpoint-uploads.md) already handles final native
checkpoints after GPU release. Intermediate publication still uses the GPU
worker's CPU. The separate checkpoint-storage-overlap worktree is not modified
by this change.

The worker image already pins `freesolo==0.4.2`. Model cache population before GPU
allocation, intermediate CPU publisher lifecycle changes, worker-count tuning,
and initialized-trainer reuse remain separate work. This patch does not claim
new throughput improvements or change worker counts or trainer lifetime.

## Validation

Offline tests cover timing overlap, attempt isolation, failed updates, malformed
records, Ray stdout prefixes, bounded retention, heartbeat/metadata propagation,
and status display. Existing checkpoint and CPU-handoff tests exercise recovery
identity and publication ordering. No new GPU benchmark or deployment was run.
For live qualification, compare matched frequent-save runs, record all phases,
and verify intermediate and final recovery before claiming broader gains.

Local validation on October 22, 2026: 186 focused tests passed and 6 dependency-
related tests skipped. The broader SFT/GRPO/OPD run had the same 25 failures as a
clean checkout at `60ce2683`: offline fixtures attempted external model-config
or dataset-script downloads. There were no additional failing test identities.
Ruff and `git diff --check` passed. The focused JUnit report is at
`/tmp/flash-timing-focused.xml` in the development workspace.

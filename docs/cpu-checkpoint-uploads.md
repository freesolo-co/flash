# CPU publication of final recovery checkpoints

New Modal training attempts always stage their final native recovery checkpoint
on a private per-attempt Volume and publish it with a CPU worker after the GPU
exits. There is no enable/disable setting. The trainer supplies its resolved final
step, so explicit `train.max_steps` and epoch-derived horizons use the same path.
Intermediate recovery saves and required adapter publication keep their existing
ordering on the GPU worker.

The controller confirms GPU termination (including Volume commit) before starting
a CPU-only uploader, capped at four cores, 8 GiB and 30 minutes, within the same
run deadline. The worker's success marker stays private until checkpoint, metrics
and DONE publication succeed. The uploader checks the staged job specification,
run/attempt identity, and checkpoint step against the completed training metrics.
A GPU handoff marker cannot complete startup recovery. Failed uploads keep the
previous durable HF checkpoint; retries can resume from it. Cancellation targets
the owned child and removes staging storage.

The staging copy and Volume commit still take time on the GPU. This avoids paying
GPU rates during the final HF transfer; it does not make periodic saves free or
promise a transfer speedup. CPU upload duration and an upper compute estimate are
reported separately from GPU training cost. Volume storage is transient and is
deleted by normal completion/cancellation cleanup. An unreconciled launch must be
reconciled before deleting a Volume; do not delete storage underneath a live run.
Legacy persisted handles without a handoff retain their original recovery behavior.
Local workers without the Modal Volume upload directly.

Earlier live checks covered an L4-to-CPU fixture and short Qwen3.5-9B SFT and GRPO
runs on one H100. The SFT run resumed a step-4 native checkpoint and reproduced the
uninterrupted final losses and all 496 adapter tensors exactly. OPD and the larger
multi-GPU model matrix have not received equivalent live training validation.

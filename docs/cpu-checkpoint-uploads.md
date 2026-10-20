# CPU publication of final recovery checkpoints

Operators can opt in with `FLASH_MODAL_CPU_UPLOADS=1` on the Modal controller.
This requires an explicit `train.max_steps` and leaves intermediate native saves
on the training worker. It stages the final native save on a private per-attempt
Modal Volume, preserving required adapter publication before leaving the GPU.

The controller confirms GPU termination (including Volume commit) before starting
a CPU-only uploader, capped at four cores, 8 GiB and 30 minutes, within the same
run deadline. The worker's success marker stays private until checkpoint, metrics
and DONE publication succeed. A GPU handoff marker cannot complete startup
recovery. Failed uploads keep the previous durable HF checkpoint; retries can
resume from it. Cancellation targets the owned child and removes staging storage.

The staging copy and Volume commit still take time on the GPU. This avoids paying
GPU rates during the final HF transfer; it does not make periodic saves free or
promise a transfer speedup. CPU upload duration and an upper compute estimate are
reported separately from GPU training cost. Volume storage is transient and is
deleted by normal completion/cancellation cleanup. An unreconciled launch must be
reconciled before deleting a Volume; do not delete storage underneath a live run.

Keep this opt-in until the full training/recovery matrix passes. A live L4-to-CPU
fixture verified GPU exit, byte-identical checkpoint publication, terminal
artifact ordering, and private-Volume cleanup. Full Qwen3.5-9B SFT and GRPO runs also completed publication after GPU exit.
The SFT run resumed a step-4 native checkpoint and reproduced the uninterrupted
final losses and all 496 adapter tensors exactly. These checks cover one H100
and short fixtures; OPD and the larger multi-GPU model matrix remain unvalidated
for this opt-in path.

# Compact LoRA recovery

FSDP2 LoRA training always saves compact recovery checkpoints, omitting immutable
frozen base parameters. Flash automatically pins the base model to a 40-character
revision and passes that identity into workers and retries. There is no checkpoint
format toggle.

Trainable LoRA tensors, persistent buffers, optimizer state, scheduler state, RNG,
and the existing data-loader checkpoint remain intact. A rank-local manifest
identifies omitted keys and the exact base. During recovery, the already loaded
pinned base supplies only those frozen keys. Flash validates the complete key
set, and native loading checks tensor shapes. A different base, missing trainable, missing buffer, or changed
model structure is rejected. Existing full-state checkpoints remain readable.

This removes repeated base bytes from disk/upload. It does not promise to remove
all FSDP state-dict materialization time, and compact recovery is not a standalone
full-model export: retain access to the pinned base weights.

Validation on the pinned worker stack: a two-H100 FSDP2 fixture restored the next
loss, every model tensor, optimizer state and scheduler state exactly. A Qwen 9B
SFT run exported a 280,919,452-byte recovery checkpoint (87,132,731-byte model
shard), then resumed from step 4 to step 8. Its four resumed losses and all 496
final adapter tensors matched the uninterrupted run exactly. These results do
not substitute for live coverage of every model, trainer or GPU topology.

An independent Qwen3.5-9B SFT check on one Modal H100 used all-linear LoRA targets
(716 tensors, including vision projections). Full checkpoints were 19.15 GB;
compact checkpoints were 329 MB. Fresh-process recovery from both mid-epoch and
an epoch boundary matched uninterrupted losses, final adapter tensors, buffers,
optimizer and scheduler state exactly. Exported adapters also matched. Native
save time was 31.25 seconds full versus about 2 seconds compact in that run,
excluding uploads; this was not a randomized performance benchmark. GRPO/OPD
recovery and changed GPU topologies remain outside this live validation.

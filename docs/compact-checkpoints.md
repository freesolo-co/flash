# Compact LoRA recovery (experimental)

An operator can set `FLASH_COMPACT_CHECKPOINTS=1` in the worker environment to
save FSDP2 recovery shards without immutable frozen base parameters. This is
opt-in until multi-rank save/load, adapter export, and resumed-update checks pass
on the supported worker stack. The base model must have a pinned 40-character
revision. Keep the setting enabled for retries that restore compact checkpoints.

Trainable LoRA tensors, persistent buffers, optimizer state, scheduler state, RNG,
and the existing data-loader checkpoint remain intact. A rank-local manifest
identifies omitted keys and the exact base. During recovery, the already loaded
pinned base supplies only those frozen keys; native strict loading still checks
the complete state. A different base, missing trainable, missing buffer, or changed
model structure is rejected. Existing full-state checkpoints remain readable.

This removes repeated base bytes from disk/upload. It does not promise to remove
all FSDP state-dict materialization time, and compact recovery is not a standalone
full-model export: retain access to the pinned base weights.

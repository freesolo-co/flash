# Packed SFT batch prefetch

The worker keeps its bounded data-loader process pool alive between epochs. Each
process prefetches two batches, so repeated epochs avoid process startup/import
stalls while the GPU consumes already tokenized, boundary-safe packed batches.
A synchronous `num_workers=0` loader remains supported. Global batch size, row
order, masks, packing boundaries, and optimizer accumulation stay unchanged.

Flash already enables Verl's dynamic token batching and remove-padding paths.
This change improves their producer pipeline; it does not raise token budgets or
change grouping to chase throughput. The pinned worker-image live test covered three epochs of 31 rows with two
persistent workers and an interrupted, mid-epoch loader restore: no rows were
skipped or replayed. A full Qwen3.5-9B SFT run crossed two epochs; restoring its
step-4 checkpoint reproduced the uninterrupted final losses and all 496 adapter
tensors exactly. These are correctness checks, not a throughput benchmark.

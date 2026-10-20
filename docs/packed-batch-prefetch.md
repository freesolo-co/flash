# Packed SFT batch prefetch

The worker keeps its bounded data-loader process pool alive between epochs. Each
process prefetches two batches, so repeated epochs avoid process startup/import
stalls while the GPU consumes already tokenized, boundary-safe packed batches.
A synchronous `num_workers=0` loader remains supported. Global batch size, row
order, masks, packing boundaries, and optimizer accumulation stay unchanged.

Flash already enables Verl's dynamic token batching and remove-padding paths.
This change improves their producer pipeline; it does not raise token budgets or
change grouping to chase throughput. Live epoch and interrupted-loader recovery
validation must pass before promotion, since prefetch must not skip or replay
training examples after a checkpoint.

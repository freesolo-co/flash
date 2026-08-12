"""How a flash run splits work across the cards it rented: by DATA, never by sequence.

One constant, shared by all three algorithms, because the reason it holds is a property of the
catalog rather than of any one trainer.
"""

from __future__ import annotations

# Ulysses sequence-parallel width for EVERY algorithm (sft, grpo, opd): off, always.
#
# Every catalog model is a Qwen3.5/3.6 GatedDeltaNet hybrid: linear attention plus a causal conv on
# most layers, whose state runs ALONG the sequence. verl gathers the full sequence only inside
# `_ulysses_flash_attention_forward` -- it patches `_flash_attention_forward` and nothing else -- so
# at a width > 1 the decoder hands each rank's SLICE straight to `self.linear_attn` with no gather,
# and every rank but rank 0 starts its recurrence mid-sequence from zero state. Measured 21%
# relative divergence on a 4-layer text model at width 2, with rank 0 bit-identical. Packed boundary
# resets cannot save it: ulysses slices at fixed `seqlen // width` offsets, blind to example
# boundaries.
#
# This costs no capacity. FSDP's mesh is built from world_size alone (`create_device_mesh`),
# independent of this width, so weights, grads and optimizer state still shard across every card and
# multi-card 32k is unaffected -- the cards become DATA-parallel ranks instead of sequence ranks.
#
# NOT the vllm rollout `tensor_model_parallel_size`, which still tracks the card count: tensor
# parallelism splits attention heads rather than the sequence, so it has no recurrent-state problem
# (and is why the allocator's head-divisibility cap is still load-bearing).
ULYSSES_SEQUENCE_PARALLEL_SIZE = 1

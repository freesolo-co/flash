"""The FSDP sharding strategy a multi-card run trains under, and what it costs per card.

ZeRO-3 (verl's default, ``reshard_after_forward=true``) re-gathers each parameter shard for the
backward and frees it again, so a card only ever holds its own slice. ZeRO-2 keeps the gathered copy
resident instead: one fewer all-gather per step, paid for in per-card memory.
"""

from __future__ import annotations

# FSDP shards parameters/optimizer across cards but replicates activations and adds collective
# buffers; a combination only counts as fitting when the combined VRAM clears the need with this
# margin. conservative on purpose: a too-optimistic shard model OOMs a paid run.
SHARD_VRAM_EFFICIENCY = 0.85
# activations/buffers replicate PER CARD regardless of sharding; every card in a combination must
# individually hold this floor on top of its parameter shard, so tiny cards cannot fake a fit by
# count alone (e.g. 4 x 12 GB is not 40 GB of usable capacity for a 24 GB-activation job).
REPLICATED_PER_CARD_GB = 8
# the public gpu.count maximum is 8, and every managed model's attention-head count is divisible by
# 8. keep the allocator aligned with that contract so a live 8-card provider SKU is reachable rather
# than silently clamping the authored ceiling to 4.
MAX_COMBINATION_CARDS = 8

# ZeRO-2 (verl `reshard_after_forward=false`) keeps each rank's full parameter copy resident after
# the forward instead of re-gathering it in the backward, which removes one all-gather per step.
# MEASURED on 2x A40 (pcie) and 2x A100-SXM (nvlink), 1.59B LoRA under Flash's own config with
# gradient checkpointing on: 1.377x and 1.150x step time, for a peak-memory cost of a CONSTANT
# 0.883 x the full bf16 weight size PER CARD, flat in card count (2.808 GB at n=2, 2.809 GB at n=4
# on the same model). Adding cards never amortizes it -- only the non-weight part shards -- so this
# is a per-card floor, exactly like `REPLICATED_PER_CARD_GB`, and not a divisible pool term.
#
# The "constant multiple of weight size" part is what the gate actually leans on, so it was measured
# a second time at a different scale: 3.37B on 2x RTX 4090, peak 5.824 -> 11.504 GB against a 6.735
# GB bf16 weight, i.e. 0.843x. Across a 2.1x weight increase the ratio moved -4.5%, so the penalty
# tracks weight size rather than growing super-linearly with it.
ZERO2_WEIGHT_RESIDENCY = 0.883
# What the GATE charges, which is deliberately more than what was measured. The measurements above
# span 1.59B-3.37B; the catalog runs up to 35B, so the gate still extrapolates the ratio about 10x
# beyond its evidence. Charging the measured value directly leaves the biggest models almost no room
# for that extrapolation to be wrong: 35B-A3B opd on 2x B200 clears the requirement by 0.4% and
# breaks if the true residency is 0.906 (2.6% above measured), and 27B grpo on 4x H200 breaks at
# 0.927 (5.0% above). An understatement there does not forgo a speedup, it OOMs a paid multi-card
# run that ZeRO-3 would have completed.
#
# 1.25 is above 1.0 -- a FULL retained bf16 copy plus a quarter -- so the charge stays conservative
# even if the residency is really "the whole weight copy and then some" at scale. It costs almost
# nothing: 155 of the 168 firing configs survive it, and the 13 it drops are exactly the ones
# sitting closest to the edge. Lower it only with matched multi-size measurements, never to admit a
# specific shape.
ZERO2_CHARGED_RESIDENCY = 1.25
_BF16_BYTES_PER_PARAM = 2.0


def zero2_replicated_floor_gb(params_b: float) -> float:
    """Per-card VRAM floor under ZeRO-2: the ZeRO-3 floor plus the retained weight copy.

    Charges ``ZERO2_CHARGED_RESIDENCY``, not the bare measured ``ZERO2_WEIGHT_RESIDENCY`` -- see
    that constant for why the extrapolation to catalog-scale models needs the margin.
    """
    if params_b <= 0:
        return float(REPLICATED_PER_CARD_GB)
    return REPLICATED_PER_CARD_GB + ZERO2_CHARGED_RESIDENCY * params_b * _BF16_BYTES_PER_PARAM


def combined_vram_gb(vram_gb: int, gpu_count: int, *, zero2_params_b: float = 0.0) -> float:
    """Return run-usable VRAM for a multi-card shape.

    All fit gates share this model. Return 0.0 when any card cannot hold the replicated floor; card
    count cannot compensate for that requirement.

    ``zero2_params_b`` prices the ZeRO-2 sharding policy: it does not change the SHAPE of the model,
    only the size of the per-card replicated floor (see ``zero2_replicated_floor_gb``). Default 0.0
    is the ZeRO-3 answer every existing caller already gets. Keeping this one function as the only
    fit model is the invariant that stops the allocator from admitting a shape on pooled capacity
    the worker's sharding strategy will not deliver.
    """
    if gpu_count <= 1:
        # PyTorch clamps every FSDP strategy to NO_SHARD at world size one, so ZeRO-2 and ZeRO-3
        # are the same run on one card and the penalty does not apply.
        return float(vram_gb)
    floor = (
        zero2_replicated_floor_gb(zero2_params_b)
        if zero2_params_b > 0
        else float(REPLICATED_PER_CARD_GB)
    )
    if vram_gb <= floor:
        return 0.0
    # the trailing addend is REPLICATED_PER_CARD_GB, never `floor`. what this returns is compared
    # against a need that was computed WITHOUT a retained weight copy, and that need decomposes as
    # `REPLICATED_PER_CARD_GB + shardable`. so the per-card constraint `floor + shardable/n <= vram`
    # rearranges to `need <= n * (vram - floor) * efficiency + REPLICATED_PER_CARD_GB`. adding the
    # ZeRO-2 floor back instead would refund exactly the retained copy this gate just charged, and
    # the gate would admit shapes whose true per-card demand exceeds the card.
    return gpu_count * (vram_gb - floor) * SHARD_VRAM_EFFICIENCY + REPLICATED_PER_CARD_GB


def zero2_enabled(vram_gb: int, executed_gpu_count: int, params_b: float, need_gb: float) -> bool:
    """THE rule for whether a run trains under ZeRO-2 instead of ZeRO-3.

    Called by the allocator (to price and admit the shape) and by the verl worker (to set
    ``actor_rollout_ref.actor.fsdp_config.reshard_after_forward``). It must stay a pure function of
    values BOTH sides hold, or the two drift and the allocator admits a run on pooled capacity the
    worker then overspends -- the failure mode that makes a worker-only flip unsafe.

    Enabled only when the shape keeps the run fitting with the ZeRO-2 floor already charged, so the
    speedup is never bought with headroom the run needs. The single-rank case is excluded because
    FSDP degrades to NO_SHARD there: no all-gather exists to remove, so the trade would be pure
    memory cost for no gain.
    """
    if executed_gpu_count <= 1 or params_b <= 0 or need_gb <= 0:
        return False
    return combined_vram_gb(vram_gb, executed_gpu_count, zero2_params_b=params_b) >= need_gb

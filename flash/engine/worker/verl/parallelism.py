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


def resolve_reshard_after_forward(
    *,
    model_id: str,
    algorithm: str,
    gpu_type: str,
    n_gpus: int,
    train=None,
    thinking: bool = False,
    model_revision: str = "",
) -> bool:
    """verl's ``reshard_after_forward`` for this run: False selects ZeRO-2, True keeps ZeRO-3.

    ZeRO-2 keeps each rank's parameter copy resident after the forward, removing one all-gather per
    step (measured 1.377x on pcie, 1.150x on nvlink) at a fixed per-card memory cost. That cost is
    only affordable on some shapes, so the decision belongs to the ALLOCATOR's fit model rather than
    to the worker: `zero2_enabled` is the one gate, and this function only re-asks it with the run's
    own resolved geometry.

    Answering it here rather than trusting a flag from the caller keeps a single source of truth
    when a run is recovered or re-launched onto a different card than it was first quoted for -- the
    policy follows the hardware the run actually lands on. Fails CLOSED to ZeRO-3 (verl's default)
    on any unknown card, unsizable model, or sizing error, because ZeRO-3 is the strictly
    lower-memory strategy and a wrong ZeRO-2 answer OOMs a paid run.
    """
    if n_gpus <= 1 or not (gpu_type or "").strip():
        return True
    try:
        from flash.engine.plan.vram import model_required_vram_gb, resolve_params_b
        from flash.providers.core.base import get_gpu_info
        from flash.providers.core.sharding import zero2_enabled

        # the PINNED revision's real weight count, not the catalog default: the retained copy is
        # priced off the weights the worker actually loads.
        params_b = float(resolve_params_b(model_id, model_revision) or 0.0)
        if params_b <= 0:
            return True
        need = float(
            model_required_vram_gb(
                model_id,
                algorithm,
                train=train,
                thinking=thinking,
                model_revision=model_revision,
            )
        )
        return not zero2_enabled(int(get_gpu_info(gpu_type).vram_gb), int(n_gpus), params_b, need)
    except Exception:
        return True

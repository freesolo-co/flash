"""KV-cache sizing for colocated vLLM rollout engines.

Split out of ``flash.engine.plan.vram`` to keep that module under the file-size limit. The
architecture-aware KV math is a self-contained unit: it answers "how much VRAM does the
rollout engine's KV pool need" and is consumed by the fit estimators next door.

The tuning constants are defined here and re-exported by ``flash.engine.plan.vram``, whose fit
estimators read them too; defining them here keeps this module importable on its own.
"""

from __future__ import annotations

import math

_KV_COEF = 2.0
_KV_CAP = 8.0
_KV_BLOCK_TOKENS = 16
# vllm profile, graph, scheduler, and allocator costs beyond raw cache tensors. the 8 gb measured
# floor remains authoritative at short context; longer contexts retain 1.5 gb plus 25% fragmentation.
_KV_PROFILE_OVERHEAD_GB = 1.5
_KV_FRAGMENTATION_MARGIN = 1.25
# the smallest share of a card vllm is given. below this the engine has no room for its own runtime
# once the kv pool is carved out, so a budget that prices under it is not a smaller engine but a
# broken one. the parameter-based estimate and multi-rank shard path both bottom out here.
_MIN_ENGINE_UTIL = 0.10
# vision-language rollout engines (catalog image-training models, e.g. Qwen3.5) need runtime memory
# beyond their bf16 weights before any kv page: vllm profiles the vision encoder on a maximum-size
# dummy image, reserves the encoder cache, and captures cuda graphs. measured lower bound: on an
# 80 GB H100, Qwen3.5-9B-Base GRPO at gpu_memory_utilization=0.3939 (31.5 GB budget, 19.3 GB weights)
# died with "No available memory for the cache blocks", so that non-kv runtime is >= 12.2 GB. 14 GB
# holds it with margin; it is charged to the engine before the kv pool in both the planner and the
# worker, so they size the same engine.
_VLM_ENGINE_RESERVE_GB = 14.0
# ceiling for a single-gpu sleep-mode vision engine whose reserve does not fit under the 0.45 cap.
# the trainer's resident bf16 copy (same size as the engine's weights) plus 10 GB of trainer headroom
# must still fit beside the lifted engine, so the lift can only take memory the card actually has.
_VLM_SINGLE_GPU_UTIL_CAP = 0.62
_TRAINER_HEADROOM_GB = 10.0


def vision_engine_reserve_gb(model_id: str) -> float:
    """Non-kv runtime reserve for a vision-language rollout engine; 0 for text-only models."""
    from flash.core.catalog import supports_image_training

    return _VLM_ENGINE_RESERVE_GB if supports_image_training(model_id) else 0.0


def _round_up(value: int, multiple: int) -> int:
    return max(multiple, math.ceil(max(1, value) / multiple) * multiple)


def _architecture_kv_raw_gb(
    model_info,
    vllm_max_len: int,
    num_generations: int,
    fp8_kv: bool,
) -> float | None:
    """Raw per-layer attention KV plus recurrent-state pages from catalog geometry."""
    attention_layers = int(getattr(model_info, "num_attention_layers", 0) or 0)
    kv_heads = int(getattr(model_info, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(model_info, "head_dim", 0) or 0)
    if not (attention_layers and kv_heads and head_dim):
        return None

    sequences = max(1, int(num_generations))
    seq_len = max(1, int(vllm_max_len))
    kv_bytes = 1 if fp8_kv else 2
    attention_bytes_per_token = 2 * kv_heads * head_dim * kv_bytes
    attention_tokens = _round_up(seq_len, _KV_BLOCK_TOKENS)
    total_bytes = attention_layers * sequences * attention_tokens * attention_bytes_per_token

    linear_layers = int(getattr(model_info, "num_linear_attention_layers", 0) or 0)
    if linear_layers:
        key_heads = int(getattr(model_info, "linear_num_key_heads", 0) or 0)
        value_heads = int(getattr(model_info, "linear_num_value_heads", 0) or 0)
        key_dim = int(getattr(model_info, "linear_key_head_dim", 0) or 0)
        value_dim = int(getattr(model_info, "linear_value_head_dim", 0) or 0)
        conv_kernel = int(getattr(model_info, "linear_conv_kernel_dim", 0) or 0)
        if not all((key_heads, value_heads, key_dim, value_dim, conv_kernel)):
            # linear-attention dims are absent from the catalog: we can't size the recurrent/conv
            # state, but the standard attention KV already accumulated in total_bytes is a better
            # (if partial) estimate than the parameter-based heuristic. The caller also retains
            # that heuristic as a floor when linear-attention geometry is incomplete.
            return total_bytes / 1e9
        # vllm stores qwen gated-deltanet recurrent and convolution state in bf16 pages.
        state_elements = value_heads * key_dim * value_dim
        state_elements += (key_heads * key_dim + value_heads * value_dim) * conv_kernel
        state_bytes = state_elements * 2
        state_block_tokens = _round_up(
            math.ceil(state_bytes / attention_bytes_per_token), _KV_BLOCK_TOKENS
        )
        catalog_block = int(getattr(model_info, "mamba_block_size", 0) or 0)
        if fp8_kv and catalog_block:
            state_block_tokens = max(state_block_tokens, catalog_block)
        state_pages = math.ceil(seq_len / state_block_tokens)
        total_bytes += linear_layers * sequences * state_pages * state_bytes

    return total_bytes / 1e9


def _resident_kv_gb(
    params_b: float | None,
    vllm_max_len: int,
    num_generations: int = 8,
    fp8_kv: bool = False,
    model_info=None,
    rollout_cache_min_gb: float = 0.0,
) -> float:
    """Estimate cache memory, keeping the runtime minimum independent of model geometry."""
    width = math.sqrt(max(float(params_b or 1.0), 0.1))
    parameter_estimate_gb = _KV_COEF * (max(1, vllm_max_len) / 1024.0) * width
    parameter_estimate_gb *= max(1, num_generations) / 8.0
    if fp8_kv:
        parameter_estimate_gb *= 0.5

    raw = _architecture_kv_raw_gb(model_info, vllm_max_len, num_generations, fp8_kv)
    if raw is None:
        return max(rollout_cache_min_gb, parameter_estimate_gb)
    profiled = max(
        _KV_CAP, rollout_cache_min_gb, _KV_PROFILE_OVERHEAD_GB + raw * _KV_FRAGMENTATION_MARGIN
    )
    declares_linear = int(getattr(model_info, "num_linear_attention_layers", 0) or 0) > 0
    linear_dims_known = all(
        int(getattr(model_info, _f, 0) or 0)
        for _f in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        )
    )
    if getattr(model_info, "sleep_unsupported", False) or (
        declares_linear and not linear_dims_known
    ):
        profiled = max(profiled, parameter_estimate_gb)
    return profiled


def _colocate_util_cap(weights_gb: float, total_vram_gb: float) -> float:
    """Utilization ceiling for the colocated vLLM executor budget.

    0.45 everywhere the blanket cap was tuned; lifted to 0.55 only when the executor carries a BIG
    weight copy (>=60 GB) AND the card leaves the trainer's resident copy room alongside the lifted
    budget (0.45*total - weights >= ~10 GB). See colocate_kv_util for the measured rationale."""
    big_weight_copy = weights_gb >= 60
    leaves_room_for_trainer_copy = (0.45 * total_vram_gb - weights_gb) >= 10.0
    return 0.55 if (big_weight_copy and leaves_room_for_trainer_copy) else 0.45


def _engine_util_cap(
    weights_gb: float,
    fixed_gb: float,
    total_vram_gb: float,
    *,
    vision_reserve_gb: float = 0.0,
    single_rank_sleep: bool = False,
) -> float:
    """The colocated engine's utilization ceiling, shared by the planner and the worker.

    ``fixed_gb`` is everything the engine holds before kv: weights, adapter, and the vision reserve.
    The standard cap applies unless a single-gpu sleep-mode vision engine cannot fit its fixed cost
    plus the minimum kv pool under it; only then is the cap lifted, never beyond what leaves the
    trainer's resident copy and its headroom free.
    """
    cap = _colocate_util_cap(weights_gb, total_vram_gb)
    card_gb = max(1.0, total_vram_gb)
    if not (vision_reserve_gb > 0 and single_rank_sleep):
        return cap
    if cap * card_gb >= fixed_gb + _KV_CAP:
        return cap
    room = (card_gb - weights_gb - _TRAINER_HEADROOM_GB) / card_gb
    return max(cap, min(_VLM_SINGLE_GPU_UTIL_CAP, room))


def _capped_engine_util(
    weights_gb: float,
    adapter_gb: float,
    requested_kv_gb: float,
    total_vram_gb: float,
    util_cap: float,
    reserve_gb: float = 0.0,
) -> float:
    """Fit weights, adapter, then KV into vllm's capped share of one card."""
    card_gb = max(1.0, total_vram_gb)
    max_engine_gb = util_cap * card_gb
    # gpu_memory_utilization is vllm's whole allocation, and its lora weights live inside it. reserve
    # weights plus adapter first; only a cap collision may reduce the requested kv pool. spelling out
    # fitted_kv_gb makes that loss visible instead of incidentally subtracting adapter bytes from a
    # weights-plus-kv fraction. if weights plus adapter alone exceed the cap, kv reaches zero and the
    # returned cap truthfully exposes that this engine shape has no viable cache budget.
    fitted_kv_gb = max(
        0.0, min(requested_kv_gb, max_engine_gb - weights_gb - adapter_gb - reserve_gb)
    )
    fitted_engine_gb = weights_gb + adapter_gb + reserve_gb + fitted_kv_gb
    return min(util_cap, fitted_engine_gb / card_gb)


def grpo_kv_floor_gb(
    params_b: float,
    vllm_max_len: int,
    group_size: int = 8,
    active_params_b: float | None = None,
    fp8_kv: bool = False,
    model_info=None,
    lora_rank: int = 0,
    tensor_parallel: int = 1,
    rollout_cache_min_gb: float = 0.0,
    vision_reserve_gb: float = 0.0,
    sleep_mode: bool = True,
) -> int:
    """Smallest card whose cap holds rank-local weights, adapter, vision reserve, and half the KV."""
    from flash.engine.plan.vram import _lora_weight_memory_gb

    kv_params_b = float(active_params_b) if active_params_b else float(params_b)
    tp_size = max(1, int(tensor_parallel))
    weights_gb = max(0.5, float(params_b)) * 2.0 / tp_size
    adapter_gb = _lora_weight_memory_gb(lora_rank, model_info, tensor_parallel=tp_size) or 0.0
    kv_gb = _resident_kv_gb(
        kv_params_b,
        vllm_max_len,
        group_size,
        fp8_kv=fp8_kv,
        model_info=model_info,
        rollout_cache_min_gb=rollout_cache_min_gb,
    )
    fixed_gb = weights_gb + adapter_gb + vision_reserve_gb
    need = fixed_gb + 0.5 * kv_gb
    single_rank_sleep = tp_size == 1 and sleep_mode
    lower = math.ceil(need / max(0.55, _VLM_SINGLE_GPU_UTIL_CAP))
    upper = math.ceil(need / 0.45)
    for total_vram_gb in range(lower, upper + 1):
        cap = _engine_util_cap(
            weights_gb,
            fixed_gb,
            total_vram_gb,
            vision_reserve_gb=vision_reserve_gb,
            single_rank_sleep=single_rank_sleep,
        )
        if cap * total_vram_gb >= need:
            return total_vram_gb
    return upper


def colocate_kv_util(
    params_b: float | None,
    vllm_max_len: int,
    total_vram_gb: float,
    sleep_mode: bool,
    num_generations: int = 8,
    active_params_b: float | None = None,
    fp8_kv: bool = False,
    model_info=None,
    tensor_parallel: int = 1,
    lora_rank: int = 0,
    rollout_cache_min_gb: float = 0.0,
    vision_reserve_gb: float = 0.0,
) -> float:
    """vllm_gpu_memory_utilization for one colocated rollout tensor-parallel rank.

    budget the rank's weight shard, rank-local bf16 lora footprint, and conservative full kv cache.
    catalog models use geometry with measured overhead and an 8 gb floor; uncataloged models retain
    the parameter-based estimate and no unverified adapter margin.
    """
    from flash.engine.plan.vram import _lora_weight_memory_gb

    total_weights_gb = max(0.5, float(params_b or 1.0)) * 2.0
    tp_size = max(1, int(tensor_parallel))
    adapter_gb = _lora_weight_memory_gb(lora_rank, model_info, tensor_parallel=tp_size) or 0.0
    # tensor parallelism shards vllm's bf16 weights, but kv heads can replicate when tp is wider than
    # the model's kv-head count. shard only the weight term and keep the full kv budget on every rank.
    weights_gb = total_weights_gb if tp_size == 1 else total_weights_gb / tp_size
    # catalog geometry is authoritative. active params remain only for the uncataloged fallback, while
    # the weight copy always uses total parameters.
    kv_params_b = float(active_params_b) if active_params_b else params_b
    # 0.45 starves KV for >=60 GB colocated weight copies on cards with residual headroom.
    # lift to 0.55 only when `0.45 * total_vram_gb - weight_copy_gb >= 10`; this admits B200
    # but not H200. `_NvidiaSmiPeakSampler` verifies runtime headroom.
    util_cap = _engine_util_cap(
        weights_gb,
        weights_gb + adapter_gb + vision_reserve_gb,
        total_vram_gb,
        vision_reserve_gb=vision_reserve_gb,
        single_rank_sleep=tp_size == 1 and sleep_mode,
    )
    kv_gb = _resident_kv_gb(
        kv_params_b,
        vllm_max_len,
        num_generations,
        fp8_kv=fp8_kv,
        model_info=model_info,
        rollout_cache_min_gb=rollout_cache_min_gb,
    )
    # Sleep mode keeps a larger pool (1.5x margin): the engine is offloaded during the backward, so a
    # bigger rollout-phase KV does not compete with the training peak.
    kv_pool_gb = max(_KV_CAP, kv_gb * (1.5 if sleep_mode else 1.0))
    sized = _capped_engine_util(
        weights_gb, adapter_gb, kv_pool_gb, total_vram_gb, util_cap, vision_reserve_gb
    )
    # Resident engines always retain a minimum allocation. Sleeping engines need it only when
    # tensor parallelism shrinks each rank's weight budget below the engine's runtime overhead.
    return max(_MIN_ENGINE_UTIL, sized) if not sleep_mode or tp_size > 1 else sized

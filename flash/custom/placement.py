"""Memory admission for one-process custom training with GPU-resident layer sharding."""

import math

MAX_GPUS = 2


def per_device_vram_gb(model_id, train, count, *, total_need=None):
    """Shard resident state, but reserve full activations/logits on every card.

    Unlike FSDP, layer placement does not distribute a vocabulary projection or
    optimizer tensors within one layer. Allow an extra layer and both vocabulary
    matrices for partition imbalance; never admit using pooled memory alone.
    """
    from flash.core.catalog import get_model
    from flash.engine.plan.vram import _lora_memory_gb, _positive_int_or_default, _sizing_value
    from flash.providers.core.allocator import required_vram_gb, vram_headroom

    need = (
        total_need if total_need is not None else required_vram_gb(model_id, "custom", train=train)
    )
    if count == 1:
        return need
    if count != MAX_GPUS:
        raise ValueError("custom training supports one or two Modal GPUs")
    info = get_model(model_id)
    rank = _positive_int_or_default(_sizing_value(train, "lora_rank"), 16)
    resident = info.params_b * 2 + _lora_memory_gb(
        rank, info.active_params_b or info.params_b, "custom", info
    )
    imbalance = resident / info.num_layers + 4 * info.vocab_size * info.hidden_size / 1e9
    return math.ceil(
        max(0, need - resident * vram_headroom()) + (resident / count + imbalance) * vram_headroom()
    )


def fitting_candidates(candidates, model_id, train, *, total_need):
    """Keep only allocations that fit the custom worker on every device."""
    return [
        candidate
        for candidate in candidates
        if candidate.vram_gb
        >= per_device_vram_gb(model_id, train, candidate.gpu_count, total_need=total_need)
    ]


def preview_gpu(model_id, train, max_count, *, total_need):
    from flash.providers.core.base import GPU_INFO, UnsupportedGpuError
    from flash.providers.core.sharding import combined_vram_gb
    from flash.providers.modal.pricing import hourly_rate

    shapes = [
        (hourly_rate(gpu.name) * count, count, gpu.name)
        for count in (1, 2)
        if count <= max_count
        for gpu in GPU_INFO.values()
        if gpu.validated
        and count <= gpu.max_count
        and combined_vram_gb(gpu.vram_gb, count) >= total_need
        and gpu.vram_gb >= per_device_vram_gb(model_id, train, count, total_need=total_need)
    ]
    if not shapes:
        raise UnsupportedGpuError("custom training cannot fit the per-device memory budget")
    return min(shapes)[2]


def allocation_shape(model_id, provider, providers, gpu_type, fallbacks, max_count):
    """Validate custom provider preferences before resolving the allocation shape."""
    from flash.providers.core.base import UnsupportedGpuError, custom_gpu_shape
    from flash.providers.core.registry import validated_provider_preferences

    if provider and providers:
        raise UnsupportedGpuError("provider and providers cannot both be set")
    validated_provider_preferences(providers, allow_empty=isinstance(providers, tuple))
    return custom_gpu_shape(
        model_id, provider, (gpu_type, *fallbacks) if gpu_type else (), max_count
    )

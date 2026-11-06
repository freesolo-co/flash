"""Keep a custom model's whole decoder layers on one or two local CUDA devices."""


def load_placement(model_id, rank, context, gpu_count):
    import torch

    from flash.custom.placement import MAX_GPUS, per_device_vram_gb

    if type(gpu_count) is not int or gpu_count not in (1, MAX_GPUS):
        raise ValueError("custom training supports one or two GPUs")
    if torch.cuda.device_count() < gpu_count:
        raise ValueError(f"custom training requires {gpu_count} visible GPUs")
    if gpu_count == 1:
        return {}
    need = per_device_vram_gb(
        model_id, {"lora_rank": rank, "max_context_tokens": context}, gpu_count
    )
    memory = {}
    for device in range(gpu_count):
        total = torch.cuda.get_device_properties(device).total_memory
        if total < need * 1e9:
            raise ValueError(f"custom training requires at least {need} GB on each GPU")
        memory[device] = int(total * 0.8)
    # Transformers/Accelerate put complete decoder blocks on devices and retain
    # autograd across their transfers. No CPU/disk offload is permitted below.
    return {"device_map": "balanced", "max_memory": memory}


def validate_placement(model, gpu_count):
    devices = {p.device for p in model.parameters()}
    if any(device.type != "cuda" for device in devices):
        raise ValueError("custom training requires GPU-resident weights; offload is unsupported")
    if len(devices) != gpu_count:
        raise ValueError("custom training did not place model weights on every allocated GPU")


def restore_cuda_rng(states):
    """Restore matching devices; seed additional shards from the saved primary RNG.

    Checkpoints can move between GPU shapes. Never address a saved device that
    does not exist on the new worker. Exact RNG replay applies to unchanged shapes.
    """
    import torch

    if states:
        for device in range(torch.cuda.device_count()):
            torch.cuda.set_rng_state(states[device] if device < len(states) else states[0], device)

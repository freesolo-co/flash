"""Fixed native vLLM cache policy for the Qwen serving family."""


def qwen_cache_args(model: str) -> dict:
    if not model.startswith(("Qwen/", "Freesolo-Co/Qwen")):
        return {}
    # vLLM owns the attention and recurrent-state layouts and includes LoRA and
    # multimodal identities in block hashes. ARC retains both recent and reused
    # CPU blocks; no DeepInfer block layout or scheduler patch is shared here.
    return {
        "enable_prefix_caching": True,
        # The connector needs stable GPU addresses even when PyTorch enables VMM.
        "enable_cumem_allocator": True,
        "kv_transfer_config": {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": 1024**3,
                "eviction_policy": "arc",
            },
        },
    }

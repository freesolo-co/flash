"""Retained Modal GPU implementation, registered only in modal/all provider modes."""

import os
import re

from flash.serving.lambda_cloud.profiles import engine_profile
from flash.serving.src.engine.lora_engine import _LoraEngineImpl


def selected_gpu(options, name, memory_bytes, count):
    """Verify the scheduler's actual card before choosing hardware-specific FP8 settings."""
    minimum_gib = {"A10G": 22, "L40S": 44, "A100-80GB": 75, "H200": 130}
    for gpu in options:
        family = "A10" if gpu == "A10G" else gpu.split("-")[0]
        if (
            count == 1
            and re.search(rf"\b{family}\b", name, re.IGNORECASE)
            and memory_bytes >= minimum_gib[gpu] * 2**30
        ):
            return gpu
    raise RuntimeError(f"Modal allocated an unapproved GPU for {options}: {name}")


class HostedModalEngine(_LoraEngineImpl):
    async def _load(self):
        import torch

        self.pinned_gpu = selected_gpu(
            self.pinned_gpu,
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory,
            torch.cuda.device_count(),
        )
        # AOT artifacts can encode hardware-specific quantization kernels.
        cache = os.environ.get("VLLM_CACHE_ROOT", "/vol/hosting-cache/vllm")
        os.environ["VLLM_CACHE_ROOT"] = f"{cache}/{self.pinned_gpu}"
        gpu = "A10" if self.pinned_gpu == "A10G" else self.pinned_gpu
        self.engine_overrides = engine_profile(self.base_model, gpu)
        await super()._load()

"""Retained Modal GPU implementation, registered only in modal/all provider modes."""

from flash.serving.lambda_cloud.profiles import engine_profile
from flash.serving.src.engine.lora_engine import _LoraEngineImpl


class HostedModalEngine(_LoraEngineImpl):
    async def _load(self):
        gpu = "A10" if self.pinned_gpu == "A10G" else self.pinned_gpu
        self.engine_overrides = engine_profile(self.base_model, gpu)
        await super()._load()

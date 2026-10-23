"""Adapter-only NCCL publication to dedicated vLLM GPUs.

The upstream remote path drops PEFT metadata and defaults to exporting base
weights. Both endpoints must agree on adapter mode before the first transfer.
"""

from copy import deepcopy

from verl.checkpoint_engine.base import CheckpointEngineWorker
from verl.single_controller.base.decorator import Dispatch, register

from flash.engine.worker.train.core.child.worker import FlashActorRolloutWorker


class PoolActorWorker(FlashActorRolloutWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_adapter_config(self):
        module = self.actor.engine.module
        module = getattr(module, "_fsdp_wrapped_module", module)
        config = getattr(module, "peft_config", {}).get("default")
        if config is None:
            raise RuntimeError("separate rollout requires a loaded training adapter")
        return config.to_dict()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps=None, mode="auto"):
        if mode != "nccl":
            raise ValueError("separate rollout adapter publication requires NCCL")
        engine = self.actor.engine
        try:
            weights, peft_config = engine.get_per_tensor_param(
                layered_summon=True, base_sync_done=True
            )
            if not peft_config:
                raise RuntimeError("refusing to publish base weights to an adapter rollout pool")
            return await self.checkpoint_engine.send_weights(weights, global_steps=global_steps)
        finally:
            if engine.is_param_offload_enabled:
                engine.to("cpu", model=True, optimizer=False, grad=False)


class PoolCheckpointWorker(CheckpointEngineWorker):
    def __init__(self, *args, **kwargs):
        # Ray unpickles the already-built model config without __post_init__.
        # Register the server adapter in this receiving process explicitly.
        from flash.engine.worker.train.core.child import registration  # noqa: F401

        super().__init__(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_adapter_config(self, config):
        if not isinstance(config, dict) or not config.get("r"):
            raise ValueError("missing training adapter metadata")
        self.adapter_config = deepcopy(config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps=None):
        config = getattr(self, "adapter_config", None)
        if not config:
            raise RuntimeError("adapter metadata must be installed before publication")
        weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)
        await self.server_adapter.update_weights(
            weights,
            global_steps=global_steps,
            peft_config=config,
            base_sync_done=True,
        )

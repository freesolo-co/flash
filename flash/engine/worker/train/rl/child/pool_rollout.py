"""Standalone vLLM replica with an adapter-aware checkpoint receiver."""

import ray
from verl.single_controller.ray import RayClassWithInitArgs

from flash.engine.worker.train.core.child.rollout import FlashVLLMReplica
from flash.engine.worker.train.rl.child.pool_weights import PoolCheckpointWorker


class PoolVLLMReplica(FlashVLLMReplica):
    def get_ray_class_with_init_args(self):
        return RayClassWithInitArgs(
            cls=ray.remote(PoolCheckpointWorker),
            rollout_config=self.config,
            model_config=self.model_config,
            replica_rank=self.replica_rank,
        )

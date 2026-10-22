"""Register Flash's rollout type using the extension contract documented by upstream."""

from verl.workers.rollout.base import _ROLLOUT_REGISTRY
from verl.workers.rollout.replica import RolloutReplicaRegistry


def load_replica():
    from flash.engine.worker.train.core.child.rollout import FlashVLLMReplica

    return FlashVLLMReplica


RolloutReplicaRegistry.register("flash_vllm", load_replica)
_ROLLOUT_REGISTRY[("flash_vllm", "async")] = "verl.workers.rollout.vllm_rollout.ServerAdapter"

"""Synchronous GRPO with dedicated, independently sized rollout replicas."""

import ray
from verl.checkpoint_engine import CheckpointEngineManager
from verl.trainer.ppo.utils import Role
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.rollout.llm_server import LLMServerManager

from flash.engine.worker.train.rl.child.pool_weights import PoolActorWorker
from flash.engine.worker.train.rl.child.trainer import FlashGrpoTrainer
from flash.engine.worker.train.rl.launch.pool import RolloutPool


class PoolServerManager(LLMServerManager):
    def __init__(self, *args, **kwargs):
        from flash.engine.worker.train.rl.child.pool_rollout import PoolVLLMReplica

        self.rollout_replica_class = PoolVLLMReplica
        super().__init__(*args, **kwargs)


class FlashSeparateGrpoTrainer(FlashGrpoTrainer):
    # The upstream base creates Actor workers and an empty rollout manager. No
    # inference engine or hybrid weight-sync endpoint is placed on training GPUs.
    _enable_hybrid_replicas = False

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        self.role_worker_mapping[Role.Actor] = ray.remote(PoolActorWorker)

    def _setup(self):
        super()._setup()
        # super loaded the recovery checkpoint, but has not called on_init_end.
        # The first publication below therefore includes restored/warm-start weights.
        configs = self.actor_rollout_wg.get_adapter_config()
        if not configs or any(config != configs[0] for config in configs):
            raise RuntimeError("training ranks disagree on adapter metadata")
        empty_manager = self.llm_server_manager
        self.llm_server_manager = PoolServerManager.create(config=self.config)
        # The empty manager has no GPU actors, only its now-unused router.
        ray.kill(empty_manager.global_load_balancer)
        replicas = self.llm_server_manager.get_replicas()
        ray.get(
            [
                worker.set_adapter_config.remote(configs[0])
                for replica in replicas
                for worker in replica.workers
            ]
        )
        self.checkpoint_manager = CheckpointEngineManager(
            config=omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine),
            actor_wg=self.actor_rollout_wg,
            replicas=replicas,
        )
        # Inherited sync hooks publish once here and after every update, waiting
        # for every replica before admitting the next batch. A failed publication
        # raises and prevents readiness / further generation.

    def on_sample_end(self):
        # Dedicated GPUs retain their base weights and KV allocation. The remote
        # checkpoint manager clears caches and gates requests at publication.
        if self.curr_step_profile:
            self._stop_rollout_profiling()

    def _get_n_gpus_for_throughput(self):
        pool = RolloutPool.decode(self.runtime.rollout_pool)
        return super()._get_n_gpus_for_throughput() + pool.gpus

    def _consume_sync_metrics(self):
        pool = RolloutPool.decode(self.runtime.rollout_pool)
        return {
            **super()._consume_sync_metrics(),
            "rollout_pool/gpus": pool.gpus,
            "rollout_pool/replicas": pool.replicas,
            "rollout_pool/training_gpus": self.runtime.dp_cards,
            "rollout_pool/published_step": self.global_steps,
        }

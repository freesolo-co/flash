"""Shared synchronous trainer policy using upstream Verl's V1 lifecycle hooks."""

import importlib

from omegaconf import OmegaConf
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

from flash.engine.worker.train.core.child.options import RuntimeOptions
from flash.engine.worker.train.core.child.readiness import publish_ready
from flash.engine.worker.train.core.child.tracking import publish_tracking_link
from flash.engine.worker.train.opd.child.policy import _init_transfer_queue
from flash.engine.worker.train.opd.child.replay import BoundedReplayBuffer


class FlashSyncTrainer(PPOTrainerSync):
    def __init__(self, config):
        self.runtime = RuntimeOptions.decode(dict(config.actor_rollout_ref.model.runtime_options))
        super().__init__(config)

    def _build_replay_buffer(self):
        sampler = self.config.trainer.v1.sampler
        return BoundedReplayBuffer(
            self,
            trainer_mode="sync",
            trainer_config=self.config.trainer.v1.sync,
            max_off_policy_threshold=sampler.max_off_policy_threshold,
            max_off_policy_strategy=sampler.max_off_policy_strategy,
            sampler_kwargs=sampler.sampler_kwargs,
            refill_fn=self._add_prompts_to_generate,
        )

    def _init_resource_pool_mgr(self):
        import ray

        from flash.engine.worker.train.core.child.worker import FlashActorRolloutWorker

        super()._init_resource_pool_mgr()
        for role in (Role.Actor, Role.ActorRollout, Role.ActorRolloutRef, Role.RefPolicy):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = ray.remote(FlashActorRolloutWorker)

    def _save_checkpoint(self):
        if self.runtime.should_save(int(self.global_steps)):
            return super()._save_checkpoint()
        return None

    def on_train_begin(self):
        publish_tracking_link(self.logger)
        return super().on_train_begin()


class FlashTaskRunner:
    def run(self, config):
        import transfer_queue as tq

        options = RuntimeOptions.decode(dict(config.actor_rollout_ref.model.runtime_options))
        module = "rl" if options.algorithm == "grpo" else "opd"
        name = "FlashGrpoTrainer" if module == "rl" else "FlashPPOTrainer"
        trainer_type = getattr(
            importlib.import_module(f"flash.engine.worker.train.{module}.child.trainer"), name
        )
        manager_type = getattr(
            importlib.import_module(f"flash.engine.worker.train.{module}.child.rollout"),
            "FlashGrpoManager" if module == "rl" else "FlashOpdManager",
        )
        OmegaConf.resolve(config)
        trainer = None
        succeeded = False
        # A timed-out init still owns a blocked daemon thread. Only enter queue cleanup
        # after successful initialization; close must never race that thread.
        _init_transfer_queue(tq.init, config.transfer_queue)
        try:
            trainer = trainer_type(config=config)
            trainer.init()
            manager = manager_type.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            publish_ready(options)
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
                if getattr(trainer, "_dump_executor", None) is not None:
                    trainer._shutdown_dump_executor()
            finally:
                # The pinned replay buffer polls synchronously and owns no thread or close
                # operation. The dump executor above must finish before the queue.
                tq.close()

"""Shared synchronous trainer policy using upstream Verl's V1 lifecycle hooks."""

import contextlib

from omegaconf import OmegaConf
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

from flash.engine.worker.runtime.phases import training_phase
from flash.engine.worker.train.core.child.options import RuntimeOptions
from flash.engine.worker.train.core.child.queue import _init_transfer_queue
from flash.engine.worker.train.core.child.readiness import publish_ready
from flash.engine.worker.train.core.child.replay import BoundedReplayBuffer
from flash.engine.worker.train.core.child.tracking import publish_tracking_link
from flash.engine.worker.train.core.lifecycle.timing import checkpoint_phase


class FlashSyncTrainer(PPOTrainerSync):
    def __init__(self, config):
        self.runtime = RuntimeOptions.decode(dict(config.actor_rollout_ref.model.runtime_options))
        super().__init__(config)

    def step(self, metrics, timing_raw):
        first = not getattr(self, "_flash_update_seen", False)
        with training_phase("first_update" if first else "steady_update", step=self.global_steps):
            result = super().step(metrics, timing_raw)
        self._flash_update_seen = True
        return result

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
            with checkpoint_phase(int(self.global_steps), "capture"):
                return super()._save_checkpoint()
        return None

    def on_train_begin(self):
        publish_tracking_link(self.logger)
        return super().on_train_begin()


class FlashTaskRunner:
    def run(self, config):
        import transfer_queue as tq

        options = RuntimeOptions.decode(dict(config.actor_rollout_ref.model.runtime_options))
        from flash.engine.worker.train.rl.child.rollout import FlashGrpoManager
        from flash.engine.worker.train.rl.child.trainer import FlashGrpoTrainer

        OmegaConf.resolve(config)
        trainer = None
        succeeded = False
        # A timed-out init still owns a blocked daemon thread. Only enter queue cleanup
        # after successful initialization; close must never race that thread.
        _init_transfer_queue(tq.init, config.transfer_queue)
        try:
            with training_phase("trainer_initialization"):
                trainer = FlashGrpoTrainer(config=config)
                trainer.init()
            manager = FlashGrpoManager.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            publish_ready(options)
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(trainer, "logger", None)
                if tracking is not None:
                    # Optional telemetry must not hide the training outcome or skip draining.
                    with contextlib.suppress(Exception):
                        tracking.finish(exit_code=0 if succeeded else 1)
                if getattr(trainer, "_dump_executor", None) is not None:
                    trainer._shutdown_dump_executor()
            finally:
                # The pinned replay buffer polls synchronously and owns no thread or close
                # operation. The dump executor above must finish before the queue.
                tq.close()

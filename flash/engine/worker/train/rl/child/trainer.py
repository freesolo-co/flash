"""GRPO trainer policy through explicit Verl extension methods."""

from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from flash.engine.worker.train.core.child.options import RuntimeOptions
from flash.engine.worker.train.core.child.readiness import publish_ready
from flash.engine.worker.train.core.child.tracking import TrackingOwner
from flash.engine.worker.train.rl.child.credit import apply_per_turn_credit


def require_trainable_responses(batch):
    response_mask = batch.batch.get("response_mask")
    if response_mask is not None and not bool(response_mask.any()):
        raise RuntimeError(
            "no trainable response tokens remain: every rollout was truncated or unusable; "
            "increase train.max_completion_tokens or disable thinking"
        )


class FlashGrpoTrainer(TrackingOwner, RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime = RuntimeOptions.decode(
            dict(self.config.actor_rollout_ref.model.runtime_options)
        )

    def init_workers(self):
        super().init_workers()
        publish_ready(self.runtime)

    def _compute_advantage(self, *args, **kwargs):
        data = super()._compute_advantage(*args, **kwargs)
        return apply_per_turn_credit(data) if self.runtime.per_turn_credit else data

    def _apply_rollout_correction(self, batch, config):
        require_trainable_responses(batch)
        return super()._apply_rollout_correction(batch, config)

    def _update_actor(self, batch):
        require_trainable_responses(batch)
        return super()._update_actor(batch)

    def _save_checkpoint(self):
        if self.runtime.should_save(int(self.global_steps)):
            return super()._save_checkpoint()
        return None


class FlashGrpoTaskRunner(TaskRunner):
    trainer_class = FlashGrpoTrainer

    def add_actor_rollout_worker(self, config):
        import ray

        from flash.engine.worker.train.core.child.worker import FlashActorRolloutWorker

        _original, worker_group = super().add_actor_rollout_worker(config)
        self.role_worker_mapping = {
            role: ray.remote(FlashActorRolloutWorker) for role in self.role_worker_mapping
        }
        return FlashActorRolloutWorker, worker_group

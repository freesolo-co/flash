"""OPD trainer with explicit teacher cycles, checkpoint policy, and replay ownership."""

import os

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from verl.single_controller.ray import ResourcePoolManager
from verl.trainer.main_ppo_sync import KVBatchMeta, PPOTrainer, tq
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.workers.engine_workers import TrainingWorker

from flash.engine.worker.train.core.child.options import RuntimeOptions
from flash.engine.worker.train.core.child.readiness import publish_ready
from flash.engine.worker.train.core.child.tracking import TrackingOwner
from flash.engine.worker.train.core.child.worker import FlashActorRolloutWorker
from flash.engine.worker.train.opd.child.bridge import (
    _post_no_signal_abandoned,
    _post_no_signal_resample,
    _update_actor_after_teacher_cycle_commit,
)
from flash.engine.worker.train.opd.child.policy import (
    _AllNoSignalBatch,
    _full_sequence_signal_sequences,
    _init_transfer_queue,
    _run_with_no_signal_replacements,
)
from flash.engine.worker.train.opd.child.replay import BoundedReplayBuffer


def filter_signal_batch(batch):
    fields = tq.kv_batch_get(
        keys=batch.keys,
        partition_id=batch.partition_id,
        select_fields=["teacher_ids"],
    )
    group_ids = fields["teacher_ids"]
    if group_ids.is_nested:
        keep = torch.tensor(
            [row.ge(0).any().item() for row in group_ids.unbind()], dtype=torch.bool
        )
    else:
        keep = _full_sequence_signal_sequences(group_ids)
    keys = [key for key, selected in zip(batch.keys, keep.tolist(), strict=True) if selected]
    tags = [tag for tag, selected in zip(batch.tags, keep.tolist(), strict=True) if selected]
    if not keys:
        raise _AllNoSignalBatch(batch)
    return KVBatchMeta(
        keys=keys,
        tags=tags,
        partition_id=batch.partition_id,
        fields=batch.fields,
        extra_info=batch.extra_info,
    )


class FlashPPOTrainer(TrackingOwner, PPOTrainer):
    def _create_replay_buffer(self):
        return BoundedReplayBuffer(self)

    def _save_checkpoint(self):
        options = RuntimeOptions.decode(dict(self.config.actor_rollout_ref.model.runtime_options))
        if options.should_save(int(self.global_steps)):
            return super()._save_checkpoint()
        return None

    def init_workers(self):
        self.use_teacher_policy = False
        super().init_workers()
        self.use_teacher_policy = True
        self.distillation_config = omega_conf_to_dataclass(self.config.distillation)
        publish_ready(
            RuntimeOptions.decode(dict(self.config.actor_rollout_ref.model.runtime_options))
        )

    def step(self, batch_dict, metrics, timing_raw):
        def run_attempt(attempt_ordinal: int):
            batch_dict["flash_no_signal_attempt"] = np.full(
                len(batch_dict["raw_prompt"]), attempt_ordinal, dtype=np.int64
            )
            return super(FlashPPOTrainer, self).step(batch_dict, metrics, timing_raw)

        def cleanup_attempt(batch) -> None:
            prompt_keys = [str(uid) for uid in batch_dict["uid"]]
            keys = list(dict.fromkeys([*batch.keys, *prompt_keys]))
            tq.kv_clear(keys=keys, partition_id=batch.partition_id)
            self.replay_buffer.remove(batch.partition_id, keys)

        def prepare_replacement() -> None:
            self.checkpoint_manager.update_weights()

        return _run_with_no_signal_replacements(
            run_attempt,
            cleanup_attempt,
            prepare_replacement,
            lambda: _post_no_signal_resample(
                os.environ["FLASH_OPD_BRIDGE_URL"],
                os.environ["FLASH_OPD_BRIDGE_TOKEN"],
            ),
            lambda: _post_no_signal_abandoned(
                os.environ["FLASH_OPD_BRIDGE_URL"],
                os.environ["FLASH_OPD_BRIDGE_TOKEN"],
            ),
            max_attempts=int(
                self.config.actor_rollout_ref.model.runtime_options.no_signal_attempts
            ),
        )

    def _compute_reward_colocate(self, batch, metrics):
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["response_mask"],
        )
        data["rm_scores"] = torch.zeros_like(data["response_mask"], dtype=torch.float32)
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=data.select("rm_scores"),
        )
        return batch

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        return dp_size

    def _balance_batch(self, batch, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        return super()._balance_batch(
            filter_signal_batch(batch),
            metrics,
            logging_prefix=logging_prefix,
            keep_minibatch=keep_minibatch,
        )

    def _update_actor(self, batch, metrics):
        real_batch_size = sum(not tag.get("is_padding", False) for tag in batch.tags)
        extra_info = {
            "calculate_entropy": self.config.actor_rollout_ref.actor.calculate_entropy,
            "distillation_use_topk": False,
            "global_batch_size": real_batch_size,
            "mini_batch_size": len(batch),
            "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
            "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
            "dataloader_kwargs": {"shuffle": False},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        batch.extra_info.update(extra_info)
        output = _update_actor_after_teacher_cycle_commit(
            self.actor_rollout_wg,
            batch,
            os.environ["FLASH_OPD_BRIDGE_URL"],
            os.environ["FLASH_OPD_BRIDGE_TOKEN"],
        )
        output = rename_dict(output["metrics"], "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        metrics.update(reduce_metrics(output))
        return batch


@ray.remote
class FlashTaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = (
            lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )
        role = (
            Role.ActorRolloutRef
            if need_reference_policy(config) and not ref_in_actor
            else Role.ActorRollout
        )
        self.role_worker_mapping[role] = ray.remote(FlashActorRolloutWorker)
        self.mapping[role] = "global_pool"

    def add_critic_worker(self, config):
        if need_critic(config):
            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        }
        config.reward.reward_model.nnodes = config.trainer.nnodes
        config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
        self.mapping[Role.RewardModel] = "global_pool"
        self.resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=self.mapping
        )

    def run(self, config):
        OmegaConf.resolve(config)
        _init_transfer_queue(tq.init, config.transfer_queue)
        trainer = None
        try:
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.init_resource_pool_mgr(config)
            trainer = FlashPPOTrainer(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.resource_pool_manager,
            )
            trainer.init_workers()
            trainer.fit()
        finally:
            if trainer:
                trainer.replay_buffer.close()
            tq.close()

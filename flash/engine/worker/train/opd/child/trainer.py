"""OPD trainer with explicit teacher cycles, checkpoint policy, and replay ownership."""

import copy
import os

import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict

from flash.engine.worker.train.core.child.trainer import FlashSyncTrainer
from flash.engine.worker.train.opd.child.bridge import (
    _post_no_signal_abandoned,
    _post_no_signal_resample,
    _update_actor_after_teacher_cycle_commit,
)
from flash.engine.worker.train.opd.child.policy import (
    _AllNoSignalBatch,
    _full_sequence_signal_sequences,
    _run_with_no_signal_replacements,
)


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


class FlashPPOTrainer(FlashSyncTrainer):
    def _init_resource_pool_mgr(self):
        original = self.config
        self.config = copy.deepcopy(original)
        self.config.distillation.enabled = False
        try:
            super()._init_resource_pool_mgr()
        finally:
            self.config = original

    def _setup(self):
        self.use_teacher_policy = False
        super()._setup()
        self.distillation_config = omega_conf_to_dataclass(self.config.distillation)

    def get_teacher_client(self):
        return None

    def _next_train_batch(self, num_prompts=None):
        if self._replacement_prompts is None:
            self._replacement_prompts = super()._next_train_batch(num_prompts)
        batch = self._replacement_prompts.clone()
        tu.assign_non_tensor_data(batch, "flash_no_signal_attempt", self._no_signal_attempt)
        return batch

    def step(self, metrics, timing_raw):
        self._replacement_prompts = None

        def run_attempt(attempt_ordinal):
            self._no_signal_attempt = attempt_ordinal
            return super(FlashPPOTrainer, self).step(metrics, timing_raw)

        def cleanup_attempt(batch):
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

        def prepare_replacement():
            self.checkpoint_manager.update_weights(self.global_steps - 1)

        try:
            return _run_with_no_signal_replacements(
                run_attempt,
                cleanup_attempt,
                prepare_replacement,
                lambda: _post_no_signal_resample(
                    os.environ["FLASH_OPD_BRIDGE_URL"], os.environ["FLASH_OPD_BRIDGE_TOKEN"]
                ),
                lambda: _post_no_signal_abandoned(
                    os.environ["FLASH_OPD_BRIDGE_URL"], os.environ["FLASH_OPD_BRIDGE_TOKEN"]
                ),
                max_attempts=self.runtime.no_signal_attempts,
            )
        finally:
            self._replacement_prompts = None

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
        selected = filter_signal_batch(batch)
        rejected = sorted(set(batch.keys) - set(selected.keys))
        if rejected:
            tq.kv_clear(keys=rejected, partition_id=batch.partition_id)
        return super()._balance_batch(
            selected,
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

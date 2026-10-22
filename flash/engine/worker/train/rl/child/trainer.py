"""GRPO credit and trainability checks on the upstream V1 trainer."""

import numpy as np
import torch
import transfer_queue as tq
from tensordict import TensorDict
from verl import DataProto
from verl.workers.utils.padding import response_to_nested

from flash.engine.worker.train.core.child.trainer import FlashSyncTrainer
from flash.engine.worker.train.rl.child.credit import apply_per_turn_credit


def require_trainable_responses(batch):
    fields = tq.kv_batch_get(
        keys=batch.keys, partition_id=batch.partition_id, select_fields=["response_mask"]
    )
    mask = fields["response_mask"]
    values = mask.values() if mask.is_nested else mask
    if not bool(values.any()):
        raise RuntimeError(
            "no trainable response tokens remain: every rollout was truncated or unusable; "
            "increase train.max_completion_tokens or disable thinking"
        )


class FlashGrpoTrainer(FlashSyncTrainer):
    def _compute_advantage(self, batch, metrics):
        require_trainable_responses(batch)
        batch = super()._compute_advantage(batch, metrics)
        if self.runtime.per_turn_credit:
            apply_queue_per_turn_credit(batch)
        return batch

    def _update_actor(self, batch, metrics):
        # rollout correction can reject the last trainable tokens after advantage calculation.
        require_trainable_responses(batch)
        return super()._update_actor(batch, metrics)


def _object_column(values):
    column = np.empty(len(values), dtype=object)
    column[:] = values
    return column


def apply_queue_per_turn_credit(batch):
    fields = tq.kv_batch_get(
        keys=batch.keys,
        partition_id=batch.partition_id,
        select_fields=["advantages", "response_mask", "uid", "extra_fields"],
    )
    mask = fields["response_mask"]
    nested = mask.is_nested
    episode = fields["advantages"].to_padded_tensor(0) if nested else fields["advantages"]
    response_mask = mask.to_padded_tensor(0) if nested else mask
    extras = list(fields["extra_fields"])
    data = DataProto(
        batch=TensorDict(
            {"advantages": episode, "response_mask": response_mask}, batch_size=len(batch)
        ),
        non_tensor_batch={
            "uid": np.array(list(fields["uid"]), dtype=object),
            "flash_turn_spans": _object_column([item.get("flash_turn_spans") for item in extras]),
            "flash_turn_rewards": _object_column(
                [item.get("flash_turn_rewards") for item in extras]
            ),
        },
    )
    apply_per_turn_credit(data)
    advantages = data.batch["advantages"]
    if not bool(torch.isfinite(advantages).all()):
        raise ValueError("per-turn advantages must be finite")
    values = response_to_nested(advantages, mask) if nested else advantages
    tq.kv_batch_put(
        keys=batch.keys,
        partition_id=batch.partition_id,
        fields=TensorDict({"advantages": values, "returns": values}, batch_size=len(batch)),
    )

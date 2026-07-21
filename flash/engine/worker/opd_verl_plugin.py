"""Standalone verl 0.8.0 extensions for flash OPD.

This file is copied into the isolated verl child workdir. It registers the exact flash groupwise
reverse-KL loss, routes teacher scoring through the parent localhost bridge, supplies deterministic
single-turn sampling, and removes the otherwise mandatory local teacher GPU pool.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib.util
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any


def deterministic_rollout_seed(
    flash_seed: int, global_step: int, example_index: int, rollout_ordinal: int
) -> int:
    """Derive one stable vLLM request seed from the complete rollout identity."""
    payload = f"{int(flash_seed)}:{int(global_step)}:{int(example_index)}:{int(rollout_ordinal)}"
    digest = hashlib.blake2b(payload.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _signal_sequences(group_ids, response_mask):
    """Return the per-sequence mask for rows carrying at least one aligned student token."""
    return ((group_ids >= 0) & response_mask.bool()).any(dim=-1)


def _full_sequence_signal_sequences(group_ids):
    """Detect aligned metadata in native full-sequence teacher tensors."""
    return group_ids.ge(0).flatten(start_dim=1).any(dim=-1)


def _flash_groupwise_reverse_kl_values(
    student_logprobs,
    teacher_logsums,
    group_ids,
    response_mask,
    kl_penalty_coef: float,
):
    """Build response-shaped loss values whose verl aggregation equals flash's sequence mean."""
    import torch

    if student_logprobs.shape != teacher_logsums.shape or student_logprobs.shape != group_ids.shape:
        raise ValueError("flash OPD teacher metadata must match student response logprob shape")
    if response_mask.shape != student_logprobs.shape:
        raise ValueError("flash OPD response mask must match student response logprob shape")
    values = torch.zeros_like(student_logprobs)
    response_mask = response_mask.bool()
    for row in range(student_logprobs.shape[0]):
        selected = response_mask[row] & group_ids[row].ge(0)
        selected_count = int(selected.sum().item())
        if selected_count == 0:
            continue
        response_count = int(response_mask[row].sum().item())
        row_groups = group_ids[row]
        for group_id in torch.unique(row_groups[selected], sorted=True):
            group_mask = selected & row_groups.eq(group_id)
            group_length = group_mask.sum().to(dtype=student_logprobs.dtype)
            student_logsum = student_logprobs[row][group_mask].detach().sum()
            teacher_logsum = teacher_logsums[row][group_mask][0]
            coefficient = (
                float(kl_penalty_coef)
                * (student_logsum - teacher_logsum)
                / group_length
            )
            values[row][group_mask] = coefficient * student_logprobs[row][group_mask]
        values[row][selected] *= response_count / selected_count
    return values


def _set_current_global_batch_info(config, data) -> None:
    """Mirror verl 0.8.0 ppo_loss metadata population for the current microbatch."""
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor


def _post_json(url: str, token: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def _install_verl_extensions() -> None:
    import ray
    import torch
    from omegaconf import OmegaConf

    try:
        import transfer_queue as tq
        from transfer_queue import KVBatchMeta
    except ImportError:
        from verl.utils.transferqueue_utils import KVBatchMeta, tq

    from verl.experimental.agent_loop.agent_loop import register
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from verl.single_controller.ray import ResourcePoolManager
    from verl.trainer.distillation.losses import (
        DistillationLossSettings,
        register_distillation_loss,
    )
    from verl.trainer.main_ppo_sync import PPOTrainer
    from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.metric import AggregationType, Metric, reduce_metrics
    from verl.utils.py_functional import rename_dict
    from verl.workers.config import DistillationConfig, DistillationLossConfig
    from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
    from verl.workers.utils.padding import no_padding_2_padding

    @register_distillation_loss(
        DistillationLossSettings(names=["flash_groupwise_reverse_kl"], use_estimator=True)
    )
    def flash_groupwise_reverse_kl(config, distillation_config, model_output, data):
        _set_current_global_batch_info(config, data)
        student_logprobs = no_padding_2_padding(model_output["log_probs"], data)
        teacher_logsums = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
        group_ids = no_padding_2_padding(data["teacher_ids"], data).squeeze(-1).long()
        response_mask = data["response_mask"]
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        losses = _flash_groupwise_reverse_kl_values(
            student_logprobs,
            teacher_logsums,
            group_ids,
            response_mask,
            distillation_config.kl_penalty_coef,
        )
        selected = (group_ids >= 0) & response_mask.bool()
        selected_losses = losses[selected]
        mean_abs = selected_losses.abs().mean() if selected_losses.numel() else losses.new_zeros(())
        return losses, {
            "distillation/abs_loss": Metric(AggregationType.MEAN, mean_abs),
            "distillation/signal_sequences": Metric(
                AggregationType.SUM, _signal_sequences(group_ids, response_mask).sum()
            ),
        }

    @dataclass
    class FlashRemoteDistillationConfig(DistillationConfig):
        bridge_url: str = ""
        bridge_token: str = ""
        kl_penalty_coef: float = 1.0

        def __post_init__(self):
            if not self.enabled:
                return
            self.distillation_loss = omega_conf_to_dataclass(
                self.distillation_loss, dataclass_type=DistillationLossConfig
            )
            if self.distillation_loss.loss_mode != "flash_groupwise_reverse_kl":
                raise ValueError("flash remote distillation requires flash_groupwise_reverse_kl")
            if not self.bridge_url or not self.bridge_token:
                raise ValueError("flash remote distillation bridge configuration is missing")
            if self.kl_penalty_coef <= 0:
                raise ValueError("flash remote distillation kl_penalty_coef must be positive")
            self.teacher_models = {}

    FlashRemoteDistillationConfig.__module__ = __name__
    globals()["FlashRemoteDistillationConfig"] = FlashRemoteDistillationConfig

    class FlashBridgeTeacherManager:
        def __init__(self, config, teacher_client=None):
            resolved = omega_conf_to_dataclass(config.distillation)
            self.bridge_url = resolved.bridge_url
            self.bridge_token = resolved.bridge_token

        async def compute_teacher_logprobs_single(
            self,
            sequence_ids: list[int],
            multi_modal_data=None,
            mm_processor_kwargs=None,
            routing_key=None,
        ):
            if multi_modal_data:
                raise RuntimeError("flash OPD bridge does not accept child-side multimodal payloads")
            if routing_key is None:
                raise RuntimeError("flash OPD bridge requires the indexed dataset row")
            payload = await asyncio.to_thread(
                _post_json,
                self.bridge_url,
                self.bridge_token,
                "/score",
                {
                    "index": int(routing_key),
                    "sequence_ids": [int(token_id) for token_id in sequence_ids],
                },
            )
            teacher_ids = torch.tensor(payload["teacher_ids"], dtype=torch.int32).unsqueeze(-1)
            teacher_logprobs = torch.tensor(
                payload["teacher_logprobs"], dtype=torch.float32
            ).unsqueeze(-1)
            if teacher_ids.shape != teacher_logprobs.shape:
                raise RuntimeError("flash OPD bridge returned inconsistent teacher tensors")
            if teacher_ids.shape[0] != len(sequence_ids):
                raise RuntimeError("flash OPD bridge returned the wrong sequence length")
            return teacher_ids, teacher_logprobs

    import verl.experimental.teacher_loop.teacher_manager as teacher_manager_module

    teacher_manager_module.AsyncTeacherLLMServerManager = FlashBridgeTeacherManager

    @register("flash_single_turn")
    class FlashSingleTurnAgentLoop(SingleTurnAgentLoop):
        async def run(self, sampling_params: dict[str, Any], **kwargs):
            params = dict(sampling_params)
            flash_seed = int(os.environ["FLASH_OPD_SEED"])
            global_step = int(kwargs["global_steps"])
            example_index = int(kwargs["index"])
            rollout_ordinal = int(kwargs.get("session_id", 0))
            params["seed"] = deterministic_rollout_seed(
                flash_seed, global_step, example_index, rollout_ordinal
            )
            stops = json.loads(os.environ.get("FLASH_OPD_STOP_SEQUENCES", "[]"))
            if stops:
                params["stop"] = stops
                params["include_stop_str_in_output"] = True
            eos_ids = json.loads(os.environ.get("FLASH_OPD_EOS_TOKEN_IDS", "[]"))
            if eos_ids:
                params["stop_token_ids"] = eos_ids
            structured = json.loads(os.environ.get("FLASH_OPD_STRUCTURED_OUTPUTS", "null"))
            if structured:
                from vllm.sampling_params import StructuredOutputsParams

                params["structured_outputs"] = StructuredOutputsParams(**structured)
                params["logprobs"] = True
            output = await super().run(params, **kwargs)
            output.reward_score = 0.0
            return output

    FlashSingleTurnAgentLoop.__module__ = __name__
    globals()["FlashSingleTurnAgentLoop"] = FlashSingleTurnAgentLoop

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
            raise RuntimeError("flash OPD step produced no aligned teacher signal")
        return KVBatchMeta(
            keys=keys,
            tags=tags,
            partition_id=batch.partition_id,
            fields=batch.fields,
            extra_info=batch.extra_info,
        )

    class FlashPPOTrainer(PPOTrainer):
        def init_workers(self):
            self.use_teacher_policy = False
            super().init_workers()
            self.use_teacher_policy = True
            self.distillation_config = omega_conf_to_dataclass(self.config.distillation)

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
            output = self.actor_rollout_wg.update_actor(batch)
            output = rename_dict(output["metrics"], "actor/")
            output["perf/mfu/actor"] = output.pop("actor/mfu")
            metrics.update(reduce_metrics(output))
            return batch

    FlashPPOTrainer.__module__ = __name__
    globals()["FlashPPOTrainer"] = FlashPPOTrainer

    @ray.remote
    class FlashTaskRunner:
        def __init__(self):
            self.role_worker_mapping = {}
            self.mapping = {}

        def add_actor_rollout_worker(self, config):
            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
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
            tq.init(config.transfer_queue)
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

    globals()["FlashTaskRunner"] = FlashTaskRunner

    import verl.trainer.main_ppo_sync as main_ppo_sync

    main_ppo_sync.TaskRunner = FlashTaskRunner

    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    original_build_optimizer = FSDPEngine._build_optimizer

    @functools.wraps(original_build_optimizer)
    def build_optimizer_with_mutation_notice(self, module):
        optimizer = original_build_optimizer(self, module)
        original_step = optimizer.step
        notified = False

        @functools.wraps(original_step)
        def step_with_notice(*args, **kwargs):
            nonlocal notified
            if not notified:
                _post_json(
                    os.environ["FLASH_OPD_BRIDGE_URL"],
                    os.environ["FLASH_OPD_BRIDGE_TOKEN"],
                    "/mutation",
                    {},
                )
                notified = True
            return original_step(*args, **kwargs)

        optimizer.step = step_with_notice
        return optimizer

    FSDPEngine._build_optimizer = build_optimizer_with_mutation_notice


def main() -> None:
    """Run verl's synchronous trainer after the external module has patched its task runner."""
    from verl.trainer.main_ppo_sync import main as verl_main

    verl_main()


if importlib.util.find_spec("verl") is not None:
    _install_verl_extensions()

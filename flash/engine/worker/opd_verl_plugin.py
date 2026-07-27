"""Standalone verl 0.8.0 extensions for flash OPD.

This file is copied into the isolated verl child workdir. It registers the exact flash groupwise
reverse-KL loss, routes teacher scoring through the parent localhost bridge, supplies deterministic
single-turn sampling, and removes the otherwise mandatory local teacher GPU pool.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

try:
    from flash_opd_verl_multiturn import build_flash_multi_turn_agent_loop
    from flash_opd_verl_structured import StructuredOutputReplay, canonical_structured_spec
except ImportError:
    from flash.engine.worker.opd_verl_multiturn import build_flash_multi_turn_agent_loop
    from flash.engine.worker.opd_verl_structured import (
        StructuredOutputReplay,
        canonical_structured_spec,
    )

_PERMANENT_TEACHER_EXIT = 86
_TRANSIENT_TEACHER_EXIT = 87
_STRUCTURED_RUNTIME_VERSIONS = {
    "verl": "0.8.0",
    "vllm": "0.11.0",
    "xgrammar": "0.1.25",
}


def _require_structured_runtime_versions() -> None:
    for package, expected in _STRUCTURED_RUNTIME_VERSIONS.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(
                f"structured OPD requires {package} {expected} exactly; found {actual}"
            )


def deterministic_rollout_seed(
    flash_seed: int,
    global_step: int,
    example_index: int,
    rollout_ordinal: int,
    assistant_turn_ordinal: int = 0,
    no_signal_attempt_ordinal: int = 0,
) -> int:
    """Derive one stable vLLM request seed from the complete rollout identity."""
    # single-turn rollouts (turn ordinal 0) keep the pre-multiturn payload byte-identical, so a
    # resumed single-turn run continues its original deterministic trajectory; the turn suffix is
    # appended only for multi-turn identities.
    payload = f"{int(flash_seed)}:{int(global_step)}:{int(example_index)}:{int(rollout_ordinal)}"
    turn_ordinal = int(assistant_turn_ordinal)
    if turn_ordinal < 0:
        raise ValueError("flash OPD assistant turn ordinal must be nonnegative")
    if turn_ordinal:
        payload += f":{turn_ordinal}"
    attempt_ordinal = int(no_signal_attempt_ordinal)
    if attempt_ordinal < 0:
        raise ValueError("flash OPD no-signal attempt ordinal must be nonnegative")
    if attempt_ordinal:
        payload += f":retry:{attempt_ordinal}"
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


class FlashTeacherBridgeError(RuntimeError):
    def __init__(self, message: str, *, classification: str) -> None:
        super().__init__(message)
        self.classification = classification


class _AllNoSignalBatch(RuntimeError):
    def __init__(self, batch) -> None:
        super().__init__("flash OPD rollout attempt produced no aligned teacher signal")
        self.batch = batch


def _run_with_no_signal_replacements(
    run_attempt,
    cleanup_attempt,
    prepare_replacement,
    record_resample,
    record_abandoned,
    *,
    max_attempts: int = 3,
):
    """retry an all-no-signal rollout with a fresh bounded dispatch."""
    if max_attempts <= 0:
        raise ValueError("flash OPD no-signal attempt limit must be positive")
    for attempt_ordinal in range(max_attempts):
        try:
            return run_attempt(attempt_ordinal)
        except _AllNoSignalBatch as error:
            cleanup_attempt(error.batch)
            if attempt_ordinal == max_attempts - 1:
                record_abandoned()
                raise RuntimeError(
                    f"flash OPD produced no aligned teacher signal after {max_attempts} rollout attempts"
                ) from error
            record_resample()
            prepare_replacement()
    raise AssertionError("unreachable no-signal replacement state")


def _multi_modal_image_count(multi_modal_data) -> int:
    if not multi_modal_data:
        return 0
    if not isinstance(multi_modal_data, dict):
        raise TypeError("verl multimodal data must be a mapping")
    # verl's rollout carries the payload under "image" (singular) for single-image rows and
    # "images" for lists; count whichever is present.
    images = multi_modal_data.get("images")
    if images is None:
        images = multi_modal_data.get("image")
    if images is None:
        return 0
    if not isinstance(images, (list, tuple)):
        return 1
    try:
        return len(images)
    except TypeError as error:
        raise TypeError("verl multimodal images must be a sized collection") from error


def _bridge_score_payload(
    index: int,
    prompt_ids: list[int],
    response_ids: list[int],
    multi_modal_data=None,
    forced: list[bool] | None = None,
) -> dict[str, Any]:
    prompt_ids = [int(token_id) for token_id in prompt_ids]
    response_ids = [int(token_id) for token_id in response_ids]
    payload = {
        "index": int(index),
        "prompt_length": len(prompt_ids),
        "sequence_ids": prompt_ids + response_ids,
        "image_count": _multi_modal_image_count(multi_modal_data),
    }
    if forced is not None:
        payload["forced"] = list(forced)
    return payload


def _raw_prompt_has_image_block(raw_prompt) -> bool:
    for message in raw_prompt or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "image" for block in content
        ):
            return True
    return False


def _resolve_image_token_id(processor, tokenizer) -> int:
    def valid(value) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            token_id = int(value)
        except (TypeError, ValueError):
            return None
        return token_id if token_id >= 0 else None

    token_id = valid(getattr(processor, "image_token_id", None))
    if token_id is not None:
        return token_id
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        image_token = getattr(processor, "image_token", None)
        candidates = [image_token] if isinstance(image_token, str) else []
        candidates.append("<|image_pad|>")
        for token in candidates:
            try:
                token_id = valid(convert(token))
            except Exception:
                token_id = None
            if token_id is not None:
                return token_id
    raise ValueError("could not resolve a valid image token id from the processor or tokenizer")


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
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
            details = payload["error"]
            classification = str(details["classification"])
            message = str(details["message"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as decode_error:
            raise FlashTeacherBridgeError(
                f"flash OPD bridge returned unclassified HTTP {error.code}",
                classification="permanent",
            ) from decode_error
        if classification not in {"permanent", "transient"}:
            raise FlashTeacherBridgeError(
                f"flash OPD bridge returned unknown teacher failure classification {classification!r}",
                classification="permanent",
            ) from error
        raise FlashTeacherBridgeError(message, classification=classification) from error
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise FlashTeacherBridgeError(
            f"flash OPD bridge transport failed: {type(error).__name__}",
            classification="transient",
        ) from error
    try:
        return json.loads(body.decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise FlashTeacherBridgeError(
            "flash OPD bridge returned malformed success JSON",
            classification="permanent",
        ) from error


def _write_mutation_failure_fallback(classification: str, message: str) -> None:
    base_path = os.environ.get("FLASH_OPD_MUTATION_FAILURE_PATH", "")
    if not base_path:
        return
    process_id = os.getpid()
    record_path = f"{base_path}.{process_id}.{classification}.json"
    directory = os.path.dirname(base_path) or "."
    prefix = f".{os.path.basename(base_path)}.{process_id}.{classification}."
    payload = json.dumps(
        {"classification": classification, "message": str(message)[:4096]},
        separators=(",", ":"),
    ).encode("utf-8")
    temporary_path = ""
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            dir=directory,
            prefix=prefix,
            suffix=".tmp",
        )
        try:
            remaining = payload
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("mutation failure fallback write did not progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary_path, record_path)
        temporary_path = ""
    except OSError:
        if temporary_path:
            with contextlib.suppress(OSError):
                os.unlink(temporary_path)


def _post_mutation_notice(url: str, token: str) -> None:
    try:
        _post_json(url, token, "/mutation", {})
    except FlashTeacherBridgeError as error:
        _write_mutation_failure_fallback(error.classification, str(error))
        exit_code = (
            _PERMANENT_TEACHER_EXIT
            if error.classification == "permanent"
            else _TRANSIENT_TEACHER_EXIT
        )
        os._exit(exit_code)
    except Exception as error:
        _write_mutation_failure_fallback(
            "permanent",
            f"unexpected mutation bridge failure: {type(error).__name__}",
        )
        os._exit(_PERMANENT_TEACHER_EXIT)


def _install_verl_extensions() -> None:
    import numpy as np
    import ray
    import torch
    from omegaconf import OmegaConf

    try:
        import transfer_queue as tq
        from transfer_queue import KVBatchMeta
    except ImportError:
        from verl.utils.transferqueue_utils import KVBatchMeta, tq

    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopOutput,
        AgentLoopWorker,
        register,
    )
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
            prompt_ids: list[int],
            response_ids: list[int],
            multi_modal_data=None,
            mm_processor_kwargs=None,
            routing_key=None,
            forced: list[bool] | None = None,
        ):
            if routing_key is None:
                raise RuntimeError("flash OPD bridge requires the indexed dataset row")
            request_payload = _bridge_score_payload(
                routing_key,
                prompt_ids,
                response_ids,
                multi_modal_data,
                forced,
            )
            try:
                payload = await asyncio.to_thread(
                    _post_json,
                    self.bridge_url,
                    self.bridge_token,
                    "/score",
                    request_payload,
                )
            except FlashTeacherBridgeError as error:
                exit_code = (
                    _PERMANENT_TEACHER_EXIT
                    if error.classification == "permanent"
                    else _TRANSIENT_TEACHER_EXIT
                )
                os._exit(exit_code)
            except Exception:
                os._exit(_PERMANENT_TEACHER_EXIT)
            teacher_ids = torch.tensor(payload["teacher_ids"], dtype=torch.int32).unsqueeze(-1)
            teacher_logprobs = torch.tensor(
                payload["teacher_logprobs"], dtype=torch.float32
            ).unsqueeze(-1)
            if teacher_ids.shape != teacher_logprobs.shape:
                raise RuntimeError("flash OPD bridge returned inconsistent teacher tensors")
            if teacher_ids.shape[0] != len(request_payload["sequence_ids"]):
                raise RuntimeError("flash OPD bridge returned the wrong sequence length")
            return teacher_ids, teacher_logprobs

    async def compute_flash_teacher_logprobs(
        self,
        output,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs=None,
    ) -> None:
        if self.distillation_enabled and not validate:
            existing_ids = output.extra_fields.get("teacher_ids")
            existing_logprobs = output.extra_fields.get("teacher_logprobs")
            if existing_ids is not None or existing_logprobs is not None:
                if existing_ids is None or existing_logprobs is None:
                    raise RuntimeError("flash OPD teacher metadata is incomplete")
                return
            routing_key = None
            if sample_kwargs is not None:
                routing_value = sample_kwargs.get(self.teacher_key)
                if routing_value is not None:
                    routing_key = (
                        routing_value.item() if hasattr(routing_value, "item") else routing_value
                    )
            forced = None
            structured_spec = os.environ.get("FLASH_OPD_STRUCTURED_OUTPUTS", "")
            if structured_spec:
                replay = getattr(self, "_flash_structured_output_replay", None)
                if replay is None:
                    replay = StructuredOutputReplay(
                        self.tokenizer,
                        int(os.environ["FLASH_OPD_MODEL_VOCAB_SIZE"]),
                    )
                    self._flash_structured_output_replay = replay
                forced = replay.forced_mask(
                    prompt_ids,
                    response_ids,
                    canonical_structured_spec(json.loads(structured_spec)),
                    thinking=os.environ.get("FLASH_OPD_THINKING") == "1",
                )
            teacher_ids, teacher_logprobs = (
                await self.teacher_server_manager.compute_teacher_logprobs_single(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    multi_modal_data=output.multi_modal_data,
                    mm_processor_kwargs=output.mm_processor_kwargs,
                    routing_key=routing_key,
                    forced=forced,
                )
            )
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs

    import verl.experimental.teacher_loop.teacher_manager as teacher_manager_module

    AgentLoopWorker._compute_teacher_logprobs = compute_flash_teacher_logprobs
    teacher_manager_module.AsyncTeacherLLMServerManager = FlashBridgeTeacherManager

    @register("flash_single_turn")
    class FlashSingleTurnAgentLoop(SingleTurnAgentLoop):
        async def run(self, sampling_params: dict[str, Any], **kwargs):
            params = dict(sampling_params)
            if _raw_prompt_has_image_block(kwargs.get("raw_prompt")):
                image_token_id = _resolve_image_token_id(self.processor, self.tokenizer)
                logit_bias = dict(params.get("logit_bias") or {})
                logit_bias[image_token_id] = -100.0
                params["logit_bias"] = logit_bias
            flash_seed = int(os.environ["FLASH_OPD_SEED"])
            global_step = int(kwargs["global_steps"])
            example_index = int(kwargs["index"])
            rollout_ordinal = int(kwargs.get("session_id", 0))
            no_signal_attempt_ordinal = int(kwargs.get("flash_no_signal_attempt", 0))
            params["seed"] = deterministic_rollout_seed(
                flash_seed,
                global_step,
                example_index,
                rollout_ordinal,
                no_signal_attempt_ordinal=no_signal_attempt_ordinal,
            )
            stops = json.loads(os.environ.get("FLASH_OPD_STOP_SEQUENCES", "[]"))
            if stops:
                params["stop"] = stops
                params["include_stop_str_in_output"] = True
            eos_ids = json.loads(os.environ.get("FLASH_OPD_EOS_TOKEN_IDS", "[]"))
            if eos_ids:
                params["stop_token_ids"] = eos_ids
            structured_spec = os.environ.get("FLASH_OPD_STRUCTURED_OUTPUTS", "")
            if structured_spec:
                _require_structured_runtime_versions()
                from vllm.sampling_params import StructuredOutputsParams

                params["structured_outputs"] = StructuredOutputsParams(
                    **json.loads(structured_spec)
                )
            output = await super().run(params, **kwargs)
            output.reward_score = 0.0
            return output

    FlashSingleTurnAgentLoop.__module__ = __name__
    globals()["FlashSingleTurnAgentLoop"] = FlashSingleTurnAgentLoop

    FlashMultiTurnAgentLoop = build_flash_multi_turn_agent_loop(
        register=register,
        agent_loop_base=AgentLoopBase,
        agent_loop_output=AgentLoopOutput,
        post_json=_post_json,
        deterministic_seed=deterministic_rollout_seed,
        permanent_teacher_exit=_PERMANENT_TEACHER_EXIT,
        transient_teacher_exit=_TRANSIENT_TEACHER_EXIT,
    )
    globals()["FlashMultiTurnAgentLoop"] = FlashMultiTurnAgentLoop

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

    class FlashPPOTrainer(PPOTrainer):
        def init_workers(self):
            self.use_teacher_policy = False
            super().init_workers()
            self.use_teacher_policy = True
            self.distillation_config = omega_conf_to_dataclass(self.config.distillation)

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

            def record(path: str) -> None:
                _post_json(
                    os.environ["FLASH_OPD_BRIDGE_URL"],
                    os.environ["FLASH_OPD_BRIDGE_TOKEN"],
                    path,
                    {},
                )

            return _run_with_no_signal_replacements(
                run_attempt,
                cleanup_attempt,
                prepare_replacement,
                lambda: record("/no-signal/resample"),
                lambda: record("/no-signal/abandoned"),
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

        @functools.wraps(original_step)
        def step_with_notice(*args, **kwargs):
            _post_mutation_notice(
                os.environ["FLASH_OPD_BRIDGE_URL"],
                os.environ["FLASH_OPD_BRIDGE_TOKEN"],
            )
            return original_step(*args, **kwargs)

        optimizer.step = step_with_notice
        return optimizer

    FSDPEngine._build_optimizer = build_optimizer_with_mutation_notice


def main() -> None:
    """Run verl's synchronous trainer after the external module has patched its task runner."""
    from verl.trainer.main_ppo_sync import main as verl_main

    verl_main()


if os.environ.get("FLASH_OPD_STRUCTURED_OUTPUTS"):
    _require_structured_runtime_versions()

if importlib.util.find_spec("verl") is not None:
    _install_verl_extensions()

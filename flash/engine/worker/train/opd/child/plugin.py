"""Standalone verl 0.8.0 extensions for flash OPD.

This file is copied into the isolated verl child workdir. It registers the exact flash groupwise
reverse-KL loss, routes teacher scoring through the parent localhost bridge, supplies deterministic
single-turn sampling, and removes the otherwise mandatory local teacher GPU pool.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib.metadata
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from packaging.version import Version

PLUGIN_LOADED_EXTERNALLY = __name__ == "flash_opd_plugin"
_PLUGIN_CONFIG: dict[str, Any] = {}

if PLUGIN_LOADED_EXTERNALLY:
    from flash_multiturn_glue import multi_modal_image_count
    from flash_opd_multiturn import build_flash_multi_turn_agent_loop
    from flash_opd_structured import StructuredOutputReplay, canonical_structured_spec
    from flash_opd_tensors import (
        full_sequence_signal_sequences as _full_sequence_signal_sequences,
    )
    from flash_opd_tensors import (
        packed_full_sequence as _packed_full_sequence,
    )
    from flash_opd_tensors import (
        signal_sequences as _signal_sequences,
    )
else:
    from flash.engine.worker.train.core.child.glue import multi_modal_image_count
    from flash.engine.worker.train.opd.child.multiturn import build_flash_multi_turn_agent_loop
    from flash.engine.worker.train.opd.child.structured import (
        StructuredOutputReplay,
        canonical_structured_spec,
    )
    from flash.engine.worker.train.opd.child.tensors import (
        full_sequence_signal_sequences as _full_sequence_signal_sequences,
    )
    from flash.engine.worker.train.opd.child.tensors import (
        packed_full_sequence as _packed_full_sequence,
    )
    from flash.engine.worker.train.opd.child.tensors import (
        signal_sequences as _signal_sequences,
    )

_PERMANENT_TEACHER_EXIT = 86
_TRANSIENT_TEACHER_EXIT = 87
_FAILURE_FALLBACK_MAX_CHARS = 8192
_STRUCTURED_RUNTIME_EXACT_VERSIONS = {
    "verl": "0.8.0",
    "xgrammar": "0.1.25",
}
_STRUCTURED_RUNTIME_MINIMUM_VERSIONS = {"vllm": "0.11.0"}


def _require_structured_runtime_versions() -> None:
    for package, expected in _STRUCTURED_RUNTIME_EXACT_VERSIONS.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(
                f"structured OPD requires {package} {expected} exactly; found {actual}"
            )
    for package, minimum in _STRUCTURED_RUNTIME_MINIMUM_VERSIONS.items():
        actual = importlib.metadata.version(package)
        if Version(actual) < Version(minimum):
            raise RuntimeError(
                f"structured OPD requires {package} {minimum} or newer; found {actual}"
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
    response_mask = response_mask.bool()
    selected = response_mask & group_ids.ge(0)
    order = torch.argsort(group_ids, dim=1, stable=True)
    sorted_selected = selected.gather(1, order)
    sorted_groups = group_ids.gather(1, order)
    sorted_student = student_logprobs.gather(1, order)
    sorted_teacher = teacher_logsums.gather(1, order)
    sorted_rows = (
        torch.arange(group_ids.shape[0], device=group_ids.device).unsqueeze(1).expand_as(group_ids)
    )

    flat_rows = sorted_rows.masked_select(sorted_selected)
    flat_groups = sorted_groups.masked_select(sorted_selected)
    flat_student = sorted_student.masked_select(sorted_selected)
    flat_teacher = sorted_teacher.masked_select(sorted_selected)
    previous_rows = torch.cat((flat_rows.new_full((1,), -1), flat_rows[:-1]))
    previous_groups = torch.cat((flat_groups.new_zeros(1), flat_groups[:-1]))
    group_starts = flat_rows.ne(previous_rows) | flat_groups.ne(previous_groups)

    # the trailing dummy group keeps the jagged reduction valid for an all-empty batch.
    #
    # `vmap(torch.sum)` over a jagged tensor, NOT scatter_add_ or a padded row-sum: those
    # reassociate the float additions and so change the per-group value. measured on this machine,
    # scatter_add_ differs BITWISE from `torch.sum` in 1056 of 3000 random fp32 groups. the
    # differences are ~1e-7 and pass any allclose check, but they move the training loss silently.
    # test_groupwise_reverse_kl_keeps_the_exact_per_group_reduction pins this by reading the source.
    grouped_starts = torch.cat((group_starts, group_starts.new_ones(1)))
    grouped_indices = grouped_starts.cumsum(0) - 1
    group_indices = grouped_indices[:-1]
    group_lengths = torch.bincount(grouped_indices)
    grouped_student = torch.cat((flat_student.detach(), flat_student.new_zeros(1)))
    offsets = torch.cat((group_lengths.new_zeros(1), group_lengths.cumsum(0)))
    nested_student = torch.nested.nested_tensor_from_jagged(grouped_student, offsets)
    student_logsums = torch.vmap(torch.sum)(nested_student).values()
    teacher_logsums_first = torch.cat(
        (flat_teacher.masked_select(group_starts), flat_teacher.new_zeros(1))
    )
    coefficients = (
        float(kl_penalty_coef)
        * (student_logsums - teacher_logsums_first)
        / group_lengths.to(dtype=student_logprobs.dtype)
    )
    sorted_values = coefficients.index_select(0, group_indices) * flat_student

    # verl aggregates over RESPONSE positions while this loss is defined over the SELECTED subset,
    # so rescale each row by response/selected to keep the two denominators equal. clamp guards a
    # row with nothing selected, which is reachable and must contribute zeros rather than nan.
    response_counts = response_mask.sum(dim=1)
    selected_counts = selected.sum(dim=1).clamp_min(1)
    ratio_dtype = (
        torch.float32
        if student_logprobs.dtype in (torch.float16, torch.bfloat16)
        else student_logprobs.dtype
    )
    ratios = response_counts.to(ratio_dtype) / selected_counts.to(ratio_dtype)
    scaled_values = (sorted_values.to(ratio_dtype) * ratios.index_select(0, flat_rows)).to(
        student_logprobs.dtype
    )
    sorted_output = torch.zeros_like(student_logprobs).masked_scatter(
        sorted_selected, scaled_values
    )
    return torch.zeros_like(student_logprobs).scatter(1, order, sorted_output)


def _set_current_global_batch_info(config, data) -> None:
    """Mirror verl 0.8.0 ppo_loss metadata population for the current microbatch."""
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor


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
    max_attempts: int | None = None,
):
    """retry an all-no-signal rollout with a fresh bounded dispatch."""
    if max_attempts is None:
        try:
            max_attempts = int(_PLUGIN_CONFIG["no_signal_attempts"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("flash OPD no-signal attempt limit is not configured") from error
    if max_attempts <= 0:
        raise ValueError("flash OPD no-signal attempt limit must be positive")
    for attempt_ordinal in range(max_attempts):
        try:
            return run_attempt(attempt_ordinal)
        except _AllNoSignalBatch as error:
            cleanup_attempt(error.batch)
            if attempt_ordinal == max_attempts - 1:
                # name WHY every rollout was dropped. the three pre-teacher gates fail for very
                # different reasons (cap reached without EOS, empty response, undecodable text) and
                # the bare message sent a paid debugging cycle after the wrong one.
                #
                # the bridge returns the delta for THIS step, not its lifetime totals, and omits
                # reasons that did not fire. an empty dict therefore means the rollouts were lost
                # after scoring, and the message correctly says nothing rather than guessing.
                skips = record_abandoned() or {}
                detail = (
                    "; dropped before teacher scoring: "
                    + ", ".join(f"{reason}={count}" for reason, count in sorted(skips.items()))
                    if skips
                    else ""
                )
                raise RuntimeError(
                    f"flash OPD produced no aligned teacher signal after {max_attempts} rollout "
                    f"attempts{detail}"
                ) from error
            record_resample()
            prepare_replacement()
    raise AssertionError("unreachable no-signal replacement state")


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
        "image_count": multi_modal_image_count(multi_modal_data),
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


_TQ_INIT_TIMEOUT_S = 600.0


def _describe_ray_resources() -> str:
    """summarise ray's cluster and free resources, or say why they could not be read.

    ray drops a resource key from available_resources() once it is fully allocated rather than reporting
    it as zero -- verified against ray 2.56.1, where consuming every cpu removes 'CPU' from the mapping
    entirely. that is exactly the exhaustion this probe exists to name, so a missing key reads as 0.0
    instead of None. cluster_resources() omits a resource the node does not have at all, which 0.0 also
    describes correctly.

    this is only ever called on the timeout path, so it must not raise: a probe failure here would
    replace the diagnosis it exists to produce.
    """
    import ray

    try:
        total = ray.cluster_resources()
        free = ray.available_resources()
    except Exception as error:  # pragma: no cover - defensive, ray is up by this point
        return f"ray resources unreadable: {type(error).__name__}: {error}"
    return (
        f"cluster CPU={total.get('CPU', 0.0)} GPU={total.get('GPU', 0.0)}, "
        f"free CPU={free.get('CPU', 0.0)} GPU={free.get('GPU', 0.0)}"
    )


def _describe_stalled_thread(ident: int | None, depth: int = 4) -> str:
    """render the innermost frames of a thread that is still parked, or say why they are unavailable.

    which of tq.init's three waits is stuck is not decidable from ray's resource counts alone: a
    satisfied placement group rules out the first, but the controller ray.get and the get_config spin
    look identical from outside. the wedged thread is still alive at this point, so its own frames name
    the wait directly. like the resource probe, this runs only on the failure path and must not raise.
    """
    import sys
    import traceback

    try:
        frame = sys._current_frames().get(ident) if ident is not None else None
        if frame is None:
            return "stack unavailable"
        frames = traceback.extract_stack(frame)[-depth:]
        return " <- ".join(
            f"{f.name} ({f.filename.rsplit('/', 1)[-1]}:{f.lineno})" for f in reversed(frames)
        )
    except Exception as error:  # pragma: no cover - defensive, the thread is alive by construction
        return f"stack unreadable: {type(error).__name__}: {error}"


def _init_transfer_queue(
    init: Callable[[Any], Any], conf: Any, timeout_s: float = _TQ_INIT_TIMEOUT_S
) -> None:
    """run verl's transfer-queue init under a deadline, reporting the resource state on timeout.

    verl force-enables TransferQueue on the opd entry point (main_ppo_sync.main sets
    transfer_queue.enable = True with no opt-out), so this runs before a single gpu is touched. tq.init
    has three separate unbounded waits, none of which can be observed from outside:

      - get_placement_group blocks in ray.get(pg.ready()) until every 1-cpu storage bundle is placed,
      - process_zmq_server_info blocks in ray.get on the controller, itself a 1-cpu actor scheduled
        outside any placement group,
      - _init_from_existing spins `while conf is None` on get_config against a controller that may
        never publish one.

    all three are silent: tq's get_logger defaults to WARNING and TQ_LOGGING_LEVEL is unset here, so
    every progress line inside tq.init is suppressed. a wedge therefore presents as a run that reaches
    the ray banner and then stops, burning the full setup grace period at 0% gpu before the plane
    calls it a stall -- with no indication that transfer_queue was even involved.

    the thread is a daemon and is deliberately not joined after the timeout: it is parked inside an
    unbounded ray.get that no signal will interrupt, and the exception below fails the run anyway. that
    is also what makes its stack readable here, which is the only thing that tells the three waits apart.
    """
    import threading

    failure: list[BaseException] = []

    def _run() -> None:
        try:
            init(conf)
        except BaseException as error:  # surfaced below, on the caller's thread
            failure.append(error)

    thread = threading.Thread(target=_run, name="flash-tq-init", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise RuntimeError(
            f"verl transfer_queue init did not finish within {timeout_s:.0f}s; "
            f"stalled at {_describe_stalled_thread(thread.ident)}; {_describe_ray_resources()}. "
            "tq.init waits without a timeout in three places -- an unplaceable storage placement group, "
            "a controller rpc that never returns, and a spin on a controller that never publishes its "
            "config -- so read the stalled frame: only the first is a capacity problem"
        )
    if failure:
        raise failure[0]


def _register_flash_distillation_loss() -> None:
    from verl.trainer.distillation.losses import (
        DistillationLossSettings,
        register_distillation_loss,
    )
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding

    @register_distillation_loss(
        DistillationLossSettings(names=["flash_groupwise_reverse_kl"], use_estimator=True)
    )
    def flash_groupwise_reverse_kl(config, distillation_config, model_output, data):
        _set_current_global_batch_info(config, data)
        student_logprobs = no_padding_2_padding(model_output["log_probs"], data)
        teacher_logsums = no_padding_2_padding(
            _packed_full_sequence(data["teacher_logprobs"], data), data
        ).squeeze(-1)
        group_ids = (
            no_padding_2_padding(_packed_full_sequence(data["teacher_ids"], data), data)
            .squeeze(-1)
            .long()
        )
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


def _build_flash_teacher_extensions():
    import torch
    from verl.utils.config import omega_conf_to_dataclass

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
                _exit_for_score_failure(error)
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
            (
                teacher_ids,
                teacher_logprobs,
            ) = await self.teacher_server_manager.compute_teacher_logprobs_single(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                routing_key=routing_key,
                forced=forced,
            )
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs

    return FlashBridgeTeacherManager, compute_flash_teacher_logprobs


def _build_flash_ppo_trainer(PPOTrainer, tq, KVBatchMeta):
    import numpy as np
    import torch
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.metric import reduce_metrics
    from verl.utils.py_functional import rename_dict

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

        def _balance_batch(
            self, batch, metrics, logging_prefix="global_seqlen", keep_minibatch=False
        ):
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

    return FlashPPOTrainer


def _build_flash_task_runner(FlashPPOTrainer, tq):
    import ray
    from omegaconf import OmegaConf
    from verl.single_controller.ray import ResourcePoolManager
    from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
    from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

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
                install_bounded_replay_buffer_sample(trainer)
                trainer.fit()
            finally:
                if trainer:
                    trainer.replay_buffer.close()
                tq.close()

    return FlashTaskRunner


def _patch_fsdp_optimizer() -> None:
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    original_build_optimizer = FSDPEngine._build_optimizer

    @functools.wraps(original_build_optimizer)
    def build_optimizer_with_mutation_notice(self, module):
        optimizer = original_build_optimizer(self, module)
        return _wrap_optimizer_with_mutation_notice(
            optimizer,
            os.environ["FLASH_OPD_BRIDGE_URL"],
            os.environ["FLASH_OPD_BRIDGE_TOKEN"],
        )

    FSDPEngine._build_optimizer = build_optimizer_with_mutation_notice


def _install_verl_extensions() -> None:
    try:
        import transfer_queue as tq
        from transfer_queue import KVBatchMeta
    except ImportError:
        from verl.utils.transferqueue_utils import KVBatchMeta, tq

    def mark_prompt_failure(uid, partition_id, global_steps, status) -> None:
        tq.kv_put(
            key=uid,
            partition_id=partition_id,
            tag={"global_steps": global_steps, "status": status},
        )

    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopOutput,
        AgentLoopWorker,
        register,
    )
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config import DistillationConfig, DistillationLossConfig

    _register_flash_distillation_loss()

    @dataclass
    class FlashRemoteDistillationConfig(DistillationConfig):
        # verl's BaseConfig.__setattr__ rejects any assignment to a field that is not listed here
        # (base_config.py:33-38), so __post_init__ below can only normalize distillation_loss if the
        # field is declared mutable. verl's own DistillationConfig declares teacher_models for exactly
        # the same reason; it never reassigns distillation_loss, so it did not need to.
        _mutable_fields = DistillationConfig._mutable_fields | {"distillation_loss"}

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

    FlashBridgeTeacherManager, compute_flash_teacher_logprobs = _build_flash_teacher_extensions()

    import verl.experimental.teacher_loop.teacher_manager as teacher_manager_module

    AgentLoopWorker._compute_teacher_logprobs = compute_flash_teacher_logprobs
    teacher_manager_module.AsyncTeacherLLMServerManager = FlashBridgeTeacherManager

    # import main_ppo_sync only after the patch above. applying `@ray.remote` to a class copies the
    # methods it inherits onto the actor class, so `AgentLoopWorkerTQ(AgentLoopWorker)` freezes
    # whatever `_compute_teacher_logprobs` resolves to at decoration time. importing this module
    # earlier snapshots verl's original, which then calls the flash teacher manager with
    # `sequence_ids=` instead of prompt_ids/response_ids and kills the first rollout.
    from verl.trainer.main_ppo_sync import PPOTrainer

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

    # verl's `register` freezes `f"{__module__}.{__qualname__}"` into the agent-loop registry at
    # decoration time, and hydra later resolves that string with a plain dotted-path lookup. this
    # class is defined inside a function, so its qualname carries `<locals>` and no lookup can
    # reach it. rewrite both dunders to the importable module-level name first, publish the class
    # there, and only then register, exactly as the multi-turn loop already does.
    FlashSingleTurnAgentLoop.__module__ = __name__
    FlashSingleTurnAgentLoop.__qualname__ = "FlashSingleTurnAgentLoop"
    globals()["FlashSingleTurnAgentLoop"] = FlashSingleTurnAgentLoop
    register("flash_single_turn")(FlashSingleTurnAgentLoop)

    FlashMultiTurnAgentLoop = build_flash_multi_turn_agent_loop(
        register=register,
        agent_loop_base=AgentLoopBase,
        agent_loop_output=AgentLoopOutput,
        post_json=_post_json,
        score_failure_handler=_defer_score_failure,
        fatal_rollout_exit_code=_fatal_rollout_exit_code,
        mark_prompt_failure=mark_prompt_failure,
        deterministic_seed=deterministic_rollout_seed,
        permanent_teacher_exit=_PERMANENT_TEACHER_EXIT,
        transient_teacher_exit=_TRANSIENT_TEACHER_EXIT,
    )
    globals()["FlashMultiTurnAgentLoop"] = FlashMultiTurnAgentLoop

    FlashPPOTrainer = _build_flash_ppo_trainer(PPOTrainer, tq, KVBatchMeta)
    FlashPPOTrainer.__module__ = __name__
    globals()["FlashPPOTrainer"] = FlashPPOTrainer

    FlashTaskRunner = _build_flash_task_runner(FlashPPOTrainer, tq)
    globals()["FlashTaskRunner"] = FlashTaskRunner

    import verl.trainer.main_ppo_sync as main_ppo_sync

    main_ppo_sync.TaskRunner = FlashTaskRunner
    _patch_fsdp_optimizer()


def main() -> None:
    """Run verl's synchronous trainer after the external module has patched its task runner."""
    from verl.trainer.main_ppo_sync import main as verl_main

    verl_main()


# re-exported so the child plugin keeps its public surface: the opd tests import these from
# `child.plugin` and patch `plugin._post_json` there. the flat import is the one that runs in
# the verl child workdir, where `flash` is not on the path and the file is `flash_opd_bridge`;
# the package import is the one that runs in-repo. same shape as the multiturn import above.
#
# this block MUST stay above `_install_verl_extensions()`: that call runs at import time and passes
# `_post_json` into the multi-turn agent loop, so binding these names afterwards leaves the install
# path reading an unbound global. the child then dies with `NameError: _post_json`, and since the
# loss decorator has already registered by that point the retry reports "flash_groupwise_reverse_kl
# is already registered" instead, which points at the wrong problem entirely.
if PLUGIN_LOADED_EXTERNALLY:
    from flash_opd_bridge import (
        FlashTeacherBridgeError,
        _coordinate_first_mutation_notice,
        _defer_score_failure,
        _exit_for_score_failure,
        _fallback_classification,
        _fatal_rollout_exit_code,
        _mutation_distributed,
        _post_json,
        _post_no_signal_abandoned,
        _post_no_signal_resample,
        _post_teacher_cycle_committed,
        _publish_mutation_notice,
        _serialize_failure_fallback,
        _unexpected_mutation_bridge_error,
        _update_actor_after_teacher_cycle_commit,
        _wrap_optimizer_with_mutation_notice,
        _write_abandonment_failure_fallback,
        _write_cycle_commit_failure_fallback,
        _write_failure_fallback,
        _write_mutation_failure_fallback,
        _write_resample_failure_fallback,
        _write_score_delivery_failure_fallback,
    )
else:
    from flash.engine.worker.train.opd.child.bridge import (  # noqa: F401
        FlashTeacherBridgeError,
        _coordinate_first_mutation_notice,
        _defer_score_failure,
        _exit_for_score_failure,
        _fallback_classification,
        _fatal_rollout_exit_code,
        _mutation_distributed,
        _post_json,
        _post_no_signal_abandoned,
        _post_no_signal_resample,
        _post_teacher_cycle_committed,
        _publish_mutation_notice,
        _serialize_failure_fallback,
        _unexpected_mutation_bridge_error,
        _update_actor_after_teacher_cycle_commit,
        _wrap_optimizer_with_mutation_notice,
        _write_abandonment_failure_fallback,
        _write_cycle_commit_failure_fallback,
        _write_failure_fallback,
        _write_mutation_failure_fallback,
        _write_resample_failure_fallback,
        _write_score_delivery_failure_fallback,
    )


if PLUGIN_LOADED_EXTERNALLY:
    from flash_opd_replay_guard import install_bounded_replay_buffer_sample
else:
    from flash.engine.worker.train.opd.child.replay_guard import (
        install_bounded_replay_buffer_sample,
    )


if PLUGIN_LOADED_EXTERNALLY:
    import flash_opd_runtime
    import flash_verl_runtime

    _PLUGIN_CONFIG.update(flash_verl_runtime.load_plugin_config("FLASH_OPD_PLUGIN_CONFIG"))
    flash_opd_runtime.install(_PLUGIN_CONFIG)

if os.environ.get("FLASH_OPD_STRUCTURED_OUTPUTS"):
    _require_structured_runtime_versions()

if PLUGIN_LOADED_EXTERNALLY:
    _install_verl_extensions()

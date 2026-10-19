"""OPD objective, deterministic sampling, and bounded teacher-cycle policy."""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections.abc import Callable
from typing import Any

from packaging.version import Version

from flash.engine.worker.train.core.child.glue import multi_modal_image_count
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
from flash.engine.worker.train.opd.child.tensors import (
    full_sequence_signal_sequences as _full_sequence_signal_sequences,  # noqa: F401
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
        raise ValueError("no-signal attempt limit is required")
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

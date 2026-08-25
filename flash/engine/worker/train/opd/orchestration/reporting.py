"""training metadata reporting helpers for opd."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flash.engine.worker.runtime.state as _worker_state
from flash.engine.worker.verl.parallelism import ULYSSES_SEQUENCE_PARALLEL_SIZE

if TYPE_CHECKING:
    from flash.engine.worker.train.opd.orchestration.state import (
        _ChildResult,
        _OpdRequest,
        _PromptState,
        _RuntimeState,
        _WorkloadState,
    )


def _build_train_note_sections(
    request: _OpdRequest,
    prompt_state: _PromptState,
    workload: _WorkloadState,
    runtime: _RuntimeState,
    result: _ChildResult,
    download_seconds: float,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], dict[str, Any]],
]:
    final_accounting = result.final_accounting
    knobs = request.knobs
    initial = {
        "epochs": knobs.epochs,
        "retained_prompts": len(prompt_state.prompts),
        "dropped_long_prompts": prompt_state.dropped_long,
        "method": "gkd",
        "init_from_adapter": request.spec.train.init_from_adapter or None,
        "teacher_model": knobs.teacher_model,
        "download_seconds": download_seconds,
        "thinking": _worker_state.THINKING,
        "loss_curve": final_accounting["loss_curve"],
        "mean_coverage": (
            float(final_accounting["coverage_sum"]) / int(final_accounting["aligned_sequences"])
            if final_accounting["aligned_sequences"]
            else 0.0
        ),
    }
    accounting = {
        "truncated_rollouts": int(final_accounting["truncated_rollouts"]),
        "forced_tokens": int(final_accounting["forced_tokens"]),
        "dropped_forced_groups": int(final_accounting["dropped_forced_groups"]),
        "teacher_input_tokens": int(final_accounting["teacher_input_tokens"]),
        "teacher_output_tokens": int(final_accounting["teacher_output_tokens"]),
        "aligned_sequences": int(final_accounting["aligned_sequences"]),
        "empty_alignments": int(final_accounting["empty_alignments"]),
        "teacher_ok": int(final_accounting["teacher_ok"]),
    }
    training = {
        "temperature": knobs.temperature,
        "group_size": knobs.group_size,
        "prompts_per_step": workload.prompts_per_step,
        "max_completion_len": knobs.max_completion,
        "multi_turn": request.multi_turn,
        "max_turns": request.max_turns if request.multi_turn else None,
        "episodes": int(final_accounting["episodes_seen"]) if request.multi_turn else None,
        "mean_turns_per_episode": (
            int(final_accounting["mt_turn_records"]) / int(final_accounting["episodes_seen"])
            if request.multi_turn and final_accounting["episodes_seen"]
            else None
        ),
    }
    backend = (
        {
            "rollout_backend": "verl_vllm",
            "verl_version": "0.8.0",
            "verl_backend": "fsdp",
            # report the executed width, not the allocation: the card count here would claim a
            # sequence-parallel run that did not happen. token-balanced batching makes every
            # allocated rank a dp rank, so unlike sft the executed dp width is the card count.
            "ulysses_sequence_parallel_size": ULYSSES_SEQUENCE_PARALLEL_SIZE,
            "data_parallel_size": runtime.gpu_count,
        },
        {
            "peak_gpu_gb": result.peak_gpu_gb,
            "warm_started": bool(workload.warmstart_adapter),
            "resumed": bool(runtime.resume_step),
            "wandb_project": runtime.project_name if "wandb" in runtime.loggers else None,
            "wandb_run_name": runtime.experiment_name if "wandb" in runtime.loggers else None,
        },
    )
    return initial, accounting, training, backend

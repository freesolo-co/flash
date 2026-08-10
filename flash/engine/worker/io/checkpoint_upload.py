"""Trainer callback state and hooks for durable checkpoint publication.

Split out of ``flash.engine.worker.io.hf`` to keep that module under the file-size limit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from flash._internal.diagnostics import sanitize_diagnostic
from flash.engine.worker.perf import RetriableInfraError


@dataclass
class _CheckpointUploadState:
    required_steps: frozenset[int]
    deployable_steps: set[int] = field(default_factory=set)
    uploaded_steps: set[int] = field(default_factory=set)


def _publish_deployable(
    callback_state: _CheckpointUploadState,
    ckpt_dir: str,
    step: int,
    *,
    provenance_ready: bool = False,
    emit_heartbeat: bool = True,
) -> None:
    """publish the trainer checkpoint's adapter without changing its contents."""
    _hf.publish_deployable_checkpoint(
        ckpt_dir,
        step,
        required=step in callback_state.required_steps,
        _provenance_ready=provenance_ready,
        _emit_heartbeat=emit_heartbeat,
    )
    if step in callback_state.required_steps:
        callback_state.deployable_steps.add(step)


def _upload(
    callback_state: _CheckpointUploadState,
    step: int,
    ckpt_dir: str,
    *,
    provenance_ready: bool = False,
    emit_heartbeat: bool = True,
    lock_timeout_s: float | None = None,
) -> bool:
    # publish the small durable deployable before the latest-only resume checkpoint.
    def _prepare() -> None:
        if step not in callback_state.deployable_steps:
            _publish_deployable(
                callback_state,
                ckpt_dir,
                step,
                provenance_ready=provenance_ready,
                emit_heartbeat=emit_heartbeat,
            )

    return _hf.upload_resume_checkpoint(
        step,
        ckpt_dir,
        before_upload=_prepare,
        after_upload=lambda: callback_state.uploaded_steps.add(step),
        skip_upload=lambda: step in callback_state.uploaded_steps,
        emit_heartbeat=emit_heartbeat,
        lock_timeout_s=lock_timeout_s,
    )


def _enqueue_optional(callback_state: _CheckpointUploadState, step: int, ckpt_dir: str) -> None:
    try:
        _hf._write_deployable_provenance(ckpt_dir)
        staged_dir, staged_checkpoint = _hf._stage_optional_directory(
            ckpt_dir, f"checkpoint-{step}"
        )
    except Exception as error:
        # surface the miss explicitly rather than logging a soft warning and continuing as if
        # the periodic save reached hf.
        print(
            f"[ckpt] step {step} snapshot failed; step not published: "
            f"{sanitize_diagnostic(error, limit=500)}"
        )
        return

    def _publish_coalesced_deployable(replaced) -> None:
        # a newer optional save coalesced this resume checkpoint away; still publish this step's
        # small durable deployable through the non-coalescing fifo path (which owns the staged
        # tree cleanup) so per-step deployables are never dropped.
        _hf._OPTIONAL_DEPLOYABLE_UPLOADER.enqueue(
            f"coalesced deployable step {step}",
            replaced.staged_dir,
            lambda: _hf.publish_deployable_checkpoint(
                staged_checkpoint, step, _provenance_ready=True, _emit_heartbeat=False
            ),
        )

    _hf._OPTIONAL_CHECKPOINT_UPLOADER.enqueue(
        f"checkpoint step {step}",
        staged_dir,
        lambda: _upload(
            callback_state,
            step,
            staged_checkpoint,
            provenance_ready=True,
            emit_heartbeat=False,
        ),
        on_coalesce=_publish_coalesced_deployable,
    )


def _on_step_end(callback_state: _CheckpointUploadState, state, control):
    if int(getattr(state, "global_step", 0) or 0) in callback_state.required_steps:
        control.should_save = True
    return control


def _on_train_begin(callback_state: _CheckpointUploadState, state, control):
    # resume credits a required save only when its deployable adapter is verified on hf.
    # crediting the restored step alone could accept a save that never reached hf.
    resumed_step = int(getattr(state, "global_step", 0) or 0)
    for step in callback_state.required_steps:
        if step <= resumed_step and _hf._deployable_adapter_on_hf(step):
            callback_state.deployable_steps.add(step)
            callback_state.uploaded_steps.add(step)
    return control


def _on_save(callback_state: _CheckpointUploadState, args, state) -> None:
    step = int(state.global_step)
    if not _hf._w.HF_REPO:
        if step in callback_state.required_steps:
            raise RuntimeError(f"required save step {step} has no artifact repository")
        return
    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
    if not os.path.isdir(ckpt_dir):
        if step in callback_state.required_steps:
            raise RuntimeError(f"required save step {step} has no trainer checkpoint directory")
        return
    if step not in callback_state.required_steps:
        _enqueue_optional(callback_state, step, ckpt_dir)
        return
    if not _upload(
        callback_state,
        step,
        ckpt_dir,
        lock_timeout_s=_hf._checkpoint_upload_lock_timeout(),
    ):
        raise RetriableInfraError(
            f"required save step {step} full-state checkpoint was not durably published"
        )


def _on_train_end(callback_state: _CheckpointUploadState, args) -> None:
    if not _hf._w.HF_REPO:
        if callback_state.required_steps:
            raise RuntimeError("required saves have no artifact repository")
        return
    latest = _hf._latest_checkpoint_dir(args.output_dir)
    if latest is not None:
        step, ckpt_dir = latest
        should_flush = not callback_state.required_steps or step in callback_state.required_steps
        if (
            should_flush
            and step not in callback_state.uploaded_steps
            and not _upload(
                callback_state,
                step,
                ckpt_dir,
                lock_timeout_s=_hf._checkpoint_upload_lock_timeout(),
            )
        ):
            if step in callback_state.required_steps:
                raise RetriableInfraError(
                    f"required save step {step} full-state checkpoint was not durably published"
                )
            print(
                f"[ckpt] final resume checkpoint step {step} not durable on HF after retries; "
                "the deployable adapter save is preserved."
            )
    missing_required = sorted(callback_state.required_steps - callback_state.deployable_steps)
    if missing_required:
        raise RuntimeError(f"required saves were not durably published: {missing_required}")


def make_checkpoint_upload_callback(save_at_steps=()):
    """stream optional saves in the background while keeping required saves synchronous."""
    from transformers import TrainerCallback

    callback_state = _CheckpointUploadState(
        required_steps=frozenset(int(step) for step in save_at_steps)
    )

    class _CheckpointUpload(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            return _on_step_end(callback_state, state, control)

        def on_train_begin(self, args, state, control, **kwargs):
            return _on_train_begin(callback_state, state, control)

        def on_save(self, args, state, control, **kwargs):
            return _on_save(callback_state, args, state)

        def on_train_end(self, args, state, control, **kwargs):
            return _on_train_end(callback_state, args)

    return _CheckpointUpload()


# bound after definitions so importing this sibling directly cannot fail on the parent re-export.
from flash.engine.worker.io import hf as _hf  # noqa: E402

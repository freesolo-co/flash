"""Managed worker: remote primitives, standard Flash checkpoints and run telemetry."""

import json
import os
import tempfile
import time
from pathlib import Path

from flash.core.checkpoint_policy import is_spot_spec
from flash.custom.backend import load_backend
from flash.custom.checkpoints import RunCheckpoints
from flash.custom.server import serve_session
from flash.custom.session import TrainingSession
from flash.engine.worker.io import artifacts as hf
from flash.engine.worker.io.heartbeat import heartbeat, liveness_heartbeat
from flash.engine.worker.train.core.child.checkpoint_schedule import CheckpointClock


def _custom_targeting(model_id):
    """Custom runs adapt only text modules; a warm-start source must have the same modality."""
    from flash.core.catalog import supports_image_training

    if not supports_image_training(model_id):
        return None
    from flash.adapters.targets import resolve_lora_targeting

    return resolve_lora_targeting(model_id, algorithm="custom", multimodal=False)


def checkpoint_due(train, clock, step, max_steps):
    """Whether an optimizer step publishes a checkpoint.

    Spot (``clock`` set, ``save_every`` normalized away): automatic recovery saves before another
    typical step would leave more than the Spot recovery target of unsaved work, plus the first,
    requested and final steps. Otherwise the authored interval, requested and final steps.
    """
    if clock is not None:
        return clock.due(step, required=train.save_at_steps, final_step=max_steps)
    return step % train.save_every == 0 or step in train.save_at_steps or step == max_steps


def _latest_checkpoint_or_retry(lookup, spec):
    """Checkpoint discovery whose transient storage failures reschedule the attempt.

    A temporary artifact-store outage must not permanently fail a Spot run that has a durable
    checkpoint waiting to be resumed, so classify it like the DONE-marker read does.
    """
    from flash.artifacts.errors import artifact_error_transience
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    try:
        return lookup(spec)
    except Exception as exc:
        if artifact_error_transience(exc) is True:
            raise RetriableInfraError(
                f"custom checkpoint discovery failed transiently ({type(exc).__name__})"
            ) from exc
        raise


def run(*, context):
    spec = context.config.spec
    checkpoints = RunCheckpoints(context)
    resolved = spec.custom.get("resolved_resume")
    spot = is_spot_spec(spec)
    if spot:
        from flash.custom.managed import latest_run_checkpoint

        # A replacement for a preempted Spot VM continues from this run's newest durable state.
        latest = _latest_checkpoint_or_retry(latest_run_checkpoint, spec)
        if latest is not None:
            resolved = {key: latest[key] for key in ("repo", "revision", "path")}
    heartbeat("custom_model_load", context=context)
    with (
        liveness_heartbeat("custom_model_load", context=context),
        tempfile.TemporaryDirectory(prefix="flash-custom-resume-adapter-") as workspace,
    ):
        init_adapter = None
        if resolved:
            # a resumed checkpoint keeps the LoRA layout it was trained with, including a
            # warm-started source's multimodal module paths; restore then loads its exact state.
            init_adapter = checkpoints.download_adapter(resolved, workspace)
        elif spec.train.init_from_adapter:
            from flash.engine.worker.train.entry.sft_train import _warmstart_adapter_path

            # downloads the prepared source and checks its rank, alpha, modules and base model.
            init_adapter = _warmstart_adapter_path(
                spec.model,
                spec.model_revision,
                spec.train.lora_rank,
                spec.train.lora_alpha,
                _custom_targeting(spec.model),
                context=context,
            )
        backend = load_backend(
            spec.model,
            revision=spec.model_revision,
            lora_rank=spec.train.lora_rank,
            lora_alpha=spec.train.lora_alpha,
            seed=spec.seed,
            checkpoint_dir="/tmp/flash-custom-checkpoints",
            max_tokens=spec.train.max_context_tokens,
            gpu_count=spec.gpu.count,
            init_adapter=init_adapter,
        )
    backend.checkpoints = checkpoints
    session = TrainingSession(
        backend, max_steps=spec.train.max_steps, idle_seconds=spec.custom["idle_seconds"]
    )
    if resolved:
        restored = checkpoints.restore(backend, resolved)
        session.step = restored["checkpoint"]["step"]
        session.restored_state = restored["algorithm_state"]
        backend.algorithm_state = session.restored_state
    started = time.monotonic()
    session.last_activity = started
    started_at = time.time()

    def save():
        if session.step not in checkpoints.published:
            backend.save_checkpoint(
                f"step-{session.step}",
                metadata={"step": session.step, "model_version": session.version},
                algorithm_state=backend.algorithm_state,
            )
        return f"{spec.run_id}/step-{session.step}"

    clock = CheckpointClock() if spot else None

    def committed(current, operation, output):
        if operation == "optim_step":
            if checkpoint_due(spec.train, clock, current.step, current.max_steps):
                output["checkpoint_ref"] = save()
                if clock is not None:
                    clock.saved()
            heartbeat(
                "custom_step",
                step=current.step,
                total_steps=current.max_steps,
                metrics_last=[{"step": current.step, **output.get("metrics", {})}],
                metrics=output.get("metrics", {}),
                session_seconds=time.monotonic() - started,
                session_started_at=started_at,
                context=context,
            )
        elif operation == "close":
            output["checkpoint_ref"] = save()
            checkpoints.finish(checkpoints.published[current.step])

    session.on_commit = committed
    last_tick = time.monotonic()

    def tick():
        nonlocal last_tick
        if time.monotonic() - last_tick >= 60:
            heartbeat(
                "custom_waiting",
                step=session.step,
                session_seconds=time.monotonic() - started,
                session_started_at=started_at,
                context=context,
            )
            last_tick = time.monotonic()

    heartbeat(
        "custom_waiting",
        step=session.step,
        session_seconds=0,
        session_started_at=started_at,
        context=context,
    )
    tls = None
    if os.environ.get("FLASH_CUSTOM_TLS_CERT") and os.environ.get("FLASH_CUSTOM_TLS_KEY"):
        tls = {
            "cert": os.environ["FLASH_CUSTOM_TLS_CERT"],
            "key": os.environ["FLASH_CUSTOM_TLS_KEY"],
        }
    serve_session(
        session,
        os.environ["FLASH_CUSTOM_SESSION_TOKEN"],
        context.remaining_wall_seconds() or spec.gpu.max_wall_seconds,
        tick=tick,
        tls=tls,
    )
    if session.failed or not session.closed:
        raise RuntimeError(
            "custom controller disconnected or timed out; resume from the last durable checkpoint"
        )
    seconds = time.monotonic() - started
    metrics = {"steps": session.step, "session_seconds": seconds, "train_seconds": seconds}
    path = Path("/tmp/metrics.json")
    path.write_text(json.dumps(metrics))
    hf.upload_artifact_file(str(path), "metrics.json", required=True, context=context)
    done = Path("/tmp/custom-DONE")
    done.write_text("done")
    hf.upload_artifact_file(str(done), "DONE", required=True, context=context)
    heartbeat(
        "done",
        step=session.step,
        session_seconds=seconds,
        session_started_at=started_at,
        context=context,
    )

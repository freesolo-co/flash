"""Managed worker: remote primitives, standard Flash checkpoints and run telemetry."""

import json
import time
from pathlib import Path

from flash.custom.backend import load_backend
from flash.custom.checkpoints import RunCheckpoints
from flash.custom.managed import session_token
from flash.custom.server import serve_session
from flash.custom.session import TrainingSession
from flash.engine.worker.io import hf
from flash.engine.worker.io.heartbeat import heartbeat, liveness_heartbeat


def run(*, context):
    spec = context.config.spec
    heartbeat("custom_model_load", context=context)
    with liveness_heartbeat("custom_model_load", context=context):
        backend = load_backend(
            spec.model,
            revision=spec.model_revision,
            lora_rank=spec.train.lora_rank,
            lora_alpha=spec.train.lora_alpha,
            seed=spec.seed,
            checkpoint_dir="/tmp/flash-custom-checkpoints",
            max_tokens=spec.train.max_context_tokens,
        )
    checkpoints = RunCheckpoints(context)
    backend.checkpoints = checkpoints
    session = TrainingSession(
        backend, max_steps=spec.train.max_steps, idle_seconds=spec.custom["idle_seconds"]
    )
    resolved = spec.custom.get("resolved_resume")
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

    def committed(current, operation, output):
        if operation == "optim_step":
            if (
                current.step % spec.train.save_every == 0
                or current.step in spec.train.save_at_steps
                or current.step == current.max_steps
            ):
                output["checkpoint_ref"] = save()
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
    serve_session(
        session,
        session_token(spec.run_id, context.config.attempt),
        context.remaining_wall_seconds() or spec.gpu.max_wall_seconds,
        tick=tick,
    )
    if session.failed or not session.closed:
        raise RuntimeError(
            "custom controller disconnected or timed out; resume from the last durable checkpoint"
        )
    seconds = time.monotonic() - started
    metrics = {"steps": session.step, "session_seconds": seconds, "train_seconds": seconds}
    path = Path("/tmp/metrics.json")
    path.write_text(json.dumps(metrics))
    hf.hf_upload_file(str(path), "metrics.json", required=True, context=context)
    done = Path("/tmp/custom-DONE")
    done.write_text("done")
    hf.hf_upload_file(str(done), "DONE", required=True, context=context)
    heartbeat(
        "done",
        step=session.step,
        session_seconds=seconds,
        session_started_at=started_at,
        context=context,
    )

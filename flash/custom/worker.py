"""Managed worker: remote primitives, standard Flash checkpoints and run telemetry."""

import json
import os
import tempfile
import time
from pathlib import Path

from flash.core.checkpoint_policy import is_spot_spec
from flash.custom.backend import load_backend
from flash.custom.billing import PRIOR_SESSION_SECONDS_ENV
from flash.custom.checkpoints import RunCheckpoints
from flash.custom.server import serve_session
from flash.custom.session import TrainingSession
from flash.engine.worker.io import artifacts as hf
from flash.engine.worker.io.heartbeat import heartbeat, liveness_heartbeat
from flash.engine.worker.train.core.child.checkpoint_schedule import CheckpointClock


def checkpoint_due(train, clock, step, max_steps):
    """Whether an optimizer step publishes a checkpoint.

    Spot (``clock`` set, ``save_every`` normalized away): automatic recovery saves before another
    typical step would leave more than the Spot recovery target of unsaved work, plus the first,
    requested and final steps. Otherwise the authored interval, requested and final steps.
    """
    if clock is not None:
        return clock.due(step, required=train.save_at_steps, final_step=max_steps)
    return step % train.save_every == 0 or step in train.save_at_steps or step == max_steps


# Stop serving this long before the cap so the reached step can still be saved durably.
CAP_SAVE_RESERVE_SECONDS = 180
SAVE_LOCK_SECONDS = 60


def session_cap_seconds(spec, context):
    """Seconds this attempt may serve: custom.max_wall_seconds counts GPU session time only.

    The run's own deadline (gpu.max_wall_seconds from submission) additionally covers queue
    time, like any other algorithm's run, and stays the hard ceiling.
    """
    try:
        prior = float(os.environ.get(PRIOR_SESSION_SECONDS_ENV) or 0)
    except ValueError:
        prior = 0.0
    if not prior >= 0 or prior == float("inf"):
        prior = 0.0
    cap = max(0.0, float(spec.custom["max_wall_seconds"]) - prior)
    remaining = context.remaining_wall_seconds()
    return cap if remaining is None else min(cap, remaining)


def cap_save_reserve(cap):
    return min(CAP_SAVE_RESERVE_SECONDS, cap / 10)


class SessionMeter:
    """Billable session time: it starts when a controller first reaches the session.

    A GPU that loaded and waited but was never reached by any controller has no session time.
    """

    def __init__(self):
        self.started = None
        self.started_at = None

    def attach(self, _session=None):
        # runs on the server's event loop; the next tick heartbeat reports it.
        self.started = time.monotonic()
        self.started_at = time.time()

    def seconds(self):
        return 0.0 if self.started is None else time.monotonic() - self.started

    def fields(self):
        fields = {
            "session_seconds": self.seconds(),
            "controller_attached": self.started_at is not None,
        }
        if self.started_at is not None:
            fields["session_started_at"] = self.started_at
        return fields


def unfinished_session_error(session, reason, saved):
    """The worker error for a session that ended without the controller closing it."""
    kept = f"; saved checkpoint {saved}" if saved else ""
    if reason == "wall":
        return RuntimeError(
            f"custom session reached its wall-clock cap{kept}; resume from the last durable checkpoint"
        )
    if session.attached_at is None:
        return RuntimeError(
            "no custom controller attached to the session; no session time was billed"
        )
    return RuntimeError(
        f"custom controller disconnected or timed out{kept}; resume from the last durable checkpoint"
    )


def save_unfinished_session(session, save, initial_step=0):
    """Publish the reached step of a session that ends without close; never a failed one.

    A failed session may hold a partially applied update, so its weights are not saved. A
    running operation is given SAVE_LOCK_SECONDS to finish; otherwise nothing is saved.
    """
    if session.failed or session.closed or session.step <= initial_step:
        return None
    if not session.lock.acquire(timeout=SAVE_LOCK_SECONDS):
        return None
    try:
        if session.failed:
            return None
        session.closed = True  # no further operations may change the saved weights
        return save()
    except Exception as error:
        print(f"custom: could not save the unfinished session ({type(error).__name__})", flush=True)
        return None
    finally:
        session.lock.release()


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

            # downloads the prepared source and checks its rank, alpha and base model. No
            # modality is imposed: custom sessions accept image rows, and the backend continues
            # the source's exact LoRA modules (text-only, or language + vision from image SFT),
            # failing if any source tensor does not land on a trainable parameter.
            init_adapter = _warmstart_adapter_path(
                spec.model,
                spec.model_revision,
                spec.train.lora_rank,
                spec.train.lora_alpha,
                None,
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
    # The session clock (billing and session_seconds) starts when a controller first reaches the
    # session, not when the GPU finished loading: a session no controller ever attached to has no
    # billable session time. The idle clock still starts now (unattached: see TrainingSession).
    session.last_activity = time.monotonic()
    meter = SessionMeter()
    session.on_attach = meter.attach
    session_fields = meter.fields

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
                **session_fields(),
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
            heartbeat("custom_waiting", step=session.step, **session_fields(), context=context)
            last_tick = time.monotonic()

    heartbeat("custom_waiting", step=session.step, **session_fields(), context=context)
    tls = None
    if os.environ.get("FLASH_CUSTOM_TLS_CERT") and os.environ.get("FLASH_CUSTOM_TLS_KEY"):
        tls = {
            "cert": os.environ["FLASH_CUSTOM_TLS_CERT"],
            "key": os.environ["FLASH_CUSTOM_TLS_KEY"],
        }
    initial_step = session.step
    cap = session_cap_seconds(spec, context)
    reason = serve_session(
        session,
        os.environ["FLASH_CUSTOM_SESSION_TOKEN"],
        max(1.0, cap - cap_save_reserve(cap)),
        tick=tick,
        tls=tls,
    )
    if session.failed or not session.closed:
        raise unfinished_session_error(
            session, reason, save_unfinished_session(session, save, initial_step)
        )
    seconds = meter.seconds()
    metrics = {"steps": session.step, "session_seconds": seconds, "train_seconds": seconds}
    path = Path("/tmp/metrics.json")
    path.write_text(json.dumps(metrics))
    hf.upload_artifact_file(str(path), "metrics.json", required=True, context=context)
    done = Path("/tmp/custom-DONE")
    done.write_text("done")
    hf.upload_artifact_file(str(done), "DONE", required=True, context=context)
    heartbeat("done", step=session.step, **session_fields(), context=context)

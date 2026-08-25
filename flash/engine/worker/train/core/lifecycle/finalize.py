"""Train-metadata + run-metrics finalize for the worker."""

from __future__ import annotations

import json
import os

from flash.engine.result.accounting import RunMetrics, sanitize_worker_metrics
from flash.engine.worker.io import hf as hf_io
from flash.engine.worker.io import progress as progress_io
from flash.engine.worker.io import result as result_io
from flash.engine.worker.perf import gpu_diagnostics
from flash.engine.worker.runtime import state as worker_state


def _train_meta_job_spec():
    """the run's spec for the uploaded metrics artifact, with the base revision it trained on.

    to_dict() is the PUBLIC spec and no longer emits model_revision, but this artifact records what
    the worker actually trained against -- a run reproduced without its base revision is not the
    same run. to_internal_dict() is the wrong fix: it would publish train.hf_repo and
    train.init_from_adapter_revision, internal storage locators, into an artifact users download.
    so add back the one field, and only when it is set.
    """
    spec = worker_state.JOB_SPEC
    if not spec:
        return None
    data = spec.to_dict()
    if spec.model_revision:
        data["model_revision"] = spec.model_revision
    return data


def write_train_meta(
    phase,
    adapter_dir,
    model_id,
    train_wall,
    setup_seconds,
    train_tokens,
    generated_tokens,
    notes,
    *,
    step=None,
    progress_fields=None,
):
    env = worker_state.require_active_env()
    meta = sanitize_worker_metrics(
        {
            "phase": phase,
            "adapter_dir": adapter_dir,
            "model_id": model_id,
            "train_wall": train_wall,
            "setup_seconds": setup_seconds,
            "train_tokens": train_tokens,
            "generated_tokens": generated_tokens,
            "notes": notes or {},
        }
    )
    with open("/tmp/train_meta.json", "w") as f:
        json.dump(meta, f)
    hf_io.hf_upload_file("/tmp/train_meta.json", "train_meta.json")
    # carry the completed optimizer step into the final cumulative progress record.
    _step_field = {"step": int(step)} if isinstance(step, (int, float)) and step > 0 else {}
    _progress_fields = progress_fields or {}
    progress_io.publish_progress(
        f"{phase}_train_done",
        **_step_field,
        **{k: meta[k] for k in ("train_wall", "train_tokens", "generated_tokens")},
        **_progress_fields,
        gpu=gpu_diagnostics(),
    )
    m = RunMetrics(
        arm=os.environ.get("FLASH_ARM", "runpod"),
        phase=phase,
        # Completed optimizer updates (opd passes step=opt_steps; sft/rl omit it -> None). _finalize
        # reads metrics.step to carry the true step onto the terminal `done` progress.
        step=step,
        seed=worker_state.SEED,
        model_id=model_id,
        wall_seconds=train_wall,
        setup_seconds=setup_seconds,
        train_throughput_toks_per_s=(
            (generated_tokens or train_tokens) / train_wall if train_wall else 0.0
        ),
        train_tokens=train_tokens,
        generated_tokens=generated_tokens,
        notes={
            **(notes or {}),
            "renderer": "flash_env",
            "thinking": worker_state.THINKING,
            "train_wall": train_wall,
            "model_id": model_id,
            "environment": env.id,
            "job_spec": _train_meta_job_spec(),
        },
    )
    _finalize(m, progress_fields=_progress_fields)


def _finalize(metrics: RunMetrics, *, progress_fields=None):
    metrics.save("/tmp/metrics.json")
    step = metrics.step
    completed_steps = int(step) if isinstance(step, (int, float)) and step > 0 else 0
    manifest = result_io.publish_result(
        outcome="succeeded",
        failure_class=None,
        started_at=worker_state.WORKER_START_TIME,
        training_entered=True,
        completed_steps=completed_steps,
        metrics=json.loads(metrics.to_json()),
        checkpoint={"failure": (progress_fields or {}).get("checkpoint_failure")},
        artifacts={"metrics": "embedded", "adapter": "published"},
    )
    progress_io.publish_progress(
        "result_published",
        step=completed_steps,
        result_path=result_io.result_path(manifest),
        gpu=gpu_diagnostics(),
    )
    print("NODE RESULT:", metrics.to_json())

"""managed gpu worker process orchestration."""

from __future__ import annotations

import os
import sys
import time
import traceback

import flash.engine.worker.entry.opd as opd_entry
import flash.engine.worker.entry.rl as rl_entry
import flash.engine.worker.entry.sft as sft_entry
import flash.engine.worker.io.hf as hf_io
import flash.engine.worker.io.progress as progress_io
import flash.engine.worker.io.result as result_io
import flash.engine.worker.perf as worker_perf
import flash.engine.worker.runtime.kernel_warmup as kernel_warmup
import flash.engine.worker.train.entry.backend_common as backend_common
from flash._internal.diagnostics import sanitize_diagnostic
from flash.core.spec import gpu_count_of
from flash.engine.worker.runtime import state
from flash.envs.loading.staged import (
    StagedEnvironmentTransientError,
)
from flash.envs.meta.identity import GitHubTransientError


def _worker_failure_flags(exc: BaseException) -> dict[str, bool]:
    # both transient families, not just the staged one: `GitHubTransientError` (quota and outage)
    # still reaches here from the environment identity path, and the worker's answer to either is
    # the same -- reschedule. classifying only the staged type would fail those runs permanently.
    retriable = isinstance(
        exc,
        (worker_perf.RetriableInfraError, StagedEnvironmentTransientError, GitHubTransientError),
    )
    return {"retriable": retriable, "oom": (not retriable and worker_perf.is_cuda_oom(exc))}


def _preflight_gpu_occupancy_for_spec() -> None:
    """Screen every GPU in the allocator-resolved worker allocation for substantial occupancy.

    The allocator stamps the chosen count onto ``JOB_SPEC`` before provider launch, so this passes
    only the cards rented for this worker. It does not re-derive required VRAM or claim exact fit.

    This must stay the first CUDA-adjacent call in boot. The reading is trustworthy only while this
    process has no context of its own, and ``_force_fla_triton_gdn_on_sm100`` below creates one.
    """
    worker_perf.preflight_gpu_occupancy(gpu_count_of(state.JOB_SPEC))


def _run_worker_mode() -> None:
    # first statement in the worker, ahead of every import below it: `huggingface_hub.constants`
    # captures HF_HUB_DISABLE_XET when it is imported, so setting it afterwards has no effect on
    # which upload path is used. anything imported before this line could freeze the default.
    hf_io._disable_xet_upload_staging()
    if result_io.read_existing_terminal_result() is not None:
        return

    modes = {
        "sft": sft_entry.run_sft,
        "rl": rl_entry.run_rl,
        "opd": opd_entry.run_opd,
    }
    handler = modes.get(state.RUN_MODE)
    if handler is None:
        raise RuntimeError("worker run mode is invalid")
    remaining = state._remaining_worker_wall_seconds()
    if remaining is not None and remaining <= 0:
        raise RuntimeError("worker run wall deadline exceeded")
    # these setups run in the parent; verl trains in FLASH_VERL_PYTHON.
    # only _force_fla_triton_gdn_on_sm100 propagates through FLA_* env vars. the install and the
    # monkeypatch are interpreter-local and must not be treated as child configuration. the tilelang
    # libcudart repoint is NOT here: it is interpreter-local too, and the interpreter that needs it is
    # the verl child, so it ships as a sitecustomize fragment (verl.child_io).
    # FIRST, and specifically before `_force_fla_triton_gdn_on_sm100`: that reads
    # `get_device_capability`, which initializes CUDA in this process, and from that moment the
    # driver's "used" includes our own context with no sound way to subtract it -- the occupancy
    # check declines to run rather than guess (see `preflight_gpu_occupancy`), so calling it later
    # means never calling it at all. this is also before `_ensure_fla_fastpath_on_hopper`, whose repair
    # path runs pip installs with 600s timeouts: a card handed over with a co-tenant's ~18GB still
    # resident fails the run either way, and the only question is whether that happens now or after
    # dependency repair, the model download and FSDP init have spent paid GPU on the same
    # conclusion. NVML answers without a context, so nothing here dirties the reading for the code
    # below.
    _preflight_gpu_occupancy_for_spec()
    # run before model imports: sm100 tilelang GDN computes wrong gradients, so use Triton.
    worker_perf._force_fla_triton_gdn_on_sm100()
    worker_perf._ensure_fla_fastpath_on_hopper()
    # AFTER the fla fast path (which may (re)install fla), BEFORE any model import / GDN
    # launch: restrict fla's Blackwell GDN bwd autotune to grad-correct configs (fla #913).
    worker_perf._restrict_fla_gdn_autotune_on_blackwell()
    progress_io.publish_progress("boot", gpu=worker_perf.gpu_diagnostics(include_torch=False))
    kernel_warmup.load_mega_cache()
    completed = False
    try:
        handler()
        completed = True
    finally:
        try:
            if completed:
                progress_io.flush_progress()
        finally:
            state._cleanup_active_env_package()
    # hard-exit: colocated vllm can deadlock on nccl/cuda teardown; all artifacts are already on hf.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    try:
        _run_worker_mode()
    except Exception as e:
        tb = sanitize_diagnostic(traceback.format_exc(), limit=16_000)
        try:
            err_name = hf_io.error_artifact_name(state.RUN_MODE, state.ATTEMPT)
            err_path = f"/tmp/{err_name}"
            with open(err_path, "w") as f:
                f.write(tb)
            hf_io.hf_upload_file(err_path, err_name)
        except Exception as up_err:
            print("error-upload warn:", sanitize_diagnostic(up_err, limit=500))
        try:
            # preserve ray session logs because raylet tracebacks show only the downstream EOF and
            # pod-local logs vanish after failure (VERL-115). all training modes start ray.
            ray_logs = backend_common.collect_ray_failure_logs(
                started_after=state.WORKER_START_TIME
            )
            if ray_logs:
                ray_name = hf_io.ray_log_artifact_name(state.RUN_MODE, state.ATTEMPT)
                ray_path = f"/tmp/{ray_name}"
                with open(ray_path, "w") as f:
                    f.write(ray_logs)
                hf_io.hf_upload_file(ray_path, ray_name)
        except Exception as ray_err:
            # this is the failure path already. a collector that could raise here would replace the
            # run's real error with its own.
            print("ray-log-collect warn:", sanitize_diagnostic(ray_err, limit=500))
        flags = _worker_failure_flags(e)
        detail = sanitize_diagnostic(e, limit=500)
        failure_class = (
            "oom" if flags["oom"] else "artifact_transport" if flags["retriable"] else "worker"
        )
        # preserve the bounded metric backlog on both the primary and fallback error progress.
        _err_metrics = (
            {"metrics_last": list(progress_io.LATEST_GRPO_METRICS)}
            if progress_io.LATEST_GRPO_METRICS
            else {}
        )
        progress_publication_error: Exception | None = None
        try:
            progress_io.publish_progress(
                f"error_{state.RUN_MODE}",
                error=detail,
                **flags,
                **_err_metrics,
                gpu=worker_perf.gpu_diagnostics(),
            )
        except Exception as progress_error:
            progress_publication_error = progress_error
        result_publication_error: Exception | None = None
        try:
            result_io.publish_result(
                outcome="failed",
                failure_class=failure_class,
                started_at=state.WORKER_START_TIME,
                training_entered=progress_io._PROGRESS_TRAINING_ENTERED,
                completed_steps=progress_io._PROGRESS_COMPLETED_STEPS,
                metrics={},
                checkpoint={
                    "failure": progress_io._PROGRESS_PENDING_CHECKPOINT_FAILURE,
                },
                artifacts={"error": err_name if "err_name" in locals() else None},
                diagnostics={
                    "error": detail,
                    "traceback": tb,
                    **(
                        {"progress_publication_error": progress_publication_error}
                        if progress_publication_error is not None
                        else {}
                    ),
                },
            )
        except Exception as result_error:
            result_publication_error = result_error
            print(
                "result-publication failed:",
                sanitize_diagnostic(result_error, limit=500),
                flush=True,
            )
        remaining = state._remaining_worker_wall_seconds()
        delay = 10.0 if remaining is None else min(10.0, remaining)
        if delay > 0:
            time.sleep(delay)
        if result_publication_error is not None:
            raise result_publication_error from e
        raise

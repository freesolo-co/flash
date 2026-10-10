"""managed gpu worker process orchestration."""

from __future__ import annotations

import os
import sys
import time
import traceback

import flash.engine.worker.entry.opd as opd_entry
import flash.engine.worker.entry.rl as rl_entry
import flash.engine.worker.entry.sft as sft_entry
import flash.engine.worker.io.heartbeat as heartbeat_io
import flash.engine.worker.io.hf as hf_io
import flash.engine.worker.perf as worker_perf
import flash.engine.worker.runtime.kernel_warmup as kernel_warmup
import flash.engine.worker.train.entry.backend_common as backend_common
from flash._internal.diagnostics import sanitize_diagnostic
from flash.core.spec import gpu_count_of
from flash.engine.support.huggingface import hub_error_transience
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
    # Idempotency: check DONE before any env-mutating pip install (fla fast path).
    if state.HF_REPO:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import RemoteEntryNotFoundError

        try:
            hf_hub_download(
                repo_id=state.HF_REPO,
                repo_type="dataset",
                filename=f"{hf_io.hf_prefix()}/DONE",
                token=os.environ.get("HF_TOKEN"),
            )
            done = True
        except Exception as exc:
            if type(exc) is RemoteEntryNotFoundError:
                done = False
            elif hub_error_transience(exc) is True:
                raise worker_perf.RetriableInfraError(
                    f"DONE marker read failed transiently ({type(exc).__name__})"
                ) from exc
            else:
                raise
        remaining = state._remaining_worker_wall_seconds()
        if remaining is not None and remaining <= 0:
            raise RuntimeError("worker run wall deadline exceeded")
        if done:
            print("Run already complete (DONE present); returning persisted metrics.")
            heartbeat_io.heartbeat(
                "already_done", gpu=worker_perf.gpu_diagnostics(include_torch=False)
            )
            # DONE is written only AFTER metrics.json uploads (required=True), so a failed read here
            # is a transient HF blip, never a missing file. Retry, then signal RETRIABLE (reschedule)
            # rather than SystemExit — a BaseException that bypasses the retriable-stamping handler
            # below and would report a genuinely-succeeded run as a fatal failure.
            last_err: Exception | None = None
            for attempt in range(3):
                remaining = state._remaining_worker_wall_seconds()
                if remaining is not None and remaining <= 0:
                    break
                try:
                    got = hf_hub_download(
                        repo_id=state.HF_REPO,
                        repo_type="dataset",
                        filename=f"{hf_io.hf_prefix()}/metrics.json",
                        token=os.environ.get("HF_TOKEN"),
                    )
                    import shutil

                    shutil.copy(got, "/tmp/metrics.json")
                    sys.stdout.flush()
                    os._exit(0)
                except Exception as e:
                    last_err = e
                    if attempt >= 2:
                        break
                    delay = 5 * (attempt + 1)
                    remaining = state._remaining_worker_wall_seconds()
                    if remaining is not None:
                        if remaining <= 0:
                            break
                        delay = min(delay, remaining)
                    if delay > 0:
                        time.sleep(delay)
            error_kind = type(last_err).__name__ if last_err is not None else "unknown error"
            raise worker_perf.RetriableInfraError(
                "DONE present but metrics.json unreadable after retries "
                f"(transient HF; {error_kind})"
            )
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
    heartbeat_io.heartbeat("boot", gpu=worker_perf.gpu_diagnostics(include_torch=False))
    kernel_warmup.load_mega_cache()
    try:
        handler()
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
        # A CUDA OOM -> stamp an ``oom`` flag so the runner retries on a LARGER GPU. Infra failures
        # keep same-size retry semantics and must never be reclassified as OOM.
        hb_flags = _worker_failure_flags(e)
        detail = sanitize_diagnostic(e, limit=500)
        # preserve the bounded metric backlog on BOTH the primary and the fallback error
        # heartbeat. compute it (and detail) before the guarded call -- both are cheap and
        # cannot raise -- so that if the primary heartbeat fails (most likely worker_perf.gpu_diagnostics()
        # or the upload itself), the fallback still carries metrics_last, which is the backlog
        # this path exists to surface for short failing RL runs.
        _err_metrics = (
            {"metrics_last": list(heartbeat_io.LATEST_GRPO_METRICS_LAST)}
            if heartbeat_io.LATEST_GRPO_METRICS_LAST
            else {}
        )
        try:
            heartbeat_io.heartbeat(
                f"error_{state.RUN_MODE}",
                error=detail,
                **hb_flags,
                **_err_metrics,
                # `gpu=`, like every other producer here: this was the one path spelling it `diag`,
                # and the consumer reads `gpu` alone. the mismatch is worse than a missing field --
                # `record_heartbeat` assigns `gpu_status` unconditionally, so an error heartbeat
                # whose diagnostics land under an unread key CLEARS the snapshot a healthy
                # heartbeat already stored, losing the evidence exactly when an oom needs it.
                gpu=worker_perf.gpu_diagnostics(),
            )
        except Exception:
            heartbeat_io.heartbeat(
                f"error_{state.RUN_MODE}", error=detail, **hb_flags, **_err_metrics
            )
        remaining = state._remaining_worker_wall_seconds()
        delay = 10.0 if remaining is None else min(10.0, remaining)
        if delay > 0:
            time.sleep(delay)
        raise

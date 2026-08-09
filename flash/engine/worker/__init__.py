"""On-GPU fine-tuning worker. Modes: sft | rl.

Namespace hub: owns run-scoped STATE globals, defines ``main`` + ``_finalize``, re-exports
every submodule name. Submodules read state via ``_w.<name>`` (live-package proxy) so
``monkeypatch.setattr(worker, "<name>", ...)`` is seen at call time, not captured at import.
"""

from __future__ import annotations

import math
import os
import sys

# Unused here; ``worker.threading.Thread`` is patched by tests to run upload threads inline.
import threading  # noqa: F401
import time
import traceback

from flash._internal.diagnostics import sanitize_diagnostic
from flash.core.spec import FIXED_SEED, load_job_spec_from_env
from flash.engine.result.accounting import RunMetrics
from flash.engine.worker.backend_common import collect_ray_failure_logs
from flash.engine.worker.entry.opd import run_opd
from flash.engine.worker.entry.rl import run_rl
from flash.engine.worker.entry.sft import run_sft
from flash.engine.worker.io.heartbeat import (
    _HB_LOCK,
    _HB_SETUP_LIVENESS_STAGES,
    _HB_THROTTLED_STAGES,
    _HB_TIGHT_LIVENESS_STAGES,
    _HB_UPLOAD_LIVENESS_STAGES,
    _HB_UPLOAD_LOCK,
    LATEST_GRPO_METRICS_LAST,
    heartbeat,
    make_sft_heartbeat_callback,
)
from flash.engine.worker.io.hf import (
    _hf_upload,
    _latest_checkpoint_dir,
    error_artifact_name,
    flush_optional_uploads,
    hf_api,
    hf_prefix,
    hf_resume_checkpoint,
    hf_upload_file,
    hf_upload_folder,
    load_tokenizer,
    make_checkpoint_upload_callback,
    model_revision_kwargs,
    prefetch_model,
    publish_deployable_checkpoint,
    publish_opd_optimizer_start_marker,
    ray_log_artifact_name,
    upload_debug_jsonl,
    upload_resume_checkpoint,
    write_base_model_provenance,
)
from flash.engine.worker.io.wandb_log import (
    wandb_finish,
    wandb_report_to,
    wandb_run_info,
    wandb_run_name,
)
from flash.engine.worker.model.adapter import (
    _download_adapter,
    _init_adapter_model,
    _resolve_adapter_ref,
    lora_target_parameters,
    make_lora,
    prepare_fresh_lora_base,
    require_vllm_for_rollout_func,
    stamp_adapter_provenance,
    validate_lora_target_parameters,
)
from flash.engine.worker.model.decoding import (
    graded_text,
    prompt_opens_thinking,
    render_prompt,
    strip_think,
    think_token_count,
    thinking_text,
)
from flash.engine.worker.model.lora import (
    assert_adapter_delta_nonzero,
    assert_adapter_load_clean,
    assert_lora_applied,
    is_vl_checkpoint,
)
from flash.engine.worker.perf import (
    RetriableInfraError,
    _attn_impl_for_capability,
    _ensure_fla_fastpath_on_hopper,
    _estimate_params,
    _flash_attn_3_available,
    _flash_attn_available,
    _force_fla_triton_gdn_on_sm100,
    _liger_default_for_model,
    _memory_mode,
    _neutralize_tilelang_cudart_stub,
    _remove_fla_from_disk,
    _restrict_fla_gdn_autotune_on_blackwell,
    gpu_diagnostics,
    grad_checkpointing_on,
    grpo_use_reentrant,
    is_cuda_oom,
    optimal_attn_impl,
    wait_for_gpu,
)
from flash.engine.worker.runtime.kernel_warmup import _current_cuda_sm, load_mega_cache
from flash.engine.worker.runtime.rng import backend_seed, seed_training_rngs
from flash.engine.worker.train.finalize import write_train_meta
from flash.engine.worker.train.rl.config import (
    build_grpo_prompt_dataset,
    grpo_mask_truncated_completions,
    grpo_overrides,
    resolve_grpo_prompts_per_step,
)
from flash.envs.adapter import GitHubRateLimitError
from flash.envs.base import load_environment
from flash.teacher.retry_contract import OPD_RESUME_REVISION_ENV


def _resolve_worker_seed(job_spec, env_seed: str | None) -> int:
    if job_spec is not None:
        return int(job_spec.seed)
    try:
        seed = int(env_seed) if env_seed is not None else FIXED_SEED
    except (TypeError, ValueError):
        return FIXED_SEED
    return seed if 0 <= seed <= 2**63 - 1 else FIXED_SEED


def _parse_attempt_env() -> int:
    raw = os.environ.get("ATTEMPT")
    if raw is None:
        return 0
    if not raw or any(char < "0" or char > "9" for char in raw):
        raise RuntimeError("managed worker ATTEMPT must be an unsigned decimal integer")
    return int(raw)


HF_REPO = os.environ.get("HF_REPO", "")
RUN_ID = os.environ.get("RUN_ID", "local")
RUN_MODE = os.environ.get("RUN_MODE", "sft")
ATTEMPT = _parse_attempt_env()
# captured at import, which is before this attempt can have started ray. any ray session older than
# this belongs to a PREVIOUS attempt on the same reused pod (/tmp survives a retry), and reporting
# one as this attempt's evidence would send the next diagnosis after a failure that never happened.
WORKER_START_TIME = time.time()
JOB_SPEC = load_job_spec_from_env()
SEED = _resolve_worker_seed(JOB_SPEC, os.environ.get("SEED"))
PHASE = os.environ.get(
    "PHASE",
    JOB_SPEC.phase
    if JOB_SPEC
    else (RUN_MODE if RUN_MODE in ("sft", "rl", "opd", "profile") else "sft"),
)
OPD_RESUME_REVISION = os.environ.get(OPD_RESUME_REVISION_ENV, "").strip()

_HB_LAST_UPLOAD = 0.0
_HB_LAST_PROGRESS_TS = 0.0
# progress-carry latch: SEQ counts real (non-liveness) heartbeat calls; UPLOADED_SEQ is the newest
# real call whose snapshot actually committed to HF. SEQ > UPLOADED_SEQ means progress happened that
# HF has not seen (throttled away or a failed upload), so the next liveness ping is upgraded to a
# real heartbeat. Without this, a liveness ping can win the shared upload slot and defer the real
# per-step heartbeat past the provider's stall window, killing a healthy run.
_HB_PROGRESS_SEQ = 0
_HB_PROGRESS_UPLOADED_SEQ = 0
# Environment-scoped artifact repos are shared by many runs. Keep heartbeat commits below the HF
# per-repo commit cap while staying under the provider poller's training stall window
# (STALL_AFTER_S=1500s in flash/providers/_lifecycle/poll.py).
_HB_MIN_INTERVAL_S = 900.0
# highest optimizer step committed by a heartbeat.
# force bypasses the 900s throttle only on step advance so cancel billing stays current.
_HB_LAST_COMMITTED_STEP = 0
# forced step commits also obey _HB_FORCE_MIN_INTERVAL_S to protect the hf commit cap.
# measure from _HB_LAST_FORCED_UPLOAD so unrelated liveness uploads cannot delay a needed force;
# only sub-floor bursts coalesce, bounding cancel under-billing to one floor window.
_HB_LAST_FORCED_UPLOAD = 0.0
_HB_FORCE_MIN_INTERVAL_S = 60.0
# Setup liveness is the user-visible signal during cold model download/load. Keep it below common
# external "frozen heartbeat" thresholds without relaxing the noisy per-step training throttle.
_HB_SETUP_LIVENESS_INTERVAL_S = 240.0
_HB_TERMINAL_ONLY = False

_WANDB_FINISH_WAIT_S = 120.0
_WANDB_FINISH_FAIL_WAIT_S = 5.0


def _remaining_worker_wall_seconds() -> float | None:
    raw_deadline = os.environ.get("FLASH_RUN_DEADLINE_AT")
    if raw_deadline is None:
        return None
    try:
        deadline = float(raw_deadline)
    except (TypeError, ValueError):
        raise RuntimeError("worker run wall deadline is invalid") from None
    now = time.time()
    if deadline <= 0 or now <= 0 or not math.isfinite(deadline) or not math.isfinite(now):
        raise RuntimeError("worker run wall deadline is invalid")
    return max(0.0, deadline - now)


def _load_active_env():
    """Load the run's Freesolo environment from the JobSpec; require an explicit env."""
    if JOB_SPEC is None:
        return None
    env_id = JOB_SPEC.environment.id
    if not env_id:
        raise RuntimeError(
            "JobSpec sets no environment: provide [environment] id "
            "(a Freesolo environment id like 'your-name/your-env', returned by "
            "`flash env push --project <project-uuid> --name <name>`)."
        )
    env = load_environment(
        env_id, JOB_SPEC.environment.params, resolved_sha=JOB_SPEC.environment.resolved_sha
    )
    # tell the env whether this run samples <think> blocks, so the multi-turn scoring path strips
    # reasoning exactly like the single-turn path does (see FreesoloEnvironment.record_model_turn).
    if hasattr(env, "thinking"):
        env.thinking = bool(JOB_SPEC.thinking)
    return env


ACTIVE_ENV = None


def require_active_env():
    """Return the run's loaded environment, raising a clear error if none is loaded."""
    global ACTIVE_ENV
    if ACTIVE_ENV is None:
        ACTIVE_ENV = _load_active_env()
    if ACTIVE_ENV is None:
        raise RuntimeError(
            "no environment is loaded: this worker was started without a JobSpec "
            "(FLASH_JOB_SPEC_JSON / FLASH_JOB_SPEC_PATH is unset). A train/eval run must "
            "carry a JobSpec naming [environment] id "
            "(a Freesolo environment id like 'your-name/your-env', returned by "
            "`flash env push --project <project-uuid> --name <name>`)."
        )
    return ACTIVE_ENV


def _worker_failure_flags(exc: BaseException) -> dict[str, bool]:
    retriable = isinstance(exc, (RetriableInfraError, GitHubRateLimitError))
    return {"retriable": retriable, "oom": (not retriable and is_cuda_oom(exc))}


THINKING = JOB_SPEC.thinking if JOB_SPEC else False


def _finalize(metrics: RunMetrics, *, heartbeat_fields=None):
    if not flush_optional_uploads():
        print("optional upload flush timed out before final publication")
    metrics.save("/tmp/metrics.json")
    hf_upload_file("/tmp/metrics.json", "metrics.json", required=True)
    with open("/tmp/DONE", "w") as f:
        f.write(str(time.time()))
    hf_upload_file("/tmp/DONE", "DONE", required=True)
    # Carry the completed optimizer step onto the terminal `done` heartbeat. Without it, a cancel
    # that races the DONE upload (this stepless `done` heartbeat recorded, but the poller hasn't
    # transitioned the run to done yet) prices from actual_steps_run(), which treats `done` as a
    # non-training stage with no step and floors a fully-trained run to 0. RunMetrics.step carries
    # the completed optimizer updates for opd; None (other phases) -> stepless as before.
    _step = metrics.step
    _step_field = {"step": int(_step)} if isinstance(_step, (int, float)) and _step > 0 else {}
    heartbeat("done", **_step_field, **(heartbeat_fields or {}), gpu=gpu_diagnostics())
    print("NODE DONE:", metrics.to_json())


def main():
    try:
        from flash.engine.worker.entry.profile import run_sft_profile

        modes = {
            "sft": run_sft,
            "rl": run_rl,
            "opd": run_opd,
            "profile": run_sft_profile,
        }
        handler = modes.get(RUN_MODE)
        if handler is None:
            raise RuntimeError("worker run mode is invalid")
        remaining = _remaining_worker_wall_seconds()
        if remaining is not None and remaining <= 0:
            raise RuntimeError("worker run wall deadline exceeded")
        if RUN_MODE == "sft" and JOB_SPEC and JOB_SPEC.train.init_from_adapter:
            raise ValueError(
                "train.init_from_adapter is supported only for GRPO and OPD continue-in-place runs; "
                "SFT adapter continuation is not supported"
            )
        # Idempotency: check DONE before any env-mutating pip install (fla fast path).
        if HF_REPO:
            from huggingface_hub import hf_hub_download

            try:
                hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    filename=f"{hf_prefix()}/DONE",
                    token=os.environ.get("HF_TOKEN"),
                )
                done = True
            except Exception:
                done = False
            remaining = _remaining_worker_wall_seconds()
            if remaining is not None and remaining <= 0:
                raise RuntimeError("worker run wall deadline exceeded")
            if done:
                print("Run already complete (DONE present); returning persisted metrics.")
                heartbeat("already_done", gpu=gpu_diagnostics(include_torch=False))
                # DONE is written only AFTER metrics.json uploads (required=True), so a failed read here
                # is a transient HF blip, never a missing file. Retry, then signal RETRIABLE (reschedule)
                # rather than SystemExit — a BaseException that bypasses the retriable-stamping handler
                # below and would report a genuinely-succeeded run as a fatal failure.
                last_err: Exception | None = None
                for attempt in range(3):
                    remaining = _remaining_worker_wall_seconds()
                    if remaining is not None and remaining <= 0:
                        break
                    try:
                        got = hf_hub_download(
                            repo_id=HF_REPO,
                            repo_type="dataset",
                            filename=f"{hf_prefix()}/metrics.json",
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
                        remaining = _remaining_worker_wall_seconds()
                        if remaining is not None:
                            if remaining <= 0:
                                break
                            delay = min(delay, remaining)
                        if delay > 0:
                            time.sleep(delay)
                error_kind = type(last_err).__name__ if last_err is not None else "unknown error"
                raise RetriableInfraError(
                    "DONE present but metrics.json unreadable after retries "
                    f"(transient HF; {error_kind})"
                )
        # A profile run tokenizes or samples on cpu and never imports a model, so it exits BEFORE
        # the kernel setup below: none of it would apply, and _ensure_fla_fastpath_on_hopper would
        # pip-install into a process that is about to leave.
        if RUN_MODE == "profile":
            heartbeat("boot")
            handler()
            wandb_finish(exit_code=0)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        # these setups run in the parent; verl trains in FLASH_VERL_PYTHON.
        # only _force_fla_triton_gdn_on_sm100 propagates through FLA_* env vars. the install,
        # symlink,
        # and monkeypatch are interpreter-local and must not be treated as child configuration.
        # run before model imports: sm100 tilelang GDN computes wrong gradients, so use Triton.
        _force_fla_triton_gdn_on_sm100()
        _ensure_fla_fastpath_on_hopper()
        # Must run AFTER fla fast path (may reinstall tilelang) and BEFORE model/vLLM import.
        _neutralize_tilelang_cudart_stub()
        # AFTER the fla fast path (which may (re)install fla), BEFORE any model import / GDN
        # launch: restrict fla's Blackwell GDN bwd autotune to grad-correct configs (fla #913).
        _restrict_fla_gdn_autotune_on_blackwell()
        heartbeat("boot", gpu=gpu_diagnostics(include_torch=False))
        load_mega_cache()
        handler()
        # Hard-exit: colocated vLLM can deadlock on NCCL/CUDA teardown; all artifacts already on HF.
        wandb_finish(exit_code=0)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception as e:
        tb = sanitize_diagnostic(traceback.format_exc(), limit=16_000)
        if not flush_optional_uploads():
            print("optional upload flush timed out on worker error")
        try:
            err_name = error_artifact_name(RUN_MODE, ATTEMPT)
            err_path = f"/tmp/{err_name}"
            with open(err_path, "w") as f:
                f.write(tb)
            hf_upload_file(err_path, err_name)
        except Exception as up_err:
            print("error-upload warn:", sanitize_diagnostic(up_err, limit=500))
        try:
            # preserve ray session logs because raylet tracebacks show only the downstream EOF and
            # pod-local logs vanish after failure (VERL-115). all training modes start ray.
            ray_logs = collect_ray_failure_logs(started_after=WORKER_START_TIME)
            if ray_logs:
                ray_name = ray_log_artifact_name(RUN_MODE, ATTEMPT)
                ray_path = f"/tmp/{ray_name}"
                with open(ray_path, "w") as f:
                    f.write(ray_logs)
                hf_upload_file(ray_path, ray_name)
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
        # cannot raise -- so that if the primary heartbeat fails (most likely gpu_diagnostics()
        # or the upload itself), the fallback still carries metrics_last, which is the backlog
        # this path exists to surface for short failing RL runs.
        _err_metrics = (
            {"metrics_last": list(LATEST_GRPO_METRICS_LAST)} if LATEST_GRPO_METRICS_LAST else {}
        )
        try:
            heartbeat(
                f"error_{RUN_MODE}",
                error=detail,
                **hb_flags,
                **_err_metrics,
                diag=gpu_diagnostics(),
            )
        except Exception:
            heartbeat(f"error_{RUN_MODE}", error=detail, **hb_flags, **_err_metrics)
        wandb_finish(exit_code=1)
        remaining = _remaining_worker_wall_seconds()
        delay = 10.0 if remaining is None else min(10.0, remaining)
        if delay > 0:
            time.sleep(delay)
        raise


__all__ = [
    "ACTIVE_ENV",
    "ATTEMPT",
    "HF_REPO",
    "JOB_SPEC",
    "OPD_RESUME_REVISION",
    "PHASE",
    "RUN_ID",
    "RUN_MODE",
    "SEED",
    "THINKING",
    "_HB_FORCE_MIN_INTERVAL_S",
    "_HB_LAST_COMMITTED_STEP",
    "_HB_LAST_FORCED_UPLOAD",
    "_HB_LAST_PROGRESS_TS",
    "_HB_LAST_UPLOAD",
    "_HB_LOCK",
    "_HB_MIN_INTERVAL_S",
    "_HB_PROGRESS_SEQ",
    "_HB_PROGRESS_UPLOADED_SEQ",
    "_HB_SETUP_LIVENESS_INTERVAL_S",
    "_HB_SETUP_LIVENESS_STAGES",
    "_HB_TERMINAL_ONLY",
    "_HB_THROTTLED_STAGES",
    "_HB_TIGHT_LIVENESS_STAGES",
    "_HB_UPLOAD_LIVENESS_STAGES",
    "_HB_UPLOAD_LOCK",
    "_WANDB_FINISH_FAIL_WAIT_S",
    "_WANDB_FINISH_WAIT_S",
    "RetriableInfraError",
    "_attn_impl_for_capability",
    "_current_cuda_sm",
    "_download_adapter",
    "_ensure_fla_fastpath_on_hopper",
    "_estimate_params",
    "_finalize",
    "_flash_attn_3_available",
    "_flash_attn_available",
    "_force_fla_triton_gdn_on_sm100",
    "_hf_upload",
    "_init_adapter_model",
    "_latest_checkpoint_dir",
    "_liger_default_for_model",
    "_load_active_env",
    "_memory_mode",
    "_neutralize_tilelang_cudart_stub",
    "_remove_fla_from_disk",
    "_resolve_adapter_ref",
    "_restrict_fla_gdn_autotune_on_blackwell",
    "assert_adapter_delta_nonzero",
    "assert_adapter_load_clean",
    "assert_lora_applied",
    "backend_seed",
    "build_grpo_prompt_dataset",
    # gpu/backend setup
    "error_artifact_name",
    "flush_optional_uploads",
    "gpu_diagnostics",
    "grad_checkpointing_on",
    "graded_text",
    "grpo_mask_truncated_completions",
    "grpo_overrides",
    "grpo_use_reentrant",
    "heartbeat",
    "hf_api",
    "hf_prefix",
    "hf_resume_checkpoint",
    "hf_upload_file",
    "hf_upload_folder",
    "is_vl_checkpoint",
    "load_mega_cache",
    "load_tokenizer",
    "lora_target_parameters",
    "main",
    "make_checkpoint_upload_callback",
    "make_lora",
    "make_sft_heartbeat_callback",
    "model_revision_kwargs",
    "optimal_attn_impl",
    "prefetch_model",
    "prepare_fresh_lora_base",
    "prompt_opens_thinking",
    "publish_deployable_checkpoint",
    "publish_opd_optimizer_start_marker",
    "ray_log_artifact_name",
    "render_prompt",
    "require_active_env",
    "require_vllm_for_rollout_func",
    "resolve_grpo_prompts_per_step",
    "run_opd",
    "run_rl",
    "run_sft",
    "seed_training_rngs",
    "stamp_adapter_provenance",
    "strip_think",
    "think_token_count",
    "thinking_text",
    "upload_debug_jsonl",
    "upload_resume_checkpoint",
    "validate_lora_target_parameters",
    "wait_for_gpu",
    "wandb_finish",
    "wandb_report_to",
    "wandb_run_info",
    "wandb_run_name",
    "write_base_model_provenance",
    "write_train_meta",
]


if __name__ == "__main__":
    main()

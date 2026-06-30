"""On-GPU fine-tuning worker. Modes: sft | rl.

Namespace hub: owns run-scoped STATE globals, defines ``main`` + ``_finalize``, re-exports
every submodule name. Submodules read state via ``_w.<name>`` (live-package proxy) so
``monkeypatch.setattr(worker, "<name>", ...)`` is seen at call time, not captured at import.
"""

from __future__ import annotations

import os
import sys

# Unused here; ``worker.threading.Thread`` is patched by tests to run upload threads inline.
import threading  # noqa: F401
import time
import traceback

from flash.engine.accounting import RunMetrics
from flash.engine.worker.adapter import (
    _download_adapter,
    _init_adapter_model,
    _resolve_adapter_ref,
    make_lora,
    recombined_warmstart_adapter_dir,
    require_vllm_for_rollout_func,
)
from flash.engine.worker.decoding import (
    graded_text,
    prompt_opens_thinking,
    render_prompt,
    strip_think,
    think_token_count,
)
from flash.engine.worker.finalize import write_train_meta
from flash.engine.worker.gpu_setup import (
    finalize_alloc_conf_for_sleep,
    force_vllm_backend_for_sm120,
    patch_trl_colocate_llm_kwargs,
)
from flash.engine.worker.grpo import (
    _grpo_is_no_op_failure,
    _grpo_resume_already_complete,
    build_grpo_prompt_dataset,
    compute_grpo_batching,
    grpo_mask_truncated_completions,
    grpo_overrides,
    resolve_grpo_prompts_per_step,
    rl_per_device_comps,
)
from flash.engine.worker.heartbeat import (
    _HB_LOCK,
    _HB_TERMINAL_ONLY_INTERVAL_S,
    _HB_TERMINAL_STAGES,
    _HB_THROTTLED_STAGES,
    _HB_UPLOAD_LOCK,
    _SFT_HEARTBEAT_INTERVAL_S,
    _STEP_GPU_DIAG_INTERVAL_S,
    heartbeat,
    make_reward_heartbeat_callback,
    make_sft_heartbeat_callback,
)
from flash.engine.worker.hf import (
    _hf_upload,
    _latest_checkpoint_dir,
    error_artifact_name,
    hf_api,
    hf_prefix,
    hf_resume_checkpoint,
    hf_upload_file,
    hf_upload_folder,
    make_checkpoint_upload_callback,
    prefetch_model,
    publish_deployable_checkpoint,
    upload_debug_jsonl,
)
from flash.engine.worker.kernel_warmup import _current_cuda_sm, load_mega_cache
from flash.engine.worker.lora import (
    _LM_SYNC_REMAP_ON,
    _VL_EXCLUDE_SEGMENTS,
    _remap_vl_sync_weights,
    assert_adapter_delta_nonzero,
    assert_adapter_load_clean,
    assert_lora_applied,
    disable_liger_grpo_torch_compile,
    is_vl_checkpoint,
    lora_exclude_modules,
    patch_grpo_mask_aware_lm_head,
    patch_vllm_language_model_only,
    patch_vllm_lm_weight_sync,
    recombine_lora_adapters,
    strip_language_model_infix,
    vllm_language_model_only_kwargs,
)
from flash.engine.worker.perf import (
    RetriableInfraError,
    _attn_impl_for_capability,
    _ensure_fla_fastpath_on_hopper,
    _estimate_params,
    _flash_attn_3_available,
    _flash_attn_available,
    _GpuPeakSampler,
    _liger_default_for_model,
    _memory_mode,
    _metric_curve,
    _neutralize_tilelang_cudart_stub,
    _peak_gpu_gb,
    _remove_fla_from_disk,
    _reset_peak_gpu,
    _sdpa_cudnn_ctx,
    free_gpu,
    fused_optim_name,
    gpu_diagnostics,
    grad_checkpointing_on,
    grpo_sleep_mode,
    is_cuda_oom,
    liger_on,
    loraplus_optimizer_cls,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)
from flash.engine.worker.rl import run_rl
from flash.engine.worker.sft import run_sft
from flash.engine.worker.wandb_log import (
    wandb_finish,
    wandb_report_to,
    wandb_run_info,
    wandb_run_name,
)
from flash.envs.adapter import GitHubRateLimitError
from flash.envs.registry import load_environment
from flash.spec import FIXED_SEED, load_job_spec_from_env

HF_REPO = os.environ.get("HF_REPO", "")
RUN_ID = os.environ.get("RUN_ID", "local")
SEED = int(os.environ.get("SEED", str(FIXED_SEED)))
RUN_MODE = os.environ.get("RUN_MODE", "sft")
ATTEMPT = os.environ.get("ATTEMPT", "")
JOB_SPEC = load_job_spec_from_env()
PHASE = os.environ.get(
    "PHASE",
    JOB_SPEC.phase if JOB_SPEC else (RUN_MODE if RUN_MODE in ("sft", "rl") else "sft"),
)

_HB_LAST_UPLOAD = 0.0
_HB_LAST_PROGRESS_TS = 0.0
# Environment-scoped artifact repos are shared by many runs. Keep heartbeat commits below the HF
# per-repo commit cap while staying under the provider poller's 1200s training stall window.
_HB_MIN_INTERVAL_S = 900.0
_HB_TERMINAL_ONLY = False

_WANDB_FINISH_WAIT_S = 120.0
_WANDB_FINISH_FAIL_WAIT_S = 5.0


def _load_active_env():
    """Load the run's Freesolo environment from the JobSpec; require an explicit env."""
    if JOB_SPEC is None:
        return None
    env_id = JOB_SPEC.environment.id
    if not env_id:
        raise RuntimeError(
            "JobSpec sets no environment: provide [environment] id "
            "(a Freesolo environment id like 'your-name/your-env', returned by "
            "`flash env push --name <name>`)."
        )
    return load_environment(
        env_id, JOB_SPEC.environment.params, resolved_sha=JOB_SPEC.environment.resolved_sha
    )


ACTIVE_ENV = None

# Set by ``_init_adapter_model`` to the SFT adapter dir ONLY when it takes the VL merge-into-base
# warm-start path (#296): the SFT is merged into the training base and a FRESH GRPO LoRA is trained,
# so the saved adapter is SFT-less and MUST be recombined with this SFT before deploy. Stays None for
# fresh-LoRA and continued-adapter (non-VL) runs, whose saved adapter is already deployable as-is.
_VL_WARMSTART_SFT_DIR: str | None = None
# The selected catalog model for the same VL warm-start path. Finalize uses it to enforce the same
# model-specific serving rank cap that init-time preflight used, even if the SFT adapter config lacks
# base_model_name_or_path.
_VL_WARMSTART_MODEL_ID: str | None = None


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
            "`flash env push --name <name>`)."
        )
    return ACTIVE_ENV


def _worker_failure_flags(exc: BaseException) -> dict[str, bool]:
    retriable = isinstance(exc, (RetriableInfraError, GitHubRateLimitError))
    return {"retriable": retriable, "oom": (not retriable and is_cuda_oom(exc))}


THINKING = JOB_SPEC.thinking if JOB_SPEC else False


def _finalize(metrics: RunMetrics):
    metrics.save("/tmp/metrics.json")
    hf_upload_file("/tmp/metrics.json", "metrics.json", required=True)
    with open("/tmp/DONE", "w") as f:
        f.write(str(time.time()))
    hf_upload_file("/tmp/DONE", "DONE", required=True)
    heartbeat("done", gpu=gpu_diagnostics())
    print("NODE DONE:", metrics.to_json())


def main():
    try:
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
            if done:
                print("Run already complete (DONE present); returning persisted metrics.")
                heartbeat("already_done", gpu=gpu_diagnostics(include_torch=False))
                # DONE is written only AFTER metrics.json uploads (required=True), so a failed read here
                # is a transient HF blip, never a missing file. Retry, then signal RETRIABLE (reschedule)
                # rather than SystemExit — a BaseException that bypasses the retriable-stamping handler
                # below and would report a genuinely-succeeded run as a fatal failure.
                last_err: Exception | None = None
                for attempt in range(3):
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
                        time.sleep(5 * (attempt + 1))
                raise RetriableInfraError(
                    f"DONE present but metrics.json unreadable after retries (transient HF): {last_err}"
                )
        _ensure_fla_fastpath_on_hopper()
        # Must run AFTER fla fast path (may reinstall tilelang) and BEFORE model/vLLM import.
        _neutralize_tilelang_cudart_stub()
        heartbeat("boot", gpu=gpu_diagnostics(include_torch=False))
        finalize_alloc_conf_for_sleep()
        load_mega_cache()
        modes = {
            "sft": run_sft,
            "rl": run_rl,
        }
        handler = modes.get(RUN_MODE)
        if handler is None:
            raise SystemExit(f"unknown RUN_MODE {RUN_MODE}; known: {sorted(modes)}")
        handler()
        # Hard-exit: colocated vLLM can deadlock on NCCL/CUDA teardown; all artifacts already on HF.
        wandb_finish(exit_code=0)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception as e:
        tb = traceback.format_exc()
        traceback.print_exc()
        try:
            err_name = error_artifact_name(RUN_MODE, ATTEMPT)
            err_path = f"/tmp/{err_name}"
            with open(err_path, "w") as f:
                f.write(tb)
            hf_upload_file(err_path, err_name)
        except Exception as up_err:
            print("error-upload warn:", up_err)
        # A CUDA OOM -> stamp an ``oom`` flag so the runner retries on a LARGER GPU. Infra failures
        # keep same-size retry semantics and must never be reclassified as OOM.
        hb_flags = _worker_failure_flags(e)
        try:
            heartbeat(f"error_{RUN_MODE}", error=str(e)[:500], **hb_flags, diag=gpu_diagnostics())
        except Exception:
            heartbeat(f"error_{RUN_MODE}", error=str(e)[:500], **hb_flags)
        wandb_finish(exit_code=1)
        time.sleep(10)
        raise


__all__ = [
    "ACTIVE_ENV",
    "ATTEMPT",
    "HF_REPO",
    "JOB_SPEC",
    "PHASE",
    "RUN_ID",
    "RUN_MODE",
    "SEED",
    "THINKING",
    "_HB_LAST_PROGRESS_TS",
    "_HB_LAST_UPLOAD",
    "_HB_LOCK",
    "_HB_MIN_INTERVAL_S",
    "_HB_TERMINAL_ONLY",
    "_HB_TERMINAL_ONLY_INTERVAL_S",
    "_HB_TERMINAL_STAGES",
    "_HB_THROTTLED_STAGES",
    "_HB_UPLOAD_LOCK",
    "_LM_SYNC_REMAP_ON",
    "_SFT_HEARTBEAT_INTERVAL_S",
    "_STEP_GPU_DIAG_INTERVAL_S",
    "_VL_EXCLUDE_SEGMENTS",
    "_WANDB_FINISH_FAIL_WAIT_S",
    "_WANDB_FINISH_WAIT_S",
    "RetriableInfraError",
    "_GpuPeakSampler",
    "_attn_impl_for_capability",
    "_current_cuda_sm",
    "_download_adapter",
    "_ensure_fla_fastpath_on_hopper",
    "_estimate_params",
    "_finalize",
    "_flash_attn_3_available",
    "_flash_attn_available",
    "_grpo_is_no_op_failure",
    "_grpo_resume_already_complete",
    "_hf_upload",
    "_init_adapter_model",
    "_latest_checkpoint_dir",
    "_liger_default_for_model",
    "_load_active_env",
    "_memory_mode",
    "_metric_curve",
    "_neutralize_tilelang_cudart_stub",
    "_peak_gpu_gb",
    "_remap_vl_sync_weights",
    "_remove_fla_from_disk",
    "_reset_peak_gpu",
    "_resolve_adapter_ref",
    "_sdpa_cudnn_ctx",
    "assert_adapter_delta_nonzero",
    "assert_adapter_load_clean",
    "assert_lora_applied",
    "build_grpo_prompt_dataset",
    "compute_grpo_batching",
    # gpu/backend setup
    "disable_liger_grpo_torch_compile",
    "error_artifact_name",
    "finalize_alloc_conf_for_sleep",
    "force_vllm_backend_for_sm120",
    "free_gpu",
    "fused_optim_name",
    "gpu_diagnostics",
    "grad_checkpointing_on",
    "graded_text",
    "grpo_mask_truncated_completions",
    "grpo_overrides",
    "grpo_sleep_mode",
    "heartbeat",
    "hf_api",
    "hf_prefix",
    "hf_resume_checkpoint",
    "hf_upload_file",
    "hf_upload_folder",
    "is_vl_checkpoint",
    "liger_on",
    "load_mega_cache",
    "lora_exclude_modules",
    "loraplus_optimizer_cls",
    "main",
    "make_checkpoint_upload_callback",
    "make_lora",
    "make_reward_heartbeat_callback",
    "make_sft_heartbeat_callback",
    "optimal_attn_impl",
    "patch_grpo_mask_aware_lm_head",
    "patch_trl_colocate_llm_kwargs",
    "patch_vllm_language_model_only",
    "patch_vllm_lm_weight_sync",
    "prefetch_model",
    "prompt_opens_thinking",
    "publish_deployable_checkpoint",
    "recombine_lora_adapters",
    "recombined_warmstart_adapter_dir",
    "render_prompt",
    "require_active_env",
    "require_vllm_for_rollout_func",
    "resolve_grpo_prompts_per_step",
    "rl_per_device_comps",
    "run_rl",
    "run_sft",
    "setup_perf_backends",
    "strip_language_model_infix",
    "strip_think",
    "think_token_count",
    "upload_debug_jsonl",
    "vllm_language_model_only_kwargs",
    "wait_for_gpu",
    "wandb_finish",
    "wandb_report_to",
    "wandb_run_info",
    "wandb_run_name",
    "write_train_meta",
]


if __name__ == "__main__":
    main()

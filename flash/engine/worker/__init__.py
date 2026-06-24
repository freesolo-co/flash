"""On-GPU fine-tuning worker (RunPod). Modes: sft | rl.

This module runs on the provisioned RunPod GPU. It uses the shared recipe
(``flash.engine.recipe``) so SFT targets and RL rewards are rendered and scored
consistently.

Artifacts (adapter, metrics.json, heartbeat.json, checkpoints) are streamed to a
Hugging Face dataset repo. HF checkpoints give preemption resilience: if a worker is
recycled mid-run we resume from the latest uploaded checkpoint. Metrics are also
returned directly to the caller by the launching provider.

Core environment variables (set by the launching provider / runner):
  RUN_MODE      sft|rl
  SEED          int
  HF_REPO       Hugging Face dataset repo for artifacts, populated per-run from the
                JobSpec's [train] hf_repo by whichever provider launches the worker
  HF_TOKEN
  RUN_ID        unique id for this run (namespacing in the repo)

The FLASH_*/RL_*/SFT_* env vars are A/B overrides documented at their use sites; the
JobSpec [train] table is the source of truth for per-run knobs.

This package is split into cohesive submodules (``hf``/``heartbeat``/``decoding``/``wandb_log``/
``grpo``/``gpu_setup``/``adapter``/``sft``/``rl``/``finalize``, plus the leaf ``lora``/``perf``);
THIS module is the namespace hub. It (1) owns the run-scoped STATE module globals, (2) keeps
``import os`` / ``import time`` at module level (tests patch ``worker.os._exit`` /
``worker.time.sleep``), (3) defines ``main`` + ``_finalize``, and (4) re-exports every moved name.

CRITICAL monkeypatch contract: tests do ``monkeypatch.setattr(worker, "<name>", ...)`` for both
functions (``heartbeat``/``hf_api``/``run_sft``/``finalize_alloc_conf_for_sleep``/
``lora_exclude_modules``/``_ensure_fla_fastpath_on_hopper``/``hf_upload_file`` ...) and STATE
(``JOB_SPEC``/``HF_REPO``/``RUN_ID``/``RUN_MODE``/``PHASE``/``SEED``/``_HB_*`` ...). For a patch on
``worker.<name>`` to reach a caller, the caller must resolve ``<name>`` THROUGH this package at call
time, NOT via a bound local. The submodules therefore import the live-package proxy
(``from flash.engine.worker._pkg import W as _w``) and call ``_w.heartbeat(...)`` / read
``_w.JOB_SPEC`` etc.; the proxy always delegates to ``sys.modules['flash.engine.worker']`` so the
patch is seen even across the ``sys.modules.pop + re-import`` / ``importlib.reload`` that some tests
use. ``main``/``_finalize`` call the moved helpers via their bare (re-exported) names, which resolve
to this module's globals.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

from flash.engine.accounting import RunMetrics
from flash.engine.worker.adapter import (
    _download_adapter,
    _init_adapter_model,
    _resolve_adapter_ref,
    make_lora,
    require_vllm_for_rollout_func,
)
from flash.engine.worker.decoding import (
    graded_text,
    render_prompt,
    strip_think,
    think_token_count,
)
from flash.engine.worker.finalize import write_train_meta
from flash.engine.worker.gpu_setup import (
    finalize_alloc_conf_for_sleep,
    force_vllm_backend_for_sm120,
)
from flash.engine.worker.grpo import (
    _grpo_is_no_op_failure,
    _grpo_resume_already_complete,
    compute_grpo_batching,
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

# Re-export the pure leaf helpers (``.lora`` / ``.perf``). The retained readers (main/_finalize)
# and several submodules call some of these by their bare name, which resolves through THIS
# module's namespace — so a test's ``monkeypatch.setattr(worker, "<name>", ...)`` still reaches
# them. Every name is listed in ``__all__`` so the re-export is not flagged as unused.
from flash.engine.worker.lora import (
    _LM_SYNC_REMAP_ON,
    _VL_EXCLUDE_SEGMENTS,
    _remap_vl_sync_weights,
    assert_adapter_delta_nonzero,
    assert_adapter_load_clean,
    assert_lora_applied,
    is_vl_checkpoint,
    lora_exclude_modules,
    model_quant,
    patch_vllm_language_model_only,
    patch_vllm_lm_weight_sync,
    remap_adapter_keys,
    remap_vl_adapter_dir,
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
    liger_on,
    loraplus_optimizer_cls,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)
from flash.engine.worker.rl import run_rl
from flash.engine.worker.sft import run_sft
from flash.engine.worker.wandb_log import (
    wandb_report_to,
    wandb_run_info,
    wandb_run_name,
)
from flash.envs.registry import load_environment
from flash.spec import load_job_spec_from_env

# ------------------------------------------------------------------------------------------------
# Run-scoped STATE (module globals). Set once at import from the launch env / JobSpec; tests patch
# these via ``monkeypatch.setattr(worker, "<name>", ...)``. The submodules read them as
# ``_w.<name>`` so those patches take effect.
# ------------------------------------------------------------------------------------------------
HF_REPO = os.environ.get("HF_REPO", "")
RUN_ID = os.environ.get("RUN_ID", "local")
SEED = int(os.environ.get("SEED", "0"))
RUN_MODE = os.environ.get("RUN_MODE", "sft")
ATTEMPT = os.environ.get("ATTEMPT", "")
JOB_SPEC = load_job_spec_from_env()
# PHASE is the stable artifact namespace (sft|rl) and matches RUN_MODE for a train run.
PHASE = os.environ.get(
    "PHASE",
    JOB_SPEC.phase if JOB_SPEC else (RUN_MODE if RUN_MODE in ("sft", "rl") else "sft"),
)

# Heartbeat HF-commit throttle knobs (patched by tests; the heartbeat reader in ``heartbeat.py``
# reads these as ``_w.<name>``). Each heartbeat() commits heartbeat.json to the HF artifact repo;
# committing every training step blows HuggingFace's per-repo commit rate limit (128/hour), so the
# per-step "rl_step" stage is throttled to once per _HB_MIN_INTERVAL_S; terminal stages always
# commit. _HB_TERMINAL_ONLY (benchmark fan-out) throttles every non-terminal stage. The local file
# + stdout line are always written regardless.
_HB_LAST_UPLOAD = 0.0
# The rl_step heartbeat-upload throttle, in seconds (fixed 60s) — keeps GRPO under HF's
# 128 commits/hour-per-repo limit when concurrent runs share one HF_REPO.
_HB_MIN_INTERVAL_S = 60.0
_HB_TERMINAL_ONLY = False


def _load_active_env():
    """Load the run's Freesolo environment from the JobSpec; require an explicit env.

    There is no default/builtin environment: a run MUST name a published Freesolo
    environment id. Failing here prevents a paid worker from training/evaluating the
    wrong task.
    """
    if JOB_SPEC is None:
        # No JobSpec at all (e.g. the module imported for a non-run path / a unit test). There
        # is nothing to select; defer the hard requirement to the JobSpec-present branch so the
        # module stays importable. A real run always carries a JobSpec.
        return None
    env_id = JOB_SPEC.environment.id
    if not env_id:
        # Every supported algorithm (sft/grpo) trains/evaluates against a Freesolo env, so a
        # missing env is always a misconfigured spec. Fail loudly rather than fall back to a
        # default and burn a paid worker on the wrong task.
        raise RuntimeError(
            "JobSpec sets no environment: provide [environment] id "
            "(a Freesolo environment id like 'your-name/your-env', returned by "
            "`flash env push --name <name>`)."
        )
    return load_environment(env_id, JOB_SPEC.environment.params)


ACTIVE_ENV = None


def require_active_env():
    """Return the run's loaded environment, or raise a CLEAR error when there is none.

    ``ACTIVE_ENV`` is None on the no-JobSpec path (the module is imported with no
    FLASH_JOB_SPEC_JSON/PATH, e.g. a misconfigured worker launch). Every train/eval consumer
    needs a real env; without this guard the first ``ACTIVE_ENV.<attr>`` access dies with an
    opaque ``AttributeError: 'NoneType' object has no attribute ...``. Fail loudly with an
    actionable message instead — mirrors the explicit RuntimeError raised when a JobSpec is
    present but names no environment.
    """
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


# Thinking/reasoning mode: one flag per run from the run config (TOML `thinking`), consumed
# identically by SFT rendering, RL rollouts, and serving. Defaults off without a JobSpec.
THINKING = JOB_SPEC.thinking if JOB_SPEC else False


# Completion: train phase writes metrics.json + the DONE sentinel (see _finalize).
def _finalize(metrics: RunMetrics):
    metrics.save("/tmp/metrics.json")
    # Required: a swallowed upload would make the control plane fail/retry a finished run.
    hf_upload_file("/tmp/metrics.json", "metrics.json", required=True)
    # DONE sentinel so the controller knows it's safe to tear down
    with open("/tmp/DONE", "w") as f:
        f.write(str(time.time()))
    hf_upload_file("/tmp/DONE", "DONE", required=True)
    heartbeat("done", gpu=gpu_diagnostics())
    print("NODE DONE:", metrics.to_json())


def main():
    # Idempotency: if DONE was already uploaded, a re-delivered job re-fetches the final
    # metrics from HF and returns them immediately. (The previous behavior — sleeping in
    # an infinite loop — kept a billable GPU worker alive until the execution timeout.)
    try:
        # Idempotency FIRST — before any env-mutating pip install / package removal: a re-delivered
        # job whose DONE already exists must return the persisted metrics and exit WITHOUT running
        # _ensure_fla_fastpath_on_hopper() (mutates the env: pip-installs tilelang/fla) — that wasted
        # a worker mutating its env on an already-complete run. It runs after the DONE check below.
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
                    raise SystemExit(f"DONE present but metrics.json unavailable: {e}") from e
        # Not a DONE re-delivery -> this worker will train. These must run before any model import:
        _ensure_fla_fastpath_on_hopper()  # Hopper: enable fla+tilelang GDN fast path (see perf.py)
        # Repoint tilelang's libcudart_stub.so at the real CUDA runtime so it can't shadow libcudart
        # in vLLM's CudaRTLibrary (intermittent `undefined symbol: cudaDeviceReset` on GRPO vLLM
        # init, any model size/arch). AFTER the fla fast path (a tilelang reinstall there rewrites
        # the stub) and BEFORE the model/vLLM import. See perf.py / flash #184.
        _neutralize_tilelang_cudart_stub()
        heartbeat("boot", gpu=gpu_diagnostics(include_torch=False))
        finalize_alloc_conf_for_sleep()  # sync CUDA alloc conf to resolved sleep (before first CUDA alloc)
        # Dispatch table — register new algorithms (e.g. ppo) here as they land.
        modes = {
            "sft": run_sft,  # SFT (TRL SFTTrainer)
            "rl": run_rl,  # GRPO (TRL GRPOTrainer + colocated vLLM)
        }
        handler = modes.get(RUN_MODE)
        if handler is None:
            raise SystemExit(f"unknown RUN_MODE {RUN_MODE}; known: {sorted(modes)}")
        handler()
        # All artifacts (adapter, train_meta, metrics, DONE) are uploaded to HF *inside* the
        # handler. The RL trainer's colocated vLLM can DEADLOCK at interpreter shutdown
        # during NCCL/IPC/CUDA teardown — not segfault-and-exit (which `check=False` on the
        # train subprocess already tolerates), but hang forever. That would block the Flash
        # handler's *blocking* `subprocess.run` (heartbeat frozen at "rl_train_done") and the
        # whole run stalls until the wall-clock cap. Hard-exit to bypass the hanging teardown now that
        # every output is safely persisted.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception as e:
        # Structured retry signal both pollers read: an infra failure -> retry on a fresh worker.
        retriable = isinstance(e, RetriableInfraError)
        tb = traceback.format_exc()
        traceback.print_exc()
        try:
            err_name = error_artifact_name(RUN_MODE)
            err_path = f"/tmp/{err_name}"
            with open(err_path, "w") as f:
                f.write(tb)
            hf_upload_file(err_path, err_name)
        except Exception as up_err:
            print("error-upload warn:", up_err)
        hb_flags = {"retriable": retriable}
        try:
            heartbeat(f"error_{RUN_MODE}", error=str(e)[:500], **hb_flags, diag=gpu_diagnostics())
        except Exception:
            heartbeat(f"error_{RUN_MODE}", error=str(e)[:500], **hb_flags)
        # keep container alive briefly so logs flush, then exit non-zero -> restart
        time.sleep(10)
        raise


__all__ = [
    "ACTIVE_ENV",
    "ATTEMPT",
    # run-scoped STATE
    "HF_REPO",
    "JOB_SPEC",
    "PHASE",
    "RUN_ID",
    "RUN_MODE",
    "SEED",
    "THINKING",
    "_HB_LAST_UPLOAD",
    "_HB_LOCK",
    "_HB_MIN_INTERVAL_S",
    "_HB_TERMINAL_ONLY",
    "_HB_TERMINAL_ONLY_INTERVAL_S",
    "_HB_TERMINAL_STAGES",
    "_HB_THROTTLED_STAGES",
    "_HB_UPLOAD_LOCK",
    # leaf lora re-exports
    "_LM_SYNC_REMAP_ON",
    "_SFT_HEARTBEAT_INTERVAL_S",
    "_STEP_GPU_DIAG_INTERVAL_S",
    "_VL_EXCLUDE_SEGMENTS",
    # leaf perf re-exports
    "RetriableInfraError",
    "_GpuPeakSampler",
    "_attn_impl_for_capability",
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
    "_liger_default_for_model",
    # env + entry + finalize (defined here)
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
    # grpo batching / no-op guards
    "compute_grpo_batching",
    # hf artifact channel
    "error_artifact_name",
    "finalize_alloc_conf_for_sleep",
    # gpu/backend setup
    "force_vllm_backend_for_sm120",
    "free_gpu",
    "fused_optim_name",
    "gpu_diagnostics",
    "grad_checkpointing_on",
    "graded_text",
    "grpo_overrides",
    # heartbeat
    "heartbeat",
    "hf_api",
    "hf_prefix",
    "hf_resume_checkpoint",
    "hf_upload_file",
    "hf_upload_folder",
    "is_vl_checkpoint",
    "liger_on",
    "lora_exclude_modules",
    "loraplus_optimizer_cls",
    "main",
    "make_checkpoint_upload_callback",
    # lora / adapter
    "make_lora",
    "make_reward_heartbeat_callback",
    "make_sft_heartbeat_callback",
    "model_quant",
    "optimal_attn_impl",
    "patch_vllm_language_model_only",
    "patch_vllm_lm_weight_sync",
    "prefetch_model",
    "publish_deployable_checkpoint",
    "remap_adapter_keys",
    "remap_vl_adapter_dir",
    # decoding
    "render_prompt",
    "require_active_env",
    "require_vllm_for_rollout_func",
    "resolve_grpo_prompts_per_step",
    "rl_per_device_comps",
    "run_rl",
    # training entrypoints
    "run_sft",
    "setup_perf_backends",
    "strip_language_model_infix",
    "strip_think",
    "think_token_count",
    "upload_debug_jsonl",
    "vllm_language_model_only_kwargs",
    "wait_for_gpu",
    # wandb
    "wandb_report_to",
    "wandb_run_info",
    "wandb_run_name",
    # finalize / meta
    "write_train_meta",
]


if __name__ == "__main__":
    main()

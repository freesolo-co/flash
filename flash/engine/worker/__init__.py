"""On-GPU fine-tuning worker (RunPod or Vast.ai). Modes: sft | rl.

This module runs on the provisioned GPU (RunPod or Vast.ai) launched by the selected
``flash.providers`` backend. It uses the shared recipe (``flash.engine.recipe``) so
SFT targets and RL rewards are rendered and scored consistently.

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
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import sys
import threading
import time
import traceback

from flash.engine.accounting import RunMetrics

# Shared, substrate-neutral fine-tuning internals (live in this same package).
from flash.engine.chalk_kernels import (
    active_kernels,
    fused_ce_available,
    install_chalk_kernels,
)
from flash.engine.recipe import RECIPE

# Re-export the pure helpers split into the leaf submodules ``.perf`` and ``.lora``.
# CRITICAL: the readers below (run_sft / run_rl / make_lora / _init_adapter_model / ...) call
# these by their bare name, which resolves through THIS module's namespace — so a test's
# ``monkeypatch.setattr(worker, "<name>", ...)`` still reaches the readers. Names actually used
# by the retained readers are imported plainly; names re-exported only for API / test access
# (no retained reader uses them) are marked unused for the linter.
from flash.engine.worker.lora import (
    _LM_SYNC_REMAP_ON,
    _VL_EXCLUDE_SEGMENTS,  # noqa: F401
    _remap_vl_sync_weights,  # noqa: F401
    assert_lora_applied,
    is_vl_checkpoint,
    lora_exclude_modules,
    patch_vllm_language_model_only,
    patch_vllm_lm_weight_sync,
    remap_adapter_keys,  # noqa: F401
    remap_vl_adapter_dir,
    strip_language_model_infix,  # noqa: F401
    vllm_language_model_only_kwargs,  # noqa: F401
)
from flash.engine.worker.perf import (
    RetriableInfraError,
    _attn_impl_for_capability,  # noqa: F401
    _ensure_fla_fastpath_on_hopper,
    _estimate_params,  # noqa: F401
    _flash_attn_available,
    _GpuPeakSampler,
    _liger_default_for_model,  # noqa: F401
    _memory_mode,
    _metric_curve,
    _peak_gpu_gb,
    _remove_fla_from_disk,  # noqa: F401
    _reset_peak_gpu,
    _sdpa_cudnn_ctx,
    assert_usable_gpu,
    flex_attn_status,
    free_gpu,
    fused_optim_name,
    gpu_diagnostics,
    grad_checkpointing_on,
    loraplus_optimizer_cls,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)
from flash.envs.registry import load_environment
from flash.spec import load_job_spec_from_env

# Disable PyTorch's NVML-based CUDA availability check BEFORE any torch/CUDA import. On a
# MIG slice / permission-restricted host (RunPod has fulfilled full-GPU requests with a MIG
# partition whose nvidia-smi reads "[Insufficient Permissions]"), the NVML-based check makes
# the caching allocator HARD-ASSERT mid-backward ("NVML_SUCCESS == r INTERNAL ASSERT FAILED
# ... CUDACachingAllocator"). Falling back to the non-NVML check lets the dedicated boot guard
# (assert_usable_gpu) detect the bad host and raise a RETRIABLE infra error instead of crashing
# opaquely deep in training. setdefault so an explicit operator/launcher value still wins. Read
# lazily by torch at first CUDA use, so setting it at import (before any torch import) is in time.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")

HF_REPO = os.environ.get("HF_REPO", "")
RUN_ID = os.environ.get("RUN_ID", "local")
SEED = int(os.environ.get("SEED", "0"))
RUN_MODE = os.environ.get("RUN_MODE", "sft")
JOB_SPEC = load_job_spec_from_env()
# PHASE is the stable artifact namespace (sft|rl) and matches RUN_MODE for a train run.
PHASE = os.environ.get(
    "PHASE",
    JOB_SPEC.phase if JOB_SPEC else (RUN_MODE if RUN_MODE in ("sft", "rl") else "sft"),
)


def _load_active_env():
    """Load the run's verifiers environment from the JobSpec; require an explicit env.

    There is no default/builtin environment (verifiers-only): a run MUST name a verifiers/
    Prime Hub env id. Failing here (instead of falling back to some default) prevents a paid
    worker from training/evaluating the wrong task.
    """
    if JOB_SPEC is None:
        # No JobSpec at all (e.g. the module imported for a non-run path / a unit test). There
        # is nothing to select; defer the hard requirement to the JobSpec-present branch so the
        # module stays importable. A real run always carries a JobSpec.
        return None
    env_id = JOB_SPEC.environment.id
    if not env_id:
        # Every supported algorithm (sft/grpo) trains/evaluates against a verifiers env, so a
        # missing env is always a misconfigured spec. Fail loudly rather than fall back to a
        # default and burn a paid worker on the wrong task.
        raise RuntimeError(
            "JobSpec sets no environment: provide [environment] id (a verifiers/Prime Hub "
            "slug, e.g. 'owner/name')."
        )
    return load_environment(env_id, JOB_SPEC.environment.params)


ACTIVE_ENV = _load_active_env()


def require_active_env():
    """Return the run's loaded environment, or raise a CLEAR error when there is none.

    ``ACTIVE_ENV`` is None on the no-JobSpec path (the module is imported with no
    FLASH_JOB_SPEC_JSON/PATH, e.g. a misconfigured worker launch). Every train/eval consumer
    needs a real env; without this guard the first ``ACTIVE_ENV.<attr>`` access dies with an
    opaque ``AttributeError: 'NoneType' object has no attribute ...``. Fail loudly with an
    actionable message instead — mirrors the explicit RuntimeError raised when a JobSpec is
    present but names no environment.
    """
    if ACTIVE_ENV is None:
        raise RuntimeError(
            "no environment is loaded: this worker was started without a JobSpec "
            "(FLASH_JOB_SPEC_JSON / FLASH_JOB_SPEC_PATH is unset). A train/eval run must "
            "carry a JobSpec naming [environment] id (a verifiers/Prime Hub slug, e.g. "
            "'owner/name')."
        )
    return ACTIVE_ENV


# Thinking/reasoning mode: one flag per run from the run config (TOML `thinking`), consumed
# identically by SFT rendering, RL rollouts, and serving. Defaults off without a JobSpec.
THINKING = JOB_SPEC.thinking if JOB_SPEC else False


# ---------------------------------------------------------------------------
# HF helpers (code-delivery + artifact channel; works without inbound network)
# ---------------------------------------------------------------------------
def error_artifact_name(mode: str) -> str:
    """Per-mode error filename (e.g. error_sft.txt) so a run's traceback is uploaded
    under a stable name even though heartbeat.json is single-file/overwritten."""
    return f"error_{mode}.txt"


def hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def hf_prefix() -> str:
    return f"{PHASE}/{RUN_ID}/seed{SEED}"


def _hf_upload(do_upload, repo_subpath: str, required: bool, label: str) -> None:
    """Shared HF upload loop for files/folders: HF_REPO guard + retry/raise-or-warn.

    ``required=True`` (completion artifacts DONE/metrics.json, the trained adapter) retries
    and finally raises: a swallowed upload failure would make the control plane mark a
    finished run failed/retried, or mark the run done while deployment can never download
    the missing adapter. Optional artifacts (generations, logs) only warn.
    """
    if not HF_REPO:
        return
    attempts = 3 if required else 1
    for attempt in range(attempts):
        try:
            do_upload()
            return
        except Exception as e:
            if required and attempt + 1 < attempts:
                print(f"{label} retry {attempt + 1}/{attempts}: {e}")
                time.sleep(5 * (attempt + 1))
                continue
            if required:
                # Already retried 3x -> the host/network is bad, not the run. Infra-shaped.
                raise RetriableInfraError(f"required upload of {repo_subpath!r} failed: {e}") from e
            print(f"{label} warn:", e)
            return


def hf_upload_file(local_path: str, repo_subpath: str, required: bool = False):
    """Upload one file to the run's HF prefix."""
    _hf_upload(
        lambda: hf_api().upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{hf_prefix()}/{repo_subpath}",
            repo_id=HF_REPO,
            repo_type="dataset",
        ),
        repo_subpath,
        required,
        "hf_upload_file",
    )


def hf_upload_folder(local_dir: str, repo_subpath: str, required: bool = False):
    """Upload a folder to the run's HF prefix."""
    _hf_upload(
        lambda: hf_api().upload_folder(
            folder_path=local_dir,
            path_in_repo=f"{hf_prefix()}/{repo_subpath}",
            repo_id=HF_REPO,
            repo_type="dataset",
        ),
        repo_subpath,
        required,
        "hf_upload_folder",
    )


def hf_resume_checkpoint() -> str | None:
    """Latest streamed trainer checkpoint for this run (or None).

    Checkpoints are uploaded DURING the run by ``make_checkpoint_upload_callback`` as
    ``<prefix>/checkpoint/checkpoint-<step>/``; a replacement worker downloads the
    newest one so a mid-run preemption costs at most one save interval.
    """
    if not HF_REPO:
        return None
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            allow_patterns=[f"{hf_prefix()}/checkpoint/**"],
            local_dir="/tmp/resume",
            token=os.environ.get("HF_TOKEN"),
        )
        base = os.path.join("/tmp/resume", hf_prefix(), "checkpoint")
        if not os.path.isdir(base):
            return None
        cands = [d for d in os.listdir(base) if d.startswith("checkpoint-")]
        if not cands:
            return None
        latest = max(cands, key=lambda d: int(d.split("-")[-1]))
        path = os.path.join(base, latest)
        print(f"[resume] found streamed checkpoint: {path}")
        return path
    except Exception as e:
        print("hf_resume_checkpoint warn:", e)
        return None


def prefetch_model(model_id: str) -> float:
    """Pull the model weights into the local HF cache up front; return seconds spent.

    The trainer/vLLM would download lazily anyway — doing it explicitly (a) makes the
    download a first-class, timed stage in the heartbeat stream (the cold-start metric
    the speed work optimizes), and (b) fails fast with a clear disk/network error
    instead of dying inside trainer construction. Idempotent: a warm cache costs ~0 s.
    """
    from huggingface_hub import snapshot_download

    t0 = time.time()
    try:
        snapshot_download(
            repo_id=model_id,
            # weights + tokenizer/config only (same exclusions as the image bake)
            ignore_patterns=["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"],
        )
    except Exception as e:
        # Surface but don't fail here: gated/local-only models still load fine through
        # the normal from_pretrained path the trainer uses next.
        print("prefetch_model warn:", e)
    secs = round(time.time() - t0, 1)
    heartbeat(
        "model_prefetched",
        model=model_id,
        download_seconds=secs,
        hf_transfer=os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
    )
    return secs


def make_checkpoint_upload_callback():
    """Stream each trainer save to HF so preemption loses <= one save interval.

    Uploads run in a background thread (the train loop never blocks on the network);
    older checkpoints are deleted in the same commit. If an upload is still in flight
    when the next save fires, the new save is skipped (the following one catches up).
    """
    from transformers import TrainerCallback

    lock = threading.Lock()

    class _CheckpointUpload(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if not HF_REPO:
                return
            step = int(state.global_step)
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            if not os.path.isdir(ckpt_dir):
                return
            if not lock.acquire(blocking=False):
                print(f"[ckpt] upload busy; skipping step {step}")
                return

            def _upload():
                try:
                    hf_api().upload_folder(
                        folder_path=ckpt_dir,
                        path_in_repo=f"{hf_prefix()}/checkpoint/checkpoint-{step}",
                        repo_id=HF_REPO,
                        repo_type="dataset",
                        delete_patterns=[f"{hf_prefix()}/checkpoint/**"],
                    )
                    heartbeat("checkpoint_uploaded", step=step)
                except Exception as e:
                    print("ckpt upload warn:", e)
                finally:
                    lock.release()

            threading.Thread(target=_upload, daemon=True).start()

    return _CheckpointUpload()


# Heartbeat HF-commit throttle. Each heartbeat() commits heartbeat.json to the HF artifact
# repo; committing every training step (the reward callback fires per step) blows HuggingFace's
# per-repo commit rate limit (128/hour), especially when several runs share one HF_REPO. Only
# the per-step "rl_step" stage is high-frequency, so throttle JUST that one to once per
# 60s; every other stage — including milestones and the terminal done/already_done — always
# commits so the control plane never misses a transition.
# The local file + stdout line are always written regardless.
_HB_LAST_UPLOAD = 0.0


# The rl_step heartbeat-upload throttle, in seconds (fixed 60s) — keeps GRPO under HF's
# 128 commits/hour-per-repo limit when concurrent runs share one HF_REPO.
_HB_MIN_INTERVAL_S = 60.0
_HB_THROTTLED_STAGES = frozenset({"rl_step"})
# Terminal transitions the control plane must never miss — always committed.
_HB_TERMINAL_STAGES = frozenset({"done", "already_done"})
_HB_TERMINAL_ONLY = False
# Even in terminal-only mode, emit a SLOW heartbeat at this cadence so the control plane's stall
# detector (poll_vast_job stall_after_s, default 1500s) keeps seeing progress through a long
# training phase and doesn't false-stall the run. 600s -> ~6 commits/hr, far under the 128/hr cap.
_HB_TERMINAL_ONLY_INTERVAL_S = 600.0


# Serializes heartbeat.json writes and _HB_LAST_UPLOAD reads/updates. During GRPO,
# heartbeat() is called concurrently from the trainer thread (reward callback) and the
# checkpoint-upload daemon thread; without this lock two writers can interleave and
# truncate/garble heartbeat.json (and race _HB_LAST_UPLOAD).
_HB_LOCK = threading.Lock()
# Serializes the actual HF upload (a slow network commit) SEPARATELY from _HB_LOCK so the
# trainer's frequent local writes never block on the network. Without it, two heartbeat
# threads can upload heartbeat.json concurrently: a slower upload could land AFTER a newer
# one on HF (reorder), so this lock makes uploads strictly ordered.
_HB_UPLOAD_LOCK = threading.Lock()


def heartbeat(stage: str, **kw):
    global _HB_LAST_UPLOAD
    payload = {
        "stage": stage,
        "ts": time.time(),
        "run_id": RUN_ID,
        "mode": RUN_MODE,
        "seed": SEED,
        **kw,
    }
    os.makedirs("/tmp/hb", exist_ok=True)
    p = "/tmp/hb/heartbeat.json"
    # _HB_LOCK guards ONLY the fast local work (atomic write + _HB_LAST_UPLOAD + snapshot capture);
    # the slow HF commit runs OUTSIDE it so the trainer's per-step reward callback never blocks on
    # the network behind the checkpoint daemon's commit (a GRPO perf regression).
    with _HB_LOCK:
        # Atomic write: temp file + os.replace() so a concurrent reader never sees a partial file.
        tmp = p + f".{os.getpid()}.{threading.get_ident()}.tmp"
        snapshot = json.dumps(payload)
        with open(tmp, "w") as f:
            f.write(snapshot)
        os.replace(tmp, p)
        now = time.time()
        if stage in _HB_TERMINAL_STAGES or stage.startswith("error_"):
            upload_due = True  # never miss a terminal transition
        elif _HB_TERMINAL_ONLY:
            # Benchmark fan-out: keep commits far under the 128/hour cap, but still emit a SLOW
            # heartbeat (~every _HB_TERMINAL_ONLY_INTERVAL_S) so the control-plane stall detector
            # sees progress during a long training phase and doesn't false-stall the run.
            upload_due = (
                _HB_LAST_UPLOAD == 0.0 or (now - _HB_LAST_UPLOAD) >= _HB_TERMINAL_ONLY_INTERVAL_S
            )
        else:
            throttled = stage in _HB_THROTTLED_STAGES
            upload_due = not throttled or (now - _HB_LAST_UPLOAD) >= _HB_MIN_INTERVAL_S
        if upload_due:
            _HB_LAST_UPLOAD = now  # claim the slot under the lock (throttle stays atomic)
    if upload_due:
        # Serialize the network commit under a SEPARATE lock so uploads can't reorder, and
        # upload the captured snapshot (via a private temp file, since hf_upload_file takes
        # a path) rather than re-reading p — which a newer heartbeat may already have
        # overwritten between our slot-claim and this upload.
        with _HB_UPLOAD_LOCK:
            up = p + f".{os.getpid()}.{threading.get_ident()}.upload.tmp"
            with open(up, "w") as f:
                f.write(snapshot)
            try:
                hf_upload_file(up, "heartbeat.json")
            finally:
                with contextlib.suppress(OSError):
                    os.remove(up)
    print("HEARTBEAT", json.dumps(payload))


# ---------------------------------------------------------------------------
# Decoding parity: render with the model's own chat template and one run-wide thinking
# flag (off by default), so SFT targets and RL rollouts use identical prompt
# formatting within a run.
# ---------------------------------------------------------------------------
def render_prompt(tokenizer, item) -> str:
    item = item if isinstance(item, dict) else {"question": item}
    msgs = require_active_env().prompt_messages(item)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=THINKING
    )


def strip_think(completion: str | None) -> str | None:
    """Drop <think>...</think> reasoning before the environment grades/rewards a
    thinking-mode completion.

    - closed block(s): keep only the text after the LAST </think>. This also covers
      always-thinking templates that pre-open <think> inside the generation prompt,
      whose completions contain </think> with no opening tag.
    - unclosed <think> (completion budget exhausted): keep only the pre-think text
      (usually empty), so answer extraction fails and the completion scores 0 —
      deliberate reward pressure to close thinking within budget, and it keeps a
      last-number fallback from matching numbers inside the reasoning.
    - no tags: unchanged.
    """
    if completion is None:
        return None
    if "</think>" in completion:
        return completion.rsplit("</think>", 1)[1]
    if "<think>" in completion:
        return completion.split("<think>", 1)[0]
    return completion


def graded_text(completion: str | None) -> str | None:
    """What the env grader/reward sees: thinking runs strip <think> blocks first (a
    completion whose reasoning never closes grades 0 — see strip_think). Applied once
    here, before ACTIVE_ENV.grade/reward, so it works for every environment."""
    return strip_think(completion) if THINKING else completion




# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------



def force_vllm_backend_for_sm120() -> str | None:
    """On RTX 5090 / consumer Blackwell (sm120), force a PTX-independent vLLM attention backend.

    vLLM's default rollout backend is flash-attn, whose PRE-BUILT PTX needs a newer driver JIT than
    many 5090 RunPod hosts have — when the JIT fails the colocated rollout silently produces NO
    completions (empty reward_history, ~1.4 s "done"; a whole 22-run sweep hit this on every 5090).
    FLASHINFER is vLLM's Blackwell-native backend (no flash-attn PTX dependency) and trains on a 5090
    (measured: FLASHINFER/TORCH_SDPA/TRITON_ATTN all train, ~116 s). This mirrors the trainer's
    cuDNN-SDPA forcing on sm120 (``_attn_impl_for_capability``). The GRPO no-op guard remains the
    backstop. Returns the backend set (None if not sm120, or the operator already pinned one)."""
    if os.environ.get("VLLM_ATTENTION_BACKEND"):
        return None  # operator override wins
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 12:
            return None
    except Exception as e:
        print("[rl] sm120 vLLM backend probe skipped:", e)
        return None
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    print("[rl] sm120 (RTX 5090): VLLM_ATTENTION_BACKEND=FLASHINFER (flash-attn PTX is unreliable "
          "on consumer Blackwell hosts -> empty-rollout failures)")
    return "FLASHINFER"



def finalize_alloc_conf_for_sleep() -> None:
    """Sync the CUDA allocator conf with the worker's RESOLVED vLLM sleep default.

    The launcher (providers/*/train.py build_worker_env) must pick PYTORCH_ALLOC_CONF before this
    process starts, but it can't always know the GRPO sleep decision: for a small model the worker
    resolves sleep OFF (the speed default), yet the launcher conservatively assumes sleep ON and
    picks the non-expandable conf (safe, but fragments a long colocate run). When the launcher cedes
    the decision (it sets FLASH_ALLOC_AUTO=1 for RL runs), we resolve the same sleep default here (we
    have the model config + GPU) and, if sleep is OFF, switch to expandable_segments — which only
    crashes WITH sleep on, a case we've just ruled out. PYTORCH_ALLOC_CONF is read lazily at the
    first CUDA allocation, so this must run before any allocation (it does — called at boot)."""
    if os.environ.get("FLASH_ALLOC_AUTO") != "1":
        return
    try:
        model_id = JOB_SPEC.model if JOB_SPEC else ""
        # Resolve the GRPO context the SAME way the sleep gate does (run_rl): the run's
        # [train].max_length, so a long-context run gets the right sleep default + alloc conf.
        _spec_len = 0
        try:
            if JOB_SPEC and JOB_SPEC.train and JOB_SPEC.train.max_length:
                _spec_len = int(JOB_SPEC.train.max_length)
        except Exception:
            _spec_len = 0
        ctx = int(_spec_len or 0)
        if not _memory_mode(model_id, ctx):  # sleep resolves OFF -> expandable is safe + better
            conf = "expandable_segments:True"
            os.environ["PYTORCH_ALLOC_CONF"] = conf
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
            print(f"[alloc] sleep resolves OFF -> {conf} (anti-fragmentation, matches worker gate)")
        else:
            print("[alloc] sleep resolves ON -> keeping launcher's non-expandable conf")
    except Exception as e:
        print("[alloc] auto-conf skipped:", e)






def wandb_report_to() -> list[str]:
    """TRL/HF ``report_to`` targets. Restores the W&B logging the legacy freesolo training path had
    but the flash migration dropped: report to W&B whenever WANDB_API_KEY is present. No key -> []
    (silent, the metrics.json artifact is still the source of truth).

    Project + run name come ONLY from the typed ``[wandb]`` config (``JOB_SPEC.wandb``) — there is
    NO WANDB_PROJECT / WANDB_NAME environment variable. HF's WandbCallback has no project argument
    and would read WANDB_PROJECT from the env, so we initialize the run directly via the wandb SDK
    here (``wandb.init(project=..., name=...)``); the Trainer's callback then reuses that run. The
    run's entity is the API key's default account/team (we don't pass ``entity=``), so the only
    W&B env var is the WANDB_API_KEY credential."""
    if not os.environ.get("WANDB_API_KEY"):
        return []
    import importlib.util

    if importlib.util.find_spec("wandb") is None:
        print("[wandb] WANDB_API_KEY set but the wandb package is missing; skipping W&B logging")
        return []
    # Best-effort, like the bitsandbytes import above: a partial/broken wandb install or an
    # init failure (auth, network, runtime import error) must NOT abort training — W&B logging is
    # optional and metrics.json is the source of truth. Any failure -> no W&B logging ([]).
    try:
        import wandb

        if wandb.run is None:  # init from the spec so the project needs no WANDB_PROJECT env
            project = (JOB_SPEC.wandb.project if JOB_SPEC else None) or "flash"
            wandb.init(project=project, name=wandb_run_name())
    except Exception as e:
        print(f"[wandb] W&B init failed ({e}); skipping W&B logging (metrics.json is still written)")
        return []
    return ["wandb"]


def wandb_run_name() -> str:
    """W&B run name, from the typed ``[wandb] run_name`` config (``JOB_SPEC.wandb.run_name``) only —
    no WANDB_NAME environment variable. An explicit name is used verbatim (the user owns the
    naming); otherwise a stable id tying the dashboard run to the Flash run
    (``flash-<phase>-<run_id>-seed<N>``). Passed to the Trainer via ``TrainingArguments.run_name``
    and to ``wandb.init`` above."""
    configured = JOB_SPEC.wandb.run_name if JOB_SPEC else None
    if configured and configured.strip():
        return configured.strip()
    return f"flash-{PHASE}-{RUN_ID}-seed{SEED}"


def wandb_run_info() -> dict:
    """The live W&B run's {url, id, project} if W&B is active, else {}. Recorded in metrics.json so
    the W&B run is verifiable + the freesolo agent's `wandb_runs` / the SDK's link_wandb can point at
    the real dashboard URL — the link the flash migration otherwise dropped. Never raises."""
    try:
        import wandb

        run = getattr(wandb, "run", None)
        if run is None:
            return {}
        return {
            "wandb_url": getattr(run, "url", None),
            "wandb_id": getattr(run, "id", None),
            "wandb_project": getattr(run, "project", None),
        }
    except Exception:
        return {}





def make_lora(model_id: str | None = None):
    """LoRA config. We target 'all-linear' (every nn.Linear) rather than a hardcoded
    q/k/v/o list: it is architecture-agnostic, so the same recipe works for the dense
    default (Qwen3-4B-Instruct-2507) and for newer models with extra projection
    types (e.g. the Qwen3.5 hybrid Gated-DeltaNet) without missing any adapters.
    For natively-multimodal checkpoints the vision tower is excluded (see
    ``lora_exclude_modules``)."""
    from peft import LoraConfig

    # Adapt every linear projection. "all-linear" is a PEFT SPECIAL string (not a module name)
    # that PEFT expands to all linear layers — the right managed default across the catalog.
    targets = "all-linear"
    rank = JOB_SPEC.train.lora_rank if JOB_SPEC else RECIPE.lora.rank
    alpha = JOB_SPEC.train.lora_alpha if JOB_SPEC else RECIPE.lora.alpha
    kwargs = {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": RECIPE.lora.dropout,
        "target_modules": targets,
        "task_type": "CAUSAL_LM",
    }
    # Adapter initialization: standard zero-B init (the LoRA delta starts at zero, so the saved
    # adapter is a plain residual that loads correctly onto the ORIGINAL base).
    # PiSSA was removed: it mutates the effective base during training, so its saved adapter only
    # reconstructs against the PiSSA-residual base. Loading that adapter onto the unmodified base
    # at SERVING or GRPO WARM-START (which is exactly our flow) corrupts the model -> the served
    # model emits only whitespace and warm-start GRPO hangs. peft can convert PiSSA->standard on
    # save, but the simpler, robust choice is the default init (the convergence gain isn't worth
    # silently breaking serve + warm-start).
    kwargs["init_lora_weights"] = True
    print("[lora] init_lora_weights=True (standard zero-B; PiSSA removed for serve/warm-start safety)")
    # rsLoRA scaling (convergence lever, always-on: measured -47% train loss in A/B (gpu-bench)).
    kwargs["use_rslora"] = True
    if model_id and targets == "all-linear":
        exclude = lora_exclude_modules(model_id)
        if exclude:
            kwargs["exclude_modules"] = exclude
            print(f"[lora] excluding modules for {model_id}: {exclude}")
    return LoraConfig(**kwargs)




def require_vllm_for_rollout_func(use_rollout_func: bool, use_vllm: bool, model_id: str) -> None:
    """Fail fast when a multi-turn GRPO run needs colocated vLLM but it's disabled.

    The multi-turn rollout closure (``multiturn_rollout.build_rollout_func``) drives generation
    through ``trainer.vllm_generation.llm``. TRL only creates that engine when ``use_vllm`` is
    True, so with vLLM disabled the rollout would AttributeError at the first turn. GRPO now always
    colocates vLLM (``use_vllm`` is unconditionally True), so this guard is defensive — keep it to
    fail fast with an actionable message should a future tier disable the rollout engine.
    """
    if use_rollout_func and not use_vllm:
        raise RuntimeError(
            f"multi-turn GRPO needs colocated vLLM, which is disabled for {model_id}. "
            "Use a single-turn environment for this model, or a model tier that keeps "
            "vLLM enabled for rollouts."
        )


def run_sft():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    require_active_env()  # fail loudly (not AttributeError: NoneType) on the no-JobSpec path
    t_start = time.time()
    heartbeat("sft_start")
    # SFT only fits the single assistant `sft_target` per row; a multi-turn/ToolEnv env's
    # tool/env turns are not represented, so SFT on one would silently mis-train (imitating a
    # collapsed single-turn target). Warn loudly so it is not mistaken for proper multi-turn SFT.
    if getattr(ACTIVE_ENV, "multi_turn", False):
        print(
            "[sft][warn] this is a multi-turn / tool verifiers environment, but SFT only fits "
            "the single assistant target per row (tool/env turns are ignored). The model will be "
            "trained on collapsed single-turn targets; multi-turn SFT is not supported. Use a "
            "single-turn environment, or expect a single-turn-only fit."
        )
    wait_for_gpu()
    setup_perf_backends()
    model_id = JOB_SPEC.model if JOB_SPEC else RECIPE.hf_model_id
    download_seconds = prefetch_model(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build SFT text dataset (seeded shuffle for reproducibility)
    train = ACTIVE_ENV.dataset()
    rng = random.Random(SEED)
    rng.shuffle(train)
    max_examples = int(
        JOB_SPEC.train.max_examples or 0
        if JOB_SPEC and JOB_SPEC.train and JOB_SPEC.train.max_examples is not None
        else 0
    )
    if max_examples > 0:
        train = train[:max_examples]
    texts = []
    for ex in train:
        msgs = [
            *ACTIVE_ENV.prompt_messages(ex),
            {"role": "assistant", "content": ACTIVE_ENV.sft_target(ex)},
        ]
        texts.append(
            {
                "text": tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False, enable_thinking=THINKING
                )
            }
        )
    if THINKING and not any("<think>" in t["text"] for t in texts[:256]):
        print(
            "WARN: thinking mode is ON but no sampled SFT target contains a <think> "
            "trace — training on non-reasoning targets teaches the model to SKIP "
            "thinking. Use a dataset with reasoning traces, or set thinking = false."
        )
    ds = Dataset.from_list(texts)

    setup_seconds = time.time() - t_start
    heartbeat("sft_model_load", setup_seconds=setup_seconds)

    # Epochs come from the run's [train] epochs (already in JOB_SPEC), else the recipe default.
    epochs = int(
        JOB_SPEC.train.epochs
        if JOB_SPEC and JOB_SPEC.train.epochs is not None
        else RECIPE.sft.num_epochs
    )
    # SDK [train] knobs override the recipe default.
    from flash.catalog import vocab_size_for
    from flash.engine.vram import resolve_params_b, sft_grad_accum, sft_logits_fused

    _t = JOB_SPEC.train if JOB_SPEC else None
    sft_lr = _t.learning_rate if _t and _t.learning_rate is not None else RECIPE.sft.learning_rate
    sft_max_len = (
        _t.max_length
        if _t and _t.max_length is not None
        else (RECIPE.sft.max_seq_len_thinking if THINKING else RECIPE.sft.max_seq_len)
    )
    # batch_size is the GLOBAL/effective batch; sft_grad_accum sizes the per-device micro-batch +
    # grad-accum to realize it (shared with the cost estimator's step count, see engine.vram).
    effective_batch = (
        _t.batch_size if _t and _t.batch_size is not None else RECIPE.sft.effective_batch
    )
    # Large-vocab OOM guard: when the fused CE (chalk's FLCE) is OFF, the SFTTrainer materializes the full
    # [per_device, seq, vocab] fp32 logits + grad — at Qwen3.5's ~248k vocab a 0.8B SFT OOM'd a
    # 24 GB card in backward. Cap the per-device micro-batch by the real model vocab + seq so those
    # logits stay within the logits budget; grad-accum rises to keep the effective batch unchanged
    # (the SFT mirror of rl_per_device_comps' GRPO cap). fused = sft_logits_fused, the CONSERVATIVE
    # >=3B / >=2048-ctx gate: chalk's FLCE actually fuses every run, but the cap still binds for a
    # small short-context run (where the allocator doesn't bank on the saving) so it's never OOM if
    # the fused path is unavailable.
    _sft_params_b = resolve_params_b(model_id)  # catalog stat else HF safetensors (open models)
    _sft_vocab = vocab_size_for(model_id)
    # chalk's FLCE fuses every run BY DEFAULT (standalone, no size / liger_kernel-importable gate),
    # so the fused-CE memory saving is normally taken. The allocator stays CONSERVATIVE about banking
    # on it: sft_logits_fused is the >=3B / >=2048-ctx mirror, so the large-vocab logits cap below
    # binds only for small short-context runs (a GPU is never undersized). It ALSO binds when the
    # fused CE can't actually run — i.e. an operator disables FLCE (FLASH_FLCE_KERNEL=0) OR
    # freesolo-chalk (the FLCE provider) isn't importable: there's then no fused saving to bank on,
    # so fused_ce_available() is False and the cap protects against the full-logits OOM. (Gating on
    # the FLAG alone would wrongly assume the saving whenever chalk is missing/failed-to-install.)
    _sft_fused = sft_logits_fused(_sft_params_b, sft_max_len) and fused_ce_available()
    per_device_bs, grad_accum = sft_grad_accum(
        effective_batch, seq_len=sft_max_len, vocab=_sft_vocab, fused=_sft_fused
    )
    if not _sft_fused and per_device_bs < min(effective_batch, 4):
        print(
            f"[sft] large-vocab logits cap: per_device={per_device_bs} grad_accum={grad_accum} "
            f"(seq={sft_max_len}, vocab={_sft_vocab}; realized batch "
            f"{per_device_bs * grad_accum} >= requested {effective_batch})"
        )
    sft_save_default = _t.save_every if _t and _t.save_every is not None else 50
    out_dir = f"/tmp/sft_seed{SEED}"
    resume_ckpt = hf_resume_checkpoint()

    # [train].max_steps>0 caps optimizer steps (used by the cheap pre-flight smoke).
    max_steps = int(_t.max_steps or 0 if _t and _t.max_steps is not None else 0)
    cfg_kwargs = {
        "output_dir": out_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": per_device_bs,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": sft_lr,
        "warmup_ratio": RECIPE.sft.warmup_frac,
        "logging_steps": 10,
        "save_steps": sft_save_default,
        "save_total_limit": 1,
        # Memory-light checkpoints: save ONLY the (small LoRA) model, not the optimizer /
        # scheduler / RNG state — skips the optimizer-state serialization spike at save and
        # writes just the adapter. (We don't resume mid-run; seeds restart cleanly.)
        "save_only_model": True,
        "max_length": sft_max_len,
        "bf16": True,
        "report_to": wandb_report_to(),  # W&B when WANDB_API_KEY present (restored post-flash-migration)
        "run_name": wandb_run_name(),
        # Dataloader parallelism: overlap host-side collation/tokenization with GPU compute so a
        # real (large) training set isn't dataloader-bound. Pure throughput, zero quality change.
        # Negligible on the tiny benchmark (pre-tokenized, in-memory); a real win at production
        # dataset sizes.
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "seed": SEED,
        "gradient_checkpointing": grad_checkpointing_on(model_id, sft_max_len),
        # Non-reentrant checkpointing: composes cleanly with autograd hooks (verl #3629) and is
        # required by TRL for correct grad flow through the LoRA adapters.
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "completion_only_loss": False,
        # Optimizer: 8-bit paged AdamW (int8 state paged to host RAM -> fits a smaller GPU).
        "optim": fused_optim_name(),
    }
    if max_steps > 0:
        cfg_kwargs["max_steps"] = max_steps
    # Example packing: concatenate short examples into full max_length sequences so a batch isn't
    # mostly pad tokens — PR #174 measured a 4.4-10.7x SFT speedup (h100 8.2x, 4090 10.7x) because
    # instruction targets are far shorter than max_seq_len; unpacked batches waste most of their
    # FLOPs on padding. TRL's 'bfd' strategy makes padding-free batches whose example boundaries are
    # honored ONLY by an attention impl that reads them — under plain SDPA packed examples
    # cross-contaminate (silent quality loss). The boundary-correct backend is FlashAttention-2
    # varlen (reads position_ids); but flash-attn has NO prebuilt wheel for torch 2.10 (PyPI
    # sdist-only; Dao-AILab wheels stop at torch 2.9) so it would build from source on every cold
    # start (~20 min, fragile) — it is NOT in the worker image, so _fa_ok is False on the current
    # stack. That alone used to disable packing — but flex_attention (block-diagonal doc mask) is a
    # SECOND boundary-correct backend that needs NO flash-attn, so it enables 'bfd' packing on archs
    # that support it even on torch 2.10 / sm120 where FA2 is absent. MEASURED ~1.7x SFT throughput
    # at equal VRAM on a Llama-arch 0.5B (5090). flex support is per-arch (Llama/Qwen2/3 yes;
    # Qwen3.5/3.6 hybrid-GDN no — HF #34809), so we only enable flex packing when the arch supports it.
    _fa_ok = _flash_attn_available()
    # When FA2 is absent, flex is the fallback boundary-correct backend — but only if the model's
    # arch supports it. Capture WHY it's unavailable so the SKIPPED message is accurate (a real arch
    # limitation vs a possibly-transient config-probe failure), instead of always blaming the arch.
    _flex_status = "unsupported" if _fa_ok else flex_attn_status(model_id)
    _flex_ok = _flex_status == "supported"
    if _fa_ok:
        cfg_kwargs["packing"] = True
        print("[sft] example packing enabled (FA2 varlen)")
    elif _flex_ok:
        cfg_kwargs["packing"] = True
        print("[sft] example packing enabled (flex_attention block-diagonal mask)")
    else:
        # flash-attn is absent on torch 2.10; distinguish the flex fallback reason so the diagnosis
        # isn't misleading when the config probe merely failed (offline / transient HF error).
        _flex_reason = (
            "the model's arch lacks flex_attention support"
            if _flex_status == "unsupported"
            else "the flex_attention support probe failed (offline / transient HF error)"
        )
        print(
            f"[sft] packing SKIPPED: no boundary-correct attn backend (flash-attn absent on torch "
            f"2.10 and {_flex_reason}). Bake flash-attn into the worker image, or use a flex-capable "
            f"arch (with a reachable config), to enable packing."
        )
    # Fused CE/RMSNorm/SwiGLU come from chalk (STANDALONE), NOT Liger: install_chalk_kernels patches
    # the live model AFTER the trainer builds it, with chalk's FLCE on by default — so the big
    # large-vocab (Qwen3.5 ~248k) memory/throughput win is preserved without use_liger_kernel.
    _attn = optimal_attn_impl()  # arch-aware FlashAttention (Kernels Hub) / SDPA
    # Packing correctness: 'bfd' packed batches are boundary-correct ONLY under a varlen/masked attn.
    # Force the boundary-correct backend whenever packing is on, over the SDPA default which would
    # cross-contaminate packed examples: FA2 varlen if available, else flex_attention's doc mask.
    # (Merge: keep this PR's flex_attention fallback; QLoRA tier dropped per dev #74 — catalog is
    # all bf16 — so model_init_kwargs is plain bf16.)
    if cfg_kwargs.get("packing"):
        if _fa_ok:
            _attn = "flash_attention_2"
            print("[sft] attn_implementation=flash_attention_2 (packing boundary-correct varlen)")
        elif _flex_ok:
            _attn = "flex_attention"
            print("[sft] attn_implementation=flex_attention (packing boundary-correct block mask)")
    # Explicit bf16 + no auto device-map: TRL/transformers-5 string loading can
    # otherwise fall back to fp32 (2x VRAM; observed 18.6 GB for a 4.66B model) or
    # accelerate-offload large models to meta ("expected device meta but got
    # cuda:0" in backward on the 9B).
    mik = {"dtype": "bfloat16", "device_map": None}
    if _attn:
        mik["attn_implementation"] = _attn
    cfg_kwargs["model_init_kwargs"] = mik
    cfg = TRLSFTConfig(**cfg_kwargs)

    # LoRA+ (convergence lever, arXiv 2402.12354; always-on: measured -52% train loss in A/B
    # (gpu-bench)): give the LoRA B matrices a higher LR than A (ratio 16). Reported ~2x fewer steps
    # to target at identical per-step FLOPs. TRL builds the model from a string inside __init__, so
    # the optimizer (which needs the instantiated params) can't be pre-built — override
    # create_optimizer to construct it from self.model once it exists.
    _lp_ratio = 16
    _SFT = SFTTrainer
    if _lp_ratio > 1:

        class _SFT(SFTTrainer):  # local LoRA+ subclass
            _loraplus_applied = False  # True only once the LoRA+ grouping actually installs

            def create_optimizer(self):
                if self.optimizer is None:
                    try:
                        from peft.optimizers import create_loraplus_optimizer

                        # Mirror the configured `optim` so LoRA+ and the 8-bit paged optimizer state
                        # coexist (instead of silently forcing fp32 AdamW); see loraplus_optimizer_cls.
                        # .value (not str()): self.args.optim is a TRL OptimizerNames enum whose
                        # str() is "OptimizerNames.PAGED_ADAMW_8BIT"; pass the raw value
                        # ("paged_adamw_8bit") so the 8-bit match works.
                        opt_cls, extra = loraplus_optimizer_cls(
                            getattr(self.args.optim, "value", self.args.optim)
                        )
                        # Forward the TrainingArguments optimizer config that the default HF
                        # create_optimizer path would have applied. Building the optimizer
                        # ourselves means we must replicate it explicitly, or LoRA+ runs would
                        # silently use the optimizer class's own defaults instead of the
                        # configured betas/eps/weight_decay. betas/eps go straight to the optimizer
                        # constructor (alongside any `extra` from loraplus_optimizer_cls);
                        # weight_decay is handled separately below.
                        fwd = dict(extra)
                        _betas = (
                            getattr(self.args, "adam_beta1", None),
                            getattr(self.args, "adam_beta2", None),
                        )
                        if None not in _betas:
                            fwd.setdefault("betas", _betas)
                        _eps = getattr(self.args, "adam_epsilon", None)
                        if _eps is not None:
                            fwd.setdefault("eps", _eps)
                        # PEFT does NOT read args.weight_decay; it applies decay via its own LoRA+
                        # param groups, keyed off the loraplus_weight_decay kwarg (which it pops
                        # before constructing the optimizer). Pass it as a top-level kwarg so it
                        # isn't forwarded into the optimizer constructor.
                        lp_extra: dict[str, object] = {}
                        _wd = getattr(self.args, "weight_decay", None)
                        if _wd is not None:
                            lp_extra["loraplus_weight_decay"] = _wd
                        # PEFT's create_loraplus_optimizer forwards extra kwargs to the optimizer;
                        # the lr keyword name has shifted across PEFT versions, so pass it via
                        # optimizer_kwargs (the stable form) and fall back to a top-level lr=.
                        try:
                            self.optimizer = create_loraplus_optimizer(
                                model=self.model,
                                optimizer_cls=opt_cls,
                                optimizer_kwargs={"lr": self.args.learning_rate, **fwd},
                                loraplus_lr_ratio=_lp_ratio,
                                **lp_extra,
                            )
                        except TypeError:
                            self.optimizer = create_loraplus_optimizer(
                                model=self.model,
                                optimizer_cls=opt_cls,
                                lr=self.args.learning_rate,
                                loraplus_lr_ratio=_lp_ratio,
                                **fwd,
                                **lp_extra,
                            )
                        self._loraplus_applied = True
                        print(
                            f"[lora+] optimizer enabled (B-matrix LR ratio={_lp_ratio}, "
                            f"cls={opt_cls.__name__})"
                        )
                        return self.optimizer
                    except Exception as e:  # never block training on the LoRA+ wiring
                        print("[lora+] setup failed, falling back to default optimizer:", e)
                return super().create_optimizer()

    # Pass model as a string id + tokenizer as processing_class so TRL takes the
    # text/causal-LM path (not the VLM processor path) for this multimodal checkpoint.
    trainer = _SFT(
        model=model_id,
        args=cfg,
        train_dataset=ds,
        peft_config=make_lora(model_id),
        processing_class=tok,
        callbacks=[make_checkpoint_upload_callback()],
    )
    # Apply chalk's standalone fused kernels (rms_norm/swiglu/FLCE + RoPE/LoRA-delta/embedding) on
    # the materialized SFT trainer.model — chalk's apply patches the LIVE module, so it must run
    # AFTER TRL builds the model. No-op unless a FLASH_* kernel flag selects it and freesolo-chalk
    # is installed.
    _chalk_report = install_chalk_kernels(getattr(trainer, "model", None))

    _reset_peak_gpu()  # so peak_gpu_gb reflects the train loop (optimizer-state A/B is measurable)
    _gpu_sampler = _GpuPeakSampler().start()  # true device peak incl. bnb managed optimizer pages
    t_train = time.time()
    with _sdpa_cudnn_ctx(_attn):  # force cuDNN SDPA on sm120 (no-op otherwise)
        trainer.train(resume_from_checkpoint=resume_ckpt)
    train_wall = time.time() - t_train
    sft_peak_gpu_gb = _peak_gpu_gb()
    sft_device_peak_gpu_gb = _gpu_sampler.stop_gb()

    adapter_dir = f"{out_dir}/adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    hf_upload_folder(adapter_dir, "adapter", required=True)
    heartbeat("sft_trained", train_wall=train_wall)

    # count train tokens
    train_tokens = int(sum(len(tok(t["text"])["input_ids"]) for t in texts) * epochs)

    # Write train metadata + the completion sentinel (metrics.json/DONE) for this phase.
    write_train_meta(
        phase="sft",
        adapter_dir=adapter_dir,
        model_id=model_id,
        train_wall=train_wall,
        setup_seconds=setup_seconds,
        train_tokens=train_tokens,
        generated_tokens=0,
        notes={
            "epochs": epochs,
            "resumed": bool(resume_ckpt),
            "download_seconds": download_seconds,
            "hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
            "thinking": THINKING,
            # Persist the loss curve so a CONVERGENCE A/B (PiSSA / LoRA+ init, etc.) is measurable
            # without a checkpoint: trainer_state.json is only written on a save_step, and the
            # console is only uploaded on failure, so a short successful run otherwise drops its
            # loss history entirely.
            "loss_curve": _metric_curve(trainer, "loss"),
            # Peak torch-allocated GPU memory during the train loop (excludes bnb managed pages, so
            # it overstates the 8-bit saving — use device_peak_gpu_gb for the true footprint).
            "peak_gpu_gb": sft_peak_gpu_gb,
            # True peak device memory (total-free, incl. bnb managed optimizer pages): the honest
            # headline for the fp32-vs-8-bit LoRA+ optimizer A/B.
            "device_peak_gpu_gb": sft_device_peak_gpu_gb,
            # Report the optimizer ACTUALLY built on the trainer, not the planned class: if the
            # LoRA+ create_optimizer override failed, training falls back to TRL's configured
            # optimizer without LoRA+ grouping. loraplus_applied records which path actually ran.
            # Accelerate wraps the optimizer (AcceleratedOptimizer) under transformers 5.x, so unwrap
            # via `.optimizer` to record the underlying PagedAdamW8bit/AdamW the A/B cares about, not
            # the wrapper name.
            "loraplus_optim": (
                type(getattr(trainer.optimizer, "optimizer", trainer.optimizer)).__name__
                if getattr(trainer, "optimizer", None) is not None
                else loraplus_optimizer_cls(fused_optim_name())[0].__name__
            ),
            "loraplus_applied": getattr(trainer, "_loraplus_applied", False),
            # Which chalk gap-filling kernels actually ENGAGED (empty/None = chalk not installed or
            # every kernel fell back) — verifies the chalk stack without the console.
            "chalk_kernels": active_kernels(_chalk_report) or None,
            **wandb_run_info(),
        },
    )
    free_gpu(trainer)


# ---------------------------------------------------------------------------
# RL (GRPO) with TRL + colocated vLLM
# ---------------------------------------------------------------------------
def compute_grpo_batching(prompts_per_step: int, group_size: int, per_device_comps: int) -> dict:
    """Translate an intended ``prompts_per_step`` into a TRL GRPO batch configuration.

    TRL's GRPO batch sizing is denominated in **completions (prompt-completion pairs), not
    prompts**. The number of *unique prompts* optimized per step is

        (per_device_train_batch_size * gradient_accumulation_steps * num_processes)
        / num_generations

    So to actually optimize ``prompts_per_step`` prompts per step, the global *completion*
    batch must equal ``prompts_per_step * group_size``. We keep ``per_device`` small (it,
    not grad-accum, sets peak VRAM) and put the rest in gradient accumulation.

    The bug this fixes: ``grad_accum = prompts_per_step // per_device`` treated
    ``per_device_train_batch_size`` as a *prompt* count, omitting the ``* group_size``
    factor, so a run intended as 64 prompts/step actually optimized only
    ``64 / group_size = 8`` prompts/step (an 8x smaller effective batch).
    """
    import math

    group_size = max(1, int(group_size))
    prompts_per_step = max(1, int(prompts_per_step))
    per_device = max(1, int(per_device_comps))
    target_comps = prompts_per_step * group_size  # total completions / optimizer step
    # Never let the per-device completion micro-batch exceed the target completion batch:
    # a small prompts_per_step would otherwise overshoot it (mirrors run_sft's
    # `min(per_device_bs, effective_batch)`). No-op at the default (prompts_per_step=64).
    per_device = max(1, min(per_device, target_comps))
    grad_accum = max(1, target_comps // per_device)
    # TRL rejects a global completion batch (per_device * grad_accum) that is not
    # divisible by num_generations (= group_size), failing only AFTER the paid worker
    # is provisioned. per_device is the fixed VRAM knob, so round grad_accum UP to the
    # next multiple that makes the batch divisible (grad_accum must be a multiple of
    # group_size // gcd(per_device, group_size)). This only ever raises the effective
    # batch slightly; the common per_device|group_size cases are unchanged.
    accum_step = group_size // math.gcd(per_device, group_size)
    grad_accum = ((grad_accum + accum_step - 1) // accum_step) * accum_step
    generations_per_step = per_device * grad_accum
    unique_prompts_per_step = generations_per_step // group_size
    return {
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": grad_accum,
        "generations_per_step": generations_per_step,
        "unique_prompts_per_step": unique_prompts_per_step,
        # TRL requires the global completion batch be divisible by num_generations.
        "divisible_by_group": (generations_per_step % group_size == 0),
    }


def rl_per_device_comps(
    completion_len: int = 0,
    vocab: int = 248_320,
    *,
    use_vllm: bool = True,
    params_b: float | None = None,
) -> int:
    """Per-device *completion* micro-batch for GRPO (TRL counts completions, not prompts).

    This, not grad-accum, sets peak trainer VRAM: the logprob pass materializes fp32 logits
    of shape [per_device, completion_len, vocab]. At Qwen3.5's ~248k vocab a long completion is
    enormous (measured: per_device 8 x 4096 tok x 248k x 4 B = ~30 GiB single alloc -> OOMs
    a small card). So we MEMORY-CAP per_device to a logits budget (6 GB) for the
    given completion length, then push the difference into grad-accum
    (compute_grpo_batching) so the effective batch is unchanged. This keeps long-completion
    GRPO on a cheaper GPU.

    The logits budget is NOT the whole story: the per-device forward also holds the model's
    attention/activation memory (the Qwen3.5 GDN/FLA kernels peak per micro-batch even with
    grad checkpointing), which the logits term can't see. Under colocated vLLM (the rollout
    engine + its card-sized KV pool + a 2nd weight copy share the GPU) that activation peak is
    what OOMs a small card -- and Liger, which fuses away the logits, does NOT touch it.
    MEASURED: Qwen3.5-2B (width ~1.41) group8 seq2048 OOMs a 32 GB card at per_device=8 but
    TRAINS at 4. So for colocate, additionally cap per_device to the live card's VRAM scaled
    by model width (~sqrt(params)): ~vram_gb/8 at 2B-width, tightened for wider models (4B/9B).
    """
    # Default prompts/step; the auto-caps below (logits budget + colocate VRAM/width) handle OOM.
    base = 2 if THINKING else 8
    if completion_len > 0:
        budget = 6.0 * 1e9
        cap = max(1, int(budget / (max(1, completion_len) * vocab * 4)))
        base = min(base, cap)
    if use_vllm:
        try:
            import torch

            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                width = (max(float(params_b), 0.1) ** 0.5) if params_b else 1.41
                act_cap = max(1, int(vram_gb / (7.5 * (width / 1.41))))
                base = min(base, act_cap)
        except Exception as e:
            print("rl_per_device_comps colocate cap probe failed (keeping logits cap):", e)
    return max(1, base)


def make_reward_heartbeat_callback():
    """A TRL/transformers callback that streams the per-step mean reward to the HF heartbeat
    channel, giving the worker a live RL signal (no pod log API) and recording a
    ``reward_history``. Built lazily so the module imports without transformers installed."""
    from transformers import TrainerCallback

    class _RewardHeartbeat(TrainerCallback):
        def __init__(self):
            self.reward_history = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            r = logs.get("reward")
            if r is None:
                return
            try:
                r = float(r)
            except (TypeError, ValueError):
                return
            self.reward_history.append(r)
            step = int(getattr(state, "global_step", len(self.reward_history)))
            heartbeat("rl_step", step=step, reward=r, reward_last=self.reward_history[-8:])

    return _RewardHeartbeat()


def grpo_overrides() -> dict:
    """The GRPO recipe knobs, read off the job spec's ``[train]`` table (``TrainSpec``).
    A field left unset (None) is omitted here so the recipe default applies downstream.

    Knobs: group_size, temperature, max_tokens (completion budget), kl_penalty_coef (the KL
    beta), advantage_clip (centered-advantage clip), and thinking_length_penalty_coef
    (a per-<think>-token reward deduction). These live in ``[train]`` — NOT in
    ``[environment.params]``, which is forwarded verbatim to the verifiers env loader."""
    if not JOB_SPEC:
        return {}
    train = JOB_SPEC.train
    cfg = {
        "group_size": train.group_size,
        "temperature": train.temperature,
        "max_tokens": train.max_tokens,
        "kl_penalty_coef": train.kl_penalty_coef,
        "advantage_clip": train.advantage_clip,
        "thinking_length_penalty_coef": train.thinking_length_penalty_coef,
    }
    return {k: v for k, v in cfg.items() if v is not None}


def think_token_count(completion: str | None, tokenizer) -> int:
    """Number of tokens inside the completion's <think>...</think> span (0 if none).

    Used for the thinking-length reward deduction: long reasoning is penalized in
    proportion to the tokens it spent, mirroring the SDK's thinking_length_penalty_coef.
    """
    if not completion or "<think>" not in completion:
        return 0
    after = completion.split("<think>", 1)[1]
    think_text = after.split("</think>", 1)[0] if "</think>" in after else after
    if not think_text:
        return 0
    return len(tokenizer(think_text, add_special_tokens=False)["input_ids"])


def _init_adapter_model(model_id: str):
    """Base model + the ``train.init_from_adapter`` adapter loaded as a trainable
    PeftModel, or the plain ``model_id`` string + a fresh LoRA when it is unset.

    GRPO continuing an SFT adapter: TRL trains the LOADED adapter (peft_config=None)
    instead of attaching a fresh one."""
    prefix = JOB_SPEC.train.init_from_adapter if JOB_SPEC else ""
    if not prefix:
        return model_id, make_lora(model_id)
    adir = _download_adapter(prefix)
    if not adir:
        # The user explicitly asked GRPO to continue from this adapter; silently
        # falling back to a fresh base-model LoRA would spend a full paid run
        # optimizing the wrong starting point. Fail hard instead.
        raise RuntimeError(
            f"train.init_from_adapter={prefix!r} could not be downloaded from the artifact "
            "store (wrong/missing prefix or no access); refusing to silently start GRPO from "
            "the base model. Fix the adapter prefix / HF credentials, or omit "
            "init_from_adapter to train a fresh LoRA."
        )
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    print(f"[init-adapter] initializing LoRA from {prefix}")
    # VL checkpoints (Qwen3.5/3.6): the SFT step saved the adapter against the FULL multimodal model
    # (keys under ``base_model.model.model.language_model.layers.*``), but we load the base here via
    # AutoModelForCausalLM (text-only tree, ``base_model.model.model.layers.*``). Strip the
    # ``.language_model.`` infix on disk so PeftModel.from_pretrained matches the SFT keys —
    # otherwise peft only WARNS about missing keys and silently trains a fresh LoRA, discarding the
    # SFT. No-op for non-VL checkpoints. See flash/engine/worker/lora.py.
    remap_vl_adapter_dir(adir, model_id)
    _attn = optimal_attn_impl()
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        **({"attn_implementation": _attn} if _attn else {}),
    )
    model = PeftModel.from_pretrained(base, adir, is_trainable=True)
    # Fail loudly if the adapter didn't actually apply (a future key-mismatch regression would
    # otherwise silently start GRPO from the base model again).
    assert_lora_applied(model, model_id)
    return model, None


def _grpo_resume_already_complete(resume_ckpt, target_steps: int, steps_run: int) -> bool:
    """True when this worker resumed a checkpoint that already reached the target step count.

    Such a resume legitimately performs ZERO new optimizer steps (so the fresh hb_cb has an empty
    reward_history) yet the policy IS fully trained — it must NOT be flagged as a no-op failure.
    """
    return bool(resume_ckpt) and target_steps > 0 and steps_run >= target_steps


def _grpo_is_no_op_failure(
    reward_history, resume_ckpt, target_steps: int, steps_run: int
) -> bool:
    """True when a GRPO run trained NOTHING and must fail loudly instead of reporting as done.

    An empty ``reward_history`` means the reward callback never fired — the rollout scored nothing
    (e.g. vLLM silently returning no completions), so no real training happened. The sole exception
    is a resume that already reached the target steps (see ``_grpo_resume_already_complete``): that
    has an empty fresh history but a fully-trained policy, so it is NOT a failure.
    """
    if reward_history:
        return False
    return not _grpo_resume_already_complete(resume_ckpt, target_steps, steps_run)


def run_rl():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    require_active_env()  # fail loudly (not AttributeError: NoneType) on the no-JobSpec path
    t_start = time.time()
    heartbeat("rl_start")
    # GRPO rollout strategy by env shape (trl 1.6 adds the hooks these need):
    #   * single-turn          -> TRL single-shot generation + per-completion reward (below);
    #   * tool (ToolEnv & subs:
    #     Stateful/Sandbox/Python) -> TRL drives the tool-call loop natively via
    #     GRPOTrainer(tools=...) (it parses tool calls, executes the tools, and masks the
    #     tool-result tokens itself); the reward scores the full transcript;
    #   * pure multi-turn      -> a custom rollout_func (flash.engine.multiturn_rollout)
    #     drives THIS env's turn loop on the colocate engine and returns the interleaved
    #     token sequence with an env_mask so only the model's tokens are trained.
    is_tool_env = getattr(ACTIVE_ENV, "is_tool_env", False)
    is_multi_turn = getattr(ACTIVE_ENV, "multi_turn", False)
    conversational = is_multi_turn  # message-list prompts (tool + pure multi-turn) vs strings
    wait_for_gpu()
    setup_perf_backends()
    model_id = JOB_SPEC.model if JOB_SPEC else RECIPE.hf_model_id
    download_seconds = prefetch_model(model_id)
    rl = RECIPE.rl
    # Steps come from the run's [train] steps (already in JOB_SPEC), else the recipe default.
    steps = int(
        JOB_SPEC.train.steps if JOB_SPEC and JOB_SPEC.train.steps is not None else rl.num_steps
    )
    # Throughput/quality knobs: the number of prompts optimized per step, completions per
    # prompt, and whether vLLM offloads weights between steps. Sleep mode frees memory for the
    # optimizer but reloads ~weights each step (a large per-step cost); it's gated OFF by model
    # size when both the policy and rollout engine fit resident.
    gcfg = grpo_overrides()
    _t = JOB_SPEC.train if JOB_SPEC else None
    # batch_size = prompts per optimizer step for GRPO.
    # prompts per optimizer step = the run config's [train].batch_size (recipe default otherwise).
    prompts_per_step = int(_t.batch_size if _t and _t.batch_size is not None else rl.prompts_per_step)
    group_size = int(gcfg.get("group_size") or rl.group_size)
    # temperature: explicit None check, NOT `or` — a configured 0.0 (greedy/deterministic
    # rollouts) must be honored, not fall back to the recipe sampling temperature.
    _gcfg_temp = gcfg.get("temperature")
    _temperature = float(_gcfg_temp if _gcfg_temp is not None else rl.sampling_temperature)
    _kl_beta = float(gcfg.get("kl_penalty_coef") or 0.0)
    _adv_clip = float(gcfg.get("advantage_clip") or 0.0)
    _think_penalty = float(gcfg.get("thinking_length_penalty_coef") or 0.0)
    # vLLM sleep mode offloads the rollout engine's weights between steps to free memory for the
    # optimizer, but reloading each step is a large per-step cost — PR #174 measured ~2-2.6x faster
    # GRPO with it OFF on models that fit. Gate it by model size (same small=speed / large=memory
    # gate as gradient checkpointing): OFF for small/fitting models, ON for large.
    # Gate on the GRPO rollout context (the run's [train].max_length sizes the engine + KV cache):
    # a long-context GRPO run is memory-tight and needs sleep mode. Matches the liger-loss gate below.
    _grpo_ctx = int(_t.max_length if _t and _t.max_length else 0)
    sleep_mode = _memory_mode(model_id, _grpo_ctx)
    # Rollout backend: always colocated vLLM (fast). The whole supported catalog runs GRPO with
    # colocated vLLM; there is no transformers-generation fallback.
    use_vllm = True
    print("[rl] rollout backend: colocated vLLM")
    from flash.catalog import MODELS as _CATALOG

    _info = _CATALOG.get(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train = ACTIVE_ENV.dataset()
    rng = random.Random(SEED)
    rng.shuffle(train)
    if conversational:
        # Message-list prompts so the chat template applies roles + (for tool envs) the tool
        # schemas; per-turn length is managed by the tool loop / rollout_func, not a flat budget.
        prompts = [{"prompt": ACTIVE_ENV.prompt_messages(ex), "example": ex} for ex in train]
    else:
        prompts = [{"prompt": render_prompt(tok, ex), "example": ex} for ex in train]
    # The colocated vLLM engine's model length is the hard cap on prompt+completion at
    # rollout. Size it from [train].max_length and derive the prompt budget from it so a
    # bigger engine or a smaller completion automatically admits longer prompts (rather than
    # a fixed rl.max_prompt_len that no env override could lift).
    _max_completion = int(
        gcfg.get("max_tokens")
        or (rl.max_completion_len_thinking if THINKING else rl.max_completion_len)
    )
    # Engine context = the run's [train].max_length (so a long-context GRPO config sized/paid for
    # by the allocator actually RUNS at that length), else the recipe default. Without the
    # train.max_length fallback the allocator provisions a big GPU for the long context but the
    # engine runs short — paying for headroom we never use.
    _train_ctx = _t.max_length if (_t and _t.max_length) else 0
    vllm_max_len = int(_train_ctx or max(1024, rl.max_prompt_len + _max_completion))
    # The engine must fit completion + at least some prompt. If [train].max_length is below the
    # completion budget, no prompt can ever fit — fail fast here rather than passing a 1-token
    # budget that lets prompts through and then OOMs/overflows mid-rollout.
    if vllm_max_len <= _max_completion:
        raise ValueError(
            f"engine length {vllm_max_len} leaves no room for the {_max_completion}-token "
            "completion; raise [train].max_length or lower [train].max_tokens"
        )
    prompt_budget = vllm_max_len - _max_completion

    # TRL 1.5's GRPOConfig has no max_prompt_length and does NOT truncate prompts, so a prompt
    # that leaves no room for the completion within the engine length would fail mid-rollout
    # AFTER the paid worker is provisioned. Drop prompts that don't fit the budget up front.
    # render_prompt returns an apply_chat_template(tokenize=False) string that already carries
    # the special tokens, so tokenize with add_special_tokens=False (the default re-adds
    # BOS/EOS and over-counts).
    # Drop prompts that leave no room for the completion within the engine length — applies to
    # BOTH single-turn (string prompts) and conversational (message-list) prompts, so a tool /
    # multi-turn rollout can't overflow the colocate engine mid-generation. Conversational
    # prompts are length-checked via the chat template (with the generation prompt).
    # Tool schemas TRL injects into the prompt for native tools= GRPO — include them in the
    # budget for a tool env so a prompt isn't undercounted at filter time vs. rollout time.
    _oai_tools = (
        getattr(getattr(ACTIVE_ENV, "_env", None), "oai_tools", None) if is_tool_env else None
    )

    def _prompt_tokens(p) -> int:
        if conversational:
            # Render to text then tokenize — the SAME path the rollout uses — so the filter
            # count matches the rollout's count (avoids a tokenize=True vs text mismatch).
            kw = {"tools": _oai_tools} if _oai_tools else {}
            try:
                text = tok.apply_chat_template(
                    p["prompt"],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=THINKING,
                    **kw,
                )
            except Exception as exc:
                # Fail fast WITH context: a tokenizer/template incompatibility would render every
                # prompt uncountable and otherwise surface as a misleading "all prompts exceed
                # budget" — raise so the model/template can be fixed before a paid run trains on
                # a degenerate dataset.
                raise RuntimeError(
                    "failed to render a conversational prompt with this model's chat template "
                    f"(fix the model/template or the env's prompts): {exc}"
                ) from exc
            return len(tok(text, add_special_tokens=False).input_ids)
        return len(tok(p["prompt"], add_special_tokens=False).input_ids)

    kept = [p for p in prompts if 0 < _prompt_tokens(p) <= prompt_budget]
    if len(kept) < len(prompts):
        print(
            f"[rl] dropped {len(prompts) - len(kept)} prompts over the {prompt_budget}-token "
            f"prompt budget (engine {vllm_max_len} - completion {_max_completion})"
        )
    if not kept:
        raise ValueError(
            f"every training prompt exceeds the {prompt_budget}-token prompt budget (engine "
            f"{vllm_max_len} - completion {_max_completion}); raise [train].max_length, lower "
            "[train].max_tokens, or shorten the environment's prompts"
        )
    prompts = kept
    ds = Dataset.from_list(prompts)

    def reward_fn(completions, **kwargs):
        # rollout_func (pure multi-turn) path: the per-rollout reward is computed by the env
        # during the rollout and forwarded as the "reward" extra field — pass it through.
        if kwargs.get("reward") is not None:
            return [float(r) for r in kwargs["reward"]]
        # Score the <think>-stripped text (graded_text), then — datums parity — deduct
        # the thinking-length penalty computed from the RAW completion's <think> span.
        examples = kwargs.get("example")
        rewards = []
        for comp, ex in zip(completions, examples, strict=False):
            if isinstance(comp, list):
                # Tool / conversational transcript (TRL passes a list of messages): score the
                # whole transcript via the rubric (no <think> stripping — multi-turn content).
                rewards.append(ACTIVE_ENV.reward_from_messages(comp, ex))
                continue
            r = ACTIVE_ENV.reward(graded_text(comp), ex)
            if _think_penalty > 0 and THINKING:
                r -= _think_penalty * think_token_count(comp, tok)
            rewards.append(r)
        return rewards

    # TRL's per_device_train_batch_size counts COMPLETIONS, not prompts. Size grad-accum so
    # the global completion batch = prompts_per_step * group_size, i.e. each optimizer step
    # actually optimizes `prompts_per_step` prompts. The per-device *completion* micro-batch
    # is the VRAM knob (thinking-aware; see rl_per_device_comps).
    from flash.engine.vram import resolve_params_b

    # Open-model (uncataloged) GRPO: size the colocate activation cap from the catalog stat, else
    # the HF safetensors metadata (no download). Without a real count a large open model falls back
    # to the ~2B-width default in rl_per_device_comps and gets too LOOSE a per-device cap ->
    # colocate OOM. Best-effort: stays None offline, keeping prior behavior.
    _params_b = resolve_params_b(model_id)
    from flash.catalog import vocab_size_for

    per_device_comps = rl_per_device_comps(
        _max_completion, vocab=vocab_size_for(model_id), use_vllm=use_vllm, params_b=_params_b
    )
    batching = compute_grpo_batching(prompts_per_step, group_size, per_device_comps)
    if not batching["divisible_by_group"]:
        print("WARN: generation batch not divisible by group size; check prompts_per_step/group_size")
    print(
        f"[rl] GRPO batching: per_device={batching['per_device_train_batch_size']} "
        f"grad_accum={batching['gradient_accumulation_steps']} "
        f"generations/step={batching['generations_per_step']} "
        f"unique_prompts/step={batching['unique_prompts_per_step']} "
        f"(target prompts/step={prompts_per_step}, group={group_size}, sleep={sleep_mode})"
    )
    out_dir = f"/tmp/rl_seed{SEED}"
    resume_ckpt = hf_resume_checkpoint()

    grpo_kwargs = {
        "output_dir": out_dir,
        "learning_rate": (
            _t.learning_rate if _t and _t.learning_rate is not None else rl.learning_rate
        ),
        "per_device_train_batch_size": batching["per_device_train_batch_size"],
        "gradient_accumulation_steps": batching["gradient_accumulation_steps"],
        "num_generations": group_size,
        # NB: GRPOConfig has no max_prompt_length field (TRL 1.5) and does not truncate
        # prompts; the dataset is pre-filtered above to prompts that fit prompt_budget
        # (vllm_max_len - completion), so every prompt fits the engine sized here.
        "max_completion_length": _max_completion,
        "max_steps": steps,
        "temperature": _temperature,
        "top_p": rl.sampling_top_p,
        "use_vllm": use_vllm,
        "logging_steps": 1,
        "save_steps": _t.save_every if _t and _t.save_every is not None else 20,
        "save_total_limit": 1,
        # Memory-light checkpoints: adapter only, no optimizer/scheduler/RNG state -> no
        # serialization spike at save (the save-step OOM guard).
        "save_only_model": True,
        "bf16": True,
        "report_to": wandb_report_to(),  # W&B when WANDB_API_KEY present (restored post-flash-migration)
        "run_name": wandb_run_name(),
        "seed": SEED,
        "gradient_checkpointing": grad_checkpointing_on(model_id, vllm_max_len),
        # Non-reentrant checkpointing: the modern path that composes correctly with autograd
        # saved-tensor hooks and avoids the reentrant path's extra graph retention. (verl #3629.)
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # Pin a stable, well-conditioned GRPO recipe instead of inheriting TRL's defaults
        # (which on a short run suppress the lift): constant LR (TRL default 'linear' decays
        # to 0 over the run), advantages centered by group mean only (no std scaling, which
        # biases by difficulty/length — matches datums.centered_advantages), and no
        # length-normalized loss. beta is the KL-to-reference coef (datums kl_masks ->
        # kl_penalty_coef).
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "beta": _kl_beta,
        "scale_rewards": "none",
        "loss_type": "dr_grpo",
        # Optimizer: 8-bit paged AdamW (int8 state paged to host RAM -> fits a smaller GPU);
        # colocated GRPO (trainer + vLLM on one GPU) is memory-tight, so this is the right default.
        "optim": fused_optim_name(),
    }
    # Fused GRPO loss comes from chalk's FLCE (STANDALONE), NOT Liger: install_chalk_kernels patches
    # the live model after the trainer builds it, with chalk's fused-linear-CE on by default. That
    # fuses the lm_head + per-token logprob so the fp32 [batch, seq, ~248k vocab] logits never
    # materialize — the documented GRPO OOM driver (MEASURED: without a fused CE even Qwen3.5-0.8B
    # GRPO OOMs a 24/32 GB card because the per-completion 248k-vocab logits dominate). flash no
    # longer sets TRL's use_liger_kernel; chalk's FLCE beats Liger's and carries this protection.
    if use_vllm:
        # RTX 5090 / sm120: pin a PTX-independent vLLM attention backend (FLASHINFER) BEFORE TRL
        # builds the colocated engine — else the rollout can silently produce no completions on
        # old-driver Blackwell hosts (flash-attn PTX JIT failure). No-op off sm120 / if pinned.
        force_vllm_backend_for_sm120()
        # Colocate shares one GPU between the policy model and the vLLM rollout engine.
        # vllm_max_model_length bounds the KV cache to what GRPO needs (else vLLM sizes for
        # the model's FULL context and won't start on a consumer GPU). vllm_gpu_memory_utilization
        # sizes vLLM's pool; sleep mode offloads its weights between steps.
        grpo_kwargs.update(
            vllm_mode="colocate",
            vllm_max_model_length=vllm_max_len,
            vllm_gpu_memory_utilization=0.45,
            vllm_enable_sleep_mode=sleep_mode,
        )
        # Rollout-memory + throughput knobs, applied ONLY if this TRL exposes the field (so an
        # older TRL never crashes on an unknown kwarg). All verl-validated for GRPO colocate (#174).
        _grpo_fields = set(getattr(GRPOConfig, "__dataclass_fields__", {}))

        def _set_vllm_field(names, value, label):
            for _f in names:
                if _f in _grpo_fields:
                    grpo_kwargs[_f] = value
                    print(f"[rl] {label} ({_f}={value})")
                    return True
            return False

        # fp8 KV cache only where the silicon has native fp8 (compute capability >= 8.9: Ada /
        # Hopper / Blackwell) — ~halves the rollout KV pool. Ampere (A100/A5000/3090, sm80) lacks
        # fp8, so it stays fp16 there (forcing it on would error / silently emulate).
        try:
            import torch as _torch

            _want_fp8 = _torch.cuda.get_device_capability() >= (8, 9)
        except Exception:
            _want_fp8 = False
        if _want_fp8:
            _set_vllm_field(("vllm_kv_cache_dtype", "kv_cache_dtype"), "fp8", "fp8 KV cache")
        # PREFIX CACHING: every GRPO group of `num_generations` rollouts shares the SAME prompt
        # prefix, so caching the prompt KV computes it once and reuses it — the dominant rollout win
        # on one GPU. CHUNKED PREFILL interleaves prefill with decode so a long prompt doesn't stall
        # the batch. CUDAGRAPH MODE sets verl's full-graph-decode + piecewise-fallback rollout mode.
        _set_vllm_field(
            ("vllm_enable_prefix_caching", "enable_prefix_caching"),
            True,
            "vLLM prefix caching (shared GRPO prompt KV reuse)",
        )
        _set_vllm_field(
            ("vllm_enable_chunked_prefill", "enable_chunked_prefill"),
            True,
            "vLLM chunked prefill",
        )
        _set_vllm_field(
            ("vllm_compilation_config", "compilation_config"),
            {"cudagraph_mode": "FULL_AND_PIECEWISE"},
            "vLLM cudagraph_mode (verl rollout default)",
        )
    # Adapter init: continue training the SFT adapter (peft_config=None, model is the
    # loaded PeftModel) when train.init_from_adapter is set, else a fresh LoRA on the
    # string model id (model_init_kwargs forces bf16 — TRL string-loading can fall back
    # to fp32 and double VRAM).
    init_model, init_peft = _init_adapter_model(model_id)
    # chalk's kernels are applied AFTER construction (below) against trainer.model: chalk's apply
    # patches the LIVE nn.Module, so there is nothing to install pre-build. On the fresh-LoRA path
    # init_model is just the model-id string (TRL builds the module), and even on the
    # continue-adapter path TRL may rebuild/wrap the PeftModel, so trainer.model is the
    # authoritative target.
    if init_peft is not None:
        # Fresh LoRA: TRL loads the string model id with these kwargs, then attaches the
        # adapter. Force bf16 (TRL string-loading can fall back to fp32 and double VRAM).
        _attn = optimal_attn_impl()  # arch-aware FlashAttention (Kernels Hub) / SDPA
        grpo_kwargs["model_init_kwargs"] = {"dtype": "bfloat16"}
        if _attn:
            grpo_kwargs["model_init_kwargs"]["attn_implementation"] = _attn
    else:
        _attn = optimal_attn_impl()
    # stop_sequences: TRL forwards generation_kwargs to the (vLLM) sampler, whose
    # SamplingParams.stop truncates each rollout at the requested delimiter — so the reward
    # sees the same completion the config intends, instead of generating to max_completion.
    if _t and _t.stop_sequences:
        grpo_kwargs["generation_kwargs"] = {"stop": list(_t.stop_sequences)}
    # advantage_clip>0 is the datums centered-advantage clamp; TRL has no advantage-value
    # clip knob (it clips the importance ratio), so honor the default (clip off ==
    # centered) and surface a note when a config asks for an explicit clamp.
    if _adv_clip > 0:
        print(f"[rl] advantage_clip={_adv_clip} recorded; TRL centers advantages (no value clip)")
    # num_iterations (the one promoted GRPO speed lever, measured 1.38x faster) is feature-detected
    # so an older TRL that lacks the field is simply skipped (GRPOConfig rejects unknown kwargs).
    # Generation dominates GRPO wall-clock, so reusing each rollout batch for 2 optimizer steps is
    # the cheapest large speedup; mu=2 is the standard GRPO config and TRL's importance-sampling
    # correction (on by default) keeps the step stable. (The GSPO/DAPO A/B levers were dropped: the
    # framework-scan in gpu-bench/RESEARCH_FINDINGS.md measured no robust win over baseline.)
    import dataclasses as _dc

    try:
        _grpo_fields = {f.name for f in _dc.fields(GRPOConfig)}
    except TypeError:
        _grpo_fields = set()  # not a dataclass on this TRL -> skip the feature-detected knob
    if "num_iterations" in _grpo_fields:
        grpo_kwargs["num_iterations"] = 2
        print("[rl] rollout amortization: num_iterations=2 (reuse each generation batch)")
    cfg = GRPOConfig(**grpo_kwargs)
    setup_seconds = time.time() - t_start
    heartbeat("rl_train_start", setup_seconds=setup_seconds)

    # VL checkpoints (Qwen3.5/3.6) train text-only: make TRL's colocated rollout
    # engine skip the vision tower (VRAM + 5090 PTX-compat; see the patch docstring).
    # Only relevant when vLLM drives rollouts; transformers generation uses the trainer
    # model (already text-only via the LoRA target/exclude config).
    if use_vllm:
        patch_vllm_language_model_only(model_id)
        # Install (but do NOT yet activate) the TRL->vLLM weight-sync name remap for Qwen3.5/3.6:
        # the trainer pushes ``model.*`` names but the VL engine's LM params live under
        # ``language_model.*``, so the first sync_weights() would raise without this. Activated
        # below, after the trainer + its initial checkpoint load are built.
        patch_vllm_lm_weight_sync(model_id)
    hb_cb = make_reward_heartbeat_callback()
    # Multi-turn / tool wiring (trl 1.6): tool envs hand TRL the tool callables so it runs the
    # tool-call loop natively; pure multi-turn envs hand TRL a rollout_func that drives the
    # env's own turn loop on the colocate engine (env_mask masks the non-model tokens).
    extra_trainer_kwargs: dict = {}
    tools = ACTIVE_ENV.tools() if is_tool_env else []
    # A tool env exposing NO tools would silently degrade to single-shot under tools=[]; drive
    # it through the rollout_func turn loop instead so it isn't mis-trained as single-turn.
    if is_tool_env and not tools:
        print("[rl][warn] tool env exposes no tools — using the multi-turn rollout_func path")
    use_rollout_func = is_multi_turn and not (is_tool_env and tools)
    require_vllm_for_rollout_func(use_rollout_func, use_vllm, model_id)
    if is_tool_env and tools:
        extra_trainer_kwargs["tools"] = tools
        print(f"[rl] tool env: handing {len(tools)} tool(s) to TRL's native tool loop")
    if use_rollout_func:
        from flash.engine.multiturn_rollout import (
            build_examples_index,
            build_rollout_func,
            index_collisions,
        )

        examples_by_key = build_examples_index(train, ACTIVE_ENV.prompt_messages)
        ncol = index_collisions(train, ACTIVE_ENV.prompt_messages)
        if ncol:
            print(
                f"[rl][warn] {ncol} duplicate prompt(s) collide in the reward index; the shared "
                "prompt scores against the last example's answer/info"
            )
        extra_trainer_kwargs["rollout_func"] = build_rollout_func(
            active_env=ACTIVE_ENV,
            tok=tok,
            examples_by_key=examples_by_key,
            max_completion=_max_completion,
            max_turns=getattr(ACTIVE_ENV, "max_turns", 10),
            temperature=_temperature,
            top_p=rl.sampling_top_p,
            stop=(list(_t.stop_sequences) if _t and _t.stop_sequences else None),
            thinking=THINKING,
            engine_max_len=vllm_max_len,
        )
        print("[rl] multi-turn env: driving the turn loop via rollout_func")
    trainer = GRPOTrainer(
        model=init_model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=reward_fn,
        peft_config=init_peft,
        processing_class=tok,
        callbacks=[hb_cb, make_checkpoint_upload_callback()],
        **extra_trainer_kwargs,
    )
    # Apply chalk's standalone fused kernels (rms_norm/swiglu/FLCE + RoPE/LoRA-delta/embedding) on
    # the module GRPOTrainer actually optimizes (trainer.model) — the fresh-LoRA path only passes
    # the model-id string to TRL, so trainer.model is the authoritative target.
    # Capture the install report so the engaged kernels land in metrics (active_kernels below).
    _chalk_report = install_chalk_kernels(getattr(trainer, "model", None))
    # The trainer (and its colocated vLLM engine + initial checkpoint load) is now built. Activate
    # the TRL->vLLM weight-sync name remap ONLY now (see patch_vllm_lm_weight_sync) so the initial
    # checkpoint load stayed untouched while the train-time syncs get remapped. No-op unless the VL
    # patch above was installed.
    if use_vllm:
        _LM_SYNC_REMAP_ON["on"] = True
        if is_vl_checkpoint(model_id):
            print("[vllm] LM weight-sync remap activated for training syncs")
    # Mid-run eval is intentionally NOT run during training: held-out evaluation happens on the
    # deploy/serving side (against the trained adapter), keeping training pure (no eval-phase cost
    # or eval-boundary stalls). Training streams only the per-step reward heartbeat.
    t_train = time.time()
    with _sdpa_cudnn_ctx(_attn):  # force cuDNN SDPA on sm120 (no-op otherwise)
        trainer.train(resume_from_checkpoint=resume_ckpt)
    train_wall = time.time() - t_train
    reward_history = list(getattr(hb_cb, "reward_history", []))
    # A GRPO run that finishes WITHOUT the reward callback ever firing (empty reward_history)
    # produced NO real training — the rollout scored nothing (e.g. vLLM generation silently
    # returning no completions, observed on RTX 5090 / sm120: ~1.4 s wall, empty reward + loss
    # curves, but the run otherwise "succeeds"). That is a FAILURE, not a success: a no-op run with
    # an unchanged adapter must not be reported as done — fail loudly so the operator/agent doesn't
    # trust it. (An env returning all-zero rewards still appends 0.0s, so an EMPTY history uniquely
    # means the reward path never ran.)
    _steps_run = int(getattr(trainer.state, "global_step", 0) or 0)
    # A resume that already reached the target steps legitimately performs ZERO new optimizer
    # steps: the previous worker uploaded the final checkpoint (and scored its rewards) but died
    # before writing metrics/DONE, so this worker's fresh hb_cb has an empty reward_history even
    # though the policy IS fully trained. Don't fail those — finalize from the resumed state. The
    # no-op guard below is only for a run that genuinely trained nothing (no resume, or the resume
    # didn't reach the target steps).
    _resumed_complete = _grpo_resume_already_complete(resume_ckpt, steps, _steps_run)
    if _grpo_is_no_op_failure(reward_history, resume_ckpt, steps, _steps_run):
        raise RuntimeError(
            f"GRPO scored no reward in {train_wall:.1f}s over {_steps_run} step(s) — the rollout "
            "produced no completions, so the policy was never actually trained. Failing loudly "
            "instead of reporting a no-op run as done (seen on RTX 5090/sm120 vLLM rollout)."
        )
    if not reward_history and _resumed_complete:
        print(
            f"[resume] no new reward in this worker but resumed checkpoint already reached "
            f"{_steps_run}/{steps} step(s) — finalizing the completed policy instead of failing."
        )
    adapter_dir = f"{out_dir}/adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    hf_upload_folder(adapter_dir, "adapter", required=True)
    heartbeat("rl_trained", train_wall=train_wall)

    # Upper bound on generated tokens: completions actually optimized (the intended
    # prompts_per_step after the batch fix) x the max completion length. Over-counts (most
    # completions are shorter); reported as an upper bound, used only for a rough throughput.
    gen_tokens = steps * batching["unique_prompts_per_step"] * group_size * _max_completion
    write_train_meta(
        phase="rl",
        adapter_dir=adapter_dir,
        model_id=model_id,
        train_wall=train_wall,
        setup_seconds=setup_seconds,
        train_tokens=0,
        generated_tokens=gen_tokens,
        notes={
            "steps": steps,
            "resumed": bool(resume_ckpt),
            "download_seconds": download_seconds,
            "hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
            "reward_history": reward_history,
            "loss_curve": _metric_curve(trainer, "loss"),
            # Which chalk gap-filling kernels actually ENGAGED (None = chalk not installed or every
            # kernel fell back) — verifies the chalk stack on a GRPO run without the console.
            "chalk_kernels": active_kernels(_chalk_report) or None,
            **wandb_run_info(),
            "gen_tokens_is_upper_bound": True,
            "thinking": THINKING,
            "max_completion_len": _max_completion,
            "prompts_per_step": batching["unique_prompts_per_step"],
            "generations_per_step": batching["generations_per_step"],
            "group_size": group_size,
            "per_device_train_batch_size": batching["per_device_train_batch_size"],
            "gradient_accumulation_steps": batching["gradient_accumulation_steps"],
            "grpo_recipe": {
                "lr_scheduler": "constant",
                "beta": _kl_beta,
                "scale_rewards": "none",
                "loss_type": "dr_grpo",
                "temperature": _temperature,
                "advantage_clip": _adv_clip,
                "thinking_length_penalty_coef": _think_penalty,
                "init_from_adapter": JOB_SPEC.train.init_from_adapter if JOB_SPEC else "",
            },
        },
    )
    free_gpu(trainer)


# ---------------------------------------------------------------------------
# Completion: train phase writes metrics.json + the DONE sentinel (see _finalize).
# ---------------------------------------------------------------------------


def write_train_meta(
    phase, adapter_dir, model_id, train_wall, setup_seconds, train_tokens, generated_tokens, notes
):
    meta = {
        "phase": phase,
        "adapter_dir": adapter_dir,
        "model_id": model_id,
        "train_wall": train_wall,
        "setup_seconds": setup_seconds,
        "train_tokens": train_tokens,
        "generated_tokens": generated_tokens,
        "notes": notes or {},
    }
    with open("/tmp/train_meta.json", "w") as f:
        json.dump(meta, f)
    hf_upload_file("/tmp/train_meta.json", "train_meta.json")
    heartbeat(
        f"{phase}_train_done",
        **{k: meta[k] for k in ("train_wall", "train_tokens", "generated_tokens")},
    )
    # Finalize directly from the training phase: build the run-metrics record (training
    # metrics only — loss/reward are streamed by the trainer; reward_history is in notes)
    # and write the completion sentinel. There is no separate eval phase.
    m = RunMetrics(
        # Substrate the worker actually ran on. Each provider's launcher sets FLASH_ARM
        # in the worker env (runpod -> "runpod", vast -> "vast"); default to "runpod" only
        # when unset so the persisted metrics correctly attribute the compute backend.
        arm=os.environ.get("FLASH_ARM", "runpod"),
        phase=phase,
        seed=SEED,
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
            "thinking": THINKING,
            "train_wall": train_wall,
            "model_id": model_id,
            "environment": ACTIVE_ENV.id,
            "job_spec": JOB_SPEC.to_dict() if JOB_SPEC else None,
        },
    )
    _finalize(m)


def _download_adapter(adapter_prefix: str | None) -> str | None:
    """Download an init_from_adapter LoRA to /tmp/evdl/<prefix>/adapter and return its dir.

    Two forms of ``adapter_prefix``:
      * ``"<prefix>"``            -> read from THIS run's own artifact repo (HF_REPO).
      * ``"<owner>/<repo>:<prefix>"`` -> CROSS-REPO warm-start: read the SFT adapter from
        another run's managed artifact repo. Required since hf_repo is now a per-run managed
        repo (Freesolo-Co/flashrun-<run_id>), so an SFT adapter never lives in the GRPO run's
        own repo. The control-plane HF_TOKEN can read sibling managed repos.
    """
    if not adapter_prefix:
        return None
    if ":" in adapter_prefix:
        repo, prefix = adapter_prefix.split(":", 1)
    else:
        repo, prefix = HF_REPO, adapter_prefix
    if not (repo and prefix):
        return None
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=[f"{prefix}/adapter/*"],
        local_dir="/tmp/evdl",
        token=os.environ.get("HF_TOKEN"),
    )
    adir = os.path.join("/tmp/evdl", prefix, "adapter")
    return adir if os.path.isdir(adir) else None


def _finalize(metrics: RunMetrics):
    metrics.save("/tmp/metrics.json")
    # Required: a swallowed upload would make the control plane fail/retry a finished run.
    hf_upload_file("/tmp/metrics.json", "metrics.json", required=True)
    # DONE sentinel so the controller knows it's safe to tear down
    with open("/tmp/DONE", "w") as f:
        f.write(str(time.time()))
    hf_upload_file("/tmp/DONE", "DONE", required=True)
    heartbeat("done")
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
                heartbeat("already_done")
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
        # Not a DONE re-delivery -> this worker will train. These must run before any model import.
        heartbeat("boot")  # FIRST: signal liveness before any slow setup (Hopper fla/tilelang install)
        # Boot guard: a MIG slice / NVML-restricted host (RunPod has fulfilled a full-GPU request
        # with a MIG partition) makes PyTorch's allocator NVML-assert mid-backward. Detect it now
        # and raise the RETRIABLE infra error so the control plane resubmits on a fresh worker,
        # instead of crashing opaquely deep in training as a non-retried job_failed.
        assert_usable_gpu()
        # Hopper: enable fla+tilelang GDN fast path (see perf.py). AFTER the GPU guard so a MIG /
        # unusable host bails RETRIABLE without first paying the multi-minute tilelang/fla install;
        # still before any model import (transformers gates GDN on is_fla_available() at load).
        _ensure_fla_fastpath_on_hopper()
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
        # Structured retry signal both pollers read: infra failure -> retry on a fresh worker;
        # exclude_class -> the GPU class is at fault (MIG), so re-allocate OFF it.
        retriable = isinstance(e, RetriableInfraError)
        exclude_class = bool(getattr(e, "exclude_class", False))
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
        hb_flags = {"retriable": retriable, "exclude_class": exclude_class}
        try:
            heartbeat(f"error_{RUN_MODE}", error=str(e)[:500], **hb_flags, diag=gpu_diagnostics())
        except Exception:
            heartbeat(f"error_{RUN_MODE}", error=str(e)[:500], **hb_flags)
        # keep container alive briefly so logs flush, then exit non-zero -> restart
        time.sleep(10)
        raise


if __name__ == "__main__":
    main()



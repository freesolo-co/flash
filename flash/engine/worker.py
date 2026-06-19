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
from flash.engine.chalk_kernels import install_chalk_kernels
from flash.engine.recipe import RECIPE
from flash.envs.registry import load_environment
from flash.spec import load_job_spec_from_env

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
                raise RuntimeError(f"required upload of {repo_subpath!r} failed: {e}") from e
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
    import threading

    from transformers import TrainerCallback

    lock = threading.Lock()

    class _CheckpointUpload(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # Multi-GPU (FSDP / accelerate) runs on_save on EVERY rank; only the single global
            # main process may upload, or ranks race on the same HF commit and one rank's
            # delete_patterns can wipe another rank's just-uploaded checkpoint. Single-GPU runs
            # are always world-process-zero, so this is a no-op there.
            if not state.is_world_process_zero:
                return
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
# FLASH_HEARTBEAT_MIN_S (default 60s); every other stage — including milestones and the
# terminal done/already_done — always commits so the control plane never misses a transition.
# The local file + stdout line are always written regardless.
_HB_LAST_UPLOAD = 0.0


def _hb_min_interval_s() -> float:
    """The rl_step heartbeat-upload throttle, in seconds. Operators raise FLASH_HEARTBEAT_MIN_S
    to stay under HuggingFace's 128 commits/hour-per-repo limit when several concurrent GRPO runs
    share one HF_REPO; default 60s. A non-positive or unparseable value falls back to 60s."""
    raw = os.environ.get("FLASH_HEARTBEAT_MIN_S")
    if raw is None:
        return 60.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 60.0
    return value if value > 0 else 60.0


_HB_MIN_INTERVAL_S = _hb_min_interval_s()
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


def _patch_peft_weight_converter_compat() -> None:
    """peft 0.19.1 x transformers 5.6-5.10: make MoE adapter loading work.

    peft's ``build_peft_weight_mapping`` reconstructs transformers ``WeightConverter``
    objects passing ``distributed_operation=`` / ``quantization_operation=`` — kwargs
    the WeightConverter in transformers <5.11 doesn't accept (init=False dataclass
    fields), so loading a LoRA adapter onto any arch WITH weight conversions dies with
    ``TypeError: unexpected keyword argument 'distributed_operation'`` (observed on a
    weight-converting checkpoint eval). The
    worker can't take transformers>=5.11 (vllm 0.19.1 compat), so accept-and-drop
    unknown kwargs; on a single GPU those fields are unused. No-op once signatures
    match.
    """
    import inspect

    try:
        from transformers import core_model_loading as cml
    except Exception:  # pragma: no cover - older stacks have no converter module
        return
    converter = getattr(cml, "WeightConverter", None)
    if converter is None or getattr(converter, "_flash_compat", False):
        return
    accepted = set(inspect.signature(converter.__init__).parameters)
    if "distributed_operation" in accepted:
        return
    orig_init = converter.__init__

    def _compat_init(self, *args, **kwargs):
        dropped = [k for k in kwargs if k not in accepted]
        for k in dropped:
            kwargs.pop(k)
        orig_init(self, *args, **kwargs)

    converter.__init__ = _compat_init
    converter._flash_compat = True
    print("[compat] WeightConverter patched (peft<->transformers signature drift)")


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
# Module-path segments that must never receive LoRA on natively-multimodal checkpoints
# trained text-only: the vision tower / projector / MTP head. Critically, adapters that
# DO touch them cannot be loaded by vLLM in text-only (language_model_only) serving —
# its LoRA loader rejects "unexpected modules" (observed with Qwen3.5-2B).
_VL_EXCLUDE_SEGMENTS = ("visual", "vision_tower", "multi_modal_projector", "mtp")


def lora_exclude_modules(model_id: str) -> str | None:
    """Regex (peft fullmatch semantics) excluding vision-tower modules from LoRA.

    Returns None when no exclusion is needed (pure text architectures). NOTE: peft's
    list-form exclude_modules uses suffix matching (like target_modules), which does
    NOT match leaf modules under 'visual.*' — a regex string is required.
    """
    excludes = {
        "qwen3_5": _VL_EXCLUDE_SEGMENTS,
        "qwen3_5_moe": _VL_EXCLUDE_SEGMENTS,
        "qwen3_6": _VL_EXCLUDE_SEGMENTS,
    }
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", "") or ""
    except Exception as e:
        print("lora_exclude_modules: config probe failed:", e)
        return None
    segments = excludes.get(model_type)
    if not segments:
        return None
    alt = "|".join(segments)
    return rf"(^|.*\.)({alt})(\..*|$)"


def is_vl_checkpoint(model_id: str) -> bool:
    """True for natively-multimodal checkpoints we train/serve text-only (Qwen3.5/3.6)."""
    return bool(lora_exclude_modules(model_id))


def vllm_language_model_only_kwargs(model_id: str) -> dict:
    """Engine kwargs to skip the vision tower for VL checkpoints (vLLM >= 0.19).

    Besides wasting VRAM, the vision tower's attention path hardcodes vLLM's bundled
    flash-attn, whose PTX needs a newer driver JIT than many RTX 5090 hosts have
    ("PTX compiled with unsupported toolchain") — text-only loading sidesteps it and
    is the officially supported way to run Qwen3.5 as a pure LLM.
    """
    return {"language_model_only": True} if is_vl_checkpoint(model_id) else {}


def _attn_impl_for_capability(major: int, minor: int) -> str | None:
    """Map a CUDA compute capability to the trainer ``attn_implementation``.

    Attention uses PyTorch SDPA (its flash/efficient backend is already selected automatically
    on Ampere/Ada/Hopper) — the HF Kernels-Hub FA path is disabled because the torch2.10-
    compatible ``kernels`` versions break transformers' import. So:
      sm120 (Blackwell consumer 5090/RTX Pro) -> "sdpa" (forced to the cuDNN backend at train
      time — its default SDPA can fall to the slow math kernel); all other archs -> None (let
      transformers pick SDPA, which already flash-backs on Ampere/Ada/Hopper). The big LoRA
      win comes from the Liger fused kernels, not the attention path. Pure function (no torch)
      so it's unit-testable on CPU; override the whole thing with FLASH_ATTN_IMPL.
    """
    if major == 12:  # Blackwell consumer: force cuDNN SDPA (avoid the math fallback)
        return "sdpa"
    return None


def _flash_attn_available() -> bool:
    """True when the ``flash_attn`` wheel is importable (our worker image builds it from source).

    Gates the packing default: TRL's ``packing_strategy='bfd'`` produces flattened/padding-free
    batches whose example boundaries are carried by ``position_ids`` and enforced ONLY by an
    attention impl that honors them (FlashAttention-2 varlen / flex_attention). Under plain SDPA,
    packed examples attend ACROSS boundaries (silent quality loss). find_spec only — no import side
    effects (and no CUDA init)."""
    try:
        import importlib.util

        return importlib.util.find_spec("flash_attn") is not None
    except Exception:
        return False


def optimal_attn_impl() -> str | None:
    """Best ``attn_implementation`` for the live GPU (None = leave transformers' default)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
    except Exception as e:
        print("optimal_attn_impl probe failed:", e)
        return None
    impl = _attn_impl_for_capability(major, minor)
    if impl:
        print(f"[attn] sm{major}{minor} -> attn_implementation={impl}")
    return impl


# Liger's fused linear cross-entropy is a MEMORY optimization (it never materializes the fp32
# [B,T,vocab] logits), not a fixed-batch speed win. PR #174 ledger: on a 1B model at fixed batch
# it is a measured NET LOSS on EVERY arch (min-of-3: A100 0.86x, H100 0.90x, RTX 3090 0.78x,
# RTX 4090 0.83x, RTX 5090 0.79x) — the per-step Triton overhead isn't repaid because the small
# model's logits don't dominate memory. Its value appears on LARGE models (lets a bigger batch
# fit / avoids OOM). So gate by estimated model size.
_LIGER_MIN_PARAMS = 3e9  # ~3B; 1B-class models measured net-negative -> Liger off below this


def _estimate_params(cfg) -> float:
    """Rough param count from a HF config: embeddings (+untied lm_head) + transformer blocks.
    For multimodal checkpoints (e.g. Qwen3.5-VL) the LM dims live under ``text_config`` — read it
    when the top-level dims are absent, else the gate underestimates and wrongly disables the
    memory path (GC/Liger) for the 4B/9B tiers."""
    tc = getattr(cfg, "text_config", None)
    src = cfg if getattr(cfg, "hidden_size", 0) else (tc or cfg)
    h = getattr(src, "hidden_size", 0) or 0
    v = getattr(src, "vocab_size", 0) or getattr(cfg, "vocab_size", 0) or 0
    n = getattr(src, "num_hidden_layers", 0) or 0
    tied = getattr(src, "tie_word_embeddings", getattr(cfg, "tie_word_embeddings", False))
    emb = v * h * (1 if tied else 2)
    blocks = n * 12 * h * h  # ~12 h^2 per transformer block (attn + MLP)
    return float(emb + blocks)


def _liger_default_for_model(model_id: str) -> bool:
    """Default Liger ON only for models large enough that fused-CE's memory win pays off
    (≥ _LIGER_MIN_PARAMS, ~3B). 1B-class models measured net-negative -> default OFF."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return _estimate_params(cfg) >= _LIGER_MIN_PARAMS
    except Exception as e:
        print("liger model-size probe failed (default off):", e)
        return False


def liger_on(default_on: bool) -> bool:
    """Whether to enable a Liger kernel path. ``default_on`` is the model-size decision (on only
    for models large enough that fused-CE's memory win pays off; 1B-class is a measured net loss).
    Even when on, require a CUDA GPU AND that ``liger_kernel`` is importable — the local
    ``flash[gpu]`` extra doesn't ship it, so blindly setting use_liger_kernel would crash a
    local GPU run. No GPU / absent -> off."""
    if not default_on:
        return False
    try:
        import importlib.util

        import torch

        return bool(
            torch.cuda.is_available() and importlib.util.find_spec("liger_kernel") is not None
        )
    except Exception:
        return False


def setup_perf_backends() -> None:
    """Universal, arch-agnostic throughput knobs — safe on every CUDA arch, no JIT/compile cost.

    - TF32 for fp32 matmuls/cuDNN (Ampere+): the residual fp32 ops in a bf16 LoRA run (some
      norms, the optimizer's fp32 master step, any fp32 GEMM) run on the TF32 tensor cores at
      ~no accuracy cost. No-op on pre-Ampere.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.set_float32_matmul_precision("high")  # TF32 for fp32 matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[perf] TF32 matmul/cuDNN enabled")
    except Exception as e:
        print("setup_perf_backends skipped:", e)


def finalize_alloc_conf_for_sleep() -> None:
    """Sync the CUDA allocator conf with the worker's RESOLVED vLLM sleep default.

    The launcher (providers/*/train.py build_worker_env) must pick PYTORCH_ALLOC_CONF before this
    process starts, but it can't always know the GRPO sleep decision: for a small model with
    RL_VLLM_SLEEP unset the worker resolves sleep OFF (the speed default), yet the launcher
    conservatively assumes sleep ON and picks the non-expandable conf (safe, but fragments a long
    colocate run). When the launcher cedes the decision (it sets FLASH_ALLOC_AUTO=1 — only when
    it applied a DEFAULT, never an operator override), we resolve the same sleep default here (we
    have the model config + GPU) and, if sleep is OFF, switch to expandable_segments — which only
    crashes WITH sleep on, a case we've just ruled out. PYTORCH_ALLOC_CONF is read lazily at the
    first CUDA allocation, so this must run before any allocation (it does — called at boot)."""
    if os.environ.get("FLASH_ALLOC_AUTO") != "1":
        return
    try:
        model_id = os.environ.get("BENCH_HF_MODEL", "")
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


def _remove_fla_from_disk() -> tuple[list[str], bool]:
    """Physically delete every importable ``fla`` package dir from the worker's REAL sys.path.

    Loops until ``find_spec('fla')`` is clean (removing one copy can expose another further down
    the path) and invalidates import caches so transformers' is_fla_available() probe sees it
    gone. ``pip uninstall`` alone is unreliable here — it targets one site-packages but the base
    image bakes ``fla`` into another dir on the path (and can report success while leaving the
    package dir). Returns ``(removed_dirs, still_importable)``. Used by the Hopper auto-drop.
    """
    import importlib
    import importlib.util
    import shutil

    removed: list[str] = []
    for _ in range(6):  # a few passes: removing one copy can reveal another earlier on the path
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("fla")
        if spec is None:
            break
        # Resolve the package directory (submodule_search_locations for a package, else the file dir).
        locs = list(getattr(spec, "submodule_search_locations", None) or [])
        if not locs and spec.origin:
            locs = [os.path.dirname(spec.origin)]
        progressed = False
        for loc in locs:
            if loc and os.path.isdir(loc) and os.path.basename(loc.rstrip("/")) == "fla":
                try:
                    shutil.rmtree(loc)
                    removed.append(loc)
                    progressed = True
                except Exception as e:
                    print(f"[fla] could not remove {loc}: {e}", flush=True)
        if not progressed:
            break
    importlib.invalidate_caches()
    return removed, importlib.util.find_spec("fla") is not None


# Long-context runs are memory-bound (activations + vLLM KV cache scale with sequence length), so
# they need the memory features even on a SMALL model — PR #174 measured a 1B model OOM on GRPO at
# 4096 ctx in speed mode, but it fits in memory mode. So "memory mode" = large model OR long ctx.
_LONG_CONTEXT_TOKENS = 2048


def _memory_mode(model_id: str, max_length: int = 0) -> bool:
    """Whether to default the memory-saving features (Liger, grad-checkpointing, vLLM sleep) ON:
    a large model (fused-CE memory win) OR a long context (activations/KV dominate). Small model +
    short context -> off (optimize for speed)."""
    if max_length and max_length >= _LONG_CONTEXT_TOKENS:
        return True
    return _liger_default_for_model(model_id)


def grad_checkpointing_on(model_id: str, max_length: int = 0) -> bool:
    """Gradient checkpointing recomputes the forward in backward (~25% slower) to save activation
    memory — a MEMORY feature, not speed. ON for large models / long context that need the
    headroom; OFF for small+short runs that fit without it (the speed win)."""
    return _memory_mode(model_id, max_length)


def fused_optim_name() -> str:
    """TRL/HF ``optim`` value: 8-bit paged AdamW (bitsandbytes int8 optimizer state paged to host
    RAM). It fits a smaller/cheaper GPU and is the better default across the catalog."""
    return "paged_adamw_8bit"


def _reset_peak_gpu() -> None:
    """Reset the CUDA peak-memory counter so a subsequent ``_peak_gpu_gb`` measures only the work
    that follows (e.g. the train loop, isolating the optimizer-state A/B from setup/model load)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_gpu_gb() -> float:
    """Peak torch-allocated CUDA memory (GB) since the last reset; 0.0 if no CUDA. Note: bnb paged
    8-bit optimizer state lives in unified/managed memory outside torch's caching allocator and is
    NOT counted here — so this OVERSTATES the 8-bit saving. _GpuPeakSampler measures the true
    device footprint (incl. bnb managed pages) for the honest A/B number."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return 0.0


class _GpuPeakSampler:
    """Background sampler of true device memory (GB) = (total - free) from cuda.mem_get_info, which
    DOES include bitsandbytes managed/paged optimizer pages while they're GPU-resident (torch's
    max_memory_allocated does not). This is the honest peak for the fp32-vs-8-bit optimizer A/B."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak_used = 0
        self._stop = False
        self._thread = None

    def _run(self):
        import torch

        while not self._stop:
            try:
                free, total = torch.cuda.mem_get_info()
                self.peak_used = max(self.peak_used, total - free)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        try:
            import threading

            import torch

            if not torch.cuda.is_available():
                return self
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            pass
        return self

    def stop_gb(self) -> float:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2)
        return round(self.peak_used / 1e9, 3)


def loraplus_optimizer_cls(optim_name: str):
    """Optimizer class for the LoRA+ ``create_optimizer`` override (returns ``(cls, extra_kwargs)``).

    The LoRA+ override has to *build* the optimizer itself (PEFT splits the LoRA A/B matrices into
    separate param groups with different LRs), so it cannot inherit TRL's ``optim=`` string — it has
    to choose a concrete class. Historically it always built a full-precision ``torch.optim.AdamW``,
    which silently discarded the catalog's ``paged_adamw_8bit`` setting whenever LoRA+ was on.

    PEFT's ``create_loraplus_optimizer`` accepts ANY ``optimizer_cls`` — including bitsandbytes 8-bit
    optimizers (it registers embedding overrides with bnb's ``GlobalOptimManager`` to keep them
    32-bit) — so LoRA+ and the 8-bit paged optimizer state coexist. An ``8bit`` ``optim`` value
    (the fleet default; ``fused_optim_name`` -> ``paged_adamw_8bit``) selects
    ``bnb.optim.PagedAdamW8bit``; a non-8-bit ``optim`` keeps fp32 AdamW. This simply mirrors the
    configured ``optim`` — there is no separate toggle: an on-GPU A/B (Qwen3.5-4B SFT, rank-128
    LoRA, same seed/data/init) measured the 8-bit paged state at -75% optimizer memory
    (1359 -> 346 MB) and -0.72 GB peak with NO convergence penalty (final loss 10.64 vs 11.16 from
    an identical start), so it's unconditionally the default wherever ``optim`` is 8-bit. Falls
    back to fp32 AdamW only if bitsandbytes is missing."""
    import torch as _torch

    # case-insensitive + str-safe: TRL normalizes optim to an OptimizerNames enum whose str() is
    # "OptimizerNames.PAGED_ADAMW_8BIT" (uppercase), so a bare `"8bit" in optim_name` would miss it.
    if "8bit" in str(optim_name or "").lower():
        try:
            import bitsandbytes as bnb

            return bnb.optim.PagedAdamW8bit, {}
        except Exception as e:  # bnb missing / no CUDA build -> safe fp32 fallback
            print(f"[lora+] bitsandbytes 8-bit optimizer unavailable ({e}); using fp32 AdamW")
    return _torch.optim.AdamW, {}


def wandb_report_to() -> list[str]:
    """TRL/HF ``report_to`` targets. Restores the W&B logging the legacy freesolo training path had
    but the flash migration dropped: report to W&B whenever WANDB_API_KEY is present. No key -> []
    (silent, the metrics.json artifact is still the source of truth). Sets a default project so runs
    land in one place."""
    if not os.environ.get("WANDB_API_KEY"):
        return []
    import importlib.util

    if importlib.util.find_spec("wandb") is None:
        print("[wandb] WANDB_API_KEY set but the wandb package is missing; skipping W&B logging")
        return []
    os.environ.setdefault("WANDB_PROJECT", "flash")
    return ["wandb"]


def wandb_run_name() -> str:
    """Stable, human-readable W&B run name tying the dashboard run to the Flash run id."""
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


def _sdpa_cudnn_ctx(attn_impl: str | None):
    """Context forcing the cuDNN SDPA backend (real Blackwell-consumer kernels) when we fell
    back to plain SDPA on sm120; a no-op context otherwise. Best-effort."""
    if attn_impl != "sdpa":
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # Priority-ordered: prefer the fast cuDNN/flash/efficient kernels, but ALWAYS include MATH
        # as the final fallback. Restricting to only [CUDNN, EFFICIENT] makes sm120 GRPO crash with
        # "RuntimeError: No available kernel" when neither has a kernel for the completion-batch
        # attention shape (MEASURED: Qwen3.5 GRPO on RTX 5090). MATH is universal, so the candidate
        # set is never empty; set_priority keeps cuDNN first whenever it CAN serve the shape (SFT
        # fast path unchanged), only falling through for the shapes cuDNN/efficient reject.
        return sdpa_kernel(
            [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
            set_priority=True,
        )
    except Exception as e:
        print("[attn] cuDNN SDPA backend unavailable, using default SDPA:", e)
        return contextlib.nullcontext()


def patch_vllm_language_model_only(model_id: str) -> bool:
    """Force ``language_model_only=True`` on vLLM engines created by third-party code
    (TRL's colocated GRPO rollout engine) for VL checkpoints. Returns True if patched."""
    extra = vllm_language_model_only_kwargs(model_id)
    if not extra:
        return False
    try:
        import vllm

        if getattr(vllm.LLM.__init__, "_flash_lmo_patched", False):
            return True
        orig = vllm.LLM.__init__

        def patched(self, *args, **kwargs):
            kwargs.setdefault("language_model_only", True)
            return orig(self, *args, **kwargs)

        patched._flash_lmo_patched = True
        vllm.LLM.__init__ = patched
        print(f"[vllm] language_model_only patch active for {model_id}")
        return True
    except Exception as e:
        print("patch_vllm_language_model_only warn:", e)
        return False


def make_lora(model_id: str | None = None):
    """LoRA config. We target 'all-linear' (every nn.Linear) rather than a hardcoded
    q/k/v/o list: it is architecture-agnostic, so the same recipe works for the dense
    default (Qwen3-4B-Instruct-2507) and for newer models with extra projection
    types (e.g. the Qwen3.5 hybrid Gated-DeltaNet) without missing any adapters.
    For natively-multimodal checkpoints the vision tower is excluded (see
    ``lora_exclude_modules``)."""
    from peft import LoraConfig

    # PEFT target_modules accepts the special string "all-linear" OR a LIST of module-name
    # suffixes. A comma-separated env (e.g. "q_proj,k_proj,v_proj,o_proj" to adapt attention only
    # and leave the MoE experts frozen) MUST be split into a list — else PEFT treats the whole
    # string as ONE module name and raises "Target modules ... not found in the base model".
    _parts = [
        t.strip() for t in os.environ.get("LORA_TARGETS", "all-linear").split(",") if t.strip()
    ]
    # "all-linear" is a PEFT SPECIAL string, not a module name — keep it as a string (incl. when a
    # stray trailing comma made it the sole element, e.g. "all-linear," -> ["all-linear"]). Any
    # real multi-module value becomes a list of suffixes.
    targets = "all-linear" if (not _parts or _parts == ["all-linear"]) else _parts
    rank = JOB_SPEC.train.lora_rank if JOB_SPEC else RECIPE.lora.rank
    alpha = JOB_SPEC.train.lora_alpha if JOB_SPEC else RECIPE.lora.alpha
    kwargs = {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": RECIPE.lora.dropout,
        "target_modules": targets,
        "task_type": "CAUSAL_LM",
    }
    # Adapter initialization (convergence lever, always-on: measured -35% train loss in A/B
    # (gpu-bench)). PiSSA inits A/B from the base weight's top singular vectors (fast SVD, ~seconds)
    # so LoRA converges faster + to higher quality than the default zero-B init (arXiv 2404.02948).
    # NOTE: PiSSA mutates the effective base, so the saved adapter is a PiSSA-residual unless
    # converted — fine for our train+eval+serve-same-stack flow.
    # PiSSA's SVD-based init fails inside the DISAGGREGATED trainer (which is CVD-pinned to a
    # non-zero device so the vLLM server can own device 0): peft's pissa_init hits
    # "set_data ... incompatible tensor type" there. FLASH_LORA_INIT=default falls back to the
    # standard LoRA init (Kaiming-A / zero-B) — the disaggregated worker sets this. The init METHOD
    # does not affect rollout/step THROUGHPUT, which is the whole point of the disaggregated path.
    # `or` (not a get-default): a present-but-blank FLASH_LORA_INIT must fall back too, else
    # init_lora_weights="" reaches PEFT as an invalid value instead of a real init method.
    _lora_init = os.environ.get("FLASH_LORA_INIT") or "pissa_niter_16"
    # PiSSA's SVD init requires an UNQUANTIZED base ("Please initialize PiSSA under
    # float32/float16/bfloat16"); it raises on a 4-bit (qlora) base at adapter creation. Force
    # standard init for qlora models regardless of the configured default, so the 4-bit trainers
    # (Qwen3.5-9B, Qwen3.6-35B-A3B) don't crash — this hit 9B colocate (asb-esc-q9b-coloB).
    if (
        model_id
        and model_quant(model_id) == "4bit-qlora"
        and _lora_init.lower() not in ("default", "standard", "plain", "true")
    ):
        print(f"[lora] {model_id} is 4-bit qlora; PiSSA needs an unquantized base -> forcing standard init")
        _lora_init = "default"
    if _lora_init.lower() in ("default", "standard", "plain", "true"):
        kwargs["init_lora_weights"] = True
        print("[lora] init_lora_weights=default (standard LoRA init; pissa disabled)")
    else:
        kwargs["init_lora_weights"] = _lora_init
        print(f"[lora] init_lora_weights={_lora_init}, rsLoRA scaling enabled")
    # rsLoRA scaling (convergence lever, always-on: measured -47% train loss in A/B (gpu-bench)).
    kwargs["use_rslora"] = True
    if model_id and targets == "all-linear":
        exclude = lora_exclude_modules(model_id)
        if exclude:
            kwargs["exclude_modules"] = exclude
            print(f"[lora] excluding modules for {model_id}: {exclude}")
    return LoraConfig(**kwargs)


def model_quant(model_id: str) -> str:
    """Quantization tier for this model: catalog entry > FLASH_QUANT env > bf16."""
    env_q = os.environ.get("FLASH_QUANT")
    if env_q:
        return env_q
    try:
        from flash.catalog import MODELS

        info = MODELS.get(model_id)
        if info is not None:
            return info.quant
    except Exception as e:
        print("model_quant: catalog probe failed:", e)
    return "bf16"


def qlora_model_init_kwargs() -> dict:
    """Model-load kwargs for the 4-bit QLoRA tier: bf16 compute + a bitsandbytes NF4
    (double-quant) config so the frozen base loads in 4-bit and only the LoRA adapter trains."""
    import torch
    from transformers import BitsAndBytesConfig

    return {
        "dtype": torch.bfloat16,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    }


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
    train = ACTIVE_ENV.dataset("train")
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

    default_epochs = (
        JOB_SPEC.train.epochs
        if JOB_SPEC and JOB_SPEC.train.epochs is not None
        else RECIPE.sft.num_epochs
    )
    epochs = int(os.environ.get("SFT_EPOCHS", str(default_epochs)))
    # SDK [train] knobs override the recipe default; an operator env var still wins last.
    _t = JOB_SPEC.train if JOB_SPEC else None
    per_device_bs = int(os.environ.get("SFT_PER_DEVICE_BS", "4"))
    # batch_size is the GLOBAL/effective batch: grad-accum is sized to reach it. Cap the
    # per-device micro-batch at the target (so a target < per_device doesn't overshoot) and
    # use CEIL division so the realized global batch is never BELOW the requested one (floor
    # would undershoot when target isn't a multiple of per_device, e.g. 16/6 -> 12).
    effective_batch = (
        _t.batch_size if _t and _t.batch_size is not None else RECIPE.sft.effective_batch
    )
    per_device_bs = max(1, min(per_device_bs, effective_batch))
    grad_accum = max(1, -(-effective_batch // per_device_bs))
    sft_lr = _t.learning_rate if _t and _t.learning_rate is not None else RECIPE.sft.learning_rate
    sft_max_len = (
        _t.max_length
        if _t and _t.max_length is not None
        else (RECIPE.sft.max_seq_len_thinking if THINKING else RECIPE.sft.max_seq_len)
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
    # start (~20 min, fragile) — it is NOT in the worker image. So _fa_ok is False on the current
    # stack and packing is effectively unavailable until flash-attn is baked into a prebuilt image.
    # Default: packing ON when FA2 is importable; else SKIP (don't silently cross-contaminate).
    # SFT_PACKING=0 disables; SFT_PACKING=1 forces (bfd under SDPA, with the contamination warning).
    _packing_env = os.environ.get("SFT_PACKING")
    _want_packing = (_packing_env or "1") not in ("0", "false", "False")
    _packing_forced = _packing_env not in (None, "")
    _fa_ok = _flash_attn_available()
    if _want_packing and _fa_ok:
        cfg_kwargs["packing"] = True
        print("[sft] example packing enabled (FA2 varlen; SFT_PACKING=0 to disable)")
    elif _want_packing and _packing_forced:
        cfg_kwargs["packing"] = True
        print(
            "[sft] WARNING: packing forced without FA2 — 'bfd' boundaries need varlen; "
            "examples may cross-contaminate under SDPA. Set SFT_PACKING=0 to disable."
        )
    elif _want_packing:
        print(
            "[sft] packing SKIPPED: no boundary-correct attn backend (flash-attn absent on torch "
            "2.10). Bake flash-attn into the worker image to enable FA2 varlen packing."
        )
    # Liger fused CE/RMSNorm/RoPE kernels, gated by model size (_memory_mode). The fused linear
    # cross-entropy is the big large-vocab (Qwen ~152k) memory/throughput win.
    if liger_on(_memory_mode(model_id, sft_max_len)):
        cfg_kwargs["use_liger_kernel"] = True
        print("[sft] liger fused kernels enabled")
    _attn = optimal_attn_impl()  # arch-aware FlashAttention (Kernels Hub) / SDPA
    # Packing correctness: 'bfd' packed batches are boundary-correct ONLY under a varlen attn.
    # With FA2 importable force flash_attention_2 — a pure win over the SDPA default which would
    # cross-contaminate packed examples.
    if cfg_kwargs.get("packing") and _fa_ok:
        _attn = "flash_attention_2"
        print("[sft] attn_implementation=flash_attention_2 (packing boundary-correct varlen)")
    quant = model_quant(model_id)
    if quant == "4bit-qlora":
        # QLoRA tier: 4-bit NF4 base + bf16 LoRA adapters (e.g. Qwen3.5-9B on a 5090).
        _patch_peft_weight_converter_compat()  # adapter (re)load, e.g. ckpt resume
        mik = qlora_model_init_kwargs()
        print(f"[sft] loading {model_id} in 4-bit (QLoRA tier)")
    else:
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

    # Install any opt-in chalk kernels (selected via FLASH_* flags) before TRL builds the model, so the
    # class/function-level patches (LoRA delta, fused MLP/QKV, RoPE) apply to it. No-op unless
    # a FLASH_* kernel flag is set and freesolo-chalk is installed.
    install_chalk_kernels()

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
    # The class/function-level chalk kernels installed above patch the layers TRL just built; the
    # INSTANCE-level ones (FP8 base, embedding, FP8 MLP) need the materialized module, so install
    # them now against the SFT trainer.model. No-op unless a FLASH_* kernel flag is set and chalk present.
    install_chalk_kernels(getattr(trainer, "model", None))

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
            **wandb_run_info(),
        },
    )
    free_gpu(trainer)


# ---------------------------------------------------------------------------
# RL (GRPO) with TRL + colocated vLLM
# ---------------------------------------------------------------------------
def compute_grpo_batching(
    prompts_per_step: int, group_size: int, per_device_comps: int, num_processes: int = 1
) -> dict:
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
    nproc = max(1, int(num_processes))
    # Never let the per-device completion micro-batch exceed the PER-RANK share of the target
    # completion batch. The smallest GLOBAL micro-batch is per_device * num_processes, so capping at
    # the full target_comps (ignoring rank count) would let an num_processes>1 FSDP run overshoot
    # prompts_per_step*group_size and inflate unique_prompts/step. Cap at target_comps // nproc
    # (mirrors run_sft's `min(per_device_bs, effective_batch)`; no-op at the default nproc=1).
    per_device = max(1, min(per_device, max(1, target_comps // nproc)))
    # The GLOBAL completion batch TRL optimizes is per_device * grad_accum * num_processes —
    # accelerate/TRL multiply by the data-parallel world size (FSDP trainer ranks). To still optimize
    # `prompts_per_step` prompts/step under an `num_processes`-rank FSDP trainer, grad_accum must be
    # divided by num_processes; otherwise the effective batch (and unique_prompts/step) scales with
    # the rank count, and a small dataset can't fill even one step (the FSDP 0-real-steps bug seen on
    # 2:2). num_processes=1 (colocate / single-trainer 1:1/1:2) is unchanged.
    # Round UP (ceil), not floor: when target_comps is not an exact multiple of the global
    # micro-batch (per_device * nproc) — e.g. batch_size=5, group=8, per_device=8, nproc=2 targets
    # 40 completions but 40 // 16 == 2 yields only 32 (4 prompts, not the configured 5) — floor
    # division silently trains a SMALLER batch than [train].batch_size requested. Ceiling reaches at
    # least the configured prompts/step; the divisibility rounding below only ever raises it further.
    _global_micro = per_device * nproc
    grad_accum = max(1, -(-target_comps // _global_micro))
    # TRL rejects a GLOBAL completion batch (per_device * grad_accum * num_processes) that is not
    # divisible by num_generations (= group_size), failing only AFTER the paid worker is
    # provisioned. per_device is the fixed VRAM knob, so round grad_accum UP to the next multiple
    # that makes the GLOBAL batch divisible. The divisibility multiple must use per_device * nproc
    # (the global micro-batch), not per_device alone — otherwise a multi-trainer (nproc>1) run whose
    # global micro-batch already divides group_size still gets grad_accum inflated, training MORE
    # prompts/step than configured (e.g. group_size=8, nproc=2, per_device=4: global micro-batch 8
    # is already one full group, but gcd(per_device, group_size) would force grad_accum=2 -> 2
    # prompts/step instead of 1). nproc=1 (colocate / single-trainer) is unchanged.
    accum_step = group_size // math.gcd(per_device * nproc, group_size)
    grad_accum = ((grad_accum + accum_step - 1) // accum_step) * accum_step
    # generations_per_step / unique_prompts_per_step are reported GLOBALLY (across all ranks) so the
    # metric matches the intended prompts_per_step regardless of the trainer world size.
    generations_per_step = per_device * grad_accum * nproc
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
    vocab: int = 152_000,
    *,
    use_vllm: bool = True,
    colocated: bool = True,
    params_b: float | None = None,
) -> int:
    """Per-device *completion* micro-batch for GRPO (TRL counts completions, not prompts).

    This, not grad-accum, sets peak trainer VRAM: the logprob pass materializes fp32 logits
    of shape [per_device, completion_len, vocab]. At Qwen's ~152k vocab a long completion is
    enormous (measured: per_device 8 x 4096 tok x 152k x 4 B = ~20 GiB single alloc -> OOMs
    a small card). So we MEMORY-CAP per_device to a logits budget (RL_LOGITS_BUDGET_GB,
    default 6) for the given completion length, then push the difference into grad-accum
    (compute_grpo_batching) so the effective batch is unchanged. This keeps long-completion
    GRPO on a cheaper GPU. RL_PER_DEVICE_PROMPTS forces an explicit value.

    The logits budget is NOT the whole story: the per-device forward also holds the model's
    attention/activation memory (the Qwen3.5 GDN/FLA kernels peak per micro-batch even with
    grad checkpointing), which the logits term can't see. Under colocated vLLM (the rollout
    engine + its card-sized KV pool + a 2nd weight copy share the GPU) that activation peak is
    what OOMs a small card -- and Liger, which fuses away the logits, does NOT touch it.
    MEASURED: Qwen3.5-2B (width ~1.41) group8 seq2048 OOMs a 32 GB card at per_device=8 but
    TRAINS at 4. So for colocate, additionally cap per_device to the live card's VRAM scaled
    by model width (~sqrt(params)): ~vram_gb/8 at 2B-width, tightened for wider models (4B/9B).

    ``colocated`` gates ONLY that activation cap. In DISAGGREGATED mode the vLLM rollout server
    runs on a SEPARATE GPU, so the trainer card carries neither the engine nor its KV pool/2nd
    weight copy — the colocate activation cap would needlessly shrink per_device, inflating
    grad-accum and undermining the very throughput the split is meant to measure. The logits
    budget (a real per-device VRAM term) still applies in both modes. Defaults True (colocate).
    """
    base = int(os.environ.get("RL_PER_DEVICE_PROMPTS", "2" if THINKING else "8"))
    if "RL_PER_DEVICE_PROMPTS" in os.environ:
        # Explicit operator force (A/B knob): bypass BOTH auto-caps -- they own the OOM risk.
        return max(1, base)
    if completion_len > 0:
        budget = float(os.environ.get("RL_LOGITS_BUDGET_GB", "6")) * 1e9
        cap = max(1, int(budget / (max(1, completion_len) * vocab * 4)))
        base = min(base, cap)
    if use_vllm and colocated:
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
            # In a disaggregated multi-trainer (DDP) group EVERY rank runs this callback; the
            # heartbeat commits heartbeat.json to the SAME HF path, so N ranks would race /
            # duplicate-commit. Only rank 0 (world-process-zero) emits — single-process and
            # train_gpus==1 paths are always rank 0, so their behavior is unchanged.
            if not getattr(state, "is_world_process_zero", True):
                return
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


def _maybe_attach_periodic_eval(
    trainer,
    tok,
    *,
    is_multi_turn: bool,
    is_tool_env: bool,
    max_new_default: int,
    stop: list[str] | None,
    engine_max_len: int,
    max_turns: int,
):
    """Attach periodic mid-run eval to the GRPO trainer when enabled — the run's
    ``[train] eval_every_steps`` (or the ``FLASH_EVAL_EVERY_STEPS`` operator override) > 0.

    Returns the ``PeriodicEval`` (so the caller can persist its ``history`` into metrics.json),
    or ``None`` when eval is disabled/unsupported for this run.

    Every N optimizer steps it greedily evaluates the policy on a FIXED held-out split and
    streams the result via ``heartbeat("rl_eval", ...)`` so the agent gets a live eval curve
    between steps. Evaluation distinct from the reward comes from the env's eval-metric rubric
    metrics (``rubric.add_metric``), surfaced via the adapter's ``evaluate``.

    Generation uses the TRAINER'S model (``trainer.model.generate``), NOT the colocate vLLM
    engine: an out-of-band ``engine.generate`` from a callback hangs GRPO (verified on a live
    GPU run). The model path works on both backends and keeps memory bounded (one prompt at a
    time). Tool envs are skipped in v1: TRL drives their tool loop natively and the greedy
    single-shot path would mis-score them.
    """
    from flash.engine import midrun_eval as _me

    # The cadence comes from the run's [train] eval_every_steps TOML (env var overrides). The
    # eval queries + grading logic + completion budget all come from the environment / the run's
    # normal settings, not config.
    _train = JOB_SPEC.train if JOB_SPEC else None
    cfg = _me.eval_config(max_new_default, spec_every=getattr(_train, "eval_every_steps", None))
    if cfg["every_steps"] <= 0:
        return None
    if is_tool_env:
        print("[rl][eval] mid-run eval is not supported for tool envs in v1; skipping")
        return None
    env = require_active_env()
    # Mid-run eval is a HELD-OUT generalization signal: if the env has no DISTINCT eval split, skip
    # it entirely (fail fast) rather than falling back to training rows — scoring train data and
    # surfacing it as an eval curve is misleading. A missing split disables eval, not the training.
    # has_eval_split() can itself raise (e.g. a separate Hub eval env whose get_eval_dataset returns
    # None and eval_dataset is then accessed); like the materialize step below, a broken probe must
    # disable eval, never abort the paid training run.
    try:
        _has_eval_split = getattr(env, "has_eval_split", lambda: True)()
    except Exception as exc:
        print(f"[rl][eval] has_eval_split() failed ({exc}); skipping mid-run eval")
        return None
    if not _has_eval_split:
        print(
            "[rl][eval] env has no held-out eval split; skipping mid-run eval "
            "(refusing to fall back to training rows)"
        )
        return None
    # Materializing the eval split can raise (e.g. a separate Hub eval env whose get_eval_dataset
    # fails) — this runs at training start, so a raise here would abort the whole paid run. Guard
    # it: a broken eval split disables mid-run eval, never the training.
    try:
        # Pass the cap as `limit` so the env getter materializes at most num_examples rows (a huge
        # Hub eval/train split isn't fully built just to be sliced); the slice is a backstop for
        # envs whose getter ignores `n`.
        _cap = cfg["num_examples"] or None
        examples = env.dataset("eval", limit=_cap)[: cfg["num_examples"]]
    except Exception as exc:  # never let an eval-split failure abort training
        print(f"[rl][eval] could not materialize the eval split ({exc}); skipping mid-run eval")
        return None
    if not examples:
        print("[rl][eval] env exposes no eval examples; skipping mid-run eval")
        return None
    def _render_messages(messages, add_generation_prompt):
        text = tok.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            enable_thinking=THINKING,
        )
        return [int(t) for t in tok(text, add_special_tokens=False).input_ids]

    def _render_prompt_ids(example):
        return _render_messages(env.prompt_messages(example), True)

    def _build_score_one(model):
        generate = _me.build_hf_greedy_generate(model, tok, stop=stop)
        if is_multi_turn:
            return _me.multi_turn_scorer(
                env,
                _render_messages,
                generate,
                max_turns=max_turns,
                max_new_tokens=cfg["max_new_tokens"],
                engine_max_len=engine_max_len,
                on_warn=print,
            )
        return _me.single_turn_scorer(
            env, _render_prompt_ids, generate, cfg["max_new_tokens"], graded_text
        )

    # eval_pass_rate threshold = the ENV's own pass_threshold (what adapter.grade uses), so an env
    # with pass_threshold=0.8 isn't silently scored at 0.5; default 0.5 when the env defines none.
    env_pass = getattr(getattr(env, "_env", None), "pass_threshold", None)
    pass_threshold = float(env_pass) if env_pass is not None else 0.5

    periodic = _me.PeriodicEval(
        examples=examples,
        score_one_builder=_build_score_one,
        every_steps=cfg["every_steps"],
        heartbeat_fn=heartbeat,
        pass_threshold=pass_threshold,
        on_warn=print,
    )
    # Resolve the live model lazily at eval time (it's the trainer's own model, always present).
    periodic.bind_model_getter(lambda: getattr(trainer, "model", None))
    trainer.add_callback(_me.make_periodic_eval_callback(periodic))
    print(
        f"[rl][eval] mid-run eval every {cfg['every_steps']} steps on {len(examples)} held-out "
        f"examples (greedy via trainer model, max_new={cfg['max_new_tokens']}, "
        f"{'multi-turn' if is_multi_turn else 'single-turn'})"
    )
    return periodic


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
    # 4-bit-QLoRA tier: load the frozen base in NF4 so a continued-adapter GRPO run fits
    # the same memory budget as a fresh-LoRA one (and TRL still sees Linear4bit modules ->
    # bitsandbytes vLLM rollout).
    if model_quant(model_id) == "4bit-qlora":
        _patch_peft_weight_converter_compat()
        _attn = optimal_attn_impl()  # arch-aware attention on the QLoRA path too
        _mik = qlora_model_init_kwargs()
        if _attn:  # else leave transformers' default (sdpa)
            _mik["attn_implementation"] = _attn
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            **_mik,
        )
    else:
        _attn = optimal_attn_impl()
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            **({"attn_implementation": _attn} if _attn else {}),
        )
    model = PeftModel.from_pretrained(base, adir, is_trainable=True)
    return model, None


def _env_inference_gpus(default: int) -> int:
    """``FLASH_INFERENCE_GPUS`` as an int, tolerating a present-but-BLANK value.

    A ``[worker_env]`` table serializes every value via ``str(v)``, so an empty/whitespace
    ``FLASH_INFERENCE_GPUS`` reaches the worker as ``""`` — a bare ``int("")`` would raise
    ValueError and abort the worker before any clear config error. Treat blank/unparseable as
    "no override" (keep the spec default) instead of crashing."""
    raw = os.environ.get("FLASH_INFERENCE_GPUS")
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        print(
            f"[rl][disagg][warn] FLASH_INFERENCE_GPUS={raw!r} is not an integer; "
            f"using inference_gpus={default}"
        )
        return default


def _require_full_node_for_disaggregated(total_gpus: int, inference_gpus: int) -> None:
    """Fail fast when a disaggregated run's node exposes FEWER GPUs than [gpu] count requested.

    A provider can under-provision (rent an N-GPU offer yet expose <N to the container — observed
    on Vast). For COLOCATE that just means fewer cards than hoped, but for a DISAGGREGATED ratio it
    silently rebuilds a DIFFERENT topology: a requested 2:2 (count=4, inference_gpus=2) on a node
    that exposes only 3 GPUs collapses to 1:2 — one trainer instead of two, skipping the FSDP/DDP
    multi-trainer path the user paid for, while the VRAM was sized for the full ratio. Refuse the
    degraded split (the node is torn down on raise) instead of quietly mis-running the config."""
    hint = os.environ.get("FLASH_GPU_COUNT")
    if hint and hint.isdigit() and int(hint) > total_gpus:
        raise RuntimeError(
            f"[rl][disagg] node under-provisioned: [gpu] count requested {hint} GPU(s) but the "
            f"container exposes only {total_gpus} (nvidia-smi). A disaggregated rollout "
            f"(inference_gpus={inference_gpus}) needs the full requested topology — refusing to "
            f"silently run a smaller split. Re-provision a node with {hint} visible GPUs."
        )


def _bind_trainer_rank_local(where: str) -> None:
    """Bind THIS accelerate rank to its LOCAL_RANK CUDA device before any CUDA context forms.

    ``accelerate launch`` sets ONE comma-separated CUDA_VISIBLE_DEVICES for the whole trainer-only
    child group and selects each rank's GPU via LOCAL_RANK — it does NOT slice CVD per rank. So
    without an explicit set_device, every CUDA probe (e.g. _drop_fla_on_hopper's
    get_device_capability in main(), wait_for_gpu, setup_perf_backends) lands on the group's cuda:0,
    stacking every rank's startup context on the FIRST train GPU and starving/OOMing rank 0 on tight
    DDP ratios. Idempotent — safe to call once early (main()) and again in run_rl (the Trainer later
    selects the same device)."""
    _local_rank = os.environ.get("LOCAL_RANK")
    if _local_rank is None or not _local_rank.isdigit():
        return
    try:
        import torch as _torch_rank

        _torch_rank.cuda.set_device(int(_local_rank))
        print(f"[rl][disagg] trainer-only child bound to cuda:{_local_rank} (LOCAL_RANK, {where})")
    except Exception as _e:
        print(f"[rl][disagg][warn] could not set_device(LOCAL_RANK={_local_rank}, {where}): {_e}")


def _fp8_native_no_cuda_context() -> bool | None:
    """Is this node's GPU fp8-native (compute capability >= 8.9) WITHOUT creating a CUDA context?

    ``torch.cuda.get_device_capability()`` binds (and RETAINS) a CUDA context on the current device.
    On the disaggregated DDP LAUNCHER that context lands on the first TRAINER card and stays alive
    while the accelerate child trains on that same card — stealing VRAM / OOMing rank 0 on a tight
    2:1/2:2 ratio (same hazard the launcher already avoids by skipping wait_for_gpu/perf_backends).
    So read compute_cap from ``nvidia-smi`` (CUDA-free) instead. The trainer card is the same class
    as the inference card (whole-machine node), so it is a valid proxy for the rollout server's fp8
    KV decision. Returns True/False on a clean read, or None when nvidia-smi is unavailable/unparsable
    (caller then falls back to the safe default — no fp8)."""
    import subprocess as _sp

    try:
        out = _sp.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        for ln in out.stdout.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            major_s, _, minor_s = ln.partition(".")
            return (int(major_s), int(minor_s or 0)) >= (8, 9)
    except Exception:
        return None
    return None


def _is_disaggregated_ddp_launcher() -> bool:
    """True when THIS main() process is the multi-trainer (train_gpus>1) disaggregated DDP launcher.

    The launcher builds no trainer: it starts the vLLM rollout server, then re-execs the worker under
    ``accelerate launch`` and waits while the children train. It STAYS ALIVE the whole run, so any
    CUDA context it binds (e.g. _drop_fla_on_hopper's get_device_capability) sits idle on the first
    trainer card and steals VRAM from / OOMs rank 0 on a tight 2:1/2:2 split — the same hazard run_rl
    already avoids by skipping wait_for_gpu/setup_perf_backends on the launcher. Computed CUDA-FREE
    (env + nvidia-smi only) so the check itself never creates the context it is guarding against."""
    if RUN_MODE != "rl":
        return False
    from flash.engine import disaggregated as _disagg

    if _disagg.trainer_only_mode():
        return False  # the accelerate child IS a trainer; it must drop fla and bind its rank device
    inference_gpus = int(JOB_SPEC.train.inference_gpus) if (JOB_SPEC and JOB_SPEC.train) else 0
    inference_gpus = _env_inference_gpus(inference_gpus)
    if inference_gpus <= 0:
        return False
    try:
        from flash.engine.rollout_bench import select_rollout_split

        total = _disagg.detect_total_gpus()
        if inference_gpus >= total:
            return False  # invalid split; run_rl raises with a clear message
        return select_rollout_split(total, inference_gpus).train_gpus > 1
    except Exception as e:
        # We are already in the disaggregated configuration (rl mode, not the trainer-only child,
        # inference_gpus>0), and the only non-launcher split — inference_gpus>=total — is returned
        # above before this try. So an UNEXPECTED split-detection failure here is NOT evidence that
        # this is a colocated/single-trainer run; swallowing it as `return False` would let the
        # caller (_drop_fla_on_hopper) run get_device_capability() and bind a retained CUDA context
        # on the pinned trainer card for the whole DDP launch — the exact rank-0 VRAM/OOM hazard this
        # guard exists to prevent. Fail SAFE to True (the launcher trains nothing, so skipping its fla
        # probe is free; each accelerate child still drops fla rank-local), and log loudly so the
        # mis-detection is visible rather than silent.
        print(
            f"[rl][disagg] WARNING: _is_disaggregated_ddp_launcher split detection failed ({e!r}); "
            "assuming multi-trainer launcher to avoid binding a CUDA context on the trainer card "
            "(skips the fla probe; children still drop fla rank-local).",
            flush=True,
        )
        return True


def _pin_trainer_devices_for_disaggregated() -> None:
    """Pin CUDA_VISIBLE_DEVICES to the TRAINER's devices before any CUDA context is created.

    Disaggregated GRPO only (``[train].inference_gpus>0``). Must run at the very top of main(),
    before _drop_fla_on_hopper / finalize_alloc_conf_for_sleep / any model import touches CUDA —
    CUDA_VISIBLE_DEVICES is honored only before the first context. The vLLM server (a separate
    subprocess) gets the inference devices via its own env (server_subprocess_env); the trainer
    must NOT also land on device 0 or TRL aborts the weight-sync init with "same CUDA device ...
    for multiple distinct roles/ranks". detect_total_gpus reads FLASH_GPU_COUNT / nvidia-smi (no
    torch context), so this stays CUDA-free. run_rl recomputes the same split (deterministic)."""
    if RUN_MODE != "rl":
        return
    from flash.engine import disaggregated as _disagg
    from flash.engine.rollout_bench import select_rollout_split

    # In the accelerate-launched trainer child (train_gpus>1), `accelerate launch --gpu_ids` already
    # pinned this rank to its single train GPU; overriding CUDA_VISIBLE_DEVICES here would make every
    # rank see all train GPUs and collide. Leave the env exactly as accelerate set it.
    if _disagg.trainer_only_mode():
        print("[rl][disagg] trainer-only child: accelerate manages CUDA_VISIBLE_DEVICES; skip pin")
        # Do NOT override CVD (accelerate's --gpu_ids already pinned the group), but DO bind this rank
        # to its LOCAL_RANK device NOW: main() calls _drop_fla_on_hopper (get_device_capability) right
        # after this, which would otherwise create the rank's CUDA context on the group's cuda:0 —
        # before run_rl's set_device runs — stacking every rank on the first train GPU (fla #640 probe
        # + DDP starvation). Binding here makes that earlier probe rank-local.
        _bind_trainer_rank_local("early")
        return
    inference_gpus = int(JOB_SPEC.train.inference_gpus) if (JOB_SPEC and JOB_SPEC.train) else 0
    inference_gpus = _env_inference_gpus(inference_gpus)
    if inference_gpus <= 0:
        return

    total = _disagg.detect_total_gpus()
    if inference_gpus >= total:
        return  # run_rl's select_rollout_split will raise with a clear message
    split = select_rollout_split(total, inference_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = _disagg.trainer_cuda_visible_devices(split)
    print(
        f"[rl][disagg] (early) pinned trainer CUDA_VISIBLE_DEVICES="
        f"{os.environ['CUDA_VISIBLE_DEVICES']} (node {total} GPUs, {inference_gpus} for inference; "
        f"server gets device(s) {','.join(map(str, split.infer_devices))})"
    )


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
    # ---- Disaggregated (multi-GPU async) rollout: device split (BEFORE any CUDA context) ----
    # [train].inference_gpus>0 dedicates GPUs to a separate vLLM rollout server so generation
    # overlaps the optimizer step instead of time-sharing one card (verl async rollout). The
    # trainer process MUST pin CUDA_VISIBLE_DEVICES to its train devices before wait_for_gpu() (or
    # any torch.cuda call) binds the context to all visible cards. The server is launched later (it
    # needs the engine length); here we only compute the split + pin the trainer + snapshot a clean
    # env for the server subprocess (which gets the GLOBAL inference device indices).
    from flash.engine import disaggregated as _disagg
    from flash.engine.rollout_bench import (
        select_rollout_split,
        validate_disaggregated_requirement,
    )

    _inference_gpus = int(JOB_SPEC.train.inference_gpus) if (JOB_SPEC and JOB_SPEC.train) else 0
    _inference_gpus = _env_inference_gpus(_inference_gpus)
    # Re-validate against the EFFECTIVE inference_gpus: submit-time validation only saw the spec
    # value, but FLASH_INFERENCE_GPUS (via [worker_env]) can change it here — e.g. =0 would force
    # colocated GRPO for a requires_disaggregated model that OOMs colocated. Fail fast on the worker.
    from flash.catalog import MODELS as _CATALOG_RL

    _info_rl = _CATALOG_RL.get(JOB_SPEC.model if JOB_SPEC else RECIPE.hf_model_id)
    validate_disaggregated_requirement(
        requires_disaggregated=bool(getattr(_info_rl, "requires_disaggregated", False)),
        algorithm="grpo",
        inference_gpus=_inference_gpus,
    )
    _rollout_split = None
    _disagg_base_env: dict | None = None
    # Defaults for the colocate (inference_gpus==0) and train_gpus==1 paths — these MUST be defined
    # before the disaggregated branch so the trainer-build block's `if _is_fsdp_launcher:` never hits
    # an UnboundLocalError on the colocate path (only the train_gpus>1 launcher sets it True).
    _trainer_only = False
    _is_fsdp_launcher = False
    if _inference_gpus > 0:
        _total_gpus = _disagg.detect_total_gpus()
        # Refuse a degraded topology if the provider under-provisioned (fewer visible GPUs than the
        # requested [gpu] count): a disaggregated ratio sized/intended for the full node must not
        # silently collapse to a smaller split.
        _require_full_node_for_disaggregated(_total_gpus, _inference_gpus)
        _rollout_split = select_rollout_split(_total_gpus, _inference_gpus)
        print(
            f"[rl][disagg] node has {_total_gpus} GPU(s); split {_rollout_split.label}: "
            f"train devices {_rollout_split.train_devices}, infer devices "
            f"{_rollout_split.infer_devices}"
        )
        # train_gpus>1 (2:1, 3:1, 2:2) needs the trainer as a DISTRIBUTED group, not one process
        # spanning many GPUs (that silently becomes nn.DataParallel, which TRL's weight-sync gather
        # can't handle). Two roles, same code path:
        #   LAUNCHER (this run, train_gpus>1, RANK unset): launches the vLLM server below, then at the
        #     trainer-build point re-execs the worker under `accelerate launch` (FSDP) across the
        #     train devices and waits — it does NOT build a trainer itself.
        #   TRAINER child (FLASH_RL_TRAINER_ONLY=1, RANK set by accelerate): skips the server launch
        #     (connects to the launcher's server via env port) and runs the FSDP GRPOTrainer; only
        #     rank 0 writes artifacts. train_gpus==1 keeps the simple single-process path untouched.
        _trainer_only = _disagg.trainer_only_mode()
        _is_fsdp_launcher = _rollout_split.train_gpus > 1 and not _trainer_only
        # Defense-in-depth (submit-time validate_disaggregated_requirement is the primary gate): a
        # single-trainer-only model (e.g. the 35B) must NOT take the multi-trainer DDP launcher path
        # — it replicates the whole policy per trainer rank (host-RAM/OOM). FLASH_GPU_COUNT /
        # FLASH_INFERENCE_GPUS are reserved in [worker_env], but reject here too rather than crash
        # the DDP launch on a paid node, in case the effective split still resolves >1 trainer GPU.
        if _rollout_split.train_gpus > 1 and bool(getattr(_info_rl, "single_trainer_only", False)):
            raise RuntimeError(
                f"[rl][disagg] {JOB_SPEC.model if JOB_SPEC else RECIPE.hf_model_id} only supports a "
                f"single-trainer (1:N) disaggregated split, but this node resolves "
                f"{_rollout_split.train_gpus} trainer GPUs ({_rollout_split.label}). The multi-trainer "
                f"DDP launcher replicates the full policy per rank, which OOMs for a model this large. "
                f"Use [gpu] count = inference_gpus + 1."
            )
        _disagg_base_env = dict(os.environ)  # clean snapshot for the server subprocess
        if not _trainer_only:
            # The launcher (and the train_gpus==1 path) pin the trainer away from the inference card.
            # The accelerate child must NOT — accelerate's --gpu_ids already pinned each rank.
            os.environ["CUDA_VISIBLE_DEVICES"] = _disagg.trainer_cuda_visible_devices(_rollout_split)
            print(f"[rl][disagg] trainer CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            print(
                f"[rl][disagg] trainer-only child rank={os.environ.get('RANK','0')} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} (accelerate-pinned)"
            )
            # Re-affirm the rank-local binding (already done in main()'s early pin) so all CUDA work
            # here (wait_for_gpu's `device="cuda"`, setup_perf_backends) is rank-local rather than
            # stacking on the group's cuda:0. Idempotent; the Trainer/Accelerator later selects the
            # same device.
            _bind_trainer_rank_local("run_rl")
        # PiSSA's SVD init crashes in the CVD-pinned (non-zero device) disaggregated trainer
        # (peft set_data incompatible-tensor-type); fall back to standard LoRA init. setdefault so
        # an explicit operator [worker_env] FLASH_LORA_INIT still wins. Init method is irrelevant
        # to the throughput this path measures.
        # setdefault would NOT override an operator-set BLANK value, leaving make_lora with an
        # invalid empty init (and PiSSA would still be attempted in this CVD-pinned trainer);
        # treat blank/unset as "use standard init" so the fallback actually engages.
        if not os.environ.get("FLASH_LORA_INIT"):
            os.environ["FLASH_LORA_INIT"] = "default"
    disaggregated = _rollout_split is not None and _rollout_split.mode == "disaggregated"
    # The FSDP/DDP LAUNCHER builds NO trainer (it re-execs the worker under `accelerate launch` and
    # returns) — but its CUDA_VISIBLE_DEVICES is pinned to ALL the train GPUs. wait_for_gpu()'s
    # `torch.zeros(..., device="cuda")` would create and RETAIN a CUDA context on the first trainer
    # card, and setup_perf_backends() likewise initializes the primary context; in a memory-tight
    # 2:1/2:2 disaggregated run that idle launcher context steals VRAM from / OOMs rank 0 on the
    # SAME card the accelerate child then trains on. Skip both probes on the launcher; each
    # accelerate child runs them in its own (rank-local) process below.
    if not _is_fsdp_launcher:
        wait_for_gpu()
        setup_perf_backends()
    model_id = JOB_SPEC.model if JOB_SPEC else RECIPE.hf_model_id
    # QLoRA tier loads the base bf16 checkpoint; vLLM/transformers quantize it to 4-bit NF4 at load.
    quant = model_quant(model_id)
    download_seconds = prefetch_model(model_id)
    rl = RECIPE.rl
    steps = int(os.environ.get("RL_STEPS", str(rl.num_steps)))
    # Throughput/quality knobs (env-overridable): the number of prompts optimized per step,
    # completions per prompt, and whether vLLM offloads weights between steps. Sleep mode
    # frees memory for the optimizer but reloads ~weights each step (a large per-step cost);
    # disable it (RL_VLLM_SLEEP=0) with a higher RL_VLLM_GPU_UTIL when both fit resident.
    # SDK-supplied GRPO knobs (datums parity) override the recipe; env vars still win.
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
    # GRPO with it OFF on models that fit. Default it by model size (same small=speed / large=memory
    # gate as gradient checkpointing): OFF for small/fitting models, ON for large. RL_VLLM_SLEEP wins.
    _sleep_env = os.environ.get("RL_VLLM_SLEEP")
    if _sleep_env is not None:
        sleep_mode = _sleep_env not in ("0", "false", "False")
    else:
        # Gate on the GRPO rollout context (the run's [train].max_length sizes the engine + KV
        # cache): a long-context GRPO run is memory-tight and needs sleep mode. Matches the
        # liger-loss gate below.
        _grpo_ctx = int(_t.max_length if _t and _t.max_length else 0)
        sleep_mode = _memory_mode(model_id, _grpo_ctx)
    if disaggregated:
        # The disaggregated rollout server runs continuously on its own GPU(s); there is no
        # trainer/rollout time-share to free memory for, so sleep mode (offload weights each step)
        # is both unnecessary and unsupported across the process boundary.
        sleep_mode = False
    # Rollout backend: vLLM. Colocate (default) shares the trainer GPU; disaggregated runs a
    # separate vLLM server on dedicated GPU(s). There is no transformers-generation fallback.
    use_vllm = True
    print(f"[rl] rollout backend: {'disaggregated vLLM server' if disaggregated else 'colocated vLLM'}")
    from flash.catalog import MODELS as _CATALOG

    _info = _CATALOG.get(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train = ACTIVE_ENV.dataset("train")
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
    from flash.engine.vram import fetch_hf_params_b, params_b_from_str

    _params_b = params_b_from_str(getattr(_info, "params", None)) if _info else None
    # Open-model (uncataloged) GRPO: _info carries no param count, so size the colocate
    # activation cap from the HF safetensors metadata (no download). Without this, a large
    # open model falls back to the ~2B-width default in rl_per_device_comps and gets too LOOSE
    # a per-device cap -> colocate OOM. Best-effort: stays None offline, keeping prior behavior.
    if _params_b is None:
        _params_b = fetch_hf_params_b(model_id)
    # colocated=not disaggregated: the colocate activation cap only applies when the rollout engine
    # shares the trainer GPU. Disaggregated runs the engine on a separate card, so skip that cap (the
    # logits-budget cap still applies) — otherwise per_device is shrunk and grad-accum inflated, which
    # would cancel out the throughput the split exists to deliver.
    per_device_comps = rl_per_device_comps(
        _max_completion, use_vllm=use_vllm, colocated=not disaggregated, params_b=_params_b
    )
    # In the FSDP trainer child the optimizer runs across train_gpus ranks, so the per-rank grad_accum
    # must be divided by that count (else the global batch over-scales and a small dataset yields 0
    # real steps). The single-process paths (colocate, 1:1/1:2 with one trainer GPU, and the launcher
    # which builds no trainer) pass num_processes=1 — unchanged.
    _grpo_nproc = _rollout_split.train_gpus if (_trainer_only and _rollout_split) else 1
    batching = compute_grpo_batching(
        prompts_per_step, group_size, per_device_comps, num_processes=_grpo_nproc
    )
    if not batching["divisible_by_group"]:
        print("WARN: generation batch not divisible by group size; check RL_PER_DEVICE_PROMPTS")
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
    # Liger fused GRPO loss: fuses the lm_head + per-token logprob so the fp32
    # [batch, seq, ~152k vocab] logits never materialize — the documented GRPO OOM driver.
    # TRL 1.6's GRPOConfig flag is `use_liger_kernel` (NOT `use_liger_loss`, which doesn't
    # exist in 1.6). DEFAULT ON for the GRPO path regardless of model size: MEASURED that
    # WITHOUT it even Qwen3.5-0.8B GRPO OOMs a 24 GB (and 32 GB) card because the per-completion
    # logits over the 152k vocab dominate — the small-scale JIT cost is far cheaper than the OOM.
    # (This differs from SFT, where Liger is gated by size since 1B-class SFT can be net-negative.)
    if liger_on(True):
        grpo_kwargs["use_liger_kernel"] = True
        print("[rl] liger fused GRPO loss enabled")
    vllm_proc = None
    # Reject an unsupported multi-turn / no-tool config BEFORE launching the paid rollout server.
    # is_multi_turn, is_tool_env, and disaggregated are all known here; the tools accessor is pure.
    # The wiring block far below re-derives the same use_rollout_func, but by then (in disaggregated
    # mode) the standalone vllm-serve has already been booted + health-checked — minutes of a large
    # rollout server, possibly surfacing a server-boot error instead of this config error. Decide it
    # up front. (build_rollout_func/index building still happens at the wiring site for colocate.)
    _tools_for_guard = ACTIVE_ENV.tools() if is_tool_env else []
    use_rollout_func = is_multi_turn and not (is_tool_env and _tools_for_guard)
    if use_rollout_func and disaggregated:
        # The multi-turn rollout_func drives generation through TRL's IN-PROCESS vLLM engine; in
        # disaggregated mode vLLM is a separate server (vllm_mode="server") with no in-process
        # engine, so the rollout would fail at the first custom turn. Fail fast BEFORE server launch.
        raise RuntimeError(
            "multi-turn GRPO (custom rollout_func) is not supported with disaggregated rollout "
            "(inference_gpus>0): it needs an in-process vLLM engine, but disaggregated runs vLLM as "
            "a separate server. Use colocated GRPO (inference_gpus=0) for multi-turn / tool envs."
        )
    if use_vllm and disaggregated:
        # Disaggregated: a separate `trl vllm-serve` process owns the inference GPU(s); the trainer
        # connects over loopback (vllm_mode="server") and TRL syncs the (PEFT-merged) policy weights
        # to it via NCCL each generation batch (sync_weights), so generation for the next batch
        # overlaps this step's optimizer instead of time-sharing one card (verl async rollout).
        # The server gets the whole inference card (no colocate util cap), stays resident (no sleep),
        # and its KV cache is bounded by --max_model_len = the GRPO engine length.
        _port = _disagg.server_port()
        _server_util = float(
            os.environ.get("RL_VLLM_SERVER_UTIL", str(_disagg.DEFAULT_SERVER_GPU_UTIL))
        )
        # verl rollout lever (mirror colocate): fp8 KV cache on fp8-native silicon (compute
        # capability >= 8.9: Ada/Hopper/Blackwell) ~halves the server's KV pool. The trainer GPU is
        # the same class as the inference GPU (whole-machine node), so its capability is a valid
        # proxy. Ampere (A100/3090, sm80) lacks fp8 -> stay fp16.
        # On the DDP LAUNCHER (train_gpus>1, not trainer_only) a torch CUDA probe here would bind a
        # context on the first trainer card that lingers while the accelerate child trains on it
        # (VRAM theft / rank-0 OOM on tight ratios — the same reason wait_for_gpu/perf_backends are
        # skipped on the launcher). Use the CUDA-free nvidia-smi compute_cap read there; only the
        # single-process / trainer-child paths (already bound to their own device) take the torch
        # probe. nvidia-smi unavailable -> default to no fp8 (safe: a larger fp16 KV pool, never OOM).
        if _is_fsdp_launcher:
            _fp8 = _fp8_native_no_cuda_context()
            _server_kv = "fp8" if _fp8 else None
        else:
            try:
                import torch as _torch

                _server_kv = "fp8" if _torch.cuda.get_device_capability() >= (8, 9) else None
            except Exception:
                _server_kv = None
        # Inference parallelism across the rollout GPUs. DEFAULT = tensor-parallel: shard the model
        # across infer_gpus so the decode (memory-bandwidth bound) gets their aggregate HBM
        # bandwidth -> faster, larger-batch generation, which is how 1:2 / 1:3 ratios clear the
        # rollout bottleneck on a generation-bound step. Works for dense AND MoE. Data-parallel
        # (replicas) is REJECTED by vLLM for dense models ("Offline data parallel ... not supported
        # ... for dense models"), so reserve FLASH_DISAGG_PARALLEL=dp for the MoE 35B-A3B only.
        _parallel = (os.environ.get("FLASH_DISAGG_PARALLEL") or "tp").strip().lower()
        if _parallel not in ("dp", "tp"):
            _parallel = "tp"
        if _parallel == "dp":
            # vLLM rejects data-parallel for DENSE models ("Offline data parallel ... not supported
            # ... for dense models"); only MoE supports it. Validate up front and fall back to tp
            # (valid for dense AND MoE) with a warning, rather than booting the server for ~20 min
            # and only then crashing on the incompatible config. tp is always a safe fallback, so an
            # uncertain probe (offline / config unreadable) also degrades to tp.
            _is_moe = False
            try:
                from transformers import AutoConfig

                _cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                _is_moe = bool(
                    getattr(_cfg, "num_experts", 0) or getattr(_cfg, "n_routed_experts", 0)
                ) or "moe" in (getattr(_cfg, "model_type", "") or "").lower()
            except Exception as _e:
                print(f"[rl][disagg] MoE probe failed ({_e}); assuming dense", flush=True)
            if not _is_moe:
                print(
                    f"[rl][disagg] WARNING: FLASH_DISAGG_PARALLEL=dp is MoE-only (vLLM rejects "
                    f"data-parallel for dense models); {model_id} is dense -> using tp.",
                    flush=True,
                )
                _parallel = "tp"
        print(
            f"[rl][disagg] inference parallelism={_parallel} across {_rollout_split.infer_gpus} "
            f"rollout GPU(s) (tp=shard-one-model/throughput, dp=replicas-MoE-only)",
            flush=True,
        )
        # FAST-FAIL the invalid-TP ratio: vLLM requires the model's attention-head count to be
        # divisible by tensor_parallel_size (= infer_gpus). E.g. MiniCPM5-1B has 16 heads, so TP=3
        # (a 1:3 split) is REJECTED ("Total number of attention heads (16) must be divisible by
        # tensor parallel size (3)") — valid infer ratios for a 16-head model are 1, 2, 4, 8. Without
        # this check the worker rents the GPU and boots the server for ~20 min only to crash on the
        # pydantic VllmConfig validation. Catch it here (config read is cheap) with an actionable
        # message naming the valid inference-GPU counts. Only relevant for TP with >1 inference GPU.
        if _parallel == "tp" and _rollout_split.infer_gpus > 1:
            try:
                from transformers import AutoConfig

                _hc = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                # VL / multimodal configs nest the LM dims under text_config (the top-level
                # num_attention_heads may be the vision encoder's). Pick the LM config the same way
                # _estimate_params does: top-level only when it carries hidden_size, else text_config.
                _tc = getattr(_hc, "text_config", None)
                _src = _hc if getattr(_hc, "hidden_size", 0) else (_tc or _hc)
                _heads = getattr(_src, "num_attention_heads", None)
            except Exception as _e:
                _heads = None
                print(f"[rl][disagg] head-count probe failed ({_e}); skipping TP divisibility check", flush=True)
            if _heads and _heads % _rollout_split.infer_gpus != 0:
                # Disaggregated rollout reserves >=1 GPU for training, so inference_gpus < _total_gpus
                # (never the full node) — cap the suggested divisors accordingly.
                _valid = [d for d in range(1, _heads + 1) if _heads % d == 0 and d < _total_gpus]
                raise RuntimeError(
                    f"[rl][disagg] invalid tensor-parallel split: {model_id} has {_heads} attention "
                    f"heads, not divisible by inference_gpus={_rollout_split.infer_gpus} (vLLM "
                    f"requires heads % TP == 0). Valid inference_gpus for this model: {_valid}. "
                    f"Pick a [train] inference_gpus from that set (e.g. a 1:2 or 1:4 split)."
                )
        # Optional --enforce_eager: skip vLLM CUDA-graph capture at server boot. For very large
        # models (e.g. the 35B-A3B MoE) graph capture dominates the boot window and can blow past
        # RL_VLLM_SERVER_TIMEOUT before the server is ever healthy; eager trades a little decode
        # throughput for a tractable boot. Opt-in via FLASH_RL_VLLM_ENFORCE_EAGER so small models
        # keep graphs (their boot is cheap and they want the decode speed).
        _vllm_extra: list[str] = []
        if os.environ.get("FLASH_RL_VLLM_ENFORCE_EAGER", "").strip().lower() in ("1", "true", "yes"):
            _vllm_extra += ["--enforce_eager", "true"]
        _cmd = _disagg.build_vllm_serve_cmd(
            model_id,
            _rollout_split,
            max_model_len=vllm_max_len,
            port=_port,
            gpu_memory_util=_server_util,
            quant=quant,
            kv_cache_dtype=_server_kv,
            parallel=_parallel,
            extra=(_vllm_extra or None),
        )
        _server_timeout = float(os.environ.get("RL_VLLM_SERVER_TIMEOUT", "1200"))
        if _trainer_only:
            # The FSDP launcher already started the server on the inference card(s); this accelerate
            # rank just connects to it (vllm_proc stays None so the finally never terminates the
            # launcher's server). No health wait — the launcher only re-execs us once it is healthy.
            print(f"[rl][disagg] trainer-only rank: connecting to existing rollout server at port {_port}")
        else:
            _server_log = "/tmp/vllm_serve.log"
            vllm_proc = _disagg.launch_vllm_server(
                _cmd, _disagg.server_subprocess_env(_disagg_base_env, _rollout_split), log_path=_server_log
            )
            # vLLM server boot under spawn is slow: re-import torch/vLLM in the spawned worker + model
            # download + load + CUDA-graph capture. A 1B model measured ~450s end-to-end; bigger models
            # take longer, so default generously (uvicorn binds the HTTP port only AFTER the worker
            # finishes loading, so the health check sees connection-refused until then).
            try:
                # Emit a heartbeat every 60s during the boot so the control plane's no-heartbeat
                # STALL detector (~25 min) doesn't kill a big model mid-boot: the 35B server
                # (70 GB bf16 + tilelang/CUDA-graph JIT) can take >20 min to bind its port, and that
                # whole stretch is silent otherwise. (This was why the 35B run got killed at 1503s.)
                _boot_t0 = time.time()
                _disagg.wait_for_server_health(
                    _port,
                    timeout=_server_timeout,
                    proc=vllm_proc,
                    log_path=_server_log,
                    on_wait=lambda: heartbeat("rl_server_boot", boot_seconds=round(time.time() - _boot_t0)),
                )
            except BaseException:
                # A server boot failure / health-check error here must NOT leak the vllm-serve
                # subprocess + its inference GPU; the trainer try/finally below only covers train().
                _disagg.terminate_server(vllm_proc)
                raise
            print(f"[rl][disagg] rollout server healthy at {_disagg.server_base_url(_port)}")
        grpo_kwargs.update(
            vllm_mode="server",
            vllm_server_base_url=_disagg.server_base_url(_port),
            vllm_server_timeout=_server_timeout,
        )
        # Keep the trainer's NCCL weight-sync group port in lockstep with the server's. Both default
        # to TRL's 51216, but FLASH_VLLM_GROUP_PORT overrides the SERVER (server_subprocess_env);
        # the trainer's VLLMClient must use the same port or sync_weights can't rendezvous. Set the
        # field only if this TRL exposes it (older TRL rejects unknown GRPOConfig kwargs).
        _group_port = _disagg.group_port()
        if "vllm_group_port" in getattr(GRPOConfig, "__dataclass_fields__", {}):
            grpo_kwargs["vllm_group_port"] = _group_port
        elif _group_port != _disagg.DEFAULT_GROUP_PORT:
            print(
                f"[rl][disagg][warn] this TRL has no GRPOConfig.vllm_group_port; cannot honor "
                f"FLASH_VLLM_GROUP_PORT={_group_port} on the trainer side (stays TRL default 51216)"
            )
    elif use_vllm:
        # Colocate shares one GPU between the policy model and the vLLM rollout engine.
        # vllm_max_model_length bounds the KV cache to what GRPO needs (else vLLM sizes for
        # the model's FULL context and won't start on a consumer GPU). RL_VLLM_GPU_UTIL
        # sizes vLLM's pool; RL_VLLM_SLEEP offloads its weights between steps.
        grpo_kwargs.update(
            vllm_mode="colocate",
            vllm_max_model_length=vllm_max_len,
            vllm_gpu_memory_utilization=float(os.environ.get("RL_VLLM_GPU_UTIL", "0.45")),
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
    trainer = None
    try:
        # Adapter init: continue training the SFT adapter (peft_config=None, model is the
        # loaded PeftModel) when train.init_from_adapter is set, else a fresh LoRA on the
        # string model id (model_init_kwargs forces bf16 — TRL string-loading can fall back
        # to fp32 and double VRAM).
        #
        # The FSDP/DDP LAUNCHER (train_gpus>1) builds NO trainer — it re-execs the worker under
        # `accelerate launch` and the child ranks load their own copies. Loading the base+adapter
        # here would pin a full extra PeftModel on the launcher's train devices that is never freed
        # before the children spawn on the SAME cards -> OOM / lost memory for the ranks (esp. a
        # common GRPO-after-SFT run via train.init_from_adapter). Skip the load entirely on the
        # launcher; the children call _init_adapter_model in their own process.
        if _is_fsdp_launcher:
            init_model, init_peft = None, None
        else:
            # Install the CLASS/FUNCTION-level opt-in chalk kernels (LoRA delta, fused MLP/QKV, RoPE)
            # FIRST — they are global monkeypatches that must be in place BEFORE the model is built.
            # This MUST run before _init_adapter_model: on the train.init_from_adapter continue path
            # that call materializes the full PeftModel, so installing after it would miss those
            # layers (the fresh-LoRA path returns only the model-id string, built later by TRL, so it
            # was unaffected either way). The INSTANCE-level kernels (FP8 base, embedding, FP8 MLP)
            # need the actual nn.Module and are installed AFTER GRPOTrainer construction (below)
            # against trainer.model. Skipped on the FSDP launcher (it builds no trainer; the child
            # ranks install their own). No-op unless a FLASH_* kernel flag is set and chalk is present.
            install_chalk_kernels()
            init_model, init_peft = _init_adapter_model(model_id)
        if init_peft is not None:
            # Fresh LoRA: TRL loads the string model id with these kwargs, then attaches the
            # adapter. For the 4-bit-QLoRA tier load the base in NF4 — TRL detects the
            # bnb.Linear4bit modules and brings up its colocated vLLM rollout engine with
            # quantization="bitsandbytes" (so a 36B MoE fits an 80 GB GPU in 4-bit on both the
            # trainer and rollout sides). Otherwise force bf16 (TRL string-loading can fall
            # back to fp32 and double VRAM).
            _attn = optimal_attn_impl()  # arch-aware FlashAttention (Kernels Hub) / SDPA
            if quant == "4bit-qlora":
                _patch_peft_weight_converter_compat()  # adapter (re)load compatibility
                grpo_kwargs["model_init_kwargs"] = qlora_model_init_kwargs()
                _vllm_note = "; vLLM rollout -> bitsandbytes" if use_vllm else ""
                print(f"[rl] loading {model_id} in 4-bit (QLoRA tier){_vllm_note}")
            else:
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
        # NOTE: an N-step weight-sync knob (RL_SYNC_EVERY via steps_per_generation) was tried and
        # REVERTED — raising steps_per_generation multiplies generation_batch_size past a small
        # dataset into an empty dataloader (0 real steps / degenerate ~12s "done"), and the measured
        # speedup was nil (2B 2:2 RL_SYNC_EVERY=4 = 118.0s vs per-step 109.7s). The real multi-GPU
        # win is GPU SCALING (more inference TP + more DDP trainers): MiniCPM 1->8 GPU = 1.0x -> 5.21x,
        # which needs no trainer-config change. Removing the broken knob; the path to >2x beyond
        # scaling is async one-step-off OVERLAP (verl HybridEngine / TRL AsyncGRPOTrainer), separate.
        cfg = GRPOConfig(**grpo_kwargs)
        setup_seconds = time.time() - t_start
        heartbeat("rl_train_start", setup_seconds=setup_seconds)

        # VL checkpoints (Qwen3.5/3.6) train text-only: make TRL's colocated rollout
        # engine skip the vision tower (VRAM + 5090 PTX-compat; see the patch docstring).
        # Only reachable for the IN-PROCESS (colocate) engine. The disaggregated `trl vllm-serve`
        # server is a separate process this monkeypatch can't reach, AND TRL's vllm-serve exposes no
        # language-model-only flag, so that server loads the full model incl. the vision tower. That is
        # acceptable in practice: the only requires_disaggregated model (Qwen3.6-35B-A3B) runs on H200,
        # where the vision tower fits and the 5090-class PTX issue doesn't apply. Revisit (a TRL patch /
        # --hf-overrides shim) only if a disaggregated VL model is ever served on a smaller GPU.
        if use_vllm and not disaggregated:
            patch_vllm_language_model_only(model_id)
        hb_cb = make_reward_heartbeat_callback()
        # Multi-turn / tool wiring (trl 1.6): tool envs hand TRL the tool callables so it runs the
        # tool-call loop natively; pure multi-turn envs hand TRL a rollout_func that drives the
        # env's own turn loop on the colocate engine (env_mask masks the non-model tokens).
        extra_trainer_kwargs: dict = {}
        # tools / use_rollout_func were already computed (and the disaggregated incompatibility
        # rejected) ABOVE, before the rollout server launched, so reuse them here rather than
        # re-deriving (single source of truth; the disaggregated raise can't reach this point).
        tools = _tools_for_guard
        # A tool env exposing NO tools would silently degrade to single-shot under tools=[]; drive
        # it through the rollout_func turn loop instead so it isn't mis-trained as single-turn.
        if is_tool_env and not tools:
            print("[rl][warn] tool env exposes no tools — using the multi-turn rollout_func path")
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
        if _is_fsdp_launcher:
            # train_gpus>1: the server is up (above); re-exec THIS worker under `accelerate launch`
            # so the GRPO trainer runs as an FSDP group across the train devices and connects to the
            # server. We build no trainer in this process; the child (rank 0) writes all artifacts to
            # the SAME /tmp + uploads to HF, so after it returns the run is complete here too.
            import subprocess as _sp

            t_train = time.time()
            # Use DDP (replicate the model on each trainer rank), NOT FSDP sharding. TRL's per-step
            # weight sync calls peft merge_adapter()->get_delta_weight (weight_B @ weight_A); FSDP
            # (FULL_SHARD and even SHARD_GRAD_OP) flattens/shards the LoRA params so the merge fails
            # ("inconsistent tensor size [32768] vs [24576]"). DDP keeps each param whole on every
            # rank, so the merge works. Every model that runs a multi-trainer ratio here (1-9B)
            # fits replicated on one trainer card; the only model that wouldn't (35B) uses the
            # single-trainer 1:1 path, so DDP covers all the 2:1/2:2/3:1 ratios.
            _acc_cmd = _disagg.build_accelerate_launch_cmd(_rollout_split, use_fsdp=False)
            _child_env = dict(os.environ)
            _child_env["FLASH_RL_TRAINER_ONLY"] = "1"
            _child_env["FLASH_VLLM_SERVER_PORT"] = str(_port)
            _child_env["FLASH_VLLM_GROUP_PORT"] = str(_disagg.group_port())
            # The launcher pinned CUDA_VISIBLE_DEVICES to all train devices; the child must NOT inherit
            # that (accelerate's --gpu_ids assigns one GPU per rank). Drop it so accelerate controls
            # device placement across the train GPUs.
            _child_env.pop("CUDA_VISIBLE_DEVICES", None)
            print(f"[rl][disagg][fsdp] launching FSDP trainer group ({_rollout_split.label}): {' '.join(_acc_cmd)}")
            # Capture the child group's stdout+stderr to a file and ALWAYS upload it: the launcher's
            # own console is not reliably captured once the child writes DONE, so this is the only way
            # to see the FSDP trainer's behavior (real steps vs 0-step empty run) for debugging.
            _child_log = "/tmp/fsdp_child.log"
            with open(_child_log, "w") as _clf:
                _rc = _sp.run(_acc_cmd, env=_child_env, stdout=_clf, stderr=_sp.STDOUT).returncode
            print("[rl][disagg][fsdp] --- accelerate child log tail (last 160) ---")
            print(_disagg._server_log_tail(_child_log, n=160))
            print("[rl][disagg][fsdp] --- end child log ---", flush=True)
            try:
                hf_upload_file(_child_log, "console_fsdp_child.txt")
            except Exception as _e:
                print(f"[rl][disagg][fsdp] could not upload child log: {_e}")
            if _rc != 0:
                raise RuntimeError(
                    f"accelerate FSDP trainer group exited {_rc} (split {_rollout_split.label}); "
                    "see console_fsdp_child.txt"
                )
            print("[rl][disagg][fsdp] trainer group finished; artifacts written by rank 0")
        else:
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
            # Now that TRL has materialized the model, install the INSTANCE-level chalk kernels (FP8
            # base, embedding, FP8 MLP) against the actual module GRPOTrainer optimizes (trainer.model).
            # Doing it here (not on init_model) is what makes them reach the fresh-LoRA path, where
            # init_model was only the model-id string. No-op unless a FLASH_* kernel flag is set and
            # freesolo-chalk is installed.
            install_chalk_kernels(getattr(trainer, "model", None))
            # Opt-in periodic mid-run eval (the run's [train] eval_every_steps, or
            # FLASH_EVAL_EVERY_STEPS, > 0): greedy eval on a held-out split, streamed via
            # heartbeat("rl_eval", ...) AND accumulated into metrics.json so the agent reads
            # the eval curve (not just the noisy reward) when judging a run.
            #
            # DISABLE it entirely on a multi-trainer (DDP) group (disaggregated train_gpus>1). The
            # eval callback runs heavy on_step_end generation on the trainer's model: attaching it on
            # ALL ranks would duplicate the work + race the HF commits, but attaching it on rank 0
            # ONLY desyncs the process group — rank 0 stalls in generation / run_final while the other
            # ranks advance into the next step's (or save_model's) NCCL collective, hanging the group.
            # A per-rank callback fundamentally can't do unsynchronized work in DDP, so skip eval for
            # the multi-trainer ratios (the reward heartbeat is symmetric — every rank's on_log
            # returns fast off-rank-0 — so it stays). Single-trainer (colocate, 1:1, 1:2) is
            # unaffected and keeps the full eval curve.
            _multi_trainer = _rollout_split is not None and _rollout_split.train_gpus > 1
            periodic_eval = None
            if _multi_trainer:
                print(
                    "[rl][eval] mid-run eval is disabled for multi-trainer (DDP) disaggregated runs "
                    "(a per-rank eval callback would desync the NCCL process group); use a "
                    "single-trainer ratio (e.g. 1:2) for the eval curve"
                )
            else:
                periodic_eval = _maybe_attach_periodic_eval(
                    trainer,
                    tok,
                    is_multi_turn=is_multi_turn,
                    is_tool_env=is_tool_env,
                    max_new_default=_max_completion,
                    stop=(list(_t.stop_sequences) if _t and _t.stop_sequences else None),
                    engine_max_len=vllm_max_len,
                    max_turns=getattr(ACTIVE_ENV, "max_turns", 10),
                )
            t_train = time.time()
            with _sdpa_cudnn_ctx(_attn):  # force cuDNN SDPA on sm120 (no-op otherwise)
                trainer.train(resume_from_checkpoint=resume_ckpt)
    finally:
        # Free the inference card + port whether training finished or raised — INCLUDING a failure
        # while CONSTRUCTING the trainer, before train() starts (generation is done once train()
        # returns; on error the node is torn down anyway, but be tidy).
        if vllm_proc is not None:
            # Dump the rollout server's log tail to stdout (captured in console_rl.txt) so a
            # post-health failure — e.g. the weight-sync NCCL/TCPStore rendezvous on init_communicator
            # — is diagnosable from the server side, which the health-check tail can't reach.
            try:
                print("[rl][disagg] --- vllm-serve log tail (last 120 lines) ---")
                print(_disagg._server_log_tail("/tmp/vllm_serve.log", n=120))
                print("[rl][disagg] --- end vllm-serve log tail ---")
            except Exception as _e:
                print(f"[rl][disagg] could not read server log: {_e}")
            _disagg.terminate_server(vllm_proc)
            print("[rl][disagg] rollout server terminated")
    if _is_fsdp_launcher:
        # The accelerate child (rank 0) already saved the adapter, wrote /tmp/metrics.json, and
        # uploaded every artifact to HF; /tmp is shared on this node so the run is complete here.
        # This launcher built no trainer, so skip the in-process artifact writes and return.
        print("[rl][disagg][fsdp] launcher done (artifacts written by the FSDP trainer group)")
        return
    train_wall = time.time() - t_train
    reward_history = list(getattr(hb_cb, "reward_history", []))
    # Final eval on the actually-saved policy: the cadence only fires on multiples of
    # eval_every_steps, so when the run length isn't a multiple the last cadence eval predates the
    # saved adapter. run_final adds one eval on the final model (no-ops if the last step already
    # coincided with a cadence eval).
    if periodic_eval is not None:
        periodic_eval.run_final(int(getattr(trainer.state, "global_step", 0) or 0))
    eval_history = periodic_eval.history_records() if periodic_eval is not None else []

    adapter_dir = f"{out_dir}/adapter"
    if _disagg.trainer_only_mode():
        # FSDP trainer group: trainer.save_model gathers the sharded (LoRA) state across ALL ranks
        # and writes on rank 0 — it must be called by every rank or the gather deadlocks.
        # model.save_pretrained is NOT FSDP-aware (would save an empty/partial shard).
        trainer.save_model(adapter_dir)
    else:
        trainer.model.save_pretrained(adapter_dir)
    if not _disagg.is_main_rank():
        # Non-rank-0 FSDP workers finished their part of the state gather above; only rank 0 does the
        # HF upload + metrics + DONE. Return so the other ranks don't double-write/upload.
        return
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
            # Rollout topology for the ratio benchmark: "colocate" (1 GPU) or the disaggregated
            # train:infer split (e.g. "1:1", "2:1"). The benchmark reads these to label each row.
            "rollout_mode": _rollout_split.mode if _rollout_split else "colocate",
            "rollout_split": _rollout_split.label if _rollout_split else "colocate",
            "inference_gpus": _inference_gpus,
            "loss_curve": _metric_curve(trainer, "loss"),
            **wandb_run_info(),
            # The mid-run eval curve (per [train] eval_every_steps): each entry has step,
            # eval_reward, eval_pass_rate, and eval_metrics{}. Empty when eval is off. The agent
            # reads this to judge the run on held-out EVAL quality, not just the training reward.
            "eval_history": eval_history,
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
def gpu_diagnostics() -> dict:
    """Collect CUDA/driver diagnostics to pin down GPU init failures on rented nodes."""
    diag = {}
    try:
        import torch

        diag["torch"] = torch.__version__
        diag["torch_cuda"] = torch.version.cuda
        diag["cuda_available"] = torch.cuda.is_available()
        try:
            diag["device_count"] = torch.cuda.device_count()
            diag["device_name"] = torch.cuda.get_device_name(0)
        except Exception as e:
            diag["device_query_err"] = str(e)[:160]
    except Exception as e:
        diag["torch_import_err"] = str(e)[:160]
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        diag["nvidia_smi"] = (out.stdout or out.stderr).strip()[:200]
    except Exception as e:
        diag["nvidia_smi_err"] = str(e)[:160]
    return diag


def wait_for_gpu(max_tries=12, sleep_s=10):
    """Rented nodes sometimes report 'CUDA device not ready' transiently at startup.
    Poll a trivial CUDA op until it succeeds before doing real work; raise if never ready."""
    import time as _t

    last = None
    for i in range(max_tries):
        try:
            import torch

            if torch.cuda.is_available():
                # Force an actual kernel launch (alloc + add) to confirm the GPU is live.
                _ = torch.zeros(8, device="cuda") + 1
                torch.cuda.synchronize()
                print(f"GPU ready after {i} retries: {torch.cuda.get_device_name(0)}")
                return True
            last = "cuda not available"
        except Exception as e:
            last = str(e)[:160]
        print(f"GPU not ready (try {i + 1}/{max_tries}): {last}; sleeping {sleep_s}s")
        _t.sleep(sleep_s)
    raise RuntimeError(f"GPU never became ready after {max_tries} tries: {last}")


def free_gpu(trainer=None):
    try:
        import gc

        import torch

        try:
            if trainer is not None and hasattr(trainer, "model"):
                trainer.model = None
        except Exception:
            # Best-effort VRAM release before gc; any failure here is non-fatal cleanup.
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print("free_gpu warn:", e)


def _metric_curve(trainer, key: str, cap: int = 400) -> list:
    """The logged values of `key` (e.g. 'loss' or 'reward') from the trainer's log history,
    rounded + capped. Lets metrics.json carry the convergence/reward curve for an A/B without
    relying on a checkpoint's trainer_state.json (only written on save_steps) or the console
    (only uploaded on failure). Never raises."""
    try:
        vals = [round(float(h[key]), 4) for h in trainer.state.log_history if key in h]
        return vals[:cap]
    except Exception:
        return []


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
    _finalize(m, adapter_dir)


def _download_adapter(adapter_prefix: str | None) -> str | None:
    if not (adapter_prefix and HF_REPO):
        return None
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=[f"{adapter_prefix}/adapter/*"],
        local_dir="/tmp/evdl",
        token=os.environ.get("HF_TOKEN"),
    )
    adir = os.path.join("/tmp/evdl", adapter_prefix, "adapter")
    return adir if os.path.isdir(adir) else None


def _finalize(metrics: RunMetrics, adapter_dir: str):
    metrics.save("/tmp/metrics.json")
    # Required: a swallowed upload would make the control plane fail/retry a finished run.
    hf_upload_file("/tmp/metrics.json", "metrics.json", required=True)
    # DONE sentinel so the controller knows it's safe to tear down
    with open("/tmp/DONE", "w") as f:
        f.write(str(time.time()))
    hf_upload_file("/tmp/DONE", "DONE", required=True)
    heartbeat("done")
    print("NODE DONE:", metrics.to_json())


def _drop_fla_on_hopper() -> None:
    """Remove flash-linear-attention when running on a Hopper GPU (sm90, H100).

    fla's gated chunk_bwd Triton kernel is miscomputed on Hopper with Triton>=3.4 and
    HARD-RAISES (fla #640), so every gated-delta (Qwen3.5/3.6 family) GRPO backward crashes.
    The worker base image BAKES fla in, and per-run installs (extra_pip / `prime env install`)
    can pull it back, so the only reliable place to drop it is HERE: in the worker process,
    after all installs and BEFORE any model import. transformers then uses the correct
    pure-PyTorch delta rule (2-3x slower but it RUNS). Runs on BOTH substrates (RunPod and
    Vast both exec this module). importlib caches are invalidated so the later
    is_fla_available() probe sees it gone. Ampere/Ada/Blackwell keep fla for the speedup.
    """
    import importlib.util
    import subprocess

    # The multi-trainer DDP launcher trains NOTHING (it re-execs the worker under accelerate and
    # waits) so it has no fla backward to protect — and it stays alive the whole run, so the
    # get_device_capability() probe below would bind a retained CUDA context on the first trainer card
    # and starve/OOM rank 0 (the very hazard run_rl avoids by skipping wait_for_gpu/perf_backends on
    # the launcher). Each accelerate child runs _drop_fla_on_hopper in its own rank-local process.
    if _is_disaggregated_ddp_launcher():
        print("[hopper] disaggregated DDP launcher: skip fla probe (children drop it rank-local)")
        return

    try:
        import torch

        if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9):
            return  # not Hopper: fla's Triton kernel is correct here, keep it.
    except Exception:
        return

    if importlib.util.find_spec("fla") is None:
        return
    # pip first (clears metadata); _remove_fla_from_disk then deletes any package dir pip left
    # behind (incomplete RECORD / non-pip base-image install / a copy on another sys.path entry).
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "flash-linear-attention"], check=False
    )
    removed, still = _remove_fla_from_disk()
    print(
        f"[hopper] fla removed {removed or 'nothing'} (still_importable={still}) -> "
        f"{'WARNING fla remains' if still else 'pure-PyTorch delta rule'} (fla #640)",
        flush=True,
    )


def main():
    # Idempotency: if DONE was already uploaded, a re-delivered job re-fetches the final
    # metrics from HF and returns them immediately. (The previous behavior — sleeping in
    # an infinite loop — kept a billable GPU worker alive until the execution timeout.)
    try:
        # Idempotency FIRST — before any env-mutating pip install / package removal: a re-delivered
        # job whose DONE already exists must return the persisted metrics and exit WITHOUT running
        # _drop_fla_on_hopper() (pip-uninstalls fla) — that wasted a worker mutating its env on an
        # already-complete run. It is called after the DONE check below (see _drop_fla_on_hopper()).
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
        # Not a DONE re-delivery -> this worker will train. These must run before any model import:
        # Pin the trainer to its devices for the disaggregated GRPO path BEFORE anything below
        # touches CUDA (_drop_fla_on_hopper / finalize_alloc_conf_for_sleep / model imports all can
        # create a CUDA context). If we wait until run_rl, the context is already bound to device 0
        # and the trainer collides with the vLLM server's device-0 rollout (TRL aborts with
        # "same CUDA device ... for multiple distinct roles"). CUDA_VISIBLE_DEVICES only takes
        # effect before the first context, so this is the earliest safe point.
        _pin_trainer_devices_for_disaggregated()
        _drop_fla_on_hopper()  # Hopper fla guard (see _drop_fla_on_hopper)
        heartbeat("boot")
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
        tb = traceback.format_exc()
        traceback.print_exc()
        # Upload the FULL traceback under a phase-specific name (error_<phase>.txt) so the
        # train (sft/rl) root-cause error survives for debugging. heartbeat.json is
        # single-file/overwritten, so the per-phase error file is the persistent signal.
        try:
            err_name = error_artifact_name(RUN_MODE)
            err_path = f"/tmp/{err_name}"
            with open(err_path, "w") as f:
                f.write(tb)
            hf_upload_file(err_path, err_name)
        except Exception as up_err:
            print("error-upload warn:", up_err)
        try:
            heartbeat(f"error_{RUN_MODE}", error=str(e)[:500], diag=gpu_diagnostics())
        except Exception:
            heartbeat(f"error_{RUN_MODE}", error=str(e)[:500])
        # keep container alive briefly so logs flush, then exit non-zero -> restart
        time.sleep(10)
        raise


if __name__ == "__main__":
    main()

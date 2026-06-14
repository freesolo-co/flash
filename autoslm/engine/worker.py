"""On-GPU fine-tuning worker (default RTX 5090). Modes: sft | rl.

This module runs **on the RunPod GPU** provisioned by ``autoslm.flash``. It uses the
shared recipe (``autoslm.engine.recipe``) so SFT targets and RL rewards are rendered
and scored consistently.

Artifacts (adapter, metrics.json, heartbeat.json, checkpoints) are streamed to a
Hugging Face dataset repo. HF checkpoints give preemption resilience: if a worker is
recycled mid-run we resume from the latest uploaded checkpoint. Metrics are also
returned directly to the caller by the Flash endpoint.

Environment variables (set by the Flash endpoint / orchestrator):
  RUN_MODE      sft|rl
  SEED          int
  HF_REPO       Hugging Face dataset repo for artifacts (e.g. your-org/autoslm-runs)
  HUGGINGFACE_TOKEN
  RUN_ID        unique id for this run (namespacing in the repo)
  RL_STEPS, SFT_EPOCHS   optional overrides
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback

from autoslm.engine.accounting import RunMetrics

# Shared, substrate-neutral fine-tuning internals (live in this same package).
from autoslm.engine.recipe import RECIPE
from autoslm.envs.registry import load_environment
from autoslm.worker_spec import load_job_spec_from_env

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
ACTIVE_ENV = (
    load_environment(
        JOB_SPEC.environment.id,
        JOB_SPEC.environment.params,
        JOB_SPEC.environment.path,
    )
    if JOB_SPEC
    else load_environment("gsm8k")
)
# Thinking/reasoning mode: one flag per run, consumed identically by SFT rendering,
# RL rollouts, and serving. The env fallback serves the bench/no-JobSpec path.
THINKING = (
    JOB_SPEC.thinking
    if JOB_SPEC
    else os.environ.get("AUTOSLM_THINKING", "0") not in ("0", "false", "False")
)


# ---------------------------------------------------------------------------
# HF helpers (code-delivery + artifact channel; works without inbound network)
# ---------------------------------------------------------------------------
def error_artifact_name(mode: str) -> str:
    """Per-mode error filename (e.g. error_sft.txt) so a run's traceback is uploaded
    under a stable name even though heartbeat.json is single-file/overwritten."""
    return f"error_{mode}.txt"


def hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HUGGINGFACE_TOKEN"))


def hf_prefix() -> str:
    return f"{PHASE}/{RUN_ID}/seed{SEED}"


def hf_upload_file(local_path: str, repo_subpath: str, required: bool = False):
    """Upload one file to the run's HF prefix.

    ``required=True`` (the Vast completion artifacts DONE/metrics.json) retries and
    finally raises: on Vast there is no queue return value, so poll_vast_job decides
    success purely from these artifacts — a swallowed upload failure would make the
    control plane mark a finished run failed or retry already-completed work.
    """
    if not HF_REPO:
        return
    import time as _time

    attempts = 3 if required else 1
    for attempt in range(attempts):
        try:
            hf_api().upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"{hf_prefix()}/{repo_subpath}",
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            return
        except Exception as e:
            if required and attempt + 1 < attempts:
                print(f"hf_upload_file retry {attempt + 1}/{attempts}: {e}")
                _time.sleep(5 * (attempt + 1))
                continue
            if required:
                raise RuntimeError(f"required upload of {repo_subpath!r} failed: {e}") from e
            print("hf_upload_file warn:", e)
            return


def hf_upload_folder(local_dir: str, repo_subpath: str, required: bool = False):
    """Upload a folder to the run's HF prefix.

    ``required=True`` (the trained adapter) retries and finally raises on failure:
    swallowing it would mark the run done while deployment can never download the
    missing adapter. Optional artifacts (generations, logs) only warn.
    """
    if not HF_REPO:
        return
    import time as _time

    attempts = 3 if required else 1
    for attempt in range(attempts):
        try:
            hf_api().upload_folder(
                folder_path=local_dir,
                path_in_repo=f"{hf_prefix()}/{repo_subpath}",
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            return
        except Exception as e:
            if required and attempt + 1 < attempts:
                print(f"hf_upload_folder retry {attempt + 1}/{attempts}: {e}")
                _time.sleep(5 * (attempt + 1))
                continue
            if required:
                raise RuntimeError(f"required upload of {repo_subpath!r} failed: {e}") from e
            print("hf_upload_folder warn:", e)
            return


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
            token=os.environ.get("HUGGINGFACE_TOKEN"),
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


def heartbeat(stage: str, **kw):
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
    with open(p, "w") as f:
        json.dump(payload, f)
    hf_upload_file(p, "heartbeat.json")
    print("HEARTBEAT", json.dumps(payload))


# ---------------------------------------------------------------------------
# Decoding parity: render with the model's own chat template and one run-wide thinking
# flag (off by default), so SFT targets and RL rollouts use identical prompt
# formatting within a run.
# ---------------------------------------------------------------------------
def render_prompt(tokenizer, item) -> str:
    item = item if isinstance(item, dict) else {"question": item}
    msgs = ACTIVE_ENV.prompt_messages(item)
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
    fields), so loading a LoRA adapter onto any arch WITH weight conversions (the MoE
    tier; dense models have none) dies with ``TypeError: unexpected keyword argument
    'distributed_operation'`` (observed live: Qwen3.6-35B-A3B eval on A100). The
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
    if converter is None or getattr(converter, "_autoslm_compat", False):
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
    converter._autoslm_compat = True
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
    if os.environ.get("AUTOSLM_TEXT_ONLY", "1") in ("0", "false", "False"):
        return {}
    return {"language_model_only": True} if is_vl_checkpoint(model_id) else {}


def patch_vllm_language_model_only(model_id: str) -> bool:
    """Force ``language_model_only=True`` on vLLM engines created by third-party code
    (TRL's colocated GRPO rollout engine) for VL checkpoints. Returns True if patched."""
    extra = vllm_language_model_only_kwargs(model_id)
    if not extra:
        return False
    try:
        import vllm

        if getattr(vllm.LLM.__init__, "_autoslm_lmo_patched", False):
            return True
        orig = vllm.LLM.__init__

        def patched(self, *args, **kwargs):
            kwargs.setdefault("language_model_only", True)
            return orig(self, *args, **kwargs)

        patched._autoslm_lmo_patched = True
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

    targets = os.environ.get("LORA_TARGETS", "all-linear")
    rank = JOB_SPEC.train.lora_rank if JOB_SPEC else RECIPE.lora.rank
    alpha = JOB_SPEC.train.lora_alpha if JOB_SPEC else RECIPE.lora.alpha
    kwargs = {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": RECIPE.lora.dropout,
        "target_modules": targets,
        "task_type": "CAUSAL_LM",
    }
    if model_id and targets == "all-linear":
        exclude = lora_exclude_modules(model_id)
        if exclude:
            kwargs["exclude_modules"] = exclude
            print(f"[lora] excluding modules for {model_id}: {exclude}")
    return LoraConfig(**kwargs)


def model_quant(model_id: str) -> str:
    """Quantization tier for this model: catalog entry > AUTOSLM_QUANT env > bf16."""
    env_q = os.environ.get("AUTOSLM_QUANT")
    if env_q:
        return env_q
    try:
        from autoslm.catalog import MODELS

        info = MODELS.get(model_id)
        if info is not None:
            return info.quant
    except Exception as e:
        print("model_quant: catalog probe failed:", e)
    return "bf16"


def quant_weights_repo(model_id: str) -> str:
    """Pre-quantized weights repo for the QLoRA tier (or "" to quantize at load).

    AUTOSLM_QUANT_REPO env (A/B / one-off override) > catalog ``quant_repo``. The
    returned repo is what the trainer loads; the tokenizer/chat template still come
    from the base ``model_id``.
    """
    env_repo = os.environ.get("AUTOSLM_QUANT_REPO")
    if env_repo:
        return env_repo
    try:
        from autoslm.catalog import MODELS

        info = MODELS.get(model_id)
        return getattr(info, "quant_repo", "") if info else ""
    except Exception as e:
        print("quant_weights_repo: catalog probe failed:", e)
        return ""


def qlora_model_init_kwargs(prequantized: bool = False) -> dict:
    """Model-load kwargs for the 4-bit QLoRA tier (large MoEs on one consumer GPU).

    ``prequantized``: the checkpoint already carries a bitsandbytes quantization
    config (a ``quant_repo``), so we must not pass a second BitsAndBytesConfig —
    transformers would ignore the checkpoint's and warn.
    """
    import torch

    kwargs: dict = {"dtype": torch.bfloat16}
    if not prequantized:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return kwargs


def run_sft():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    t_start = time.time()
    heartbeat("sft_start")
    wait_for_gpu()
    model_id = JOB_SPEC.model if JOB_SPEC else RECIPE.hf_model_id
    # QLoRA tier may load a pre-quantized export (~3.5x smaller download/disk);
    # the tokenizer below still comes from the base id.
    weights_id = model_id
    if model_quant(model_id) == "4bit-qlora":
        weights_id = quant_weights_repo(model_id) or model_id
    download_seconds = prefetch_model(weights_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build SFT text dataset (seeded shuffle for reproducibility)
    train = ACTIVE_ENV.dataset("train")
    rng = random.Random(SEED)
    rng.shuffle(train)
    max_examples = int(os.environ.get("SFT_MAX_EXAMPLES", "0"))
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

    # SFT_MAX_STEPS>0 caps optimizer steps (used by the cheap pre-flight smoke).
    max_steps = int(os.environ.get("SFT_MAX_STEPS", "0"))
    cfg_kwargs = {
        "output_dir": out_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": per_device_bs,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": sft_lr,
        "warmup_ratio": RECIPE.sft.warmup_frac,
        "logging_steps": 10,
        "save_steps": int(os.environ.get("SFT_SAVE_STEPS", str(sft_save_default))),
        "save_total_limit": 1,
        "max_length": sft_max_len,
        "bf16": True,
        "report_to": [],
        "seed": SEED,
        "gradient_checkpointing": True,
        "completion_only_loss": False,
    }
    if max_steps > 0:
        cfg_kwargs["max_steps"] = max_steps
    # A/B-gated throughput knobs (default off until the live A/B shows a win):
    # SFT_PACKING=1 packs short examples into full max_length sequences (GSM8K targets
    # are far shorter than max_seq_len, so unpacked batches are mostly pad tokens).
    if os.environ.get("SFT_PACKING", "0") not in ("0", "false"):
        cfg_kwargs["packing"] = True
        print("[sft] packing enabled (SFT_PACKING)")
    # SFT_LIGER=1 swaps in Liger's fused CE/RMSNorm/RoPE kernels (needs the
    # liger-kernel package on the worker: AUTOSLM_WORKER_EXTRA_DEPS=liger-kernel).
    if os.environ.get("SFT_LIGER", "0") not in ("0", "false"):
        cfg_kwargs["use_liger_kernel"] = True
        print("[sft] liger kernels enabled (SFT_LIGER)")
    quant = model_quant(model_id)
    if quant == "4bit-qlora":
        # Large-MoE tier: 4-bit NF4 base + LoRA adapters (e.g. Qwen3.6-35B-A3B on a 5090).
        _patch_peft_weight_converter_compat()  # MoE adapter (re)load, e.g. ckpt resume
        cfg_kwargs["model_init_kwargs"] = qlora_model_init_kwargs(
            prequantized=weights_id != model_id
        )
        print(f"[sft] loading {weights_id} in 4-bit (QLoRA tier)")
    else:
        # Explicit bf16 + no auto device-map: TRL/transformers-5 string loading can
        # otherwise fall back to fp32 (2x VRAM; observed 18.6 GB for a 4.66B model) or
        # accelerate-offload large models to meta ("expected device meta but got
        # cuda:0" in backward on the 9B).
        cfg_kwargs["model_init_kwargs"] = {"dtype": "bfloat16", "device_map": None}
    cfg = TRLSFTConfig(**cfg_kwargs)
    # Pass model as a string id + tokenizer as processing_class so TRL takes the
    # text/causal-LM path (not the VLM processor path) for this multimodal checkpoint.
    trainer = SFTTrainer(
        model=weights_id,
        args=cfg,
        train_dataset=ds,
        peft_config=make_lora(model_id),
        processing_class=tok,
        callbacks=[make_checkpoint_upload_callback()],
    )

    t_train = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    train_wall = time.time() - t_train

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


def rl_per_device_comps() -> int:
    """Per-device *completion* micro-batch for GRPO (TRL counts completions, not prompts).

    This, not grad-accum, sets peak trainer VRAM: the logprob pass materializes
    fp32 logits of shape [per_device, seq_len, vocab]. At Qwen's ~152k vocab a
    2048-token thinking sequence is ~1.25 GiB per completion — micro-batch 8 OOMs a
    24 GB card (observed live: 6.96 GiB single alloc on an RTX 4090), so thinking
    runs default to 2. compute_grpo_batching puts the difference into grad-accum,
    leaving the effective batch unchanged. RL_PER_DEVICE_PROMPTS overrides.
    """
    return int(os.environ.get("RL_PER_DEVICE_PROMPTS", "2" if THINKING else "8"))


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
    """The freesolo SDK's GRPO knobs (datums parity), read off the job spec's
    ``[train]`` (``TrainSpec``). Missing (None) -> recipe defaults apply.

    Same parameters the deleted client-side datums.py used, now honored server-side:
    group_size, temperature, max_tokens (completion budget), kl_penalty_coef (the KL
    beta), advantage_clip (centered-advantage clip), and thinking_length_penalty_coef
    (a per-<think>-token reward deduction). The SDK ships these in [train] (they are
    TrainSpec fields), so only forward the ones actually set."""
    if not JOB_SPEC:
        return {}
    t = JOB_SPEC.train
    pairs = {
        "group_size": t.group_size,
        "temperature": t.temperature,
        "max_tokens": t.max_tokens,
        "kl_penalty_coef": t.kl_penalty_coef,
        "advantage_clip": t.advantage_clip,
        "thinking_length_penalty_coef": t.thinking_length_penalty_coef,
    }
    return {k: v for k, v in pairs.items() if v is not None}


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


def _init_adapter_model(model_id: str, tokenizer):
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
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adir, is_trainable=True)
    return model, None


def run_rl():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    t_start = time.time()
    heartbeat("rl_start")
    wait_for_gpu()
    model_id = JOB_SPEC.model if JOB_SPEC else RECIPE.hf_model_id
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
    _pps_default = _t.batch_size if _t and _t.batch_size is not None else rl.prompts_per_step
    prompts_per_step = int(os.environ.get("RL_PROMPTS_PER_STEP", str(_pps_default)))
    group_size = int(os.environ.get("RL_GROUP_SIZE", gcfg.get("group_size") or rl.group_size))
    # temperature: explicit None check, NOT `or` — a configured 0.0 (greedy/deterministic
    # rollouts) must be honored, not fall back to the recipe sampling temperature.
    _gcfg_temp = gcfg.get("temperature")
    _temperature = float(_gcfg_temp if _gcfg_temp is not None else rl.sampling_temperature)
    _kl_beta = float(gcfg.get("kl_penalty_coef") or 0.0)
    _adv_clip = float(gcfg.get("advantage_clip") or 0.0)
    _think_penalty = float(gcfg.get("thinking_length_penalty_coef") or 0.0)
    sleep_mode = os.environ.get("RL_VLLM_SLEEP", "1") not in ("0", "false", "False")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train = ACTIVE_ENV.dataset("train")
    rng = random.Random(SEED)
    rng.shuffle(train)
    prompts = [{"prompt": render_prompt(tok, ex), "example": ex} for ex in train]
    # The colocated vLLM engine's model length is the hard cap on prompt+completion at
    # rollout. Size it (honoring RL_VLLM_MAX_LEN) and derive the prompt budget from it so a
    # bigger engine or a smaller completion automatically admits longer prompts (rather than
    # a fixed rl.max_prompt_len that no env override could lift).
    _max_completion = int(
        os.environ.get(
            "RL_MAX_COMPLETION",
            gcfg.get("max_tokens")
            or (rl.max_completion_len_thinking if THINKING else rl.max_completion_len),
        )
    )
    vllm_max_len = int(
        os.environ.get("RL_VLLM_MAX_LEN", str(max(1024, rl.max_prompt_len + _max_completion)))
    )
    # The engine must fit completion + at least some prompt. If RL_VLLM_MAX_LEN is set below
    # the completion budget, no prompt can ever fit — fail fast here rather than passing a
    # 1-token budget that lets prompts through and then OOMs/overflows mid-rollout.
    if vllm_max_len <= _max_completion:
        raise ValueError(
            f"RL_VLLM_MAX_LEN={vllm_max_len} leaves no room for the {_max_completion}-token "
            "completion; raise RL_VLLM_MAX_LEN or lower RL_MAX_COMPLETION"
        )
    prompt_budget = vllm_max_len - _max_completion
    # TRL 1.5's GRPOConfig has no max_prompt_length and does NOT truncate prompts, so a prompt
    # that leaves no room for the completion within the engine length would fail mid-rollout
    # AFTER the paid worker is provisioned. Drop prompts that don't fit the budget up front.
    # render_prompt returns an apply_chat_template(tokenize=False) string that already carries
    # the special tokens, so tokenize with add_special_tokens=False (the default re-adds
    # BOS/EOS and over-counts).
    kept = [
        p
        for p in prompts
        if len(tok(p["prompt"], add_special_tokens=False).input_ids) <= prompt_budget
    ]
    if len(kept) < len(prompts):
        print(
            f"[rl] dropped {len(prompts) - len(kept)} prompts over the {prompt_budget}-token "
            f"prompt budget (engine {vllm_max_len} - completion {_max_completion})"
        )
    if not kept:
        raise ValueError(
            f"every training prompt exceeds the {prompt_budget}-token prompt budget (engine "
            f"{vllm_max_len} - completion {_max_completion}); raise RL_VLLM_MAX_LEN, lower "
            "RL_MAX_COMPLETION, or shorten the environment's prompts"
        )
    prompts = kept
    ds = Dataset.from_list(prompts)

    def reward_fn(completions, **kwargs):
        # Score the <think>-stripped text (graded_text), then — datums parity — deduct
        # the thinking-length penalty computed from the RAW completion's <think> span.
        examples = kwargs.get("example")
        rewards = []
        for raw, ex in zip(completions, examples, strict=False):
            r = ACTIVE_ENV.reward(graded_text(raw), ex)
            if _think_penalty > 0 and THINKING:
                r -= _think_penalty * think_token_count(raw, tok)
            rewards.append(r)
        return rewards

    # TRL's per_device_train_batch_size counts COMPLETIONS, not prompts. Size grad-accum so
    # the global completion batch = prompts_per_step * group_size, i.e. each optimizer step
    # actually optimizes `prompts_per_step` prompts. The per-device *completion* micro-batch
    # is the VRAM knob (thinking-aware; see rl_per_device_comps).
    per_device_comps = rl_per_device_comps()
    batching = compute_grpo_batching(prompts_per_step, group_size, per_device_comps)
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
        "use_vllm": True,
        "vllm_mode": "colocate",
        # Bound the colocated engine's KV allocation to what GRPO actually needs (computed
        # above as vllm_max_len). Without this, vLLM sizes its KV cache for the model's FULL
        # context (262k for Qwen3-4B-Instruct-2507) and refuses to start on a consumer GPU.
        "vllm_max_model_length": vllm_max_len,
        # Colocate shares one GPU between the policy model and the vLLM rollout engine; on a
        # 32 GB RTX 5090 a 4B policy (~8 GB) + vLLM weights/KV easily OOMs. RL_VLLM_GPU_UTIL
        # sizes vLLM's pool; RL_VLLM_SLEEP offloads its weights between steps (saves memory
        # but costs a reload each step). Default util 0.45 fits colocate on a 5090.
        "vllm_gpu_memory_utilization": float(os.environ.get("RL_VLLM_GPU_UTIL", "0.45")),
        "vllm_enable_sleep_mode": sleep_mode,
        "logging_steps": 1,
        "save_steps": int(
            os.environ.get(
                "RL_SAVE_STEPS", str(_t.save_every if _t and _t.save_every is not None else 20)
            )
        ),
        "save_total_limit": 1,
        "bf16": True,
        "report_to": [],
        "seed": SEED,
        "gradient_checkpointing": True,
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
    }
    # Adapter init: continue training the SFT adapter (peft_config=None, model is the
    # loaded PeftModel) when train.init_from_adapter is set, else a fresh LoRA on the
    # string model id (model_init_kwargs forces bf16 — TRL string-loading can fall back
    # to fp32 and double VRAM).
    init_model, init_peft = _init_adapter_model(model_id, tok)
    if init_peft is not None:
        grpo_kwargs["model_init_kwargs"] = {"dtype": "bfloat16"}
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
    cfg = GRPOConfig(**grpo_kwargs)
    setup_seconds = time.time() - t_start
    heartbeat("rl_train_start", setup_seconds=setup_seconds)

    # VL checkpoints (Qwen3.5/3.6) train text-only: make TRL's colocated rollout
    # engine skip the vision tower (VRAM + 5090 PTX-compat; see the patch docstring).
    patch_vllm_language_model_only(model_id)
    hb_cb = make_reward_heartbeat_callback()
    trainer = GRPOTrainer(
        model=init_model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=reward_fn,
        peft_config=init_peft,
        processing_class=tok,
        callbacks=[hb_cb, make_checkpoint_upload_callback()],
    )
    t_train = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    train_wall = time.time() - t_train
    reward_history = list(getattr(hb_cb, "reward_history", []))

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
        arm="runpod",
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
            "renderer": "autoslm_env",
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
        token=os.environ.get("HUGGINGFACE_TOKEN"),
    )
    adir = os.path.join("/tmp/evdl", adapter_prefix, "adapter")
    return adir if os.path.isdir(adir) else None


def _finalize(metrics: RunMetrics, adapter_dir: str):
    metrics.save("/tmp/metrics.json")
    # Required: on Vast these two artifacts ARE the success signal (no queue return),
    # so a swallowed upload would make the control plane fail/retry a finished run.
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
        if HF_REPO:
            from huggingface_hub import hf_hub_download

            try:
                hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    filename=f"{hf_prefix()}/DONE",
                    token=os.environ.get("HUGGINGFACE_TOKEN"),
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
                        token=os.environ.get("HUGGINGFACE_TOKEN"),
                    )
                    import shutil

                    shutil.copy(got, "/tmp/metrics.json")
                    sys.stdout.flush()
                    os._exit(0)
                except Exception as e:
                    raise SystemExit(f"DONE present but metrics.json unavailable: {e}") from e
        heartbeat("boot")
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
        # single-file/overwritten, so the per-phase error file is the durable signal.
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

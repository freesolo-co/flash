"""LoRA config + warm-start adapter loading for the fine-tuning worker."""

from __future__ import annotations

import os

from flash.engine.recipe import RECIPE
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.lora import (
    adapter_is_vl_warmstart,
    assert_adapter_delta_nonzero,
    assert_adapter_load_clean,
    assert_lora_applied,
    is_vl_checkpoint,
)
from flash.engine.worker.perf import optimal_attn_impl


def make_lora(model_id: str | None = None):
    """Build LoRA config targeting all linear layers (VL models included: the vision tower /
    projector / MTP linears are adapted too; on examples without images they simply get no gradient)."""
    from peft import LoraConfig

    targets = "all-linear"
    rank = _w.JOB_SPEC.train.lora_rank if _w.JOB_SPEC else RECIPE.lora.rank
    alpha = _w.JOB_SPEC.train.lora_alpha if _w.JOB_SPEC else RECIPE.lora.alpha
    model_revision = getattr(_w.JOB_SPEC, "model_revision", "") if _w.JOB_SPEC else ""
    kwargs = {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": RECIPE.lora.dropout,
        "target_modules": targets,
        "task_type": "CAUSAL_LM",
        "revision": model_revision or None,
    }
    # PiSSA removed: it mutates the base, so its adapter corrupts serve + warm-start on the unmodified base.
    kwargs["init_lora_weights"] = True
    print(
        "[lora] init_lora_weights=True (standard zero-B; PiSSA removed for serve/warm-start safety)"
    )
    # rsLoRA removed: ~5.6x effective LR inflation with catalog LRs causes SFT divergence + serve corruption.
    kwargs["use_rslora"] = False
    return LoraConfig(**kwargs)


def stamp_adapter_provenance(model, model_id: str, model_revision: str = "") -> None:
    """Validate and stamp the saved default adapter's immutable base identity."""
    configs = getattr(model, "peft_config", None) or {}
    config = configs.get("default") if isinstance(configs, dict) else None
    if config is None:
        raise RuntimeError("trained model has no default PEFT adapter config")
    current_base = str(getattr(config, "base_model_name_or_path", "") or "").strip()
    if current_base and current_base != model_id:
        raise RuntimeError(
            f"adapter base model {current_base!r} does not match validated target {model_id!r}"
        )
    current_revision = str(getattr(config, "revision", "") or "").strip()
    if current_revision and current_revision != model_revision:
        raise RuntimeError("adapter base revision does not match the validated target commit")
    config.base_model_name_or_path = model_id
    config.revision = model_revision or None


def prepare_fresh_lora_base(
    model_source: str,
    model_id: str,
    model_init_kwargs: dict,
    *,
    force: bool = False,
    phase: str = "train",
    model_revision: str = "",
):
    """Prepare the correct base object/path for fresh-LoRA training.

    PEFT expands ``target_modules`` against the concrete model object it wraps. TRL's default string
    loader can resolve Qwen VL checkpoints to a language-only tree, while OPD and serving use the full
    image-text tree. Therefore every fresh LoRA on a VL checkpoint must preload
    ``AutoModelForImageTextToText`` so SFT/GRPO/OPD adapter tensor sets stay identical, including
    zero-gradient vision-tower LoRA tensors on examples without images. Non-VL checkpoints keep the original
    model path for TRL's normal loader. ``force`` pins the full-VL loader when the caller already knows
    the checkpoint is VL, so a transient config-probe failure cannot send the fresh stage back to a
    language-only loader.
    """
    loader_revision = str(model_init_kwargs.get("revision") or "").strip()
    if loader_revision and model_revision and loader_revision != model_revision:
        raise ValueError(
            "fresh adapter architecture probe revision must match from_pretrained revision"
        )
    effective_revision = model_revision or loader_revision
    if effective_revision and not loader_revision:
        model_init_kwargs = {**model_init_kwargs, "revision": effective_revision}
    if not (force or is_vl_checkpoint(model_id, revision=effective_revision)):
        return model_source
    from transformers import AutoModelForImageTextToText

    print(
        f"[{phase}] VL checkpoint: loading full multimodal model for fresh LoRA target parity "
        f"({model_source})"
    )
    return AutoModelForImageTextToText.from_pretrained(
        model_source, trust_remote_code=True, **model_init_kwargs
    )


def require_vllm_for_rollout_func(use_rollout_func: bool, use_vllm: bool, model_id: str) -> None:
    """Fail fast when multi-turn GRPO needs colocated vLLM but it's disabled."""
    if use_rollout_func and not use_vllm:
        raise RuntimeError(
            f"multi-turn GRPO needs colocated vLLM, which is disabled for {model_id}. "
            "Use a single-turn environment for this model, or a model tier that keeps "
            "vLLM enabled for rollouts."
        )


def _assert_warmstart_adapter_applied(model, model_id: str, load_result) -> None:
    """Fail closed if a warm-start adapter didn't fully apply: clean load (no silently-discarded
    keys), LoRA actually present, and a non-zero delta. Shared by the VL and non-VL load paths."""
    assert_adapter_load_clean(load_result, model_id)
    assert_lora_applied(model, model_id)
    assert_adapter_delta_nonzero(model, model_id)


def _init_adapter_model(model_id: str):
    """Load init_from_adapter as a trainable PeftModel that CONTINUES the prior adapter in place, or
    return ``(model_id, fresh LoRA config)`` when no warm-start is requested.

    Warm-start (SFT→GRPO/OPD) keeps ONE LoRA adapter for the whole pipeline: it loads the prior
    adapter as the trainable ``default`` and training continues it — the saved/deployed adapter is
    that same rank-r adapter on the ORIGINAL catalog base, so it serves directly (no merge into the
    base, no rank-stack recombine). VL checkpoints (Qwen3.5/3.6) load the FULL multimodal model so
    (a) the adapter's ``language_model.*`` keys line up with the module tree and (b) the trainer arch
    matches the VL vLLM rollout engine, so TRL's/opd's weight-sync lands the continued policy and
    rollouts stay on-policy from the warm-started weights (the arch match is the fix for the #296
    trainer<->vLLM mismatch — continuing a *language-only* adapter is what collapsed VL SFT to base).
    Non-VL checkpoints use the lighter causal-LM loader. Returns ``(model, None)`` for warm-start (TRL
    and opd train the LOADED adapter, ``peft_config=None``); for a fresh LoRA TRL/opd build it from the
    returned ``LoraConfig`` — VL fresh runs load the full multimodal base via ``prepare_fresh_lora_base``.
    """
    prefix = _w.JOB_SPEC.train.init_from_adapter if _w.JOB_SPEC else ""
    model_revision = getattr(_w.JOB_SPEC, "model_revision", "") if _w.JOB_SPEC else ""
    if not prefix:
        return model_id, make_lora(model_id)
    adir = _download_adapter(prefix)
    if not adir:
        raise RuntimeError(
            "the warm-start source adapter could not be downloaded from the artifact store; "
            "refusing to silently start from the base model. Verify the source run and access, "
            "or omit init_from_adapter to train a fresh LoRA."
        )
    print("[init-adapter] continuing the prepared source LoRA")
    _attn = optimal_attn_impl()
    attn_kw = {"attn_implementation": _attn} if _attn else {}

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    # Only the base class differs: VL checkpoints load the full multimodal model (so the adapter's
    # language_model.* keys line up with the module tree AND the trainer arch matches the VL vLLM
    # rollout engine); non-VL uses the lighter causal-LM loader.
    is_vl = adapter_is_vl_warmstart(adir, model_id, revision=model_revision)
    if is_vl:
        print("[init-adapter] VL checkpoint: continuing the adapter on the full multimodal base")
    base_cls = AutoModelForImageTextToText if is_vl else AutoModelForCausalLM
    from flash.engine.worker.hf import model_revision_kwargs

    base = base_cls.from_pretrained(
        model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        **attn_kw,
        **model_revision_kwargs(model_revision),
    )
    # PeftModel.from_pretrained builds the wrapper + a trainable "default" adapter, but it doesn't
    # forward the HF `_checkpoint_conversion_mapping`; re-load "default" with key_mapping so a VL
    # checkpoint's keys remap onto the current module tree (mapping is None/no-op for non-VL).
    model = PeftModel.from_pretrained(base, adir, is_trainable=True)
    key_mapping = getattr(base, "_checkpoint_conversion_mapping", None)
    load_result = model.load_adapter(
        adir, adapter_name="default", is_trainable=True, key_mapping=key_mapping
    )
    _assert_warmstart_adapter_applied(model, model_id, load_result)
    return model, None


def _resolve_adapter_ref(adapter_ref: str) -> tuple[str, str] | None:
    """Resolve the INTERNAL adapter storage reference into (repo, prefix).

    Users write the short ``<run_id>[/step-N]`` form (see ``flash.schema.parse_checkpoint_ref``);
    the control plane resolves it against the source run's metadata into the storage reference
    the worker receives here (``flash.runner._resolve_init_from_adapter``). Per-step deployable
    adapters live at the identical ``<prefix>/adapter`` layout in the artifact repo (see
    ``publish_deployable_checkpoint``), so the same download path serves both.
    """
    from flash.schema import parse_adapter_storage_ref

    return parse_adapter_storage_ref(adapter_ref)


def _download_adapter(adapter_prefix: str | None) -> str | None:
    """Download an init_from_adapter LoRA to /tmp/evdl/<prefix>/adapter and return its dir.

    ``adapter_prefix`` is the internal storage ref resolved by the control plane:
    ``<owner>/<repo>:<phase>/<run_id>[/checkpoints/step-N]``.
    """
    if not adapter_prefix:
        return None
    resolved = _resolve_adapter_ref(adapter_prefix)
    if not resolved:
        return None
    repo, prefix = resolved
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            allow_patterns=[f"{prefix}/adapter/*"],
            local_dir="/tmp/evdl",
            token=os.environ.get("HF_TOKEN"),
            revision=(_w.JOB_SPEC.train.init_from_adapter_revision if _w.JOB_SPEC else None)
            or None,
        )
    except Exception:
        raise RuntimeError(
            "the prepared warm-start source adapter could not be downloaded"
        ) from None
    adir = os.path.join("/tmp/evdl", prefix, "adapter")
    return adir if os.path.isdir(adir) else None

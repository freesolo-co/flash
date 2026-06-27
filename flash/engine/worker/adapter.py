"""LoRA config + warm-start adapter loading for the fine-tuning worker."""

from __future__ import annotations

import os
import re

from flash.engine.recipe import RECIPE
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.lora import (
    assert_adapter_delta_nonzero,
    assert_adapter_load_clean,
    assert_lora_applied,
    remap_vl_adapter_dir,
)
from flash.engine.worker.perf import optimal_attn_impl


def make_lora(model_id: str | None = None):
    """Build LoRA config targeting all linear layers; vision tower excluded for VL models."""
    from peft import LoraConfig

    targets = "all-linear"
    rank = _w.JOB_SPEC.train.lora_rank if _w.JOB_SPEC else RECIPE.lora.rank
    alpha = _w.JOB_SPEC.train.lora_alpha if _w.JOB_SPEC else RECIPE.lora.alpha
    kwargs = {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": RECIPE.lora.dropout,
        "target_modules": targets,
        "task_type": "CAUSAL_LM",
    }
    # PiSSA removed: it mutates the base, so its adapter corrupts serve + warm-start on the unmodified base.
    kwargs["init_lora_weights"] = True
    print(
        "[lora] init_lora_weights=True (standard zero-B; PiSSA removed for serve/warm-start safety)"
    )
    # rsLoRA removed: ~5.6x effective LR inflation with catalog LRs causes SFT divergence + serve corruption.
    kwargs["use_rslora"] = False
    if model_id and targets == "all-linear":
        exclude = _w.lora_exclude_modules(model_id)
        if exclude:
            kwargs["exclude_modules"] = exclude
            print(f"[lora] excluding modules for {model_id}: {exclude}")
    return LoraConfig(**kwargs)


def require_vllm_for_rollout_func(use_rollout_func: bool, use_vllm: bool, model_id: str) -> None:
    """Fail fast when multi-turn GRPO needs colocated vLLM but it's disabled."""
    if use_rollout_func and not use_vllm:
        raise RuntimeError(
            f"multi-turn GRPO needs colocated vLLM, which is disabled for {model_id}. "
            "Use a single-turn environment for this model, or a model tier that keeps "
            "vLLM enabled for rollouts."
        )


def _init_adapter_model(model_id: str):
    """Load init_from_adapter as a trainable PeftModel, or return model_id + fresh LoRA."""
    prefix = _w.JOB_SPEC.train.init_from_adapter if _w.JOB_SPEC else ""
    if not prefix:
        return model_id, make_lora(model_id)
    adir = _download_adapter(prefix)
    if not adir:
        raise RuntimeError(
            f"train.init_from_adapter={prefix!r} could not be downloaded from the artifact "
            "store (wrong/missing prefix or no access); refusing to silently start GRPO from "
            "the base model. Fix the adapter prefix / HF credentials, or omit "
            "init_from_adapter to train a fresh LoRA."
        )
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    print(f"[init-adapter] initializing LoRA from {prefix}")
    # VL adapters saved with .language_model. key infix must be remapped before PeftModel.from_pretrained
    # or peft silently discards mismatched keys and trains a fresh LoRA from scratch.
    remap_vl_adapter_dir(adir, model_id)
    _attn = optimal_attn_impl()
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        **({"attn_implementation": _attn} if _attn else {}),
    )
    model = PeftModel.from_pretrained(base, adir, is_trainable=True)
    # from_pretrained uses strict=False and only warns on key mismatches; reload to catch silent discards.
    # key_mapping must match from_pretrained's remapping for renamed-arch checkpoints.
    key_mapping = getattr(base, "_checkpoint_conversion_mapping", None)
    load_result = model.load_adapter(
        adir, adapter_name="default", is_trainable=True, key_mapping=key_mapping
    )
    assert_adapter_load_clean(load_result, model_id)
    assert_lora_applied(model, model_id)
    assert_adapter_delta_nonzero(model, model_id)
    return model, None


def _resolve_adapter_ref(adapter_ref: str) -> tuple[str, str] | None:
    """Resolve an adapter_ref string into (repo, prefix)."""
    adapter_ref = adapter_ref.strip()
    match = re.fullmatch(
        r"(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*):"
        r"(?P<phase>sft|rl)/(?P<run_id>[A-Za-z0-9][A-Za-z0-9._-]{0,127})/seed(?P<seed>\d+)",
        adapter_ref,
    )
    if not match:
        return None
    repo, phase, run_id, seed = match.groups()
    return repo, f"{phase}/{run_id}/seed{seed}"


def _download_adapter(adapter_prefix: str | None) -> str | None:
    """Download a LoRA adapter to /tmp/evdl/<prefix>/adapter and return its path."""
    if not adapter_prefix:
        return None
    resolved = _resolve_adapter_ref(adapter_prefix)
    if not resolved:
        return None
    repo, prefix = resolved
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

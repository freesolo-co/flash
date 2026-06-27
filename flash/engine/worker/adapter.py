"""LoRA config + warm-start adapter loading for the fine-tuning worker."""

from __future__ import annotations

import os
import re

from flash.engine.recipe import RECIPE
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.lora import (
    adapter_is_vl_warmstart,
    assert_adapter_delta_nonzero,
    assert_adapter_load_clean,
    assert_lora_applied,
    recombine_lora_adapters,
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

    print(f"[init-adapter] initializing LoRA from {prefix}")
    _attn = optimal_attn_impl()
    attn_kw = {"attn_implementation": _attn} if _attn else {}

    if adapter_is_vl_warmstart(adir, model_id):
        # VL checkpoints (Qwen3.5/3.6): MERGE the SFT into the base and train a FRESH LoRA on the
        # merged weights, instead of continuing the live SFT LoRA. Continuing it makes the colocated
        # vLLM rollout engine AND the KL reference key off the BARE base — the SFT only reaches vLLM
        # through the text-trainer<->VL-vLLM weight-sync, which round-trips poorly for these
        # `*ForConditionalGeneration` models, so GRPO rolls out base-verbose and collapses a working
        # concise-thinking SFT back to base (observed: every Qwen3.5 GRPO reverts; non-VL MiniCPM
        # does not). We merge into the FULL multimodal model (NOT the text-only tree) so the saved
        # checkpoint keeps the VL config + ``language_model.*`` keys that BOTH the trainer reload and
        # vLLM's VL loader (language_model_only skips the vision weights) expect — a text-only
        # AutoModelForCausalLM merge saves a Qwen3_5TextConfig that vLLM's VL loader rejects. The SFT
        # adapter was trained against the full VL model, so its keys match here WITHOUT the infix strip.
        from transformers import AutoModelForImageTextToText

        base = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype="bfloat16", trust_remote_code=True, **attn_kw
        )
        model = PeftModel.from_pretrained(base, adir, is_trainable=False)
        key_mapping = getattr(base, "_checkpoint_conversion_mapping", None)
        load_result = model.load_adapter(
            adir, adapter_name="default", is_trainable=False, key_mapping=key_mapping
        )
        assert_adapter_load_clean(load_result, model_id)
        assert_lora_applied(model, model_id)
        assert_adapter_delta_nonzero(model, model_id)
        import gc
        import tempfile

        merged = model.merge_and_unload()
        # UNIQUE per call (mkdtemp): a fixed /tmp/flash_sft_merged would let two GRPO warm-starts on
        # the SAME host clobber each other's merged weights (or load a partially-written tree). The
        # dir is the run's training base, so it persists for the run (the worker is ephemeral); the
        # old fixed path never cleaned up either, so uniqueness adds no leak it didn't already have.
        merged_dir = tempfile.mkdtemp(prefix="flash_sft_merged_")
        merged.save_pretrained(merged_dir, safe_serialization=True)
        from transformers import AutoProcessor

        # processor (preferred for VL) so vLLM/loaders find tokenizer + image-processor config; fall
        # back to the bare tokenizer if no processor is published.
        try:
            AutoProcessor.from_pretrained(model_id, trust_remote_code=True).save_pretrained(merged_dir)
        except Exception:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).save_pretrained(merged_dir)
        del base, model, merged
        gc.collect()
        print(f"[init-adapter] merged VL SFT {prefix!r} -> {merged_dir}; training a fresh LoRA on it")
        # Record the SFT adapter dir: the saved GRPO LoRA will be SFT-less (trained on the merged
        # base), so finalize MUST recombine it with this SFT before deploy (recombined_warmstart_
        # adapter_dir). The dir persists for the run (worker is ephemeral; init's download stays).
        _w._VL_WARMSTART_SFT_DIR = adir
        return merged_dir, make_lora(merged_dir)

    # Non-VL checkpoints (e.g. MiniCPM): the continued-LoRA path works (GRPO keeps the SFT behavior),
    # so keep it — TRL trains the LOADED adapter (peft_config=None).
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="bfloat16", trust_remote_code=True, **attn_kw
    )
    model = PeftModel.from_pretrained(base, adir, is_trainable=True)
    key_mapping = getattr(base, "_checkpoint_conversion_mapping", None)
    load_result = model.load_adapter(
        adir, adapter_name="default", is_trainable=True, key_mapping=key_mapping
    )
    assert_adapter_load_clean(load_result, model_id)
    assert_lora_applied(model, model_id)
    assert_adapter_delta_nonzero(model, model_id)
    return model, None


def recombined_warmstart_adapter_dir(src_adapter_dir: str) -> str | None:
    """For a VL merge-into-base warm-start GRPO run, return a NEW dir holding the SFT⊕GRPO
    recombined adapter (deployable on the ORIGINAL catalog base); otherwise ``None``.

    ``_init_adapter_model`` (VL path, #296) merges the SFT into the base and trains a FRESH LoRA on
    the merged weights — so ``src_adapter_dir`` (the just-saved trainer adapter) is GRPO-ONLY and,
    deployed on the catalog base, drops the SFT (served output collapses to ~base: malformed,
    SFT-less filters). Recombining the original SFT LoRA back in produces a rank-(r_sft+r_grpo)
    adapter that reproduces ``base + SFT + GRPO`` on the unmodified base.

    Gated on the SFT dir ``_init_adapter_model`` recorded when (and only when) it took the VL merge
    path — so this is a NO-OP for the continued-adapter (non-VL) path, which already carries the SFT
    in its saved adapter, and for fresh-LoRA runs (no ``init_from_adapter``). When the merge DID
    happen, the recombine is REQUIRED: if the recorded SFT dir has vanished, raise rather than
    silently ship the SFT-less GRPO adapter (the exact broken deploy this guards against).
    """
    sft_adir = getattr(_w, "_VL_WARMSTART_SFT_DIR", None)
    if not sft_adir:
        return None
    if not os.path.isdir(sft_adir):
        raise RuntimeError(
            f"VL warm-start merged the SFT into the base at init but its adapter dir {sft_adir!r} is "
            "gone at finalize — the saved GRPO adapter is SFT-less and cannot be recombined. Refusing "
            "to deploy an SFT-less adapter (re-run; check /tmp/evdl was not evicted mid-run)."
        )

    import fnmatch
    import shutil
    import tempfile

    from flash.engine.worker.hf import _CHECKPOINT_TRAINER_STATE

    out_dir = tempfile.mkdtemp(prefix="flash_recomb_adapter_")
    try:
        rank = recombine_lora_adapters(sft_adir, src_adapter_dir, out_dir)
        # Carry tokenizer/aux files from the raw save (serving uses the base tokenizer, but keep the
        # deployed dir at parity with the un-recombined save); the recombined config+weights stay.
        # Skip trainer state — for the per-step path src is a `checkpoint-<n>` dir carrying optimizer/
        # scheduler state that the deployable adapter must not duplicate.
        for name in os.listdir(src_adapter_dir):
            if name in ("adapter_model.safetensors", "adapter_config.json", "adapter_model.bin"):
                continue
            if any(fnmatch.fnmatch(name, pat) for pat in _CHECKPOINT_TRAINER_STATE):
                continue
            src = os.path.join(src_adapter_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(out_dir, name))
    except Exception:
        # recombine (or the aux copy) raised — remove the just-created temp dir so a caller that
        # catches and continues (the per-step publish in hf.py) doesn't accumulate
        # flash_recomb_adapter_* dirs under /tmp across repeated failures.
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    print(
        f"[recombine] VL warm-start: stacked SFT⊕GRPO -> rank-{rank} deployable adapter at "
        f"{out_dir} (reproduces base+SFT+GRPO on the catalog base)"
    )
    return out_dir


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

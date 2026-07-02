"""LoRA config + warm-start adapter loading for the fine-tuning worker."""

from __future__ import annotations

import os

from flash.catalog import serving_lora_rank_cap
from flash.engine.recipe import RECIPE
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.lora import (
    adapter_is_vl_warmstart,
    assert_adapter_delta_nonzero,
    assert_adapter_load_clean,
    assert_lora_applied,
    recombine_lora_adapters,
    validate_recombined_lora_rank,
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


def _assert_warmstart_adapter_applied(model, model_id: str, load_result) -> None:
    """Fail closed if a warm-start adapter didn't fully apply: clean load (no silently-discarded
    keys), LoRA actually present, and a non-zero delta. Shared by the VL and non-VL load paths."""
    assert_adapter_load_clean(load_result, model_id)
    assert_lora_applied(model, model_id)
    assert_adapter_delta_nonzero(model, model_id)


def _merge_vl_warmstart_adapter(adir: str, model_id: str, attn_kw: dict) -> str:
    """VL warm-start (#296): MERGE the SFT into the FULL multimodal base and save the merged model to a
    fresh temp dir — the new training base for a GRPO LoRA trained from scratch. Continuing the live SFT
    LoRA instead makes the colocated vLLM rollout AND KL reference run off the BARE base (the SFT only
    reaches vLLM via a text<->VL weight-sync that round-trips poorly for ``*ForConditionalGeneration``
    models), so GRPO rolls out base-verbose and collapses a working SFT back to base (observed: every
    Qwen3.5 GRPO reverts; non-VL MiniCPM does not). Merging into the full multimodal model (NOT the
    text-only tree) keeps the VL config + ``language_model.*`` keys that both the trainer reload and
    vLLM's VL loader expect; the SFT keys match here WITHOUT the infix strip. Records the SFT dir for the
    finalize recombine and returns the merged dir."""
    import gc
    import tempfile

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    base = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype="bfloat16", trust_remote_code=True, **attn_kw
    )
    model = PeftModel.from_pretrained(base, adir, is_trainable=False)
    key_mapping = getattr(base, "_checkpoint_conversion_mapping", None)
    load_result = model.load_adapter(
        adir, adapter_name="default", is_trainable=False, key_mapping=key_mapping
    )
    _assert_warmstart_adapter_applied(model, model_id, load_result)
    merged = model.merge_and_unload()
    # UNIQUE per call (mkdtemp): a fixed /tmp/flash_sft_merged would let two GRPO warm-starts on the
    # SAME host clobber each other's merged weights. The dir is the run's training base, so it persists
    # for the run (the worker is ephemeral).
    merged_dir = tempfile.mkdtemp(prefix="flash_sft_merged_")
    merged.save_pretrained(merged_dir, safe_serialization=True)
    from transformers import AutoProcessor

    # processor (preferred for VL) so vLLM/loaders find tokenizer + image-processor config; fall back
    # to the bare tokenizer if no processor is published.
    try:
        AutoProcessor.from_pretrained(model_id, trust_remote_code=True).save_pretrained(merged_dir)
    except Exception:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).save_pretrained(merged_dir)
    del base, model, merged
    gc.collect()
    # The saved GRPO LoRA will be SFT-less (trained on the merged base), so finalize MUST recombine it
    # with this SFT before deploy (recombined_warmstart_adapter_dir).
    _w._VL_WARMSTART_SFT_DIR = adir
    _w._VL_WARMSTART_MODEL_ID = model_id
    return merged_dir


def _init_adapter_model(model_id: str):
    """Load init_from_adapter as a trainable PeftModel, or return model_id + fresh LoRA."""
    _w._VL_WARMSTART_SFT_DIR = None
    _w._VL_WARMSTART_MODEL_ID = None
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
    print(f"[init-adapter] initializing LoRA from {prefix}")
    _attn = optimal_attn_impl()
    attn_kw = {"attn_implementation": _attn} if _attn else {}

    if adapter_is_vl_warmstart(adir, model_id):
        grpo_rank = _w.JOB_SPEC.train.lora_rank if _w.JOB_SPEC else RECIPE.lora.rank
        max_rank = serving_lora_rank_cap(model_id)
        sft_rank, grpo_rank, recombined_rank = validate_recombined_lora_rank(
            adir, grpo_rank, max_rank=max_rank
        )
        cap_note = f"serving cap {max_rank}" if max_rank is not None else "no catalog serving cap"
        print(
            "[init-adapter] VL warm-start rank preflight: "
            f"SFT rank {sft_rank} + GRPO rank {grpo_rank} = deploy rank {recombined_rank} "
            f"({cap_note})"
        )
        merged_dir = _merge_vl_warmstart_adapter(adir, model_id, attn_kw)
        print(
            f"[init-adapter] merged VL SFT {prefix!r} -> {merged_dir}; training a fresh LoRA on it"
        )
        return merged_dir, make_lora(merged_dir)

    # Non-VL checkpoints (e.g. MiniCPM): the continued-LoRA path works (GRPO keeps the SFT behavior),
    # so keep it — TRL trains the LOADED adapter (peft_config=None).
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="bfloat16", trust_remote_code=True, **attn_kw
    )
    model = PeftModel.from_pretrained(base, adir, is_trainable=True)
    key_mapping = getattr(base, "_checkpoint_conversion_mapping", None)
    load_result = model.load_adapter(
        adir, adapter_name="default", is_trainable=True, key_mapping=key_mapping
    )
    _assert_warmstart_adapter_applied(model, model_id, load_result)
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

    from flash.adapter_artifacts import ADAPTER_WEIGHT_FILES
    from flash.engine.worker.hf import _CHECKPOINT_TRAINER_STATE

    out_dir = tempfile.mkdtemp(prefix="flash_recomb_adapter_")
    try:
        rank = recombine_lora_adapters(
            sft_adir,
            src_adapter_dir,
            out_dir,
            model_id=getattr(_w, "_VL_WARMSTART_MODEL_ID", None),
        )
        # Carry tokenizer/aux files from the raw save (serving uses the base tokenizer, but keep the
        # deployed dir at parity with the un-recombined save); the recombined config+weights stay.
        # Skip trainer state — for the per-step path src is a `checkpoint-<n>` dir carrying optimizer/
        # scheduler state that the deployable adapter must not duplicate.
        for name in os.listdir(src_adapter_dir):
            if name == "adapter_config.json" or name in ADAPTER_WEIGHT_FILES:
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

    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=[f"{prefix}/adapter/*"],
        local_dir="/tmp/evdl",
        token=os.environ.get("HF_TOKEN"),
    )
    adir = os.path.join("/tmp/evdl", prefix, "adapter")
    return adir if os.path.isdir(adir) else None

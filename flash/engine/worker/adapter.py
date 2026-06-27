"""LoRA config + warm-start adapter loading for the fine-tuning worker.

Run-scoped state (``JOB_SPEC``) and the patchable ``lora_exclude_modules`` are read THROUGH the
worker package (``_w.<name>``) at CALL time, so tests that ``monkeypatch.setattr(worker,
"lora_exclude_modules", ...)`` then call ``worker.make_lora(...)`` take effect.
"""

from __future__ import annotations

import os
import re

from flash.engine.recipe import RECIPE
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.lora import (
    assert_adapter_delta_nonzero,
    assert_adapter_load_clean,
    assert_lora_applied,
    is_vl_checkpoint,
)
from flash.engine.worker.perf import optimal_attn_impl


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
    rank = _w.JOB_SPEC.train.lora_rank if _w.JOB_SPEC else RECIPE.lora.rank
    alpha = _w.JOB_SPEC.train.lora_alpha if _w.JOB_SPEC else RECIPE.lora.alpha
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
    print(
        "[lora] init_lora_weights=True (standard zero-B; PiSSA removed for serve/warm-start safety)"
    )
    # Standard LoRA scaling (alpha/r). rsLoRA was removed: it scales by alpha/sqrt(r) (~5.6x larger
    # for r=32/alpha=64), so with the usual LoRA LR (e.g. 2e-4) the effective update is ~5.6x too
    # large -> SFT diverges to a degenerate adapter (served model repeats a single token / emits
    # whitespace) and the adapter is also fragile under vLLM's rsLoRA handling at serve time.
    # Standard scaling keeps the catalog LRs sane and the saved adapter serve-safe.
    kwargs["use_rslora"] = False
    if model_id and targets == "all-linear":
        exclude = _w.lora_exclude_modules(model_id)
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


def _init_adapter_model(model_id: str):
    """Base model + the ``train.init_from_adapter`` adapter loaded as a trainable
    PeftModel, or the plain ``model_id`` string + a fresh LoRA when it is unset.

    GRPO continuing an SFT adapter: TRL trains the LOADED adapter (peft_config=None)
    instead of attaching a fresh one."""
    prefix = _w.JOB_SPEC.train.init_from_adapter if _w.JOB_SPEC else ""
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

    print(f"[init-adapter] initializing LoRA from {prefix}")
    _attn = optimal_attn_impl()
    attn_kw = {"attn_implementation": _attn} if _attn else {}

    if is_vl_checkpoint(model_id):
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
        return merged_dir, make_lora(model_id)

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


def _resolve_adapter_ref(adapter_ref: str) -> tuple[str, str] | None:
    """Resolve init_from_adapter into (repo, prefix).

    The only public form is the exact adapter_ref emitted by ``flash status``:
    ``<owner>/<repo>:<phase>/<run_id>/seed<N>``.
    """
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
    """Download an init_from_adapter LoRA to /tmp/evdl/<prefix>/adapter and return its dir.

    ``adapter_prefix`` must be the full ``adapter_ref`` string emitted by ``flash status``:
    ``<owner>/<repo>:<phase>/<run_id>/seed<N>``.
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

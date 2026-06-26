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
    remap_vl_adapter_dir,
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
    """The base for GRPO + the LoRA config to attach — ALWAYS a ``(model_path_str, LoraConfig)``
    pair, so the warm-start path is structurally identical to the from-base path.

    From base (``init_from_adapter`` unset): returns ``(model_id, make_lora(model_id))``. TRL loads
    and GPU-places the string itself, then attaches a fresh LoRA.

    Warm start (``init_from_adapter`` set): the SFT adapter is MERGED into a disk copy of the base
    and we return ``(merged_base_dir, make_lora(model_id))`` — the SAME shape. GRPO then trains a
    FRESH LoRA on the SFT-merged base. Two reasons this is the right design:

    * **Fixes the warm-start init hang.** The old path handed GRPOTrainer a live, CPU-resident
      PeftModel (``peft_config=None``). Under colocate vLLM (always on for GRPO) that pre-loaded
      object stalls trainer construction at 0% GPU for ~25 min with no error. Passing a model-path
      STRING + a LoraConfig makes TRL load/place the model itself — the proven from-base init path —
      so colocate vLLM never sees an unexpected pre-built module.
    * **Decouples LoRA rank/alpha from SFT.** The SFT delta is baked into the merged base, so the
      fresh GRPO LoRA's ``train.lora_rank``/``train.lora_alpha`` (read by ``make_lora``) no longer
      have to match the SFT run's adapter shape.

    Caveat handled at finalize: the GRPO LoRA alone, served on the ORIGINAL catalog base, would MISS
    the merged SFT. ``combine_warmstart_into_adapter`` recombines SFT+GRPO into one deployable
    adapter before upload, so the existing deploy/serve path (catalog base + the run's one adapter)
    stays correct without any serving changes."""
    prefix = _w.JOB_SPEC.train.init_from_adapter if _w.JOB_SPEC else ""
    if not prefix:
        _w.WARMSTART_MERGED = False  # from-base: per-step adapters ARE catalog-base-deployable
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
    print(f"[init-adapter] warm-start: merging SFT adapter {prefix} into the base for GRPO")
    # VL checkpoints (Qwen3.5/3.6): the SFT step saved the adapter against the FULL multimodal model
    # (keys under ``base_model.model.model.language_model.layers.*``), but we load the base here via
    # AutoModelForCausalLM (text-only tree, ``base_model.model.model.layers.*``). Strip the
    # ``.language_model.`` infix on disk so PeftModel.from_pretrained matches the SFT keys —
    # otherwise peft only WARNS about missing keys and silently trains a fresh LoRA, discarding the
    # SFT. No-op for non-VL checkpoints. See flash/engine/worker/lora.py. The return value also tells
    # us the adapter carried VL keys (evidence-based, robust to a flaky is_vl_checkpoint probe).
    n_vl_remapped = remap_vl_adapter_dir(adir, model_id)
    if n_vl_remapped > 0 or is_vl_checkpoint(model_id):
        # Merge-to-string is not yet safe for VL checkpoints: merging the SFT into the text-only
        # AutoModelForCausalLM tree and saving it writes a text-only config/weight layout, which the
        # colocate vLLM engine (which loads the merged dir as the VL arch) cannot reconcile. Fail
        # FAST (before importing peft/loading the base or wasting a paid GPU on a broken merged base —
        # the old path hung here instead). Follow-up: merge against the full VL model / preserve the
        # VL config so the merged dir round-trips through vLLM.
        raise RuntimeError(
            f"warm-start GRPO (train.init_from_adapter) is not yet supported for the natively-"
            f"multimodal checkpoint {model_id!r}. Run GRPO from the base model (omit "
            "init_from_adapter), or deploy/continue from the SFT adapter directly."
        )
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    _attn = optimal_attn_impl()
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        **({"attn_implementation": _attn} if _attn else {}),
    )
    model = PeftModel.from_pretrained(base, adir, is_trainable=True)
    # Fail loudly if the adapter didn't actually apply (a key mismatch would otherwise silently merge
    # an all-zero identity and start GRPO from the base model). from_pretrained loads with
    # load_state_dict(strict=False) and only WARNS on a mismatch, discarding the load result — so
    # re-run load_adapter to CAPTURE which keys matched and assert matched==saved (peft injects the
    # LoRA modules from target_modules BEFORE loading weights, so the module-count check alone can't
    # see a silent weight discard). The reload is idempotent: same weights into the same "default"
    # adapter. See flash/engine/worker/lora.py. Mirror from_pretrained's key_mapping: for transformers
    # models that define a ``_checkpoint_conversion_mapping`` (renamed-arch checkpoints),
    # from_pretrained remaps the adapter keys before loading; the reload must apply the SAME mapping
    # or it would reinterpret valid keys as mismatched and falsely abort.
    key_mapping = getattr(base, "_checkpoint_conversion_mapping", None)
    load_result = model.load_adapter(
        adir, adapter_name="default", is_trainable=True, key_mapping=key_mapping
    )
    assert_adapter_load_clean(load_result, model_id)
    assert_lora_applied(model, model_id)
    assert_adapter_delta_nonzero(model, model_id)
    # Bake the (verified non-zero) SFT delta into the base weights and hand TRL a STRING path. GRPO
    # trains a fresh LoRA on this merged base; the deployable SFT⊕GRPO adapter is reassembled at
    # finalize (combine_warmstart_into_adapter).
    merged = model.merge_and_unload()
    merged_dir = _save_merged_base(merged, model_id)
    # GRPO now trains on a base with SFT MERGED in -> per-step adapter snapshots are GRPO-only deltas,
    # not deployable on the catalog base. Flag it so publish_deployable_checkpoint stops advertising
    # intermediate steps as deployable (only the recombined FINAL adapter is catalog-base-correct).
    _w.WARMSTART_MERGED = True
    # Drop the CPU-resident copies promptly — TRL reloads merged_dir straight onto the GPU.
    del model, base, merged
    return merged_dir, make_lora(model_id)


def _save_merged_base(merged_model, model_id: str) -> str:
    """Save a merge_and_unload()'d base (SFT folded in) to a fresh temp dir + its tokenizer, and
    return the dir. TRL/colocate-vLLM load it as a plain base model (string path), exactly like the
    from-base ``model_id``. Ephemeral: the per-run GPU pod is torn down after the run."""
    import tempfile

    from transformers import AutoTokenizer

    out = tempfile.mkdtemp(prefix="flash-warmstart-merged-")
    merged_model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).save_pretrained(out)
    print(f"[init-adapter] SFT merged into base at {out} — GRPO trains a fresh LoRA on it")
    return out


def combine_warmstart_into_adapter(model_id: str, grpo_adapter_dir: str) -> bool:
    """Recombine a warm start's SFT adapter with the freshly-trained GRPO LoRA into ONE adapter
    deployable on the ORIGINAL catalog base, rewritten IN PLACE at ``grpo_adapter_dir``.

    Warm-start GRPO trains a fresh LoRA on a base that already has the SFT adapter merged in (see
    ``_init_adapter_model``), so the GRPO adapter alone, served on the catalog base, would miss the
    SFT gain. A LoRA delta is just an additive low-rank term, independent of which base it sits on,
    so PEFT's ``add_weighted_adapter(..., combination_type="cat")`` concatenates the SFT and GRPO
    LoRAs into a single rank-(r_sft+r_grpo) adapter whose delta on the original base is exactly
    ΔSFT + ΔGRPO — i.e. the trained model (base + SFT + GRPO). cat sets the combined adapter's
    ``lora_alpha == r`` (scaling 1) and folds each source adapter's alpha/r scaling into the
    concatenated weights, so weights ``[1.0, 1.0]`` reproduce the plain sum. Ranks may differ
    (the SFT and GRPO runs can use different lora_rank/lora_alpha).

    Saves the combined adapter as the ``"default"`` adapter, which PEFT writes to the dir ROOT
    (overwriting the GRPO-only ``adapter_config.json`` + ``adapter_model.safetensors``), so the
    existing deploy/serve path (catalog base + the run's one adapter) stays correct with no serving
    change. Returns False (no-op) for a from-base run.

    Note: warm-start runs do NOT advertise per-step deployable checkpoints at all — those snapshots
    are GRPO-only deltas on the merged base, so ``publish_deployable_checkpoint`` skips them while
    ``WARMSTART_MERGED`` is set (a mid-RL / cancelled ``flash deploy --step N`` would otherwise serve
    SFT-less weights). Only the run's FINAL adapter is recombined here into a catalog-base-correct,
    deployable adapter."""
    prefix = _w.JOB_SPEC.train.init_from_adapter if _w.JOB_SPEC else ""
    if not prefix:
        return False
    adir = _download_adapter(prefix)  # cached locally from _init_adapter_model; re-resolve statelessly
    if not adir:
        raise RuntimeError(
            f"warm-start finalize: SFT adapter {prefix!r} is no longer resolvable to recombine with "
            "the GRPO LoRA. The deployed adapter would silently miss the SFT — failing instead."
        )
    remap_vl_adapter_dir(adir, model_id)  # idempotent (already stripped at init); VL is guarded out

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    # The cat only manipulates LoRA weights; the base is loaded (CPU, bf16) just to host the adapter
    # modules, so this never contends with the trainer's still-resident GPU copy.
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adir, adapter_name="sft", is_trainable=False)
    model.load_adapter(grpo_adapter_dir, adapter_name="grpo", is_trainable=False)
    r_sft = model.peft_config["sft"].r
    r_grpo = model.peft_config["grpo"].r
    model.add_weighted_adapter(
        ["sft", "grpo"], [1.0, 1.0], adapter_name="default", combination_type="cat"
    )
    # selected_adapters=["default"] + the "default" name => PEFT writes to grpo_adapter_dir ROOT
    # (peft_model.save_pretrained), overwriting the GRPO-only adapter files in place.
    model.save_pretrained(grpo_adapter_dir, selected_adapters=["default"])
    print(
        f"[finalize] recombined warm-start SFT(r={r_sft}) + GRPO(r={r_grpo}) -> one deployable "
        f"rank-{r_sft + r_grpo} adapter at {grpo_adapter_dir} (serves on the catalog base)"
    )
    del model, base
    return True


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

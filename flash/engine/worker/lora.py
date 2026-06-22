"""Pure LoRA-target / VL-checkpoint helpers for the fine-tuning worker.

These helpers take the model id as an ARGUMENT and read NONE of the worker's run-scoped
module globals, so they live here as a leaf module. ``flash.engine.worker`` re-exports
them; this module must NOT import that package (no cycle). Heavy deps (transformers, peft,
vllm, the catalog) are imported lazily inside the functions so the module stays
CPU-importable.
"""

from __future__ import annotations


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


# Flipped to True only AFTER the GRPO trainer (and its colocated vLLM engine + initial
# checkpoint load) is constructed, but BEFORE ``trainer.train()`` runs the first weight sync.
# See ``patch_vllm_lm_weight_sync``. A module dict (not a bare bool) so the gating flag is shared
# by reference between this module and the worker package that flips it.
_LM_SYNC_REMAP_ON = {"on": False}


def _remap_vl_sync_weights(weights):
    """Rewrite TRL's trainer weight names to vLLM's VL-engine names for the train-time sync.

    The trainer (built via ``AutoModelForCausalLM``) names its LM params ``model.layers.*`` /
    ``model.norm`` / ``model.embed_tokens`` / ``lm_head.*``; the colocated vLLM engine loaded the
    same checkpoint as ``Qwen3_5ForConditionalGeneration`` whose LM params live under
    ``language_model.*``. Prefix incoming ``model.``/``lm_head.`` names with ``language_model.`` so
    they resolve. Also tolerate a peft ``base_model.model.`` prefix (a merged-adapter sync can yield
    base-model names through that wrapper) by stripping it before the language_model. prefix is
    added. Names that already start with ``language_model.`` (or anything else) pass through
    untouched. A generator so vLLM's loader still streams one (name, tensor) at a time.
    """
    for name, tensor in weights:
        # A continued-adapter (PeftModel) sync can surface names through the peft wrapper as
        # ``base_model.model.model.layers.*`` / ``base_model.model.lm_head.*``; strip the wrapper
        # so the same model./lm_head. rule applies.
        if name.startswith("base_model.model."):
            name = name[len("base_model.model.") :]
        if name.startswith(("model.", "lm_head.")):
            name = "language_model." + name
        yield name, tensor


def patch_vllm_lm_weight_sync(model_id: str) -> bool:
    """Make TRL's GRPO ``sync_weights`` work for ``*ForConditionalGeneration`` checkpoints
    (the whole Qwen3.5/3.6 family). Returns True if any vLLM model class was patched.

    The trainer loads via ``AutoModelForCausalLM`` so its params are named ``model.layers.*`` /
    ``model.norm`` / ``model.embed_tokens`` / ``lm_head.*``. vLLM loads the same checkpoint as
    ``Qwen3_5ForConditionalGeneration`` whose LM params live under ``language_model.*``. TRL's
    ``sync_weights`` pushes the trainer names verbatim, so vLLM's loader raises "There is no module
    or parameter named 'model' in Qwen3_5ForConditionalGeneration" at the first generation step and
    GRPO dies (even with ``language_model_only=True``: that only skips loading the vision tower, it
    does NOT rename the surviving LM params out from under ``language_model.``).

    The fix wraps the vLLM model class ``load_weights`` to remap incoming ``model.``/``lm_head.``
    names to ``language_model.*`` (see ``_remap_vl_sync_weights``) — but ONLY while
    ``_LM_SYNC_REMAP_ON`` is set. The INITIAL checkpoint load (during trainer construction) runs
    with it OFF, so vLLM's own ``hf_to_vllm_mapper`` handles the on-disk checkpoint untouched; the
    remap activates only for the train-time TRL syncs. The flag is flipped on between trainer
    construction and ``train()``. Works for BOTH from-base and warm-started (init_from_adapter)
    GRPO. No-op for non-VL checkpoints."""
    if not is_vl_checkpoint(model_id):
        return False
    patched_any = False
    try:
        import importlib

        # The dense class is REQUIRED for the whole Qwen3.5/3.6 family — if its module/class can't
        # be imported (vLLM not installed where it should be, or the class renamed in a new vLLM)
        # we must NOT silently no-op: the run would crash again at the first ``sync_weights()`` with
        # a far less actionable error. Log loudly for the required one; the MoE class is OPTIONAL
        # (only some models are MoE, and older vLLM lacks the module) so its absence stays quiet.
        for mod_name, cls_name, required in (
            ("vllm.model_executor.models.qwen3_5", "Qwen3_5ForConditionalGeneration", True),
            ("vllm.model_executor.models.qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration", False),
        ):
            try:
                mod = importlib.import_module(mod_name)
            except Exception as e:
                mod = None
                if required:
                    print(
                        f"[vllm] WARN patch_vllm_lm_weight_sync: could not import required module "
                        f"{mod_name} ({e!r}); GRPO weight-sync will NOT be remapped and the run may "
                        f"crash at the first sync_weights() for this VL checkpoint."
                    )
            cls = getattr(mod, cls_name, None) if mod is not None else None
            if cls is None:
                if required and mod is not None:
                    print(
                        f"[vllm] WARN patch_vllm_lm_weight_sync: module {mod_name} imported but has "
                        f"no {cls_name} (vLLM API changed?); GRPO weight-sync will NOT be remapped "
                        f"and the run may crash at the first sync_weights() for this VL checkpoint."
                    )
                continue
            if getattr(cls.load_weights, "_flash_sync_patched", False):
                continue
            orig_load = cls.load_weights

            def _make_patched(orig):
                def patched(self, weights, *args, **kwargs):
                    if _LM_SYNC_REMAP_ON["on"]:
                        weights = _remap_vl_sync_weights(weights)
                    return orig(self, weights, *args, **kwargs)

                patched._flash_sync_patched = True
                return patched

            cls.load_weights = _make_patched(orig_load)
            patched_any = True
            print(f"[vllm] LM weight-sync name patch installed for {cls_name} (gated)")
    except Exception as e:
        print("patch_vllm_lm_weight_sync warn:", e)
    return patched_any


def model_quant(model_id: str) -> str:
    """Quantization tier for this model: catalog entry > bf16 (managed; no override)."""
    try:
        from flash.catalog import MODELS

        info = MODELS.get(model_id)
        if info is not None:
            return info.quant
    except Exception as e:
        print("model_quant: catalog probe failed:", e)
    return "bf16"

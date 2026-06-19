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

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


# --------------------------------------------------------------------------------------------
# Warm-start (init_from_adapter) SFT-adapter key remap for VL checkpoints.
#
# SFT (run_sft) trains the FULL multimodal model: ``SFTTrainer(model=model_id,
# peft_config=make_lora(...))`` loads ``Qwen3_5ForConditionalGeneration`` whose LM modules live
# under ``language_model.``, so the SAVED adapter's keys are
# ``base_model.model.model.language_model.layers.X...``. But warm-started GRPO
# (``_init_adapter_model``) loads the base via ``AutoModelForCausalLM`` — a TEXT-ONLY module tree
# whose LoRA targets are named ``base_model.model.model.layers.X...`` (no ``language_model.``
# infix). ``PeftModel.from_pretrained`` then can't match the SFT keys: peft logs a *warning* about
# missing adapter keys and SILENTLY keeps the fresh zero-init LoRA, so the SFT is thrown away and
# GRPO restarts from the base model (observed: linkd-search warm-start reward ~= 0.001).
#
# Stripping the ``.language_model.`` infix from the saved adapter keys makes them line up with the
# ``AutoModelForCausalLM`` trainer (proven workaround: remapped adapters train correctly). We keep
# the trainer as ``AutoModelForCausalLM`` so the train-time vLLM weight-sync remap
# (``patch_vllm_lm_weight_sync`` / ``_remap_vl_sync_weights``) stays consistent.
# --------------------------------------------------------------------------------------------

_LANGUAGE_MODEL_INFIX = ".language_model."


def strip_language_model_infix(key: str) -> str:
    """Strip the FIRST ``.language_model.`` infix from a peft adapter weight key.

    ``base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_A.default.weight``
    -> ``base_model.model.model.layers.0.linear_attn.out_proj.lora_A.default.weight``.

    Only the first occurrence is removed (the LM-vs-VL boundary appears once in the path); keys
    without the infix are returned unchanged.
    """
    i = key.find(_LANGUAGE_MODEL_INFIX)
    if i == -1:
        return key
    # Replace ".language_model." with "." (keep one separator dot).
    return key[:i] + "." + key[i + len(_LANGUAGE_MODEL_INFIX) :]


def remap_adapter_keys(keys):
    """Map an iterable of adapter weight keys -> a dict {old_key: new_key} for keys that change.

    Pure (no I/O); used both by the on-disk rewriter and by tests to assert the post-remap key set
    matches an ``AutoModelForCausalLM``-named LoRA param set.
    """
    out = {}
    for k in keys:
        nk = strip_language_model_infix(k)
        if nk != k:
            out[k] = nk
    return out


def _rewrite_safetensors_header_keys(path: str, rename) -> int:
    """Rename tensor keys in a ``.safetensors`` file IN PLACE, editing only the header.

    safetensors layout: 8-byte little-endian header length, then a JSON header mapping
    ``name -> {dtype, shape, data_offsets}`` (plus an optional ``__metadata__`` entry), then the
    raw tensor data. ``data_offsets`` are relative to the data section, so a pure key rename leaves
    every byte of the data section valid — we only rewrite the JSON header and its length prefix.

    ``rename`` is a callable ``old_key -> new_key``. Returns the number of keys renamed. No torch /
    safetensors dependency (keeps this module CPU-importable on the server venv).
    """
    import json
    import os
    import shutil
    import struct

    with open(path, "rb") as f:
        len_bytes = f.read(8)
        if len(len_bytes) < 8:
            raise ValueError(f"{path}: too small to be a safetensors file")
        (hdr_len,) = struct.unpack("<Q", len_bytes)
        header_bytes = f.read(hdr_len)
        if len(header_bytes) < hdr_len:
            raise ValueError(f"{path}: truncated safetensors header")
        header = json.loads(header_bytes)
    data_start = 8 + hdr_len

    new_header = {}
    renamed = 0
    for k, v in header.items():
        if k == "__metadata__":
            new_header[k] = v
            continue
        nk = rename(k)
        if nk != k:
            if nk in header or nk in new_header:
                raise ValueError(
                    f"{path}: remapped key {nk!r} collides with an existing key; refusing to "
                    f"overwrite (adapter may already be remapped or malformed)"
                )
            renamed += 1
        new_header[nk] = v

    if renamed == 0:
        return 0

    # Re-serialize compactly. safetensors does not require any specific key order or padding; the
    # only constraint is that data_offsets stay consistent with the (unchanged) data section.
    new_header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    # Stream the (possibly multi-GB) tensor data straight from the original to a temp file instead
    # of slurping the whole file into memory; os.replace makes the swap atomic so an interrupted
    # rewrite can't corrupt the adapter.
    tmp = path + ".remap.tmp"
    try:
        with open(path, "rb") as src, open(tmp, "wb") as out:
            src.seek(data_start)
            out.write(struct.pack("<Q", len(new_header_bytes)))
            out.write(new_header_bytes)
            shutil.copyfileobj(src, out, 8 * 1024 * 1024)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)
    return renamed


def _rewrite_bin_keys(path: str, rename) -> int:
    """Rename keys in a PyTorch ``.bin`` (pickled ``state_dict``) adapter IN PLACE.

    Used only when the saved adapter is the legacy ``.bin`` format (no ``.safetensors``). Needs
    torch to (de)serialize; that's fine because this path runs only on the GPU worker.
    """
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=True)
    new_sd = {}
    renamed = 0
    for k, v in sd.items():
        nk = rename(k)
        if nk != k:
            if nk in sd or nk in new_sd:
                raise ValueError(
                    f"{path}: remapped key {nk!r} collides with an existing key; refusing to "
                    f"overwrite (adapter may already be remapped or malformed)"
                )
            renamed += 1
        new_sd[nk] = v
    if renamed == 0:
        return 0
    torch.save(new_sd, path)
    return renamed


def remap_vl_adapter_dir(adir: str, model_id: str) -> int:
    """For a VL warm-start, strip the ``.language_model.`` infix from the downloaded SFT adapter so
    its keys match the ``AutoModelForCausalLM`` trainer used by ``_init_adapter_model``.

    No-op (returns 0) when ``model_id`` is not a VL checkpoint, or the adapter has no
    ``.language_model.`` keys (e.g. already remapped, or a text-only SFT). Rewrites whichever weight
    file exists in ``adir`` (``adapter_model.safetensors`` preferred, else ``adapter_model.bin``).
    Returns the number of keys renamed. Idempotent: a second call finds nothing to strip.
    """
    import os

    if not is_vl_checkpoint(model_id):
        return 0

    st_path = os.path.join(adir, "adapter_model.safetensors")
    bin_path = os.path.join(adir, "adapter_model.bin")
    if os.path.isfile(st_path):
        n = _rewrite_safetensors_header_keys(st_path, strip_language_model_infix)
    elif os.path.isfile(bin_path):
        n = _rewrite_bin_keys(bin_path, strip_language_model_infix)
    else:
        print(
            f"[init-adapter] remap_vl_adapter_dir: no adapter_model.safetensors/.bin in {adir!r}; "
            "nothing to remap"
        )
        return 0
    if n:
        print(
            f"[init-adapter] remapped {n} VL SFT adapter key(s): stripped '.language_model.' infix "
            f"to match the AutoModelForCausalLM trainer for {model_id}"
        )
    else:
        print(
            f"[init-adapter] no '.language_model.' adapter keys to remap for {model_id} "
            "(already remapped or a text-only adapter)"
        )
    return n


def assert_lora_applied(model, model_id: str) -> int:
    """After ``PeftModel.from_pretrained``, verify the adapter's LoRA actually loaded (non-empty)
    so a future key-mismatch regression fails LOUDLY instead of silently training a fresh LoRA.

    Counts the LoRA A/B submodules present on the PeftModel. Raises for ANY warm-start that ended
    up with ZERO LoRA modules (a key mismatch from any cause; the VL ``.language_model.`` mismatch
    this remap fixes is the common one). Returns the count.
    """
    count = 0
    for name, _ in model.named_modules():
        # peft names the per-target adapter submodules ``...lora_A.<adapter>`` / ``...lora_B.*``.
        if name.endswith("lora_A.default") or name.endswith("lora_B.default"):
            count += 1
    if count == 0:
        raise RuntimeError(
            f"warm-start adapter for {model_id} loaded ZERO LoRA modules — the SFT adapter was NOT "
            "applied (key mismatch). GRPO would silently restart from the base model. For Qwen3.5/"
            "3.6 VL this is usually the '.language_model.' key-mismatch (check remap_vl_adapter_dir "
            "ran on the adapter); otherwise verify the adapter's keys match the model."
        )
    print(f"[init-adapter] verified {count} LoRA submodule(s) applied for {model_id}")
    return count


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

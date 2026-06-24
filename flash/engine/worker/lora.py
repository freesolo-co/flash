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


def patch_grpo_mask_aware_lm_head(trainer) -> bool:
    """Skip the 248k-vocab ``lm_head`` projection at MASKED completion positions in the GRPO loss.

    In multi-turn GRPO the completion is the full interleaved transcript; ~half of it is
    environment/tool tokens masked out of the loss (the rollout's ``env_mask`` -> TRL's
    ``tool_mask``), plus right-padding. TRL 1.6's ``compute_liger_loss`` hands the FULL-length hidden
    states to ``liger_grpo_loss``, and the Liger kernel runs the lm_head matmul + log-softmax for
    EVERY position (in the forward AND the backward recompute). Masked positions contribute zero
    loss and zero gradient but still pay the full FLOPs of the single most expensive GRPO op (the
    248k-vocab projection Liger exists to tame).

    Wrap ``trainer.liger_grpo_loss`` to GATHER the unmasked positions — ONE shared index applied
    identically to every per-token tensor (``_input``, ``selected_token_ids``, ``attention_mask``,
    ``old_per_token_logps``, ``ref_per_token_logps``, and a 2-D ``vllm_is_ratio``) — before the call,
    so the kernel only projects the kept positions. Per-sequence ``advantages`` ``(B,)`` and the loss
    object's ``max_completion_length`` are left untouched. This is EXACTLY loss-preserving: dr_grpo's
    numerator only ever summed unmasked positions, and its normalizer is ``B * max_completion_length``
    (a config constant on the loss object, independent of the gathered length); the gathered
    sequence is re-padded with masked positions whose new mask is 0, so loss + credit assignment are
    unchanged while the gathered length T' < T cuts the projection FLOPs by ~the masked fraction.
    No-op when nothing is masked (single-turn / full mask) or the loss object isn't present. Returns
    True if wrapped."""
    orig = getattr(trainer, "liger_grpo_loss", None)
    if orig is None:
        return False
    import torch

    def _gather(x, idx, tprime):
        if x is None:
            return None
        if x.dim() == 2:  # (B, T) per-token tensor
            return torch.gather(x, 1, idx)
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(idx.size(0), tprime, x.size(-1)))

    def masked_liger_loss(*args, **kwargs):
        mask = kwargs.get("attention_mask")  # loss mask = completion_mask * tool_mask, shape (B, T)
        if args or mask is None or mask.dim() != 2:
            return orig(*args, **kwargs)  # unexpected call shape -> never alter the loss
        keep = mask != 0
        full_t = mask.size(1)
        tprime = int(keep.sum(dim=1).max().item())
        if tprime == 0 or tprime == full_t:
            return orig(**kwargs)  # nothing maskable to skip
        # One shared gather index: unmasked positions first (stable -> original order preserved),
        # the remainder filled from the trailing masked positions whose gathered mask is 0 (so they
        # add zero loss/grad and can't perturb the per-token ratio/KL alignment).
        order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
        idx = order[:, :tprime].contiguous()
        gk = dict(kwargs)
        gk["attention_mask"] = torch.gather(mask, 1, idx)
        gk["_input"] = _gather(kwargs.get("_input"), idx, tprime)
        gk["selected_token_ids"] = _gather(kwargs.get("selected_token_ids"), idx, tprime)
        for key in ("old_per_token_logps", "ref_per_token_logps"):
            if kwargs.get(key) is not None:
                gk[key] = _gather(kwargs[key], idx, tprime)
        ratio = kwargs.get("vllm_is_ratio")
        if ratio is not None and ratio.dim() == 2 and ratio.size(1) == full_t:
            gk["vllm_is_ratio"] = _gather(ratio, idx, tprime)
        return orig(**gk)

    trainer.liger_grpo_loss = masked_liger_loss
    return True


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


# Substrings that identify a peft LoRA weight key (vs a base-model param). The whole adapter file
# is LoRA weights, but a wrong-arch / corrupt checkpoint can contain non-LoRA tensors, so we filter.
_LORA_KEY_MARKERS = (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B.", "lora_")


def _is_lora_key(key: str) -> bool:
    return any(m in key for m in _LORA_KEY_MARKERS)


# A safetensors header is small even for huge models (a few hundred KB at most); 100 MB is a wildly
# generous ceiling that still refuses a corrupt/hostile file declaring a multi-GB header length
# before we allocate/read it.
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


def _read_adapter_tensor_keys(adir: str) -> list[str] | None:
    """Tensor key names in the downloaded adapter.

    For ``.safetensors`` this reads ONLY the JSON header (pure stdlib, no tensor data — keeps this
    module CPU-importable). For the legacy ``.bin`` format the pickled ``state_dict`` must be
    materialized via ``torch.load`` to enumerate its keys (a pickle can't be read key-only without
    unpickling the tensor payloads — GPU-worker only). Returns ``None`` when neither weight file
    exists in ``adir``.
    """
    import json
    import os
    import struct

    st_path = os.path.join(adir, "adapter_model.safetensors")
    bin_path = os.path.join(adir, "adapter_model.bin")
    if os.path.isfile(st_path):
        # safetensors layout: 8-byte LE header length, then the JSON header, then the tensor data.
        # Bound the DECLARED header length against the real file size (and an absolute ceiling)
        # BEFORE reading it, so a corrupt/hostile file can't trigger a huge allocation / long read.
        file_size = os.path.getsize(st_path)
        with open(st_path, "rb") as f:
            len_bytes = f.read(8)
            if len(len_bytes) < 8:
                raise ValueError(f"{st_path}: too small to be a safetensors file")
            (hdr_len,) = struct.unpack("<Q", len_bytes)
            if hdr_len > file_size - 8 or hdr_len > _MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError(
                    f"{st_path}: declared safetensors header length {hdr_len} is implausible "
                    f"(file is {file_size} bytes) — refusing to read a corrupt/oversized header"
                )
            header_bytes = f.read(hdr_len)
            if len(header_bytes) < hdr_len:
                raise ValueError(f"{st_path}: truncated safetensors header")
            header = json.loads(header_bytes)
        # The safetensors header MUST be a JSON object keyed by tensor name. A corrupt/hostile file
        # could decode to a list/int/str or carry non-string keys, which would later blow up with a
        # confusing TypeError in _is_lora_key (substring search on a non-str). Validate the shape here
        # (this reader is the "reject hostile headers early" hardening) and fail with a clear message.
        if not isinstance(header, dict) or not all(isinstance(k, str) for k in header):
            raise ValueError(
                f"{st_path}: safetensors header is not a JSON object with string keys "
                "(corrupt or not a safetensors file)"
            )
        return [k for k in header if k != "__metadata__"]
    if os.path.isfile(bin_path):
        import torch

        sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        return list(sd.keys())
    return None


def remap_vl_adapter_dir(adir: str, model_id: str) -> int:
    """For a VL warm-start, strip the ``.language_model.`` infix from the downloaded SFT adapter so
    its keys match the ``AutoModelForCausalLM`` trainer used by ``_init_adapter_model``.

    The remap decision is driven by the ADAPTER'S OWN keys, not only the ``is_vl_checkpoint`` config
    probe. ``is_vl_checkpoint`` calls ``AutoConfig.from_pretrained`` and swallows EVERY exception to
    return False, so an HF rate-limit / network hiccup / uncached config silently turned a genuine
    VL warm-start into a no-op: the ``.language_model.`` keys were left in place, the text-only base
    couldn't match them, peft kept the zero-init LoRA, and GRPO aborted at
    ``assert_adapter_delta_nonzero`` with all-zero ``lora_B`` (issue #286). Any adapter that actually
    carries ``.language_model.`` LoRA keys was saved against the full multimodal model and MUST be
    stripped regardless of the probe, so we key off the file contents and only fall back to the probe
    for the (already-stripped / text-only) no-infix case.

    Fails LOUDLY instead of silently dropping a mismatched adapter:
    - a VL warm-start whose adapter has NO LoRA keys at all (corrupt / wrong-architecture) raises;
    - any ``.language_model.`` LoRA key that SURVIVES the rewrite raises (it would be silently
      discarded by the text-only base -> all-zero ``lora_B``).

    Returns the number of keys renamed. No-op (returns 0) for a genuinely text-only model, or an
    already-remapped adapter. Idempotent: a second call finds nothing to strip.
    """
    import os

    keys = _read_adapter_tensor_keys(adir)
    if keys is None:
        print(
            f"[init-adapter] remap_vl_adapter_dir: no adapter_model.safetensors/.bin in {adir!r}; "
            "nothing to remap"
        )
        return 0

    lora_keys = [k for k in keys if _is_lora_key(k)]
    infixed = [k for k in lora_keys if _LANGUAGE_MODEL_INFIX in k]

    # No '.language_model.' LoRA keys -> nothing to strip from the file itself. The ONLY reason to act
    # is the config probe, so it runs HERE (the fallback case) rather than on every warm-start: a key
    # already in text-only form needs no network round-trip to confirm. is_vl distinguishes a genuine
    # text-only model (return 0) from an already-remapped / text-only-SFT VL adapter (diagnostic).
    if not infixed:
        if not is_vl_checkpoint(model_id):
            return 0  # genuinely text-only model with text-only adapter keys
        if not lora_keys:
            # A VL warm-start whose adapter carries no LoRA weights can't hold a real SFT delta — it
            # would load as the all-zero identity. Fail here, before the base-model download.
            raise RuntimeError(
                f"warm-start adapter in {adir!r} for {model_id} contains NO LoRA weight keys "
                f"(found {len(keys)} tensor(s), 0 with a lora_ marker) — the adapter is corrupt, "
                "incomplete, or from a different architecture, so GRPO would train from the base "
                "model. Re-export the SFT adapter, or omit train.init_from_adapter for a fresh LoRA."
            )
        # VL checkpoint but nothing to strip: legitimately already-remapped (idempotent re-run) or a
        # text-only SFT. Surface the adapter's actual LoRA prefix so a real key mismatch isn't a
        # silent no-op — if GRPO later aborts with all-zero lora_B, these keys didn't match the base.
        sample_prefix = next(
            (k.split(".lora_")[0] for k in lora_keys if ".lora_" in k), lora_keys[0]
        )
        print(
            f"[init-adapter] remap_vl_adapter_dir: 0 '.language_model.' keys to strip for VL "
            f"checkpoint {model_id} ({len(lora_keys)} LoRA key(s); e.g. prefix {sample_prefix!r}) — "
            "treating as already-remapped/text-only. If the warm-start later aborts with all-zero "
            "lora_B, these keys did not match the base model."
        )
        return 0

    # The adapter carries '.language_model.' LoRA keys: it was saved against the full multimodal model
    # and MUST be stripped to match the AutoModelForCausalLM trainer — regardless of the config probe
    # (a flaky/failed AutoConfig probe must not silently skip a needed remap -> issue #286). We don't
    # call is_vl_checkpoint at all on this path: the adapter's own keys are sufficient evidence.
    # Fail CLOSED *before* touching disk: strip_language_model_infix removes only the FIRST infix, so a
    # key carrying it twice would still match no text-only module and be silently discarded (the #286
    # all-zero-lora_B failure). Predict the post-strip keys from the in-memory list (no file re-read).
    survivors = [
        nk for nk in (strip_language_model_infix(k) for k in infixed) if _LANGUAGE_MODEL_INFIX in nk
    ]
    if survivors:
        raise RuntimeError(
            f"remap_vl_adapter_dir: {len(survivors)} LoRA key(s) in {adir!r} for {model_id} would "
            f"still carry '.language_model.' after the remap (e.g. {survivors[0]!r}) — they will NOT "
            "match the AutoModelForCausalLM trainer and would be silently discarded -> all-zero "
            "lora_B. The adapter's key layout is unexpected; verify it was saved by this SFT pipeline."
        )

    st_path = os.path.join(adir, "adapter_model.safetensors")
    bin_path = os.path.join(adir, "adapter_model.bin")
    if os.path.isfile(st_path):
        n = _rewrite_safetensors_header_keys(st_path, strip_language_model_infix)
    else:  # bin_path exists — keys were read from one of the two files above
        n = _rewrite_bin_keys(bin_path, strip_language_model_infix)

    print(
        f"[init-adapter] remapped {n} VL SFT adapter key(s): stripped '.language_model.' infix "
        f"to match the AutoModelForCausalLM trainer for {model_id}"
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


def assert_adapter_load_clean(load_result, model_id: str) -> None:
    """Assert a peft adapter load matched ALL saved keys — fail closed on a silent discard.

    ``PeftModel.from_pretrained`` loads adapter weights with ``load_state_dict(strict=False)`` and
    only WARNS on a key mismatch (it throws the load result away), so an SFT adapter whose keys don't
    line up with the target base is silently dropped and GRPO restarts from the base model (bug #67).
    ``assert_lora_applied`` can't catch this: peft INJECTS the LoRA modules from ``target_modules``
    BEFORE loading any weights, so the module count is non-zero even when zero saved weights matched.

    ``load_result`` is the object returned by ``PeftModel.load_adapter`` (a ``_IncompatibleKeys`` with
    ``missing_keys`` / ``unexpected_keys``). We only care about LoRA keys: an adapter-only checkpoint
    loaded with ``strict=False`` legitimately leaves the base-model params out, so they can surface as
    "missing" without anything being wrong. peft's ``load_adapter`` already filters ``missing_keys`` to
    the tuner prefix, but we re-filter to keys carrying the LoRA prefix (``lora_``) ourselves so a
    benign base-weight miss never aborts a correct warm-start even if peft's internal filtering
    changes. Raises if any injected LoRA module got no saved weight (``missing_keys``) or any saved
    LoRA key matched no module (``unexpected_keys``) — i.e. matched != saved.
    """

    def _lora_only(keys):
        # the #67 mismatch keys (e.g. ...lora_A.default.weight) all carry this prefix; base-model
        # params do not, so this drops the benign base misses peft can report under strict=False.
        return [k for k in (keys or []) if "lora_" in k]

    missing = _lora_only(getattr(load_result, "missing_keys", None))
    unexpected = _lora_only(getattr(load_result, "unexpected_keys", None))
    if missing or unexpected:
        raise RuntimeError(
            f"warm-start adapter for {model_id} did NOT load cleanly: {len(missing)} injected LoRA "
            f"module(s) got no saved weight (missing) and {len(unexpected)} saved adapter key(s) "
            "matched no module (unexpected). The adapter was silently discarded -> GRPO would restart "
            "from the base model. For Qwen3.5/3.6 VL this is the '.language_model.' key mismatch "
            "(check remap_vl_adapter_dir ran on the adapter); otherwise the adapter's keys don't match "
            f"the base. missing[:3]={missing[:3]} unexpected[:3]={unexpected[:3]}"
        )
    print(
        f"[init-adapter] adapter load matched all saved keys for {model_id} (no missing/unexpected)"
    )


def assert_adapter_delta_nonzero(model, model_id: str) -> int:
    """Assert at least one ``lora_B`` weight is non-zero — the adapter is not an identity no-op.

    With standard zero-B init (``init_lora_weights=True``), a freshly-injected-but-unloaded adapter
    has ``lora_B == 0`` everywhere, so the effective delta ``(B @ A) * scaling`` is identically zero
    and the warm-started model equals the base. A real SFT adapter that actually loaded has non-zero
    ``lora_B``. This is an API-independent backstop to ``assert_adapter_load_clean``: it catches a
    silent discard even if peft's load-result shape changes. Returns the count of non-zero ``lora_B``
    modules. When no ``lora_B`` modules exist at all, defers to ``assert_lora_applied`` (no raise).
    """
    seen = 0
    nonzero = 0
    for name, module in model.named_modules():
        if not name.endswith("lora_B.default"):
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        seen += 1
        if bool(weight.detach().ne(0).any()):
            nonzero += 1
    if seen and nonzero == 0:
        raise RuntimeError(
            f"warm-start adapter for {model_id} has ALL-ZERO lora_B weights across {seen} module(s) — "
            "the adapter delta is identically zero (an unloaded / silently-discarded adapter). GRPO "
            "would train from the base model. Verify the adapter's keys match the base (see "
            "remap_vl_adapter_dir)."
        )
    print(f"[init-adapter] verified non-zero lora_B in {nonzero}/{seen} module(s) for {model_id}")
    return nonzero


def model_quant(model_id: str) -> str:
    """Quantization tier for this model: catalog entry > bf16 (managed; no override).

    The whole catalog is bf16, so this always returns ``"bf16"`` today; kept as the single
    source of truth a future non-bf16 tier could feed (no caller branches on it now)."""
    try:
        from flash.catalog import MODELS

        info = MODELS.get(model_id)
        if info is not None:
            return info.quant
    except Exception as e:
        print("model_quant: catalog probe failed:", e)
    return "bf16"

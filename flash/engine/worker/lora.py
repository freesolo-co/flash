"""Pure LoRA-target / VL-checkpoint helpers for the fine-tuning worker.

These helpers take the model id as an ARGUMENT and read NONE of the worker's run-scoped
module globals, so they live here as a leaf module. ``flash.engine.worker`` re-exports
them; this module must NOT import that package (no cycle). Heavy deps (transformers, peft,
vllm, the catalog) are imported lazily inside the functions so the module stays
CPU-importable.
"""

from __future__ import annotations

# Module-path segments that must never receive LoRA on natively-multimodal checkpoints
# trained text-only: the vision tower / projector / MTP head. Critically, adapters that
# DO touch them cannot be loaded by vLLM in text-only (language_model_only) serving —
# its LoRA loader rejects "unexpected modules" (observed with Qwen3.5-2B).
_VL_EXCLUDE_SEGMENTS = ("visual", "vision_tower", "multi_modal_projector", "mtp")
SERVING_MAX_LORA_RANK = 32


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

    The trainer (built via ``AutoModelForCausalLM``) names its LM params under ``model.*`` while the
    colocated vLLM engine loaded the same checkpoint as ``Qwen3_5ForConditionalGeneration`` whose LM
    params live under ``language_model.*``. Each incoming ``(name, tensor)`` is remapped per the form
    the trainer's checkpoint class produced:

    - peft wrapper: a ``base_model.model.`` prefix (a continued/merged-adapter sync surfacing
      base-model names through the PeftModel wrapper) is STRIPPED first, so the rules below apply to
      the unwrapped name.
    - multimodal-named trainer (``model.language_model.*`` / ``model.visual.*`` / ``lm_head.*`` — what
      ``AutoModelForCausalLM`` yields when it resolves the FULL ``*ForConditionalGeneration``, e.g. the
      Qwen3.6-35B-A3B MoE): passed through UNTOUCHED, because vLLM's own ``hf_to_vllm_mapper`` already
      maps these. Prepending ``language_model.`` here would double the prefix and crash the fused-MoE
      expert lookup (see the inline comment).
    - text-only dense trainer (bare ``model.*`` — the dense Qwen3.5 family's ``Qwen3_5ForCausalLM``,
      no infix, no vision tower): prefixed with ``language_model.`` so it lands under
      ``language_model.model.*``.
    - anything already ``language_model.*`` (or otherwise unmatched): passed through untouched.

    A generator so vLLM's loader still streams one (name, tensor) at a time.
    """
    for name, tensor in weights:
        # A continued-adapter (PeftModel) sync can surface names through the peft wrapper as
        # ``base_model.model.model.layers.*`` / ``base_model.model.lm_head.*``; strip the wrapper
        # so the same model./lm_head. rule applies.
        if name.startswith("base_model.model."):
            name = name[len("base_model.model.") :]
        # MULTIMODAL-named trainers: when AutoModelForCausalLM resolves the checkpoint to the FULL
        # ``*ForConditionalGeneration`` (the Qwen3.6-35B-A3B MoE does — its params are
        # ``model.language_model.*`` / ``model.visual.*`` / ``lm_head.*``), the vLLM engine's OWN
        # ``hf_to_vllm_mapper`` already maps these (``model.language_model.`` -> ``language_model.model.``,
        # ``model.visual.`` -> ``visual.``, ``lm_head.`` -> ``language_model.lm_head.``). Pass them
        # through UNTOUCHED so the sync is byte-identical to the proven initial on-disk load.
        # Prepending ``language_model.`` here would yield ``language_model.model.language_model.*``,
        # which the mapper's ``startswith("model.language_model.")`` rule no longer matches -> the
        # fused-MoE expert lookup then crashes with
        # ``KeyError: 'language_model.layers.N.mlp.experts.w13_weight'`` (the ``.model.`` segment is
        # lost in the AutoWeightsLoader recursion).
        if name.startswith(("model.language_model.", "model.visual.", "lm_head.")):
            yield name, tensor
            continue
        # TEXT-ONLY trainers: the dense Qwen3.5 family resolves to ``Qwen3_5ForCausalLM``
        # (``model.layers.*`` / ``model.norm`` / ``model.embed_tokens``, no infix, no vision tower).
        # The mapper has NO bare-``model.`` rule, so prepend ``language_model.`` ourselves to land
        # them under ``language_model.model.*`` (unchanged behavior for dense GRPO).
        if name.startswith("model."):
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
            # NB: the MoE class lives in the SAME ``qwen3_5`` module (there is no ``qwen3_5_moe``
            # module). It also inherits the dense class's (patched) ``load_weights``, but patch it
            # explicitly too so the remap is guaranteed active for the MoE rollout engine.
            ("vllm.model_executor.models.qwen3_5", "Qwen3_5MoeForConditionalGeneration", False),
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

    Targets MULTI-TURN GRPO, where the masked set is the env/tool text (~half-to-most of the
    transcript: the rollout's ``env_mask`` -> TRL's ``tool_mask``) that EVERY row carries, so the
    micro-batch has maskable headroom in all rows. TRL 1.6's ``compute_liger_loss`` hands the
    FULL-length hidden states to ``liger_grpo_loss``, and the Liger kernel runs the lm_head matmul +
    log-softmax for EVERY position (in the forward AND the backward recompute). Masked positions
    contribute zero loss and zero gradient but still pay the full FLOPs of the single most expensive
    GRPO op (the 248k-vocab projection Liger exists to tame). The saving scales with the env-masked
    fraction. (SINGLE-TURN is effectively a no-op: its only mask is right-padding, and TRL pads
    completions to the LONGEST in the micro-batch, so the deepest row has ``keep.sum() == full_t`` and
    the across-batch no-op below triggers — there is no shared headroom to gather. It would engage
    only if ``pad_to_multiple_of`` padded every row past the longest completion.)

    Wrap ``trainer.liger_grpo_loss`` to GATHER the unmasked positions — ONE shared index applied
    identically to every per-token tensor (``_input``, ``selected_token_ids``, ``attention_mask``,
    ``old_per_token_logps``, ``ref_per_token_logps``, and a 2-D ``vllm_is_ratio``) — before the call,
    so the kernel only projects the kept positions. Per-sequence ``advantages`` ``(B,)`` and the loss
    object's ``max_completion_length`` are left untouched. This is EXACTLY loss-preserving: dr_grpo's
    numerator only ever summed unmasked positions, and its normalizer is ``B * max_completion_length``
    (a config constant on the loss object, independent of the gathered length); the gathered
    sequence is re-padded with masked positions whose new mask is 0, so loss + credit assignment are
    unchanged while the gathered length T' < T cuts the projection FLOPs by ~the masked fraction.
    No-op when the deepest row is full-length (``max(unmasked) == T`` — e.g. single-turn padded to the
    batch max), when nothing is masked at all, or when the loss object isn't present. Returns True if
    wrapped."""
    orig = getattr(trainer, "liger_grpo_loss", None)
    if orig is None:
        return False
    if getattr(orig, "_flash_mask_aware", False):
        return True  # already wrapped — idempotent (mirrors the other patch helpers' sentinels)
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
            # Nothing maskable to skip across the batch: the DEEPEST row is full-length (max unmasked
            # == T). Standard single-turn vLLM GRPO pads completions to the longest in the micro-batch,
            # so this is its common case — the patch only engages where every row has masked headroom.
            return orig(**kwargs)
        # Defensive: we gather a KNOWN set of per-token tensors below. If TRL/Liger starts passing any
        # OTHER per-token tensor shaped (B, T[, *]), it would stay full-length while the rest are
        # gathered to T' -> a shape mismatch or misaligned credit. Bail to the unmodified loss instead.
        # (Per-sequence ``advantages`` is (B,) and 2-D ``vllm_is_ratio`` is handled explicitly below.)
        _known = {"attention_mask", "_input", "selected_token_ids", "old_per_token_logps",
                  "ref_per_token_logps", "vllm_is_ratio"}
        for _k, _v in kwargs.items():
            if (_k not in _known and isinstance(_v, torch.Tensor) and _v.dim() >= 2
                    and _v.size(0) == mask.size(0) and _v.size(1) == full_t):
                return orig(**kwargs)  # unknown per-token tensor -> don't risk a misaligned gather
        # One shared gather index: the unmasked positions first (stable argsort -> their original
        # order preserved), then the remaining masked positions in original order. Keep only the
        # first tprime columns; a sequence with fewer than tprime unmasked positions has its filler
        # entries taken from its masked positions, whose gathered mask is 0 — so they add zero
        # loss/grad and can't perturb the per-token ratio/KL alignment.
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
        # The gathered tensors have shape (B, tprime) where tprime varies per micro-batch
        # (it is the max unmasked-position count across the batch). torch.compile inside
        # liger_kernel's compiled_compute_loss builds SHAPE_ENV guards keyed on static tensor
        # dimensions; when tprime changes between calls, guard recompilation hits a
        # symbol_to_source IndexError (InternalTorchDynamoError). Running the gathered call
        # without torch.compile is still faster than the unmasked path: the gather already
        # eliminated the masked FLOPs; eager overhead is negligible at 0.8B scale.
        import torch._dynamo as _dynamo

        _disabled_orig = getattr(masked_liger_loss, "_flash_disabled_orig", None)
        if _disabled_orig is None:
            _disabled_orig = _dynamo.disable(orig)
            masked_liger_loss._flash_disabled_orig = _disabled_orig
        return _disabled_orig(**gk)

    masked_liger_loss._flash_mask_aware = True  # sentinel for the idempotency check above
    trainer.liger_grpo_loss = masked_liger_loss
    return True


def disable_liger_grpo_torch_compile(trainer) -> bool:
    """Run liger's fused GRPO loss EAGER — drop only its ``torch.compile``, keep the memory path.

    ``LigerFusedLinearGRPOLoss`` wraps ONLY the loss math
    (``fused_linear_ppo._compute_loss_from_logps``) in ``torch.compile`` (gated by its ``compiled``
    flag, default True); the memory-efficient part — the chunked custom-autograd ``chunk_forward``
    that never materializes the fp32 ``[batch, seq, ~248k vocab]`` logits — ALWAYS runs eager. On
    torch 2.10 that ``torch.compile`` is BROKEN: its SHAPE_ENV guards are keyed on the per-call tensor
    dims and guard generation trips a torch bug (``symbol_to_source`` IndexError surfaced as
    ``InternalTorchDynamoError`` — "list index out of range" at ``symbolic_shapes.issue_guard``) that
    crashes the FIRST GRPO step on EVERY path (single-turn, multi-turn, tool). It fires during
    guard-build (after tracing), so neither the multi-turn ``suppress_errors=True`` nor the mask-aware
    path's ``_dynamo.disable`` catches it.

    Setting ``compiled=False`` makes liger skip the ``torch.compile`` wrapper entirely while KEEPING
    the chunked memory path — so the 248k-vocab fp32-logit OOM fix (the whole reason
    ``use_liger_kernel`` stays on for GRPO) is fully retained; only the loss-math JIT is dropped, and
    its eager overhead is negligible at these tiny per-token GEMMs. Call this BEFORE
    ``patch_grpo_mask_aware_lm_head`` (which replaces ``liger_grpo_loss`` with a closure) so it lands
    on the live ``LigerFusedLinearGRPOLoss`` instance. No-op (returns False) when the loss isn't
    present, predates the ``compiled`` flag, or already has it off. Returns True if it flipped it."""
    loss = getattr(trainer, "liger_grpo_loss", None)
    if loss is None or not getattr(loss, "compiled", False):
        return False
    loss.compiled = False
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
    """Tensor key names in the downloaded adapter, read ONLY from the ``.safetensors`` JSON header
    (pure stdlib, no tensor data — keeps this module CPU-importable). Returns ``None`` when no
    ``adapter_model.safetensors`` exists in ``adir``.
    """
    import json
    import os
    import struct

    st_path = os.path.join(adir, "adapter_model.safetensors")
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
            try:
                header = json.loads(header_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                # A bare JSONDecodeError ("Expecting value: line 1 column 1") — or a
                # UnicodeDecodeError from non-UTF8 header bytes — gives no clue WHICH adapter is
                # corrupt. Re-raise with the file path so a bad download is diagnosable.
                raise ValueError(
                    f"{st_path}: safetensors header is not valid JSON "
                    f"(corrupt or not a safetensors file): {exc}"
                ) from exc
        # The safetensors header MUST be a JSON object keyed by tensor name. A corrupt/hostile file
        # could decode to a list/int/str, which would later blow up with a confusing TypeError in
        # _is_lora_key (substring search on a non-str). (JSON object keys are always str, so only the
        # container type needs checking.) Reject a non-object header early with a clear message.
        if not isinstance(header, dict):
            raise ValueError(
                f"{st_path}: safetensors header is not a JSON object "
                "(corrupt or not a safetensors file)"
            )
        return [k for k in header if k != "__metadata__"]
    return None


def adapter_is_vl_warmstart(adir: str, model_id: str) -> bool:
    """Whether a warm-start adapter should take the VL merge-into-base path.

    Robust to a transient ``is_vl_checkpoint`` config-probe failure (it calls
    ``AutoConfig.from_pretrained`` and swallows EVERY exception to return False, so an HF
    rate-limit / network hiccup / uncached config could silently route a genuine VL warm-start down
    the text-only path and reintroduce the trainer<->vLLM mismatch — issue #286). An adapter that
    actually carries ``.language_model.`` LoRA keys was saved against the full multimodal model and
    IS a VL warm-start regardless of the probe (the adapter's own keys are the authoritative signal).
    Falls back to the config probe only when the adapter can't be read or carries no
    ``.language_model.`` LoRA keys (already-text-only / non-VL)."""
    try:
        keys = _read_adapter_tensor_keys(adir)
        if keys and any(_LANGUAGE_MODEL_INFIX in k for k in keys if _is_lora_key(k)):
            return True
    except Exception as e:  # best-effort: never let a key-read failure abort the launch
        # Name the adapter dir + model so a transient failure is tied to the specific warm-start
        # when several workers log into the same stream.
        print(
            f"[init-adapter] adapter VL-key probe failed for adir={adir!r} model={model_id!r}; "
            f"deferring to the config probe: {e}"
        )
    return is_vl_checkpoint(model_id)


def adapter_lora_rank(adapter_dir: str) -> int:
    """Read a saved PEFT LoRA adapter's scalar rank from adapter_config.json."""
    import json
    import os

    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        rank = int(cfg["r"])
    except FileNotFoundError as exc:
        raise ValueError(f"adapter rank preflight: missing {cfg_path!r}") from exc
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"adapter rank preflight: {cfg_path!r} must contain a positive integer `r`"
        ) from exc
    if rank <= 0:
        raise ValueError(
            f"adapter rank preflight: {cfg_path!r} has non-positive rank r={rank}"
        )
    return rank


def validate_recombined_lora_rank(
    sft_dir: str,
    grpo_rank: int,
    *,
    max_rank: int = SERVING_MAX_LORA_RANK,
) -> tuple[int, int, int]:
    """Fail before training when a VL SFT+GRPO recombine would exceed serving's rank cap."""
    try:
        grpo_rank = int(grpo_rank)
    except (TypeError, ValueError) as exc:
        raise ValueError("VL warm-start rank preflight: GRPO train.lora_rank must be an integer") from exc
    if grpo_rank <= 0:
        raise ValueError(
            f"VL warm-start rank preflight: GRPO train.lora_rank must be positive, got {grpo_rank}"
        )

    sft_rank = adapter_lora_rank(sft_dir)
    recombined_rank = sft_rank + grpo_rank
    if recombined_rank <= max_rank:
        return sft_rank, grpo_rank, recombined_rank

    allowed_grpo_rank = max_rank - sft_rank
    if allowed_grpo_rank >= 1:
        guidance = f"set GRPO [train].lora_rank <= {allowed_grpo_rank}"
    else:
        allowed_sft_rank = max_rank - grpo_rank
        if allowed_sft_rank >= 1:
            guidance = f"retrain the SFT adapter at rank <= {allowed_sft_rank}"
        else:
            guidance = f"lower both SFT and GRPO ranks so their sum is <= {max_rank}"
    raise ValueError(
        "VL warm-start rank preflight failed: recombined SFT+GRPO adapter would be "
        f"rank {recombined_rank} (SFT rank {sft_rank} + GRPO rank {grpo_rank}), "
        f"exceeding the serving LoRA rank cap {max_rank}. Because this warm-start path "
        f"rank-stacks the adapters for deploy, {guidance}."
    )


def recombine_lora_adapters(sft_dir: str, grpo_dir: str, out_dir: str) -> int:
    """Stack two LoRA adapters (SFT ⊕ GRPO) into ONE rank-(r_sft+r_grpo) adapter in ``out_dir``.

    The VL warm-start path (#296) MERGES the SFT into the base and trains a FRESH LoRA on the merged
    weights, so the saved GRPO adapter is a delta RELATIVE TO base+SFT. Deploying it alone on the
    catalog base drops the SFT entirely (served output collapses to ~base). Concatenating the two
    LoRAs reproduces ``base + SFT_delta + GRPO_delta`` — the exact model GRPO trained — on the
    ORIGINAL base: for each module, ``A_out = cat([A_sft, A_grpo], 0)`` and
    ``B_out = cat([s_sft·B_sft, s_grpo·B_grpo], 1)`` with each adapter's own scale (``alpha/r``, or
    ``alpha/√r`` under rsLoRA) BAKED into its ``B`` and the combined adapter set to unit scale
    (``alpha=r`` ⇒ ``alpha/r=1``). Then ``delta_out = B_out @ A_out = s_sft·B_sft@A_sft +
    s_grpo·B_grpo@A_grpo`` exactly, for ANY input scales. Returns the combined rank.

    Both adapters must target the SAME modules (true for managed flash: target_modules is
    model-derived, not user-set) and be plain LoRA — DoRA / ``modules_to_save`` / non-LoRA tensors
    raise (a wrong recombine would silently mis-deploy, so fail LOUDLY instead).
    """
    import json
    import math
    import os

    import torch
    from safetensors.torch import load_file, save_file

    def _load(d: str):
        cfg_path = os.path.join(d, "adapter_config.json")
        st_path = os.path.join(d, "adapter_model.safetensors")
        if os.path.isfile(st_path):
            sd = load_file(st_path)
        else:
            raise ValueError(
                f"recombine: {d!r} has no adapter_model.safetensors "
                f"(dir contents: {sorted(os.listdir(d)) if os.path.isdir(d) else 'MISSING'}); "
                "only safetensors LoRA adapters can be recombined"
            )
        if not os.path.isfile(cfg_path):
            raise ValueError(
                f"recombine: {d!r} has no adapter_config.json "
                f"(dir contents: {sorted(os.listdir(d)) if os.path.isdir(d) else 'MISSING'}); "
                "the adapter config is required to read its rank/alpha/scale"
            )
        with open(cfg_path) as f:
            return json.load(f), sd

    sft_cfg, sft_sd = _load(sft_dir)
    grpo_cfg, grpo_sd = _load(grpo_dir)

    # Normalize the ``.language_model.`` infix on BOTH adapters' keys before comparing/stacking.
    # The VL merge warm-start (#296) trains the SFT against the FULL multimodal model, so the SFT
    # adapter's keys carry the infix (``base_model.model.model.language_model.layers...``), while the
    # fresh GRPO LoRA is saved by the text-only ``AutoModelForCausalLM`` trainer with no infix
    # (``base_model.model.model.layers...`` — see _remap_vl_sync_weights / the warm-start remap note).
    # Without this, the equivalent LM modules compare as DIFFERENT targets and the recombine wrongly
    # aborts for the default Qwen3.5 warm-start path. Stripping is idempotent (a no-op for an already
    # text-only adapter) and emits the text-only key form serving deploys on the catalog base.
    def _normalize_infix(sd, which):
        norm: dict = {}
        for k, v in sd.items():
            nk = strip_language_model_infix(k)
            # Fail closed on ANY post-normalization collision. A malformed adapter carrying BOTH the
            # infixed and the already-text-only form of a key would otherwise let the second write
            # silently overwrite the first (the text-only key has nk == k, so a ``nk != k`` guard
            # would skip it) — recombining whichever duplicate wins instead of rejecting the mix.
            if nk in norm:
                raise ValueError(
                    f"recombine: {which} adapter key {k!r} collides with another after stripping "
                    "the '.language_model.' infix — cannot normalize a mixed VL adapter"
                )
            norm[nk] = v
        return norm

    sft_sd = _normalize_infix(sft_sd, "SFT")
    grpo_sd = _normalize_infix(grpo_sd, "GRPO")

    for name, cfg in (("SFT", sft_cfg), ("GRPO", grpo_cfg)):
        # peft_type defaults to LORA for older configs that omit it; anything else (e.g. ADALORA)
        # has different tensor/scale semantics the cat-recombine math doesn't model.
        peft_type = (cfg.get("peft_type") or "LORA").upper()
        if peft_type != "LORA":
            raise ValueError(
                f"recombine: {name} adapter peft_type={peft_type!r} (not plain LORA) — "
                "cat-recombine is unsupported"
            )
        if cfg.get("use_dora"):
            raise ValueError(f"recombine: {name} adapter uses DoRA — cat-recombine is unsupported")
        if cfg.get("modules_to_save"):
            raise ValueError(
                f"recombine: {name} adapter has modules_to_save={cfg['modules_to_save']!r} "
                "(full-weight tensors) — cat-recombine is unsupported"
            )
        # Per-module rank/alpha overrides break the single-(r, alpha) assumption _scale() and the
        # rank-stacking below rely on — and out_cfg blanks both, which would SILENTLY drop them.
        for key in ("rank_pattern", "alpha_pattern"):
            if cfg.get(key):
                raise ValueError(
                    f"recombine: {name} adapter has a non-empty {key}={cfg[key]!r} (per-module "
                    "rank/alpha) — cat-recombine assumes a uniform rank and is unsupported"
                )

    # Real PEFT adapters embed the adapter NAME in the saved key (``...lora_A.default.weight`` — see
    # tests/test_vl_warmstart_adapter_keys.py and _is_lora_key), not the bare ``...lora_A.weight``
    # form. Match the ``.lora_A.``/``.lora_B.`` weight tensors by infix so BOTH the ``.default.`` and
    # the bare form are recognized — otherwise real keys fall into ``extra`` below and the recombine
    # wrongly aborts as "non-LoRA tensors present", blocking the VL warm-start deploy.
    def _is_lora_a(k):
        return ".lora_A." in k and k.endswith(".weight")

    def _is_lora_b(k):
        return ".lora_B." in k and k.endswith(".weight")

    def _ab(sd):
        return {k for k in sd if _is_lora_a(k) or _is_lora_b(k)}

    sft_ab, grpo_ab = _ab(sft_sd), _ab(grpo_sd)
    extra = (set(sft_sd) - sft_ab) | (set(grpo_sd) - grpo_ab)
    if extra:
        raise ValueError(
            f"recombine: non-LoRA tensors present (e.g. {sorted(extra)[:4]}) — only plain "
            "lora_A/lora_B adapters can be recombined"
        )
    if sft_ab != grpo_ab:
        only_sft, only_grpo = sorted(sft_ab - grpo_ab)[:3], sorted(grpo_ab - sft_ab)[:3]
        raise ValueError(
            "recombine: SFT and GRPO adapters target DIFFERENT modules "
            f"(only-SFT={only_sft}, only-GRPO={only_grpo}); their target_modules must match for a "
            "rank-stacked recombine"
        )
    # Symmetric to the per-A pairing guard in the loop below: the loop iterates only lora_A keys, so a
    # lora_B with no paired lora_A would be silently DROPPED from the output (an incomplete adapter),
    # not caught by the equal-key-set check above. Fail loudly, like the A-without-B case.
    orphan_b = sorted(
        bk for bk in sft_ab if _is_lora_b(bk) and bk.replace(".lora_B.", ".lora_A.", 1) not in sft_ab
    )
    if orphan_b:
        raise ValueError(
            f"recombine: lora_B key(s) {orphan_b[:3]} have no matching lora_A — the adapter is "
            "malformed (unpaired LoRA tensors)"
        )

    def _scale(cfg) -> float:
        r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
        return alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r

    s_sft, s_grpo = _scale(sft_cfg), _scale(grpo_cfg)
    r_sft, r_grpo = int(sft_cfg["r"]), int(grpo_cfg["r"])
    r_out = r_sft + r_grpo
    if r_out > SERVING_MAX_LORA_RANK:
        raise ValueError(
            "recombine: rank-stacked SFT+GRPO adapter would be "
            f"rank {r_out} (SFT rank {r_sft} + GRPO rank {r_grpo}), exceeding the serving "
            f"LoRA rank cap {SERVING_MAX_LORA_RANK}"
        )

    out: dict[str, torch.Tensor] = {}
    # Sorted so ``out``'s insertion order — and thus the serialized safetensors byte layout — is
    # deterministic across runs (``sft_ab`` is a set; its iteration order is not). Stable output
    # keeps the recombined adapter content-addressable for uploads/caching.
    for ak in sorted(k for k in sft_ab if _is_lora_a(k)):
        # Pair B by swapping the A/B marker — robust to the ``.default.`` adapter-name segment that
        # bare ``lora_A.weight`` -> ``lora_B.weight`` suffix slicing would miss.
        bk = ak.replace(".lora_A.", ".lora_B.", 1)
        # Both adapters carry the same A/B key set (asserted above), but a malformed adapter could
        # have a lora_A with no paired lora_B — name the missing key rather than throw a bare
        # KeyError on the indexing below. (``bk in sft_ab`` ⇒ present in both state dicts.)
        if bk not in sft_ab:
            raise ValueError(
                f"recombine: lora_A key {ak!r} has no matching lora_B key {bk!r} — the adapter is "
                "malformed (unpaired LoRA tensors)"
            )
        # A: (r, in_features) — stacked along the rank axis (no scaling; scale lives on B).
        out[ak] = torch.cat([sft_sd[ak], grpo_sd[ak]], dim=0).contiguous()
        # B: (out_features, r) — bake each adapter's own scale, then stack along the rank axis. The
        # combined adapter carries unit scale, so the per-component scales survive intact.
        # Promote to the higher of the two input dtypes (don't force the SFT's, which would downcast a
        # higher-precision GRPO B); A is cat'd as-is and auto-promotes the same way, so A/B stay consistent.
        dt = torch.promote_types(sft_sd[bk].dtype, grpo_sd[bk].dtype)
        b_sft = sft_sd[bk].to(torch.float32) * s_sft
        b_grpo = grpo_sd[bk].to(torch.float32) * s_grpo
        out[bk] = torch.cat([b_sft, b_grpo], dim=1).to(dt).contiguous()

    out_cfg = dict(grpo_cfg)
    out_cfg["r"] = r_out
    out_cfg["lora_alpha"] = r_out  # alpha/r == 1.0 (scales already baked into each B block)
    out_cfg["use_rslora"] = False
    out_cfg["rank_pattern"] = {}
    out_cfg["alpha_pattern"] = {}
    # The GRPO adapter was trained on the ephemeral SFT-merged temp dir, so out_cfg (copied from
    # grpo_cfg) still names that temp path as base_model_name_or_path. Replace it with the real catalog
    # base from the SFT config; if the SFT config carries none (e.g. an external/legacy adapter), DROP
    # the field rather than ship a deployed config pointing at a now-deleted temp dir.
    sft_base = sft_cfg.get("base_model_name_or_path")
    if sft_base:
        out_cfg["base_model_name_or_path"] = sft_base
    else:
        out_cfg.pop("base_model_name_or_path", None)

    os.makedirs(out_dir, exist_ok=True)
    save_file(out, os.path.join(out_dir, "adapter_model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)
    return r_out


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
            "3.6 VL this is usually a '.language_model.' key-mismatch; otherwise verify the adapter's "
            "keys match the model."
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
            "from the base model. For Qwen3.5/3.6 VL this is the '.language_model.' key mismatch; "
            "otherwise the adapter's keys don't match "
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
            "would train from the base model. Verify the adapter's keys match the base."
        )
    print(f"[init-adapter] verified non-zero lora_B in {nonzero}/{seen} module(s) for {model_id}")
    return nonzero

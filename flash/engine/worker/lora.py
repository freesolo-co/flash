"""Pure LoRA-target / VL-checkpoint helpers for the fine-tuning worker."""

from __future__ import annotations

# vLLM text-only serving rejects adapters that touch these vision/projector/MTP segments.
_VL_EXCLUDE_SEGMENTS = ("visual", "vision_tower", "multi_modal_projector", "mtp")


def lora_exclude_modules(model_id: str) -> str | None:
    """Regex (peft fullmatch semantics) excluding vision-tower modules from LoRA.

    peft's list-form exclude_modules suffix-matches and won't match leaf modules under
    'visual.*' — a regex string is required.
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
    """Engine kwargs to skip the vision tower for VL checkpoints (vLLM >= 0.19)."""
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


# A dict (not a bare bool) so the gating flag is shared by reference with the worker that flips it.
_LM_SYNC_REMAP_ON = {"on": False}


def _remap_vl_sync_weights(weights):
    """Rewrite TRL's trainer weight names to vLLM's VL-engine names for the train-time sync."""
    for name, tensor in weights:
        if name.startswith("base_model.model."):
            name = name[len("base_model.model.") :]
        # Multimodal-named params: vLLM's hf_to_vllm_mapper already maps these; prepending
        # language_model. would crash the fused-MoE expert lookup.
        if name.startswith(("model.language_model.", "model.visual.", "lm_head.")):
            yield name, tensor
            continue
        # Dense text-only: the mapper has no bare-model. rule, so prepend language_model. ourselves.
        if name.startswith("model."):
            name = "language_model." + name
        yield name, tensor


def patch_vllm_lm_weight_sync(model_id: str) -> bool:
    """Make TRL's GRPO ``sync_weights`` work for ``*ForConditionalGeneration`` checkpoints.

    Returns True if any vLLM model class was patched. The remap only runs while
    ``_LM_SYNC_REMAP_ON`` is set, so the initial on-disk load stays untouched.
    """
    if not is_vl_checkpoint(model_id):
        return False
    patched_any = False
    try:
        import importlib

        # Dense class is required (log loudly if missing); the MoE class is optional (stays quiet).
        for mod_name, cls_name, required in (
            ("vllm.model_executor.models.qwen3_5", "Qwen3_5ForConditionalGeneration", True),
            # MoE class lives in the same qwen3_5 module; patch it explicitly for the MoE engine.
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

    Targets multi-turn GRPO: gathers the unmasked positions before the Liger call so the kernel
    only projects them. Exactly loss-preserving; no-op when the deepest row is full-length.
    """
    orig = getattr(trainer, "liger_grpo_loss", None)
    if orig is None:
        return False
    if getattr(orig, "_flash_mask_aware", False):
        return True
    import torch

    def _gather(x, idx, tprime):
        if x is None:
            return None
        if x.dim() == 2:
            return torch.gather(x, 1, idx)
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(idx.size(0), tprime, x.size(-1)))

    def masked_liger_loss(*args, **kwargs):
        mask = kwargs.get("attention_mask")
        if args or mask is None or mask.dim() != 2:
            return orig(*args, **kwargs)
        keep = mask != 0
        full_t = mask.size(1)
        tprime = int(keep.sum(dim=1).max().item())
        if tprime == 0 or tprime == full_t:
            # Deepest row is full-length: no shared headroom to gather (single-turn's common case).
            return orig(**kwargs)
        # Bail if TRL/Liger passes an unknown per-token (B,T) tensor we don't gather — would misalign.
        _known = {"attention_mask", "_input", "selected_token_ids", "old_per_token_logps",
                  "ref_per_token_logps", "vllm_is_ratio"}
        for _k, _v in kwargs.items():
            if (_k not in _known and isinstance(_v, torch.Tensor) and _v.dim() >= 2
                    and _v.size(0) == mask.size(0) and _v.size(1) == full_t):
                return orig(**kwargs)
        # Shared gather index: unmasked positions first (stable argsort), keep first tprime columns.
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
        # Run eager: a varying tprime trips a torch.compile SHAPE_ENV guard bug (symbol_to_source).
        import torch._dynamo as _dynamo

        _disabled_orig = getattr(masked_liger_loss, "_flash_disabled_orig", None)
        if _disabled_orig is None:
            _disabled_orig = _dynamo.disable(orig)
            masked_liger_loss._flash_disabled_orig = _disabled_orig
        return _disabled_orig(**gk)

    masked_liger_loss._flash_mask_aware = True
    trainer.liger_grpo_loss = masked_liger_loss
    return True


def disable_liger_grpo_torch_compile(trainer) -> bool:
    """Run liger's fused GRPO loss EAGER — drop only its ``torch.compile``, keep the memory path.

    On torch 2.10 that compile is broken (SHAPE_ENV symbol_to_source IndexError) and crashes the
    first GRPO step. Call BEFORE ``patch_grpo_mask_aware_lm_head``. Returns True if it flipped it.
    """
    loss = getattr(trainer, "liger_grpo_loss", None)
    if loss is None or not getattr(loss, "compiled", False):
        return False
    loss.compiled = False
    return True


# VL SFT adapters save LM keys under .language_model.; strip the infix so they match the text-only
# AutoModelForCausalLM trainer used by warm-started GRPO (else peft silently keeps a zero-init LoRA).
_LANGUAGE_MODEL_INFIX = ".language_model."


def strip_language_model_infix(key: str) -> str:
    """Strip the FIRST ``.language_model.`` infix from a peft adapter weight key."""
    i = key.find(_LANGUAGE_MODEL_INFIX)
    if i == -1:
        return key
    return key[:i] + "." + key[i + len(_LANGUAGE_MODEL_INFIX) :]


def remap_adapter_keys(keys):
    """Map an iterable of adapter weight keys -> a dict {old_key: new_key} for keys that change."""
    out = {}
    for k in keys:
        nk = strip_language_model_infix(k)
        if nk != k:
            out[k] = nk
    return out


# Generous ceiling that still refuses a corrupt file declaring a multi-GB header before we read it.
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


def _read_safetensors_header(path: str) -> tuple[dict, int]:
    """Read and FULLY validate a ``.safetensors`` file's JSON header.

    Returns ``(header, data_start)`` where ``data_start`` is ``8 + hdr_len``. Pure stdlib so this
    module stays CPU-importable. Validates the declared length before allocating; errors name #198.
    """
    import json
    import os
    import struct

    # Bound the declared header length against the real file size before reading it.
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        len_bytes = f.read(8)
        if len(len_bytes) < 8:
            raise ValueError(f"{path}: too small to be a safetensors file")
        (hdr_len,) = struct.unpack("<Q", len_bytes)
        if hdr_len > file_size - 8 or hdr_len > _MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(
                f"{path}: declared safetensors header length {hdr_len} is implausible "
                f"(file is {file_size} bytes) — refusing to read a corrupt/oversized header"
            )
        header_bytes = f.read(hdr_len)
        if len(header_bytes) < hdr_len:
            raise ValueError(f"{path}: truncated safetensors header")
        try:
            header = json.loads(header_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # Re-raise with the file path so a bad download is diagnosable.
            raise ValueError(
                f"{path}: safetensors header is not valid JSON "
                f"(corrupt or not a safetensors file): {exc}"
            ) from exc
    # Header must be a JSON object; reject early (else a confusing TypeError later in _is_lora_key).
    if not isinstance(header, dict):
        raise ValueError(
            f"{path}: safetensors header is not a JSON object "
            "(corrupt or not a safetensors file)"
        )
    return header, 8 + hdr_len


def _rename_keys(mapping, rename, *, path: str, skip=()):
    """Apply ``rename`` to every key of ``mapping`` -> ``(new_mapping, renamed_count)``.

    Raises ``ValueError`` if a renamed key would collide with an existing/already-renamed key.
    """
    new_mapping = {}
    renamed = 0
    for k, v in mapping.items():
        if k in skip:
            new_mapping[k] = v
            continue
        nk = rename(k)
        if nk != k:
            if nk in mapping or nk in new_mapping:
                raise ValueError(
                    f"{path}: remapped key {nk!r} collides with an existing key; refusing to "
                    f"overwrite (adapter may already be remapped or malformed)"
                )
            renamed += 1
        new_mapping[nk] = v
    return new_mapping, renamed


def _rewrite_safetensors_header_keys(path: str, rename) -> int:
    """Rename tensor keys in a ``.safetensors`` file IN PLACE, editing only the header."""
    import json
    import os
    import shutil
    import struct

    header, data_start = _read_safetensors_header(path)
    new_header, renamed = _rename_keys(header, rename, path=path, skip=("__metadata__",))
    if renamed == 0:
        return 0

    new_header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    # Stream to a temp file + atomic os.replace so an interrupted rewrite can't corrupt the adapter.
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
    """Rename keys in a PyTorch ``.bin`` (pickled ``state_dict``) adapter IN PLACE."""
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=True)
    new_sd, renamed = _rename_keys(sd, rename, path=path)
    if renamed == 0:
        return 0
    torch.save(new_sd, path)
    return renamed


# Substrings identifying a peft LoRA weight key (vs base params in a wrong-arch/corrupt checkpoint).
_LORA_KEY_MARKERS = (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B.", "lora_")


def _is_lora_key(key: str) -> bool:
    return any(m in key for m in _LORA_KEY_MARKERS)


def _read_adapter_tensor_keys(adir: str) -> list[str] | None:
    """Tensor key names in the downloaded adapter, or ``None`` when no weight file exists."""
    import os

    st_path = os.path.join(adir, "adapter_model.safetensors")
    bin_path = os.path.join(adir, "adapter_model.bin")
    if os.path.isfile(st_path):
        header, _ = _read_safetensors_header(st_path)
        return [k for k in header if k != "__metadata__"]
    if os.path.isfile(bin_path):
        import torch

        sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        return list(sd.keys())
    return None


def remap_vl_adapter_dir(adir: str, model_id: str) -> int:
    """Strip the ``.language_model.`` infix from a VL warm-start's downloaded SFT adapter.

    Driven by the adapter's OWN keys, not just the (exception-swallowing) ``is_vl_checkpoint`` probe
    (#286). Fails loudly on a mismatched/corrupt adapter. Returns the number of keys renamed.
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

    # No infixed LoRA keys: fall back to the config probe (only needed in this branch).
    if not infixed:
        if not is_vl_checkpoint(model_id):
            return 0
        if not lora_keys:
            # No LoRA weights at all: fail before the base-model download.
            raise RuntimeError(
                f"warm-start adapter in {adir!r} for {model_id} contains NO LoRA weight keys "
                f"(found {len(keys)} tensor(s), 0 with a lora_ marker) — the adapter is corrupt, "
                "incomplete, or from a different architecture, so GRPO would train from the base "
                "model. Re-export the SFT adapter, or omit train.init_from_adapter for a fresh LoRA."
            )
        # Nothing to strip: surface the actual LoRA prefix so a real key mismatch isn't a silent no-op.
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

    # Infixed keys must be stripped regardless of the config probe (#286). Fail closed before touching
    # disk: a key carrying the infix twice would survive the single strip and be silently discarded.
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
    else:
        n = _rewrite_bin_keys(bin_path, strip_language_model_infix)

    print(
        f"[init-adapter] remapped {n} VL SFT adapter key(s): stripped '.language_model.' infix "
        f"to match the AutoModelForCausalLM trainer for {model_id}"
    )
    return n


def adapter_is_vl_warmstart(adir: str, model_id: str) -> bool:
    """Whether a warm-start adapter should take the VL merge-into-base path.

    Robust to a transient ``is_vl_checkpoint`` config-probe failure (it calls
    ``AutoConfig.from_pretrained`` and swallows EVERY exception to return False, so an HF
    rate-limit / network hiccup / uncached config could silently route a genuine VL warm-start down
    the text-only path and reintroduce the trainer<->vLLM mismatch — issue #286). An adapter that
    actually carries ``.language_model.`` LoRA keys was saved against the full multimodal model and
    IS a VL warm-start regardless of the probe (the SAME authoritative file-content signal
    ``remap_vl_adapter_dir`` keys off). Falls back to the config probe only when the adapter can't be
    read or carries no ``.language_model.`` LoRA keys (already-text-only / non-VL)."""
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


def assert_lora_applied(model, model_id: str) -> int:
    """After ``PeftModel.from_pretrained``, verify the adapter's LoRA actually loaded (non-empty)."""
    count = 0
    for name, _ in model.named_modules():
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
    """Assert a peft adapter load matched ALL saved keys — fail closed on a silent discard (#67)."""

    def _lora_only(keys):
        # Filter to LoRA keys, dropping the benign base-weight misses peft reports under strict=False.
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
    """Assert at least one ``lora_B`` weight is non-zero — the adapter is not an identity no-op."""
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

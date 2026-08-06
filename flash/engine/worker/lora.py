"""Pure LoRA-target / VL-checkpoint helpers for the fine-tuning worker.

These helpers take the model id as an ARGUMENT and read NONE of the worker's run-scoped
module globals, so they live here as a leaf module. ``flash.engine.worker`` re-exports
them; this module must NOT import that package (no cycle). Heavy deps (transformers, peft,
vllm, the catalog) are imported lazily inside the functions so the module stays
CPU-importable.
"""

from __future__ import annotations

# Natively-multimodal model types (Qwen3.5/3.6). Their LoRA adapters adapt the FULL module
# tree — vision tower / projector / MTP head included, like every other linear (on no-image
# data those get no gradient, so their lora_B stays zero-init). The engine loads and serves
# the whole VL model (vision tower included); there is no language-only VL adapter path.
_VL_MODEL_TYPES = ("qwen3_5", "qwen3_5_moe", "qwen3_6")


def is_vl_checkpoint(model_id: str, revision: str = "") -> bool:
    """True for natively-multimodal checkpoints (Qwen3.5/3.6) — routes VL warm-start handling."""
    try:
        from transformers import AutoConfig

        from flash.engine.worker.hf import model_revision_kwargs

        cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            **model_revision_kwargs(revision),
        )
        model_type = getattr(cfg, "model_type", "") or ""
    except Exception as e:
        print("is_vl_checkpoint: config probe failed:", e)
        return False
    return model_type in _VL_MODEL_TYPES


# --------------------------------------------------------------------------------------------
# Warm-start (init_from_adapter) SFT-adapter key namespace for VL checkpoints.
#
# SFT/GRPO/OPD all train VL checkpoints through the FULL multimodal model, so their adapter module
# sets match exactly and a warm-start CONTINUES the one LoRA in place (no merge, no rank-stack).
# ``_LANGUAGE_MODEL_INFIX`` is the signal ``adapter_is_vl_warmstart`` reads to detect a full-VL
# warm-start adapter from its keys and load the matching multimodal base.

_LANGUAGE_MODEL_INFIX = ".language_model."


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

    For safetensors, read ONLY the JSON header (pure stdlib, no tensor data). For PEFT
    ``adapter_model.bin``, use Torch's weights-only loader and inspect the state-dict keys. Returns
    ``None`` when no adapter weights exist in ``adir``.
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
        data_size = file_size - 8 - hdr_len
        for key, tensor_info in header.items():
            if key == "__metadata__":
                continue
            offsets = tensor_info.get("data_offsets") if isinstance(tensor_info, dict) else None
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets
                )
                or offsets[0] < 0
                or offsets[0] > offsets[1]
                or offsets[1] > data_size
            ):
                raise ValueError(f"{st_path}: invalid or truncated tensor data for {key!r}")
        return [k for k in header if k != "__metadata__"]
    bin_path = os.path.join(adir, "adapter_model.bin")
    if os.path.isfile(bin_path):
        import torch

        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(
                f"{bin_path}: expected a tensor state dict, got {type(state).__name__}"
            )
        bad = [
            k for k, v in state.items() if not isinstance(k, str) or not isinstance(v, torch.Tensor)
        ]
        if bad:
            raise ValueError(
                f"{bin_path}: contains non-tensor entries (e.g. {bad[:4]}); "
                "expected a plain PEFT adapter state dict"
            )
        return list(state)
    return None


def adapter_is_vl_warmstart(adir: str, model_id: str, revision: str = "") -> bool:
    """Whether a warm-start adapter must be continued on the FULL multimodal base (VL loader).

    Robust to a transient ``is_vl_checkpoint`` config-probe failure (it calls
    ``AutoConfig.from_pretrained`` and swallows EVERY exception to return False, so an HF
    rate-limit / network hiccup / uncached config could silently route a genuine VL warm-start onto
    the language-only loader — whose module names wouldn't match the adapter's ``.language_model.``
    keys, and whose trainer arch wouldn't match the VL vLLM rollout engine — issue #286). An adapter
    that actually carries ``.language_model.`` LoRA keys was saved against the full multimodal model and
    IS a VL warm-start regardless of the probe (the adapter's own keys are the authoritative signal).
    Falls back to the config probe only when the adapter can't be read or carries no
    ``.language_model.`` LoRA keys."""
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
    if revision:
        return is_vl_checkpoint(model_id, revision=revision)
    return is_vl_checkpoint(model_id)


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
        if name.endswith(("lora_A.default", "lora_B.default")):
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

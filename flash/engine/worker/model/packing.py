"""Architecture probes and tokenization helpers for the packed training path.

verl does the packing itself (``pad_mode: no_padding`` collates to one concatenated
``(1, total_nnz)`` row), so what remains here is deciding whether a checkpoint may take that path.
GDN hybrids (Qwen3.5/3.6) may only pack when the interpreter running the forward can reset the
linear-attention recurrence (fla's ``cu_seqlens``) and the causal conv (``seq_idx``) at example
boundaries -- without both, state carries across examples inside a packed micro-batch.
"""

from __future__ import annotations


def _text_config(cfg):
    """Return the decoder sub-config (multimodal checkpoints nest it under ``text_config``)."""
    return getattr(cfg, "text_config", None) or cfg


def _load_text_config(model_id: str, revision: str):
    """The checkpoint's decoder config. Raises when the config cannot be resolved."""
    from transformers import AutoConfig

    from flash.engine.worker.io.hf import model_revision_kwargs

    return _text_config(
        AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            **model_revision_kwargs(revision),
        )
    )


def probe_is_pure_attention(model_id: str, revision: str = "") -> bool:
    """``model_is_pure_attention`` without the swallow: a failed probe RAISES.

    Callers that freeze the answer into a digest must use this. Returning False for "the hub timed
    out" is fine for a runtime gate that only has to avoid packing, but it is a wrong ANSWER, and a
    wrong answer written into a profile is compared byte-for-byte against a later re-derivation
    that may have gotten the config just fine.
    """
    cfg = _load_text_config(model_id, revision)
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types:
        return all(t == "full_attention" for t in layer_types)
    for attr in ("linear_num_key_heads", "linear_key_head_dim", "linear_conv_kernel_dim"):
        if getattr(cfg, attr, None):
            return False
    # Qwen2.5 has sliding_window but disables it via use_sliding_window=False -> still packs.
    # Mistral-style has no use_sliding_window flag -> assume window is active.
    sliding = getattr(cfg, "sliding_window", None)
    return not (sliding and getattr(cfg, "use_sliding_window", True))


def model_is_pure_attention(model_id: str, revision: str = "") -> bool:
    """True when every decoder layer is full softmax attention (safe to pack unconditionally).

    Returns False for GDN hybrids, sliding-window arches, and on any config error.

    the profile needs this and the runtime gate does not: the runtime asks the narrower question
    "is this a gdn hybrid whose child can reset boundaries", which it answers on the worker with the
    checkpoint in hand. the profile runs BEFORE any worker exists and has to commit to a packing
    mode for the quote, so it can only fail closed -- pack when the architecture is provably safe
    for every backend, otherwise price the unpacked path.
    """
    try:
        return probe_is_pure_attention(model_id, revision)
    except Exception as e:  # network/parse/arch failure -> do NOT pack (boundary-safe default)
        print(f"[pack] pure-attention probe failed for {model_id!r} (treating as NOT pure): {e}")
        return False


def probe_is_gdn_hybrid(model_id: str, revision: str = "") -> bool:
    """``model_is_gdn_hybrid`` without the swallow: a failed probe RAISES.

    See ``probe_is_pure_attention`` for why a digest-freezing caller needs the distinction.
    """
    cfg = _load_text_config(model_id, revision)
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types and any(t == "linear_attention" for t in layer_types):
        return True
    return any(
        getattr(cfg, a, None)
        for a in ("linear_num_key_heads", "linear_key_head_dim", "linear_conv_kernel_dim")
    )


def model_is_gdn_hybrid(model_id: str, revision: str = "") -> bool:
    """True for a GatedDeltaNet hybrid (Qwen3.5/3.6) that needs cu_seqlens + seq_idx to pack."""
    try:
        return probe_is_gdn_hybrid(model_id, revision)
    except Exception as e:
        print(f"[pack] gdn-hybrid probe failed for {model_id!r} (treating as NOT gdn): {e}")
        return False


def gdn_model_type(model_id: str | None, revision: str = "") -> str:
    """The checkpoint's ``model_type``, naming the ``transformers.models.*`` module to inspect.

    Read from the checkpoint's config, so the answer is a property of the weights rather than of
    whichever transformers is installed -- which is what lets the parent resolve it on behalf of the
    verl child (see backend_common.gdn_probe_module).
    """
    if not model_id:
        return "qwen3_5"
    try:
        from transformers import AutoConfig

        from flash.engine.worker.io.hf import model_revision_kwargs

        cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            **model_revision_kwargs(revision),
        )
        return getattr(cfg, "model_type", None) or "qwen3_5"
    except Exception:
        return "qwen3_5"


def _gdn_forward_threads_reset_kwargs(model_id: str | None, revision: str = "") -> bool:
    """Check that THIS arch's GDN forward actually accepts cu_seq_lens_q and seq_idx (varies by transformers version)."""
    try:
        import importlib
        import inspect

        model_type = gdn_model_type(model_id, revision=revision)
        mod = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
        gdn_cls = next(
            (
                c
                for n, c in vars(mod).items()
                if isinstance(c, type) and n.endswith("GatedDeltaNet")
            ),
            None,
        )
        if gdn_cls is None:
            return False
        fwd = inspect.getsource(gdn_cls.forward)
        return ("cu_seq_lens_q" in fwd) and ("seq_idx" in fwd)
    except Exception:
        return False


def gdn_packing_contract_available(model_id: str | None = None, revision: str = "") -> bool:
    """True when the installed gdn stack exposes the boundary-reset contract without opening cuda."""
    try:
        import importlib

        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        if not (is_flash_linear_attention_available() and is_causal_conv1d_available()):
            return False
        importlib.import_module("causal_conv1d")
        return _gdn_forward_threads_reset_kwargs(model_id, revision=revision)
    except Exception:
        return False


def gdn_packing_available(model_id: str | None = None, revision: str = "") -> bool:
    """True when the gdn reset contract is installed and functional on the current device."""
    if not gdn_packing_contract_available(model_id, revision=revision):
        return False
    try:
        # causal_conv1d compiled without the current gpu arch imports fine but raises at first forward;
        # smoke it now so training fails before optimizer work rather than at a packed boundary.
        import torch

        if torch.cuda.is_available():
            from causal_conv1d import causal_conv1d_fn

            _x = torch.zeros(1, 4, 8, device="cuda", dtype=torch.bfloat16)
            conv_weight = torch.zeros(4, 3, device="cuda", dtype=torch.bfloat16)
            causal_conv1d_fn(_x, conv_weight)
            torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _eos_terminated(texts: list[str], tokenizer) -> list[str]:
    """Append EOS (once) to each text.

    EOS is appended because a packed row concatenates examples with no separator, so it is the only
    thing marking where one example ends -- without it the model learns to run past the end of an
    answer into the next one.
    """
    eos = tokenizer.eos_token or ""
    return [t if (eos and t.endswith(eos)) else t + eos for t in texts]


def tokenize_for_packing(texts: list[str], tokenizer, max_length: int) -> list[list[int]]:
    """Tokenize texts for packing: EOS appended, tokenizer's default add_special_tokens."""
    enc = tokenizer(_eos_terminated(texts, tokenizer), truncation=True, max_length=max_length)
    return enc["input_ids"]


def untruncated_lengths_for_packing(texts: list[str], tokenizer) -> list[int]:
    """Token count per text with NO cap, under the same EOS rule ``tokenize_for_packing`` applies.

    Sharing ``_eos_terminated`` is what makes the two measurements comparable. Encoding the raw text
    instead would drop the appended EOS, so every row that does not already end in one would measure
    a token SHORT of its own realized length -- and "untruncated < realized" is exactly the
    contradiction that makes a truncation count untrustworthy.
    """
    enc = tokenizer(_eos_terminated(texts, tokenizer), truncation=False)
    return [len(ids) for ids in enc["input_ids"]]


def completion_mask_from_ids(prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    """Return per-token mask: 0 over the shared prompt prefix, 1 over the completion.

    Uses the longest shared token prefix rather than len(prompt_ids) so the boundary is robust to
    the thinking template's pre-opened ``<think>\\n`` (which diverges from the full render by a token).
    Returns ``[]`` for empty full_ids; all-zero when truncation removed the entire completion.
    """
    n_full = len(full_ids)
    if n_full == 0:
        return []
    n = 0
    for a, b in zip(prompt_ids, full_ids, strict=False):  # different lengths by design (prefix)
        if a != b:
            break
        n += 1
    if n >= n_full:
        return [0] * n_full
    return [0] * n + [1] * (n_full - n)

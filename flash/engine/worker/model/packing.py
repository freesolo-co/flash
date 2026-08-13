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


def _load_text_config(model_id: str, revision: str, *, trust_remote_code: bool = True):
    """The checkpoint's decoder config. Raises when the config cannot be resolved."""
    from transformers import AutoConfig

    from flash.engine.worker.io.hf import model_revision_kwargs

    return _text_config(
        AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            **model_revision_kwargs(revision),
        )
    )


def probe_is_pure_attention(
    model_id: str, revision: str = "", *, trust_remote_code: bool = True
) -> bool:
    """Pure-attention probe that RAISES rather than swallowing a failed config read.

    Callers that freeze the answer into a digest must use this. Returning False for "the hub timed
    out" is fine for a runtime gate that only has to avoid packing, but it is a wrong ANSWER, and a
    wrong answer written into a profile is compared byte-for-byte against a later re-derivation
    that may have gotten the config just fine.
    """
    cfg = _load_text_config(model_id, revision, trust_remote_code=trust_remote_code)
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


def probe_is_gdn_hybrid(
    model_id: str, revision: str = "", *, trust_remote_code: bool = True
) -> bool:
    """``model_is_gdn_hybrid`` without the swallow: a failed probe RAISES.

    See ``probe_is_pure_attention`` for why a digest-freezing caller needs the distinction.
    """
    cfg = _load_text_config(model_id, revision, trust_remote_code=trust_remote_code)
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


def worker_image_packing_support(model_id: str, revision: str = "") -> tuple[str, bool]:
    """Return the packing contract supplied by the fixed worker image.

    the worker image pins transformers, fla, and causal_conv1d as one tested stack. the control plane
    deliberately does not install those gpu-only modules, so curated catalog architecture identifies
    the GDN contract without importing repository code or the worker stack. uncataloged models still
    use a data-only config probe. the worker verifies its installed modules and forward signature
    before executing a packed quote.
    """
    from flash.core.catalog import MODELS

    info = MODELS.get(model_id)
    if info is not None:
        if info.num_linear_attention_layers:
            return "gdn-hybrid", True
        if info.num_attention_layers:
            return "pure-attention", True
        return "unsupported", False
    if probe_is_pure_attention(model_id, revision=revision, trust_remote_code=False):
        return "pure-attention", True
    if probe_is_gdn_hybrid(model_id, revision=revision, trust_remote_code=False):
        return "gdn-hybrid", True
    return "unsupported", False


def gdn_packing_contract_available(model_id: str | None = None, revision: str = "") -> bool:
    """True when the installed gdn stack exposes the boundary-reset contract without opening cuda.

    device-independent by construction because the gpu worker must verify capability without opening
    cuda before it accepts a packed quote. a gate that consults ``torch.cuda`` could change the answer
    across worker lifecycle stages.

    this asks whether the installed transformers threads the reset kwargs into this arch's GDN
    forward and whether both kernels are importable. ``is_*_available()`` is deliberately not used
    because those open with ``is_torch_cuda_available()``.
    """
    try:
        import importlib

        importlib.import_module("fla")
        importlib.import_module("causal_conv1d")
        return _gdn_forward_threads_reset_kwargs(model_id, revision=revision)
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


def _chatml_message_spans(full_ids: list[int], tokenizer) -> list[tuple[int, int, str]] | None:
    """Split a rendered transcript into ``(start, end, role)`` per ChatML message.

    Reads the ONE full render rather than re-rendering message prefixes. Re-rendering cannot be
    trusted here: Qwen3.5/3.6 pick each assistant turn's ``<think>`` layout from ``last_query_index``,
    computed over the WHOLE message list, so ``render(messages[:k])`` is not a prefix of
    ``render(messages)`` -- a shorter list re-renders earlier turns with a different tag layout and
    the derived offsets slide off the real turn boundaries.

    Returns ``None`` when the render is not ChatML (no ``<|im_start|>``/``<|im_end|>`` pair, or no
    message parsed), which is the signal for the caller to leave the mask untouched rather than
    guess. Roles are lowercased; a header the tokenizer splits across several tokens is joined
    before comparison so a multi-token role name still resolves.
    """
    to_id = getattr(tokenizer, "convert_tokens_to_ids", None)
    decode = getattr(tokenizer, "decode", None)
    if not callable(to_id) or not callable(decode):
        return None
    im_start = to_id("<|im_start|>")
    im_end = to_id("<|im_end|>")
    unk = getattr(tokenizer, "unk_token_id", None)
    if im_start is None or im_end is None or im_start == unk or im_end == unk:
        return None
    if im_start not in full_ids or im_end not in full_ids:
        return None

    spans: list[tuple[int, int, str]] = []
    index = 0
    total = len(full_ids)
    while index < total:
        if full_ids[index] != im_start:
            index += 1
            continue
        # the header runs from just after <|im_start|> to the newline that ends the role line.
        cursor = index + 1
        role_ids: list[int] = []
        while cursor < total and full_ids[cursor] not in (im_start, im_end):
            piece = tokenizer.decode([full_ids[cursor]])
            if "\n" in piece:
                break
            role_ids.append(full_ids[cursor])
            cursor += 1
        role = tokenizer.decode(role_ids).strip().lower() if role_ids else ""
        # the message body ends at its <|im_end|>; an unterminated final message runs to the end.
        cursor = index + 1
        while cursor < total and full_ids[cursor] != im_end:
            cursor += 1
        end = min(cursor + 1, total)
        # the template writes a newline after <|im_end|>; it belongs to the message it closes.
        if end < total and tokenizer.decode([full_ids[end]]) == "\n":
            end += 1
        spans.append((index, end, role))
        index = end
    return spans or None


def assistant_only_mask(loss_mask: list[int], full_ids: list[int], tokenizer) -> list[int]:
    """Clear supervision over every non-assistant span of a rendered transcript.

    ``completion_mask_from_ids`` returns ONE contiguous supervised span, which is right for a
    prompt -> completion row but wrong for a multi-turn target: everything after the prompt is
    supervised, so the interleaved environment/tool/user observations between assistant turns train
    the model to emit the environment's replies.

    Strictly subtractive: it only ever turns a 1 into a 0, so the prompt boundary, the truncation
    behaviour, and the pre-opened ``<think>\\n`` handling all stay exactly as they were. A row whose
    target is a single assistant turn is returned unchanged. When the transcript does not parse as
    ChatML the mask is returned unchanged -- narrowing supervision on a guess would silently drop
    real training signal.
    """
    if not full_ids or not any(loss_mask):
        return loss_mask
    spans = _chatml_message_spans(full_ids, tokenizer)
    if spans is None:
        return loss_mask
    masked = list(loss_mask)
    for start, end, role in spans:
        if role == "assistant":
            continue
        for position in range(start, min(end, len(masked))):
            masked[position] = 0
    return masked

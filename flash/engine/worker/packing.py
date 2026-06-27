"""True token packing with a block-diagonal SDPA attention mask.

Concatenate short SFT examples into ``max_length`` blocks and feed the trainer a 4D
**block-diagonal causal** attention mask so packed examples never attend across their
boundaries. Crucially this is boundary-correct under PLAIN SDPA — it needs neither
``flash_attn`` (no prebuilt wheel for torch 2.10 / no sm120 kernel) nor ``flex_attention``
(unsupported on the Qwen3.5/3.6 arch). It is exactly what lets packing run on flash's DEFAULT
RTX 5090 (sm120), where the FA2/FA3 varlen path the worker otherwise relies on is unavailable,
and on any arch whose flash-attn build did not land.

Why packing is a win: instruction targets are far shorter than ``max_seq_len``, so an unpacked
batch spends most of its FLOPs on padding. Concatenating examples into full blocks removes that
waste (PR #174 measured 4.4-10.7x on the FA2 path; the SDPA-mask path keeps the same packing win
minus the block-sparse-attention speedup FA2 varlen gives, so ~1.5-2x in practice). The dense
[T,T] mask is O(T^2) memory, but attention is a small fraction of total FLOPs for these models,
so the masked-attention overhead is dwarfed by the pad-removal win.

GATING — pure full-attention only. A 4D mask isolates examples only in layers that READ the
attention mask. Hybrid GatedDeltaNet models (Qwen3.5/3.6) interleave linear-attention layers
whose recurrence + short causal conv1d carry state ACROSS example boundaries regardless of any
mask — their boundaries reset only via the ``fla`` kernel's ``cu_seq_lens_q/k`` and
``causal_conv1d``'s ``seq_idx``. So a pure full-attention arch (``model_is_pure_attention``) packs with the 4D mask
alone, while a GDN hybrid ALSO needs the varlen path: ``BlockDiagonalCollator(emit_varlen=True)``
emits ``cu_seq_lens_q/k`` + ``seq_idx``, gated on both kernels being importable + arch-correct
(``gdn_packing_available`` + ``model_is_gdn_hybrid``). Without those kernels the hybrid tier stays
unpacked.

This is a leaf module: torch is imported lazily inside the collator so it stays CPU-importable
(the arch probe needs only ``transformers.AutoConfig``). ``flash.engine.worker`` re-exports the
public names.
"""

from __future__ import annotations

from dataclasses import dataclass


def _text_config(cfg):
    """The decoder/LM sub-config. Multimodal checkpoints (Qwen3.5-VL) keep the LM dims under
    ``text_config``; read it when present so the layer-type probe sees the real decoder."""
    return getattr(cfg, "text_config", None) or cfg


def model_is_pure_attention(model_id: str) -> bool:
    """True only when EVERY decoder layer is full softmax attention, so a 4D block-diagonal mask
    fully isolates packed examples under SDPA. Config-only probe (no weights, no CUDA). Returns
    safe-False on any error or on a hybrid / linear-attention / sliding-window arch.

    Excluded (return False):
      * GatedDeltaNet hybrids (Qwen3.5/3.6): ``layer_types`` contains ``"linear_attention"`` (their
        recurrence/conv cross boundaries a mask can't reset), or the config declares linear-attn
        dims directly.
      * Sliding-window models (e.g. Gemma): a layer typed ``"sliding_attention"`` applies a window
        the model builds itself — passing a pre-built 4D mask BYPASSES that window (wrong
        semantics), so exclude them too. Only ``"full_attention"`` everywhere is safe.

    Included (return True): standard dense decoders (Llama/MiniCPM5, Qwen2/Qwen3) that expose no
    per-layer ``layer_types`` and no linear-attn dims — every layer reads the mask.
    """
    try:
        from transformers import AutoConfig

        cfg = _text_config(AutoConfig.from_pretrained(model_id, trust_remote_code=True))
        layer_types = getattr(cfg, "layer_types", None)
        if layer_types:
            return all(t == "full_attention" for t in layer_types)
        # No per-layer types: still exclude anything that advertises a linear-attention (DeltaNet)
        # block via its dims — a hybrid arch can omit layer_types but always sets these.
        for attr in ("linear_num_key_heads", "linear_key_head_dim", "linear_conv_kernel_dim"):
            if getattr(cfg, attr, None):
                return False
        # A GLOBALLY sliding-window model (no per-layer layer_types, e.g. Mistral / Qwen2 configs)
        # builds its own LOCAL-attention causal mask; a pre-built full block-diagonal mask would
        # BYPASS the window and train with global attention instead of the checkpoint's intended
        # local attention. Exclude when a window is configured AND active: honor use_sliding_window
        # when the config exposes it (Qwen2.5 ships a sliding_window value but DISABLES it via
        # use_sliding_window=False -> still packs), else assume a configured window is active
        # (Mistral-style configs have no such flag).
        sliding = getattr(cfg, "sliding_window", None)
        return not (sliding and getattr(cfg, "use_sliding_window", True))
    except Exception as e:  # network/parse/arch failure -> do NOT pack (boundary-safe default)
        print(f"[pack] pure-attention probe failed for {model_id!r} (treating as NOT pure): {e}")
        return False


def model_is_gdn_hybrid(model_id: str) -> bool:
    """True for a GatedDeltaNet *hybrid* (Qwen3.5/3.6): the config interleaves ``"linear_attention"``
    layers with full attention. These need the varlen GDN path (cu_seqlens + seq_idx) to pack
    boundary-correctly — a 4D mask alone can't reset their recurrent/conv state. Distinct from the
    sliding-window case (also non-pure, but NOT packable this way). Config-only; safe-False on error.
    """
    try:
        from transformers import AutoConfig

        cfg = _text_config(AutoConfig.from_pretrained(model_id, trust_remote_code=True))
        layer_types = getattr(cfg, "layer_types", None)
        if layer_types and any(t == "linear_attention" for t in layer_types):
            return True
        # No layer_types but linear-attn dims declared -> still a GDN hybrid.
        return any(
            getattr(cfg, a, None)
            for a in ("linear_num_key_heads", "linear_key_head_dim", "linear_conv_kernel_dim")
        )
    except Exception as e:
        print(f"[pack] gdn-hybrid probe failed for {model_id!r} (treating as NOT gdn): {e}")
        return False


def _gdn_forward_threads_reset_kwargs(model_id: str | None) -> bool:
    """Does THIS model's GatedDeltaNet forward actually thread cu_seq_lens_q AND seq_idx? Different GDN
    families live in different modeling modules (qwen3_5 -> modeling_qwen3_5.Qwen3_5GatedDeltaNet, a
    future qwen3_6 -> modeling_qwen3_6.Qwen3_6GatedDeltaNet), so resolve the ACTUAL arch from the
    model's config and probe ITS DeltaNet class — a hardcoded qwen3_5 probe would wrongly pass for an
    arch that drops the kwargs (or whose layer hard-codes seq_idx=None on an older transformers).
    Falls back to qwen3_5 when no model_id is given. Safe-False on any failure."""
    try:
        import importlib
        import inspect

        model_type = "qwen3_5"
        if model_id:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model_type = getattr(cfg, "model_type", None) or model_type
        mod = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
        gdn_cls = next(
            (c for n, c in vars(mod).items()
             if isinstance(c, type) and n.endswith("GatedDeltaNet")),
            None,
        )
        if gdn_cls is None:
            return False
        fwd = inspect.getsource(gdn_cls.forward)
        return ("cu_seq_lens_q" in fwd) and ("seq_idx" in fwd)
    except Exception:
        return False


def gdn_packing_available(model_id: str | None = None) -> bool:
    """True only when BOTH varlen kernels a GatedDeltaNet hybrid needs to pack boundary-correctly are
    importable: ``flash-linear-attention`` (resets the DeltaNet recurrence via ``cu_seq_lens_q/k`` — the
    pure-torch fallback IGNORES it) AND ``causal_conv1d`` (resets the short causal conv via
    ``seq_idx``). Without both, a packed GDN run would cross-contaminate across example boundaries,
    so packing must stay off. GPU-validated (RTX 5090, Qwen3.5-0.8B): with both present, a packed
    example's outputs are byte-identical regardless of its neighbors' content (zero information
    leakage); the only difference vs unpacked is benign bf16 kernel-tiling numerics (~0.3 on logits,
    the same order as flash-attn-vs-SDPA drift).

    Two guards beyond the find_spec probes: (a) REALLY import causal_conv1d — its availability check
    is find_spec-based, so a built-but-broken wheel (ABI/symbol mismatch) would pass it and then crash
    at model load; (b) verify the INSTALLED Qwen3.5 DeltaNet forward actually threads cu_seq_lens_q AND
    seq_idx — transformers 5.6-5.8 hard-coded seq_idx=None / dropped cu_seq_lens_q, so on those builds
    the collator's reset kwargs are silently ignored and packed examples would still leak. Either
    guard failing -> packing stays off (the model trains unpacked, safely)."""
    try:
        import importlib

        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        if not (is_flash_linear_attention_available() and is_causal_conv1d_available()):
            return False
        importlib.import_module("causal_conv1d")  # (a) fail a built-but-broken wheel here, not at load
        if not _gdn_forward_threads_reset_kwargs(model_id):  # (b) version/API gate, per ACTUAL arch
            return False
        # (c) RUN the conv kernel on the LIVE GPU: a causal_conv1d wheel compiled WITHOUT this device's
        # arch imports fine but raises "CUDA error: no kernel image is available for execution on the
        # device" at the FIRST forward — which would crash the run mid-train. A tiny conv here surfaces
        # that now so packing stays off and the model trains unpacked instead. (fla's kernels are
        # Triton-JIT — always compiled for the present arch — so they need no such smoke.)
        import torch

        if torch.cuda.is_available():
            from causal_conv1d import causal_conv1d_fn

            _x = torch.zeros(1, 4, 8, device="cuda", dtype=torch.bfloat16)
            _w = torch.zeros(4, 3, device="cuda", dtype=torch.bfloat16)
            causal_conv1d_fn(_x, _w)
            torch.cuda.synchronize()
        return True
    except Exception:
        return False


def pack_token_ids(
    sequences: list[list[int]],
    max_length: int,
    completion_masks: list[list[int]] | None = None,
) -> list[dict]:
    """Greedily bin-pack tokenized examples into blocks of at most ``max_length`` tokens WITHOUT
    splitting an example (first-fit-decreasing, like TRL's ``bfd``: tighter blocks = less padding).

    An example longer than ``max_length`` is truncated to a single full-length block (matches the
    unpacked trainer's right-truncation). Empty sequences are dropped. Returns rows shaped
    ``{"input_ids": [...], "seq_lengths": [l1, l2, ...]}`` where ``sum(seq_lengths) == len(input_ids)``
    — the collator turns ``seq_lengths`` into the block-diagonal mask + per-example position_ids.

    ``completion_masks`` (optional, parallel to ``sequences``): per-token completion flags (1 = a
    completion token trained on, 0 = a prompt token masked from the loss). When provided, each packed
    row additionally carries a ``"completion_mask"`` aligned with its ``"input_ids"`` so completion-
    only SFT loss survives packing (the collator turns it into per-token label masking). Each mask is
    truncated to ``max_length`` in lockstep with its sequence, so the per-example invariant holds.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if completion_masks is not None and len(completion_masks) != len(sequences):
        raise ValueError(
            f"completion_masks must be parallel to sequences: {len(completion_masks)} != {len(sequences)}"
        )
    # Keep each sequence with its completion mask (if any) so truncation + the FFD reorder can't
    # desync them. Drop empty sequences (and their masks) exactly as the no-mask path does.
    if completion_masks is None:
        items = [(s[:max_length], None) for s in sequences if s]
    else:
        # Each SURVIVING (non-empty) mask must align 1:1 with its (pre-truncation) sequence — otherwise
        # the lockstep ``m[:max_length]`` truncation below would leave a packed row whose
        # completion_mask is misaligned with its input_ids, silently masking (or training on) the WRONG
        # tokens. Validate up front so a caller bug fails loud here instead of corrupting the loss
        # target. Empty sequences are dropped below regardless, so their mask length is irrelevant.
        for s, m in zip(sequences, completion_masks, strict=True):
            if s and len(m) != len(s):
                raise ValueError(
                    f"each completion_mask must match its sequence length: {len(m)} != {len(s)}"
                )
        items = [
            (s[:max_length], m[:max_length])
            for s, m in zip(sequences, completion_masks, strict=True)
            if s
        ]
    # First-fit-decreasing: place the longest examples first so the small ones fill the gaps.
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]), reverse=True)
    bins: list[dict] = []  # each: {"input_ids": [...], "seq_lengths": [...], "completion_mask"?, "free": int}
    for i in order:
        s, m = items[i]
        need = len(s)
        for b in bins:
            if b["free"] >= need:
                b["input_ids"].extend(s)
                b["seq_lengths"].append(need)
                if m is not None:
                    b["completion_mask"].extend(m)
                b["free"] -= need
                break
        else:  # no open bin fits -> start a new one
            nb = {"input_ids": list(s), "seq_lengths": [need], "free": max_length - need}
            if m is not None:
                nb["completion_mask"] = list(m)
            bins.append(nb)
    rows: list[dict] = []
    for b in bins:
        row = {"input_ids": b["input_ids"], "seq_lengths": b["seq_lengths"]}
        if "completion_mask" in b:
            row["completion_mask"] = b["completion_mask"]
        rows.append(row)
    return rows


def packing_efficiency(rows: list[dict], max_length: int) -> float:
    """Fraction of block capacity filled with real tokens (1.0 = no padding). Diagnostic only."""
    if not rows or max_length <= 0:
        return 0.0
    real = sum(sum(r["seq_lengths"]) for r in rows)
    return real / (len(rows) * max_length)


def tokenize_for_packing(texts: list[str], tokenizer, max_length: int) -> list[list[int]]:
    """Tokenize chat-templated ``text`` rows for packing, MATCHING TRL's non-packed SFT prep EXACTLY
    so a packed run trains on the SAME token sequences as the unpacked/FA2 path (no quality drift):
      * append the EOS token to any row that doesn't already end with it — TRL's add_eos step does
        this for the language-modeling ``text`` case, and skipping it would stop teaching the model
        the final stop token (it'd never learn to halt);
      * tokenize with the tokenizer's DEFAULT add_special_tokens — TRL's ``_tokenize`` for a non-
        conversational ``text`` field calls ``processing_class(text=input)`` with no override, so for
        Llama-family tokenizers (e.g. the MiniCPM pure-attention tier) it prepends BOS. Forcing
        add_special_tokens=False here would drop that BOS and diverge from the unpacked path. (Qwen
        tokenizers have no BOS, so the Qwen3.x / GDN tier is unaffected either way.)
      * truncate to ``max_length`` (same cap pack_token_ids would apply) so a pathological long row
        never materializes a huge id list; batched (one call) for speed.
    """
    eos = tokenizer.eos_token or ""
    rows = [t if (eos and t.endswith(eos)) else t + eos for t in texts]
    enc = tokenizer(rows, truncation=True, max_length=max_length)  # default add_special_tokens (TRL parity)
    return enc["input_ids"]


def build_completion_mask(
    prompt_text: str, full_ids: list[int], tokenizer, max_length: int
) -> list[int]:
    """Token-level completion mask for completion-only SFT loss: ``0`` for the prompt tokens, ``1``
    for the completion (the assistant turn(s) the model must learn to generate). ``full_ids`` are the
    example's tokens from :func:`tokenize_for_packing` (the SAME tokens the trainer sees);
    ``prompt_text`` is the chat-templated prompt rendered with ``add_generation_prompt=True``.

    We tokenize the prompt the SAME way :func:`tokenize_for_packing` tokenizes the full row (default
    ``add_special_tokens``, truncate to ``max_length``; NO appended EOS — the prompt never ends a
    turn) and mask the LONGEST SHARED TOKEN PREFIX of the prompt and the full row. The shared-prefix
    (rather than ``len(prompt_ids)``) is what makes this robust to the thinking chat template, whose
    ``add_generation_prompt=True`` render pre-opens ``<think>\\n`` so the prompt diverges from the
    full render by a token — we mask up to that divergence and train on everything after. This mirrors
    TRL's own prompt-completion masking (it likewise derives the boundary from the prompt token
    length), but in flash's single pre-tokenization pass so the unpacked and packed paths share one
    boundary.

    Returns an ALL-ZERO mask (the whole row is prompt) in the degenerate case where ``max_length``
    truncation removed the entire completion — the full row's tokens are a prefix of, or equal to, the
    prompt. That row carries no loss; :func:`run_sft <flash.engine.worker.sft.run_sft>` filters these
    no-completion-target rows out before training, so a fully-masked (NaN) micro-batch/packed block
    can't form. (Returns ``[]`` for an empty ``full_ids``.)
    """
    n_full = len(full_ids)
    if n_full == 0:
        return []
    # List form + [0] mirrors tokenize_for_packing's call EXACTLY (default add_special_tokens, same
    # truncation), so the prompt tokens line up with the full row's prefix token-for-token.
    prompt_ids = tokenizer([prompt_text], truncation=True, max_length=max_length)["input_ids"][0]
    n = 0
    for a, b in zip(prompt_ids, full_ids, strict=False):  # different lengths by design (prefix)
        if a != b:
            break
        n += 1
    if n >= n_full:
        # max_length truncation removed the ENTIRE completion: the full row is all prompt (its tokens
        # are a prefix of, or equal to, the prompt). Mask the WHOLE row — a row that contributes no
        # loss is correct here. The old ``min(n, n_full - 1)`` clamp instead forced the last PROMPT
        # token to be a trainable "completion" target, training the model to reproduce prompt text for
        # long-prompt examples where nothing survived. A packed bin still trains on its OTHER examples;
        # an unpacked all-prompt row simply becomes a no-op (all labels -100).
        return [0] * n_full
    # A real completion survived truncation: mask the shared prompt prefix, train on the rest (the
    # ``n < n_full`` guarantees at least one completion token).
    return [0] * n + [1] * (n_full - n)


# Process-local cache of the lower-triangular causal matrix: the collator runs on every batch, and
# torch.tril(torch.ones(T, T)) is a non-trivial CPU alloc at T=2048+. Keep the LARGEST one seen and
# slice it for smaller T (it's read-only). Dataloader workers are separate processes, so each holds
# its own copy — no cross-thread race.
_CAUSAL_TRIL: dict = {}


def _causal_lower_triangular(total: int, torch):
    cached = _CAUSAL_TRIL.get("m")
    if cached is None or cached.shape[0] < total:
        cached = torch.tril(torch.ones(total, total, dtype=torch.bool))
        _CAUSAL_TRIL["m"] = cached
    return cached[:total, :total]


@dataclass
class BlockDiagonalCollator:
    """Collate pre-packed rows (from :func:`pack_token_ids`) into a batch whose 4D **block-diagonal
    causal** attention mask keeps packed examples from attending across their boundaries under
    PLAIN SDPA — no flash-attn, no flex_attention.

    Emits per batch:
      * ``input_ids``      ``[B, T]`` (right-padded with ``pad_token_id``)
      * ``attention_mask`` ``[B, 1, T, T]`` BOOL — ``True`` = query may attend key. Block-diagonal
        (same example) AND causal (key <= query). A bool mask is dtype-agnostic, so it composes
        with bf16/fp16 runs without an ``-inf`` dtype mismatch. Pad tokens form their own segment
        so no query row is all-False (which would NaN the softmax); pad rows never contribute to
        loss (their labels are -100) and real tokens never attend pad keys.
      * ``position_ids``   ``[B, T]`` reset to 0 at each example start (RoPE per example)
      * ``labels``         ``[B, T]`` = ``input_ids`` for real tokens, with each example's FIRST
        token set to -100 (so the cross-boundary next-token pair is never scored — matches the
        unpacked trainer, whose first token is also never a target after HF's internal shift) and
        pad set to -100.

    ``pad_to_multiple_of`` rounds T up (tensor-core friendliness); the extra positions are pad.

    ``emit_varlen`` (GatedDeltaNet hybrids, e.g. Qwen3.5/3.6): additionally emit ``cu_seq_lens_q/k``
    (resets the DeltaNet recurrence per example in the fla kernel) and ``seq_idx`` (resets the causal
    conv in causal_conv1d) so the LINEAR-attention layers are boundary-correct too — the 4D mask only
    fixes the full-attention layers. This path requires ``per_device_train_batch_size == 1`` (one
    packed block per step; cu_seqlens spans that block) and does NOT pad (cu_seqlens must cover the
    whole row), so set ``pad_to_multiple_of`` irrelevant here.
    """

    pad_token_id: int
    label_pad_token_id: int = -100
    pad_to_multiple_of: int = 8
    emit_varlen: bool = False

    def __call__(self, features: list[dict]) -> dict:
        import torch

        rows = [list(f["input_ids"]) for f in features]
        seglens = [list(f["seq_lengths"]) for f in features]
        bsz = len(rows)
        if self.emit_varlen and bsz != 1:
            raise ValueError("emit_varlen packing requires per_device_train_batch_size == 1")
        # Fail fast on a broken row rather than silently mis-tag tokens as pad (or vice versa): the
        # whole mask/labels/cu_seqlens construction assumes sum(seq_lengths) == len(input_ids).
        for ids, lens in zip(rows, seglens, strict=True):
            if sum(lens) != len(ids):
                raise ValueError(
                    f"packed row invariant broken: sum(seq_lengths)={sum(lens)} != "
                    f"len(input_ids)={len(ids)} (rows must come from pack_token_ids)"
                )
        longest = max((len(r) for r in rows), default=0)
        m = self.pad_to_multiple_of
        # No padding on the varlen path: cu_seqlens must cover the whole sequence (a trailing pad
        # region not spanned by cu_seqlens would break the fla varlen kernel).
        total = longest if self.emit_varlen else (((longest + m - 1) // m) * m if m and m > 1 else longest)
        total = max(total, 1)

        input_ids = torch.full((bsz, total), self.pad_token_id, dtype=torch.long)
        position_ids = torch.zeros((bsz, total), dtype=torch.long)
        # segment id per token: 0..k-1 for the k examples in the block, -1 for trailing pad.
        seg = torch.full((bsz, total), -1, dtype=torch.long)

        for b, (ids, lens) in enumerate(zip(rows, seglens, strict=True)):
            n = len(ids)
            input_ids[b, :n] = torch.tensor(ids, dtype=torch.long)
            start = 0
            for ex_idx, length in enumerate(lens):
                end = start + length
                position_ids[b, start:end] = torch.arange(length)
                seg[b, start:end] = ex_idx
                start = end

        # Block-diagonal causal mask, fully vectorized:
        #   same-example: seg[q] == seg[k]   (pad shares segment -1, so pad rows attend pad -> no
        #                 all-False row; real tokens never attend pad because real seg != -1)
        #   causal:       k <= q
        same = seg.unsqueeze(2) == seg.unsqueeze(1)  # [B, T, T]
        causal = _causal_lower_triangular(total, torch)  # cached + sliced (not rebuilt per batch)
        attention_mask = (same & causal).unsqueeze(1)  # [B, 1, T, T]

        # Labels: real tokens predict their own continuation; first token of each example (and all
        # pad) -> ignore. position_ids == 0 marks exactly each example's first token (pad is 0 too,
        # and pad is already excluded below), so the boundary next-token pair is never scored.
        labels = input_ids.clone()
        labels[seg < 0] = self.label_pad_token_id
        labels[position_ids == 0] = self.label_pad_token_id

        # Completion-only loss under packing: when the packed rows carry per-token completion masks
        # (1 = completion token trained on, 0 = prompt token), additionally ignore every prompt token
        # so the loss matches the unpacked completion_only_loss path. Each mask spans only its row's
        # REAL tokens (sum(seq_lengths)); trailing pad keeps its already-ignored label. Prompt-start
        # tokens are mask 0 here AND position_ids == 0 above, so the two masks agree at boundaries.
        if any("completion_mask" in f for f in features):
            keep = torch.zeros((bsz, total), dtype=torch.bool)  # True == contributes to the loss
            for b, f in enumerate(features):
                cm = f.get("completion_mask")
                if cm is None:
                    # A row WITHOUT a completion_mask in a mixed batch keeps ALL its tokens (full-
                    # transcript loss), rather than being silently zeroed out. Pad/boundary tokens were
                    # already set to -100 above and keep=True can't un-mask them (``labels[~keep]`` only
                    # ADDS masking), so this just avoids dropping the loss for an unmasked row. An
                    # explicit None is the ONLY "no mask" signal — an empty list on a real row is a bug
                    # (caught by the length check below), not a silent "keep all".
                    keep[b, :] = True
                    continue
                # A present mask must align 1:1 with the row's REAL (pre-pad) tokens: the mask spans
                # sum(seq_lengths) == len(input_ids) (the invariant asserted above). Silently slicing a
                # mis-sized mask would shift the prompt/completion boundary and train on the wrong
                # positions, so fail loud instead of guessing.
                n_real = len(rows[b])
                if len(cm) != n_real:
                    raise ValueError(
                        f"completion_mask length {len(cm)} != row real-token count {n_real} "
                        "(mask must span sum(seq_lengths) == len(input_ids))"
                    )
                keep[b, :n_real] = torch.tensor([bool(x) for x in cm], dtype=torch.bool)
            labels[~keep] = self.label_pad_token_id

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
        }
        if self.emit_varlen:
            # bsz == 1 (asserted above): cu_seqlens covers this one block's examples, and seq_idx is
            # the per-token segment id (no pad on this path, so seg has no -1). These reach the
            # linear-attention layers via model(**batch) -> the fla chunk kernel (cu_seq_lens_q) and
            # causal_conv1d (seq_idx), resetting their state at each example boundary.
            lens = seglens[0]
            cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
            cu[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
            batch["cu_seq_lens_q"] = cu
            batch["cu_seq_lens_k"] = cu
            batch["max_length_q"] = int(max(lens))
            batch["max_length_k"] = int(max(lens))
            batch["seq_idx"] = seg.to(torch.int32)  # [1, T], non-negative (no pad on this path)
        return batch

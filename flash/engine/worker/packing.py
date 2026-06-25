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
mask — their boundaries reset only via the ``fla`` kernel's ``cu_seqlens`` and ``causal_conv1d``'s
``seq_idx``, which a mask cannot supply. So this module is gated to architectures whose every
layer is full softmax attention (``model_is_pure_attention``); the GDN-hybrid tier stays unpacked
(same as today), pending the separate cu_seqlens/seq_idx packing path.

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
        return True
    except Exception as e:  # network/parse/arch failure -> do NOT pack (boundary-safe default)
        print(f"[pack] pure-attention probe failed for {model_id!r} (treating as NOT pure): {e}")
        return False


def pack_token_ids(sequences: list[list[int]], max_length: int) -> list[dict]:
    """Greedily bin-pack tokenized examples into blocks of at most ``max_length`` tokens WITHOUT
    splitting an example (first-fit-decreasing, like TRL's ``bfd``: tighter blocks = less padding).

    An example longer than ``max_length`` is truncated to a single full-length block (matches the
    unpacked trainer's right-truncation). Empty sequences are dropped. Returns rows shaped
    ``{"input_ids": [...], "seq_lengths": [l1, l2, ...]}`` where ``sum(seq_lengths) == len(input_ids)``
    — the collator turns ``seq_lengths`` into the block-diagonal mask + per-example position_ids.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    seqs = [s[:max_length] for s in sequences if s]
    # First-fit-decreasing: place the longest examples first so the small ones fill the gaps.
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    bins: list[dict] = []  # each: {"input_ids": [...], "seq_lengths": [...], "free": int}
    for i in order:
        s = seqs[i]
        need = len(s)
        for b in bins:
            if b["free"] >= need:
                b["input_ids"].extend(s)
                b["seq_lengths"].append(need)
                b["free"] -= need
                break
        else:  # no open bin fits -> start a new one
            bins.append({"input_ids": list(s), "seq_lengths": [need], "free": max_length - need})
    return [{"input_ids": b["input_ids"], "seq_lengths": b["seq_lengths"]} for b in bins]


def packing_efficiency(rows: list[dict], max_length: int) -> float:
    """Fraction of block capacity filled with real tokens (1.0 = no padding). Diagnostic only."""
    if not rows:
        return 0.0
    real = sum(sum(r["seq_lengths"]) for r in rows)
    return real / (len(rows) * max_length)


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
    """

    pad_token_id: int
    label_pad_token_id: int = -100
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict:
        import torch

        rows = [list(f["input_ids"]) for f in features]
        seglens = [list(f["seq_lengths"]) for f in features]
        bsz = len(rows)
        longest = max((len(r) for r in rows), default=0)
        m = self.pad_to_multiple_of
        total = ((longest + m - 1) // m) * m if m and m > 1 else longest
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
        causal = torch.tril(torch.ones(total, total, dtype=torch.bool))
        attention_mask = (same & causal).unsqueeze(1)  # [B, 1, T, T]

        # Labels: real tokens predict their own continuation; first token of each example (and all
        # pad) -> ignore. position_ids == 0 marks exactly each example's first token (pad is 0 too,
        # and pad is already excluded below), so the boundary next-token pair is never scored.
        labels = input_ids.clone()
        labels[seg < 0] = self.label_pad_token_id
        labels[position_ids == 0] = self.label_pad_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
        }

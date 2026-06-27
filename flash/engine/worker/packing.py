"""True token packing with a block-diagonal SDPA attention mask.

Works under plain SDPA (no flash-attn required), so it runs on sm120/RTX 5090.
GDN hybrids (Qwen3.5/3.6) additionally need ``emit_varlen=True`` to reset linear-attention
recurrence and causal conv across example boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass


def _text_config(cfg):
    """Return the decoder sub-config (multimodal checkpoints nest it under ``text_config``)."""
    return getattr(cfg, "text_config", None) or cfg


def model_is_pure_attention(model_id: str) -> bool:
    """True when every decoder layer is full softmax attention (safe for 4D block-diagonal mask).
    Returns False for GDN hybrids, sliding-window arches, and on any config error.
    """
    try:
        from transformers import AutoConfig

        cfg = _text_config(AutoConfig.from_pretrained(model_id, trust_remote_code=True))
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
    except Exception as e:  # network/parse/arch failure -> do NOT pack (boundary-safe default)
        print(f"[pack] pure-attention probe failed for {model_id!r} (treating as NOT pure): {e}")
        return False


def model_is_gdn_hybrid(model_id: str) -> bool:
    """True for a GatedDeltaNet hybrid (Qwen3.5/3.6) that needs cu_seqlens + seq_idx to pack."""
    try:
        from transformers import AutoConfig

        cfg = _text_config(AutoConfig.from_pretrained(model_id, trust_remote_code=True))
        layer_types = getattr(cfg, "layer_types", None)
        if layer_types and any(t == "linear_attention" for t in layer_types):
            return True
        return any(
            getattr(cfg, a, None)
            for a in ("linear_num_key_heads", "linear_key_head_dim", "linear_conv_kernel_dim")
        )
    except Exception as e:
        print(f"[pack] gdn-hybrid probe failed for {model_id!r} (treating as NOT gdn): {e}")
        return False


def _gdn_forward_threads_reset_kwargs(model_id: str | None) -> bool:
    """Check that THIS arch's GDN forward actually accepts cu_seq_lens_q and seq_idx (varies by transformers version)."""
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
    """True when flash-linear-attention and causal_conv1d are both present, functional, and the
    GDN forward actually threads cu_seq_lens_q + seq_idx (varies by transformers version).
    """
    try:
        import importlib

        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        if not (is_flash_linear_attention_available() and is_causal_conv1d_available()):
            return False
        importlib.import_module("causal_conv1d")  # fail a built-but-broken ABI here, not at model load
        if not _gdn_forward_threads_reset_kwargs(model_id):
            return False
        # causal_conv1d compiled without the current GPU arch imports fine but raises at first forward;
        # smoke it now so we fall back to unpacked rather than crashing mid-train.
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
    """First-fit-decreasing bin-pack into blocks of at most ``max_length`` tokens.

    Returns rows ``{"input_ids": [...], "seq_lengths": [...]}`` where ``sum(seq_lengths) == len(input_ids)``.
    If ``completion_masks`` is provided, each row also carries ``"completion_mask"`` aligned with its ids.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if completion_masks is not None and len(completion_masks) != len(sequences):
        raise ValueError(
            f"completion_masks must be parallel to sequences: {len(completion_masks)} != {len(sequences)}"
        )
    if completion_masks is None:
        items = [(s[:max_length], None) for s in sequences if s]
    else:
        # Validate mask/sequence alignment before truncation so a mismatch fails loud here.
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
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]), reverse=True)
    bins: list[dict] = []
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
        else:
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
    """Tokenize texts for packing, matching TRL's non-packed SFT prep (EOS appended, default add_special_tokens)."""
    eos = tokenizer.eos_token or ""
    rows = [t if (eos and t.endswith(eos)) else t + eos for t in texts]
    enc = tokenizer(rows, truncation=True, max_length=max_length)  # default add_special_tokens (TRL parity)
    return enc["input_ids"]


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


# Cache the largest causal lower-triangular seen; slice for smaller T (read-only, per-process).
_CAUSAL_TRIL: dict = {}


def _causal_lower_triangular(total: int, torch):
    cached = _CAUSAL_TRIL.get("m")
    if cached is None or cached.shape[0] < total:
        cached = torch.tril(torch.ones(total, total, dtype=torch.bool))
        _CAUSAL_TRIL["m"] = cached
    return cached[:total, :total]


@dataclass
class BlockDiagonalCollator:
    """Collate pre-packed rows into a batch with a 4D block-diagonal causal attention mask (plain SDPA).

    Emits ``input_ids [B,T]``, ``attention_mask [B,1,T,T]`` bool, ``position_ids [B,T]`` (reset per
    example), and ``labels [B,T]`` (-100 at boundaries and pad).

    ``emit_varlen=True`` (GDN hybrids): also emits ``cu_seq_lens_q/k`` and ``seq_idx`` to reset
    linear-attention recurrence and causal conv. Requires batch size == 1 and no padding.
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
        for ids, lens in zip(rows, seglens, strict=True):
            if sum(lens) != len(ids):
                raise ValueError(
                    f"packed row invariant broken: sum(seq_lengths)={sum(lens)} != "
                    f"len(input_ids)={len(ids)} (rows must come from pack_token_ids)"
                )
        longest = max((len(r) for r in rows), default=0)
        m = self.pad_to_multiple_of
        # No padding on the varlen path: trailing pad not covered by cu_seqlens breaks the fla kernel.
        total = longest if self.emit_varlen else (((longest + m - 1) // m) * m if m and m > 1 else longest)
        total = max(total, 1)

        input_ids = torch.full((bsz, total), self.pad_token_id, dtype=torch.long)
        position_ids = torch.zeros((bsz, total), dtype=torch.long)
        seg = torch.full((bsz, total), -1, dtype=torch.long)  # -1 for pad, 0..k-1 per example

        for b, (ids, lens) in enumerate(zip(rows, seglens, strict=True)):
            n = len(ids)
            input_ids[b, :n] = torch.tensor(ids, dtype=torch.long)
            start = 0
            for ex_idx, length in enumerate(lens):
                end = start + length
                position_ids[b, start:end] = torch.arange(length)
                seg[b, start:end] = ex_idx
                start = end

        # Pad shares segment -1 so no query row is all-False (which would NaN softmax).
        same = seg.unsqueeze(2) == seg.unsqueeze(1)  # [B, T, T]
        causal = _causal_lower_triangular(total, torch)
        attention_mask = (same & causal).unsqueeze(1)  # [B, 1, T, T]

        labels = input_ids.clone()
        labels[seg < 0] = self.label_pad_token_id
        labels[position_ids == 0] = self.label_pad_token_id  # first token of each example

        if any("completion_mask" in f for f in features):
            keep = torch.zeros((bsz, total), dtype=torch.bool)
            for b, f in enumerate(features):
                cm = f.get("completion_mask")
                if cm is None:
                    keep[b, :] = True  # no mask -> full-transcript loss
                    continue
                n_real = len(rows[b])
                if len(cm) != n_real:
                    raise ValueError(
                        f"completion_mask length {len(cm)} != row real-token count {n_real} "
                        "(mask must span sum(seq_lengths) == len(input_ids))"
                    )
                keep[b, :n_real] = torch.tensor(cm, dtype=torch.bool)
            labels[~keep] = self.label_pad_token_id

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
        }
        if self.emit_varlen:
            lens = seglens[0]
            cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
            cu[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
            batch["cu_seq_lens_q"] = cu
            batch["cu_seq_lens_k"] = cu
            batch["max_length_q"] = int(max(lens))
            batch["max_length_k"] = int(max(lens))
            batch["seq_idx"] = seg.to(torch.int32)  # [1, T], non-negative (no pad on this path)
        return batch

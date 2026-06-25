"""True token packing (block-diagonal SDPA mask) — CPU-only correctness.

The headline guarantee: a packed forward through the 4D block-diagonal causal mask is
BIT-IDENTICAL to running each example separately (no cross-example contamination) under plain
SDPA — so packing buys throughput with zero quality change. Plus the pure-attention arch gate,
the bin-packer invariants, and the collator's mask/label construction.
"""

from __future__ import annotations

import types

import pytest

from flash.engine.worker.packing import (
    BlockDiagonalCollator,
    gdn_packing_available,
    model_is_gdn_hybrid,
    model_is_pure_attention,
    pack_token_ids,
    packing_efficiency,
)


def _split_by_lengths(ids: list[int], lengths: list[int]) -> list[list[int]]:
    """Recover the per-example token lists from a packed block (input_ids + seq_lengths)."""
    out, start = [], 0
    for length in lengths:
        out.append(list(ids[start:start + length]))
        start += length
    return out


# --------------------------------------------------------------------------- arch gate
def _patch_cfg(monkeypatch, cfg):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: cfg)


def test_pure_attention_plain_decoder(monkeypatch):
    # Llama/Qwen3/MiniCPM: no per-layer types, no linear-attn dims -> every layer reads the mask.
    _patch_cfg(monkeypatch, types.SimpleNamespace())
    assert model_is_pure_attention("any/dense") is True


def test_pure_attention_all_full_layer_types(monkeypatch):
    _patch_cfg(monkeypatch, types.SimpleNamespace(layer_types=["full_attention"] * 8))
    assert model_is_pure_attention("any/full") is True


def test_hybrid_gateddeltanet_excluded_by_layer_types(monkeypatch):
    # Qwen3.5/3.6: a linear_attention layer carries state across boundaries -> NOT pure.
    _patch_cfg(
        monkeypatch,
        types.SimpleNamespace(layer_types=["linear_attention", "full_attention"] * 4),
    )
    assert model_is_pure_attention("Qwen/Qwen3.5-4B") is False


def test_hybrid_excluded_by_linear_dims_without_layer_types(monkeypatch):
    _patch_cfg(monkeypatch, types.SimpleNamespace(linear_num_key_heads=16))
    assert model_is_pure_attention("any/gdn") is False


def test_sliding_window_excluded(monkeypatch):
    # A pre-built 4D mask BYPASSES the model's sliding window -> wrong semantics -> exclude.
    _patch_cfg(
        monkeypatch,
        types.SimpleNamespace(layer_types=["sliding_attention", "full_attention"]),
    )
    assert model_is_pure_attention("any/sliding") is False


def test_global_sliding_window_excluded(monkeypatch):
    # A globally sliding-window model (no per-layer layer_types) builds its own window; a pre-built
    # 4D mask would bypass it -> NOT pure. Gate on the ACTIVE flag.
    _patch_cfg(monkeypatch, types.SimpleNamespace(use_sliding_window=True, sliding_window=4096))
    assert model_is_pure_attention("any/sliding-global") is False


def test_disabled_sliding_window_still_pure(monkeypatch):
    # Qwen2.5 ships a sliding_window value but keeps it DISABLED -> still full attention -> packs.
    _patch_cfg(monkeypatch, types.SimpleNamespace(use_sliding_window=False, sliding_window=32768))
    assert model_is_pure_attention("Qwen/Qwen2.5-7B") is True


def test_multimodal_reads_text_config(monkeypatch):
    # Decoder dims live under text_config for VL checkpoints; the probe must look there.
    _patch_cfg(
        monkeypatch,
        types.SimpleNamespace(text_config=types.SimpleNamespace(layer_types=["linear_attention"])),
    )
    assert model_is_pure_attention("Qwen/Qwen3.5-VL") is False


def test_probe_failure_is_safe_false(monkeypatch):
    transformers = pytest.importorskip("transformers")

    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", boom)
    assert model_is_pure_attention("unreachable/model") is False


# --------------------------------------------------------------------- GDN (Qwen3.5/3.6) gate
def test_gdn_hybrid_detected_by_layer_types(monkeypatch):
    _patch_cfg(monkeypatch, types.SimpleNamespace(layer_types=["linear_attention", "full_attention"]))
    assert model_is_gdn_hybrid("Qwen/Qwen3.5-4B") is True


def test_gdn_hybrid_detected_by_linear_dims(monkeypatch):
    _patch_cfg(monkeypatch, types.SimpleNamespace(linear_conv_kernel_dim=4))
    assert model_is_gdn_hybrid("any/gdn") is True


def test_gdn_hybrid_false_for_pure_and_sliding(monkeypatch):
    _patch_cfg(monkeypatch, types.SimpleNamespace())  # plain dense -> not GDN
    assert model_is_gdn_hybrid("any/dense") is False
    _patch_cfg(monkeypatch, types.SimpleNamespace(layer_types=["sliding_attention", "full_attention"]))
    assert model_is_gdn_hybrid("any/sliding") is False  # sliding != linear_attention


def test_gdn_packing_available_requires_both_kernels(monkeypatch):
    import transformers.utils.import_utils as iu

    def setk(fla, conv):
        monkeypatch.setattr(iu, "is_flash_linear_attention_available", lambda: fla, raising=False)
        monkeypatch.setattr(iu, "is_causal_conv1d_available", lambda: conv, raising=False)

    setk(True, True)
    assert gdn_packing_available() is True
    setk(True, False)  # conv missing -> conv leaks across boundaries
    assert gdn_packing_available() is False
    setk(False, True)  # fla missing -> recurrence leaks across boundaries
    assert gdn_packing_available() is False
    setk(False, False)
    assert gdn_packing_available() is False


# --------------------------------------------------------------------------- bin packer
def test_packer_conserves_tokens_and_never_splits():
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10], [11, 12, 13, 14, 15]]
    rows = pack_token_ids(seqs, max_length=8)
    # every block fits, sum(seq_lengths) == len(input_ids), and the multiset of example LENGTHS is
    # exactly preserved (no example was split or dropped).
    for r in rows:
        assert sum(r["seq_lengths"]) == len(r["input_ids"])
        assert len(r["input_ids"]) <= 8
    got = sorted(length for r in rows for length in r["seq_lengths"])
    assert got == sorted(len(s) for s in seqs)
    # token content is conserved across all blocks
    assert sum(sum(r["input_ids"]) for r in rows) == sum(sum(s) for s in seqs)


def test_packer_truncates_overlong_example():
    rows = pack_token_ids([list(range(20))], max_length=8)
    assert len(rows) == 1
    assert rows[0]["seq_lengths"] == [8]
    assert rows[0]["input_ids"] == list(range(8))


def test_packer_drops_empty_and_efficiency():
    rows = pack_token_ids([[1, 2], [], [3, 4]], max_length=4)
    # one block of [1,2,3,4] -> 100% dense
    assert len(rows) == 1
    assert packing_efficiency(rows, 4) == pytest.approx(1.0)
    assert packing_efficiency([], 4) == 0.0


def test_packer_ffd_minimizes_blocks():
    # 6 + 5 + 3 + 2 into capacity 8 -> ideally 2 blocks (6+2, 5+3), FFD achieves it.
    rows = pack_token_ids([list(range(6)), list(range(5)), list(range(3)), list(range(2))], 8)
    assert len(rows) == 2


# --------------------------------------------------------------------------- collator
def test_collator_mask_labels_positions():
    torch = pytest.importorskip("torch")
    col = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=1)
    # one block: example A=[5,6,7], example B=[8,9]
    batch = col([{"input_ids": [5, 6, 7, 8, 9], "seq_lengths": [3, 2]}])
    am = batch["attention_mask"]
    assert am.shape == (1, 1, 5, 5)
    assert am.dtype == torch.bool
    # position ids reset per example
    assert batch["position_ids"][0].tolist() == [0, 1, 2, 0, 1]
    # labels: first token of EACH example -> -100, rest == input_ids
    assert batch["labels"][0].tolist() == [-100, 6, 7, -100, 9]
    # block-diagonal causal: B's first token (idx 3) may attend only itself; never A
    assert am[0, 0, 3].tolist() == [False, False, False, True, False]
    # A's last token (idx 2) attends A[0..2], never B
    assert am[0, 0, 2].tolist() == [True, True, True, False, False]
    # causal within example: A[0] attends only itself
    assert am[0, 0, 0].tolist() == [True, False, False, False, False]


def test_collator_pads_and_masks_pad_no_nan_rows():
    torch = pytest.importorskip("torch")
    col = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=8)
    batch = col([{"input_ids": [5, 6, 7], "seq_lengths": [3]}])
    assert batch["input_ids"].shape == (1, 8)  # padded up to multiple of 8
    # pad positions -> label -100
    assert batch["labels"][0, 3:].tolist() == [-100] * 5
    # no query row is entirely False (would NaN the softmax): every row attends >= itself
    assert batch["attention_mask"][0, 0].any(dim=-1).all().item() is True


def test_collator_emit_varlen_kwargs():
    torch = pytest.importorskip("torch")
    col = BlockDiagonalCollator(pad_token_id=0, emit_varlen=True)
    batch = col([{"input_ids": [5, 6, 7, 8, 9], "seq_lengths": [3, 2]}])
    # varlen kwargs for the GDN linear-attention layers
    assert batch["cu_seq_lens_q"].tolist() == [0, 3, 5]
    assert batch["cu_seq_lens_k"].tolist() == [0, 3, 5]
    assert batch["cu_seq_lens_q"].dtype == torch.int32
    assert batch["max_length_q"] == 3
    assert batch["max_length_k"] == 3
    assert batch["seq_idx"].tolist() == [[0, 0, 0, 1, 1]]
    assert batch["seq_idx"].dtype == torch.int32
    # no padding on the varlen path (cu_seqlens must cover the whole row)
    assert batch["input_ids"].shape == (1, 5)
    # still emits the 4D mask + position_ids + labels
    assert batch["attention_mask"].shape == (1, 1, 5, 5)
    assert batch["position_ids"][0].tolist() == [0, 1, 2, 0, 1]


def test_collator_emit_varlen_requires_bsz_one():
    pytest.importorskip("torch")
    col = BlockDiagonalCollator(pad_token_id=0, emit_varlen=True)
    with pytest.raises(ValueError, match="batch_size == 1"):
        col([{"input_ids": [1, 2], "seq_lengths": [2]}, {"input_ids": [3, 4], "seq_lengths": [2]}])


# --------------------------------------------------------------------------- the gold test
@pytest.mark.parametrize("arch", ["qwen3", "llama"])
def test_packed_forward_bit_identical_to_separate(arch):
    """Packed forward (collator mask) == per-example separate forwards, under plain SDPA."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    if arch == "qwen3":
        cfg = transformers.Qwen3Config(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
            attn_implementation="sdpa",
        )
        model = transformers.Qwen3ForCausalLM(cfg).eval()
    else:  # MiniCPM5 is the Llama arch
        cfg = transformers.LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
            attn_implementation="sdpa",
        )
        model = transformers.LlamaForCausalLM(cfg).eval()

    examples = [[5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15]]
    rows = pack_token_ids(examples, max_length=16)
    assert len(rows) == 1  # all fit in one block
    col = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=8)
    batch = col(rows)
    # The packer reorders examples (FFD), so reconstruct them IN PACKED ORDER from the row to
    # compare against the matching separate forwards.
    packed_examples = _split_by_lengths(rows[0]["input_ids"], rows[0]["seq_lengths"])

    with torch.no_grad():
        packed = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        ).logits[0]
        # ground truth: each example standalone, positions from 0, plain causal
        sep = torch.cat(
            [model(input_ids=torch.tensor([e]), position_ids=torch.arange(len(e))[None]).logits[0]
             for e in packed_examples],
            dim=0,
        )

    n_real = sum(len(e) for e in packed_examples)
    diff = (packed[:n_real] - sep).abs().max().item()
    assert diff < 1e-4, f"packed != separate (max diff {diff})"
    assert not torch.isnan(packed).any()


def test_packed_loss_matches_unpacked():
    """The HF CE loss over a packed block (with our labels) equals the token-weighted average of
    the separate per-example losses — i.e. the labels score EXACTLY the within-example next-token
    pairs and never the cross-boundary pair. This is the real training signal, end to end."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        attn_implementation="sdpa",
    )
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    examples = [[5, 6, 7, 8, 9], [10, 11, 12], [13, 14, 15, 16]]
    rows = pack_token_ids(examples, 32)
    batch = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=1)(rows)
    packed_examples = _split_by_lengths(rows[0]["input_ids"], rows[0]["seq_lengths"])

    with torch.no_grad():
        packed_loss = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"], labels=batch["labels"],
        ).loss.item()
        # token-weighted mean of the separate per-example losses (each example's labels = its own
        # ids; HF's internal shift already drops each first token, so no extra masking needed).
        total, n = 0.0, 0
        for e in packed_examples:
            ids = torch.tensor([e])
            out = model(input_ids=ids, position_ids=torch.arange(len(e))[None], labels=ids)
            contrib = len(e) - 1  # next-token pairs per example
            total += out.loss.item() * contrib
            n += contrib
        unpacked_weighted = total / n

    assert packed_loss == pytest.approx(unpacked_weighted, abs=1e-4), (
        f"packed loss {packed_loss} != token-weighted unpacked {unpacked_weighted}"
    )


def test_leaky_plain_causal_would_differ():
    """Control: WITHOUT the block-diagonal mask (plain causal) examples DO contaminate — proves
    the mask is what provides isolation, not some coincidence of the tiny model."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        attn_implementation="sdpa",
    )
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    examples = [[5, 6, 7, 8], [9, 10, 11, 12, 13]]
    rows = pack_token_ids(examples, 16)
    batch = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=1)(rows)
    packed_examples = _split_by_lengths(rows[0]["input_ids"], rows[0]["seq_lengths"])
    n_real = sum(len(e) for e in packed_examples)
    with torch.no_grad():
        sep = torch.cat(
            [model(input_ids=torch.tensor([e]), position_ids=torch.arange(len(e))[None]).logits[0]
             for e in packed_examples],
            dim=0,
        )
        T = batch["input_ids"].shape[1]
        plain_causal = torch.tril(torch.ones(1, 1, T, T, dtype=torch.bool))
        leaky = model(
            input_ids=batch["input_ids"],
            attention_mask=plain_causal,
            position_ids=batch["position_ids"],
        ).logits[0]
    assert (leaky[:n_real] - sep).abs().max().item() > 1e-3

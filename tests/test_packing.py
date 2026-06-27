"""True token packing (block-diagonal SDPA mask) — CPU-only correctness.

The headline guarantee: a packed forward through the 4D block-diagonal causal mask matches running
each example separately to FLOAT PRECISION (~1e-7 reduction-order noise, no cross-example
contamination) under plain SDPA — so packing buys throughput with zero training-relevant quality
change. Plus the pure-attention arch gate, the bin-packer invariants, and the collator's
mask/label construction.
"""

from __future__ import annotations

import types

import pytest

from flash.engine.worker.packing import (
    BlockDiagonalCollator,
    build_completion_mask,
    gdn_packing_available,
    model_is_gdn_hybrid,
    model_is_pure_attention,
    pack_token_ids,
    packing_efficiency,
    tokenize_for_packing,
)


class _FakeTok:
    """Minimal tokenizer for the tokenize_for_packing logic (char-code ids)."""

    def __init__(self, eos="<e>"):
        self.eos_token = eos

    def __call__(self, rows, truncation, max_length, add_special_tokens=True):
        # mirror TRL: default add_special_tokens (no override); prepend a BOS marker (id 1) when on,
        # so the test can assert parity with TRL's text-field tokenization. Truncate the TOTAL.
        assert truncation is True
        bos = [1] if add_special_tokens else []
        return {"input_ids": [(bos + [ord(c) for c in r])[:max_length] for r in rows]}


def test_tokenize_for_packing_eos_and_bos_parity():
    tok = _FakeTok(eos="!")
    # row WITHOUT eos -> eos appended (model learns to stop, matching TRL add_eos); WITH eos -> not
    # doubled; default add_special_tokens prepends BOS (id 1), matching TRL's text-field _tokenize.
    ids = tokenize_for_packing(["ab", "cd!"], tok, max_length=100)
    assert ids[0] == [1, ord("a"), ord("b"), ord("!")]
    assert ids[1] == [1, ord("c"), ord("d"), ord("!")]


def test_tokenize_for_packing_truncates_and_handles_no_eos():
    # BOS-inclusive truncation to max_length
    assert tokenize_for_packing(["abcdef"], _FakeTok("!"), max_length=3) == [[1, ord("a"), ord("b")]]
    # eos_token None -> no append, no crash (BOS still added)
    assert tokenize_for_packing(["ab"], _FakeTok(eos=None), max_length=100) == [[1, ord("a"), ord("b")]]


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


def test_mistral_style_sliding_window_excluded(monkeypatch):
    # Mistral-style: sliding_window set, NO use_sliding_window flag -> window is ACTIVE -> exclude.
    _patch_cfg(monkeypatch, types.SimpleNamespace(sliding_window=4096))
    assert model_is_pure_attention("mistralai/Mistral-7B") is False


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


def test_gdn_forward_probe_resolves_actual_arch():
    # The version/API probe resolves the model's ACTUAL DeltaNet class (not a hardcoded qwen3_5), so
    # it stays correct for a future qwen3_6 module. With no model_id it falls back to qwen3_5, whose
    # installed forward threads both reset kwargs -> True; a bogus arch -> safe-False.
    pytest.importorskip("transformers")
    from flash.engine.worker.packing import _gdn_forward_threads_reset_kwargs

    assert _gdn_forward_threads_reset_kwargs(None) is True


def test_gdn_packing_available_false_when_either_kernel_missing(monkeypatch):
    # Safety-critical: if EITHER find_spec probe is False the gate short-circuits to False BEFORE the
    # heavy real-import / source-API checks — a missing kernel must NEVER enable packing (it would
    # leak). (The both-present path additionally requires the real kernels + a kwargs-aware
    # transformers; that is GPU-validated end-to-end, not unit-tested here.)
    pytest.importorskip("transformers")
    import transformers.utils.import_utils as iu

    for fla, conv in [(False, True), (True, False), (False, False)]:
        monkeypatch.setattr(iu, "is_flash_linear_attention_available", lambda fla=fla: fla, raising=False)
        monkeypatch.setattr(iu, "is_causal_conv1d_available", lambda conv=conv: conv, raising=False)
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
    # token CONTENT is conserved exactly — a multiset comparison (not just the sum, which a swap or
    # duplicate could preserve).
    from collections import Counter

    assert Counter(t for r in rows for t in r["input_ids"]) == Counter(t for s in seqs for t in s)


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
    assert packing_efficiency(rows, 0) == 0.0  # no ZeroDivisionError on a bad max_length


def test_collator_rejects_broken_row_invariant():
    pytest.importorskip("torch")
    col = BlockDiagonalCollator(pad_token_id=0)
    with pytest.raises(ValueError, match="invariant broken"):
        col([{"input_ids": [1, 2, 3], "seq_lengths": [2]}])  # sum(lens)=2 != len=3


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


def test_causal_tril_cache_grows_and_slices():
    torch = pytest.importorskip("torch")
    from flash.engine.worker.packing import _CAUSAL_TRIL, _causal_lower_triangular

    _CAUSAL_TRIL.clear()
    big = _causal_lower_triangular(5, torch)
    assert big.shape == (5, 5)
    assert big.equal(torch.tril(torch.ones(5, 5, dtype=torch.bool)))
    small = _causal_lower_triangular(3, torch)  # sliced from the cached 5x5, not rebuilt
    assert small.shape == (3, 3)
    assert small.equal(torch.tril(torch.ones(3, 3, dtype=torch.bool)))


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
def test_packed_forward_matches_separate(arch):
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
    # NUMERICALLY identical: masked positions get ~0 softmax weight, so a real token's output equals
    # its standalone forward up to float reduction-order noise (the longer packed sequence sums more
    # terms) — ~1e-7 here, far below any training-relevant scale. NOT cross-boundary contamination
    # (which the leaky control below shows is ~0.5).
    assert diff < 1e-5, f"packed != separate (max diff {diff})"
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


# ---------------------------------------------------------------- completion-only loss masking
def test_build_completion_mask_clean_prefix():
    # Non-thinking: the prompt render is a clean token prefix of the full row -> mask exactly the
    # prompt, train on the completion (assistant turn + appended EOS).
    tok = _FakeTok(eos="!")
    full = tokenize_for_packing(["ABxy"], tok, max_length=100)[0]  # [1, A, B, x, y, !]
    mask = build_completion_mask("AB", full, tok, max_length=100)
    assert len(mask) == len(full)
    assert mask == [0, 0, 0, 1, 1, 1]  # BOS+A+B masked; xy! trained


def test_build_completion_mask_robust_to_divergent_prompt_suffix():
    # Thinking template: add_generation_prompt=True pre-opens a token (e.g. `<think>`) that the full
    # render diverges from -> the LONGEST SHARED PREFIX is masked, the rest (reasoning + answer) is
    # trained. Here the prompt ends with a token ('<') absent from the full row at that position.
    tok = _FakeTok(eos="!")
    full = tokenize_for_packing(["ABxy"], tok, max_length=100)[0]  # [1, A, B, x, y, !]
    mask = build_completion_mask("AB<", full, tok, max_length=100)  # prompt_ids [1,A,B,<]
    assert mask == [0, 0, 0, 1, 1, 1]  # diverges at idx 3 -> only the shared AB prompt is masked


def test_build_completion_mask_all_prompt_masks_everything():
    # Degenerate: max_length truncation left the WHOLE row as prompt (no completion survived). The row
    # is fully masked (all zeros) — NOT forced to keep the last PROMPT token as a trained target, which
    # would teach the model to reproduce prompt text. A fully-masked row is a loss no-op instead.
    tok = _FakeTok(eos="!")
    full = tokenize_for_packing(["AB"], tok, max_length=100)[0]  # [1, A, B, !]
    mask = build_completion_mask("AB!", full, tok, max_length=100)  # prompt == full
    assert mask == [0, 0, 0, 0]
    assert sum(mask) == 0  # no token trained on -> the example contributes no (prompt) loss


def test_build_completion_mask_empty_full():
    assert build_completion_mask("anything", [], _FakeTok("!"), max_length=100) == []


def test_pack_carries_completion_mask_aligned():
    # The completion mask rides along with input_ids through the FFD reorder + truncation, staying
    # token-aligned (sum(seq_lengths) == len(input_ids) == len(completion_mask)).
    seqs = [[1, 2, 3], [4, 5]]
    masks = [[0, 0, 1], [0, 1]]
    rows = pack_token_ids(seqs, max_length=8, completion_masks=masks)
    assert len(rows) == 1
    r = rows[0]
    assert len(r["completion_mask"]) == len(r["input_ids"]) == sum(r["seq_lengths"])
    # FFD places the longer example ([1,2,3]) first, then [4,5]; masks follow the same order.
    assert r["input_ids"] == [1, 2, 3, 4, 5]
    assert r["completion_mask"] == [0, 0, 1, 0, 1]


def test_pack_truncates_mask_in_lockstep():
    rows = pack_token_ids([[1, 2, 3, 4, 5]], max_length=3, completion_masks=[[0, 0, 1, 1, 1]])
    assert rows[0]["input_ids"] == [1, 2, 3]
    assert rows[0]["completion_mask"] == [0, 0, 1]  # truncated identically to input_ids


def test_pack_drops_empty_with_its_mask():
    rows = pack_token_ids([[1, 2], [], [3]], max_length=8, completion_masks=[[0, 1], [9], [1]])
    # the empty sequence (and its mask) is dropped; survivors stay aligned
    flat_ids = [t for r in rows for t in r["input_ids"]]
    flat_mask = [m for r in rows for m in r["completion_mask"]]
    assert sorted(flat_ids) == [1, 2, 3]
    assert len(flat_mask) == len(flat_ids)
    assert 9 not in flat_mask  # the dropped row's mask never leaked in


def test_pack_rejects_mismatched_mask_count():
    with pytest.raises(ValueError, match="parallel"):
        pack_token_ids([[1, 2], [3, 4]], max_length=8, completion_masks=[[0, 1]])


def test_pack_rejects_mismatched_mask_length():
    # Each mask must align 1:1 with ITS sequence (not just have the right count): a shorter/longer mask
    # would desync from input_ids after truncation, masking the wrong tokens. Reject it up front.
    with pytest.raises(ValueError, match="match its sequence length"):
        pack_token_ids([[1, 2, 3]], max_length=8, completion_masks=[[0, 1]])  # len 2 != 3


def test_pack_without_masks_omits_column():
    # Back-compat: no completion_masks -> rows have no "completion_mask" key (collator stays in its
    # original first-token/pad-only masking mode).
    rows = pack_token_ids([[1, 2], [3, 4]], max_length=8)
    assert all("completion_mask" not in r for r in rows)


def test_collator_masks_prompt_tokens_via_completion_mask():
    torch = pytest.importorskip("torch")
    col = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=1)
    # one example [5,6,7,8] with a 2-token prompt (completion_mask [0,0,1,1]). Base masking ignores
    # only the first token; completion masking ADDITIONALLY ignores token 6 (the 2nd prompt token).
    batch = col([{"input_ids": [5, 6, 7, 8], "seq_lengths": [4], "completion_mask": [0, 0, 1, 1]}])
    assert batch["labels"][0].tolist() == [-100, -100, 7, 8]


def test_collator_completion_mask_optional():
    # Without a completion_mask key the labels are the original first-token-only masking (unchanged).
    torch = pytest.importorskip("torch")
    col = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=1)
    batch = col([{"input_ids": [5, 6, 7, 8], "seq_lengths": [4]}])
    assert batch["labels"][0].tolist() == [-100, 6, 7, 8]


def test_collator_mixed_batch_keeps_unmasked_row_full_loss():
    # In a MIXED batch (one row carries a completion_mask, one does not), the row WITHOUT a mask keeps
    # full-transcript loss (only its first/pad tokens ignored) — it must NOT be silently zeroed to all
    # -100 just because a sibling row in the batch carried a completion_mask.
    torch = pytest.importorskip("torch")
    col = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=1)
    batch = col(
        [
            {"input_ids": [5, 6, 7, 8], "seq_lengths": [4], "completion_mask": [0, 0, 1, 1]},
            {"input_ids": [9, 10, 11, 12], "seq_lengths": [4]},  # no mask -> full-transcript
        ]
    )
    # masked row: first token + the 2nd prompt token ignored
    assert batch["labels"][0].tolist() == [-100, -100, 7, 8]
    # unmasked row: ONLY the first token ignored (full-transcript), not all -100
    assert batch["labels"][1].tolist() == [-100, 10, 11, 12]


def test_collator_rejects_misaligned_completion_mask():
    # A present completion_mask must span the row's REAL tokens 1:1 (len == sum(seq_lengths) ==
    # len(input_ids)). A mis-sized mask, if silently sliced, would shift the prompt/completion boundary
    # and train on the wrong positions — so the collator fails loud instead of guessing.
    pytest.importorskip("torch")
    col = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=1)
    with pytest.raises(ValueError, match="completion_mask length"):
        col([{"input_ids": [5, 6, 7, 8], "seq_lengths": [4], "completion_mask": [0, 1]}])  # len 2 != 4


def test_packed_completion_loss_matches_unpacked():
    """End to end: the HF CE loss over a packed block whose prompt tokens are masked equals the
    token-weighted average of the separate per-example losses computed ONLY on completion tokens —
    so completion-only loss is preserved exactly through packing."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        attn_implementation="sdpa",
    )
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    examples = [[5, 6, 7, 8, 9], [10, 11, 12, 13]]
    cmasks = [[0, 0, 1, 1, 1], [0, 1, 1, 1]]  # 2- and 1-token prompts
    rows = pack_token_ids(examples, 32, completion_masks=cmasks)
    batch = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=1)(rows)
    packed_examples = _split_by_lengths(rows[0]["input_ids"], rows[0]["seq_lengths"])
    packed_cmasks = _split_by_lengths(rows[0]["completion_mask"], rows[0]["seq_lengths"])

    with torch.no_grad():
        packed_loss = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"], labels=batch["labels"],
        ).loss.item()
        total, n = 0.0, 0
        for e, m in zip(packed_examples, packed_cmasks, strict=True):
            ids = torch.tensor([e])
            labels = torch.tensor([[tid if keep else -100 for tid, keep in zip(e, m, strict=True)]])
            out = model(input_ids=ids, position_ids=torch.arange(len(e))[None], labels=labels)
            contrib = sum(1 for i in range(1, len(e)) if m[i])  # shifted, scored positions
            total += out.loss.item() * contrib
            n += contrib
        unpacked_weighted = total / n

    assert packed_loss == pytest.approx(unpacked_weighted, abs=1e-4), (
        f"packed completion-loss {packed_loss} != token-weighted unpacked {unpacked_weighted}"
    )


# ------------------------------------------------ end-to-end across arches (Qwen3 + MiniCPM/Llama)
def _tiny(arch, transformers):
    """Tiny model per arch — qwen3 (pure-attn flagship tier) and llama (the MiniCPM5-1B tier, which
    really is LlamaForCausalLM: verified model_is_pure_attention(openbmb/MiniCPM5-1B) is True)."""
    common = {
        "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 64,
        "attn_implementation": "sdpa",
    }
    if arch == "qwen3":
        return transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**common)).train()
    return transformers.LlamaForCausalLM(transformers.LlamaConfig(**common)).train()


@pytest.mark.parametrize("arch", ["qwen3", "llama"])
def test_e2e_completion_only_packing_per_arch(arch):
    """The whole thing, per arch: pre-tokenized {input_ids, completion_mask} -> pack -> 4D-mask
    collate -> real forward/backward, proving the prior packing optimization (boundary isolation)
    AND the new completion-only masking hold together, and the model actually trains."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = _tiny(arch, transformers)

    # examples = [prompt_len tokens (masked) || completion tokens (trained)]
    raw = [
        ([5, 6, 7], [8, 9, 10, 11]),        # 3-token prompt, 4-token completion
        ([12, 13], [14, 15, 16]),           # 2-token prompt
        ([17, 18, 19, 20], [21, 22]),       # 4-token prompt
    ]
    seqs = [p + c for p, c in raw]
    cmasks = [[0] * len(p) + [1] * len(c) for p, c in raw]
    rows = pack_token_ids(seqs, max_length=32, completion_masks=cmasks)
    col = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=8)
    batch = col(rows)
    packed_examples = _split_by_lengths(rows[0]["input_ids"], rows[0]["seq_lengths"])
    packed_cmasks = _split_by_lengths(rows[0]["completion_mask"], rows[0]["seq_lengths"])

    # (1) ISOLATION: packed per-token logits == each example run standalone (no cross-contamination).
    with torch.no_grad():
        packed_logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        ).logits[0]
        sep = torch.cat(
            [model(input_ids=torch.tensor([e]), position_ids=torch.arange(len(e))[None]).logits[0]
             for e in packed_examples],
            dim=0,
        )
    n_real = sum(len(e) for e in packed_examples)
    assert (packed_logits[:n_real] - sep).abs().max().item() < 1e-5, f"{arch}: packing leaked"

    # (2) COMPLETION-ONLY LOSS == token-weighted per-example completion-only loss.
    with torch.no_grad():
        packed_loss = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"], labels=batch["labels"],
        ).loss.item()
        total, n = 0.0, 0
        for e, m in zip(packed_examples, packed_cmasks, strict=True):
            ids = torch.tensor([e])
            labels = torch.tensor([[t if keep else -100 for t, keep in zip(e, m, strict=True)]])
            out = model(input_ids=ids, position_ids=torch.arange(len(e))[None], labels=labels)
            contrib = sum(1 for i in range(1, len(e)) if m[i])
            total += out.loss.item() * contrib
            n += contrib
    assert packed_loss == pytest.approx(total / n, abs=1e-4), f"{arch}: completion loss wrong"

    # (3) MASKING IS REAL (not a no-op): the completion-only loss differs from the full-sequence loss
    # (same inputs, all-token labels) — proves the prompt is actually excluded.
    full_labels = batch["input_ids"].clone()
    full_labels[batch["position_ids"] == 0] = -100  # only first-token masked (the old behavior)
    pad = batch["input_ids"] == 0
    full_labels[pad & (batch["position_ids"] == 0)] = -100
    with torch.no_grad():
        full_loss = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"], labels=full_labels,
        ).loss.item()
    assert abs(full_loss - packed_loss) > 1e-4, f"{arch}: completion mask had no effect vs full-seq loss"

    # (4) TRAINS: a few SGD steps reduce the completion-only loss on this batch.
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    losses = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"], labels=batch["labels"])
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    assert losses[-1] < losses[0], f"{arch}: loss did not decrease ({losses[0]:.3f} -> {losses[-1]:.3f})"


@pytest.mark.parametrize("arch", ["qwen3", "llama"])
def test_e2e_masked_prompt_positions_get_zero_gradient(arch):
    """No leak through masked tokens: the gradient of the completion-only loss w.r.t. the LOGITS at
    prompt/pad positions is exactly zero (HF's -100 ignore_index), while completion positions get a
    non-zero gradient. This is the property that makes 'train only on the completion' real."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = _tiny(arch, transformers)
    raw = [([5, 6, 7, 8], [9, 10, 11])]
    seqs = [p + c for p, c in raw]
    cmasks = [[0] * len(p) + [1] * len(c) for p, c in raw]
    rows = pack_token_ids(seqs, max_length=16, completion_masks=cmasks)
    batch = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=8)(rows)

    # Forward to logits, then CE against our labels, and inspect d(loss)/d(logits) per position.
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"])
    logits = out.logits
    logits.retain_grad()
    # HF shift: logits[:, :-1] predict labels[:, 1:]
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = batch["labels"][:, 1:].reshape(-1)
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    loss.backward()
    gnorm = logits.grad[0].norm(dim=-1)  # per-position grad magnitude (length T)
    labels0 = batch["labels"][0]
    # grad at position i comes from predicting token i+1; a position contributes iff labels[i+1] != -100
    for i in range(len(labels0) - 1):
        target_kept = labels0[i + 1].item() != -100
        if target_kept:
            assert gnorm[i] > 0, f"{arch}: completion position {i} should have gradient"
        else:
            assert gnorm[i].item() == pytest.approx(0.0, abs=1e-9), f"{arch}: masked position {i} leaked gradient"

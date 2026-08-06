"""Architecture probes and tokenization helpers for the packed training path — CPU-only.

verl owns the packing itself now (``pad_mode: no_padding``), so what is covered here is the part
flash still decides: whether a checkpoint is a GDN hybrid, whether the interpreter running the
forward can reset linear-attention state at example boundaries, and the tokenization/completion-mask
helpers that feed the packed dataset.
"""

from __future__ import annotations

import types

import pytest

from flash.engine.worker import packing
from flash.engine.worker.packing import (
    completion_mask_from_ids,
    gdn_packing_available,
    model_is_gdn_hybrid,
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
    assert tokenize_for_packing(["abcdef"], _FakeTok("!"), max_length=3) == [
        [1, ord("a"), ord("b")]
    ]
    # eos_token None -> no append, no crash (BOS still added)
    assert tokenize_for_packing(["ab"], _FakeTok(eos=None), max_length=100) == [
        [1, ord("a"), ord("b")]
    ]


# --------------------------------------------------------------------------- arch gate
def _patch_cfg(monkeypatch, cfg):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: cfg)


# --------------------------------------------------------------------- GDN (Qwen3.5/3.6) gate
def test_gdn_hybrid_detected_by_layer_types(monkeypatch):
    _patch_cfg(
        monkeypatch, types.SimpleNamespace(layer_types=["linear_attention", "full_attention"])
    )
    assert model_is_gdn_hybrid("Qwen/Qwen3.5-4B") is True


def test_gdn_hybrid_detected_by_linear_dims(monkeypatch):
    _patch_cfg(monkeypatch, types.SimpleNamespace(linear_conv_kernel_dim=4))
    assert model_is_gdn_hybrid("any/gdn") is True


def test_gdn_hybrid_false_for_pure_and_sliding(monkeypatch):
    _patch_cfg(monkeypatch, types.SimpleNamespace())  # plain dense -> not GDN
    assert model_is_gdn_hybrid("any/dense") is False
    _patch_cfg(
        monkeypatch, types.SimpleNamespace(layer_types=["sliding_attention", "full_attention"])
    )
    assert model_is_gdn_hybrid("any/sliding") is False  # sliding != linear_attention


def test_gdn_forward_probe_resolves_actual_arch():
    # The version/API probe resolves the model's ACTUAL DeltaNet class (not a hardcoded qwen3_5), so
    # it stays correct for a future qwen3_6 module. With no model_id it falls back to qwen3_5, whose
    # installed forward threads both reset kwargs -> True; a bogus arch -> safe-False.
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
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
        monkeypatch.setattr(
            iu, "is_flash_linear_attention_available", lambda fla=fla: fla, raising=False
        )
        monkeypatch.setattr(iu, "is_causal_conv1d_available", lambda conv=conv: conv, raising=False)
        assert gdn_packing_available() is False


# ---------------------------------------------------------------- completion-only loss masking
def _mask(tok, prompt, full):
    # Mirror the SFT path: tokenize the prompt the SAME way the full row is tokenized (one batched
    # call, default add_special_tokens, no appended EOS) then derive the boundary from the ids.
    prompt_ids = tok([prompt], truncation=True, max_length=100)["input_ids"][0]
    return completion_mask_from_ids(prompt_ids, full)


def test_completion_mask_clean_prefix():
    # Non-thinking: the prompt render is a clean token prefix of the full row -> mask exactly the
    # prompt, train on the completion (assistant turn + appended EOS).
    tok = _FakeTok(eos="!")
    full = tokenize_for_packing(["ABxy"], tok, max_length=100)[0]  # [1, A, B, x, y, !]
    mask = _mask(tok, "AB", full)
    assert len(mask) == len(full)
    assert mask == [0, 0, 0, 1, 1, 1]  # BOS+A+B masked; xy! trained


def test_completion_mask_robust_to_divergent_prompt_suffix():
    # Thinking template: add_generation_prompt=True pre-opens a token (e.g. `<think>`) that the full
    # render diverges from -> the LONGEST SHARED PREFIX is masked, the rest (reasoning + answer) is
    # trained. Here the prompt ends with a token ('<') absent from the full row at that position.
    tok = _FakeTok(eos="!")
    full = tokenize_for_packing(["ABxy"], tok, max_length=100)[0]  # [1, A, B, x, y, !]
    mask = _mask(tok, "AB<", full)  # prompt_ids [1,A,B,<]
    assert mask == [0, 0, 0, 1, 1, 1]  # diverges at idx 3 -> only the shared AB prompt is masked


def test_completion_mask_all_prompt_masks_everything():
    # Degenerate: max_length truncation left the WHOLE row as prompt (no completion survived). The row
    # is fully masked (all zeros) — NOT forced to keep the last PROMPT token as a trained target, which
    # would teach the model to reproduce prompt text. A fully-masked row is a loss no-op instead.
    tok = _FakeTok(eos="!")
    full = tokenize_for_packing(["AB"], tok, max_length=100)[0]  # [1, A, B, !]
    mask = _mask(tok, "AB!", full)  # prompt == full
    assert mask == [0, 0, 0, 0]
    assert sum(mask) == 0  # no token trained on -> the example contributes no (prompt) loss


def test_completion_mask_from_ids_empty_full():
    # Empty full row -> empty mask, regardless of prompt ids.
    assert completion_mask_from_ids([1, 2, 3], []) == []


def test_pretokenize_completion_only_drops_empty_and_keeps_lockstep():
    # The SFT pre-tokenizer batches both tokenizations, builds the masks, and drops rows with no
    # completion target — keeping texts and pretok in lockstep so downstream token accounting matches.
    from flash.engine.worker.sft import _pretokenize_completion_only

    tok = _FakeTok(eos="!")
    texts = [
        {"text": "ABxy", "prompt_text": "AB"},  # real completion -> kept
        {"text": "AB", "prompt_text": "AB!"},  # prompt == full row -> all-zero mask -> dropped
        {"text": "CDz", "prompt_text": "CD"},  # real completion -> kept
    ]
    kept_texts, pretok, n_dropped = _pretokenize_completion_only(texts, tok, max_length=100)
    assert n_dropped == 1
    assert [t["text"] for t in kept_texts] == ["ABxy", "CDz"]  # lockstep: the dropped row is gone
    assert len(pretok) == 2
    assert all(any(r["completion_mask"]) for r in pretok)  # every kept row has a completion token
    assert all(len(r["completion_mask"]) == len(r["input_ids"]) for r in pretok)


def test_pretokenize_drops_content_free_completion():
    # A completion whose only masked token is special (an empty assistant turn -> just EOS/turn-closer)
    # has no REAL target and must be dropped — not train the model to emit EOS immediately. "Special" is
    # the tokenizer's own all_special_ids, so the check is template-agnostic.
    from flash.engine.worker.sft import _pretokenize_completion_only

    class _TokWithSpecials(_FakeTok):
        all_special_ids = (1, ord("!"))  # BOS + EOS marked special

    tok = _TokWithSpecials(eos="!")
    texts = [
        {
            "text": "AB",
            "prompt_text": "AB",
        },  # full [1,A,B,!]; prompt [1,A,B] -> completion=[!] (EOS) -> dropped
        {"text": "ABx", "prompt_text": "AB"},  # completion=[x,!] has real token x -> kept
    ]
    kept_texts, _pretok, n_dropped = _pretokenize_completion_only(texts, tok, max_length=100)
    assert n_dropped == 1
    assert [t["text"] for t in kept_texts] == ["ABx"]
    # Without special ids the filter degrades to "any masked token" (NaN guard only): the EOS-only row
    # survives, confirming the special-id set is what distinguishes a content-free target.
    kept2, _, dropped2 = _pretokenize_completion_only(texts, _FakeTok(eos="!"), max_length=100)
    assert dropped2 == 0
    assert [t["text"] for t in kept2] == ["AB", "ABx"]


# ------------------------------------------------- raising probes vs the swallowing wrappers
def _patch_cfg_raises(monkeypatch, exc):
    transformers = pytest.importorskip("transformers")

    def _boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _boom)


def test_swallowing_wrappers_keep_returning_false_on_probe_failure(monkeypatch):
    """The runtime gates must keep their fail-closed-to-False contract.

    sft_train/rl_train/opd_train only ever ask "may I pack"; answering False when the config cannot
    be read is the safe answer there and must not become an exception.
    """
    _patch_cfg_raises(monkeypatch, OSError("hub read timed out"))
    assert packing.model_is_gdn_hybrid("any/model") is False
    assert packing.model_is_pure_attention("any/model") is False


def test_raising_probes_propagate_so_a_digest_is_never_frozen_from_a_failure(monkeypatch):
    """The profile path needs the opposite: a probe that could not answer must RAISE.

    False-on-error is a wrong ANSWER, and a wrong answer frozen into a workload profile is compared
    byte-for-byte against a later re-derivation that may have reached the config.
    """
    _patch_cfg_raises(monkeypatch, OSError("hub read timed out"))
    with pytest.raises(OSError, match="hub read timed out"):
        packing.probe_is_gdn_hybrid("any/model")
    with pytest.raises(OSError, match="hub read timed out"):
        packing.probe_is_pure_attention("any/model")


def test_raising_and_swallowing_probes_agree_when_the_config_resolves(monkeypatch):
    """Same answer on the happy path -- the wrapper only adds the except clause."""
    _patch_cfg(
        monkeypatch, types.SimpleNamespace(layer_types=["linear_attention", "full_attention"])
    )
    assert packing.probe_is_gdn_hybrid("any/gdn") is packing.model_is_gdn_hybrid("any/gdn") is True
    _patch_cfg(monkeypatch, types.SimpleNamespace(layer_types=["full_attention"]))
    assert (
        packing.probe_is_pure_attention("any/dense")
        is packing.model_is_pure_attention("any/dense")
        is True
    )

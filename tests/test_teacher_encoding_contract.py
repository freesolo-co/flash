"""Span and tokenizer-loading contract for managed OPD teacher scoring.

Every token the teacher scores is attributed back to a character span of the source text, and
those spans are what decide which teacher logprob lands on which student token. A tokenizer that
returns a span set which merely *looks* plausible -- one short offset, one reordered pair, one
character of drift on a multi-byte boundary -- silently misaligns supervision rather than
failing, so the alignment is only as trustworthy as the rejection rules around it.

tests/test_parasail_teacher.py covers the two accepted shapes. This file covers the rules that
say no, plus the normalization that turns an under-covering offset list into an exact tiling,
because those are the paths that stand between a malformed tokenizer response and corrupted
training signal.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys

import pytest

from flash.engine.plan.recipe import TEACHER_MODELS, TeacherModel
from flash.engine.worker.teacher import encoding as encoding_module
from flash.engine.worker.teacher.encoding import (
    EncodedTeacherToken,
    _byte_piece_offsets,
    _exact_source_spans,
    _sha256,
    load_teacher_tokenizer,
)

_INVALID_SPAN = "returned an invalid source span"
_NO_COVER = "offsets do not cover the source text"


# --- rejection rules -------------------------------------------------------------------------


def test_id_and_offset_counts_must_match():
    # one id short of its offsets means the pairing below is guesswork; zip(strict=True) would
    # raise far from the cause, so the count is checked before any span is interpreted.
    with pytest.raises(RuntimeError, match="inconsistent ids and offsets"):
        _exact_source_spans("ab", [1], [(0, 1), (1, 2)])


def test_empty_text_accepts_no_tokens_and_rejects_any_token():
    assert _exact_source_spans("", [], []) == []
    with pytest.raises(RuntimeError, match="emitted tokens for empty text"):
        _exact_source_spans("", [1], [(0, 0)])


def test_nonempty_text_must_produce_at_least_one_token():
    # an empty id list would otherwise normalize to "covered", reporting full coverage of text
    # that nothing was scored against.
    with pytest.raises(RuntimeError, match="emitted no ids for nonempty text"):
        _exact_source_spans("ab", [], [])


@pytest.mark.parametrize(
    ("case", "offsets"),
    [
        # a list is not a tuple: the isinstance check is deliberate, since a mutable span could
        # be rewritten after validation.
        ("list instead of tuple", [[0, 2]]),
        ("three-element span", [(0, 1, 2)]),
        ("one-element span", [(0,)]),
        # bools are ints in Python, so `True` would pass a bare isinstance(int) check and index
        # as 1. Rejecting it keeps a JSON `true` from being read as a coordinate.
        ("bool start", [(True, 2)]),
        ("bool end", [(0, True)]),
        ("float start", [(0.0, 2)]),
        ("negative start", [(-1, 2)]),
        ("end before start", [(2, 1)]),
        ("end past end of text", [(0, 5)]),
    ],
)
def test_malformed_span_shapes_are_rejected(case, offsets):
    with pytest.raises(RuntimeError, match=_INVALID_SPAN):
        _exact_source_spans("ab", [1] * len(offsets), offsets)


def test_starts_must_not_move_backwards():
    # tokens arrive in generation order, so a start that regresses means the response was
    # reordered; accepting it would attribute a later token's logprob to earlier text.
    with pytest.raises(RuntimeError, match=_INVALID_SPAN):
        _exact_source_spans("abc", [1, 2], [(1, 2), (0, 1)])


def test_equal_starts_are_allowed_for_split_multibyte_characters():
    # two tokens inside one character legitimately share a start; only a decrease is a fault.
    tokens = _exact_source_spans("é", [1, 2], [(0, 1), (0, 1)])
    assert [(token.start, token.end) for token in tokens] == [(0, 1), (0, 1)]


# --- normalization to an exact tiling --------------------------------------------------------


def test_leading_gap_is_pulled_back_to_the_start_of_the_text():
    # a tokenizer that drops leading whitespace from its first offset still has to account for
    # that text, or the first scored token would start mid-string.
    assert _exact_source_spans("abc", [1], [(1, 3)]) == [EncodedTeacherToken(1, 0, 3)]


def test_interior_gap_extends_the_preceding_token():
    tokens = _exact_source_spans("abcd", [1, 2], [(0, 1), (2, 4)])
    assert [(token.start, token.end) for token in tokens] == [(0, 2), (2, 4)]


def test_trailing_gap_extends_the_final_token():
    assert _exact_source_spans("abcd", [1], [(0, 2)]) == [EncodedTeacherToken(1, 0, 4)]


def test_normalized_spans_tile_the_text_exactly():
    text = "hello world"
    tokens = _exact_source_spans(text, [1, 2, 3], [(0, 5), (5, 6), (6, 11)])
    assert "".join(text[token.start : token.end] for token in tokens) == text


def test_normalization_covers_the_text_for_every_accepted_offset_list():
    """Normalization, not the guard below it, is what makes coverage hold.

    ``_exact_source_spans`` ends with a re-check that the normalized spans reach the end of the
    text. That re-check is a backstop: an exhaustive sweep of every offset list that passes the
    per-span validation (all texts up to 4 characters, up to 3 tokens, 1776 inputs) leaves it
    unreached, because the three gap repairs above already close every hole. The property it
    guards is asserted here directly, so a change that breaks the repairs fails on the property
    rather than silently relying on a branch nothing exercises.
    """
    for length in range(1, 5):
        text = "a" * length
        candidates = [(s, e) for s in range(length + 1) for e in range(s, length + 1)]
        for count in range(1, 4):
            for combo in itertools.product(candidates, repeat=count):
                starts = [span[0] for span in combo]
                if any(starts[i] < starts[i - 1] for i in range(1, count)):
                    continue
                tokens = _exact_source_spans(text, list(range(count)), list(combo))
                assert tokens[0].start == 0
                # coverage is a property of the UNION, not of the final token: a trailing
                # zero-width span leaves the last token short while an earlier, longer span
                # already reaches the end. That is the same union the guard checks.
                covered = 0
                for token in sorted(tokens, key=lambda t: (t.start, t.end)):
                    assert token.start <= covered, (text, combo)
                    covered = max(covered, token.end)
                assert covered == length, (text, combo)


def test_token_ids_are_coerced_to_int_and_paired_in_order():
    tokens = _exact_source_spans("ab", [1, 2], [(0, 1), (1, 2)])
    assert [token.token_id for token in tokens] == [1, 2]
    assert all(isinstance(token.token_id, int) for token in tokens)


# --- byte-piece offsets ----------------------------------------------------------------------


def test_pieces_must_reproduce_the_source_bytes_exactly():
    # the byte join is the only proof that the decoded pieces are the text that was scored; a
    # dropped piece would otherwise shift every following span by its length.
    with pytest.raises(RuntimeError, match="do not exactly reproduce the source text"):
        _byte_piece_offsets("ab", [b"a"])


def test_ascii_pieces_map_to_one_character_each():
    assert _byte_piece_offsets("abc", [b"a", b"b", b"c"]) == [(0, 1), (1, 2), (2, 3)]


def test_piece_spanning_two_characters_covers_both():
    assert _byte_piece_offsets("ab", [b"ab"]) == [(0, 2)]


def test_multibyte_character_split_across_pieces_keeps_both_inside_it():
    # a BPE merge can cut a 4-byte emoji in half. Neither half is a character, so both map onto
    # the whole character rather than to an empty or out-of-range span.
    spans = _byte_piece_offsets("é!", [b"\xc3", b"\xa9", b"!"])
    assert spans == [(0, 1), (0, 1), (1, 2)]
    tokens = _exact_source_spans("é!", [1, 2, 3], spans)
    assert [(token.start, token.end) for token in tokens] == spans


# --- pinned tokenizer loading ----------------------------------------------------------------


def test_sha256_matches_hashlib_over_a_chunked_read(tmp_path):
    # read in 1 MiB chunks, so a file larger than one chunk is the case that matters.
    payload = b"x" * (1024 * 1024 + 7)
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)
    assert _sha256(str(path)) == hashlib.sha256(payload).hexdigest()


def _stub_download(monkeypatch, contents: dict[str, bytes], tmp_path):
    """Serve tokenizer files from disk instead of the hub, recording what was requested."""
    calls: list[tuple[str, str, str]] = []

    def fake_download(repo, filename, revision=None):
        calls.append((repo, filename, revision))
        path = tmp_path / filename
        path.write_bytes(contents[filename])
        return str(path)

    # _download_files imports huggingface_hub inside the function body, so the stub has to be in
    # sys.modules at call time rather than patched onto the module under test.
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        type("_HubStub", (), {"hf_hub_download": staticmethod(fake_download)}),
    )
    return calls


def test_download_verifies_each_file_against_its_pinned_hash(monkeypatch, tmp_path):
    payload = b'{"ok": true}'
    calls = _stub_download(monkeypatch, {"tokenizer.json": payload}, tmp_path)
    model = TeacherModel(
        alias="stub",
        model_id="stub-model",
        display_name="Stub",
        usd_per_1m=(1.0, 2.0),
        tokenizer_repo="stub/repo",
        tokenizer_revision="deadbeef",
        tokenizer_kind="tokenizer_json",
        tokenizer_files=(("tokenizer.json", hashlib.sha256(payload).hexdigest()),),
    )

    paths = encoding_module._download_files(model)

    assert list(paths) == ["tokenizer.json"]
    # the pinned revision has to reach the hub call, or the hash gate would be verifying
    # whatever the branch currently points at.
    assert calls == [("stub/repo", "tokenizer.json", "deadbeef")]


def test_download_rejects_a_file_whose_hash_does_not_match_the_pin(monkeypatch, tmp_path):
    _stub_download(monkeypatch, {"tokenizer.json": b"tampered"}, tmp_path)
    model = TeacherModel(
        alias="stub",
        model_id="stub-model",
        display_name="Stub",
        usd_per_1m=(1.0, 2.0),
        tokenizer_repo="stub/repo",
        tokenizer_revision="deadbeef",
        tokenizer_kind="tokenizer_json",
        tokenizer_files=(("tokenizer.json", "0" * 64),),
    )

    with pytest.raises(RuntimeError, match="tokenizer hash mismatch"):
        encoding_module._download_files(model)


def test_tokenizer_json_kind_encodes_through_the_span_contract(monkeypatch, tmp_path):
    """A real `tokenizers` model, built in-test, reaches encode() and yields exact spans."""
    tokenizers = pytest.importorskip("tokenizers")

    vocab = {"a": 0, "b": 1, "ab": 2}
    model = tokenizers.models.WordPiece(vocab, unk_token="a", max_input_chars_per_word=100)
    tokenizer = tokenizers.Tokenizer(model)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    payload = tokenizer_path.read_bytes()

    _stub_download(monkeypatch, {"tokenizer.json": payload}, tmp_path)
    stub = TeacherModel(
        alias="stub",
        model_id="stub-json-model",
        display_name="Stub",
        usd_per_1m=(1.0, 2.0),
        tokenizer_repo="stub/repo",
        tokenizer_revision="rev",
        tokenizer_kind="tokenizer_json",
        tokenizer_files=(("tokenizer.json", hashlib.sha256(payload).hexdigest()),),
    )
    monkeypatch.setattr(encoding_module, "teacher_for_model_id", lambda _value: stub)
    load_teacher_tokenizer.cache_clear()

    tokens = load_teacher_tokenizer("stub-json-model").encode("ab")

    assert [token.token_id for token in tokens] == [2]
    assert [(token.start, token.end) for token in tokens] == [(0, 2)]
    load_teacher_tokenizer.cache_clear()


def test_kimi_tiktoken_kind_builds_an_encoding_and_spans_its_output(monkeypatch, tmp_path):
    """The Kimi path decodes to bytes rather than offsets, so it is exercised on a tiny BPE.

    The real 2.8 MB tiktoken model is deliberately not vendored into the repo; a synthetic
    mergeable-rank table drives the same construction and encode path.
    """
    pytest.importorskip("tiktoken")

    import base64

    ranks = {bytes([value]): value for value in range(256)}
    model_blob = "\n".join(
        f"{base64.b64encode(token).decode()} {rank}" for token, rank in ranks.items()
    ).encode()
    config_blob = json.dumps(
        {"added_tokens_decoder": {str(len(ranks)): {"content": "<|im_end|>"}}}
    ).encode()

    _stub_download(
        monkeypatch,
        {"tokenizer_config.json": config_blob, "tiktoken.model": model_blob},
        tmp_path,
    )
    stub = TeacherModel(
        alias="stub-kimi",
        model_id="stub-kimi-model",
        display_name="Stub Kimi",
        usd_per_1m=(1.0, 2.0),
        tokenizer_repo="stub/kimi",
        tokenizer_revision="rev",
        tokenizer_kind="kimi_tiktoken",
        tokenizer_files=(
            ("tokenizer_config.json", hashlib.sha256(config_blob).hexdigest()),
            ("tiktoken.model", hashlib.sha256(model_blob).hexdigest()),
        ),
    )
    monkeypatch.setattr(encoding_module, "teacher_for_model_id", lambda _value: stub)
    load_teacher_tokenizer.cache_clear()

    tokens = load_teacher_tokenizer("stub-kimi-model").encode("hi")

    assert [(token.start, token.end) for token in tokens] == [(0, 1), (1, 2)]
    load_teacher_tokenizer.cache_clear()


def test_kimi_config_without_an_added_token_decoder_is_rejected(monkeypatch, tmp_path):
    pytest.importorskip("tiktoken")
    config_path = tmp_path / "tokenizer_config.json"
    config_path.write_text(json.dumps({"model_max_length": 128}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="no added token decoder"):
        encoding_module._KimiTikToken(str(config_path), str(tmp_path / "unused.model"))


@pytest.mark.parametrize(
    "decoder",
    [
        {"1": "not-a-dict"},
        {"1": {"no_content_key": True}},
        {"1": {"content": 5}},
    ],
)
def test_kimi_config_with_an_invalid_added_token_is_rejected(monkeypatch, tmp_path, decoder):
    pytest.importorskip("tiktoken")
    config_path = tmp_path / "tokenizer_config.json"
    config_path.write_text(json.dumps({"added_tokens_decoder": decoder}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid added token"):
        encoding_module._KimiTikToken(str(config_path), str(tmp_path / "unused.model"))


def test_unsupported_tokenizer_kind_is_rejected_by_name(monkeypatch):
    stub = TeacherModel(
        alias="stub",
        model_id="stub-unknown",
        display_name="Stub",
        usd_per_1m=(1.0, 2.0),
        tokenizer_repo="stub/repo",
        tokenizer_revision="rev",
        tokenizer_kind="sentencepiece",
        tokenizer_files=(),
    )
    monkeypatch.setattr(encoding_module, "teacher_for_model_id", lambda _value: stub)
    monkeypatch.setattr(encoding_module, "_download_files", lambda _model: {})
    load_teacher_tokenizer.cache_clear()

    with pytest.raises(RuntimeError, match="unsupported managed teacher tokenizer kind"):
        load_teacher_tokenizer("stub-unknown")
    load_teacher_tokenizer.cache_clear()


def test_every_catalog_teacher_declares_a_supported_tokenizer_kind():
    # load_teacher_tokenizer only implements these two kinds, so a catalog entry naming a third
    # would raise at scoring time rather than at review time.
    kinds = {teacher.tokenizer_kind for teacher in TEACHER_MODELS.values()}
    assert kinds <= {"tokenizer_json", "kimi_tiktoken"}


def test_every_catalog_teacher_pins_a_full_length_hash_for_each_tokenizer_file():
    for teacher in TEACHER_MODELS.values():
        assert teacher.tokenizer_files, f"{teacher.alias} pins no tokenizer file"
        for filename, expected_hash in teacher.tokenizer_files:
            assert len(expected_hash) == 64, f"{teacher.alias} {filename} is not a sha256"
            assert expected_hash == expected_hash.lower().strip()

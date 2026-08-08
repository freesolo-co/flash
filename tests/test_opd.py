"""On-policy distillation (opd): groupwise reverse-KL (gkd) cross-tokenizer alignment, the teacher
client, spec/cost plumbing, and the loss math.

All CPU-only. The loss-math tests need torch and are skipped where it is unavailable (they run in
CI, which has the training stack).
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from flash.engine.worker.teacher import TeacherClient
from flash.engine.worker.tokenizer_align import (
    StudentToken,
    TeacherToken,
    groupwise_alignment,
    groupwise_coverage,
)


def _student(spans):
    return [StudentToken(token_id=i, start=a, end=b) for i, (a, b) in enumerate(spans)]


def _teacher(spans):
    """TeacherToken list from (start, end) spans; logprob = -(index+1)."""
    return [
        TeacherToken(text="", logprob=-(i + 1.0), start=a, end=b) for i, (a, b) in enumerate(spans)
    ]


def test_drop_fully_forced_groups_removes_all_forced_spans():
    from flash.engine.worker.opd import _drop_fully_forced_groups

    groups = [([0], -1.0), ([1, 2], -2.0), ([3], -3.0)]
    # Student tokens 0 and 3 were grammar-forced; the [1, 2] group has a free token so it survives.
    assert _drop_fully_forced_groups(groups, (True, False, False, True)) == [([1, 2], -2.0)]


def test_drop_fully_forced_groups_is_a_noop_without_a_mask():
    from flash.engine.worker.opd import _drop_fully_forced_groups

    groups = [([0], -1.0), ([1], -2.0)]
    assert _drop_fully_forced_groups(groups, ()) == groups


def test_drop_fully_forced_groups_keeps_a_partially_forced_span():
    from flash.engine.worker.opd import _drop_fully_forced_groups

    # Token 0 forced, token 1 free -> the group still carries real signal, so it is kept.
    assert _drop_fully_forced_groups([([0, 1], -1.0)], (True, False)) == [([0, 1], -1.0)]


# The two normalization tests that stood here asserted over TRL's `_prepare_gkd_groups` /
# `_gkd_loss_from_logps`. verl carries the same invariant in one place instead of two, and it is now
# proved against the shipped implementation in
# test_opd_train.py::test_dropped_forced_groups_renormalize_over_surviving_tokens_only.


def _install_student_loader_fakes(monkeypatch, *, causal_raises=False, vl_raises=False):
    """Install tiny peft/transformers fakes for _student_model loader-selection tests."""
    calls = []

    class _Base:
        def __init__(self, loader):
            self.loader = loader

        def to(self, device):
            calls.append(("to", self.loader, device))
            return self

    class _Causal:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(("causal", args, kwargs))
            if causal_raises:
                raise AssertionError("AutoModelForCausalLM should not be used")
            return _Base("causal")

    class _ImageText:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(("vl", args, kwargs))
            if vl_raises:
                raise AssertionError("AutoModelForImageTextToText should not be used")
            return _Base("vl")

    def get_peft_model(base, peft_config):
        calls.append(("peft", base.loader, peft_config))
        return {"loader": base.loader, "peft_config": peft_config}

    peft = types.ModuleType("peft")
    peft.get_peft_model = get_peft_model
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = _Causal
    transformers.AutoModelForImageTextToText = _ImageText
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


# --------------------------------------------------------------------------------------------------
# gkd groupwise alignment — the coarsest common refinement of the two tokenizations
# --------------------------------------------------------------------------------------------------
def test_gkd_groups_are_one_per_shared_boundary_when_tokenizers_agree():
    # Both tokenizers segment identically -> every student token is its own group.
    student = _student([(0, 2), (2, 5)])
    teacher = _teacher([(0, 2), (2, 5)])
    groups = groupwise_alignment(student, teacher)
    assert [s_idx for s_idx, _ in groups] == [[0], [1]]
    assert groups[0][1] == -1.0  # teacher logprob of the first span
    assert groups[1][1] == -2.0
    assert groupwise_coverage(groups, student) == 1.0


def test_gkd_span_grows_across_disagreement_and_covers_every_token():
    # Teacher emits one token where the student emits two; no shared interior boundary at char 3,
    # so both student tokens fall in a single group summing the (single) teacher logprob.
    student = _student([(0, 3), (3, 6)])
    teacher = _teacher([(0, 6)])
    groups = groupwise_alignment(student, teacher)
    assert len(groups) == 1
    assert groups[0][0] == [0, 1]  # both student tokens grouped together
    assert groups[0][1] == -1.0
    assert groupwise_coverage(groups, student) == 1.0  # NO masking


def test_gkd_partial_agreement_splits_at_shared_boundaries_only():
    # Shared boundaries at chars 0 and 2; the student's interior boundary at 4 is not shared.
    student = _student([(0, 2), (2, 4), (4, 6)])
    teacher = _teacher([(0, 2), (2, 6)])
    groups = groupwise_alignment(student, teacher)
    assert [s_idx for s_idx, _ in groups] == [[0], [1, 2]]
    assert groupwise_coverage(groups, student) == 1.0


def test_gkd_merges_leading_student_only_span_so_no_token_is_dropped():
    # Student starts at char 0 but the first teacher token starts at char 2, so [0,2) is a
    # student-only span. Those tokens must NOT be dropped — they merge into the first teacher-bearing
    # group (coverage stays 100%).
    student = _student([(0, 1), (1, 2), (2, 5)])
    teacher = _teacher([(2, 5)])
    groups = groupwise_alignment(student, teacher)
    assert len(groups) == 1
    assert groups[0][0] == [0, 1, 2]  # all three student tokens covered
    assert groupwise_coverage(groups, student) == 1.0


def test_gkd_empty_inputs_yield_no_groups():
    assert groupwise_alignment([], _teacher([(0, 1)])) == []
    assert groupwise_alignment(_student([(0, 1)]), []) == []
    assert groupwise_coverage([], []) == 0.0


def test_gkd_coverage_never_exceeds_100pct_with_in_span_zero_width_token():
    # A zero-width student token (partial-byte fragment / mid-completion special) at char 2 rides
    # ALONG inside the group (so the span's student logprob sum stays complete) but must NOT be
    # counted as covered: 2 alignable tokens, both grouped -> exactly 100%, not 150% (the real-run
    # bug the probe surfaced).
    student = _student([(0, 2), (2, 2), (2, 5)])  # middle token is zero-width
    teacher = _teacher([(0, 5)])
    groups = groupwise_alignment(student, teacher)
    assert groups[0][0] == [0, 1, 2]  # all three ride in the group (logprob sum stays whole)
    assert groupwise_coverage(groups, student) == 1.0


# --------------------------------------------------------------------------------------------------
# student tokenization: the loss trains the SAMPLED ids (not a re-tokenization of decoded text)
# --------------------------------------------------------------------------------------------------
def test_student_tokens_use_sampled_ids_with_offsets_into_completion_text():
    from flash.engine.worker.opd_gkd import student_tokens_with_offsets

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            # id 1 -> 'h', id 2 -> 'i', id 3 -> a special token that decodes to nothing.
            m = {1: "h", 2: "i", 3: ""}
            return "".join(m[i] for i in ids)

    ids, toks = student_tokens_with_offsets(_Tok(), [1, 2, 3], "hi")
    assert ids == [1, 2, 3]  # SAMPLED ids preserved verbatim — no lossy re-tokenization
    assert (toks[0].start, toks[0].end) == (0, 1)  # 'h'
    assert (toks[1].start, toks[1].end) == (1, 2)  # 'i'
    assert (toks[2].start, toks[2].end) == (2, 2)  # special token -> zero-width span (excluded)


def test_student_tokens_share_span_for_split_multibyte_char():
    """Regression (opd.py): a byte-level tokenizer can split one multi-byte char across
    two ids; the first decodes to U+FFFD until the second arrives. Measuring each id's decoded length
    independently gave one id the whole char and the other a ZERO-WIDTH span — dropping a real
    byte-token from the alignment and undercounting the char's student logprob. Both byte-ids must
    share the completed-char span so neither is dropped."""
    from flash.engine.worker.opd_gkd import student_tokens_with_offsets

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            ids = [int(i) for i in ids]
            out = ""
            k = 0
            while k < len(ids):
                if ids[k] == 7:
                    out += "x"
                    k += 1
                elif ids[k] == 10 and k + 1 < len(ids) and ids[k + 1] == 11:
                    out += "😀"  # both byte-ids present -> the real char
                    k += 2
                elif ids[k] == 10:
                    out += "�"  # first byte only -> Unicode replacement char
                    k += 1
                else:
                    k += 1
            return out

    # "x😀": id 7 -> 'x'; ids 10,11 -> the two bytes of the emoji.
    ids, toks = student_tokens_with_offsets(_Tok(), [7, 10, 11], "x😀")
    assert ids == [7, 10, 11]
    assert (toks[0].start, toks[0].end) == (0, 1)  # 'x'
    # both halves of the split char share the SAME [1, 2) span (before the fix, id 11 was (2, 2),
    # zero-width -> dropped from the alignment / coverage denominator).
    assert (toks[1].start, toks[1].end) == (1, 2)
    assert (toks[2].start, toks[2].end) == (1, 2)


def test_student_tokens_do_not_over_merge_a_genuine_replacement_char():
    """Regression (opd.py:120-126): the U+FFFD merge heuristic must NOT fire when a token
    LEGITIMATELY decodes to the replacement glyph (the model actually emitted U+FFFD as content). Such
    a token is already reflected in completion_text, so decode(prefix) is a prefix of it — the loop
    must stop and keep it as its own span instead of swallowing the following token."""
    from flash.engine.worker.opd_gkd import student_tokens_with_offsets

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            m = {20: "�", 21: "y"}  # id 20 IS a genuine U+FFFD content token
            return "".join(m[int(x)] for x in ids)

    # completion "�y": the genuine replacement char (id 20) and 'y' (id 21) are SEPARATE tokens.
    ids, toks = student_tokens_with_offsets(_Tok(), [20, 21], "�y")
    assert ids == [20, 21]
    # not over-merged: distinct spans (the buggy heuristic gave both [0, 2)).
    assert (toks[0].start, toks[0].end) == (0, 1)
    assert (toks[1].start, toks[1].end) == (1, 2)


def test_student_tokens_offsets_decode_is_not_quadratic():
    """Regression (opd.py): offsets must be built by decoding a SMALL window per step, not
    the whole growing prefix ids[:i+1] (which was O(len^2) and dominated CPU on long completions).
    Assert the longest id-slice handed to tok.decode stays bounded regardless of completion length."""
    from flash.engine.worker.opd_gkd import student_tokens_with_offsets

    class _Tok:
        def __init__(self):
            self.max_ids = 0

        def decode(self, ids, skip_special_tokens=True):
            ids = list(ids)
            self.max_ids = max(self.max_ids, len(ids))
            return "".join("abcdefghij"[int(x) % 10] for x in ids)  # 1 char/id, no split chars

    tok = _Tok()
    n = 200
    ids = list(range(n))
    text = "".join("abcdefghij"[i % 10] for i in ids)
    out_ids, toks = student_tokens_with_offsets(tok, ids, text)
    assert out_ids == ids
    assert len(toks) == n
    # each step decodes only its own ~1-token window -> max slice is tiny, NOT ~n (a prefix decode).
    assert tok.max_ids <= 2, f"decode saw up to {tok.max_ids} ids -> quadratic prefix decoding"
    assert (toks[0].start, toks[0].end) == (0, 1)
    assert (toks[-1].start, toks[-1].end) == (n - 1, n)  # offsets still correct


def test_trim_trailing_stop_drops_delimiter_from_ids_and_text():
    from flash.engine.worker.opd_gkd import _trim_trailing_stop

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            m = {1: "A", 2: "n", 3: "s", 4: "</", 5: "answer>"}
            return "".join(m[int(i)] for i in ids)

    # completion "Ans</answer>"; stop "</answer>" spans ids 4,5 -> keep "Ans" / ids [1,2,3].
    ids, text = _trim_trailing_stop(_Tok(), [1, 2, 3, 4, 5], "Ans</answer>", ["</answer>"])
    assert text == "Ans"  # teacher scores the answer only, not the delimiter
    assert ids == [1, 2, 3]  # ids trimmed in lockstep with the text (no loss/count desync)
    # no trailing stop -> unchanged (ids normalized to a list)
    assert _trim_trailing_stop(_Tok(), [1, 2, 3], "Ans", ["</answer>"]) == ([1, 2, 3], "Ans")


def test_trim_trailing_stop_keeps_ids_and_text_synced_when_stop_starts_inside_token():
    """Regression (opd.py): when the stop delimiter starts INSIDE the final sampled token
    (that token decodes to "B</answer>"), the whole token is dropped from the kept ids — so returning
    completion_text[:keep_len] would keep a "B" the ids can no longer represent, desyncing the
    teacher-scored text from the student ids. The returned text must equal decode(kept ids)."""
    from flash.engine.worker.opd_gkd import _trim_trailing_stop

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            m = {1: "A", 4: "B</answer>"}  # id 4 fuses a real char with the stop delimiter
            return "".join(m[int(i)] for i in ids)

    ids, text = _trim_trailing_stop(_Tok(), [1, 4], "AB</answer>", ["</answer>"])
    # id 4 ("B</answer>") crosses the keep boundary -> excluded; text is what the KEPT ids decode to.
    assert ids == [1]
    assert text == "A"
    # ids/text stay consistent (the old code returned "AB", which the kept ids [1] cannot represent).
    assert text == _Tok().decode(ids)


def test_trim_trailing_stop_prefers_longest_overlapping_stop():
    r"""Regression (opd.py:150): with overlapping delimiters like ["\n", "\n\n"] listed
    shortest-first, a "\n\n" tail must have BOTH newlines trimmed (the longest/earliest matching stop),
    not just the first-listed "\n" — otherwise the teacher still scores a leftover delimiter newline."""
    from flash.engine.worker.opd_gkd import _trim_trailing_stop

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            m = {1: "h", 2: "i", 3: "\n", 4: "\n"}
            return "".join(m[int(i)] for i in ids)

    # completion "hi\n\n"; stops list the SHORTER "\n" first. Longest match "\n\n" -> keep "hi".
    ids, text = _trim_trailing_stop(_Tok(), [1, 2, 3, 4], "hi\n\n", ["\n", "\n\n"])
    assert text == "hi"  # both newlines gone, not just the first-listed one
    assert ids == [1, 2]
    # order-independent: same result when the longer stop is listed first
    assert _trim_trailing_stop(_Tok(), [1, 2, 3, 4], "hi\n\n", ["\n\n", "\n"]) == ([1, 2], "hi")


def test_stop_detection_and_trim_handle_special_token_delimiter():
    """Regression (opd.py): a [train] stop_sequence can be a tokenizer SPECIAL token (e.g.
    <|im_end|>). A skip_special_tokens=True decode STRIPS it, so the clean text no longer ends with the
    delimiter — _rollout_terminated would misclassify the rollout as truncated and _trim_trailing_stop
    would never remove it, skipping every usable sample for that config. Detection/trim must run on the
    special-tokens-INCLUDED decode."""
    from flash.engine.worker.opd_gkd import _rollout_terminated, _trim_trailing_stop

    IM_END = 9  # a special token; renders to "<|im_end|>" ONLY when specials are kept

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            answer = "".join({1: "4", 2: "2"}.get(int(i), "") for i in ids)
            if not skip_special_tokens and any(int(i) == IM_END for i in ids):
                return answer + "<|im_end|>"
            return answer

    ids = [1, 2, IM_END]
    stop_text = _Tok().decode(ids, skip_special_tokens=False)  # "42<|im_end|>"
    stops = ["<|im_end|>"]

    # The clean decode drops the delimiter, but the raw stop_text keeps it -> terminated, NOT truncated.
    assert _Tok().decode(ids, skip_special_tokens=True).endswith("<|im_end|>") is False
    assert _rollout_terminated(ids, stop_text, frozenset(), stops) is True
    # trim drops the special-token delimiter id; teacher/alignment text is the clean answer "42".
    kept_ids, text = _trim_trailing_stop(_Tok(), ids, stop_text, stops)
    assert kept_ids == [1, 2]
    assert text == "42"


def test_trim_trailing_stop_scans_from_end_not_quadratically():
    """Regression (opd.py:153): trimming the stop must scan from the END (a few decodes of
    the dropped tail), not decode every growing prefix ids[:1..n] — which was O(completion^2) and could
    dominate CPU before teacher scoring once [train].max_completion_tokens is raised. Assert decode is called only
    a bounded number of times, independent of completion length."""
    from flash.engine.worker.opd_gkd import _trim_trailing_stop

    class _Tok:
        def __init__(self):
            self.calls = 0

        def decode(self, ids, skip_special_tokens=True):
            self.calls += 1
            return "".join("abcdefghij"[int(x) % 10] for x in ids)  # 1 char/id, no split chars

    n = 500
    ids = list(range(n))
    text = "".join("abcdefghij"[i % 10] for i in ids)
    tok = _Tok()
    stop = text[-3:]  # a 3-char trailing delimiter (3 clean tokens)
    out_ids, out_text = _trim_trailing_stop(tok, ids, text, [stop])
    assert out_ids == ids[: n - 3]
    assert out_text == text[: n - 3]
    # ~4-5 decodes (drop 3 tail tokens + the satisfying check + the final return), NOT ~n. The old
    # forward loop decoded a growing prefix per token -> ~n calls.
    assert tok.calls <= 10, (
        f"decode called {tok.calls}x on a {n}-token completion -> quadratic trim"
    )


def test_rollout_terminated_requires_eos_or_stop_not_length():
    """A rollout is safe to distil only if it terminated NATURALLY — EOS in the ids, or (with
    stop_sequences) the decoded text ends with a stop delimiter. A max_new_tokens cap hit OR a
    gen_cfg.max_time cut ends without either and is a partial mid-output fragment OPD must skip (it
    can't supervise the stop token). Length is NOT the criterion."""
    from flash.engine.worker.opd_gkd import _rollout_terminated

    EOS = frozenset({99})
    # EOS in the ids -> terminated (HF appends EOS when it stops on it), regardless of length.
    assert _rollout_terminated([1, 2, 3, 99], "abc", EOS, ()) is True
    # no EOS, no stops -> NOT terminated: a cap hit OR a max_time cut, both partial fragments -> skip.
    assert _rollout_terminated([1, 2, 3, 4], "abcd", EOS, ()) is False  # cap hit, no EOS
    assert _rollout_terminated([1, 2], "ab", EOS, ()) is False  # short: max_time cut, no EOS/stop
    # A model with MULTIPLE eos ids (generation_config.eos_token_id is a list) stops on ANY member,
    # so a completion ending in a SECONDARY eos is terminated, not a truncation to skip.
    assert _rollout_terminated([1, 2, 88], "abc", frozenset({99, 88}), ()) is True
    # stop delimiter is the trailing text -> terminated even without EOS AND even at the cap (codex#587).
    assert _rollout_terminated([1, 2, 3, 4], "ans</answer>", frozenset(), ("</answer>",)) is True
    # stop configured but text doesn't end with it, no EOS -> not terminated -> skip.
    assert _rollout_terminated([1, 2, 3, 4], "ans", frozenset(), ("</answer>",)) is False
    # no termination signal at all (empty eos set, no stops) -> fail OPEN (distil, don't skip all).
    assert _rollout_terminated([1, 2, 3, 4], "abcd", frozenset(), ()) is True


def test_generation_eos_ids_unions_tokenizer_and_generation_config_lists():
    """_rollout_terminated must see EVERY halting id. _generation_eos_ids unions the tokenizer's
    eos_token_id with the model's generation_config/config eos_token_id, each of which HF allows to be
    a scalar OR a list — so a model that halts on a secondary eos id from a list while its tokenizer
    exposes a different primary eos gets both ids, and the rollout is not misread as truncated
    bool is an int subclass but never a token id, so it's excluded."""
    from flash.engine.worker.opd_gkd import _generation_eos_ids

    tok = SimpleNamespace(eos_token_id=2)
    # generation_config carries a LIST (primary + secondary); config repeats one — union dedups.
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[2, 73]),
        config=SimpleNamespace(eos_token_id=151645),
    )
    assert _generation_eos_ids(model, tok) == frozenset({2, 73, 151645})

    # Scalar-only tokenizer, model without generation config -> just the tokenizer id.
    assert _generation_eos_ids(SimpleNamespace(), SimpleNamespace(eos_token_id=5)) == frozenset({5})
    # Nothing defines an eos -> empty set (the fail-open signal for _rollout_terminated).
    assert _generation_eos_ids(SimpleNamespace(), SimpleNamespace()) == frozenset()
    # bool must not leak in as a token id (True == 1 would poison the set).
    assert _generation_eos_ids(SimpleNamespace(), SimpleNamespace(eos_token_id=True)) == frozenset()


def test_opd_vram_sizing_uses_completion_budget_not_sft_default():
    # OPD generates on-policy (loss forward runs model(prompt+completion)), so allocator sizing must
    # use the prompt+completion budget, not the SFT 1024 default — else a raised max_tokens OOMs an
    # under-sized GPU.
    from flash.engine.vram import opd_rollout_seq_len

    assert opd_rollout_seq_len(0, None, False) == 1536  # 1024 prompt + 512 completion default
    assert opd_rollout_seq_len(0, 8192, False) == 9216  # raised max_tokens sizes up (was 1024)
    assert opd_rollout_seq_len(4096, 8192, False) == 4096  # explicit max_length pins the sequence


def test_opd_selects_only_managed_parasail_aliases():
    from flash.schema import ConfigError, spec_from_dict

    def _spec(teacher):
        return spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-4B",
                "algorithm": "opd",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"epochs": 1, "max_examples": 5, "teacher_model": teacher},
            },
            run_id="x",
        )

    assert _spec("kimi-k3").train.teacher_model == "kimi-k3"
    assert _spec("GLM 5.2").train.teacher_model == "glm-5.2"
    assert _spec("qwen3.5-397b-a17b").train.teacher_model == "qwen3.5-397b-a17b"
    assert _spec("deepseek-v4-pro").train.teacher_model == "deepseek-v4-pro"
    assert _spec("DeepSeek V4 Pro").train.teacher_model == "deepseek-v4-pro"
    assert _spec("").train.teacher_model == ""

    rejected = (
        "kimi-k2.6",
        "deepseek-ai/DeepSeek-V4-Pro",
        "parasail-deepseek-v4-pro",
        "moonshotai/Kimi-K3",
        "nvidia/GLM-5.2-NVFP4",
        "Qwen/Qwen3.5-397B-A17B-FP8",
        "parasail-glm-52",
        "accounts/fireworks/models/glm-5p2",
    )
    for teacher in rejected:
        with pytest.raises(ConfigError, match="teacher_model"):
            _spec(teacher)


def test_opd_rejects_prompt_budget_at_parse_time_before_provisioning():
    """max_context_tokens <= max_completion_tokens leaves no prompt budget; opd must reject it at spec-parse time
    (before a paid worker is provisioned), not only inside run_opd after GPU setup."""
    from flash.schema import ConfigError, spec_from_dict

    def _spec(train_extra):
        return spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-4B",
                "algorithm": "opd",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"epochs": 1, "max_examples": 5, **train_extra},
            },
            run_id="x",
        )

    # max_context_tokens leaves room after an explicit max_completion_tokens -> ok.
    _spec({"max_context_tokens": 2048, "max_completion_tokens": 512})
    # max_context_tokens <= max_completion_tokens -> no prompt budget -> reject at parse.
    with pytest.raises(ConfigError, match="prompt budget"):
        _spec({"max_context_tokens": 400, "max_completion_tokens": 512})
    # max_completion_tokens omitted -> resolves to the opd recipe default (512); context below it -> reject.
    with pytest.raises(ConfigError, match="prompt budget"):
        _spec({"max_context_tokens": 256})


def test_opd_rejects_zero_kl_penalty_at_parse_time():
    from flash.schema import ConfigError, spec_from_dict

    def _spec(algorithm, train_extra):
        return spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-4B",
                "algorithm": algorithm,
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"epochs": 1, "max_examples": 5, **train_extra},
            },
            run_id="x",
        )

    with pytest.raises(ConfigError, match=r"kl_penalty_coef must be > 0 for opd"):
        _spec("opd", {"kl_penalty_coef": 0})

    assert _spec("opd", {}).train.kl_penalty_coef is None
    assert _spec("grpo", {"kl_penalty_coef": 0}).train.kl_penalty_coef == 0


@pytest.mark.parametrize("max_context_tokens", [256, 512])
def test_opd_accepts_short_hybrid_mamba_context_with_conditional_worker_floor(
    max_context_tokens,
):
    from flash.schema import spec_from_dict

    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "algorithm": "opd",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {
                "epochs": 1,
                "max_examples": 5,
                "max_context_tokens": max_context_tokens,
                "max_completion_tokens": 128,
            },
            # 35B opd trains the routed experts, so it no longer fits one card even at this short
            # context; the run needs two. the context floor under test is unchanged.
            "gpu": {"count": 2},
        },
        run_id="x",
    )

    assert spec.train.max_context_tokens == max_context_tokens


def test_opd_rejects_tool_environments(monkeypatch):
    """opd owns its vLLM rollout loop instead of TRL's native tool-call loop, so a tool-calling env
    must still fail fast. Pure multi-turn (episode) envs ARE supported now — see
    test_opd_multi_turn_distills_every_assistant_turn."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from flash.engine.worker import opd as opd_mod

    env = SimpleNamespace(is_tool_env=True)
    monkeypatch.setattr(
        opd_mod,
        "_w",
        SimpleNamespace(SEED=0, require_active_env=lambda e=env: e),
    )
    with pytest.raises(RuntimeError, match="tool-calling"):
        opd_mod.run_opd()


def test_opd_validates_dynamic_image_compatibility_before_gpu_wait():
    # The ordering invariant lives in the verl worker, which owns the whole OPD path; `run_opd` is a
    # one-line delegation to it. Asserting on the delegator's source would pass on any body at all,
    # so this follows the wiring to where the validation and the GPU probe actually sit. verl probes
    # the GPU in a subprocess rather than calling wait_for_gpu, but the invariant is the same one:
    # an incompatible model must fail before any paid GPU work starts.
    import inspect

    from flash.engine.worker.opd_train import run_opd_train

    source = inspect.getsource(run_opd_train)
    validation = 'validate_multimodal_training(model_id, "opd")'

    assert source.index(validation) < source.index("_probe_gpu_in_subprocess(")


class _CharTok:
    """A tiny char-level chat tokenizer for the multi-turn e2e test. Every char in a rendered/encoded
    string is one id (its index in ``alpha``); a reserved ``eos_token_id`` decodes to "" so a completion
    ending in it reads as a NATURAL termination and as a zero-width alignment token. ``apply_chat_template``
    inserts message content VERBATIM (``role:content|``), so make_env_glue's probe resolves — the
    load-bearing property real Qwen templates also have."""

    alpha = "uas:|gn42ok "

    def __init__(self):
        self.eos_token_id = len(self.alpha)
        self.pad_token_id = self.eos_token_id
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def _enc(self, text):
        return [self.alpha.index(c) for c in text]

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=self._enc(text))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.alpha[int(i)] for i in ids if int(i) != self.eos_token_id)

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ):
        text = "".join(f"{str(m.get('role', 'user'))[0]}:{m.get('content', '')}|" for m in messages)
        if add_generation_prompt:
            text += "a:"
        return text

    def save_pretrained(self, *a, **k):
        pass


def test_opd_filtering_stage_is_setup_not_training():
    """Regression (_poll.py): opd_filtering_prompts emits REAL progress heartbeats, so
    is_training_heartbeat would classify it as TRAINING (the tight, sticky stall window) mid-setup
    unless it's registered as a setup stage. It must be treated as cold-start setup."""
    from flash.providers._poll import SETUP_HEARTBEAT_STAGES, is_training_heartbeat

    assert "opd_filtering_prompts" in SETUP_HEARTBEAT_STAGES
    assert is_training_heartbeat("opd_filtering_prompts", 0) is False
    assert (
        is_training_heartbeat("opd_filtering_prompts", 5) is False
    )  # progress count doesn't flip it


def test_opd_preprocessing_stages_are_setup_on_the_provider_side():
    """Regression (_poll.py): the worker-side registry gained opd_prompt_scan and
    opd_image_prep, but SETUP_HEARTBEAT_STAGES kept only the retired opd_filtering_prompts. Both
    stages carry a progress callback, so is_training_heartbeat classified a run that was still
    preprocessing as TRAINING and judged it by the tight (sticky) stall window instead of the
    cold-start grace -- a large split is torn down as "stalled" before its first step."""
    from flash.providers._poll import SETUP_HEARTBEAT_STAGES, is_training_heartbeat

    for stage in ("opd_prompt_scan", "opd_image_prep"):
        assert stage in SETUP_HEARTBEAT_STAGES
        assert is_training_heartbeat(stage, 0) is False
        # these emit a REAL progress heartbeat per advance; a nonzero count must not flip them.
        assert is_training_heartbeat(stage, 5) is False


def test_opd_liveness_stages_are_throttled_at_setup_cadence():
    """opd liveness threads must use the throttled setup-liveness cadence."""
    from flash.engine.worker.heartbeat import _HB_SETUP_LIVENESS_STAGES, _HB_THROTTLED_STAGES

    opd_liveness_stages = {"opd_prompt_scan", "opd_image_prep", "opd_finalizing"}
    assert opd_liveness_stages <= _HB_SETUP_LIVENESS_STAGES
    assert opd_liveness_stages <= _HB_THROTTLED_STAGES

    assert "opd_filtering_prompts" in _HB_THROTTLED_STAGES
    assert "opd_filtering_prompts" in _HB_SETUP_LIVENESS_STAGES
    # parity with the sft pre-tokenize stage this mirrors (same dual membership).
    assert "sft_pretokenizing" in _HB_THROTTLED_STAGES
    assert "sft_pretokenizing" in _HB_SETUP_LIVENESS_STAGES


def test_liveness_heartbeat_merges_fields_into_every_emission(monkeypatch):
    """Regression (heartbeat.py): the liveness thread emits stage=<stage> with NO step,
    and because it shares the opd_step upload-throttle slot it can win the slot and overwrite the
    main thread's stepped heartbeat -- actual_steps_run then sees a training-stage heartbeat with no
    step and floors a cancelled run to 1 step. A `fields` callback must be merged into every emission
    so the step rides along on the liveness pings too."""
    import importlib
    import time

    # The worker package re-exports the `heartbeat` FUNCTION, shadowing the submodule name, so import
    # the module object explicitly rather than via attribute access.
    hb = importlib.import_module("flash.engine.worker.heartbeat")

    emitted: list[tuple[str, dict]] = []
    fake_w = SimpleNamespace(
        heartbeat=lambda stage, **kw: emitted.append((stage, kw)),
        _HB_LAST_PROGRESS_TS=0.0,
    )
    monkeypatch.setattr(hb, "_w", fake_w)
    monkeypatch.setattr(hb, "gpu_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(hb, "_LIVENESS_TICK_S", 0.001)

    with hb.liveness_heartbeat("opd_step", progress=lambda: 1, fields=lambda: {"step": 7}):
        deadline = time.time() + 2.0
        while not emitted and time.time() < deadline:
            time.sleep(0.005)
    assert emitted, "liveness thread never emitted a heartbeat"
    assert any(kw.get("step") == 7 for (s, kw) in emitted if s == "opd_step"), (
        f"fields must stamp the step onto opd_step liveness emissions; saw {emitted}"
    )


def test_opd_teacher_prompt_includes_thinking_prefill():
    """Regression (opd.py:93): in thinking mode the student template opens a reasoning
    block (e.g. <think>) AFTER the generation prompt and samples its completion after it. The teacher
    must condition on that SAME trailing prefill; the plain 'Assistant: ' prompt (empty prefill) would
    score every thinking-mode logprob against a prefix that never opened the block."""
    from flash.engine.worker import opd_gkd

    msgs = [{"role": "user", "content": "hi"}]
    # default (thinking off / no prefill) -> ends at the plain generation boundary.
    assert opd_gkd._teacher_prompt_text(msgs).endswith("Assistant: ")
    # with a prefill -> the teacher conditions on the exact text the student sampled after.
    assert opd_gkd._teacher_prompt_text(msgs, "<think>\n").endswith("Assistant: <think>\n")


def test_thinking_prefill_text_is_template_delta(monkeypatch):
    """Regression (opd.py): the thinking prefill is the DELTA a thinking-mode chat template
    opens after the generation prompt (enable_thinking True vs False). Empty when thinking is off (the
    plain teacher prompt already matches) or the template ignores enable_thinking."""
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, enable_thinking
        ):
            return "<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else "")

    monkeypatch.setattr(opd_mod, "_w", SimpleNamespace(THINKING=False))
    assert opd_mod._thinking_prefill_text(_Tok()) == ""  # thinking off -> no prefill
    monkeypatch.setattr(opd_mod, "_w", SimpleNamespace(THINKING=True))
    assert opd_mod._thinking_prefill_text(_Tok()) == "<think>\n"  # exact opened delta

    class _NoThinkTok:  # template that ignores enable_thinking -> renders identically -> empty delta
        def apply_chat_template(self, messages, **kw):
            return "<|im_start|>assistant\n"

    assert opd_mod._thinking_prefill_text(_NoThinkTok()) == ""


def test_thinking_prefill_derives_opener_from_hybrid_template(monkeypatch):
    """Regression (opd.py): _thinking_prefill_text must handle a HYBRID template where the
    thinking render is NOT a prefix-extension of the non-thinking render — the opener is inserted BEFORE
    shared trailing template text, so base is not a prefix of think. The old think.startswith(base) test
    returned "", dropping the opener the student pre-fills so the teacher scored reasoning tokens against
    the wrong prefix. The common prefix/suffix derivation must recover the opener from think's unique
    middle."""
    from flash.engine.worker import opd as opd_mod

    class _HybridTok:
        # non-thinking: no opener; thinking: inserts "<think>\n" BEFORE the shared "END" suffix, so
        # "A:\nEND" is NOT a prefix of "A:\n<think>\nEND".
        def apply_chat_template(self, messages, *, enable_thinking, **kw):
            return "A:\n<think>\nEND" if enable_thinking else "A:\nEND"

    monkeypatch.setattr(opd_mod, "_w", SimpleNamespace(THINKING=True))
    assert opd_mod._thinking_prefill_text(_HybridTok()) == "<think>\n"


def test_thinking_prefill_recovers_opener_from_closed_block_hybrid(monkeypatch):
    """Regression (opd.py): a HYBRID template that disables thinking by force-CLOSING
    the block — enable_thinking=False -> '...<think></think>\\n', enable_thinking=True -> '...<think>\\n'
    — shares '<think>' in BOTH renders, so the common-prefix delta eats it and the previous fix returned
    only '\\n'. The student still pre-fills '<think>\\n', so the teacher must condition on the full
    opener; recover it from base's closing tag."""
    from flash.engine.worker import opd as opd_mod

    class _ClosedBlockTok:
        def apply_chat_template(self, messages, *, enable_thinking, **kw):
            # both open <think>; non-thinking force-closes it with </think>.
            return "A:\n<think>\n" if enable_thinking else "A:\n<think></think>\n"

    monkeypatch.setattr(opd_mod, "_w", SimpleNamespace(THINKING=True))
    assert opd_mod._thinking_prefill_text(_ClosedBlockTok()) == "<think>\n"


def test_thinking_prefill_recovers_opener_from_whitespace_empty_block_hybrid(monkeypatch):
    """Regression (opd.py): a hybrid whose disabled render is an EMPTY block WITH whitespace
    (enable_thinking=False -> '...<think>\\n\\n</think>\\n', True -> '...<think>\\n') shares '<think>\\n'
    in the common prefix, so base's unique middle is '\\n</think>\\n' -- the closer behind a newline. The
    closed-block recovery must lstrip that intra-block whitespace before the '</' test, else it returns ''
    and thinking-mode OPD scores the student's reasoning against a teacher prompt that never opened
    <think>. The opener returned is still the thinking-side '<think>\\n'."""
    from flash.engine.worker import opd as opd_mod

    class _WhitespaceEmptyBlockTok:
        def apply_chat_template(self, messages, *, enable_thinking, **kw):
            # both open <think>\n; non-thinking force-closes with a whitespace-only empty block.
            return "A:\n<think>\n" if enable_thinking else "A:\n<think>\n\n</think>\n"

    monkeypatch.setattr(opd_mod, "_w", SimpleNamespace(THINKING=True))
    assert opd_mod._thinking_prefill_text(_WhitespaceEmptyBlockTok()) == "<think>\n"


def test_thinking_prefill_recovers_opener_when_closed_block_leaves_whitespace_remainder(
    monkeypatch,
):
    """Regression (opd.py:134): a closed-block hybrid whose disabled render closes IMMEDIATELY
    after the opener (enable_thinking=False -> '...<think></think>', True -> '...<think>\\n') shares only
    '<think>' in the common prefix, so think_mid is the NON-EMPTY whitespace remainder '\\n'. The old
    `if think_mid: return think_mid` early-return handed back '\\n' and skipped the closed-block recovery,
    conditioning the teacher on a prompt that opened but never continued <think>. The recovery must run
    FIRST and return the real thinking-side opener '<think>\\n'."""
    from flash.engine.worker import opd as opd_mod

    class _ClosedImmediatelyTok:
        def apply_chat_template(self, messages, *, enable_thinking, **kw):
            # thinking opens "<think>\n"; non-thinking force-closes right after the opener.
            return "A:\n<think>\n" if enable_thinking else "A:\n<think></think>"

    monkeypatch.setattr(opd_mod, "_w", SimpleNamespace(THINKING=True))
    assert opd_mod._thinking_prefill_text(_ClosedImmediatelyTok()) == "<think>\n"


def test_student_tokens_absorb_dropped_leading_space_sentencepiece():
    """Regression (opd.py:175): a SentencePiece/LLaMA tokenizer decodes a mid-completion
    word token IN ISOLATION without its leading word-boundary space (decode([▁world]) == 'world', not
    ' world'). prev + len(decode(window)) would then undercount that span by one char and drift every
    following offset, misassigning teacher spans to the wrong sampled ids. Offsets must be anchored to
    completion_text so the dropped space is absorbed into the token's start and spans stay contiguous
    and exact."""
    from flash.engine.worker.opd_gkd import student_tokens_with_offsets

    class _SPTok:  # decode of token 11 in isolation drops the leading space (SentencePiece behavior)
        def decode(self, ids, skip_special_tokens=True):
            m = {10: "hi", 11: "world"}  # 11 standalone -> "world", NOT " world"
            return "".join(m[int(x)] for x in ids)

    completion_text = "hi world"  # the ground-truth full decode (space at index 2)
    ids, toks = student_tokens_with_offsets(_SPTok(), [10, 11], completion_text)
    assert ids == [10, 11]
    assert (toks[0].start, toks[0].end) == (0, 2)  # "hi"
    # token 11 spans " world" [2,8): start pinned at prev (the dropped space is absorbed), not [2,7).
    assert (toks[1].start, toks[1].end) == (2, 8), (
        f"dropped leading space must be absorbed into the span; got {(toks[1].start, toks[1].end)}"
    )
    assert toks[-1].end == len(completion_text)  # no drift: spans cover the whole completion


def test_groupwise_alignment_cursor_walk_groups_denser_student_span():
    """Regression (tokenizer_align.py:73): the cursor walk that replaced the per-boundary
    rescan (O(C^2) -> O(S+T+B)) must still produce the coarsest common refinement — carrying a span's
    extra student tokens into the teacher-bearing span that closes it. Here the student tokenizes
    [0,3)+[3,6) where the teacher has one [0,6) token, so both student indices group under that
    teacher logprob; the tail [6,9) aligns 1:1."""
    from flash.engine.worker.tokenizer_align import groupwise_alignment

    student = [
        StudentToken(token_id=0, start=0, end=3),
        StudentToken(token_id=1, start=3, end=6),
        StudentToken(token_id=2, start=6, end=9),
    ]
    teacher = [
        TeacherToken(text="", logprob=-1.0, start=0, end=6),
        TeacherToken(text="", logprob=-2.0, start=6, end=9),
    ]
    assert groupwise_alignment(student, teacher) == [([0, 1], -1.0), ([2], -2.0)]


def test_opd_all_over_budget_prompts_fail_before_loading_student(monkeypatch):
    """Regression (opd.py): when every prompt exceeds the context budget the run fails
    deterministically — and that guard must fire BEFORE _student_model (which for a VL warm-start
    downloads the base and MERGES the SFT into it) AND before prefetch_model (the tens-of-GB base
    snapshot download), which is now deferred until after the pool is confirmed non-empty. Otherwise a
    misconfigured dataset pays for a full download + model load before failing. Trip if _student_model
    is reached, and assert prefetch_model was never called."""
    pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0

        def apply_chat_template(self, messages, **kw):
            return "PROMPT"

        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[1] * 100000)  # every prompt is over budget

    env = SimpleNamespace(
        dataset=lambda: [{"q": "a"}, {"q": "b"}],
        prompt_messages=lambda ex: [{"role": "user", "content": ex["q"]}],
    )
    prefetched: list = []
    fake_w = SimpleNamespace(
        require_active_env=lambda: env,
        JOB_SPEC=SimpleNamespace(
            train=SimpleNamespace(init_from_adapter=""),
            model="fake/model",
            gpu=SimpleNamespace(type=None),
        ),
        THINKING=False,
        SEED=0,
        OPD_RESUME_REVISION="",
        heartbeat=lambda stage, **kw: None,
        prefetch_model=lambda mid: (prefetched.append(mid), 0.0)[1],
    )
    monkeypatch.setattr(opd_mod, "_w", fake_w)
    monkeypatch.setattr(
        opd_mod,
        "_resolve_opd_knobs",
        lambda: opd_mod.OpdKnobs(
            teacher_model="glm-5.2",
            epochs=1,
            learning_rate=1e-4,
            temperature=0.0,
            top_p=1.0,
            max_completion=8,
            prompts_per_step=1,
            group_size=1,
            kl_coef=1.0,
            save_every=0,
            max_length=0,
            stop_sequences=(),
        ),
    )

    def _boom(*a, **k):
        raise AssertionError("_student_model was loaded before the all-over-budget guard fired")

    monkeypatch.setattr(opd_mod, "_student_model", _boom)
    monkeypatch.setattr(opd_mod, "wait_for_gpu", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "setup_perf_backends", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "gpu_diagnostics", lambda *a, **k: {})

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok())
    import flash.engine.worker.teacher as tmod

    monkeypatch.setattr(tmod, "TeacherClient", lambda *a, **k: object())
    monkeypatch.setenv("FLASH_CONTROL_PANEL_URL", "https://broker.example")
    monkeypatch.setenv("FLASH_TEACHER_CAPABILITY", "unit-test-teacher-capability")

    # The all-over-budget guard (RuntimeError) must fire; _student_model's AssertionError would
    # escape pytest.raises(RuntimeError) and fail the test (its "before the fix" behavior).
    with pytest.raises(RuntimeError, match="every prompt exceeds"):
        opd_mod.run_opd()
    # ...and the base-weight prefetch must have been deferred: an all-over-budget dataset fails
    # without paying for the tens-of-GB snapshot download.
    assert prefetched == [], "prefetch_model must not run when every prompt is over budget"


def test_opd_vram_reserves_dense_logits_unlike_fused_sft():
    """opd's gkd loss materializes dense logits (no fused CE), so its VRAM estimate must reserve the
    logits a >=3B SFT job fuses away — else a long-completion opd run is sized for a card that OOMs."""
    from flash.engine.vram import estimate_vram_gb

    kw = {"seq_len": 9216, "max_tokens": 8192, "vocab": 248_320, "lora_rank": 16}
    sft = estimate_vram_gb(4.0, "sft", "bf16", **kw)  # >=3B fuses CE -> 0 logits budgeted
    opd = estimate_vram_gb(4.0, "opd", "bf16", **kw)  # dense logits reserved (fwd + bwd)
    assert opd > sft + 10  # dense logits for opd vs 0 for fused SFT


def test_opd_vram_reserves_colocated_vllm_rollout_copy():
    """OPD student generation uses a resident vLLM engine, so VRAM includes a second weight/KV copy."""
    from flash.engine.vram import estimate_vram_gb

    kw = {"seq_len": 1536, "max_tokens": 512, "vocab": 248_320, "lora_rank": 16}
    grpo_without_vllm = estimate_vram_gb(4.0, "grpo", "bf16", use_vllm=False, **kw)
    opd_with_vllm = estimate_vram_gb(4.0, "opd", "bf16", use_vllm=True, **kw)
    opd_flag_ignored = estimate_vram_gb(4.0, "opd", "bf16", use_vllm=False, **kw)
    assert opd_with_vllm > grpo_without_vllm + 8.0  # second bf16 4B copy plus KV
    assert opd_flag_ignored == opd_with_vllm
    assert estimate_vram_gb(4.0, "opd", "bf16", **kw) == opd_with_vllm


def test_opd_vram_sizes_rollout_kv_for_full_prompt_batch():
    from flash.engine.vram import estimate_vram_gb, opd_rollout_concurrency

    assert opd_rollout_concurrency(8, 3) == 24
    kw = {"seq_len": 8192, "max_tokens": 512, "vocab": 128_000, "lora_rank": 16}
    one_prompt = estimate_vram_gb(4.0, "opd", "bf16", batch_size=1, group_size=1, **kw)
    eight_prompts = estimate_vram_gb(4.0, "opd", "bf16", batch_size=8, group_size=1, **kw)
    assert eight_prompts > one_prompt + 20.0


def test_model_required_vram_uses_opd_group_default_not_grpo_default():
    from flash.engine.vram import model_required_vram_gb

    train = {"max_length": 8192, "max_tokens": 512, "batch_size": 8, "lora_rank": 16}
    default_group = model_required_vram_gb("Qwen/Qwen3.5-4B", "opd", train=train, headroom=1.0)
    explicit_opd_default = model_required_vram_gb(
        "Qwen/Qwen3.5-4B", "opd", train={**train, "group_size": 1}, headroom=1.0
    )
    grpo_default_group = model_required_vram_gb(
        "Qwen/Qwen3.5-4B", "opd", train={**train, "group_size": 8}, headroom=1.0
    )

    assert default_group == explicit_opd_default
    assert grpo_default_group > default_group


def test_opd_35b_vllm_rollout_routes_past_any_single_card():
    """35B OPD with colocated student vLLM sizes past every single card once the experts train."""
    from flash.engine.vram import model_required_vram_gb

    need = model_required_vram_gb(
        "Qwen/Qwen3.6-35B-A3B",
        "opd",
        train={
            "max_length": 1536,
            "max_tokens": 512,
            "batch_size": 1,
            "group_size": 8,
            "lora_rank": 16,
        },
    )
    # the routed-expert adapter pushed this past the 180 GB B200, so the run needs two cards. the
    # upper bound keeps the estimate honest: it must not silently inflate past what two cards hold.
    assert 180 < need <= 2 * 180


def test_opd_35b_full_context_group1_is_rejected_because_it_only_fits_under_fp8():
    """A full-context group-1 35B OPD run does NOT fit any card, and must be rejected at parse time.

    This config used to route to the B200 on the strength of an fp8-KV discount. The 35B is a GDN
    hybrid and the OPD worker refuses fp8 KV for those (vllm's init_fp8_kv_scales crashes on the
    hybrid cache under verl's sleep/wake), so the run really allocates a bf16 cache. The discount
    was therefore admitting a run onto a card that cannot hold it, to OOM at rollout init on a paid
    GPU. Rejecting it during sizing is the correct outcome: the numbers below are exactly why.
    """
    import math

    from flash.catalog import MODELS, vocab_size_for
    from flash.engine.vram import estimate_vram_gb, model_required_vram_gb
    from flash.providers.allocator import vram_headroom
    from flash.providers.base import GPU_INFO, UnsupportedGpuError, cheapest_gpu

    moe = "Qwen/Qwen3.6-35B-A3B"
    info = MODELS[moe]
    train = {
        "max_context_tokens": 4096,
        "max_completion_tokens": 2048,
        "batch_size": 8,
        "group_size": 1,
        "lora_rank": 32,
    }
    need = model_required_vram_gb(moe, "opd", train=train, headroom=vram_headroom())
    biggest = max(g.vram_gb for g in GPU_INFO.values() if g.validated)
    assert need > biggest
    with pytest.raises(UnsupportedGpuError):
        cheapest_gpu(need)
    # the discount is what used to admit it: fp8 lands under the 180 gb b200, bf16 does not.
    kw = {
        "seq_len": 4096,
        "max_tokens": 2048,
        "batch_size": 8,
        "group_size": 1,
        "lora_rank": 32,
        "vocab": vocab_size_for(moe),
        "active_params_b": info.active_params_b,
    }
    fp8 = estimate_vram_gb(info.params_b, "opd", "bf16", fp8_kv=True, **kw)
    bf16 = estimate_vram_gb(info.params_b, "opd", "bf16", fp8_kv=False, **kw)
    hr = vram_headroom()
    assert fp8 * hr <= 180 < bf16 * hr
    # and the routed requirement must be the bf16 one, since that is the cache the worker allocates.
    assert need >= math.ceil(bf16 * hr)


def test_opd_fp8_kv_gate_does_not_downroute_below_the_fp8_ceiling():
    """The fp8-KV discount must apply only when a run can ONLY land on a modern (cc >= 8.9) card. A
    smaller OPD run that fits the 80 GB A100 (sm80, no fp8) must keep its bf16 KV sizing and its A100
    route — never dropping onto a card that would not actually use fp8 (and would then OOM)."""
    from flash.engine.vram import model_required_vram_gb
    from flash.providers.base import cheapest_gpu, max_non_fp8_kv_vram_gb, supports_fp8_kv

    train = {"max_completion_tokens": 128, "lora_rank": 32, "lora_alpha": 64}
    need = model_required_vram_gb("Qwen/Qwen3.5-2B", "opd", train=train, headroom=1.1)
    assert need <= max_non_fp8_kv_vram_gb()  # stays within the non-fp8 (<= 80 GB) band...
    assert not supports_fp8_kv(
        cheapest_gpu(need)
    )  # ...on the A100 (sm80), which does NOT use fp8 KV


def test_gdn_fp8_exclusion_survives_a_pinned_revision():
    """The GDN fp8 exclusion must key off the model, not off the sizing struct, which pinning nulls.

    Pinned GDN models still allocate bf16 KV because runtime rejects hybrid fp8 wake; reading only
    ``sizing_info`` under-reserves them. This exclusion is empirically validated and may drift.
    """
    import math
    from unittest import mock

    from flash.catalog import MODELS, vocab_size_for
    from flash.engine import vram as vram_module
    from flash.engine.vram import (
        _declares_linear_attention,
        estimate_vram_gb,
        model_required_vram_gb,
    )
    from flash.providers.allocator import vram_headroom

    # the guard itself, at the exact input the pinned path hands it.
    assert _declares_linear_attention(None, "Qwen/Qwen3.6-35B-A3B")
    # and it must not fire without evidence: no id, or an id the catalog does not route.
    assert not _declares_linear_attention(None, "")
    assert not _declares_linear_attention(None, "meta-llama/Llama-3.1-8B")

    # compare the pinned reservation with the bf16 estimate, not the unpinned total: generic sizing
    # can inflate both sides enough to hide an incorrectly applied fp8 discount.
    train = {
        "max_context_tokens": 1024,
        "max_completion_tokens": 512,
        "batch_size": 1,
        "group_size": 16,
        "lora_rank": 32,
    }
    gdn = "Qwen/Qwen3.6-35B-A3B"
    hr = vram_headroom()
    # generic architecture sizing (model_info=None) is what the pinned path uses, so the baseline
    # has to be computed the same way or it is not the number under test.
    bf16_need = math.ceil(
        estimate_vram_gb(
            MODELS[gdn].params_b,
            "opd",
            "bf16",
            seq_len=1024,
            max_tokens=512,
            batch_size=1,
            group_size=16,
            lora_rank=32,
            vocab=vocab_size_for(gdn),
            active_params_b=0.0,
            fp8_kv=False,
            model_info=None,
        )
        * hr
    )
    # a real pin resolves geometry over the network; stub it to the catalog's own numbers so the
    # only thing that varies is the nulled sizing_info -- which is the variable under test.
    with mock.patch.object(
        vram_module,
        "_validated_revision_geometry",
        lambda mid, rev, info: (info.params_b, info.vocab_size),
    ):
        pinned = model_required_vram_gb(
            gdn, "opd", train=train, headroom=hr, model_revision="a" * 40
        )
    assert pinned >= bf16_need


def test_opd_worker_fp8_kv_flag_matches_the_sizing_assumption():
    """The worker's fp8 flag must follow the cc probe AND the GDN exclusion, because vram.py's
    _opd_fp8_adjust sizes OPD against an fp8 KV pool above the non-fp8 card ceiling.

    Pins the flag itself rather than a VRAM number: this is the worker half of the pair whose sizing
    half is asserted by test_gdn_fp8_exclusion_survives_a_pinned_revision. GDN hybrids are excluded
    because vllm's fp8-kv wake path (init_fp8_kv_scales) crashes on the hybrid cache.
    """
    import inspect

    from flash.engine.worker import opd_train

    src = inspect.getsource(opd_train.run_opd_train)
    assert "model_is_gdn_hybrid(model_id, revision=model_revision)" in src
    assert "fp8_kv = _cc_ok and not gdn_hybrid" in src
    assert "get_device_capability() >= (8, 9)" in src

    # and the override is emitted only when the resolved flag is true, so a bf16 worker never sends
    # fp8 (an absent key means bf16, which is the conservative direction).
    assert 'if config.get("fp8_kv")' in inspect.getsource(opd_train.build_opd_overrides)


def test_opd_oversized_reject_names_the_knobs_to_shrink():
    """When even the biggest GPU can't hold an OPD run, the reject must be actionable: it names that
    OPD is resident-only (trainer + colocated vLLM student = two weight copies + rollout KV) and the
    knobs that shrink it, not the opaque 'no GPU that big' message the raw cheapest_gpu emits."""
    from flash.providers.base import UnsupportedGpuError, provisional_gpu

    train = {
        "max_context_tokens": 4096,
        "max_completion_tokens": 2048,
        "batch_size": 8,
        "group_size": 4,
    }
    with pytest.raises(UnsupportedGpuError) as exc:
        provisional_gpu("Qwen/Qwen3.6-35B-A3B", "opd", train=train)
    msg = str(exc.value)
    assert "resident-only" in msg
    assert "group_size" in msg
    assert "batch_size" in msg
    assert "max_completion_tokens" in msg


def test_opd_oversized_reject_reports_the_rentable_odd_ceiling(monkeypatch):
    """A ceiling of 5-7 buys four cards, so the diagnostic must never claim a fictitious shape."""
    from flash.providers.base import UnsupportedGpuError, provisional_gpu

    monkeypatch.setattr("flash.engine.vram.model_required_vram_gb", lambda *_a, **_k: 700)
    with pytest.raises(UnsupportedGpuError) as exc:
        provisional_gpu("Qwen/Qwen3.6-35B-A3B", "opd", gpu_count=7)
    msg = str(exc.value)
    assert "any 4-card validated GPU combination" in msg
    assert "7-card" not in msg
    assert "592.8 GB max" in msg


def test_opd_vram_keeps_chunked_text_peak_when_it_exceeds_dense_image_peak():
    """opd reserves the larger of one checkpointed text ce chunk and one dense image sample."""
    from flash.engine.vram import (
        _OPD_CE_PEAK_BYTES_PER_LOGIT,
        OPD_CE_CHUNK_SIZE,
        estimate_vram_gb,
    )

    kw = {
        "seq_len": 1,
        "max_tokens": OPD_CE_CHUNK_SIZE,
        "lora_rank": 16,
        "batch_size": 1,
        "group_size": 1,
    }
    v1, v2 = 100_000, 248_320
    delta = estimate_vram_gb(4.0, "opd", "bf16", vocab=v2, **kw) - estimate_vram_gb(
        4.0, "opd", "bf16", vocab=v1, **kw
    )
    expected = OPD_CE_CHUNK_SIZE * (v2 - v1) * _OPD_CE_PEAK_BYTES_PER_LOGIT / 1e9
    assert delta == pytest.approx(expected, rel=1e-9)


def test_opd_vram_dense_image_peak_grows_with_completion_budget():
    """the dense image fallback grows with the completion rows retained for its loss."""
    from flash.engine.vram import estimate_vram_gb

    kw = {"seq_len": 4096, "vocab": 248_320, "lora_rank": 16}
    non_think = estimate_vram_gb(4.0, "opd", "bf16", thinking=False, **kw)
    think = estimate_vram_gb(4.0, "opd", "bf16", thinking=True, **kw)
    assert think > non_think


def test_opd_vram_scales_to_loss_microbatch_not_full_batch():
    """OPD's dense-logit loss budget tracks the worker's loss microbatch.

    It should grow from one to four samples for <=10B models, then stop at the loss microbatch cap
    instead of scaling with the full prompt batch. The 35B path remains serial by default.
    """
    from flash.engine.vram import estimate_vram_gb

    kw = {"seq_len": 1024, "vocab": 248_320, "lora_rank": 16}
    opd_bs1 = estimate_vram_gb(4.0, "opd", "bf16", batch_size=1, group_size=1, **kw)
    opd_bs4 = estimate_vram_gb(4.0, "opd", "bf16", batch_size=4, group_size=1, **kw)
    opd_bs16 = estimate_vram_gb(4.0, "opd", "bf16", batch_size=16, group_size=1, **kw)
    assert opd_bs4 > opd_bs1
    assert opd_bs16 == opd_bs4  # capped at OPD_LOSS_MICROBATCH_SIZE, not full batch_size
    kw_35b = {"seq_len": 1024, "lora_rank": 16, "group_size": 1}
    v1, v2 = 100_000, 248_320
    opd_35b_delta_bs1 = estimate_vram_gb(
        35.0, "opd", "bf16", batch_size=1, vocab=v2, **kw_35b
    ) - estimate_vram_gb(35.0, "opd", "bf16", batch_size=1, vocab=v1, **kw_35b)
    opd_35b_delta_bs16 = estimate_vram_gb(
        35.0, "opd", "bf16", batch_size=16, vocab=v2, **kw_35b
    ) - estimate_vram_gb(35.0, "opd", "bf16", batch_size=16, vocab=v1, **kw_35b)
    assert opd_35b_delta_bs16 == pytest.approx(opd_35b_delta_bs1, rel=1e-9)
    # contrast: SFT DOES scale with the micro-batch when it is not floored by the dense-logits cap.
    sft_bs1 = estimate_vram_gb(
        4.0, "sft", "bf16", batch_size=1, seq_len=1024, vocab=1, lora_rank=16
    )
    sft_bs16 = estimate_vram_gb(
        4.0, "sft", "bf16", batch_size=16, seq_len=1024, vocab=1, lora_rank=16
    )
    assert sft_bs16 > sft_bs1


def test_opd_teacher_price_table_covers_exact_parasail_catalog():
    from flash.cost.facts import teacher_price_per_1m
    from flash.engine.recipe import TEACHER_MODELS

    assert teacher_price_per_1m("") == (1.40, 4.40)
    for alias, info in TEACHER_MODELS.items():
        assert teacher_price_per_1m(alias) == info.usd_per_1m
    assert teacher_price_per_1m("kimi-k3") == (3.00, 15.00)
    assert teacher_price_per_1m("qwen3.5-397b-a17b") == (0.50, 3.60)
    assert teacher_price_per_1m("deepseek-v4-pro") == (1.74, 3.48)
    with pytest.raises(ValueError, match="not a supported teacher"):
        teacher_price_per_1m("parasail-deepseek-v4-pro")


# --------------------------------------------------------------------------------------------------
# teacher client (mocked HTTP)
# --------------------------------------------------------------------------------------------------
def test_resolve_opd_knobs_rejects_zero_kl_penalty(monkeypatch):
    """Regression (opd.py:64): kl_penalty_coef scales the gkd objective, so an explicit 0
    (allowed by the shared schema for GRPO) makes every OPD backward a zero gradient while opt_steps
    still advances -> a fully-untrained adapter is published/charged. _resolve_opd_knobs must reject 0;
    omitting the field (None) still resolves to the positive recipe default."""
    from flash.engine.worker import opd as opd_mod

    class _Train:  # any [train] field not set returns None (falls back to the recipe default)
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, name):
            return None

    monkeypatch.setattr(
        opd_mod,
        "_w",
        SimpleNamespace(
            JOB_SPEC=SimpleNamespace(train=_Train(kl_penalty_coef=0.0)), THINKING=False
        ),
    )
    with pytest.raises(RuntimeError, match="kl_penalty_coef must be > 0"):
        opd_mod._resolve_opd_knobs()

    # unset (None) -> positive recipe default, no raise.
    monkeypatch.setattr(
        opd_mod,
        "_w",
        SimpleNamespace(
            JOB_SPEC=SimpleNamespace(train=_Train(kl_penalty_coef=None)), THINKING=False
        ),
    )
    assert opd_mod._resolve_opd_knobs().kl_coef > 0.0


def test_resolve_opd_knobs_maps_alias_to_parasail_model(monkeypatch):
    from flash.engine.worker import opd as opd_mod

    class _Train:  # any [train] field not set returns None (falls back to the recipe default)
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, name):
            return None

    def _knobs(teacher):
        monkeypatch.setattr(
            opd_mod,
            "_w",
            SimpleNamespace(
                JOB_SPEC=SimpleNamespace(train=_Train(teacher_model=teacher)), THINKING=False
            ),
        )
        return opd_mod._resolve_opd_knobs()

    assert _knobs("kimi-k3").teacher_model == "parasail-kimi-k3-fast"
    assert _knobs("qwen3.5-397b-a17b").teacher_model == "parasail-qwen35-397b-a17b"
    assert _knobs("deepseek-v4-pro").teacher_model == "parasail-deepseek-v4-pro"
    assert _knobs("").teacher_model == "parasail-glm-52"
    assert _knobs(None).teacher_model == "parasail-glm-52"
    for teacher in (
        "kimi-k2.6",
        "deepseek-ai/DeepSeek-V4-Pro",
        "Qwen/Qwen3.5-397B-A17B-FP8",
        "accounts/fireworks/models/deepseek-v4-pro",
    ):
        with pytest.raises(RuntimeError, match="teacher_model"):
            _knobs(teacher)


def test_groupwise_alignment_emits_no_empty_student_group():
    # Teacher covers [0,2) but the student's first token starts at char 2 (teacher-only leading
    # span). No group may have an empty student-index list.
    student = _student([(2, 3), (3, 5)])
    teacher = _teacher([(0, 2), (2, 5)])
    groups = groupwise_alignment(student, teacher)
    assert all(s_idx for s_idx, _ in groups)  # every group has >= 1 student token
    assert [s_idx for s_idx, _ in groups] == [[0, 1]]


def test_teacher_client_requires_capability():
    from flash.engine.worker.teacher import TeacherError

    with pytest.raises(TeacherError):
        TeacherClient("", "https://api.example/v1", "glm")


# --------------------------------------------------------------------------------------------------
# spec + cost plumbing
# --------------------------------------------------------------------------------------------------
def test_opd_spec_json_round_trip():
    from flash.schema import spec_from_dict
    from flash.spec import JobSpec

    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "opd",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {
                "epochs": 25,
                "max_examples": 8,
                "batch_size": 8,
            },
        },
        run_id="x",
    )
    restored = JobSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.phase == "opd"
    # the provider credential is control-plane-only and is never added to environment.secrets.
    assert "PARASAIL_API_KEY" not in restored.environment.secrets


def test_opd_cost_is_step_priced_and_bills_teacher_tokens():
    from flash.cost.spec import estimate_for_spec, spec_steps
    from flash.schema import spec_from_dict

    # No [train].max_examples set — opd falls back to one prompt batch per epoch for pricing.
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "opd",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 30},
        },
        run_id="x",
    )
    assert spec_steps(spec) == 30
    est = estimate_for_spec(spec)
    assert est.method == "opd"
    assert est.teacher_api_usd > 0.0  # external teacher token spend is itemized (diagnostic)
    # teacher tokens are billed by parasail to the platform-managed teacher key, tracked separately
    # from the gpu charge: total_usd is gpu (platform-billed) time only, never total + teacher.
    assert est.total_usd == pytest.approx(est.billable_hours * est.gpu_hourly_usd)
    assert "opd step" in " ".join(est.notes)


# --------------------------------------------------------------------------------------------------
# loss math (needs torch)
# --------------------------------------------------------------------------------------------------
class _TinyLM:
    """Minimal stand-in for a causal LM: per-position learnable logits, ignores the input ids."""

    def __init__(self, torch, T, V):
        self.w = torch.zeros(T, V, requires_grad=True)
        self.config = SimpleNamespace(use_cache=True)
        self.lm_head = torch.nn.Identity()
        self.input_embeddings = torch.nn.Identity()

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        logits_to_keep=0,
    ):
        B = input_ids.shape[0]
        T = input_ids.shape[1]
        hidden = self.w[:T].unsqueeze(0).expand(B, -1, -1)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return SimpleNamespace(logits=self.lm_head(hidden))

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.input_embeddings

    def parameters(self):
        return [self.w]

    def train(self, mode=True):  # _resolve_samples_batched flips the model into train mode
        return self


def test_opd_worker_rejects_images_before_gpu_or_teacher_use(monkeypatch):
    """Image-bearing OPD must fail before teacher construction or paid GPU work."""
    teacher_model = "glm-5.2"
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda _seed: None
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    import flash.engine.worker.teacher as teacher_mod
    from flash.engine.worker import opd_train as opd_mod

    env = SimpleNamespace(
        is_tool_env=False,
        multi_turn=False,
        dataset=lambda: [{"image": "dataset/red.png"}],
        prompt_messages=lambda _record: [{"role": "user", "content": [{"type": "image"}]}],
    )
    train = SimpleNamespace(
        init_from_adapter="",
        max_examples=1,
        teacher_model=teacher_model,
        epochs=1,
        temperature=None,
        save_at_steps=(),
        stop_sequences=(),
        structured_outputs="",
    )
    monkeypatch.setattr(
        opd_mod,
        "_w",
        SimpleNamespace(
            SEED=0,
            THINKING=False,
            require_active_env=lambda: env,
            JOB_SPEC=SimpleNamespace(
                train=train,
                model="Qwen/Qwen3.5-4B",
                model_revision="",
                model_policy="catalog",
                gpu=SimpleNamespace(type=None),
            ),
            heartbeat=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setattr(
        teacher_mod,
        "TeacherClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("teacher client must not be constructed")
        ),
    )
    monkeypatch.setattr(
        opd_mod,
        "_probe_gpu_in_subprocess",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("gpu allocation must not be reached")
        ),
    )

    # ValueError, not RuntimeError: TRL re-wrapped this as `RuntimeError(f"opd: {exc}")`, verl lets it
    # propagate. The type carries no behavioral difference -- `_worker_failure_flags` branches only on
    # RetriableInfraError/GitHubRateLimitError and CUDA OOM, so both are fatal and non-retriable.
    with pytest.raises(ValueError, match="image-bearing opd is not supported"):
        opd_mod.run_opd_train()

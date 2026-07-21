"""On-policy distillation (opd): groupwise reverse-KL (gkd) cross-tokenizer alignment, the teacher
client, spec/cost plumbing, and the loss math.

All CPU-only. The loss-math tests need torch and are skipped where it is unavailable (they run in
CI, which has the training stack).
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import types
from types import SimpleNamespace
from typing import ClassVar

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


def _skip(**k):
    """A resolve-sample stub whose every sample skips: no loss, teacher not reached."""
    from flash.engine.worker.opd import SampleResult

    return SampleResult()


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


def test_masking_then_prepare_normalizes_over_surviving_tokens_only():
    """After forced-group masking, the prepared loss inputs contain ONLY surviving-group tokens, so
    the downstream per-token mean normalizes over the kept (content) tokens -- dropping a fully-forced
    span re-normalizes the reverse-KL rather than leaving a shrunken sum over the original count."""
    from flash.engine.worker.opd import _drop_fully_forced_groups, _prepare_gkd_groups

    groups = [([0], -1.0), ([1, 2], -2.0), ([3], -3.0)]  # student tokens 0 and 3 fully-forced
    kept = _drop_fully_forced_groups(groups, (True, False, False, True))
    prepared = _prepare_gkd_groups(kept)
    # Tokens 0 and 3 are gone from BOTH the numerator and the mean's denominator (token_indices).
    assert prepared.token_indices == (1, 2)
    assert prepared.group_lengths == (2,)
    assert prepared.teacher_logsums == (-2.0,)


def test_masked_loss_equals_loss_without_the_forced_groups():
    """End-to-end normalization: the masked reverse-KL equals the loss computed as if the forced
    groups never existed -- the per-token mean re-normalizes over survivors, it is neither diluted by
    nor retains the dropped forced positions."""
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import _drop_fully_forced_groups, _gkd_loss_from_logps

    sp = torch.tensor([-0.5, -1.0, -1.5, -2.0], requires_grad=True)
    with_forced = [([0], -1.0), ([1, 2], -2.0), ([3], -3.0)]  # tokens 0 and 3 grammar-forced
    kept = _drop_fully_forced_groups(with_forced, (True, False, False, True))
    loss_masked = _gkd_loss_from_logps(sp, kept, kl_coef=0.25)
    loss_reference = _gkd_loss_from_logps(sp, [([1, 2], -2.0)], kl_coef=0.25)
    assert torch.allclose(loss_masked, loss_reference)


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


def test_opd_fresh_lora_keeps_causal_loader(monkeypatch):
    """Fresh OPD runs still use the lighter causal-LM loader."""
    from flash.engine.worker import opd as opd_mod

    calls = _install_student_loader_fakes(monkeypatch, vl_raises=True)
    fake_w = SimpleNamespace(
        is_vl_checkpoint=lambda model_id, revision="": False,
        _init_adapter_model=lambda model_id: (model_id, "fresh-lora-config"),
    )
    monkeypatch.setattr(opd_mod, "_w", fake_w)

    model, rollout_model_source = opd_mod._student_model(
        "Qwen/Qwen3.5-4B", {"dtype": "bf16"}, "cuda"
    )

    assert model == {"loader": "causal", "peft_config": "fresh-lora-config"}
    assert rollout_model_source == "Qwen/Qwen3.5-4B"
    assert calls[0] == (
        "causal",
        ("Qwen/Qwen3.5-4B",),
        {"trust_remote_code": True, "dtype": "bf16"},
    )
    assert ("to", "causal", "cuda") in calls
    assert not any(kind == "vl" for kind, *_ in calls)


def test_opd_fresh_student_forwards_model_revision(monkeypatch):
    from flash.engine.worker import opd as opd_mod

    calls = _install_student_loader_fakes(monkeypatch, vl_raises=True)
    fake_w = SimpleNamespace(
        JOB_SPEC=SimpleNamespace(model_revision="refs/pr/123"),
        is_vl_checkpoint=lambda model_id, revision="": False,
        _init_adapter_model=lambda model_id: (model_id, "fresh-lora-config"),
    )
    monkeypatch.setattr(opd_mod, "_w", fake_w)

    opd_mod._student_model("org/model", {"dtype": "bf16"}, "cuda")

    assert calls[0] == (
        "causal",
        ("org/model",),
        {"trust_remote_code": True, "dtype": "bf16", "revision": "refs/pr/123"},
    )


def test_opd_fresh_vl_lora_uses_multimodal_loader(monkeypatch):
    """Fresh OPD on a VL checkpoint should still train LoRA on the full multimodal tree."""
    from flash.engine.worker import opd as opd_mod

    calls = _install_student_loader_fakes(monkeypatch, causal_raises=True)
    fake_w = SimpleNamespace(
        is_vl_checkpoint=lambda model_id, revision="": True,
        _init_adapter_model=lambda model_id: (model_id, "fresh-lora-config"),
    )
    monkeypatch.setattr(opd_mod, "_w", fake_w)

    model, rollout_model_source = opd_mod._student_model(
        "Qwen/Qwen3.5-4B", {"dtype": "bf16"}, "cuda"
    )

    assert model == {"loader": "vl", "peft_config": "fresh-lora-config"}
    assert rollout_model_source == "Qwen/Qwen3.5-4B"
    assert calls[0] == (
        "vl",
        ("Qwen/Qwen3.5-4B",),
        {"trust_remote_code": True, "dtype": "bf16"},
    )
    assert ("to", "vl", "cuda") in calls
    assert not any(kind == "causal" for kind, *_ in calls)


def _patch_opd_run_vllm_stub(monkeypatch, opd_mod, *, sample_result=None, outputs=None):
    """Patch run_opd's mandatory vLLM engine with a CPU fake.

    When ``sample_result`` is supplied, the fake returns that ``SampleResult`` from the batched resolver
    after the vLLM-shaped generation/scoring phases have run.
    """

    monkeypatch.setattr(
        opd_mod,
        "_opd_vllm_kwargs",
        lambda *a, **k: {
            "gpu_memory_utilization": 0.10,
            "kv_cache_dtype": None,
            "max_num_batched_tokens": None,
            "attention_backend": None,
            "mm_encoder_attn_backend": None,
            "enforce_eager": None,
            "compilation_config": None,
        },
    )
    queued = list(outputs or [])

    class _FakeOpdVllmRolloutEngine:
        instances: ClassVar[list] = []

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.sync_count = 0
            self.closed = False
            _FakeOpdVllmRolloutEngine.instances.append(self)

        def sync_from_model(self, model):
            self.model = model
            self.sync_count += 1

        def generate(
            self,
            prompt_ids_batch,
            *,
            max_tokens,
            request_seeds=None,
            multi_modal_data_batch=None,
        ):
            self.request_seeds = list(request_seeds or [])
            out = []
            for _prompt_ids in prompt_ids_batch:
                if queued:
                    out.append(queued.pop(0))
                else:
                    out.append(opd_mod.OpdVllmOutput([3], "x", finish_reason="stop"))
            return out

        def close(self):
            self.closed = True

    monkeypatch.setattr(opd_mod, "OpdVllmRolloutEngine", _FakeOpdVllmRolloutEngine)
    if sample_result is not None:
        monkeypatch.setattr(
            opd_mod,
            "_score_one",
            lambda *a, **k: opd_mod._ScoreResult(teacher_toks=[], status="ok"),
        )

        def _resolve_samples_batched(
            model, tok, device, samples, knobs, microbatch, *, backward_scale=None, **_kwargs
        ):
            out = [
                sample_result(
                    model=model, tok=tok, device=device, prompt_ids=prompt_ids, knobs=knobs
                )
                for (_gen, _score, prompt_ids) in samples
            ]
            if backward_scale is not None:
                losses = [r.loss for r in out if r.loss is not None]
                if losses:
                    (sum(losses) * backward_scale).backward()
            return out

        monkeypatch.setattr(opd_mod, "_resolve_samples_batched", _resolve_samples_batched)
    return _FakeOpdVllmRolloutEngine


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
    from flash.engine.worker.opd import student_tokens_with_offsets

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
    """Regression (codex[bot], opd.py): a byte-level tokenizer can split one multi-byte char across
    two ids; the first decodes to U+FFFD until the second arrives. Measuring each id's decoded length
    independently gave one id the whole char and the other a ZERO-WIDTH span — dropping a real
    byte-token from the alignment and undercounting the char's student logprob. Both byte-ids must
    share the completed-char span so neither is dropped."""
    from flash.engine.worker.opd import student_tokens_with_offsets

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
    """Regression (cursor[bot], opd.py:120-126): the U+FFFD merge heuristic must NOT fire when a token
    LEGITIMATELY decodes to the replacement glyph (the model actually emitted U+FFFD as content). Such
    a token is already reflected in completion_text, so decode(prefix) is a prefix of it — the loop
    must stop and keep it as its own span instead of swallowing the following token."""
    from flash.engine.worker.opd import student_tokens_with_offsets

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
    """Regression (codex[bot], opd.py): offsets must be built by decoding a SMALL window per step, not
    the whole growing prefix ids[:i+1] (which was O(len^2) and dominated CPU on long completions).
    Assert the longest id-slice handed to tok.decode stays bounded regardless of completion length."""
    from flash.engine.worker.opd import student_tokens_with_offsets

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
    from flash.engine.worker.opd import _trim_trailing_stop

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
    """Regression (codex[bot], opd.py): when the stop delimiter starts INSIDE the final sampled token
    (that token decodes to "B</answer>"), the whole token is dropped from the kept ids — so returning
    completion_text[:keep_len] would keep a "B" the ids can no longer represent, desyncing the
    teacher-scored text from the student ids. The returned text must equal decode(kept ids)."""
    from flash.engine.worker.opd import _trim_trailing_stop

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
    r"""Regression (codex[bot], opd.py:150): with overlapping delimiters like ["\n", "\n\n"] listed
    shortest-first, a "\n\n" tail must have BOTH newlines trimmed (the longest/earliest matching stop),
    not just the first-listed "\n" — otherwise the teacher still scores a leftover delimiter newline."""
    from flash.engine.worker.opd import _trim_trailing_stop

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
    """Regression (codex[bot], opd.py): a [train] stop_sequence can be a tokenizer SPECIAL token (e.g.
    <|im_end|>). A skip_special_tokens=True decode STRIPS it, so the clean text no longer ends with the
    delimiter — _rollout_terminated would misclassify the rollout as truncated and _trim_trailing_stop
    would never remove it, skipping every usable sample for that config. Detection/trim must run on the
    special-tokens-INCLUDED decode."""
    from flash.engine.worker.opd import _rollout_terminated, _trim_trailing_stop

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
    """Regression (codex[bot], opd.py:153): trimming the stop must scan from the END (a few decodes of
    the dropped tail), not decode every growing prefix ids[:1..n] — which was O(completion^2) and could
    dominate CPU before teacher scoring once [train].max_completion_tokens is raised. Assert decode is called only
    a bounded number of times, independent of completion length."""
    from flash.engine.worker.opd import _trim_trailing_stop

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
    can't supervise the stop token). Length is NOT the criterion (codex[bot])."""
    from flash.engine.worker.opd import _rollout_terminated

    EOS = frozenset({99})
    # EOS in the ids -> terminated (HF appends EOS when it stops on it), regardless of length.
    assert _rollout_terminated([1, 2, 3, 99], "abc", EOS, ()) is True
    # no EOS, no stops -> NOT terminated: a cap hit OR a max_time cut, both partial fragments -> skip.
    assert _rollout_terminated([1, 2, 3, 4], "abcd", EOS, ()) is False  # cap hit, no EOS
    assert _rollout_terminated([1, 2], "ab", EOS, ()) is False  # short: max_time cut, no EOS/stop
    # A model with MULTIPLE eos ids (generation_config.eos_token_id is a list) stops on ANY member, so
    # a completion ending in a SECONDARY eos is terminated, not a truncation to skip (codex[bot]).
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
    a scalar OR a list — so a model like MiniCPM5 that halts on a secondary <|im_end|> (a list member)
    while its primary eos is </s> gets both ids, and a secondary-eos rollout is not misread as truncated
    (codex[bot]). bool is an int subclass but never a token id, so it's excluded."""
    from flash.engine.worker.opd import _generation_eos_ids

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


def test_opd_selects_managed_teacher_and_rejects_unknown():
    """[train].teacher_model selects the managed teacher from a fixed allow-list: a supported alias
    (or the raw Fireworks id, or a spaced/mixed-case form) parses and is stored as its canonical
    Fireworks model id; an unsupported teacher is rejected at PARSE time (before a paid GPU)."""
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

    # Supported aliases parse and are stored as the canonical Fireworks model id.
    assert _spec("kimi-k2.6").train.teacher_model == "accounts/fireworks/models/kimi-k2p6"
    assert (
        _spec("deepseek-v4-pro").train.teacher_model == "accounts/fireworks/models/deepseek-v4-pro"
    )
    # A spaced / mixed-case form normalizes to the same model id.
    assert _spec("GLM 5.2").train.teacher_model == "accounts/fireworks/models/glm-5p2"
    # The raw Fireworks model id is also accepted (identity), including with stray surrounding
    # whitespace (stripped like the alias branch).
    assert (
        _spec("accounts/fireworks/models/glm-5p2").train.teacher_model
        == "accounts/fireworks/models/glm-5p2"
    )
    assert (
        _spec("  accounts/fireworks/models/glm-5p2  ").train.teacher_model
        == "accounts/fireworks/models/glm-5p2"
    )
    # Omitting/blank leaves it unset ("" => the worker uses the default GLM 5.2 teacher).
    assert _spec("").train.teacher_model == ""

    # An unsupported teacher is rejected at parse time with a teacher-specific ConfigError.
    with pytest.raises(ConfigError, match="teacher_model"):
        _spec("gpt-5.5")
    # qwen-3.7-max (on-demand only) and minimax-m3 (serverless chat, but its /completions echo
    # endpoint OPD needs does not respond) are NOT allow-listed teachers, so both are rejected.
    with pytest.raises(ConfigError, match="teacher_model"):
        _spec("qwen-3.7-max")
    with pytest.raises(ConfigError, match="teacher_model"):
        _spec("minimax-m3")


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
                "train": {"epochs": 1, "max_examples": 5, "hf_repo": "owner/runs", **train_extra},
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
        },
        run_id="x",
    )

    assert spec.train.max_context_tokens == max_context_tokens


def test_all_skip_step_emits_stall_refresh_opd_step_heartbeat(monkeypatch):
    """Regression (codex[bot], opd.py:380-381): when EVERY sample in a step skips (empty completion
    / no teacher signal, or an over-budget re-render), the per-sample SUCCESS ping is never reached.
    Without a skip-path ping the step would emit only liveness heartbeats — which the pollers ignore
    — so a prolonged all-skip stretch on a later step could be reaped as stalled. Assert the skip path
    emits a NON-liveness opd_step heartbeat, and that it reports step==opt_steps (==0 while the first
    step is still accumulating) so it keeps the WIDE setup grace rather than flipping to the tight
    training window (opd_step is step-gated in the poller)."""
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    beats: list[tuple[str, dict]] = []

    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0

        def apply_chat_template(self, messages, **kw):
            return "PROMPT"

        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[1, 2])  # 2 tokens, well within budget

        def decode(self, ids, skip_special_tokens=True):
            return "".join("x" for _ in ids)

    class _Model(_TinyLM):
        def __init__(self):
            super().__init__(torch, T=4, V=8)
            self.config = SimpleNamespace(use_cache=False)

    env = SimpleNamespace(
        dataset=lambda: [{"q": "a"}, {"q": "b"}],
        prompt_messages=lambda ex: [{"role": "user", "content": ex["q"]}],
    )
    fake_w = SimpleNamespace(
        require_active_env=lambda: env,
        JOB_SPEC=SimpleNamespace(
            train=SimpleNamespace(init_from_adapter=""),
            model="fake/model",
            gpu=SimpleNamespace(type=None, exact_type=""),
        ),
        THINKING=False,
        SEED=0,
        OPD_RESUME_REVISION="",
        heartbeat=lambda stage, **kw: beats.append((stage, kw)),
        prefetch_model=lambda mid: 0.0,
        hf_resume_checkpoint=lambda **_kwargs: "",
        publish_deployable_checkpoint=lambda *a, **k: None,
        hf_upload_folder=lambda *a, **k: None,
        write_train_meta=lambda **k: None,
        wandb_report_to=lambda: [],  # W&B off by default in unit tests
        wandb_run_info=lambda: {},
    )
    monkeypatch.setattr(opd_mod, "_w", fake_w)
    # Deterministic knobs: 1 step / 1 prompt / group 1 -> a single sample, forced to skip.
    monkeypatch.setattr(
        opd_mod,
        "_resolve_opd_knobs",
        lambda: opd_mod.OpdKnobs(
            teacher_model="accounts/fireworks/models/glm-5p2",
            teacher_base_url="http://teacher.invalid",
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
    monkeypatch.setattr(opd_mod, "_student_model", lambda *a, **k: (_Model(), "fake/model"))
    monkeypatch.setattr(opd_mod, "wait_for_gpu", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "setup_perf_backends", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "grad_checkpointing_on", lambda *a, **k: False)
    monkeypatch.setattr(opd_mod, "gpu_diagnostics", lambda *a, **k: {})
    _patch_opd_run_vllm_stub(monkeypatch, opd_mod, sample_result=_skip)  # EVERY sample skips

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok())
    import flash.engine.worker.teacher as tmod

    monkeypatch.setattr(tmod, "TeacherClient", lambda *a, **k: object())
    monkeypatch.setenv("FIREWORKS_API_KEY", "unit-test-teacher-key")

    # An all-skip run lands no optimizer step, so run_opd raises its no-trained-step guard AFTER the
    # loop; we assert on the heartbeats captured before that raise.
    with pytest.raises(RuntimeError, match="no trained step"):
        opd_mod.run_opd()

    # Per-sample opd_step pings carry samples_done (liveness pings never do), so this isolates the
    # skip-path progress ping from any liveness heartbeat.
    per_sample = [kw for (stage, kw) in beats if stage == "opd_step" and "samples_done" in kw]
    assert per_sample, (
        "an all-skip step emitted no per-sample opd_step ping -> stall clock unrefreshed"
    )
    assert all(kw.get("step") == 0 for kw in per_sample), (
        "skip-path ping must report opt_steps (0 during the first, still-accumulating step) so the "
        "poller keeps the wide setup grace instead of the tight training window"
    )


def _opd_harness(
    monkeypatch,
    *,
    sample_result,
    beats=None,
    liveness=None,
    epochs=1,
    group=1,
    stop_sequences=(),
    structured_outputs="",
    metas=None,
    outputs=None,
    save_every=0,
    save_at_steps=(),
    env=None,
    teacher_model="accounts/fireworks/models/glm-5p2",
):
    """Wire run_opd's fakes (torch student, tokenizer, teacher, deterministic knobs) for a 1-prompt
    loop and install the caller's sample stub behind the mandatory vLLM rollout. Returns the opd
    module."""
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0

        def apply_chat_template(self, messages, **kw):
            return "PROMPT"

        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[1, 2])

        def decode(self, ids, skip_special_tokens=True):
            return "".join("x" for _ in ids)

    class _Model(_TinyLM):
        def __init__(self):
            super().__init__(torch, T=4, V=8)
            self.config = SimpleNamespace(use_cache=False)

    if env is None:
        env = SimpleNamespace(
            dataset=lambda: [{"q": "a"}],
            prompt_messages=lambda ex: [{"role": "user", "content": ex["q"]}],
        )
    fake_w = SimpleNamespace(
        require_active_env=lambda: env,
        JOB_SPEC=SimpleNamespace(
            train=SimpleNamespace(init_from_adapter=""),
            model="fake/model",
            # exact_type mirrors the real jobspec.gpu attribute run_opd reads at startup, so the
            # shared harness can drive run_opd end-to-end (e.g. the sample-completion capture test).
            gpu=SimpleNamespace(type=None, exact_type=""),
        ),
        THINKING=False,
        SEED=0,
        OPD_RESUME_REVISION="",
        publish_opd_optimizer_start_marker=lambda: None,
        heartbeat=(
            (lambda stage, **kw: beats.append((stage, kw)))
            if beats is not None
            else (lambda stage, **kw: None)
        ),
        prefetch_model=lambda mid: 0.0,
        hf_resume_checkpoint=lambda **_kwargs: "",
        publish_deployable_checkpoint=lambda *a, **k: None,
        hf_upload_folder=lambda *a, **k: None,
        write_train_meta=(
            (lambda **k: metas.append(k)) if metas is not None else (lambda **k: None)
        ),
        wandb_report_to=lambda: [],  # W&B off by default in unit tests
        wandb_run_info=lambda: {},
    )
    monkeypatch.setattr(opd_mod, "_w", fake_w)
    monkeypatch.setattr(
        opd_mod,
        "_resolve_opd_knobs",
        lambda: opd_mod.OpdKnobs(
            teacher_model=teacher_model,
            teacher_base_url="http://teacher.invalid",
            epochs=epochs,
            learning_rate=1e-4,
            temperature=0.0,
            top_p=1.0,
            max_completion=8,
            prompts_per_step=1,
            group_size=group,
            kl_coef=1.0,
            save_every=save_every,
            save_at_steps=tuple(save_at_steps),
            max_length=0,
            stop_sequences=stop_sequences,
            structured_outputs=structured_outputs,
        ),
    )
    monkeypatch.setattr(opd_mod, "_student_model", lambda *a, **k: (_Model(), "fake/model"))
    monkeypatch.setattr(opd_mod, "_save_adapter", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "_save_opd_resume_checkpoint", lambda **kwargs: None)
    monkeypatch.setattr(opd_mod, "wait_for_gpu", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "setup_perf_backends", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "grad_checkpointing_on", lambda *a, **k: False)
    monkeypatch.setattr(opd_mod, "gpu_diagnostics", lambda *a, **k: {})
    _patch_opd_run_vllm_stub(
        monkeypatch,
        opd_mod,
        sample_result=sample_result,
        outputs=outputs,
    )
    if liveness is not None:
        monkeypatch.setattr(opd_mod, "liveness_heartbeat", liveness)
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok())
    import flash.engine.worker.teacher as tmod

    class _TeacherTokenBatch(list):
        input_tokens = 3

    class _Teacher:
        def score_many_multimodal(self, items):
            return [_TeacherTokenBatch() for _item in items]

    monkeypatch.setattr(tmod, "TeacherClient", lambda *a, **k: _Teacher())
    monkeypatch.setenv("FIREWORKS_API_KEY", "unit-test-teacher-key")
    return opd_mod


def test_opd_train_meta_reports_truncated_rollouts_without_special_diagnostics(monkeypatch):
    calls = 0

    def _trained_sample(*, model, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return opd_mod.SampleResult(
                truncated=True,
                gen_tokens=1,
                skip_reason="truncated_rollout",
            )
        return opd_mod.SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    metas = []
    opd_mod = _opd_harness(monkeypatch, sample_result=_trained_sample, metas=metas)

    opd_mod.run_opd()

    notes = metas[-1]["notes"]
    assert notes["truncated_rollouts"] == 1
    assert "mean_eos_logprob" not in notes
    assert "final_empty_rate_ema" not in notes
    assert "final_truncation_rate_ema" not in notes


def test_opd_truncated_rollouts_bypass_teacher_and_gkd(monkeypatch, capsys):
    from flash.engine.worker import opd as opd_mod

    outputs = [
        opd_mod.OpdVllmOutput([3], "x", finish_reason="length") for _ in range(13)
    ]
    opd_mod = _opd_harness(
        monkeypatch,
        sample_result=None,
        outputs=outputs,
    )
    calls = {"teacher": 0, "gkd": 0}

    def _score_many(_teacher, pendings, **_kwargs):
        calls["teacher"] += 1
        return [opd_mod._ScoreResult(teacher_toks=[], status="ok") for _ in pendings]

    original_resolve = opd_mod._resolve_samples_batched

    def _resolve_samples_batched(*args, **kwargs):
        calls["gkd"] += 1
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(opd_mod, "_score_many", _score_many)
    monkeypatch.setattr(opd_mod, "_resolve_samples_batched", _resolve_samples_batched)

    with pytest.raises(RuntimeError, match="no trained step"):
        opd_mod.run_opd()

    assert calls == {"teacher": 0, "gkd": 0}
    assert "truncated_rollout=1" in capsys.readouterr().out


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


def test_opd_accepts_single_turn_image_prompts_in_cached_filter_render(monkeypatch):
    torch = pytest.importorskip("torch")
    image_module = pytest.importorskip("PIL.Image")
    transformers = pytest.importorskip("transformers")
    import flash.engine.worker.teacher as teacher_mod
    from flash.engine.worker import opd as opd_mod

    events = []
    processor_calls = []

    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return "PROMPT"

        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[1, 2])

        def convert_tokens_to_ids(self, token):
            assert token == "<|image_pad|>"
            return 99

    class _Processor:
        tokenizer = _Tok()

        def apply_chat_template(self, **kwargs):
            processor_calls.append(kwargs)
            return {
                "input_ids": torch.tensor([[1, 99, 99, 2]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "pixel_values": torch.ones((4, 3)),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]]),
            }

    image = image_module.new("RGB", (2, 2), "red")
    env = SimpleNamespace(
        is_tool_env=False,
        multi_turn=False,
        package_root=None,
        dataset=lambda: events.append("dataset")
        or [{"input": "describe", "image": image}],
        prompt_messages=lambda _record: events.append("prompt_messages")
        or [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image"},
                ],
            }
        ],
    )
    fake_w = SimpleNamespace(
        require_active_env=lambda: events.append("require_active_env") or env,
        JOB_SPEC=SimpleNamespace(
            train=SimpleNamespace(init_from_adapter="", max_examples=1),
            model="Qwen/Qwen3.5-4B",
            model_revision="",
            gpu=SimpleNamespace(type=None, exact_type=""),
        ),
        THINKING=False,
        SEED=7,
        heartbeat=lambda stage, **kwargs: events.append(stage),
        prefetch_model=lambda *args, **kwargs: events.append("prefetch_model") or 0.0,
    )
    monkeypatch.setattr(opd_mod, "_w", fake_w)
    monkeypatch.setattr(
        opd_mod, "seed_training_rngs", lambda seed: events.append(f"seed:{seed}")
    )
    monkeypatch.setattr(
        opd_mod,
        "_resolve_opd_knobs",
        lambda: opd_mod.OpdKnobs(
            teacher_model="accounts/fireworks/models/kimi-k2p6",
            teacher_base_url="http://teacher.invalid",
            epochs=1,
            learning_rate=1e-4,
            temperature=0.0,
            top_p=1.0,
            max_completion=8,
            prompts_per_step=1,
            group_size=1,
        ),
    )
    monkeypatch.setattr(teacher_mod, "TeacherClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(opd_mod, "wait_for_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(opd_mod, "setup_perf_backends", lambda: None)
    monkeypatch.setattr(opd_mod, "gpu_diagnostics", lambda: {})
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda: None)
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: _Processor(),
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "unit-test-teacher-key")

    def fail_student(*args, **kwargs):
        raise AssertionError("image prompt reached student model loading")

    monkeypatch.setattr(opd_mod, "_student_model", fail_student)

    with pytest.raises(AssertionError, match="reached student model loading"):
        opd_mod.run_opd()

    assert events[:3] == ["seed:7", "require_active_env", "dataset"]
    assert events.count("prompt_messages") == 1
    assert events.count("prefetch_model") == 1
    assert len(processor_calls) == 1
    assert processor_calls[0]["return_tensors"] == "pt"
    assert set(opd_mod._PromptRecord.__dataclass_fields__) == {
        "example",
        "student_messages",
        "teacher_messages",
        "prompt_ids",
        "rollout_prompt_ids",
        "descriptors",
    }


def test_opd_validates_dynamic_image_compatibility_before_gpu_wait():
    import inspect

    from flash.engine.worker import opd as opd_mod

    source = inspect.getsource(opd_mod.run_opd)
    validation = 'validate_multimodal_training(model_id, "opd", multi_turn=multi_turn)'

    assert source.index(validation) < source.index("wait_for_gpu(")


@pytest.mark.parametrize(
    "teacher_model",
    [
        "",
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/deepseek-v4-pro",
    ],
)
def test_opd_worker_rejects_nonvision_teacher_before_gpu_or_teacher_use(
    monkeypatch, teacher_model
):
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda _seed: None
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    import flash.engine.worker.teacher as teacher_mod
    from flash.engine.worker import opd as opd_mod

    env = SimpleNamespace(
        is_tool_env=False,
        multi_turn=False,
        dataset=lambda: [{"image": "dataset/red.png"}],
        prompt_messages=lambda _record: [
            {"role": "user", "content": [{"type": "image"}]}
        ],
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
                gpu=SimpleNamespace(type=None, exact_type=""),
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
        "wait_for_gpu",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("gpu allocation must not be reached")
        ),
    )

    with pytest.raises(RuntimeError, match=r"requires .*kimi-k2\.6"):
        opd_mod.run_opd()


def test_opd_image_deployable_save_uses_full_processor(monkeypatch):
    torch = pytest.importorskip("torch")
    image_module = pytest.importorskip("PIL.Image")
    transformers = pytest.importorskip("transformers")

    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return "PROMPT"

        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[1, 2])

        def convert_tokens_to_ids(self, token):
            assert token == "<|image_pad|>"
            return 99

        def decode(self, ids, skip_special_tokens=True):
            return "".join("x" for _ in ids)

    class _Processor:
        def __init__(self):
            self.tokenizer = _Tok()

        def apply_chat_template(self, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 99, 99, 2]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "pixel_values": torch.ones((4, 3)),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }

    image = image_module.new("RGB", (2, 2), "red")
    env = SimpleNamespace(
        is_tool_env=False,
        multi_turn=False,
        package_root=None,
        dataset=lambda: [{"input": "describe", "image": image}],
        prompt_messages=lambda _record: [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image"},
                ],
            }
        ],
    )

    def one_update(*, model, **_kwargs):
        return opd_mod.SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    processor = _Processor()
    opd_mod = _opd_harness(
        monkeypatch,
        sample_result=one_update,
        env=env,
        teacher_model="accounts/fireworks/models/kimi-k2p6",
    )
    opd_mod._w.JOB_SPEC.model = "Qwen/Qwen3.5-4B"
    opd_mod._w.JOB_SPEC.model_revision = ""
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: processor,
    )
    saved_processing_classes = []
    monkeypatch.setattr(
        opd_mod,
        "_save_adapter",
        lambda _model, processing_class, _adapter_dir: saved_processing_classes.append(
            processing_class
        ),
    )

    opd_mod.run_opd()

    rollout = opd_mod.OpdVllmRolloutEngine.instances[-1]
    assert rollout.image_pad_token_id == 99
    assert saved_processing_classes == [processor]
    assert saved_processing_classes[0] is not processor.tokenizer


def test_opd_image_prompt_fingerprint_is_json_safe_for_pil_and_bytes():
    image_module = pytest.importorskip("PIL.Image")
    from flash.engine.worker import opd as opd_mod
    from flash.multimodal import normalize_image_source

    pil_image = image_module.new("RGB", (2, 2), "red")
    encoded = io.BytesIO()
    pil_image.save(encoded, format="PNG")
    image_bytes = encoded.getvalue()
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    teacher_messages = [{"role": "user", "content": "describe"}]

    records = []
    for example in (
        {"id": "pil", "metadata": {"split": "train"}, "image": pil_image},
        {"id": "bytes", "metadata": {"split": "train"}, "image": image_bytes},
    ):
        descriptor = normalize_image_source(example["image"], None)
        records.append(
            opd_mod._PromptRecord(
                example=example,
                student_messages=messages,
                teacher_messages=teacher_messages,
                prompt_ids=[1, 99, 99, 2],
                rollout_prompt_ids=[1, 99, 2],
                descriptors=(descriptor,),
            )
        )

    first = opd_mod._opd_prompt_pool_fingerprint(records)
    second = opd_mod._opd_prompt_pool_fingerprint(records)

    assert first == second
    assert len(first) == 64


def test_opd_image_prompt_fingerprint_uses_descriptors_and_teacher_messages():
    from flash.engine.worker import opd as opd_mod

    base = {
        "example": {"id": 1, "image": b"raw-image-placeholder"},
        "student_messages": [{"role": "user", "content": [{"type": "image"}]}],
        "teacher_messages": [{"role": "user", "content": "describe"}],
        "prompt_ids": [1, 99, 99, 2],
        "rollout_prompt_ids": [1, 99, 2],
        "descriptors": ("descriptor-a",),
    }
    first = opd_mod._PromptRecord(**base)
    same_content = opd_mod._PromptRecord(**base)
    changed_descriptor = opd_mod._PromptRecord(
        **{**base, "descriptors": ("descriptor-b",)}
    )
    changed_teacher = opd_mod._PromptRecord(
        **{
            **base,
            "teacher_messages": [{"role": "user", "content": "different"}],
        }
    )

    fingerprint = opd_mod._opd_prompt_pool_fingerprint([first])

    assert fingerprint == opd_mod._opd_prompt_pool_fingerprint([same_content])
    assert fingerprint != opd_mod._opd_prompt_pool_fingerprint([changed_descriptor])
    assert fingerprint != opd_mod._opd_prompt_pool_fingerprint([changed_teacher])


def test_opd_text_prompt_fingerprint_is_legacy_byte_identical():
    from flash.engine.worker import opd as opd_mod

    record = opd_mod._PromptRecord(
        example={"id": 1, "input": "hello"},
        student_messages=[{"role": "user", "content": "hello"}],
        teacher_messages=[{"role": "user", "content": "hello"}],
        prompt_ids=[1, 2, 3],
        rollout_prompt_ids=[1, 2, 3],
    )

    assert (
        opd_mod._opd_prompt_pool_fingerprint([record])
        == "5f5531e538fe92f40eb1726dbd8973b9a4439beffcfba848667c91f3e8ee42e3"
    )
    assert opd_mod._opd_prompt_pool_fingerprint([record]) == opd_mod._opd_prompt_pool_fingerprint(
        [(record.example, record.student_messages, record.prompt_ids)]
    )


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


def test_opd_multi_turn_distills_every_assistant_turn(monkeypatch):
    """END-TO-END proof that multi-turn opd distils EACH assistant turn against the teacher, conditioned
    on the growing transcript. Drives the REAL run_opd -> rollout_one_records -> vLLM rollout shim ->
    batched teacher scoring -> _resolve_samples_batched path on CPU with a scripted 3-turn "guess"
    episode: a fake student that emits a distinct completion per turn and a fake teacher that
    echo-scores each. We assert (1) three turns were distilled with real gradients, (2) the second turn's
    teacher prompt and loss prefix strictly GREW over the first (per-turn transcript conditioning, not
    a re-scored first turn), and (3) train_meta reports the multi-turn shape."""
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    tok = _CharTok()
    V = len(tok.alpha) + 1  # +1 for the eos id (== len(alpha))

    class _MTModel(_TinyLM):
        """Scripted generating student: turn t emits guesses[t] + eos; per-position logits for the loss."""

        def __init__(self, guesses):
            super().__init__(torch, T=256, V=V)
            self.config = SimpleNamespace(use_cache=False)
            self._guesses = guesses
            self._i = 0

        def eval(self):
            return self

        def train(self):
            return self

        def to(self, device):
            return self

        def generate(self, prompt_tensor, **cfg):
            guess = self._guesses[min(self._i, len(self._guesses) - 1)]
            self._i += 1
            comp = [*tok._enc(guess), tok.eos_token_id]
            comp_t = torch.tensor([comp], dtype=prompt_tensor.dtype)
            return torch.cat([prompt_tensor, comp_t], dim=1)

        def save_pretrained(self, *a, **k):
            pass

    model = _MTModel(["42", "ok", "42"])

    teacher_batches = []

    class _CountingTeacher:
        def score_many(self, items):
            teacher_batches.append(list(items))
            return [
                [TeacherToken(text=completion, logprob=-1.0, start=0, end=len(completion))]
                for _prompt, completion in items
            ]

    class _GuessEnv:
        """three-assistant-turn episode with an environment nudge between guesses."""

        multi_turn = True
        is_tool_env = False
        max_turns = 4

        def dataset(self):
            return [{"input": "guess", "output": "42", "id": "e0"}]

        def prompt_messages(self, ex):
            return [{"role": "user", "content": "g"}]

        def new_rollout_state(self, ex):
            return {
                "prompt": [{"role": "user", "content": "g"}],
                "messages": [{"role": "user", "content": "g"}],
                "turn": 0,
                "done": False,
            }

        def record_model_turn(self, state, content):
            state["last"] = content

        def env_reply(self, messages, state):
            state["turn"] += 1
            if state["turn"] >= 3:
                state["done"] = True
                return []
            return [{"role": "user", "content": "n"}]

        def rollout_done(self, state, max_turns=None):
            return bool(state.get("done")) or state["turn"] >= (max_turns or 4)

    env = _GuessEnv()

    loss_calls = []
    real_resolve_samples_batched = opd_mod._resolve_samples_batched

    def _spy_resolve_samples_batched(
        model, tok_, device, samples, knobs, microbatch, *, backward_scale=None, **_kwargs
    ):
        out = real_resolve_samples_batched(
            model, tok_, device, samples, knobs, microbatch, backward_scale=backward_scale, **_kwargs
        )
        for (gen, _score, prompt_ids), r in zip(samples, out, strict=True):
            loss_calls.append((len(prompt_ids), gen.completion_text, r.loss is not None))
        return out

    monkeypatch.setattr(opd_mod, "_resolve_samples_batched", _spy_resolve_samples_batched)

    meta = {}
    fake_w = SimpleNamespace(
        require_active_env=lambda: env,
        JOB_SPEC=SimpleNamespace(
            train=SimpleNamespace(init_from_adapter=""),
            model="fake/model",
            gpu=SimpleNamespace(type=None, exact_type=""),
        ),
        THINKING=False,
        SEED=0,
        OPD_RESUME_REVISION="",
        publish_opd_optimizer_start_marker=lambda: None,
        heartbeat=lambda stage, **kw: None,
        prefetch_model=lambda mid: 0.0,
        hf_resume_checkpoint=lambda **_kwargs: "",
        publish_deployable_checkpoint=lambda *a, **k: None,
        hf_upload_folder=lambda *a, **k: None,
        write_train_meta=lambda **k: meta.update(k),
        wandb_report_to=lambda: [],
        wandb_run_info=lambda: {},
    )
    monkeypatch.setattr(opd_mod, "_w", fake_w)
    monkeypatch.setattr(opd_mod, "_opd_teacher_batch_size", lambda _total: 2)
    monkeypatch.setattr(
        opd_mod,
        "_resolve_opd_knobs",
        lambda: opd_mod.OpdKnobs(
            teacher_model="accounts/fireworks/models/glm-5p2",
            teacher_base_url="http://teacher.invalid",
            epochs=1,
            learning_rate=1e-4,
            temperature=0.0,
            top_p=1.0,
            max_completion=8,
            prompts_per_step=1,
            group_size=1,
            kl_coef=1.0,
            save_every=0,
            max_length=128,
            stop_sequences=(),
        ),
    )
    monkeypatch.setattr(opd_mod, "_student_model", lambda *a, **k: (model, "fake/model"))
    monkeypatch.setattr(opd_mod, "_save_adapter", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "wait_for_gpu", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "setup_perf_backends", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "grad_checkpointing_on", lambda *a, **k: False)
    monkeypatch.setattr(opd_mod, "gpu_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(opd_mod, "install_chalk_kernels", lambda *a, **k: {})
    monkeypatch.setattr(opd_mod, "active_kernels", lambda *a, **k: [])
    monkeypatch.setattr(opd_mod, "free_gpu", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "_save_opd_resume_checkpoint", lambda **kwargs: None)
    monkeypatch.setattr(opd_mod, "_save_adapter", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", lambda *a, **k: None)
    _patch_opd_run_vllm_stub(
        monkeypatch,
        opd_mod,
        outputs=[
            opd_mod.OpdVllmOutput([*tok._enc("42"), tok.eos_token_id], "42", finish_reason="stop"),
            opd_mod.OpdVllmOutput([*tok._enc("ok"), tok.eos_token_id], "ok", finish_reason="stop"),
            opd_mod.OpdVllmOutput([*tok._enc("42"), tok.eos_token_id], "42", finish_reason="stop"),
        ],
    )
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    import flash.engine.worker.teacher as tmod

    monkeypatch.setattr(tmod, "TeacherClient", lambda *a, **k: _CountingTeacher())
    monkeypatch.setenv("FIREWORKS_API_KEY", "unit-test-teacher-key")

    # Snapshot the trainable weight so we can prove the optimizer actually moved it.
    before = model.w.detach().clone()

    opd_mod.run_opd()

    assert len(loss_calls) == 3, f"expected 3 per-turn losses, got {loss_calls}"
    assert [c[1] for c in loss_calls] == ["42", "ok", "42"]
    assert all(c[2] for c in loss_calls), "every distilled turn must produce a real loss"
    # The second turn's loss prefix (transcript so far) is strictly LONGER than the first's.
    assert loss_calls[1][0] > loss_calls[0][0]

    assert [len(batch) for batch in teacher_batches] == [2, 1]
    scored_items = [item for batch in teacher_batches for item in batch]
    first_prompt, _first_completion = scored_items[0]
    second_prompt, _second_completion = scored_items[1]
    assert [completion for _prompt, completion in scored_items] == ["42", "ok", "42"]
    assert "Assistant: 42" not in first_prompt
    assert "User: n" not in first_prompt
    assert "Assistant: 42" in second_prompt
    assert "User: n" in second_prompt

    # (3) A real optimizer step landed and moved the weights; train_meta shows the multi-turn shape.
    assert not torch.equal(before, model.w.detach())
    assert meta["notes"]["multi_turn"] is True
    assert meta["notes"]["episodes"] == 1
    assert meta["notes"]["mean_turns_per_episode"] == 3.0
    assert meta["notes"]["max_turns"] == 4
    assert meta["step"] == 1  # one optimizer update over the three turn losses


def test_opd_passes_worker_env_teacher_key_to_client(monkeypatch):
    """run_opd reads the platform-injected FIREWORKS_API_KEY from the worker env and uses it to
    construct the TeacherClient."""
    opd_mod = _opd_harness(
        monkeypatch, sample_result=_skip
    )  # sets FIREWORKS_API_KEY=unit-test-teacher-key
    captured = {}
    import flash.engine.worker.teacher as tmod

    def _capture_client(api_key, *a, **k):
        captured["key"] = api_key
        return object()

    monkeypatch.setattr(tmod, "TeacherClient", _capture_client)
    with pytest.raises(RuntimeError):  # all-skip -> no trained step, after TeacherClient is built
        opd_mod.run_opd()
    assert captured["key"] == "unit-test-teacher-key"


def test_opd_missing_teacher_key_raises_platform_managed_error(monkeypatch):
    """With no key in the worker env, run_opd fails with the platform-managed diagnostic (a
    platform-side injection failure), not the old 'declare and export it' message."""
    opd_mod = _opd_harness(monkeypatch, sample_result=_skip)
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="platform-managed"):
        opd_mod.run_opd()


def test_opd_liveness_heartbeat_gets_monotonic_progress_callback(monkeypatch):
    """Regression (codex[bot], opd.py): opd must hand liveness_heartbeat a progress callback (parity
    with sft/rl) so its thread emits REAL progress on sample advance instead of pure liveness=true
    pings that share — and can starve — the opd_step upload throttle. Confirm the progress arg is a
    callable that reflects the monotonic sample count."""
    import contextlib

    captured = {}

    @contextlib.contextmanager
    def _fake_liveness(stage, progress=None, fields=None, **_kwargs):
        if stage == "opd_step" and progress is not None:
            captured["stage"] = stage
            captured["progress"] = progress
        yield

    opd_mod = _opd_harness(monkeypatch, sample_result=_skip, liveness=_fake_liveness)
    with pytest.raises(RuntimeError):  # all-skip -> no trained step
        opd_mod.run_opd()
    assert captured["stage"] == "opd_step"
    assert callable(captured["progress"]), "opd must pass a progress callback to liveness_heartbeat"
    # The callback reports the monotonic sample count. An all-skip run lands no optimizer update, so
    # the bounded-retry loop visits its full budget of max_iters = 3*steps + 10 = 13 fresh slices
    # (1 prompt x 1 group each) before the post-loop guard raises -> samples_seen advanced to 13.
    assert captured["progress"]() == 13


def test_opd_vllm_generation_uses_keepalive_heartbeat(monkeypatch):
    """A large batched vLLM rollout can block before samples_seen advances; keepalive emits real
    opd_step heartbeats during that blocking generate call so provider stall detection sees the job is
    alive and actual_steps_run floors a cancellation during first-step GPU work to one step."""
    import contextlib

    calls = []

    @contextlib.contextmanager
    def _fake_liveness(stage, progress=None, fields=None, **kwargs):
        calls.append((stage, progress, fields, kwargs))
        yield

    opd_mod = _opd_harness(monkeypatch, sample_result=_skip, liveness=_fake_liveness)
    with pytest.raises(RuntimeError):  # all-skip -> no trained step
        opd_mod.run_opd()

    generate_calls = [c for c in calls if c[0] == "opd_step" and c[3].get("keepalive") is True]
    assert generate_calls, f"missing vLLM generate keepalive context; saw {[c[0] for c in calls]}"
    assert all(c[3].get("keepalive") is True for c in generate_calls)
    assert callable(generate_calls[0][2])
    assert generate_calls[0][2]() == {"step": 0}


def test_opd_step_heartbeat_carries_distilled_sample_completions(monkeypatch):
    """The forced post-update opd_step heartbeat surfaces the distilled student completions (with each
    sample's distillation loss) so `flash log` shows what the student generated -- the OPD analog of
    GRPO's reward samples. The default rollout stub emits completion text "x" for the one prompt."""
    pytest.importorskip("torch")

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    beats: list = []
    opd_mod = _opd_harness(monkeypatch, sample_result=_one_update, beats=beats)
    opd_mod.run_opd()

    sample_beats = [
        kw for (stage, kw) in beats if stage == "opd_step" and "sampled_completions" in kw
    ]
    assert len(sample_beats) == 1, (
        f"exactly one post-update opd_step heartbeat should carry samples; saw {len(sample_beats)}"
    )
    payload = sample_beats[0]
    # Samples ride the forced post-update ping, which also carries the optimizer step + loss.
    assert payload["force"] is True
    assert payload["step"] == 1
    assert "loss" in payload
    samples = payload["sampled_completions"]
    assert len(samples) == 1
    sample = samples[0]
    assert sample["completion"] == "x"
    assert sample["prompt_tail"] == "user: a"
    # OPD samples carry a distillation loss, never a reward.
    assert "reward" not in sample
    assert isinstance(sample["loss"], float)
    # Labelled with the PRE-update step whose policy generated it (0 here), even though the heartbeat
    # reports the completed update (step 1) -- parity with GRPO's generation-time step.
    assert sample["generated_at_step"] == 0


def test_opd_reconciles_required_resume_companion_before_restored_sync_and_generation(
    monkeypatch,
):
    pytest.importorskip("torch")
    events = []

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    opd_mod = _opd_harness(
        monkeypatch,
        sample_result=_one_update,
        epochs=2,
        group=1,
        save_at_steps=(1,),
    )
    opd_mod._w.OPD_RESUME_REVISION = "pinned-sha"
    opd_mod._w.hf_resume_checkpoint = lambda **_kwargs: "/tmp/checkpoint-1"
    monkeypatch.setattr(
        opd_mod,
        "_restore_opd_full_state",
        lambda *args, **kwargs: {
            "opt_steps": 1,
            "step": 1,
            "loss_curve": [0.1],
            "coverage_curve": [1.0],
            "rollout_seed_ordinal": 1,
        },
    )
    monkeypatch.setattr(
        opd_mod,
        "_reconcile_required_opd_deployable",
        lambda path, step, required: events.append(("reconcile", path, step, required)),
    )
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", lambda *args, **kwargs: None)
    engine_type = opd_mod.OpdVllmRolloutEngine
    original_sync = engine_type.sync_from_model
    original_generate = engine_type.generate

    def _sync(self, model):
        events.append("sync")
        return original_sync(self, model)

    def _generate(self, *args, **kwargs):
        events.append("generate")
        return original_generate(self, *args, **kwargs)

    monkeypatch.setattr(engine_type, "sync_from_model", _sync)
    monkeypatch.setattr(engine_type, "generate", _generate)

    opd_mod.run_opd()

    assert events == [
        ("reconcile", "/tmp/checkpoint-1", 1, (1,)),
        "sync",
        "generate",
    ]


def test_opd_initial_sync_failure_happens_after_required_resume_reconciliation(monkeypatch):
    pytest.importorskip("torch")
    events = []
    opd_mod = _opd_harness(monkeypatch, sample_result=_skip, save_at_steps=(1,))
    opd_mod._w.OPD_RESUME_REVISION = "pinned-sha"
    opd_mod._w.hf_resume_checkpoint = lambda **_kwargs: "/tmp/checkpoint-1"
    monkeypatch.setattr(
        opd_mod,
        "_restore_opd_full_state",
        lambda *args, **kwargs: {
            "opt_steps": 1,
            "step": 1,
            "loss_curve": [0.1],
            "coverage_curve": [1.0],
            "rollout_seed_ordinal": 1,
        },
    )
    monkeypatch.setattr(
        opd_mod,
        "_reconcile_required_opd_deployable",
        lambda path, step, required: events.append(("reconcile", path, step, required)),
    )
    engine_type = opd_mod.OpdVllmRolloutEngine

    def _fail_sync(self, model):
        events.append("sync")
        raise RuntimeError("initial sync failed")

    monkeypatch.setattr(engine_type, "sync_from_model", _fail_sync)
    monkeypatch.setattr(
        engine_type,
        "generate",
        lambda *args, **kwargs: events.append("generate"),
    )

    with pytest.raises(RuntimeError, match="initial sync failed"):
        opd_mod.run_opd()

    assert events == [
        ("reconcile", "/tmp/checkpoint-1", 1, (1,)),
        "sync",
    ]


def test_opd_due_checkpoint_is_complete_before_nonfinal_sync(monkeypatch):
    pytest.importorskip("torch")
    events = []
    checkpoints = []

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=0.75,
            gen_tokens=2,
            teacher_tokens=3,
        )

    opd_mod = _opd_harness(
        monkeypatch,
        sample_result=_one_update,
        epochs=2,
        group=1,
        save_at_steps=(1,),
    )
    engine_type = opd_mod.OpdVllmRolloutEngine
    original_sync = engine_type.sync_from_model
    original_generate = engine_type.generate

    def _sync(self, model):
        events.append("sync")
        return original_sync(self, model)

    def _generate(self, *args, **kwargs):
        events.append("generate")
        return original_generate(self, *args, **kwargs)

    def _save_checkpoint(**kwargs):
        events.append("checkpoint")
        checkpoints.append(kwargs)
        if kwargs.get("after_upload") is not None:
            kwargs["after_upload"]()

    monkeypatch.setattr(engine_type, "sync_from_model", _sync)
    monkeypatch.setattr(engine_type, "generate", _generate)
    monkeypatch.setattr(opd_mod, "_save_adapter", lambda *args, **kwargs: events.append("save_adapter"))
    monkeypatch.setattr(opd_mod, "_save_opd_resume_checkpoint", _save_checkpoint)
    monkeypatch.setattr(
        opd_mod,
        "_publish_opd_deployable",
        lambda _path, step, **_kwargs: events.append(f"deployable:{step}"),
    )

    opd_mod.run_opd()

    first_generate = events.index("generate")
    checkpoint = events.index("checkpoint")
    companion = events.index("deployable:1")
    post_update_sync = events.index("sync", first_generate + 1)
    second_generate = events.index("generate", first_generate + 1)
    assert checkpoint < companion < post_update_sync < second_generate
    saved = checkpoints[0]
    assert saved["opt_steps"] == 1
    assert saved["step"] == 1
    assert saved["rollout_seed_ordinal"] == 1
    assert len(saved["accounting"]["loss_curve"]) == 1
    assert saved["accounting"]["coverage_curve"] == [0.75]
    assert saved["accounting"]["samples_seen"] == 1
    assert saved["accounting"]["generated_tokens"] == 2
    assert saved["accounting"]["teacher_input_tokens"] == 3
    assert saved["accounting"]["teacher_ok"] == 1
    assert saved["accounting"]["opd_phase_counts"]["optimizer_steps"] == 1
    assert saved["accounting"]["opd_phase_counts"].get("vllm_syncs", 0) == 0


def test_opd_sync_failure_after_due_step_preserves_checkpoint_and_blocks_next_rollout(monkeypatch):
    pytest.importorskip("torch")
    events = []

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    opd_mod = _opd_harness(
        monkeypatch,
        sample_result=_one_update,
        epochs=2,
        group=1,
        save_at_steps=(1,),
    )
    engine_type = opd_mod.OpdVllmRolloutEngine
    original_sync = engine_type.sync_from_model
    original_generate = engine_type.generate

    def _sync(self, model):
        events.append("sync")
        if events.count("sync") == 2:
            raise RuntimeError("sync failed")
        return original_sync(self, model)

    def _generate(self, *args, **kwargs):
        events.append("generate")
        return original_generate(self, *args, **kwargs)

    def _save_checkpoint(**kwargs):
        events.append("checkpoint")
        if kwargs.get("after_upload") is not None:
            kwargs["after_upload"]()

    monkeypatch.setattr(engine_type, "sync_from_model", _sync)
    monkeypatch.setattr(engine_type, "generate", _generate)
    monkeypatch.setattr(opd_mod, "_save_adapter", lambda *args, **kwargs: None)
    monkeypatch.setattr(opd_mod, "_save_opd_resume_checkpoint", _save_checkpoint)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="sync failed"):
        opd_mod.run_opd()

    assert events == ["sync", "generate", "checkpoint", "sync"]


def test_opd_periodic_checkpoint_failure_still_syncs_and_runs_next_rollout(monkeypatch):
    pytest.importorskip("torch")
    events = []
    save_calls = []

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    opd_mod = _opd_harness(
        monkeypatch,
        sample_result=_one_update,
        epochs=2,
        group=1,
        save_every=1,
    )
    engine_type = opd_mod.OpdVllmRolloutEngine
    original_sync = engine_type.sync_from_model
    original_generate = engine_type.generate

    def _sync(self, model):
        events.append("sync")
        return original_sync(self, model)

    def _generate(self, *args, **kwargs):
        events.append("generate")
        return original_generate(self, *args, **kwargs)

    def _save_checkpoint(**kwargs):
        save_calls.append(kwargs)
        events.append("checkpoint")
        if not kwargs["required"]:
            events.append("periodic_failure_suppressed")

    monkeypatch.setattr(engine_type, "sync_from_model", _sync)
    monkeypatch.setattr(engine_type, "generate", _generate)
    monkeypatch.setattr(opd_mod, "_save_adapter", lambda *args, **kwargs: None)
    monkeypatch.setattr(opd_mod, "_save_opd_resume_checkpoint", _save_checkpoint)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", lambda *args, **kwargs: None)

    opd_mod.run_opd()

    assert events == [
        "sync",
        "generate",
        "checkpoint",
        "periodic_failure_suppressed",
        "sync",
        "generate",
        "checkpoint",
    ]
    assert [call["required"] for call in save_calls] == [False, True]


def test_opd_final_nondue_step_saves_required_full_state_before_default(monkeypatch):
    pytest.importorskip("torch")
    events = []
    checkpoints = []

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    opd_mod = _opd_harness(
        monkeypatch,
        sample_result=_one_update,
        epochs=1,
        group=1,
        save_every=2,
    )

    def _save_checkpoint(**kwargs):
        checkpoints.append(kwargs)
        events.append("full_state")

    def _publish(_path, _step, **kwargs):
        events.append(("deployable", kwargs))

    monkeypatch.setattr(opd_mod, "_save_opd_resume_checkpoint", _save_checkpoint)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", _publish)

    opd_mod.run_opd()

    assert events == ["full_state", ("deployable", {"as_default": True, "publish_checkpoint": True})]
    assert len(checkpoints) == 1
    assert checkpoints[0]["opt_steps"] == 1
    assert checkpoints[0]["required"] is True
    assert checkpoints[0].get("after_upload") is None


def test_opd_final_periodic_step_publishes_one_full_state_and_one_step_deployable(monkeypatch):
    pytest.importorskip("torch")
    checkpoints = []
    deployables = []

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    opd_mod = _opd_harness(
        monkeypatch,
        sample_result=_one_update,
        epochs=1,
        group=1,
        save_every=1,
    )
    monkeypatch.setattr(
        opd_mod,
        "_save_opd_resume_checkpoint",
        lambda **kwargs: checkpoints.append(kwargs),
    )
    monkeypatch.setattr(
        opd_mod,
        "_publish_opd_deployable",
        lambda path, step, **kwargs: deployables.append((path, step, kwargs)),
    )

    opd_mod.run_opd()

    assert len(checkpoints) == 1
    assert checkpoints[0]["required"] is True
    assert checkpoints[0].get("after_upload") is None
    assert len(deployables) == 1
    assert deployables[0][1:] == (
        1,
        {"as_default": True, "publish_checkpoint": True},
    )


def test_opd_final_exact_step_has_one_required_transaction_without_republish(monkeypatch):
    pytest.importorskip("torch")
    events = []
    checkpoints = []

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    opd_mod = _opd_harness(
        monkeypatch,
        sample_result=_one_update,
        epochs=1,
        group=1,
        save_at_steps=(1,),
    )

    def _save_checkpoint(**kwargs):
        checkpoints.append(kwargs)
        events.append("full_state")
        kwargs["after_upload"]()

    def _publish(_path, _step, **kwargs):
        events.append(("deployable", kwargs))

    monkeypatch.setattr(
        opd_mod,
        "_save_adapter",
        lambda *args, **kwargs: events.append("stage_adapter"),
    )
    monkeypatch.setattr(opd_mod, "_save_opd_resume_checkpoint", _save_checkpoint)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", _publish)

    opd_mod.run_opd()

    assert events == [
        "stage_adapter",
        "full_state",
        (
            "deployable",
            {"as_default": False, "best_effort": False, "save_required": True},
        ),
        ("deployable", {"as_default": True, "publish_checkpoint": False}),
    ]
    assert len(checkpoints) == 1
    assert checkpoints[0]["required"] is True
    assert callable(checkpoints[0]["after_upload"])


def test_opd_final_deployable_failure_happens_after_required_full_state(monkeypatch):
    pytest.importorskip("torch")
    events = []

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    opd_mod = _opd_harness(monkeypatch, sample_result=_one_update, epochs=1, group=1)

    def _save_checkpoint(**kwargs):
        assert kwargs["required"] is True
        events.append("full_state")

    def _fail_deployable(*args, **kwargs):
        events.append("deployable")
        raise RuntimeError("final deployable failed")

    monkeypatch.setattr(opd_mod, "_save_opd_resume_checkpoint", _save_checkpoint)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", _fail_deployable)

    with pytest.raises(RuntimeError, match="final deployable failed"):
        opd_mod.run_opd()

    assert events == ["full_state", "deployable"]


def test_opd_resumed_at_final_step_does_not_repeat_full_state(monkeypatch):
    pytest.importorskip("torch")
    events = []

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    opd_mod = _opd_harness(monkeypatch, sample_result=_one_update, epochs=1, group=1)
    opd_mod._w.OPD_RESUME_REVISION = "pinned-sha"
    opd_mod._w.hf_resume_checkpoint = lambda **_kwargs: "/tmp/checkpoint-1"
    monkeypatch.setattr(
        opd_mod,
        "_restore_opd_full_state",
        lambda *args, **kwargs: {
            "opt_steps": 1,
            "step": 1,
            "loss_curve": [0.1],
            "coverage_curve": [1.0],
            "rollout_seed_ordinal": 1,
        },
    )
    monkeypatch.setattr(opd_mod, "_reconcile_required_opd_deployable", lambda *args: None)
    monkeypatch.setattr(
        opd_mod,
        "_save_opd_resume_checkpoint",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not repeat final full state")),
    )
    monkeypatch.setattr(
        opd_mod,
        "_save_adapter",
        lambda *args, **kwargs: events.append("stage_adapter"),
    )
    monkeypatch.setattr(
        opd_mod,
        "_publish_opd_deployable",
        lambda *args, **kwargs: events.append(("deployable", kwargs)),
    )

    opd_mod.run_opd()

    assert events == [
        "stage_adapter",
        ("deployable", {"as_default": True, "publish_checkpoint": True}),
    ]


def test_opd_nondue_step_syncs_before_next_rollout(monkeypatch):
    pytest.importorskip("torch")
    events = []

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    opd_mod = _opd_harness(monkeypatch, sample_result=_one_update, epochs=2, group=1)
    engine_type = opd_mod.OpdVllmRolloutEngine
    original_sync = engine_type.sync_from_model
    original_generate = engine_type.generate

    def _sync(self, model):
        events.append("sync")
        return original_sync(self, model)

    def _generate(self, *args, **kwargs):
        events.append("generate")
        return original_generate(self, *args, **kwargs)

    monkeypatch.setattr(engine_type, "sync_from_model", _sync)
    monkeypatch.setattr(engine_type, "generate", _generate)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", lambda *args, **kwargs: None)

    opd_mod.run_opd()

    assert events == ["sync", "generate", "sync", "generate"]


def test_opd_skips_final_vllm_sync_after_last_optimizer_step(monkeypatch):
    torch = pytest.importorskip("torch")

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        loss = model.w.float().sum() * 1e-6
        return SampleResult(
            loss=loss, teacher_status="ok", coverage=1.0, gen_tokens=1, teacher_tokens=1
        )

    opd_mod = _opd_harness(monkeypatch, sample_result=_one_update, epochs=1, group=1)
    monkeypatch.setattr(opd_mod, "_save_adapter", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", lambda *a, **k: None)

    opd_mod.run_opd()

    engine = opd_mod.OpdVllmRolloutEngine.instances[0]
    assert engine.sync_count == 1  # initial LoRA sync only; no rollout remains after step 1


def test_opd_resolves_one_halt_set_for_generation(monkeypatch):
    torch = pytest.importorskip("torch")

    def _one_update(*, model, **_kwargs):
        from flash.engine.worker.opd import SampleResult

        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    opd_mod = _opd_harness(monkeypatch, sample_result=_one_update, epochs=1, group=1)
    halt_set = frozenset({5, 7})
    calls = []
    monkeypatch.setattr(
        opd_mod,
        "_generation_eos_ids",
        lambda model, tok: calls.append((model, tok)) or halt_set,
    )
    monkeypatch.setattr(opd_mod, "_save_adapter", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", lambda *a, **k: None)

    opd_mod.run_opd()

    assert len(calls) == 1
    assert opd_mod.OpdVllmRolloutEngine.instances[0].eos_token_ids == (5, 7)


def test_opd_accounts_teacher_scores_as_they_finish(monkeypatch):
    """Regression: a slow teacher response must not hold back loss/backward for faster responses in the
    same OPD step. The old step barrier waited for every teacher future before resolving any sample."""
    import threading
    import time

    torch = pytest.importorskip("torch")

    opd_mod = _opd_harness(monkeypatch, sample_result=_skip, epochs=1, group=2)
    monkeypatch.setattr(opd_mod, "_opd_teacher_batch_size", lambda _total: 1)
    monkeypatch.setattr(opd_mod, "_save_adapter", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", lambda *a, **k: None)

    events: list[tuple[str, int]] = []
    lock = threading.Lock()
    calls = {"n": 0}

    def _score_one(*_args, **_kwargs):
        with lock:
            idx = calls["n"]
            calls["n"] += 1
        if idx == 0:
            time.sleep(0.05)
        events.append(("score", idx))
        return opd_mod._ScoreResult(teacher_toks=[idx], status="ok")

    def _resolve_samples_batched(
        model, tok, device, samples, knobs, microbatch, *, backward_scale=None, **_kwargs
    ):
        out = []
        for _gen, score, _prompt_ids in samples:
            idx = int(score.teacher_toks[0])
            events.append(("resolve", idx))
            loss = model.w.float().sum() * 1e-6
            out.append(
                opd_mod.SampleResult(
                    loss=loss, teacher_status="ok", coverage=1.0, gen_tokens=1, teacher_tokens=1
                )
            )
        if backward_scale is not None:
            losses = [r.loss for r in out if r.loss is not None]
            if losses:
                (sum(losses) * backward_scale).backward()
        return out

    monkeypatch.setattr(opd_mod, "_score_one", _score_one)
    monkeypatch.setattr(opd_mod, "_resolve_samples_batched", _resolve_samples_batched)

    opd_mod.run_opd()

    assert events.index(("resolve", 1)) < events.index(("score", 0))


def test_opd_rollout_chunking_scales_for_heavy_steps():
    import flash.engine.worker.opd as opd_mod

    assert opd_mod._opd_rollout_pipeline_chunks(1) == 1
    assert opd_mod._opd_rollout_pipeline_chunks(7) == 1
    assert opd_mod._opd_rollout_pipeline_chunks(8) == 2
    assert opd_mod._opd_rollout_chunk_size(8) == 4
    assert opd_mod._opd_rollout_pipeline_chunks(32) == 2
    assert opd_mod._opd_rollout_chunk_size(32) == 16
    assert opd_mod._opd_rollout_pipeline_chunks(64) == 4
    assert opd_mod._opd_rollout_chunk_size(64) == 16
    assert opd_mod._opd_rollout_pipeline_chunks(256) == 8


def test_opd_chunks_single_turn_rollout_to_overlap_teacher(monkeypatch):
    """Default OPD steps have 8 rollouts. Generate them in chunks so teacher scoring for the first
    chunk can run while vLLM generates the later chunk."""
    import threading

    torch = pytest.importorskip("torch")

    opd_mod = _opd_harness(monkeypatch, sample_result=_skip, epochs=1, group=8)
    monkeypatch.setattr(opd_mod, "_save_adapter", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", lambda *a, **k: None)

    events: list[tuple[str, int, int]] = []
    first_score_started = threading.Event()

    def _generate_many_vllm(
        _rollout,
        _tok,
        prompt_ids_batch,
        _knobs,
        _eos_ids,
        *,
        max_tokens,
        request_seeds=None,
    ):
        call_idx = sum(1 for e in events if e[0] == "generate")
        if call_idx == 1:
            first_score_started.wait(timeout=1.0)
        events.append(("generate", call_idx, len(prompt_ids_batch)))
        return [
            opd_mod._GenResult(completion_ids=[3], completion_text="x", gen_tokens=1)
            for _ in prompt_ids_batch
        ]

    def _score_one(*_args, **_kwargs):
        events.append(("score", len([e for e in events if e[0] == "score"]), 0))
        first_score_started.set()
        return opd_mod._ScoreResult(teacher_toks=[], status="ok")

    def _resolve_samples_batched(
        model, tok, device, samples, knobs, microbatch, *, backward_scale=None, **_kwargs
    ):
        out = [
            opd_mod.SampleResult(
                loss=model.w.float().sum() * 1e-6,
                teacher_status="ok",
                coverage=1.0,
                gen_tokens=1,
                teacher_tokens=1,
            )
            for _sample in samples
        ]
        if backward_scale is not None:
            losses = [r.loss for r in out if r.loss is not None]
            if losses:
                (sum(losses) * backward_scale).backward()
        return out

    monkeypatch.setattr(opd_mod, "_generate_many_vllm", _generate_many_vllm)
    monkeypatch.setattr(opd_mod, "_score_one", _score_one)
    monkeypatch.setattr(opd_mod, "_resolve_samples_batched", _resolve_samples_batched)

    opd_mod.run_opd()

    assert [e[2] for e in events if e[0] == "generate"] == [4, 4]
    first_score = next(i for i, e in enumerate(events) if e[0] == "score")
    second_generate = events.index(("generate", 1, 4))
    assert first_score < second_generate


def test_opd_teacher_batch_workers_and_loss_microbatch_defaults():
    import flash.engine.worker.opd as opd_mod

    assert opd_mod._opd_teacher_batch_size(64) == 8
    assert opd_mod._opd_teacher_workers(64, 8) == 8
    assert opd_mod._opd_teacher_workers(64, 16) == 4
    assert opd_mod._opd_loss_microbatch_size("Qwen/Qwen3.5-2B", 64) == 4
    assert opd_mod._opd_loss_microbatch_size("Qwen/Qwen3.6-35B-A3B", 64) == 1


def test_opd_scores_generated_chunk_in_teacher_batches(monkeypatch):
    torch = pytest.importorskip("torch")

    opd_mod = _opd_harness(monkeypatch, sample_result=_skip, epochs=1, group=8)
    monkeypatch.setattr(opd_mod, "_opd_teacher_batch_size", lambda _total: 4)
    monkeypatch.setattr(opd_mod, "_opd_teacher_workers", lambda _total, _batch: 2)
    monkeypatch.setattr(opd_mod, "_save_adapter", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", lambda *a, **k: None)

    batches: list[int] = []

    def _score_many(_teacher, pendings, **_kwargs):
        batches.append(len(pendings))
        return [opd_mod._ScoreResult(teacher_toks=[], status="ok") for _ in pendings]

    def _resolve_samples_batched(
        model, tok, device, samples, knobs, microbatch, *, backward_scale=None, **_kwargs
    ):
        out = [
            opd_mod.SampleResult(
                loss=model.w.float().sum() * 1e-6,
                teacher_status="ok",
                coverage=1.0,
                gen_tokens=1,
                teacher_tokens=1,
            )
            for _sample in samples
        ]
        if backward_scale is not None:
            losses = [r.loss for r in out if r.loss is not None]
            if losses:
                (sum(losses) * backward_scale).backward()
        return out

    monkeypatch.setattr(opd_mod, "_score_many", _score_many)
    monkeypatch.setattr(opd_mod, "_resolve_samples_batched", _resolve_samples_batched)

    opd_mod.run_opd()

    assert batches == [4, 4]


def test_opd_no_signal_from_transient_teacher_is_retriable(monkeypatch):
    """Regression (codex[bot], opd.py): a run where EVERY teacher.score fails transiently (a Fireworks
    outage spanning the run) and none succeed must raise a RetriableInfraError so the supervisor
    retries — not a plain RuntimeError, which it treats as permanent. A no-signal run where the
    teacher DID respond (but alignment yielded nothing) stays a permanent RuntimeError."""
    from flash.engine.worker.perf import RetriableInfraError

    def _all_transient(**k):
        return opd_mod.SampleResult(teacher_status="transient")

    opd_mod = _opd_harness(monkeypatch, sample_result=_all_transient)
    with pytest.raises(RetriableInfraError, match="failed transiently"):
        opd_mod.run_opd()

    # contrast: teacher responded ("ok") but no loss -> permanent RuntimeError, NOT retriable.
    def _ok_no_align(**k):
        return opd_mod.SampleResult(teacher_status="ok")

    opd_mod = _opd_harness(monkeypatch, sample_result=_ok_no_align)
    with pytest.raises(RuntimeError) as ei:
        opd_mod.run_opd()
    assert not isinstance(ei.value, RetriableInfraError)
    assert "no trained step" in str(ei.value)


def test_opd_resamples_no_signal_rollout_before_skipping_step(monkeypatch):
    """A single all-skip rollout attempt should not consume a requested optimizer update. OPD should
    resample within the same optimizer step and train when the replacement yields teacher signal."""

    torch = pytest.importorskip("torch")

    state = {"n": 0}
    metas = []

    def _skip_once_then_update(*, model, **_k):
        from flash.engine.worker.opd import SampleResult

        state["n"] += 1
        if state["n"] == 1:
            return SampleResult(teacher_status="ok", skip_reason="alignment_empty")
        return SampleResult(
            loss=model.w.float().sum() * 1e-6,
            teacher_status="ok",
            coverage=1.0,
            gen_tokens=1,
            teacher_tokens=1,
        )

    opd_mod = _opd_harness(monkeypatch, sample_result=_skip_once_then_update, epochs=1, group=1)
    monkeypatch.setattr(opd_mod, "_save_adapter", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "_publish_opd_deployable", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod._w, "write_train_meta", lambda **kw: metas.append(kw))

    opd_mod.run_opd()

    assert state["n"] == 2
    notes = metas[-1]["notes"]
    assert notes["opt_steps"] == 1
    assert notes["no_signal_resamples"] == 1
    assert notes["no_signal_skipped_steps"] == 0
    assert notes["skip_reasons"] == {"alignment_empty": 1}


def test_opd_no_signal_log_includes_skip_reasons(monkeypatch, capsys):
    """The skipped-step line must explain why the signal was unusable."""

    def _all_empty(**_k):
        return opd_mod.SampleResult(skip_reason="empty_completion")

    opd_mod = _opd_harness(monkeypatch, sample_result=_all_empty)
    with pytest.raises(RuntimeError, match="no trained step"):
        opd_mod.run_opd()

    out = capsys.readouterr().out
    assert "no usable teacher signal" in out
    assert "empty_completion" in out


def test_opd_emits_progress_heartbeat_while_filtering_prompts(monkeypatch):
    """Regression (codex[bot], opd.py:350): the prompt-budget filter scan runs after the last setup
    heartbeat and before model-load liveness; on a large split it can outlast the poller's setup grace.
    Pure-liveness pings don't reset that grace -- only progress heartbeats do -- so the scan must run
    under a liveness_heartbeat WITH a progress callback. Confirm an 'opd_filtering_prompts' stage is
    entered with a callable progress that advances."""
    import contextlib

    calls = []

    @contextlib.contextmanager
    def _fake_liveness(stage, progress=None, fields=None, **_kwargs):
        calls.append((stage, progress))
        yield

    opd_mod = _opd_harness(monkeypatch, sample_result=_skip, liveness=_fake_liveness)
    with pytest.raises(RuntimeError):  # all-skip -> no trained step, but filtering ran first
        opd_mod.run_opd()
    filt = [(s, p) for (s, p) in calls if s == "opd_filtering_prompts"]
    assert filt, f"filter scan must run under a liveness heartbeat; saw {[s for s, _ in calls]}"
    assert callable(filt[0][1]), "filter heartbeat needs a progress callback, not pure liveness"
    assert filt[0][1]() >= 1  # progress advanced as prompts were scanned


def test_opd_filtering_stage_is_setup_not_training():
    """Regression (codex[bot], _poll.py): opd_filtering_prompts emits REAL progress heartbeats, so
    is_training_heartbeat would classify it as TRAINING (the tight, sticky stall window) mid-setup
    unless it's registered as a setup stage. It must be treated as cold-start setup."""
    from flash.providers._poll import SETUP_HEARTBEAT_STAGES, is_training_heartbeat

    assert "opd_filtering_prompts" in SETUP_HEARTBEAT_STAGES
    assert is_training_heartbeat("opd_filtering_prompts", 0) is False
    assert (
        is_training_heartbeat("opd_filtering_prompts", 5) is False
    )  # progress count doesn't flip it


def test_opd_filtering_prompts_is_throttled_like_sft_pretokenizing():
    """Regression (codex[bot], heartbeat.py): opd_filtering_prompts emits a REAL progress heartbeat
    per liveness tick while it renders+tokenizes the whole split. Unthrottled that is one HF commit
    per tick -- ~120/hr on a large split before model load, blowing the 128/hr commit cap. It must be
    registered in BOTH heartbeat sets its SFT analogue sft_pretokenizing lives in: the throttle set
    (bounds commit rate) and the setup-liveness set (keeps the tighter cold-start upload cadence)."""
    from flash.engine.worker.heartbeat import _HB_SETUP_LIVENESS_STAGES, _HB_THROTTLED_STAGES

    assert "opd_filtering_prompts" in _HB_THROTTLED_STAGES
    assert "opd_filtering_prompts" in _HB_SETUP_LIVENESS_STAGES
    # parity with the SFT pre-tokenize stage this mirrors (same dual membership).
    assert "sft_pretokenizing" in _HB_THROTTLED_STAGES
    assert "sft_pretokenizing" in _HB_SETUP_LIVENESS_STAGES


def test_liveness_heartbeat_merges_fields_into_every_emission(monkeypatch):
    """Regression (codex[bot], heartbeat.py): the liveness thread emits stage=<stage> with NO step,
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


def test_opd_step_liveness_heartbeat_carries_opt_steps_in_fields(monkeypatch):
    """Regression (codex[bot], opd.py): opd must hand the opd_step liveness_heartbeat a `fields`
    callback that stamps the current opt_steps, so its (throttle-sharing) pings carry the billing step
    instead of overwriting the main thread's stepped heartbeat with a stepless one."""
    import contextlib

    captured = {}

    @contextlib.contextmanager
    def _fake_liveness(stage, progress=None, fields=None, **_kwargs):
        if stage == "opd_step":
            captured["fields"] = fields
        yield

    opd_mod = _opd_harness(monkeypatch, sample_result=_skip, liveness=_fake_liveness)
    with pytest.raises(RuntimeError):  # all-skip -> no trained step
        opd_mod.run_opd()
    assert callable(captured.get("fields")), (
        "opd_step liveness needs a fields callback carrying the current opt_steps"
    )
    out = captured["fields"]()
    assert isinstance(out, dict), f"fields callback must return a dict, got {out!r}"
    assert "step" in out, f"fields must carry the step, got {out}"
    # all-skip run reached 0 optimizer updates -> step is a real 0, not an absent/stale value.
    assert out["step"] == 0


def test_opd_teacher_prompt_includes_thinking_prefill():
    """Regression (codex[bot], opd.py:93): in thinking mode the student template opens a reasoning
    block (e.g. <think>) AFTER the generation prompt and samples its completion after it. The teacher
    must condition on that SAME trailing prefill; the plain 'Assistant: ' prompt (empty prefill) would
    score every thinking-mode logprob against a prefix that never opened the block."""
    from flash.engine.worker import opd as opd_mod

    msgs = [{"role": "user", "content": "hi"}]
    # default (thinking off / no prefill) -> ends at the plain generation boundary.
    assert opd_mod._teacher_prompt_text(msgs).endswith("Assistant: ")
    # with a prefill -> the teacher conditions on the exact text the student sampled after.
    assert opd_mod._teacher_prompt_text(msgs, "<think>\n").endswith("Assistant: <think>\n")


def test_thinking_prefill_text_is_template_delta(monkeypatch):
    """Regression (codex[bot], opd.py): the thinking prefill is the DELTA a thinking-mode chat template
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
    """Regression (codex[bot], opd.py): _thinking_prefill_text must handle a HYBRID template where the
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
    """Regression (codex[bot]/cursor, opd.py): a HYBRID template that disables thinking by force-CLOSING
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
    """Regression (codex[bot], opd.py): a hybrid whose disabled render is an EMPTY block WITH whitespace
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
    """Regression (codex[bot], opd.py:134): a closed-block hybrid whose disabled render closes IMMEDIATELY
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


def test_opd_loop_drives_by_optimizer_updates_and_fails_permanently_on_deterministic_shortfall(
    monkeypatch,
):
    """Regression (codex[bot], opd.py:467 + shortfall guard): the loop is driven by optimizer UPDATES,
    not raw iterations -- a no-signal iteration skips optimizer.step(), so `for step in range(steps)`
    could exit with opt_steps < steps and publish an under-trained adapter as the default while billing
    the full `steps` quote. when the shortfall is deterministic (updates land, then every sample skips
    with no transient teacher failure), the run must fail permanently, not RetriableInfraError: resuming
    the same cursor reproduces the insufficient successful-update rate and burns gpu on an unfixable run."""
    from flash.engine.worker.perf import RetriableInfraError

    torch = pytest.importorskip("torch")

    state = {"n": 0}

    def _one_update_then_skip(*, model, **k):
        from flash.engine.worker.opd import SampleResult

        state["n"] += 1
        # exactly one real, backward-able update lands first; then every sample skips (loss=None). The
        # teacher stays "ok" throughout, so teacher_transient == 0 -> the shortfall is deterministic.
        loss = model.w.float().sum() * 1e-6 if state["n"] == 1 else None
        return SampleResult(
            loss=loss, teacher_status="ok", coverage=1.0, gen_tokens=1, teacher_tokens=1
        )

    # epochs=3 but only ONE optimizer update can ever land -> the bounded loop exhausts its iteration
    # budget at opt_steps=1 and the post-loop guard must fail permanently (deterministic), not retry.
    opd_mod = _opd_harness(monkeypatch, sample_result=_one_update_then_skip, epochs=3, group=1)
    with pytest.raises(RuntimeError, match="deterministic") as ei:
        opd_mod.run_opd()
    assert not isinstance(ei.value, RetriableInfraError)  # permanent, not the retriable path
    assert "no optimizer step landed" not in str(ei.value)
    assert "insufficient successful-update rate" in str(ei.value)
    # the loop is bounded: it did not spin forever waiting for updates that never come.
    assert state["n"] <= 3 * 3 + 10


def test_opd_transient_teacher_shortfall_is_retriable(monkeypatch):
    """The OTHER branch of the shortfall guard: when a transient teacher outage (not deterministic
    local skips) caused opt_steps < steps, retry. One update lands, then teacher.score flakes
    transiently for the rest -> teacher_transient > 0 -> RetriableInfraError, because a healthier
    teacher next attempt may finish the run (codex[bot])."""
    from flash.engine.worker.perf import RetriableInfraError

    pytest.importorskip("torch")

    state = {"n": 0}

    def _one_update_then_transient(*, model, **k):
        from flash.engine.worker.opd import SampleResult

        state["n"] += 1
        if state["n"] == 1:  # one real update lands...
            return SampleResult(
                loss=model.w.float().sum() * 1e-6,
                teacher_status="ok",
                coverage=1.0,
                gen_tokens=1,
                teacher_tokens=1,
            )
        return SampleResult(teacher_status="transient", gen_tokens=1)  # ...then a retryable outage

    opd_mod = _opd_harness(monkeypatch, sample_result=_one_update_then_transient, epochs=3, group=1)
    with pytest.raises(RetriableInfraError, match="optimizer updates"):
        opd_mod.run_opd()


def test_student_tokens_absorb_dropped_leading_space_sentencepiece():
    """Regression (codex[bot], opd.py:175): a SentencePiece/LLaMA tokenizer decodes a mid-completion
    word token IN ISOLATION without its leading word-boundary space (decode([▁world]) == 'world', not
    ' world'). prev + len(decode(window)) would then undercount that span by one char and drift every
    following offset, misassigning teacher spans to the wrong sampled ids. Offsets must be anchored to
    completion_text so the dropped space is absorbed into the token's start and spans stay contiguous
    and exact."""
    from flash.engine.worker.opd import student_tokens_with_offsets

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
    """Regression (codex[bot], tokenizer_align.py:73): the cursor walk that replaced the per-boundary
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


def test_opd_installs_chalk_kernels_on_student(monkeypatch):
    """Regression (codex[bot], opd.py:433): the OPD HF/PEFT student drives the loss forward, so it
    must get chalk kernels like sft/rl build after their trainer — else the default
    Qwen catalog model silently runs eager and the distillation is much slower. Assert run_opd calls
    install_chalk_kernels on the built student model."""
    from flash.engine.worker import opd as _opd

    captured = {}

    def _fake_install(model=None):
        captured["model"] = model
        return {"rms_norm": {"applied": True}}

    monkeypatch.setattr(_opd, "install_chalk_kernels", _fake_install)
    monkeypatch.setattr(_opd, "active_kernels", lambda report: ["rms_norm"] if report else [])
    opd_mod = _opd_harness(monkeypatch, sample_result=_skip)
    with pytest.raises(RuntimeError):  # all-skip -> no trained step, but init (chalk) ran first
        opd_mod.run_opd()
    assert "model" in captured, "run_opd must call install_chalk_kernels on the student"
    assert captured["model"] is not None


def test_opd_wraps_training_loop_in_sdpa_cudnn_ctx(monkeypatch):
    """Regression (codex[bot], opd.py): on Blackwell (sm10x/sm120) optimal_attn_impl() returns 'sdpa';
    sft/rl wrap their forwards in _sdpa_cudnn_ctx(_attn) (rl.py:596) but opd only set attn_implementation
    at LOAD, so both the on-policy generate and the gkd loss forward ran under the default SDPA dispatch
    and silently lost the cuDNN kernel. run_opd must ENTER _sdpa_cudnn_ctx with the resolved _attn around
    the training loop."""
    import contextlib as _cl

    entered = {}

    @_cl.contextmanager
    def _rec_ctx(attn_impl):
        entered["attn"] = attn_impl
        yield

    opd_mod = _opd_harness(monkeypatch, sample_result=_skip)
    # Force the Blackwell branch and record what the loop wraps itself in.
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda *a, **k: "sdpa")
    monkeypatch.setattr(opd_mod, "_sdpa_cudnn_ctx", _rec_ctx)
    with pytest.raises(
        RuntimeError
    ):  # all-skip -> no trained step, but the ctx wrapped the loop first
        opd_mod.run_opd()
    assert entered.get("attn") == "sdpa", (
        "run_opd must wrap the training loop in _sdpa_cudnn_ctx(_attn) so the cuDNN SDPA backend is "
        "used on Blackwell (parity with sft/rl)"
    )


def test_opd_initializes_and_logs_to_wandb_when_configured(monkeypatch):
    """Regression (codex[bot], opd.py): sft/rl init W&B by passing report_to into the HF Trainer; opd's
    custom loop has no Trainer, so it must call wandb_report_to() (which CREATES the run) and log per
    optimizer step -- else wandb.run stays None (no dashboard) and the wandb_run_info() threaded into
    train_meta is empty. With W&B on, each landed optimizer step must log opd/loss + opd/coverage keyed
    by opt_steps."""
    import sys
    import types

    torch = pytest.importorskip("torch")

    logs = []
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.log = lambda data, step=None: logs.append((dict(data), step))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    state = {"n": 0}

    def _one_update_then_skip(*, model, **k):
        from flash.engine.worker.opd import SampleResult

        state["n"] += 1
        # one real optimizer step lands (W&B logs it), then the run shortfalls (loss=None)
        loss = model.w.float().sum() * 1e-6 if state["n"] == 1 else None
        return SampleResult(
            loss=loss, teacher_status="ok", coverage=1.0, gen_tokens=1, teacher_tokens=1
        )

    opd_mod = _opd_harness(monkeypatch, sample_result=_one_update_then_skip, epochs=3, group=1)
    # Turn W&B ON for this run (harness defaults it off): wandb_report_to() truthy -> _wandb_on.
    monkeypatch.setattr(opd_mod._w, "wandb_report_to", lambda: ["wandb"])
    # Deterministic shortfall fails permanently (not retriable); the one landed update still logs first.
    with pytest.raises(RuntimeError):
        opd_mod.run_opd()
    assert logs, "W&B on -> run_opd must call wandb.log for each landed optimizer step"
    data, step = logs[0]
    assert "opd/loss" in data
    assert "opd/coverage" in data
    assert step == 1, "wandb.log must be keyed by opt_steps (1 after the single update)"


def test_opd_skips_wandb_logging_when_not_configured(monkeypatch):
    """W&B OFF (no WANDB_API_KEY -> wandb_report_to() returns []): run_opd must NOT import/log to wandb,
    so a run completes its optimizer steps without touching the (here, exploding) wandb module."""
    import sys
    import types

    pytest.importorskip("torch")

    boom = types.ModuleType("wandb")

    def _explode(*a, **k):
        raise AssertionError("wandb.log must not be called when W&B is not configured")

    boom.log = _explode
    monkeypatch.setitem(sys.modules, "wandb", boom)

    state = {"n": 0}

    def _one_update_then_skip(*, model, **k):
        from flash.engine.worker.opd import SampleResult

        state["n"] += 1
        # one real, backward-able update lands first; then every sample skips (loss=None)
        loss = model.w.float().sum() * 1e-6 if state["n"] == 1 else None
        return SampleResult(
            loss=loss, teacher_status="ok", coverage=1.0, gen_tokens=1, teacher_tokens=1
        )

    opd_mod = _opd_harness(monkeypatch, sample_result=_one_update_then_skip, epochs=3, group=1)
    # Harness default wandb_report_to -> [] (off). The suppress(Exception) around wandb.log would hide a
    # raise, so the guard here is _wandb_on being False (wandb.log never reached), which _explode proves.
    with pytest.raises(RuntimeError):
        opd_mod.run_opd()


def test_run_opd_seeds_torch_before_building_student_model(monkeypatch):
    """Regression (codex[bot], opd.py): _student_model builds the LoRA via get_peft_model, which
    samples the LoRA A matrix (init_lora_weights=True) from the torch default generator. run_opd must
    seed torch BEFORE that call, else the fixed Flash seed can't reproduce the adapter init
    run-to-run. Record the order of torch.manual_seed vs the _student_model call."""
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    order: list[str] = []

    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0

        def apply_chat_template(self, messages, **kw):
            return "PROMPT"

        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[1, 2])  # within budget

        def decode(self, ids, skip_special_tokens=True):
            return "".join("x" for _ in ids)

    class _Model(_TinyLM):
        def __init__(self):
            super().__init__(torch, T=4, V=8)
            self.config = SimpleNamespace(use_cache=False)

    env = SimpleNamespace(
        dataset=lambda: [{"q": "a"}],
        prompt_messages=lambda ex: [{"role": "user", "content": ex["q"]}],
    )
    fake_w = SimpleNamespace(
        require_active_env=lambda: env,
        JOB_SPEC=SimpleNamespace(
            train=SimpleNamespace(init_from_adapter=""),
            model="fake/model",
            gpu=SimpleNamespace(type=None, exact_type=""),
        ),
        THINKING=False,
        SEED=1234,
        OPD_RESUME_REVISION="",
        heartbeat=lambda stage, **kw: None,
        prefetch_model=lambda mid: 0.0,
        hf_resume_checkpoint=lambda **_kwargs: "",
        publish_deployable_checkpoint=lambda *a, **k: None,
        hf_upload_folder=lambda *a, **k: None,
        write_train_meta=lambda **k: None,
        wandb_report_to=lambda: [],  # W&B off by default in unit tests
        wandb_run_info=lambda: {},
    )
    monkeypatch.setattr(opd_mod, "_w", fake_w)
    monkeypatch.setattr(
        opd_mod,
        "_resolve_opd_knobs",
        lambda: opd_mod.OpdKnobs(
            teacher_model="accounts/fireworks/models/glm-5p2",
            teacher_base_url="http://teacher.invalid",
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

    def _rec_student(*a, **k):
        order.append("student_model")
        return _Model(), "fake/model"

    monkeypatch.setattr(opd_mod, "_student_model", _rec_student)
    monkeypatch.setattr(opd_mod, "wait_for_gpu", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "setup_perf_backends", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "grad_checkpointing_on", lambda *a, **k: False)
    monkeypatch.setattr(opd_mod, "gpu_diagnostics", lambda *a, **k: {})
    _patch_opd_run_vllm_stub(monkeypatch, opd_mod, sample_result=_skip)

    real_manual_seed = torch.manual_seed

    def _rec_seed(s):
        order.append(f"manual_seed:{s}")
        return real_manual_seed(s)

    monkeypatch.setattr(torch, "manual_seed", _rec_seed)

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok())
    import flash.engine.worker.teacher as tmod

    monkeypatch.setattr(tmod, "TeacherClient", lambda *a, **k: object())
    monkeypatch.setenv("FIREWORKS_API_KEY", "unit-test-teacher-key")

    with pytest.raises(RuntimeError, match="no trained step"):
        opd_mod.run_opd()

    assert "student_model" in order
    assert f"manual_seed:{fake_w.SEED}" in order
    assert order.index(f"manual_seed:{fake_w.SEED}") < order.index("student_model"), (
        "torch.manual_seed(SEED) must run BEFORE _student_model builds the LoRA (LoRA A determinism)"
    )


def test_run_opd_releases_torch_cache_before_vllm_sizing(monkeypatch):
    """Warm-started OPD must release PyTorch's cached CUDA blocks before constructing vLLM.

    vLLM's EngineCore starts outside the trainer's allocator view, so reserved-but-unused PyTorch
    blocks can make vLLM's startup preflight fail even though the trainer could reuse that memory.
    """
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    order: list[str] = []

    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0

        def apply_chat_template(self, messages, **kw):
            return "PROMPT"

        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[1, 2])

        def decode(self, ids, skip_special_tokens=True):
            return "".join("x" for _ in ids)

    class _Model(_TinyLM):
        def __init__(self):
            super().__init__(torch, T=4, V=8)
            self.config = SimpleNamespace(use_cache=False)

    env = SimpleNamespace(
        dataset=lambda: [{"q": "a"}],
        prompt_messages=lambda ex: [{"role": "user", "content": ex["q"]}],
    )
    fake_w = SimpleNamespace(
        require_active_env=lambda: env,
        JOB_SPEC=SimpleNamespace(
            train=SimpleNamespace(init_from_adapter=""),
            model="fake/model",
            gpu=SimpleNamespace(type=None, exact_type=""),
        ),
        THINKING=False,
        SEED=1234,
        OPD_RESUME_REVISION="",
        heartbeat=lambda stage, **kw: None,
        prefetch_model=lambda mid: 0.0,
        hf_resume_checkpoint=lambda **_kwargs: "",
        publish_deployable_checkpoint=lambda *a, **k: None,
        hf_upload_folder=lambda *a, **k: None,
        write_train_meta=lambda **k: None,
        wandb_report_to=lambda: [],
        wandb_run_info=lambda: {},
    )
    monkeypatch.setattr(opd_mod, "_w", fake_w)
    monkeypatch.setattr(
        opd_mod,
        "_resolve_opd_knobs",
        lambda: opd_mod.OpdKnobs(
            teacher_model="accounts/fireworks/models/glm-5p2",
            teacher_base_url="http://teacher.invalid",
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
    monkeypatch.setattr(opd_mod, "wait_for_gpu", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "setup_perf_backends", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "grad_checkpointing_on", lambda *a, **k: False)
    monkeypatch.setattr(opd_mod, "gpu_diagnostics", lambda *a, **k: {})
    _patch_opd_run_vllm_stub(monkeypatch, opd_mod, sample_result=_skip)

    def _rec_student(*a, **k):
        order.append("student_model")
        return _Model(), "fake/model"

    def _rec_free_gpu(*a, **k):
        order.append("free_gpu")

    def _rec_vllm_kwargs(*a, **k):
        order.append("_opd_vllm_kwargs")
        return {
            "gpu_memory_utilization": 0.10,
            "kv_cache_dtype": None,
            "max_num_batched_tokens": None,
            "attention_backend": None,
            "mm_encoder_attn_backend": None,
            "enforce_eager": None,
            "compilation_config": None,
        }

    monkeypatch.setattr(opd_mod, "_student_model", _rec_student)
    monkeypatch.setattr(opd_mod, "free_gpu", _rec_free_gpu)
    monkeypatch.setattr(opd_mod, "_opd_vllm_kwargs", _rec_vllm_kwargs)

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok())
    import flash.engine.worker.teacher as tmod

    monkeypatch.setattr(tmod, "TeacherClient", lambda *a, **k: object())
    monkeypatch.setenv("FIREWORKS_API_KEY", "unit-test-teacher-key")

    with pytest.raises(RuntimeError, match="no trained step"):
        opd_mod.run_opd()

    assert order[:3] == ["student_model", "free_gpu", "_opd_vllm_kwargs"]


def test_opd_all_over_budget_prompts_fail_before_loading_student(monkeypatch):
    """Regression (codex[bot], opd.py): when every prompt exceeds the context budget the run fails
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
            gpu=SimpleNamespace(type=None, exact_type=""),
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
            teacher_model="accounts/fireworks/models/glm-5p2",
            teacher_base_url="http://teacher.invalid",
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
    monkeypatch.setenv("FIREWORKS_API_KEY", "unit-test-teacher-key")

    # The all-over-budget guard (RuntimeError) must fire; _student_model's AssertionError would
    # escape pytest.raises(RuntimeError) and fail the test (its "before the fix" behavior).
    with pytest.raises(RuntimeError, match="every prompt exceeds"):
        opd_mod.run_opd()
    # ...and the base-weight prefetch must have been deferred: an all-over-budget dataset fails without
    # paying for the tens-of-GB snapshot download (codex[bot]).
    assert prefetched == [], "prefetch_model must not run when every prompt is over budget"


def test_student_model_continues_warmstart_adapter_in_place(monkeypatch):
    """opd warm-start CONTINUES the prior adapter in place (VL and non-VL alike): _init_adapter_model
    returns a live trainable PeftModel + no fresh config (init_peft is None), and _student_model just
    moves it to the device and deploys that same adapter on the catalog base — no merge, no fresh
    LoRA, no recombine, and no base reload through either loader."""
    from flash.engine.worker import opd as opd_mod

    moved = []

    class _LiveModel:
        def to(self, device):
            moved.append(device)
            return self

    live = _LiveModel()

    def _fake_init_adapter_model(model_id):
        # Warm-start: a live trainable PeftModel continuing the prior (e.g. SFT) adapter, no config.
        return live, None

    monkeypatch.setattr(opd_mod._w, "_init_adapter_model", _fake_init_adapter_model, raising=False)
    # Neither loader may be touched for a continue-in-place warm-start (both raise if called).
    calls = _install_student_loader_fakes(monkeypatch, causal_raises=True, vl_raises=True)

    out, rollout_model_source = opd_mod._student_model("Qwen/Qwen3.5-4B", {"dtype": "bf16"}, "cpu")
    assert out is live  # the same live adapter, moved to device
    assert moved == ["cpu"]
    assert rollout_model_source == "Qwen/Qwen3.5-4B"  # continued adapter deploys on the catalog base
    assert calls == []  # no base reload: neither causal nor VL loader is used


def test_publish_opd_deployable_deploys_adapter_dir(tmp_path, monkeypatch):
    """opd CONTINUES the one warm-started adapter in place, so publish deploys adapter_dir directly
    (no recombine). as_default=False publishes only the step checkpoint; as_default=True also uploads
    the served default."""
    from flash.engine.worker import opd as opd_mod

    calls = {"upload": [], "publish": []}
    monkeypatch.setattr(
        opd_mod._w,
        "hf_upload_folder",
        lambda d, sub, required=False: calls["upload"].append((d, sub)),
        raising=False,
    )
    monkeypatch.setattr(
        opd_mod._w,
        "publish_deployable_checkpoint",
        lambda d, step: calls["publish"].append((d, step)),
        raising=False,
    )

    adir = str(tmp_path / "adapter")
    opd_mod._publish_opd_deployable(adir, 7, as_default=False)
    assert calls["upload"] == []  # as_default=False -> no served-default upload
    assert calls["publish"] == [(adir, 7)]  # deploys adapter_dir directly

    opd_mod._publish_opd_deployable(adir, 42, as_default=True)
    assert calls["upload"] == [(adir, "adapter")]  # served default = adapter_dir directly
    assert calls["publish"] == [(adir, 7), (adir, 42)]


def test_publish_opd_deployable_best_effort_survives_publish_failure(tmp_path, monkeypatch):
    """Per-step publish is best-effort: a publish failure (e.g. a transient HF upload error) is
    swallowed so training continues; the strict finalize path re-raises."""
    from flash.engine.worker import opd as opd_mod

    def _boom(d, step):
        raise RuntimeError("HF upload failed")

    monkeypatch.setattr(opd_mod._w, "publish_deployable_checkpoint", _boom, raising=False)
    monkeypatch.setattr(
        opd_mod._w, "hf_upload_folder", lambda d, sub, required=False: None, raising=False
    )

    # best_effort=True (per-step): swallowed, training continues (no raise).
    opd_mod._publish_opd_deployable(str(tmp_path / "a"), 20, as_default=False, best_effort=True)
    # best_effort=False (finalize): fatal.
    with pytest.raises(RuntimeError, match="HF upload failed"):
        opd_mod._publish_opd_deployable(str(tmp_path / "a"), 100, as_default=True)


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


def test_opd_35b_vllm_rollout_routes_above_h200_to_b200():
    """35B OPD with colocated student vLLM routes above the old H200-sized OPD estimate."""
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
    assert 141 < need <= 180


def test_opd_35b_fp8_kv_admits_full_context_group1_on_b200():
    """L5 regression: OPD's colocated vLLM rollout reserves an fp8 KV cache on cc >= 8.9 hardware
    (worker/opd_vllm.py), but model_required_vram_gb sized that KV as bf16 — DOUBLE the real bytes —
    so a full-context (4096) group_size=1 35B OPD run that actually fits a 180 GB B200 was rejected
    ('no validated GPU has >= 185 GB VRAM'). Size the KV fp8 once the run is provably modern-card-only,
    so the B200-fitting config is admitted."""
    from flash.catalog import MODELS, vocab_size_for
    from flash.engine.vram import estimate_vram_gb, model_required_vram_gb
    from flash.providers.allocator import vram_headroom
    from flash.providers.base import cheapest_gpu

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
    assert need <= 180
    assert cheapest_gpu(need) == "B200"
    # Prove the fp8 KV sizing is what flips it: the bf16-KV estimate * headroom overflows the B200,
    # the fp8-KV estimate clears it (same shape as the GRPO resident-fit fp8 test).
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
    assert fp8 < bf16
    hr = vram_headroom()
    assert fp8 * hr <= 180 < bf16 * hr


def test_opd_fp8_kv_gate_does_not_downroute_below_the_fp8_ceiling():
    """The fp8-KV discount must apply only when a run can ONLY land on a modern (cc >= 8.9) card. A
    smaller OPD run that fits the 80 GB A100 (sm80, no fp8) must keep its bf16 KV sizing and its A100
    route — never dropping onto a card that would not actually use fp8 (and would then OOM)."""
    from flash.engine.vram import model_required_vram_gb
    from flash.providers.base import cheapest_gpu, max_non_fp8_kv_vram_gb, supports_fp8_kv

    train = {"max_completion_tokens": 128, "lora_rank": 32, "lora_alpha": 64}
    need = model_required_vram_gb("Qwen/Qwen3.5-2B", "opd", train=train, headroom=1.1)
    assert need <= max_non_fp8_kv_vram_gb()  # stays within the non-fp8 (<= 80 GB) band...
    assert not supports_fp8_kv(cheapest_gpu(need))  # ...on the A100 (sm80), which does NOT use fp8 KV


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


def test_opd_vram_budgets_dense_logit_backward_buffers():
    """Regression (codex[bot], vram.py): OPD's loss has no fused CE and, at the loss BACKWARD peak, holds
    the fp32 completion rows + their fp32 gradient AND the bf16 full-sequence logits + their bf16
    gradient. The estimate must budget the backward buffers too, not only the two forward ones — else a
    long-completion / large-vocab (248k) opd job under-budgets OPD loss backward and routes to a GPU
    that OOMs. Isolate the logit term via a vocab delta (base + activations are vocab-independent): it
    must equal the FORWARD+BACKWARD size, i.e. 2x the forward-only (seq*2 + completion*4)*vocab."""
    from flash.engine.vram import estimate_vram_gb

    seq, comp = 9216, 8192
    kw = {
        "seq_len": seq,
        "max_tokens": comp,
        "lora_rank": 16,
        "batch_size": 1,
        "group_size": 1,
    }
    v1, v2 = 100_000, 248_320
    delta = estimate_vram_gb(4.0, "opd", "bf16", vocab=v2, **kw) - estimate_vram_gb(
        4.0, "opd", "bf16", vocab=v1, **kw
    )
    forward_only = (
        (seq * 2 + comp * 4) * (v2 - v1) / 1e9
    )  # what the old fwd-only formula would grow by
    assert delta == pytest.approx(2 * forward_only, rel=1e-9), (
        "opd dense-logit budget must include the backward buffers (2x the forward-only reservation); "
        f"got delta={delta} GB, forward-only would be {forward_only} GB"
    )


def test_opd_vram_thinking_completion_default_not_underbudgeted():
    """With max_tokens unset, opd's logits term must use the OPD recipe completion default (thinking
    = max_completion_len_thinking, 1536), not the GRPO-style min(seq_len, 1024) fallback — else a
    thinking opd job is under-budgeted and can OOM."""
    from flash.engine.vram import estimate_vram_gb

    kw = {"seq_len": 4096, "vocab": 248_320, "lora_rank": 16}  # seq high so min(seq,1024)=1024
    non_think = estimate_vram_gb(4.0, "opd", "bf16", thinking=False, **kw)  # completion=512
    think = estimate_vram_gb(4.0, "opd", "bf16", thinking=True, **kw)  # completion=1536, not 1024
    assert think > non_think  # thinking's longer completion budgets strictly more logits


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


def test_opd_teacher_rate_matches_fireworks_glm5p2_input_price():
    """glm-5p2 (and the omitted-teacher default) price at Fireworks' $1.40/M input, not the old $0.90
    — opd echo-scoring bills input tokens from the submit-time quote."""
    from flash.cost.facts import teacher_price_per_1m

    assert teacher_price_per_1m("accounts/fireworks/models/glm-5p2")[0] == 1.40
    assert teacher_price_per_1m("")[0] == 1.40  # omitted teacher -> representative default rate


def test_opd_teacher_price_table_covers_every_allowlisted_teacher():
    """Every allow-listed teacher is priced by its exact row (pricing routes through resolve_teacher
    over recipe.TEACHER_MODELS, so there is no unpriced teacher), the new teachers carry their own
    input prices (not silently GLM-priced), and an unknown teacher falls back to the default rate."""
    from flash.cost.facts import teacher_price_per_1m
    from flash.engine.recipe import TEACHER_MODELS

    # One exact price per allow-listed teacher, looked up by its provider model id.
    for info in TEACHER_MODELS.values():
        assert teacher_price_per_1m(info.model_id) == info.usd_per_1m

    # The two added teachers carry their own input prices (distinct from GLM's $1.40/M).
    assert teacher_price_per_1m("accounts/fireworks/models/deepseek-v4-pro")[0] == 1.74
    assert teacher_price_per_1m("accounts/fireworks/models/kimi-k2p6")[0] == 0.95
    # Removed teachers (qwen-3.7-max on-demand only; minimax-m3 no echo support) are unknown ids
    # now -> priced defensively at the default (GLM) rate.
    assert teacher_price_per_1m("accounts/fireworks/models/qwen3p7-max")[0] == 1.40
    assert teacher_price_per_1m("accounts/fireworks/models/minimax-m3")[0] == 1.40

    # An unknown teacher id falls back defensively to the default (GLM) rate.
    assert teacher_price_per_1m("accounts/fireworks/models/does-not-exist")[0] == 1.40


# --------------------------------------------------------------------------------------------------
# teacher client (mocked HTTP)
# --------------------------------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mock_urlopen(monkeypatch, payload, capture=None):
    import flash.engine.worker.teacher as tm

    def fake_urlopen(_transport, req, timeout=None):
        if capture is not None:
            capture["url"] = req.full_url
            capture["body"] = json.loads(req.data.decode())
        return _FakeResp(payload)

    monkeypatch.setattr(tm._ThreadLocalHttpsTransport, "urlopen", fake_urlopen)


def test_teacher_score_returns_completion_region_with_rebased_offsets_and_logprobs(monkeypatch):
    # prompt "P: " (len 3) + completion "hi" ; teacher tokens: "P", ":", " ", "hi".
    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["P", ":", " ", "hi"],
                    "token_logprobs": [0.0, -1.0, -2.0, -0.5],
                    "text_offset": [0, 1, 2, 3],
                }
            }
        ]
    }
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    toks = client.score("P: ", "hi")
    # only the completion token survives; offset rebased to the completion (start 0).
    assert len(toks) == 1
    assert toks[0].text == "hi"
    assert toks[0].start == 0
    assert toks[0].logprob == -0.5  # the realized-token logprob gkd consumes
    # scoring must not pay for generation, and asks for the minimal logprobs that return token_logprobs.
    assert capture["body"]["max_tokens"] == 0
    assert capture["body"]["echo"] is True
    assert capture["body"]["logprobs"] == 1


def test_teacher_score_many_sends_prompt_list_and_maps_choice_indexes(monkeypatch):
    payload = {
        "choices": [
            {
                "index": 1,
                "logprobs": {
                    "tokens": ["Q", "2", "B"],
                    "token_logprobs": [0.0, -1.0, -0.2],
                    "text_offset": [0, 1, 2],
                },
            },
            {
                "index": 0,
                "logprobs": {
                    "tokens": ["Q", "1", "A"],
                    "token_logprobs": [0.0, -1.0, -0.1],
                    "text_offset": [0, 1, 2],
                },
            },
        ]
    }
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient("k", "https://api.example/v1", "glm")

    out = client.score_many([("Q1", "A"), ("Q2", "B")])

    assert capture["body"] == {
        "model": "glm",
        "prompt": ["Q1A", "Q2B"],
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
        "temperature": 0,
    }
    assert [[t.text for t in toks] for toks in out] == [["A"], ["B"]]
    assert [out[0][0].logprob, out[1][0].logprob] == [-0.1, -0.2]


def test_teacher_score_many_multimodal_sends_nested_images_and_extracts_completion_suffix(
    monkeypatch,
):
    payload = {
        "choices": [
            {
                "index": 1,
                "logprobs": {
                    "tokens": ["User", ": ", "<expanded-image>", "\nAssistant:", " blue"],
                    "token_logprobs": [None, -0.1, -0.2, -0.3, -0.7],
                    "text_offset": [0, 4, 6, 106, 118],
                },
            },
            {
                "index": 0,
                "logprobs": {
                    "tokens": ["User", ": ", "<expanded-image>", "\nAssistant:", " red"],
                    "token_logprobs": [None, -0.1, -0.2, -0.3, -0.4],
                    "text_offset": [0, 4, 4, 104, 116],
                },
            },
        ]
    }
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient("k", "https://api.example/v1", "kimi")
    prompt = "User: <|media_pad|>\nAssistant: "

    scored = client.score_many_multimodal(
        [
            (prompt, "red", ["data:image/png;base64,red"]),
            (prompt, "blue", ["data:image/png;base64,blue"]),
        ]
    )

    assert capture["body"] == {
        "model": "kimi",
        "prompt": [prompt + "red", prompt + "blue"],
        "images": [
            ["data:image/png;base64,red"],
            ["data:image/png;base64,blue"],
        ],
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
        "temperature": 0,
    }
    assert [[token.text for token in tokens] for tokens in scored] == [[" red"], [" blue"]]
    assert [(scored[0][0].start, scored[0][0].end), (scored[1][0].start, scored[1][0].end)] == [
        (0, 3),
        (0, 4),
    ]
    assert [scored[0][0].logprob, scored[1][0].logprob] == [-0.4, -0.7]
    assert [scored[0].input_tokens, scored[1].input_tokens] == [5, 5]


def test_teacher_score_multimodal_single_request_uses_flat_image_list(monkeypatch):
    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["prompt", " answer"],
                    "token_logprobs": [None, -0.2],
                    "text_offset": [0, 100],
                }
            }
        ]
    }
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient("k", "https://api.example/v1", "kimi")

    client.score_many_multimodal(
        [("prompt<|media_pad|>", "answer", ["data:image/png;base64,image"])]
    )

    assert capture["body"]["prompt"] == "prompt<|media_pad|>answer"
    assert capture["body"]["images"] == ["data:image/png;base64,image"]


def test_teacher_multimodal_echo_drops_trailing_zero_width_token(monkeypatch):
    # a trailing zero-width token after the completion must not be scored as a completion token;
    # only tokens overlapping the completion region [0, len(completion)) count.
    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["prompt", " red", ""],
                    "token_logprobs": [None, -0.4, -0.9],
                    "text_offset": [0, 100, 104],
                }
            }
        ]
    }
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient("k", "https://api.example/v1", "kimi")

    scored = client.score_many_multimodal(
        [("prompt<|media_pad|>", "red", ["data:image/png;base64,image"])]
    )

    assert [token.text for token in scored[0]] == [" red"]
    assert (scored[0][0].start, scored[0][0].end) == (0, 3)
    assert scored[0][0].logprob == -0.4


@pytest.mark.parametrize(
    ("tokens", "logprobs", "offsets", "completion", "message"),
    [
        (["p", " x"], [None], [0, 100], "x", "length"),
        (["p", " x"], [None, None], [0, 100], "x", "null"),
        (["p", " x"], [None, float("nan")], [0, 100], "x", "non-finite"),
        (["p", " x"], [None, 0.2], [0, 100], "x", "positive"),
        (["p", " y"], [None, -0.2], [0, 100], "x", "exact completion suffix"),
    ],
)
def test_teacher_multimodal_echo_validator_rejects_bad_completion_contract(
    monkeypatch, tokens, logprobs, offsets, completion, message
):
    from flash.engine.worker.teacher import TeacherError

    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": logprobs,
                    "text_offset": offsets,
                }
            }
        ]
    }
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient("k", "https://api.example/v1", "kimi")

    with pytest.raises(TeacherError, match=message) as exc_info:
        client.score_many_multimodal(
            [("prompt<|media_pad|>", completion, ["data:image/png;base64,image"])]
        )
    assert exc_info.value.permanent is True


def test_teacher_transport_reuses_connection_and_reconnects_after_eof(monkeypatch):
    import http.client

    import flash.engine.worker.teacher as tm

    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["P", "hi"],
                    "token_logprobs": [0.0, -0.5],
                    "text_offset": [0, 1],
                }
            }
        ]
    }
    instances = []
    delays = []

    class _Socket:
        def settimeout(self, timeout):
            self.timeout = timeout

    class _Response:
        status = 200
        reason = "OK"
        will_close = False

        def __init__(self):
            self.headers = {}

        def read(self):
            return json.dumps(payload).encode()

        def close(self):
            pass

    class _Connection:
        def __init__(self, host, port, timeout):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.sock = _Socket()
            self.request_count = 0
            instances.append(self)

        def request(self, method, selector, body=None, headers=None):
            self.request_count += 1
            if len(instances) == 1 and self.request_count == 3:
                raise http.client.RemoteDisconnected("stale keep-alive socket")

        def getresponse(self):
            return _Response()

        def close(self):
            self.sock = None

    monkeypatch.setattr(tm.http.client, "HTTPSConnection", _Connection)
    monkeypatch.setattr(tm.time, "sleep", delays.append)
    client = TeacherClient("k", "https://api.example/v1", "glm", max_retries=2)

    client.score("P", "hi")
    client.score("P", "hi")
    client.score("P", "hi")
    client.score("P", "hi")

    assert len(instances) == 2
    assert instances[0].request_count == 3
    assert instances[1].request_count == 2
    assert delays == [2.0]


def test_teacher_score_keeps_boundary_crossing_token_clamped_to_completion(monkeypatch):
    # Prompt ends in whitespace ("P: ", plen=3); the teacher emits a leading-space merge token
    # " hi" that starts at char 2 (inside the prompt) and ends at 5 (inside the completion). Rather
    # than DROP it — which for a one-token completion would leave zero teacher tokens and skip the
    # sample — score() KEEPS it with its completion span clamped to [0, end-plen) so the first
    # completion token still carries a teacher logprob. Only tokens ENTIRELY in the prompt (end<=plen)
    # are dropped.
    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["P", ":", " hi", "!"],
                    "token_logprobs": [0.0, -1.0, -0.5, -0.2],
                    "text_offset": [0, 1, 2, 5],
                }
            }
        ]
    }
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    toks = client.score("P: ", "hi!")
    # "P" and ":" lie entirely in the prompt (end <= plen=3) -> dropped. The boundary-crossing " hi"
    # is kept, clamped to completion span [0, 2); "!" keeps [2, 3). "hi!" is fully covered.
    assert [t.text for t in toks] == [" hi", "!"]
    assert (toks[0].start, toks[0].end) == (0, 2)  # max(0, 2-3)=0 ; 5-3=2
    assert (toks[1].start, toks[1].end) == (2, 3)  # 5-3=2 ; 6-3=3
    assert toks[0].logprob == -0.5  # the merged token's realized logprob is preserved


def test_teacher_score_raises_on_malformed_response(monkeypatch):
    from flash.engine.worker.teacher import TeacherError

    _mock_urlopen(monkeypatch, {"choices": [{"logprobs": {}}]})
    client = TeacherClient("k", "https://api.example/v1", "glm")
    with pytest.raises(TeacherError) as ei:
        client.score("", "hi")
    assert ei.value.permanent is True  # malformed response -> abort the run, not skip-and-burn


def test_teacher_score_treats_mismatched_array_lengths_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): a 200 response whose tokens / token_logprobs / text_offset
    arrays disagree in length would IndexError inside the per-token loop and escape as a generic
    (non-TeacherError) exception, so a teacher that consistently returns malformed arrays could burn
    every OPD step before the run fails with "no trained step". A length mismatch is a broken contract
    -> PERMANENT (abort now)."""
    from flash.engine.worker.teacher import TeacherError

    def _payload(tokens, logprobs, offsets):
        return {
            "choices": [
                {"logprobs": {"tokens": tokens, "token_logprobs": logprobs, "text_offset": offsets}}
            ]
        }

    # Both directions of length disagreement are a broken contract and must abort PERMANENTLY:
    #  - logprobs SHORTER than tokens -> the per-token loop IndexErrors.
    #  - logprobs/offsets LONGER than tokens -> n=len(tokens) silently ignores the tail AND the last
    #    token (i==n-1) takes end=len(full), reinterpreting a mid-string token as spanning the whole
    #    completion and training on the wrong logprob (codex[bot]). `!=` (not `< n`) catches both.
    cases = [
        (["a", "b", "c"], [0.0, -1.0], [0, 1, 2]),  # 2 logprobs < 3 tokens
        (["a", "b"], [0.0, -1.0, -2.0], [0, 1]),  # 3 logprobs > 2 tokens (tail ignored)
        (["a", "b"], [0.0, -1.0], [0, 1, 2]),  # 3 offsets > 2 tokens
    ]
    for tokens, logprobs, offsets in cases:
        _mock_urlopen(monkeypatch, _payload(tokens, logprobs, offsets))
        client = TeacherClient("k", "https://api.example/v1", "glm")
        with pytest.raises(TeacherError) as ei:
            client.score("", "".join(tokens))
        assert ei.value.permanent is True, f"{(tokens, logprobs, offsets)} must be PERMANENT"
        assert "length" in str(ei.value).lower()


def test_teacher_score_rejects_null_logprob_on_completion_token_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): a null (None) realized logprob is legitimate ONLY for
    unscored PROMPT context. A None on a token that overlaps the COMPLETION (the ones score() keeps)
    means the teacher did not score it; coercing it to 0.0 (log-prob 1.0 == full confidence) would
    train the gkd loss on fabricated teacher confidence, so it must abort like the other contract
    violations. A prompt-context null (dropped anyway) must NOT trip it."""
    from flash.engine.worker.teacher import TeacherError

    # prompt "P" (plen=1) + completion "hi". Token "hi" spans [1,3) (end>plen -> KEPT) with a null
    # logprob -> reject as permanent.
    bad = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["P", "hi"],
                    "token_logprobs": [0.0, None],  # completion token "hi" unscored -> abort
                    "text_offset": [0, 1],
                }
            }
        ]
    }
    _mock_urlopen(monkeypatch, bad)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is True
    assert "null" in str(ei.value).lower()

    # A PROMPT-context null (token entirely in the prompt, end<=plen) is fine: it's dropped, not kept.
    ok = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["P", "hi"],
                    "token_logprobs": [None, -0.5],  # prompt token "P" null (dropped); "hi" scored
                    "text_offset": [0, 1],
                }
            }
        ]
    }
    _mock_urlopen(monkeypatch, ok)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    toks = client.score("P", "hi")
    assert toks
    assert toks[0].logprob == -0.5


def test_teacher_4xx_is_permanent_but_5xx_is_transient(monkeypatch):
    import urllib.error

    import flash.engine.worker.teacher as tm
    from flash.engine.worker.teacher import TeacherError

    def raise_http(code):
        def fake_urlopen(_transport, req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, f"HTTP {code}", {}, None)

        monkeypatch.setattr(tm._ThreadLocalHttpsTransport, "urlopen", fake_urlopen)

    client = TeacherClient("k", "https://api.example/v1", "glm", max_retries=1)
    # 401 (bad key) is permanent -> raised immediately so the worker aborts, not burns every step.
    raise_http(401)
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is True
    # 503 is transient -> retries exhaust to a non-permanent error (a skipped sample, run continues).
    raise_http(503)
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is False


def test_teacher_http_error_diagnostic_omits_opaque_response_body(monkeypatch):
    import io
    import traceback
    import urllib.error

    import flash.engine.worker.teacher as tm
    from flash.engine.worker.teacher import TeacherError

    private = b"opaque-private-teacher-sentinel-91ad"

    def raise_http(_transport, req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url,
            403,
            private.decode(),
            {},
            io.BytesIO(private),
        )

    monkeypatch.setattr(tm._ThreadLocalHttpsTransport, "urlopen", raise_http)
    client = TeacherClient("k", "https://api.example/v1", "glm", max_retries=1)

    with pytest.raises(TeacherError) as exc_info:
        client.score("P", "hi")

    detail = str(exc_info.value)
    formatted = "".join(
        traceback.format_exception(exc_info.type, exc_info.value, exc_info.tb)
    )
    assert exc_info.value.permanent is True
    assert "teacher HTTP 403" in detail
    assert "/completions" in detail
    assert "permanent" in detail
    assert private.decode() not in detail
    assert private.decode() not in formatted


def test_teacher_score_rejects_non_list_logprob_fields_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py:130): the length check assumes tokens/token_logprobs/
    text_offset are sequences. A malformed 200 with token_logprobs=null (or a scalar text_offset) makes
    len()/indexing raise TypeError OUTSIDE TeacherError, so a consistently malformed teacher could burn
    every OPD step. Non-list fields must raise a PERMANENT TeacherError up front."""
    from flash.engine.worker.teacher import TeacherError

    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["a", "b"],
                    "token_logprobs": None,  # malformed: null instead of a list
                    "text_offset": [0, 1],
                }
            }
        ]
    }
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    with pytest.raises(TeacherError) as ei:
        client.score("", "ab")
    assert ei.value.permanent is True
    assert "not all lists" in str(ei.value)


def test_teacher_malformed_200_body_is_transient_teacher_error(monkeypatch):
    """Regression (codex[bot], teacher.py:65): an HTTP 200 with a non-JSON body must surface as a
    TRANSIENT TeacherError, not a raw json.JSONDecodeError. A raw decode error escapes _post's except
    clauses, so a run hammered by malformed 200s could fail as permanent no-signal instead of retrying
    as teacher infra."""
    import flash.engine.worker.teacher as tm
    from flash.engine.worker.teacher import TeacherError

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"<html>502 Bad Gateway</html>"  # HTTP 200 status, non-JSON body

    monkeypatch.setattr(
        tm._ThreadLocalHttpsTransport,
        "urlopen",
        lambda _transport, req, timeout=None: _Resp(),
    )
    monkeypatch.setattr(tm.time, "sleep", lambda *a, **k: None)  # skip real backoff sleeps
    client = TeacherClient("k", "https://api.example/v1", "glm", max_retries=2)
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is False  # transient -> retried as infra, not permanent no-signal
    assert "unparseable" in str(ei.value).lower()


def test_teacher_incomplete_read_body_is_transient_teacher_error(monkeypatch):
    """Regression (codex[bot], teacher.py:65): an HTTP 200 whose body is truncated mid-read() raises
    http.client.IncompleteRead — an HTTPException, NOT an OSError — so without an explicit clause it
    escapes _post's retry loop, failing a truncated-200 run as permanent no-signal instead of retrying
    as infra."""
    import http.client

    import flash.engine.worker.teacher as tm
    from flash.engine.worker.teacher import TeacherError

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            raise http.client.IncompleteRead(b"half", 100)  # body truncated mid-read

    monkeypatch.setattr(
        tm._ThreadLocalHttpsTransport,
        "urlopen",
        lambda _transport, req, timeout=None: _Resp(),
    )
    monkeypatch.setattr(tm.time, "sleep", lambda *a, **k: None)  # skip real backoff sleeps
    client = TeacherClient("k", "https://api.example/v1", "glm", max_retries=2)
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is False  # transient -> retried as infra, not permanent no-signal
    assert "truncated" in str(ei.value).lower()


def test_teacher_score_rejects_non_numeric_or_unordered_offsets_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): a malformed 200 can put a value in text_offset that passes
    the list/length guards yet corrupts the alignment: non-numeric (null/string) or out-of-order (as
    before), and also non-finite (int(NaN) RAISES outside TeacherError -> unclassified skip), fractional
    (int() silently truncates to a wrong char index), or out-of-[0, len(full)] (a span outside the
    completion region). All must be rejected up front as PERMANENT so the worker aborts, not skip-burns.
    full = 'P' + 'hi' = 'Phi', len 3."""
    from flash.engine.worker.teacher import TeacherError

    def _payload(offsets):
        return {
            "choices": [
                {
                    "logprobs": {
                        "tokens": ["P", "hi"],
                        "token_logprobs": [0.0, -0.5],
                        "text_offset": offsets,
                    }
                }
            ]
        }

    for bad, needle in (
        ([0, None], "non-numeric"),
        ([0, "x"], "non-numeric"),
        ([2, 1], "non-decreasing"),  # both in-range so the order check (not range) is what fires
        ([0, 1.5], "not an integer"),  # fractional -> int() truncates to a wrong index
        ([0, float("nan")], "not finite"),  # NaN -> int(NaN) raises outside TeacherError
        ([0, 9], "outside"),  # 9 > len('Phi')=3 -> span past the string
        ([-1, 0], "outside"),  # negative start -> span before the completion
    ):
        _mock_urlopen(monkeypatch, _payload(bad))
        client = TeacherClient("k", "https://api.example/v1", "glm")
        with pytest.raises(TeacherError) as ei:
            client.score("P", "hi")
        assert ei.value.permanent is True, f"{bad!r} must be PERMANENT"
        assert needle in str(ei.value).lower(), f"{bad!r}: expected {needle!r} in {ei.value}"


def test_teacher_score_rejects_non_numeric_or_nonfinite_logprobs_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): token_logprobs[i] is coerced with float(...) below (None ->
    0.0 for a null realized logprob). A malformed 200 can still carry a non-numeric value (float() raises
    ValueError OUTSIDE TeacherError and can burn every OPD step) or a non-finite NaN/inf (feeds a
    poisoned gradient straight into the gkd loss). Both must be
    rejected up front as PERMANENT; a null logprob stays allowed (handled as 0.0)."""
    from flash.engine.worker.teacher import TeacherError

    def _payload(logprobs):
        return {
            "choices": [
                {
                    "logprobs": {
                        "tokens": ["P", "hi"],
                        "token_logprobs": logprobs,
                        "text_offset": [0, 1],
                    }
                }
            ]
        }

    for bad, needle in (
        ([0.0, "x"], "non-numeric"),
        ([0.0, float("nan")], "non-finite"),
        ([0.0, float("inf")], "non-finite"),
    ):
        _mock_urlopen(monkeypatch, _payload(bad))
        client = TeacherClient("k", "https://api.example/v1", "glm")
        with pytest.raises(TeacherError) as ei:
            client.score("P", "hi")
        assert ei.value.permanent is True, f"{bad!r} must be PERMANENT"
        assert needle in str(ei.value).lower(), f"{bad!r}: expected {needle!r} in {ei.value}"

    # A NULL realized logprob (first token) is legitimate and must NOT raise -> it becomes 0.0.
    _mock_urlopen(monkeypatch, _payload([None, -0.5]))
    client = TeacherClient("k", "https://api.example/v1", "glm")
    toks = client.score("P", "hi")
    assert toks  # completion token survives; null prompt logprob dropped
    assert toks[0].logprob == -0.5


def test_teacher_score_rejects_positive_logprob_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): a log-probability cannot exceed 0. A malformed 200 with a
    POSITIVE token_logprob is a probability > 1; summed into teacher_logsum it poisons the reverse-KL
    coefficient with impossible teacher mass, so OPD would train on a bogus signal instead of aborting.
    Reject as PERMANENT like the other teacher-contract violations. A ~0 logprob (near-deterministic
    token) stays allowed via the small float-rounding tolerance."""
    from flash.engine.worker.teacher import TeacherError

    def _payload(logprobs):
        return {
            "choices": [
                {
                    "logprobs": {
                        "tokens": ["P", "hi"],
                        "token_logprobs": logprobs,
                        "text_offset": [0, 1],
                    }
                }
            ]
        }

    # A clearly-positive completion logprob (prob e^2.5 >> 1) is rejected as PERMANENT.
    _mock_urlopen(monkeypatch, _payload([0.0, 2.5]))
    client = TeacherClient("k", "https://api.example/v1", "glm")
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is True
    assert "positive" in str(ei.value).lower()

    # A ~0 logprob within float-rounding tolerance (near-deterministic token) is NOT rejected.
    _mock_urlopen(monkeypatch, _payload([None, 1e-9]))
    client = TeacherClient("k", "https://api.example/v1", "glm")
    toks = client.score("P", "hi")
    assert toks  # completion token survives
    assert abs(toks[0].logprob - 1e-9) < 1e-12


def test_teacher_score_rejects_truncated_echo_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): a malformed 200 with equal-length arrays can still OMIT a
    suffix of `full`. The final token's end falls back to len(full), so a truncated echo stretches the
    last returned token across text the teacher never scored (and, if that token sits in the prompt,
    drags it across the boundary into the completion) — a fabricated span. The echoed tokens must tile
    the whole input, so a last token whose own text ends short of len(full) is rejected as PERMANENT.
    prompt 'P' (plen 1) + 'hello' = 'Phello' (len 6); an echo of only ['P','h'] ends at char 2."""
    from flash.engine.worker.teacher import TeacherError

    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["P", "h"],  # covers only "Ph"; omits "ello"
                    "token_logprobs": [0.0, -0.5],
                    "text_offset": [0, 1],
                }
            }
        ]
    }
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hello")
    assert ei.value.permanent is True
    assert "does not tile" in str(ei.value).lower()


def test_teacher_score_rejects_same_length_wrong_text_token_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): the tiling guard must compare the token TEXT to the echoed
    substring, not just its LENGTH. A malformed 200 echoing a same-length-but-different token over the
    right offsets (here 'XY' where full[1:3]=='hi') passes a length-only check yet trains the gkd loss
    on the wrong token's logprob. full 'P'+'hi'='Phi' (len 3); token 1 'XY' (len 2, == the span length)
    must still be rejected as PERMANENT because its text isn't the echoed substring."""
    from flash.engine.worker.teacher import TeacherError

    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["P", "XY"],  # len(XY)==len(hi)==2, tiles by length but wrong text
                    "token_logprobs": [0.0, -0.5],
                    "text_offset": [0, 1],
                }
            }
        ]
    }
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is True
    assert "does not tile" in str(ei.value).lower()


def test_teacher_score_rejects_interior_tiling_gap_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): the coverage guard must validate EVERY token boundary, not
    only the final one. An INTERIOR gap/overlap — offsets[i+1] != offsets[i] + len(tokens[i]) — makes the
    emit loop use offsets[i+1] as token i's end, assigning token i's logprob to text the teacher never
    scored (a fabricated completion span when the gap straddles plen). full 'P'+'hiyo' = 'Phiyo' (len 5);
    a mid-sequence offset jump (token 1 'h' ends at char 2 but the next offset is 3) must be PERMANENT."""
    from flash.engine.worker.teacher import TeacherError

    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["P", "h", "yo"],
                    "token_logprobs": [0.0, -0.3, -0.5],
                    "text_offset": [
                        0,
                        1,
                        3,
                    ],  # token 1 'h' ends at 2 but next offset is 3 -> gap at 2
                }
            }
        ]
    }
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hiyo")
    assert ei.value.permanent is True
    assert "does not tile" in str(ei.value).lower()


def test_teacher_score_rejects_echo_not_starting_at_offset_0_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): the tiling guard proves coverage of full[offsets[0]:] only —
    it never requires offsets[0] == 0. A malformed 200 that DROPS a prompt prefix and echoes a cleanly-
    tiling SUFFIX passes every offset/tiling check, but its completion logprobs were computed over a
    TRUNCATED prompt, so the gkd signal is scored against context the student never saw. full 'AB'+'cd' =
    'ABcd' (len 4); an echo of ['B','cd'] at offsets [1,2] tiles full[1:4] cleanly yet omits 'A' — the
    first offset is 1, not 0, and must be rejected as PERMANENT."""
    from flash.engine.worker.teacher import TeacherError

    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": [
                        "B",
                        "cd",
                    ],  # tiles full[1:4]=='Bcd' cleanly, but drops the 'A' prefix
                    "token_logprobs": [0.0, -0.5],
                    "text_offset": [1, 2],  # starts at 1, not 0
                }
            }
        ]
    }
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    with pytest.raises(TeacherError) as ei:
        client.score("AB", "cd")
    assert ei.value.permanent is True
    assert "offset 0" in str(ei.value).lower()


def test_teacher_score_rejects_echo_with_no_completion_tokens_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): an echo that yields NO completion-region token for a
    non-empty completion (here the degenerate empty-arrays 200) scored nothing to distil; score() must
    reject it as PERMANENT instead of returning an empty list that then burns every OPD step on no
    signal before the generic no-trained-step failure."""
    from flash.engine.worker.teacher import TeacherError

    payload = {"choices": [{"logprobs": {"tokens": [], "token_logprobs": [], "text_offset": []}}]}
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is True
    assert "no completion-region tokens" in str(ei.value).lower()


def test_score_many_preserves_mixed_text_and_image_sample_order():
    from flash.engine.worker import opd as opd_mod

    class _ImageTokens(list):
        def __init__(self, label, input_tokens):
            super().__init__([TeacherToken(label, -0.5, 0, 1)])
            self.input_tokens = input_tokens

    class _Teacher:
        def score_many(self, items):
            assert [completion for _prompt, completion in items] == ["t"]
            return [[TeacherToken("t", -0.1, 0, 1)]]

        def score_many_multimodal(self, items):
            assert [completion for _prompt, completion, _images in items] == ["a", "b"]
            return [_ImageTokens("a", 11), _ImageTokens("b", 12)]

    pendings = [
        opd_mod._Pending(
            gen=opd_mod._GenResult(completion_text="a"),
            prompt_ids=[1],
            prompt_messages=[{"role": "user", "content": "<|media_pad|>"}],
            teacher_images=("data:image/png;base64,a",),
        ),
        opd_mod._Pending(
            gen=opd_mod._GenResult(completion_text="t"),
            prompt_ids=[2],
            prompt_messages=[{"role": "user", "content": "text"}],
        ),
        opd_mod._Pending(
            gen=opd_mod._GenResult(completion_text="b"),
            prompt_ids=[3],
            prompt_messages=[{"role": "user", "content": "<|media_pad|>"}],
            teacher_images=("data:image/png;base64,b",),
        ),
    ]

    results = opd_mod._score_many(_Teacher(), pendings, thinking_prefill="")

    assert [result.teacher_toks[0].text for result in results] == ["a", "t", "b"]
    assert [result.teacher_input_tokens for result in results] == [11, None, 12]


def test_score_one_retries_same_completion_after_transient_teacher_failure():
    """A flaky teacher response should retry scoring the realized completion before OPD spends another
    GPU rollout."""

    from flash.engine.worker import opd as opd_mod
    from flash.engine.worker.teacher import TeacherError

    calls = {"n": 0}

    class _Teacher:
        def score(self, prompt, completion):
            calls["n"] += 1
            assert completion == "hi"
            if calls["n"] == 1:
                raise TeacherError("temporary 503")
            return [TeacherToken(text="hi", logprob=-1.0, start=0, end=2)]

    result = opd_mod._score_one(
        _Teacher(),
        opd_mod._GenResult(completion_text="hi"),
        prompt_messages=[{"role": "user", "content": "say hi"}],
        thinking_prefill="",
    )

    assert calls["n"] == 2
    assert result.status == "ok"
    assert result.teacher_toks


def test_score_many_deduplicates_exact_pairs_and_scatters_results():
    from flash.engine.worker import opd as opd_mod

    calls = []

    class _Teacher:
        def score_many(self, items):
            calls.append(list(items))
            return [["first-score"], ["second-score"]]

    shared_messages = [{"role": "user", "content": "same prompt"}]
    pendings = [
        opd_mod._Pending(
            gen=opd_mod._GenResult(completion_text="same completion"),
            prompt_ids=[],
            prompt_messages=shared_messages,
        ),
        opd_mod._Pending(
            gen=opd_mod._GenResult(completion_text="different completion"),
            prompt_ids=[],
            prompt_messages=shared_messages,
        ),
        opd_mod._Pending(
            gen=opd_mod._GenResult(completion_text="same completion"),
            prompt_ids=[],
            prompt_messages=shared_messages,
        ),
    ]

    results = opd_mod._score_many(_Teacher(), pendings, thinking_prefill="")

    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert results[0].teacher_toks == ["first-score"]
    assert results[1].teacher_toks == ["second-score"]
    assert results[2].teacher_toks == ["first-score"]


def test_teacher_http_error_with_unreadable_body_still_classified_by_code(monkeypatch):
    """Regression (codex[bot], teacher.py:62): a retryable 5xx whose error body is truncated makes
    e.read() raise IncompleteRead BEFORE last_err is set — without a guard it escapes _post as a generic
    exception before classification, so repeated retryable errors end as permanent no-signal. The
    preview read must be guarded and the error still classified by e.code."""
    import http.client
    import urllib.error

    import flash.engine.worker.teacher as tm
    from flash.engine.worker.teacher import TeacherError

    class _BadBodyHTTPError(urllib.error.HTTPError):
        def read(self, *a, **k):
            raise http.client.IncompleteRead(b"", 10)

    err = _BadBodyHTTPError("http://x", 503, "Service Unavailable", {}, None)
    err.fp = object()  # force the `if e.fp` branch so the guarded read() is attempted

    def raise_503(_transport, req, timeout=None):
        raise err

    monkeypatch.setattr(tm._ThreadLocalHttpsTransport, "urlopen", raise_503)
    monkeypatch.setattr(tm.time, "sleep", lambda *a, **k: None)  # skip real backoff
    client = TeacherClient("k", "https://api.example/v1", "glm", max_retries=2)
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert (
        ei.value.permanent is False
    )  # 503 retryable -> transient TeacherError, not a raw exception
    assert "503" in str(ei.value)


def test_resolve_opd_knobs_rejects_zero_kl_penalty(monkeypatch):
    """Regression (codex[bot], opd.py:64): kl_penalty_coef scales the gkd objective, so an explicit 0
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


def test_resolve_opd_knobs_resolves_teacher_from_train(monkeypatch):
    """_resolve_opd_knobs defensively re-resolves [train].teacher_model at the worker's (tolerant)
    deserialization boundary: parse already canonicalized it to a Fireworks model id, but the worker
    still validates — accepting an alias or the model id — so the TeacherClient sends a supported model.
    An unset value keeps the default GLM 5.2 teacher; the shared base_url is unchanged; an unsupported
    teacher fails loudly on the worker."""
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

    # A friendly alias resolves to the provider model id.
    assert _knobs("kimi-k2.6").teacher_model == "accounts/fireworks/models/kimi-k2p6"
    assert _knobs("deepseek-v4-pro").teacher_model == "accounts/fireworks/models/deepseek-v4-pro"
    # Unset / blank / None -> the default GLM 5.2 teacher (historical behavior preserved).
    assert _knobs("").teacher_model == "accounts/fireworks/models/glm-5p2"
    assert _knobs(None).teacher_model == "accounts/fireworks/models/glm-5p2"
    # base_url is shared across every allow-listed teacher (one Fireworks endpoint + one managed key).
    assert _knobs("deepseek-v4-pro").teacher_base_url == opd_mod.RECIPE.opd.teacher_base_url
    # An unsupported teacher fails loudly on the worker (defensive guard, mirrors the kl_coef check).
    with pytest.raises(RuntimeError, match="teacher_model"):
        _knobs("gpt-5.5")


def test_opd_loss_skips_empty_student_group_without_crashing():
    # A group with an empty student-index list (a teacher-only span) must be skipped, not divide by
    # zero in the per-span coefficient (len(s_idx) == 0).
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    rows = torch.zeros(1, 8, requires_grad=True)
    loss = opd_mod._gkd_loss_from_logits_rows(
        rows, [2], [([], -1.0), ([0], -2.0)], kl_coef=1.0
    )
    assert loss is not None  # the empty group is ignored; the real group still trains
    loss.backward()
    assert rows.grad is not None
    assert rows.grad[0].abs().sum() > 0


def test_groupwise_alignment_emits_no_empty_student_group():
    # Teacher covers [0,2) but the student's first token starts at char 2 (teacher-only leading
    # span). No group may have an empty student-index list.
    student = _student([(2, 3), (3, 5)])
    teacher = _teacher([(0, 2), (2, 5)])
    groups = groupwise_alignment(student, teacher)
    assert all(s_idx for s_idx, _ in groups)  # every group has >= 1 student token
    assert [s_idx for s_idx, _ in groups] == [[0, 1]]


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("FLASH_LIVE") != "1" or not os.environ.get("FIREWORKS_API_KEY"),
    reason="set FLASH_LIVE=1 and FIREWORKS_API_KEY to run the live Fireworks teacher test",
)
def test_live_kimi_multimodal_teacher_conditions_red_completion_on_image():
    image_module = pytest.importorskip("PIL.Image")

    def data_uri(color):
        image = image_module.new("RGB", (48, 48), color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    prompt = (
        "User: <|media_pad|>\nWhat color is this image? Reply with one lowercase word.\n"
        "Assistant: "
    )
    client = TeacherClient(
        os.environ["FIREWORKS_API_KEY"],
        "https://api.fireworks.ai/inference/v1",
        "accounts/fireworks/models/kimi-k2p6",
    )
    red_tokens, blue_tokens = client.score_many_multimodal(
        [
            (prompt, "red", [data_uri((220, 20, 20))]),
            (prompt, "red", [data_uri((20, 20, 220))]),
        ]
    )
    delta = sum(token.logprob for token in red_tokens) - sum(
        token.logprob for token in blue_tokens
    )

    assert delta > 0.05


def test_teacher_client_requires_key():
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
                "hf_repo": "owner/runs",
            },
        },
        run_id="x",
    )
    restored = JobSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.phase == "opd"
    # The teacher key is platform-managed (control-plane-injected into the worker env, like
    # HF_TOKEN) — NOT a user secret, so it is never added to environment.secrets.
    assert "FIREWORKS_API_KEY" not in restored.environment.secrets


def test_opd_cost_is_step_priced_and_bills_teacher_tokens():
    from flash.cost.spec import estimate_for_spec, spec_steps
    from flash.schema import spec_from_dict

    # No [train].max_examples set — opd falls back to one prompt batch per epoch for pricing.
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "opd",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 30, "hf_repo": "owner/runs"},
        },
        run_id="x",
    )
    assert spec_steps(spec) == 30
    est = estimate_for_spec(spec)
    assert est.method == "opd"
    assert est.teacher_api_usd > 0.0  # external teacher token spend is itemized (diagnostic)
    # Teacher tokens are billed by Fireworks to the platform-managed teacher key, tracked separately
    # from the GPU charge: total_usd is GPU (platform-billed) time only, never total + teacher.
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

    def __call__(self, input_ids):
        B = input_ids.shape[0]
        T = input_ids.shape[1]
        return SimpleNamespace(logits=self.w[:T].unsqueeze(0).expand(B, -1, -1))

    def parameters(self):
        return [self.w]

    def train(self, mode=True):  # _resolve_samples_batched flips the model into train mode
        return self


def test_opd_loss_backpropagates_over_grouped_spans():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    V = 8
    student_ids = [2, 3]  # 2 completion tokens
    rows = torch.zeros(len(student_ids), V, requires_grad=True)
    # One group covering both completion tokens (as when the teacher tokenizes them as one span).
    groups = [([0, 1], -1.5)]
    loss = opd_mod._gkd_loss_from_logits_rows(rows, student_ids, groups, kl_coef=1.0)
    assert loss is not None
    assert loss.requires_grad
    loss.backward()
    assert rows.grad[0].abs().sum() > 0
    assert rows.grad[1].abs().sum() > 0


def test_resolve_samples_batched_returns_differentiable_gkd_losses():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "".join({2: "a", 3: "b"}.get(int(i), "x") for i in ids)

    model = _TinyLM(torch, T=3, V=8)
    knobs = SimpleNamespace(kl_coef=1.0)
    samples = [
        (
            opd_mod._GenResult(completion_ids=[2], completion_text="a", gen_tokens=1),
            opd_mod._ScoreResult(teacher_toks=[TeacherToken("a", -0.5, 0, 1)], status="ok"),
            [1],
        ),
        (
            opd_mod._GenResult(completion_ids=[3], completion_text="b", gen_tokens=1),
            opd_mod._ScoreResult(teacher_toks=[TeacherToken("b", -0.7, 0, 1)], status="ok"),
            [1],
        ),
    ]

    out = opd_mod._resolve_samples_batched(model, _Tok(), "cpu", samples, knobs, microbatch=2)

    assert len(out) == 2
    assert all(r.loss is not None and r.loss.requires_grad for r in out)
    (sum(r.loss for r in out if r.loss is not None) / 2).backward()
    assert model.w.grad[0].abs().sum() > 0


def test_resolve_samples_batched_backprops_before_next_loss_microbatch():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "".join({2: "a", 3: "b"}.get(int(i), "x") for i in ids)

    class _GradObservedLM(_TinyLM):
        def __init__(self):
            super().__init__(torch, T=3, V=8)
            self.grad_seen_before_forward = []

        def __call__(self, input_ids):
            self.grad_seen_before_forward.append(
                bool(self.w.grad is not None and self.w.grad.abs().sum() > 0)
            )
            return super().__call__(input_ids)

    model = _GradObservedLM()
    knobs = SimpleNamespace(kl_coef=1.0)
    samples = [
        (
            opd_mod._GenResult(completion_ids=[2], completion_text="a", gen_tokens=1),
            opd_mod._ScoreResult(teacher_toks=[TeacherToken("a", -0.5, 0, 1)], status="ok"),
            [1],
        ),
        (
            opd_mod._GenResult(completion_ids=[3], completion_text="b", gen_tokens=1),
            opd_mod._ScoreResult(teacher_toks=[TeacherToken("b", -0.7, 0, 1)], status="ok"),
            [1],
        ),
    ]

    out = opd_mod._resolve_samples_batched(
        model, _Tok(), "cpu", samples, knobs, microbatch=1, backward_scale=0.5
    )

    assert [r.loss is not None and not r.loss.requires_grad for r in out] == [True, True]
    assert model.grad_seen_before_forward == [False, True]
    assert model.w.grad is not None
    assert model.w.grad.abs().sum() > 0


def test_resolve_samples_batched_uses_full_logits():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "".join({2: "a"}.get(int(i), "x") for i in ids)

    class _FullLM(_TinyLM):
        def __init__(self):
            super().__init__(torch, T=2, V=8)
            self.config = SimpleNamespace(use_cache=False)
            self.calls = []

        def train(self):
            return self

        def __call__(self, input_ids, **kwargs):
            self.calls.append(dict(kwargs))
            return super().__call__(input_ids)

    model = _FullLM()
    samples = [
        (
            opd_mod._GenResult(completion_ids=[2], completion_text="a", gen_tokens=1),
            opd_mod._ScoreResult(teacher_toks=[TeacherToken("a", -0.5, 0, 1)], status="ok"),
            [1],
        )
    ]

    out = opd_mod._resolve_samples_batched(
        model, _Tok(), "cpu", samples, SimpleNamespace(kl_coef=1.0), microbatch=1
    )

    assert out[0].loss is not None
    assert "logits_to_keep" not in model.calls[0]
    assert model._flash_opd_full_logits_batches == 1


def test_resolve_samples_batched_truncated_sample_has_no_forward_or_gradients():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        pad_token_id = 0

    class _NoForwardLM(_TinyLM):
        def __init__(self):
            super().__init__(torch, T=3, V=8)
            self.calls = 0

        def __call__(self, input_ids, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                logits=self.w[: input_ids.shape[1]]
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1, -1)
            )

    model = _NoForwardLM()
    sample = (
        opd_mod._GenResult(
            completion_ids=[2],
            completion_text="a",
            gen_tokens=1,
            truncated=True,
            finish_reason="length",
            skip_reason="truncated_rollout",
        ),
        opd_mod._ScoreResult(teacher_toks=[TeacherToken("a", -0.5, 0, 1)], status="ok"),
        [1],
    )

    out = opd_mod._resolve_samples_batched(
        model,
        _Tok(),
        "cpu",
        [sample],
        SimpleNamespace(kl_coef=1.0),
        microbatch=1,
    )[0]

    assert out.loss is None
    assert out.teacher_status is None
    assert out.truncated is True
    assert out.skip_reason == "truncated_rollout"
    assert model.calls == 0
    assert model.w.grad is None


def test_resolve_samples_batched_mixes_distillation_and_truncated_no_loss():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "".join({2: "a", 3: "b"}.get(int(i), "") for i in ids)

    class _CountingLM(_TinyLM):
        def __init__(self):
            super().__init__(torch, T=4, V=8)
            self.calls = 0

        def __call__(self, input_ids, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                logits=self.w[: input_ids.shape[1]]
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1, -1)
            )

    model = _CountingLM()
    samples = [
        (
            opd_mod._GenResult(
                completion_ids=[2], completion_text="a", gen_tokens=1, finish_reason="stop"
            ),
            opd_mod._ScoreResult(teacher_toks=[TeacherToken("a", -0.5, 0, 1)], status="ok"),
            [1],
        ),
        (
            opd_mod._GenResult(
                completion_ids=[2, 3],
                completion_text="ab",
                gen_tokens=2,
                truncated=True,
                finish_reason="length",
                skip_reason="truncated_rollout",
            ),
            None,
            [1],
        ),
    ]
    out = opd_mod._resolve_samples_batched(
        model,
        _Tok(),
        "cpu",
        samples,
        SimpleNamespace(kl_coef=1.0),
        microbatch=2,
        backward_scale=0.5,
    )

    assert [result.loss is not None for result in out] == [True, False]
    assert out[0].teacher_status == "ok"
    assert out[1].teacher_status is None
    assert out[1].truncated is True
    assert out[1].skip_reason == "truncated_rollout"
    assert model.calls == 1
    assert model.w.grad[0].abs().sum() > 0
    assert model.w.grad[1].abs().sum() == 0
    assert model.w.grad[2].abs().sum() == 0


def test_resolve_image_sample_lazily_processes_vision_inputs_and_extends_token_types():
    torch = pytest.importorskip("torch")
    image_module = pytest.importorskip("PIL.Image")
    pytest.importorskip("trl")
    from flash.engine.worker import opd as opd_mod
    from flash.multimodal import normalize_image_source

    processor_calls = []

    class _Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "".join({3: "a"}.get(int(i), "x") for i in ids)

    class _Processor:
        def apply_chat_template(self, **kwargs):
            processor_calls.append(kwargs)
            return {
                "input_ids": torch.tensor([[1, 99, 99, 2]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "pixel_values": torch.ones((4, 3)),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]]),
            }

    class _VisionLM:
        def __init__(self):
            self.w = torch.nn.Parameter(torch.randn(5, 8))
            self.config = SimpleNamespace(use_cache=False)
            self.calls = []

        def train(self):
            return self

        def __call__(self, input_ids, **kwargs):
            self.calls.append((input_ids.detach().clone(), kwargs))
            return SimpleNamespace(logits=self.w.unsqueeze(0))

        def parameters(self):
            return [self.w]

    descriptor = normalize_image_source(image_module.new("RGB", (2, 2), "red"), None)
    model = _VisionLM()
    sample = opd_mod._ImageLossSample(
        gen=opd_mod._GenResult(completion_ids=[3], completion_text="a", gen_tokens=1),
        score=opd_mod._ScoreResult(
            teacher_toks=[TeacherToken("a", -0.5, 0, 1)], status="ok"
        ),
        prompt_ids=[1, 99, 99, 2],
        student_messages=[{"role": "user", "content": [{"type": "image"}]}],
        descriptors=(descriptor,),
        processor=_Processor(),
        package_root=None,
        teacher_input_tokens=6,
    )

    assert processor_calls == []
    result = opd_mod._resolve_samples_batched(
        model,
        _Tok(),
        "cpu",
        [sample],
        SimpleNamespace(kl_coef=1.0),
        microbatch=8,
        backward_scale=1.0,
    )[0]

    assert result.loss is not None
    assert result.teacher_tokens == 6
    assert len(processor_calls) == 1
    assert processor_calls[0]["return_tensors"] == "pt"
    assert len(model.calls) == 1
    input_ids, kwargs = model.calls[0]
    assert input_ids.tolist() == [[1, 99, 99, 2, 3]]
    assert kwargs["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert kwargs["mm_token_type_ids"].tolist() == [[0, 1, 1, 0, 0]]
    assert kwargs["pixel_values"].shape == (4, 3)
    assert kwargs["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert model.w.grad is not None


def test_resolve_image_sample_rejects_rematerialized_prompt_id_mismatch(monkeypatch):
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "a"

    model = _TinyLM(torch, T=3, V=8)
    sample = opd_mod._ImageLossSample(
        gen=opd_mod._GenResult(completion_ids=[3], completion_text="a", gen_tokens=1),
        score=opd_mod._ScoreResult(
            teacher_toks=[TeacherToken("a", -0.5, 0, 1)], status="ok"
        ),
        prompt_ids=[1, 99, 2],
        student_messages=[{"role": "user", "content": [{"type": "image"}]}],
        descriptors=("descriptor",),
        processor=object(),
        package_root=None,
        teacher_input_tokens=4,
    )
    monkeypatch.setattr(
        opd_mod,
        "_materialize_image_prompt",
        lambda *_args: ([1, 99, 99, 2], {}),
    )

    with pytest.raises(RuntimeError, match="tokenization changed"):
        opd_mod._resolve_samples_batched(
            model,
            _Tok(),
            "cpu",
            [sample],
            SimpleNamespace(kl_coef=1.0),
            microbatch=1,
        )


def test_gkd_loss_from_logits_rows_matches_manual_logprob_math():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    rows = torch.tensor(
        [[0.2, -0.3, 0.7], [-0.5, 1.0, 0.1], [0.4, -0.2, 0.3]],
        dtype=torch.float32,
        requires_grad=True,
    )
    student_ids = [2, 1, 0]
    groups = [([0, 1], -0.75), ([2], -0.25)]

    loss = opd_mod._gkd_loss_from_logits_rows(rows, student_ids, groups, kl_coef=0.5)
    manual_logps = rows.gather(1, torch.tensor(student_ids).unsqueeze(1)).squeeze(1) - torch.logsumexp(
        rows, dim=-1
    )
    coeff0 = 0.5 * (manual_logps[:2].detach().sum() - groups[0][1]) / 2
    coeff1 = 0.5 * (manual_logps[2:].detach().sum() - groups[1][1])
    expected = torch.stack([coeff0 * manual_logps[0], coeff0 * manual_logps[1], coeff1 * manual_logps[2]]).mean()

    assert loss is not None
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert rows.grad is not None
    assert rows.grad.abs().sum() > 0

    rows2 = rows.detach().clone().requires_grad_(True)
    prepared = opd_mod._prepare_gkd_groups(groups)
    prepared_loss = opd_mod._gkd_loss_from_logits_rows(rows2, student_ids, prepared, kl_coef=0.5)
    assert prepared_loss is not None
    torch.testing.assert_close(prepared_loss, expected.detach())


def test_opd_loss_none_without_groups_or_tokens():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    rows = torch.zeros(2, 4, requires_grad=True)
    assert opd_mod._gkd_loss_from_logits_rows(rows, [2, 3], [], kl_coef=1.0) is None
    assert opd_mod._gkd_loss_from_logits_rows(rows[:0], [], [([0], -1.0)], kl_coef=1.0) is None


def test_opd_loss_coefficient_tracks_student_minus_teacher_logprob():
    # The per-span coefficient is (student_logsum.detach() - teacher_logsum)/|span|; a more
    # confident teacher (lower/more-negative teacher_logsum) makes the coefficient larger.
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    V = 8
    student_ids = [2]
    rows_hi = torch.zeros(1, V, requires_grad=True)  # uniform logits -> student logprob = -log V
    rows_lo = torch.zeros(1, V, requires_grad=True)
    hi = opd_mod._gkd_loss_from_logits_rows(
        rows_hi, student_ids, [([0], -5.0)], kl_coef=1.0
    )
    lo = opd_mod._gkd_loss_from_logits_rows(
        rows_lo, student_ids, [([0], -0.5)], kl_coef=1.0
    )
    # loss = coeff * student_logprob, student_logprob < 0, and coeff = (s_det - teacher)/1.
    # teacher=-5.0 -> larger coeff -> more-negative loss than teacher=-0.5.
    assert hi is not None
    assert lo is not None
    assert float(hi.detach()) < float(lo.detach())

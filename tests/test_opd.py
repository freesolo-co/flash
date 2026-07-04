"""On-policy distillation (opd): groupwise reverse-KL (gkd) cross-tokenizer alignment, the teacher
client, spec/cost plumbing, and the loss math.

All CPU-only. The loss-math tests need torch and are skipped where it is unavailable (they run in
CI, which has the training stack).
"""

from __future__ import annotations

import json
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


def _skip(**k):
    """A ``_train_one`` stub whose every sample skips: no loss, teacher not reached."""
    from flash.engine.worker.opd import SampleResult

    return SampleResult()


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
    assert ids == [1, 2, 3]  # ids trimmed in lockstep with the text (no gkd_loss/count desync)
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


def test_trim_trailing_stop_scans_from_end_not_quadratically():
    """Regression (codex[bot], opd.py:153): trimming the stop must scan from the END (a few decodes of
    the dropped tail), not decode every growing prefix ids[:1..n] — which was O(completion^2) and could
    dominate CPU before teacher scoring once [train].max_tokens is raised. Assert decode is called only
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


def test_opd_sampled_ids_moved_off_gpu_in_one_transfer():
    """Regression (codex[bot], opd.py:113): model.generate returns the sampled ids on the GPU;
    _to_cpu_ids must do ONE detach().cpu().tolist() bulk copy, not a per-token int(t) CUDA->CPU scalar
    sync (thousands of tiny syncs per sample once [train].max_tokens is raised)."""
    from flash.engine.worker.opd import _to_cpu_ids

    class _FakeGpuTensor:
        def __init__(self, data):
            self.data = list(data)
            self.detach_calls = 0
            self.iterated = 0

        def detach(self):
            self.detach_calls += 1
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self.data)

        def __iter__(self):
            self.iterated += 1  # element-wise iteration == the per-token CUDA-sync path to avoid
            return iter(self.data)

    t = _FakeGpuTensor([5, 6, 7])
    assert _to_cpu_ids(t) == [5, 6, 7]
    assert t.detach_calls == 1  # single bulk transfer
    assert t.iterated == 0  # never iterated element-by-element
    # a plain list (the already-trimmed path re-passes a list) is normalized, not treated as a tensor:
    assert _to_cpu_ids([1, 2, 3]) == [1, 2, 3]


def test_rollout_terminated_requires_eos_or_stop_not_length():
    """A rollout is safe to distil only if it terminated NATURALLY — EOS in the ids, or (with
    stop_sequences) the decoded text ends with a stop delimiter. A max_new_tokens cap hit OR a
    gen_cfg.max_time cut ends without either and is a partial mid-output fragment OPD must skip (it
    can't supervise the stop token). Length is NOT the criterion (codex[bot])."""
    from flash.engine.worker.opd import _rollout_terminated

    EOS = 99
    # EOS in the ids -> terminated (HF appends EOS when it stops on it), regardless of length.
    assert _rollout_terminated([1, 2, 3, EOS], "abc", EOS, ()) is True
    # no EOS, no stops -> NOT terminated: a cap hit OR a max_time cut, both partial fragments -> skip.
    assert _rollout_terminated([1, 2, 3, 4], "abcd", EOS, ()) is False  # cap hit, no EOS
    assert _rollout_terminated([1, 2], "ab", EOS, ()) is False  # short: max_time cut, no EOS/stop
    # stop delimiter is the trailing text -> terminated even without EOS AND even at the cap (codex#587).
    assert _rollout_terminated([1, 2, 3, 4], "ans</answer>", None, ("</answer>",)) is True
    # stop configured but text doesn't end with it, no EOS -> not terminated -> skip.
    assert _rollout_terminated([1, 2, 3, 4], "ans", None, ("</answer>",)) is False
    # no termination signal at all (no eos id, no stops) -> fail OPEN (distil, don't skip everything).
    assert _rollout_terminated([1, 2, 3, 4], "abcd", None, ()) is True


def test_opd_vram_sizing_uses_completion_budget_not_sft_default():
    # OPD generates on-policy (loss forward runs model(prompt+completion)), so allocator sizing must
    # use the prompt+completion budget, not the SFT 1024 default — else a raised max_tokens OOMs an
    # under-sized GPU.
    from flash.engine.vram import opd_rollout_seq_len

    assert opd_rollout_seq_len(0, None, False) == 1536  # 1024 prompt + 512 completion default
    assert opd_rollout_seq_len(0, 8192, False) == 9216  # raised max_tokens sizes up (was 1024)
    assert opd_rollout_seq_len(4096, 8192, False) == 4096  # explicit max_length pins the sequence


def test_opd_rejects_unpriced_teacher_model_but_accepts_priced():
    from flash.schema import ConfigError, spec_from_dict

    def _spec(teacher):
        return spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-4B",
                "algorithm": "opd",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"steps": 5, "hf_repo": "owner/runs", "teacher_model": teacher},
            },
            run_id="x",
        )

    _spec("accounts/fireworks/models/glm-5p2")  # priced -> ok
    with pytest.raises(ConfigError):
        _spec("accounts/fireworks/models/mystery-9000")  # unpriced override -> reject at parse


def test_opd_rejects_prompt_budget_at_parse_time_before_provisioning():
    """max_length <= max_tokens leaves no prompt budget; opd must reject it at spec-parse time
    (before a paid worker is provisioned), not only inside run_opd after GPU setup."""
    from flash.schema import ConfigError, spec_from_dict

    def _spec(train_extra):
        return spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-4B",
                "algorithm": "opd",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"steps": 5, "hf_repo": "owner/runs", **train_extra},
            },
            run_id="x",
        )

    # max_length leaves room after an explicit max_tokens -> ok.
    _spec({"max_length": 2048, "max_tokens": 512})
    # max_length <= max_tokens -> no prompt budget -> reject at parse.
    with pytest.raises(ConfigError, match="prompt budget"):
        _spec({"max_length": 400, "max_tokens": 512})
    # max_tokens omitted -> resolves to the opd recipe default (512); max_length below it -> reject.
    with pytest.raises(ConfigError, match="prompt budget"):
        _spec({"max_length": 256})


def test_train_one_full_loop_forwards_sampled_ids_and_ignores_zero_width_eos():
    """Exercise the PRODUCTION caller _train_one end-to-end (the direct-call unit test above can't
    catch a broken call site). The completion ends in a zero-width eos, so this also pins the
    coverage denominator: 2 alignable tokens fully covered -> 100%, not 2/3."""
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    completion_ids = [2, 3, 5]  # 2->'h', 3->'i', 5->eos (in-vocab id, decodes to '')

    class _Tok:
        eos_token_id = 5  # completion ends in id 5 -> _rollout_terminated sees natural EOS termination

        def decode(self, ids, skip_special_tokens=True):
            m = {2: "h", 3: "i", 5: ""}
            return "".join(m[int(x)] for x in ids)

    class _Teacher:
        def score(self, prompt, completion):  # one teacher token spanning all of "hi"
            return [TeacherToken(text="hi", logprob=-1.0, start=0, end=2)]

    class _GenLM(_TinyLM):
        def __init__(self, torch, prompt_len, completion_ids, V):
            super().__init__(torch, T=prompt_len + len(completion_ids), V=V)
            self.config = SimpleNamespace(use_cache=True)
            self._completion = torch.tensor([completion_ids])

        def eval(self):
            return self

        def train(self):
            return self

        def generate(self, prompt_tensor, **cfg):
            return torch.cat([prompt_tensor, self._completion], dim=1)

    model = _GenLM(torch, prompt_len=1, completion_ids=completion_ids, V=8)
    r = opd_mod._train_one(
        model=model,
        tok=_Tok(),
        teacher=_Teacher(),
        device="cpu",
        prompt_ids=[1],
        prompt_tensor=torch.tensor([[1]]),
        prompt_messages=[{"role": "user", "content": "say hi"}],
        gen_cfg={},
        knobs={"kl_coef": 1.0, "stop_sequences": ()},
        torch=torch,
    )
    assert r.loss is not None
    assert r.loss.requires_grad
    r.loss.backward()  # the sampled ids reached gkd_loss and produce a real gradient
    assert model.w.grad is not None
    assert model.w.grad.abs().sum() > 0
    # eos is zero-width and joins no group; coverage is over the 2 alignable tokens -> 100%.
    assert r.coverage == 1.0
    assert r.gen_tokens == 3


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
            gpu=SimpleNamespace(type=None),
        ),
        THINKING=False,
        SEED=0,
        heartbeat=lambda stage, **kw: beats.append((stage, kw)),
        prefetch_model=lambda mid: 0.0,
        hf_resume_checkpoint=lambda: "",
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
        lambda: {
            "teacher_model": "accounts/fireworks/models/glm-5p2",
            "teacher_base_url": "http://teacher.invalid",
            "steps": 1,
            "learning_rate": 1e-4,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_completion": 8,
            "prompts_per_step": 1,
            "group_size": 1,
            "kl_coef": 1.0,
            "save_every": 0,
            "max_length": 0,
            "stop_sequences": (),
        },
    )
    monkeypatch.setattr(opd_mod, "_student_model", lambda *a, **k: _Model())
    monkeypatch.setattr(opd_mod, "wait_for_gpu", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "setup_perf_backends", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "grad_checkpointing_on", lambda *a, **k: False)
    monkeypatch.setattr(opd_mod, "gpu_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(opd_mod, "_train_one", _skip)  # EVERY sample skips

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


def _opd_harness(monkeypatch, *, train_one, beats=None, liveness=None, steps=1, group=1):
    """Wire run_opd's fakes (torch student, tokenizer, teacher, deterministic knobs) for a 1-prompt
    loop and install the caller's _train_one stub. Returns the opd module."""
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
            gpu=SimpleNamespace(type=None),
        ),
        THINKING=False,
        SEED=0,
        heartbeat=(
            (lambda stage, **kw: beats.append((stage, kw)))
            if beats is not None
            else (lambda stage, **kw: None)
        ),
        prefetch_model=lambda mid: 0.0,
        hf_resume_checkpoint=lambda: "",
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
        lambda: {
            "teacher_model": "accounts/fireworks/models/glm-5p2",
            "teacher_base_url": "http://teacher.invalid",
            "steps": steps,
            "learning_rate": 1e-4,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_completion": 8,
            "prompts_per_step": 1,
            "group_size": group,
            "kl_coef": 1.0,
            "save_every": 0,
            "max_length": 0,
            "stop_sequences": (),
        },
    )
    monkeypatch.setattr(opd_mod, "_student_model", lambda *a, **k: _Model())
    monkeypatch.setattr(opd_mod, "wait_for_gpu", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "setup_perf_backends", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "grad_checkpointing_on", lambda *a, **k: False)
    monkeypatch.setattr(opd_mod, "gpu_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(opd_mod, "_train_one", train_one)
    if liveness is not None:
        monkeypatch.setattr(opd_mod, "liveness_heartbeat", liveness)
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok())
    import flash.engine.worker.teacher as tmod

    monkeypatch.setattr(tmod, "TeacherClient", lambda *a, **k: object())
    monkeypatch.setenv("FIREWORKS_API_KEY", "unit-test-teacher-key")
    return opd_mod


def test_opd_rejects_multi_turn_and_tool_environments(monkeypatch):
    """Regression (codex[bot], opd.py): opd samples one completion per prompt and cannot drive the
    turn/tool loop, so it must fail fast on a multi-turn or tool-calling env instead of silently
    distilling only the first assistant turn."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from flash.engine.worker import opd as opd_mod

    for attr in ("multi_turn", "is_tool_env"):
        env = SimpleNamespace(**{attr: True})
        monkeypatch.setattr(opd_mod, "_w", SimpleNamespace(require_active_env=lambda e=env: e))
        with pytest.raises(RuntimeError, match="multi-turn or tool"):
            opd_mod.run_opd()


def test_opd_liveness_heartbeat_gets_monotonic_progress_callback(monkeypatch):
    """Regression (codex[bot], opd.py): opd must hand liveness_heartbeat a progress callback (parity
    with sft/rl) so its thread emits REAL progress on sample advance instead of pure liveness=true
    pings that share — and can starve — the opd_step upload throttle. Confirm the progress arg is a
    callable that reflects the monotonic sample count."""
    import contextlib

    captured = {}

    @contextlib.contextmanager
    def _fake_liveness(stage, progress=None, fields=None):
        captured["stage"] = stage
        captured["progress"] = progress
        yield

    opd_mod = _opd_harness(monkeypatch, train_one=_skip, liveness=_fake_liveness)
    with pytest.raises(RuntimeError):  # all-skip -> no trained step
        opd_mod.run_opd()
    assert captured["stage"] == "opd_step"
    assert callable(captured["progress"]), "opd must pass a progress callback to liveness_heartbeat"
    # The callback reports the monotonic sample count. An all-skip run lands no optimizer update, so
    # the bounded-retry loop visits its full budget of max_iters = 2*steps + 10 = 12 fresh slices
    # (1 prompt x 1 group each) before the post-loop guard raises -> samples_seen advanced to 12.
    assert captured["progress"]() == 12


def test_opd_no_signal_from_transient_teacher_is_retriable(monkeypatch):
    """Regression (codex[bot], opd.py): a run where EVERY teacher.score fails transiently (a Fireworks
    outage spanning the run) and none succeed must raise a RetriableInfraError so the supervisor
    retries — not a plain RuntimeError, which it treats as permanent. A no-signal run where the
    teacher DID respond (but alignment yielded nothing) stays a permanent RuntimeError."""
    from flash.engine.worker.perf import RetriableInfraError

    def _all_transient(**k):
        return opd_mod.SampleResult(teacher_status="transient")

    opd_mod = _opd_harness(monkeypatch, train_one=_all_transient)
    with pytest.raises(RetriableInfraError, match="failed transiently"):
        opd_mod.run_opd()

    # contrast: teacher responded ("ok") but no loss -> permanent RuntimeError, NOT retriable.
    def _ok_no_align(**k):
        return opd_mod.SampleResult(teacher_status="ok")

    opd_mod = _opd_harness(monkeypatch, train_one=_ok_no_align)
    with pytest.raises(RuntimeError) as ei:
        opd_mod.run_opd()
    assert not isinstance(ei.value, RetriableInfraError)
    assert "no trained step" in str(ei.value)


def test_opd_emits_progress_heartbeat_while_filtering_prompts(monkeypatch):
    """Regression (codex[bot], opd.py:350): the prompt-budget filter scan runs after the last setup
    heartbeat and before model-load liveness; on a large split it can outlast the poller's setup grace.
    Pure-liveness pings don't reset that grace -- only progress heartbeats do -- so the scan must run
    under a liveness_heartbeat WITH a progress callback. Confirm an 'opd_filtering_prompts' stage is
    entered with a callable progress that advances."""
    import contextlib

    calls = []

    @contextlib.contextmanager
    def _fake_liveness(stage, progress=None, fields=None):
        calls.append((stage, progress))
        yield

    opd_mod = _opd_harness(monkeypatch, train_one=_skip, liveness=_fake_liveness)
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
    def _fake_liveness(stage, progress=None, fields=None):
        if stage == "opd_step":
            captured["fields"] = fields
        yield

    opd_mod = _opd_harness(monkeypatch, train_one=_skip, liveness=_fake_liveness)
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


def test_opd_loop_drives_by_optimizer_updates_and_retries_on_shortfall(monkeypatch):
    """Regression (codex[bot], opd.py:467): the loop must be driven by optimizer UPDATES, not raw
    iterations -- a no-signal iteration skips optimizer.step(), so `for step in range(steps)` could
    exit with opt_steps < steps and publish an under-trained adapter as the default while billing the
    full `steps` quote. A run that lands SOME updates but cannot reach `steps` within the bounded
    iteration budget must raise RetriableInfraError (retry), not ship short."""
    from flash.engine.worker.perf import RetriableInfraError

    torch = pytest.importorskip("torch")

    state = {"n": 0}

    def _one_update_then_skip(*, model, **k):
        from flash.engine.worker.opd import SampleResult

        state["n"] += 1
        # exactly one real, backward-able update lands first; then every sample skips (loss=None)
        loss = model.w.float().sum() * 1e-6 if state["n"] == 1 else None
        return SampleResult(loss=loss, teacher_status="ok", coverage=1.0, gen_tokens=1, teacher_tokens=1)

    # steps=3 but only ONE optimizer update can ever land -> the bounded loop exhausts its iteration
    # budget at opt_steps=1 and the post-loop guard must retry rather than publish 1/3.
    opd_mod = _opd_harness(
        monkeypatch, train_one=_one_update_then_skip, steps=3, group=1
    )
    with pytest.raises(RetriableInfraError, match="optimizer updates"):
        opd_mod.run_opd()
    # The loop is BOUNDED: it did not spin forever waiting for updates that never come.
    assert state["n"] <= 2 * 3 + 10


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
    """Regression (codex[bot], opd.py:433): the OPD student drives BOTH on-policy generation and the
    loss forward, so it must get chalk kernels like sft/rl build after their trainer — else the default
    Qwen catalog model silently runs eager and the distillation is much slower. Assert run_opd calls
    install_chalk_kernels on the built student model."""
    from flash.engine.worker import opd as _opd

    captured = {}

    def _fake_install(model=None):
        captured["model"] = model
        return {"rms_norm": {"applied": True}}

    monkeypatch.setattr(_opd, "install_chalk_kernels", _fake_install)
    monkeypatch.setattr(_opd, "active_kernels", lambda report: ["rms_norm"] if report else [])
    opd_mod = _opd_harness(monkeypatch, train_one=_skip)
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

    opd_mod = _opd_harness(monkeypatch, train_one=_skip)
    # Force the Blackwell branch and record what the loop wraps itself in.
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda *a, **k: "sdpa")
    monkeypatch.setattr(opd_mod, "_sdpa_cudnn_ctx", _rec_ctx)
    with pytest.raises(RuntimeError):  # all-skip -> no trained step, but the ctx wrapped the loop first
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
        return SampleResult(loss=loss, teacher_status="ok", coverage=1.0, gen_tokens=1, teacher_tokens=1)

    from flash.engine.worker.perf import RetriableInfraError

    opd_mod = _opd_harness(monkeypatch, train_one=_one_update_then_skip, steps=3, group=1)
    # Turn W&B ON for this run (harness defaults it off): wandb_report_to() truthy -> _wandb_on.
    monkeypatch.setattr(opd_mod._w, "wandb_report_to", lambda: ["wandb"])
    with pytest.raises(RetriableInfraError):  # 1/3 updates -> retry, after logging the one that landed
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
        return SampleResult(loss=loss, teacher_status="ok", coverage=1.0, gen_tokens=1, teacher_tokens=1)

    from flash.engine.worker.perf import RetriableInfraError

    opd_mod = _opd_harness(monkeypatch, train_one=_one_update_then_skip, steps=3, group=1)
    # Harness default wandb_report_to -> [] (off). The suppress(Exception) around wandb.log would hide a
    # raise, so the guard here is _wandb_on being False (wandb.log never reached), which _explode proves.
    with pytest.raises(RetriableInfraError):
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
            gpu=SimpleNamespace(type=None),
        ),
        THINKING=False,
        SEED=1234,
        heartbeat=lambda stage, **kw: None,
        prefetch_model=lambda mid: 0.0,
        hf_resume_checkpoint=lambda: "",
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
        lambda: {
            "teacher_model": "accounts/fireworks/models/glm-5p2",
            "teacher_base_url": "http://teacher.invalid",
            "steps": 1,
            "learning_rate": 1e-4,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_completion": 8,
            "prompts_per_step": 1,
            "group_size": 1,
            "kl_coef": 1.0,
            "save_every": 0,
            "max_length": 0,
            "stop_sequences": (),
        },
    )

    def _rec_student(*a, **k):
        order.append("student_model")
        return _Model()

    monkeypatch.setattr(opd_mod, "_student_model", _rec_student)
    monkeypatch.setattr(opd_mod, "wait_for_gpu", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "setup_perf_backends", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "grad_checkpointing_on", lambda *a, **k: False)
    monkeypatch.setattr(opd_mod, "gpu_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(opd_mod, "_train_one", _skip)

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
            gpu=SimpleNamespace(type=None),
        ),
        THINKING=False,
        SEED=0,
        heartbeat=lambda stage, **kw: None,
        prefetch_model=lambda mid: (prefetched.append(mid), 0.0)[1],
    )
    monkeypatch.setattr(opd_mod, "_w", fake_w)
    monkeypatch.setattr(
        opd_mod,
        "_resolve_opd_knobs",
        lambda: {
            "teacher_model": "accounts/fireworks/models/glm-5p2",
            "teacher_base_url": "http://teacher.invalid",
            "steps": 1,
            "learning_rate": 1e-4,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_completion": 8,
            "prompts_per_step": 1,
            "group_size": 1,
            "kl_coef": 1.0,
            "save_every": 0,
            "max_length": 0,
            "stop_sequences": (),
        },
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


def test_student_model_accepts_vl_warmstart_without_raising(monkeypatch):
    """opd VL warm-start (Qwen3.5/3.6) must build a fresh LoRA on the SFT-merged base — parity with
    GRPO — instead of raising. _init_adapter_model returns (merged_dir, fresh_lora) and records the
    VL merge; _student_model must LOAD the merged dir and wrap it in a PeftModel (run_opd then
    recombines SFT⊕opd at publish)."""
    pytest.importorskip("torch")  # AutoModelForCausalLM (patched below) needs torch present
    pytest.importorskip("transformers")
    import sys
    import types

    from flash.engine.worker import opd as opd_mod

    sentinel = object()
    fake_lora = object()
    seen = {}

    class _Base:
        def to(self, device):
            seen["device"] = device
            return self

    def _fake_init_adapter_model(model_id):
        opd_mod._w._VL_WARMSTART_SFT_DIR = "/tmp/fake_sft_dir"  # the VL merge path records this
        return "/tmp/merged_vl_dir", fake_lora  # merged base dir + FRESH LoRA config

    monkeypatch.setattr(opd_mod._w, "_VL_WARMSTART_SFT_DIR", None, raising=False)
    monkeypatch.setattr(opd_mod._w, "_init_adapter_model", _fake_init_adapter_model, raising=False)

    import transformers

    def _fake_from_pretrained(path, **kw):
        seen["path"] = path
        return _Base()

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(_fake_from_pretrained)
    )
    # _student_model does `from peft import get_peft_model`; peft is a worker-only dep, so inject a
    # stub module (works with or without a real peft install — hermetic local + CI).
    fake_peft = types.ModuleType("peft")
    fake_peft.get_peft_model = lambda base, cfg: (sentinel, base, cfg)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    out = opd_mod._student_model("Qwen/Qwen3.5-4B", {"dtype": "bf16"}, "cpu")
    assert out[0] is sentinel  # did NOT raise; wrapped the merged base in a PeftModel
    assert out[2] is fake_lora  # used the fresh LoRA from _init_adapter_model
    assert seen["path"] == "/tmp/merged_vl_dir"  # loaded the MERGED dir, not the raw base id


def test_publish_opd_deployable_recombines_for_vl_and_cleans_up(tmp_path, monkeypatch):
    """VL warm-start: the trained adapter is SFT-less, so publish must deploy the RECOMBINED SFT⊕opd
    adapter (as the served default AND the step checkpoint) and clean up the temp recombine dir."""
    from flash.engine.worker import opd as opd_mod

    calls = {"upload": [], "publish": []}
    recomb = tmp_path / "recomb"
    recomb.mkdir()
    monkeypatch.setattr(
        opd_mod._w, "recombined_warmstart_adapter_dir", lambda d: str(recomb), raising=False
    )
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

    opd_mod._publish_opd_deployable(str(tmp_path / "adapter"), 42, as_default=True)
    assert calls["upload"] == [(str(recomb), "adapter")]  # served default = RECOMBINED, not raw
    assert calls["publish"] == [(str(recomb), 42)]  # step checkpoint = RECOMBINED
    assert not recomb.exists()  # temp recombine dir cleaned up


def test_publish_opd_deployable_noop_recombine_deploys_adapter_dir(tmp_path, monkeypatch):
    """Non-VL / fresh runs: recombine is a no-op (returns None); the trained adapter_dir already
    carries the SFT, so it is deployed as-is. as_default=False publishes only the step checkpoint."""
    from flash.engine.worker import opd as opd_mod

    calls = {"upload": [], "publish": []}
    monkeypatch.setattr(
        opd_mod._w, "recombined_warmstart_adapter_dir", lambda d: None, raising=False
    )
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
    assert calls["publish"] == [(adir, 7)]  # no-op recombine -> deploys adapter_dir as-is


def test_publish_opd_deployable_best_effort_survives_recombine_failure(tmp_path, monkeypatch):
    """Per-step publish is best-effort: a recombine failure (e.g. an evicted SFT dir) is swallowed so
    training continues; the strict finalize path re-raises (it must never ship an SFT-less default)."""
    from flash.engine.worker import opd as opd_mod

    def _boom(d):
        raise RuntimeError("SFT dir evicted")

    monkeypatch.setattr(opd_mod._w, "recombined_warmstart_adapter_dir", _boom, raising=False)
    monkeypatch.setattr(
        opd_mod._w, "publish_deployable_checkpoint", lambda d, s: None, raising=False
    )
    monkeypatch.setattr(
        opd_mod._w, "hf_upload_folder", lambda d, sub, required=False: None, raising=False
    )

    # best_effort=True (per-step): swallowed, training continues (no raise).
    opd_mod._publish_opd_deployable(str(tmp_path / "a"), 20, as_default=False, best_effort=True)
    # best_effort=False (finalize): fatal — must not ship an SFT-less served default.
    with pytest.raises(RuntimeError, match="SFT dir evicted"):
        opd_mod._publish_opd_deployable(str(tmp_path / "a"), 100, as_default=True)


def test_opd_vram_reserves_dense_logits_unlike_fused_sft():
    """opd's gkd loss materializes dense logits (no fused CE), so its VRAM estimate must reserve the
    logits a >=3B SFT job fuses away — else a long-completion opd run is sized for a card that OOMs."""
    from flash.engine.vram import estimate_vram_gb

    kw = {"seq_len": 9216, "max_tokens": 8192, "vocab": 248_320, "lora_rank": 16}
    sft = estimate_vram_gb(4.0, "sft", "bf16", **kw)  # >=3B fuses CE -> 0 logits budgeted
    opd = estimate_vram_gb(4.0, "opd", "bf16", **kw)  # dense logits reserved (fwd + bwd)
    assert opd > sft + 10  # dense logits for opd vs 0 for fused SFT


def test_opd_vram_budgets_dense_logit_backward_buffers():
    """Regression (codex[bot], vram.py): gkd_loss has no fused CE and, at the loss BACKWARD peak, holds
    the fp32 completion rows + their fp32 gradient AND the bf16 full-sequence logits + their bf16
    gradient. The estimate must budget the backward buffers too, not only the two forward ones — else a
    long-completion / large-vocab (248k) opd job under-budgets gkd_loss.backward and routes to a GPU
    that OOMs. Isolate the logit term via a vocab delta (base + activations are vocab-independent): it
    must equal the FORWARD+BACKWARD size, i.e. 2x the forward-only (seq*2 + completion*4)*vocab."""
    from flash.engine.vram import estimate_vram_gb

    seq, comp = 9216, 8192
    kw = {"seq_len": seq, "max_tokens": comp, "lora_rank": 16}
    v1, v2 = 100_000, 248_320
    delta = estimate_vram_gb(4.0, "opd", "bf16", vocab=v2, **kw) - estimate_vram_gb(
        4.0, "opd", "bf16", vocab=v1, **kw
    )
    forward_only = (seq * 2 + comp * 4) * (v2 - v1) / 1e9  # what the old fwd-only formula would grow by
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


def test_opd_vram_is_single_sequence_not_batch_scaled():
    """Regression (codex[bot], vram.py): run_opd backprops ONE completion at a time (_train_one), so
    opd's VRAM estimate must NOT scale its activations with batch_size (the SFT per-device micro-batch
    term over-budgeted opd and bumped the GPU tier). At a short seq_len where SFT packs a batch, opd's
    estimate stays flat across batch_size while SFT's grows."""
    from flash.engine.vram import estimate_vram_gb

    kw = {"seq_len": 1024, "vocab": 248_320, "lora_rank": 16}
    opd_bs1 = estimate_vram_gb(4.0, "opd", "bf16", batch_size=1, **kw)
    opd_bs16 = estimate_vram_gb(4.0, "opd", "bf16", batch_size=16, **kw)
    assert opd_bs1 == opd_bs16  # single-sequence: batch_size does not change opd VRAM
    # contrast: SFT at the same short seq DOES scale with the micro-batch, so the invariant is meaningful.
    sft_bs1 = estimate_vram_gb(4.0, "sft", "bf16", batch_size=1, **kw)
    sft_bs16 = estimate_vram_gb(4.0, "sft", "bf16", batch_size=16, **kw)
    assert sft_bs16 > sft_bs1


def test_opd_teacher_rate_matches_fireworks_glm5p2_input_price():
    """glm-5p2 (and the omitted-teacher default) price at Fireworks' $1.40/M input, not the old $0.90
    — opd echo-scoring bills input tokens from the submit-time quote."""
    from flash.cost.facts import teacher_price_per_1m

    assert teacher_price_per_1m("accounts/fireworks/models/glm-5p2")[0] == 1.40
    assert teacher_price_per_1m("")[0] == 1.40  # omitted teacher -> representative default rate


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

    def fake_urlopen(req, timeout=None):
        if capture is not None:
            capture["url"] = req.full_url
            capture["body"] = json.loads(req.data.decode())
        return _FakeResp(payload)

    monkeypatch.setattr(tm.urllib.request, "urlopen", fake_urlopen)


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
    (non-TeacherError) exception — which _train_one's broad `except Exception` treats as a TRANSIENT
    skip, so a teacher that consistently returns malformed arrays burns every OPD step before the run
    fails with "no trained step". A length mismatch is a broken contract -> PERMANENT (abort now)."""
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
        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, f"HTTP {code}", {}, None)

        monkeypatch.setattr(tm.urllib.request, "urlopen", fake_urlopen)

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


def test_teacher_score_rejects_non_list_logprob_fields_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py:130): the length check assumes tokens/token_logprobs/
    text_offset are sequences. A malformed 200 with token_logprobs=null (or a scalar text_offset) makes
    len()/indexing raise TypeError OUTSIDE TeacherError -> _train_one swallows it as a generic
    (transient) skip without setting last_teacher_status, so a consistently malformed teacher burns
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
    clauses, and _train_one swallows it as an unclassified skip (last_teacher_status stays None), so a
    run hammered by malformed 200s fails as permanent no-signal instead of retrying as teacher infra."""
    import flash.engine.worker.teacher as tm
    from flash.engine.worker.teacher import TeacherError

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"<html>502 Bad Gateway</html>"  # HTTP 200 status, non-JSON body

    monkeypatch.setattr(tm.urllib.request, "urlopen", lambda req, timeout=None: _Resp())
    monkeypatch.setattr(tm.time, "sleep", lambda *a, **k: None)  # skip real backoff sleeps
    client = TeacherClient("k", "https://api.example/v1", "glm", max_retries=2)
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is False  # transient -> retried as infra, not permanent no-signal
    assert "unparseable" in str(ei.value).lower()


def test_teacher_incomplete_read_body_is_transient_teacher_error(monkeypatch):
    """Regression (codex[bot], teacher.py:65): an HTTP 200 whose body is truncated mid-read() raises
    http.client.IncompleteRead — an HTTPException, NOT an OSError — so without an explicit clause it
    escapes _post's retry loop and _train_one swallows it as an unclassified skip (last_teacher_status
    stays None), failing a truncated-200 run as permanent no-signal instead of retrying as infra."""
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

    monkeypatch.setattr(tm.urllib.request, "urlopen", lambda req, timeout=None: _Resp())
    monkeypatch.setattr(tm.time, "sleep", lambda *a, **k: None)  # skip real backoff sleeps
    client = TeacherClient("k", "https://api.example/v1", "glm", max_retries=2)
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is False  # transient -> retried as infra, not permanent no-signal
    assert "truncated" in str(ei.value).lower()


def test_train_one_refreshes_stall_clock_between_generation_and_scoring():
    """Regression (codex[bot], opd.py:491): _train_one must call on_generated() AFTER generation and
    BEFORE teacher scoring — both block for a long time and the caller's per-sample ping only fires
    after scoring returns, so a slow generation + teacher outage could otherwise span the poller's
    ~1200s stall window with no heartbeat. Assert the callback fires exactly once, before scoring."""
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    order: list[str] = []
    completion_ids = [2, 3]

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join({2: "h", 3: "i"}[int(x)] for x in ids)

    class _Teacher:
        def score(self, prompt, completion):
            order.append("score")
            return [TeacherToken(text="hi", logprob=-1.0, start=0, end=2)]

    class _GenLM(_TinyLM):
        def __init__(self):
            super().__init__(torch, T=1 + len(completion_ids), V=8)
            self.config = SimpleNamespace(use_cache=True)
            self._c = torch.tensor([completion_ids])

        def eval(self):
            return self

        def train(self):
            return self

        def generate(self, prompt_tensor, **cfg):
            return torch.cat([prompt_tensor, self._c], dim=1)

    opd_mod._train_one(
        model=_GenLM(),
        tok=_Tok(),
        teacher=_Teacher(),
        device="cpu",
        prompt_ids=[1],
        prompt_tensor=torch.tensor([[1]]),
        prompt_messages=[{"role": "user", "content": "hi"}],
        gen_cfg={},
        knobs={"kl_coef": 1.0, "stop_sequences": ()},
        torch=torch,
        on_generated=lambda: order.append("heartbeat"),
    )
    assert order == ["heartbeat", "score"], f"heartbeat must precede teacher scoring; got {order}"


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
    ValueError OUTSIDE TeacherError -> _train_one swallows it as an unclassified skip and burns every OPD
    step) or a non-finite NaN/inf (feeds a poisoned gradient straight into the gkd loss). Both must be
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
                    "text_offset": [0, 1, 3],  # token 1 'h' ends at 2 but next offset is 3 -> gap at 2
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
                    "tokens": ["B", "cd"],  # tiles full[1:4]=='Bcd' cleanly, but drops the 'A' prefix
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
    reject it as PERMANENT instead of returning an empty list that _train_one marks "ok" and then burns
    every OPD step on no signal before the generic no-trained-step failure."""
    from flash.engine.worker.teacher import TeacherError

    payload = {"choices": [{"logprobs": {"tokens": [], "token_logprobs": [], "text_offset": []}}]}
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is True
    assert "no completion-region tokens" in str(ei.value).lower()


def test_teacher_http_error_with_unreadable_body_still_classified_by_code(monkeypatch):
    """Regression (codex[bot], teacher.py:62): a retryable 5xx whose error body is truncated makes
    e.read() raise IncompleteRead BEFORE last_err is set — without a guard it escapes _post as a generic
    exception that _train_one skips without classifying, so repeated retryable errors end as permanent
    no-signal. The preview read must be guarded and the error still classified by e.code."""
    import http.client
    import urllib.error

    import flash.engine.worker.teacher as tm
    from flash.engine.worker.teacher import TeacherError

    class _BadBodyHTTPError(urllib.error.HTTPError):
        def read(self, *a, **k):
            raise http.client.IncompleteRead(b"", 10)

    err = _BadBodyHTTPError("http://x", 503, "Service Unavailable", {}, None)
    err.fp = object()  # force the `if e.fp` branch so the guarded read() is attempted

    def raise_503(req, timeout=None):
        raise err

    monkeypatch.setattr(tm.urllib.request, "urlopen", raise_503)
    monkeypatch.setattr(tm.time, "sleep", lambda *a, **k: None)  # skip real backoff
    client = TeacherClient("k", "https://api.example/v1", "glm", max_retries=2)
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is False  # 503 retryable -> transient TeacherError, not a raw exception
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
        SimpleNamespace(JOB_SPEC=SimpleNamespace(train=_Train(kl_penalty_coef=0.0)), THINKING=False),
    )
    with pytest.raises(RuntimeError, match="kl_penalty_coef must be > 0"):
        opd_mod._resolve_opd_knobs()

    # unset (None) -> positive recipe default, no raise.
    monkeypatch.setattr(
        opd_mod,
        "_w",
        SimpleNamespace(JOB_SPEC=SimpleNamespace(train=_Train(kl_penalty_coef=None)), THINKING=False),
    )
    assert opd_mod._resolve_opd_knobs()["kl_coef"] > 0.0


def test_gkd_loss_skips_empty_student_group_without_crashing():
    # A group with an empty student-index list (a teacher-only span) must be skipped, not divide by
    # zero in the per-span coefficient (len(s_idx) == 0).
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import gkd_loss

    model = _TinyLM(torch, T=2, V=8)
    loss = gkd_loss(model, [1], [2], [([], -1.0), ([0], -2.0)], device="cpu", kl_coef=1.0)
    assert loss is not None  # the empty group is ignored; the real group still trains
    loss.backward()
    assert model.w.grad[0].abs().sum() > 0


def test_groupwise_alignment_emits_no_empty_student_group():
    # Teacher covers [0,2) but the student's first token starts at char 2 (teacher-only leading
    # span). No group may have an empty student-index list.
    student = _student([(2, 3), (3, 5)])
    teacher = _teacher([(0, 2), (2, 5)])
    groups = groupwise_alignment(student, teacher)
    assert all(s_idx for s_idx, _ in groups)  # every group has >= 1 student token
    assert [s_idx for s_idx, _ in groups] == [[0, 1]]


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
                "steps": 25,
                "teacher_model": "accounts/fireworks/models/glm-5p1",
                "hf_repo": "owner/runs",
            },
        },
        run_id="x",
    )
    restored = JobSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.phase == "opd"
    assert restored.train.teacher_model == "accounts/fireworks/models/glm-5p1"
    # FIREWORKS_API_KEY is auto-declared a required secret for opd.
    assert "FIREWORKS_API_KEY" in restored.environment.secrets


def test_opd_cost_is_step_priced_and_bills_teacher_tokens():
    from flash.cost.spec import estimate_for_spec, spec_steps
    from flash.schema import spec_from_dict

    # No [train].max_examples set — opd must NOT fall into the SFT example-count path (which
    # would raise); it is step-driven like GRPO.
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "opd",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"steps": 30, "hf_repo": "owner/runs"},
        },
        run_id="x",
    )
    assert spec_steps(spec) == 30
    est = estimate_for_spec(spec)
    assert est.method == "opd"
    assert est.teacher_api_usd > 0.0  # external teacher token spend is itemized
    assert est.total_usd >= est.teacher_api_usd
    assert "opd step" in " ".join(est.notes)


# --------------------------------------------------------------------------------------------------
# loss math (needs torch)
# --------------------------------------------------------------------------------------------------
class _TinyLM:
    """Minimal stand-in for a causal LM: per-position learnable logits, ignores the input ids."""

    def __init__(self, torch, T, V):
        self.w = torch.zeros(T, V, requires_grad=True)

    def __call__(self, input_ids):
        T = input_ids.shape[1]
        return SimpleNamespace(logits=self.w[:T].unsqueeze(0))

    def parameters(self):
        return [self.w]


def test_gkd_loss_backpropagates_over_grouped_spans():
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import gkd_loss

    V = 8
    prompt_ids = [1]
    student_ids = [2, 3]  # 2 completion tokens
    model = _TinyLM(torch, T=len(prompt_ids) + len(student_ids), V=V)
    # One group covering both completion tokens (as when the teacher tokenizes them as one span).
    groups = [([0, 1], -1.5)]
    loss = gkd_loss(model, prompt_ids, student_ids, groups, device="cpu", kl_coef=1.0)
    assert loss is not None
    assert loss.requires_grad
    loss.backward()
    # logits at index P+j-1 predict completion token j: index 0 -> tok 0, index 1 -> tok 1.
    assert model.w.grad[0].abs().sum() > 0
    assert model.w.grad[1].abs().sum() > 0


def test_gkd_loss_none_without_groups_or_tokens():
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import gkd_loss

    model = _TinyLM(torch, T=3, V=4)
    assert gkd_loss(model, [1], [2, 3], [], device="cpu") is None
    assert gkd_loss(model, [1], [], [([0], -1.0)], device="cpu") is None


def test_gkd_loss_coefficient_tracks_student_minus_teacher_logprob():
    # The per-span coefficient is (student_logsum.detach() - teacher_logsum)/|span|; a more
    # confident teacher (lower/more-negative teacher_logsum) makes the coefficient larger.
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import gkd_loss

    V = 8
    prompt_ids = [1]
    student_ids = [2]
    model = _TinyLM(torch, T=2, V=V)  # uniform logits -> student logprob = -log V per token
    hi = gkd_loss(model, prompt_ids, student_ids, [([0], -5.0)], device="cpu", kl_coef=1.0)
    lo = gkd_loss(model, prompt_ids, student_ids, [([0], -0.5)], device="cpu", kl_coef=1.0)
    # loss = coeff * student_logprob, student_logprob < 0, and coeff = (s_det - teacher)/1.
    # teacher=-5.0 -> larger coeff -> more-negative loss than teacher=-0.5.
    assert float(hi.detach()) < float(lo.detach())

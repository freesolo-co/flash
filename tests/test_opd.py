"""CPU coverage for active OPD helpers, teacher transport, schema, and cost plumbing."""

from __future__ import annotations

import io
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
    return [TeacherToken(text='', logprob=-(i + 1.0), start=a, end=b) for i, (a, b) in enumerate(spans)]

def test_drop_fully_forced_groups_removes_all_forced_spans():
    from flash.engine.worker.opd import _drop_fully_forced_groups
    groups = [([0], -1.0), ([1, 2], -2.0), ([3], -3.0)]
    assert _drop_fully_forced_groups(groups, (True, False, False, True)) == [([1, 2], -2.0)]

def test_drop_fully_forced_groups_is_a_noop_without_a_mask():
    from flash.engine.worker.opd import _drop_fully_forced_groups
    groups = [([0], -1.0), ([1], -2.0)]
    assert _drop_fully_forced_groups(groups, ()) == groups

def test_drop_fully_forced_groups_keeps_a_partially_forced_span():
    from flash.engine.worker.opd import _drop_fully_forced_groups
    assert _drop_fully_forced_groups([([0, 1], -1.0)], (True, False)) == [([0, 1], -1.0)]

def test_masking_then_prepare_normalizes_over_surviving_tokens_only():
    """After forced-group masking, the prepared loss inputs contain ONLY surviving-group tokens, so
    the downstream per-token mean normalizes over the kept (content) tokens -- dropping a fully-forced
    span re-normalizes the reverse-KL rather than leaving a shrunken sum over the original count."""
    from flash.engine.worker.opd import _drop_fully_forced_groups, _prepare_gkd_groups
    groups = [([0], -1.0), ([1, 2], -2.0), ([3], -3.0)]
    kept = _drop_fully_forced_groups(groups, (True, False, False, True))
    prepared = _prepare_gkd_groups(kept)
    assert prepared.token_indices == (1, 2)
    assert prepared.group_lengths == (2,)
    assert prepared.teacher_logsums == (-2.0,)

def test_masked_loss_equals_loss_without_the_forced_groups():
    """End-to-end normalization: the masked reverse-KL equals the loss computed as if the forced
    groups never existed -- the per-token mean re-normalizes over survivors, it is neither diluted by
    nor retains the dropped forced positions."""
    torch = pytest.importorskip('torch')
    from flash.engine.worker.opd import _drop_fully_forced_groups, _gkd_loss_from_logps
    sp = torch.tensor([-0.5, -1.0, -1.5, -2.0], requires_grad=True)
    with_forced = [([0], -1.0), ([1, 2], -2.0), ([3], -3.0)]
    kept = _drop_fully_forced_groups(with_forced, (True, False, False, True))
    loss_masked = _gkd_loss_from_logps(sp, kept, kl_coef=0.25)
    loss_reference = _gkd_loss_from_logps(sp, [([1, 2], -2.0)], kl_coef=0.25)
    assert torch.allclose(loss_masked, loss_reference)

def test_gkd_groups_are_one_per_shared_boundary_when_tokenizers_agree():
    student = _student([(0, 2), (2, 5)])
    teacher = _teacher([(0, 2), (2, 5)])
    groups = groupwise_alignment(student, teacher)
    assert [s_idx for s_idx, _ in groups] == [[0], [1]]
    assert groups[0][1] == -1.0
    assert groups[1][1] == -2.0
    assert groupwise_coverage(groups, student) == 1.0

def test_gkd_span_grows_across_disagreement_and_covers_every_token():
    student = _student([(0, 3), (3, 6)])
    teacher = _teacher([(0, 6)])
    groups = groupwise_alignment(student, teacher)
    assert len(groups) == 1
    assert groups[0][0] == [0, 1]
    assert groups[0][1] == -1.0
    assert groupwise_coverage(groups, student) == 1.0

def test_gkd_partial_agreement_splits_at_shared_boundaries_only():
    student = _student([(0, 2), (2, 4), (4, 6)])
    teacher = _teacher([(0, 2), (2, 6)])
    groups = groupwise_alignment(student, teacher)
    assert [s_idx for s_idx, _ in groups] == [[0], [1, 2]]
    assert groupwise_coverage(groups, student) == 1.0

def test_gkd_merges_leading_student_only_span_so_no_token_is_dropped():
    student = _student([(0, 1), (1, 2), (2, 5)])
    teacher = _teacher([(2, 5)])
    groups = groupwise_alignment(student, teacher)
    assert len(groups) == 1
    assert groups[0][0] == [0, 1, 2]
    assert groupwise_coverage(groups, student) == 1.0

def test_gkd_empty_inputs_yield_no_groups():
    assert groupwise_alignment([], _teacher([(0, 1)])) == []
    assert groupwise_alignment(_student([(0, 1)]), []) == []
    assert groupwise_coverage([], []) == 0.0

def test_gkd_coverage_never_exceeds_100pct_with_in_span_zero_width_token():
    student = _student([(0, 2), (2, 2), (2, 5)])
    teacher = _teacher([(0, 5)])
    groups = groupwise_alignment(student, teacher)
    assert groups[0][0] == [0, 1, 2]
    assert groupwise_coverage(groups, student) == 1.0

def test_student_tokens_use_sampled_ids_with_offsets_into_completion_text():
    from flash.engine.worker.opd import student_tokens_with_offsets

    class _Tok:

        def decode(self, ids, skip_special_tokens=True):
            m = {1: 'h', 2: 'i', 3: ''}
            return ''.join(m[i] for i in ids)
    ids, toks = student_tokens_with_offsets(_Tok(), [1, 2, 3], 'hi')
    assert ids == [1, 2, 3]
    assert (toks[0].start, toks[0].end) == (0, 1)
    assert (toks[1].start, toks[1].end) == (1, 2)
    assert (toks[2].start, toks[2].end) == (2, 2)

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
            out = ''
            k = 0
            while k < len(ids):
                if ids[k] == 7:
                    out += 'x'
                    k += 1
                elif ids[k] == 10 and k + 1 < len(ids) and (ids[k + 1] == 11):
                    out += '😀'
                    k += 2
                elif ids[k] == 10:
                    out += '�'
                    k += 1
                else:
                    k += 1
            return out
    ids, toks = student_tokens_with_offsets(_Tok(), [7, 10, 11], 'x😀')
    assert ids == [7, 10, 11]
    assert (toks[0].start, toks[0].end) == (0, 1)
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
            m = {20: '�', 21: 'y'}
            return ''.join(m[int(x)] for x in ids)
    ids, toks = student_tokens_with_offsets(_Tok(), [20, 21], '�y')
    assert ids == [20, 21]
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
            return ''.join('abcdefghij'[int(x) % 10] for x in ids)
    tok = _Tok()
    n = 200
    ids = list(range(n))
    text = ''.join('abcdefghij'[i % 10] for i in ids)
    out_ids, toks = student_tokens_with_offsets(tok, ids, text)
    assert out_ids == ids
    assert len(toks) == n
    assert tok.max_ids <= 2, f'decode saw up to {tok.max_ids} ids -> quadratic prefix decoding'
    assert (toks[0].start, toks[0].end) == (0, 1)
    assert (toks[-1].start, toks[-1].end) == (n - 1, n)

def test_trim_trailing_stop_drops_delimiter_from_ids_and_text():
    from flash.engine.worker.opd import _trim_trailing_stop

    class _Tok:

        def decode(self, ids, skip_special_tokens=True):
            m = {1: 'A', 2: 'n', 3: 's', 4: '</', 5: 'answer>'}
            return ''.join(m[int(i)] for i in ids)
    ids, text = _trim_trailing_stop(_Tok(), [1, 2, 3, 4, 5], 'Ans</answer>', ['</answer>'])
    assert text == 'Ans'
    assert ids == [1, 2, 3]
    assert _trim_trailing_stop(_Tok(), [1, 2, 3], 'Ans', ['</answer>']) == ([1, 2, 3], 'Ans')

def test_trim_trailing_stop_keeps_ids_and_text_synced_when_stop_starts_inside_token():
    """Regression (codex[bot], opd.py): when the stop delimiter starts INSIDE the final sampled token
    (that token decodes to "B</answer>"), the whole token is dropped from the kept ids — so returning
    completion_text[:keep_len] would keep a "B" the ids can no longer represent, desyncing the
    teacher-scored text from the student ids. The returned text must equal decode(kept ids)."""
    from flash.engine.worker.opd import _trim_trailing_stop

    class _Tok:

        def decode(self, ids, skip_special_tokens=True):
            m = {1: 'A', 4: 'B</answer>'}
            return ''.join(m[int(i)] for i in ids)
    ids, text = _trim_trailing_stop(_Tok(), [1, 4], 'AB</answer>', ['</answer>'])
    assert ids == [1]
    assert text == 'A'
    assert text == _Tok().decode(ids)

def test_trim_trailing_stop_prefers_longest_overlapping_stop():
    """Regression (codex[bot], opd.py:150): with overlapping delimiters like ["\\n", "\\n\\n"] listed
    shortest-first, a "\\n\\n" tail must have BOTH newlines trimmed (the longest/earliest matching stop),
    not just the first-listed "\\n" — otherwise the teacher still scores a leftover delimiter newline."""
    from flash.engine.worker.opd import _trim_trailing_stop

    class _Tok:

        def decode(self, ids, skip_special_tokens=True):
            m = {1: 'h', 2: 'i', 3: '\n', 4: '\n'}
            return ''.join(m[int(i)] for i in ids)
    ids, text = _trim_trailing_stop(_Tok(), [1, 2, 3, 4], 'hi\n\n', ['\n', '\n\n'])
    assert text == 'hi'
    assert ids == [1, 2]
    assert _trim_trailing_stop(_Tok(), [1, 2, 3, 4], 'hi\n\n', ['\n\n', '\n']) == ([1, 2], 'hi')

def test_stop_detection_and_trim_handle_special_token_delimiter():
    """Regression (codex[bot], opd.py): a [train] stop_sequence can be a tokenizer SPECIAL token (e.g.
    <|im_end|>). A skip_special_tokens=True decode STRIPS it, so the clean text no longer ends with the
    delimiter — _rollout_terminated would misclassify the rollout as truncated and _trim_trailing_stop
    would never remove it, skipping every usable sample for that config. Detection/trim must run on the
    special-tokens-INCLUDED decode."""
    from flash.engine.worker.opd import _rollout_terminated, _trim_trailing_stop
    IM_END = 9

    class _Tok:

        def decode(self, ids, skip_special_tokens=True):
            answer = ''.join({1: '4', 2: '2'}.get(int(i), '') for i in ids)
            if not skip_special_tokens and any(int(i) == IM_END for i in ids):
                return answer + '<|im_end|>'
            return answer
    ids = [1, 2, IM_END]
    stop_text = _Tok().decode(ids, skip_special_tokens=False)
    stops = ['<|im_end|>']
    assert _Tok().decode(ids, skip_special_tokens=True).endswith('<|im_end|>') is False
    assert _rollout_terminated(ids, stop_text, frozenset(), stops) is True
    kept_ids, text = _trim_trailing_stop(_Tok(), ids, stop_text, stops)
    assert kept_ids == [1, 2]
    assert text == '42'

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
            return ''.join('abcdefghij'[int(x) % 10] for x in ids)
    n = 500
    ids = list(range(n))
    text = ''.join('abcdefghij'[i % 10] for i in ids)
    tok = _Tok()
    stop = text[-3:]
    out_ids, out_text = _trim_trailing_stop(tok, ids, text, [stop])
    assert out_ids == ids[:n - 3]
    assert out_text == text[:n - 3]
    assert tok.calls <= 10, f'decode called {tok.calls}x on a {n}-token completion -> quadratic trim'

def test_rollout_terminated_requires_eos_or_stop_not_length():
    """A rollout is safe to distil only if it terminated NATURALLY — EOS in the ids, or (with
    stop_sequences) the decoded text ends with a stop delimiter. A max_new_tokens cap hit OR a
    gen_cfg.max_time cut ends without either and is a partial mid-output fragment OPD must skip (it
    can't supervise the stop token). Length is NOT the criterion (codex[bot])."""
    from flash.engine.worker.opd import _rollout_terminated
    EOS = frozenset({99})
    assert _rollout_terminated([1, 2, 3, 99], 'abc', EOS, ()) is True
    assert _rollout_terminated([1, 2, 3, 4], 'abcd', EOS, ()) is False
    assert _rollout_terminated([1, 2], 'ab', EOS, ()) is False
    assert _rollout_terminated([1, 2, 88], 'abc', frozenset({99, 88}), ()) is True
    assert _rollout_terminated([1, 2, 3, 4], 'ans</answer>', frozenset(), ('</answer>',)) is True
    assert _rollout_terminated([1, 2, 3, 4], 'ans', frozenset(), ('</answer>',)) is False
    assert _rollout_terminated([1, 2, 3, 4], 'abcd', frozenset(), ()) is True

def test_generation_eos_ids_unions_tokenizer_and_generation_config_lists():
    """_rollout_terminated must see EVERY halting id. _generation_eos_ids unions the tokenizer's
    eos_token_id with the model's generation_config/config eos_token_id, each of which HF allows to be
    a scalar OR a list — so a model like MiniCPM5 that halts on a secondary <|im_end|> (a list member)
    while its primary eos is </s> gets both ids, and a secondary-eos rollout is not misread as truncated
    (codex[bot]). bool is an int subclass but never a token id, so it's excluded."""
    from flash.engine.worker.opd import _generation_eos_ids
    tok = SimpleNamespace(eos_token_id=2)
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2, 73]), config=SimpleNamespace(eos_token_id=151645))
    assert _generation_eos_ids(model, tok) == frozenset({2, 73, 151645})
    assert _generation_eos_ids(SimpleNamespace(), SimpleNamespace(eos_token_id=5)) == frozenset({5})
    assert _generation_eos_ids(SimpleNamespace(), SimpleNamespace()) == frozenset()
    assert _generation_eos_ids(SimpleNamespace(), SimpleNamespace(eos_token_id=True)) == frozenset()

def test_opd_vram_sizing_uses_completion_budget_not_sft_default():
    from flash.engine.vram import opd_rollout_seq_len
    assert opd_rollout_seq_len(0, None, False) == 1536
    assert opd_rollout_seq_len(0, 8192, False) == 9216
    assert opd_rollout_seq_len(4096, 8192, False) == 4096

def test_opd_selects_managed_teacher_and_rejects_unknown():
    """[train].teacher_model selects the managed teacher from a fixed allow-list: a supported alias
    (or the raw Fireworks id, or a spaced/mixed-case form) parses and is stored as its canonical
    Fireworks model id; an unsupported teacher is rejected at PARSE time (before a paid GPU)."""
    from flash.schema import ConfigError, spec_from_dict

    def _spec(teacher):
        return spec_from_dict({'model': 'Qwen/Qwen3.5-4B', 'algorithm': 'opd', 'environment': {'id': 'github:owner/repo@main:env/environment.py'}, 'train': {'epochs': 1, 'max_examples': 5, 'teacher_model': teacher}}, run_id='x')
    assert _spec('kimi-k2.6').train.teacher_model == 'accounts/fireworks/models/kimi-k2p6'
    assert _spec('deepseek-v4-pro').train.teacher_model == 'accounts/fireworks/models/deepseek-v4-pro'
    assert _spec('GLM 5.2').train.teacher_model == 'accounts/fireworks/models/glm-5p2'
    assert _spec('accounts/fireworks/models/glm-5p2').train.teacher_model == 'accounts/fireworks/models/glm-5p2'
    assert _spec('  accounts/fireworks/models/glm-5p2  ').train.teacher_model == 'accounts/fireworks/models/glm-5p2'
    assert _spec('').train.teacher_model == ''
    with pytest.raises(ConfigError, match='teacher_model'):
        _spec('gpt-5.5')
    with pytest.raises(ConfigError, match='teacher_model'):
        _spec('qwen-3.7-max')
    with pytest.raises(ConfigError, match='teacher_model'):
        _spec('minimax-m3')

def test_opd_rejects_prompt_budget_at_parse_time_before_provisioning():
    """max_context_tokens <= max_completion_tokens leaves no prompt budget; opd must reject it at spec-parse time
    (before a paid worker is provisioned), not only inside run_opd after GPU setup."""
    from flash.schema import ConfigError, spec_from_dict

    def _spec(train_extra):
        return spec_from_dict({'model': 'Qwen/Qwen3.5-4B', 'algorithm': 'opd', 'environment': {'id': 'github:owner/repo@main:env/environment.py'}, 'train': {'epochs': 1, 'max_examples': 5, 'hf_repo': 'owner/runs', **train_extra}}, run_id='x')
    _spec({'max_context_tokens': 2048, 'max_completion_tokens': 512})
    with pytest.raises(ConfigError, match='prompt budget'):
        _spec({'max_context_tokens': 400, 'max_completion_tokens': 512})
    with pytest.raises(ConfigError, match='prompt budget'):
        _spec({'max_context_tokens': 256})

def test_opd_rejects_zero_kl_penalty_at_parse_time():
    from flash.schema import ConfigError, spec_from_dict

    def _spec(algorithm, train_extra):
        return spec_from_dict({'model': 'Qwen/Qwen3.5-4B', 'algorithm': algorithm, 'environment': {'id': 'github:owner/repo@main:env/environment.py'}, 'train': {'epochs': 1, 'max_examples': 5, **train_extra}}, run_id='x')
    with pytest.raises(ConfigError, match='kl_penalty_coef must be > 0 for opd'):
        _spec('opd', {'kl_penalty_coef': 0})
    assert _spec('opd', {}).train.kl_penalty_coef is None
    assert _spec('grpo', {'kl_penalty_coef': 0}).train.kl_penalty_coef == 0

@pytest.mark.parametrize('max_context_tokens', [256, 512])
def test_opd_accepts_short_hybrid_mamba_context_with_conditional_worker_floor(max_context_tokens):
    from flash.schema import spec_from_dict
    spec = spec_from_dict({'model': 'Qwen/Qwen3.6-35B-A3B', 'algorithm': 'opd', 'environment': {'id': 'github:owner/repo@main:env/environment.py'}, 'train': {'epochs': 1, 'max_examples': 5, 'max_context_tokens': max_context_tokens, 'max_completion_tokens': 128}}, run_id='x')
    assert spec.train.max_context_tokens == max_context_tokens

def test_opd_teacher_prompt_includes_thinking_prefill():
    """Regression (codex[bot], opd.py:93): in thinking mode the student template opens a reasoning
    block (e.g. <think>) AFTER the generation prompt and samples its completion after it. The teacher
    must condition on that SAME trailing prefill; the plain 'Assistant: ' prompt (empty prefill) would
    score every thinking-mode logprob against a prefix that never opened the block."""
    from flash.engine.worker import opd as opd_mod
    msgs = [{'role': 'user', 'content': 'hi'}]
    assert opd_mod._teacher_prompt_text(msgs).endswith('Assistant: ')
    assert opd_mod._teacher_prompt_text(msgs, '<think>\n').endswith('Assistant: <think>\n')

def test_thinking_prefill_text_is_template_delta(monkeypatch):
    """Regression (codex[bot], opd.py): the thinking prefill is the DELTA a thinking-mode chat template
    opens after the generation prompt (enable_thinking True vs False). Empty when thinking is off (the
    plain teacher prompt already matches) or the template ignores enable_thinking."""
    from flash.engine.worker import opd as opd_mod

    class _Tok:

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            return '<|im_start|>assistant\n' + ('<think>\n' if enable_thinking else '')
    monkeypatch.setattr(opd_mod, '_w', SimpleNamespace(THINKING=False))
    assert opd_mod._thinking_prefill_text(_Tok()) == ''
    monkeypatch.setattr(opd_mod, '_w', SimpleNamespace(THINKING=True))
    assert opd_mod._thinking_prefill_text(_Tok()) == '<think>\n'

    class _NoThinkTok:

        def apply_chat_template(self, messages, **kw):
            return '<|im_start|>assistant\n'
    assert opd_mod._thinking_prefill_text(_NoThinkTok()) == ''

def test_thinking_prefill_derives_opener_from_hybrid_template(monkeypatch):
    """Regression (codex[bot], opd.py): _thinking_prefill_text must handle a HYBRID template where the
    thinking render is NOT a prefix-extension of the non-thinking render — the opener is inserted BEFORE
    shared trailing template text, so base is not a prefix of think. The old think.startswith(base) test
    returned "", dropping the opener the student pre-fills so the teacher scored reasoning tokens against
    the wrong prefix. The common prefix/suffix derivation must recover the opener from think's unique
    middle."""
    from flash.engine.worker import opd as opd_mod

    class _HybridTok:

        def apply_chat_template(self, messages, *, enable_thinking, **kw):
            return 'A:\n<think>\nEND' if enable_thinking else 'A:\nEND'
    monkeypatch.setattr(opd_mod, '_w', SimpleNamespace(THINKING=True))
    assert opd_mod._thinking_prefill_text(_HybridTok()) == '<think>\n'

def test_thinking_prefill_recovers_opener_from_closed_block_hybrid(monkeypatch):
    """Regression (codex[bot]/cursor, opd.py): a HYBRID template that disables thinking by force-CLOSING
    the block — enable_thinking=False -> '...<think></think>\\n', enable_thinking=True -> '...<think>\\n'
    — shares '<think>' in BOTH renders, so the common-prefix delta eats it and the previous fix returned
    only '\\n'. The student still pre-fills '<think>\\n', so the teacher must condition on the full
    opener; recover it from base's closing tag."""
    from flash.engine.worker import opd as opd_mod

    class _ClosedBlockTok:

        def apply_chat_template(self, messages, *, enable_thinking, **kw):
            return 'A:\n<think>\n' if enable_thinking else 'A:\n<think></think>\n'
    monkeypatch.setattr(opd_mod, '_w', SimpleNamespace(THINKING=True))
    assert opd_mod._thinking_prefill_text(_ClosedBlockTok()) == '<think>\n'

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
            return 'A:\n<think>\n' if enable_thinking else 'A:\n<think>\n\n</think>\n'
    monkeypatch.setattr(opd_mod, '_w', SimpleNamespace(THINKING=True))
    assert opd_mod._thinking_prefill_text(_WhitespaceEmptyBlockTok()) == '<think>\n'

def test_thinking_prefill_recovers_opener_when_closed_block_leaves_whitespace_remainder(monkeypatch):
    """Regression (codex[bot], opd.py:134): a closed-block hybrid whose disabled render closes IMMEDIATELY
    after the opener (enable_thinking=False -> '...<think></think>', True -> '...<think>\\n') shares only
    '<think>' in the common prefix, so think_mid is the NON-EMPTY whitespace remainder '\\n'. The old
    `if think_mid: return think_mid` early-return handed back '\\n' and skipped the closed-block recovery,
    conditioning the teacher on a prompt that opened but never continued <think>. The recovery must run
    FIRST and return the real thinking-side opener '<think>\\n'."""
    from flash.engine.worker import opd as opd_mod

    class _ClosedImmediatelyTok:

        def apply_chat_template(self, messages, *, enable_thinking, **kw):
            return 'A:\n<think>\n' if enable_thinking else 'A:\n<think></think>'
    monkeypatch.setattr(opd_mod, '_w', SimpleNamespace(THINKING=True))
    assert opd_mod._thinking_prefill_text(_ClosedImmediatelyTok()) == '<think>\n'

def test_student_tokens_absorb_dropped_leading_space_sentencepiece():
    """Regression (codex[bot], opd.py:175): a SentencePiece/LLaMA tokenizer decodes a mid-completion
    word token IN ISOLATION without its leading word-boundary space (decode([▁world]) == 'world', not
    ' world'). prev + len(decode(window)) would then undercount that span by one char and drift every
    following offset, misassigning teacher spans to the wrong sampled ids. Offsets must be anchored to
    completion_text so the dropped space is absorbed into the token's start and spans stay contiguous
    and exact."""
    from flash.engine.worker.opd import student_tokens_with_offsets

    class _SPTok:

        def decode(self, ids, skip_special_tokens=True):
            m = {10: 'hi', 11: 'world'}
            return ''.join(m[int(x)] for x in ids)
    completion_text = 'hi world'
    ids, toks = student_tokens_with_offsets(_SPTok(), [10, 11], completion_text)
    assert ids == [10, 11]
    assert (toks[0].start, toks[0].end) == (0, 2)
    assert (toks[1].start, toks[1].end) == (2, 8), f'dropped leading space must be absorbed into the span; got {(toks[1].start, toks[1].end)}'
    assert toks[-1].end == len(completion_text)

def test_groupwise_alignment_cursor_walk_groups_denser_student_span():
    """Regression (codex[bot], tokenizer_align.py:73): the cursor walk that replaced the per-boundary
    rescan (O(C^2) -> O(S+T+B)) must still produce the coarsest common refinement — carrying a span's
    extra student tokens into the teacher-bearing span that closes it. Here the student tokenizes
    [0,3)+[3,6) where the teacher has one [0,6) token, so both student indices group under that
    teacher logprob; the tail [6,9) aligns 1:1."""
    from flash.engine.worker.tokenizer_align import groupwise_alignment
    student = [StudentToken(token_id=0, start=0, end=3), StudentToken(token_id=1, start=3, end=6), StudentToken(token_id=2, start=6, end=9)]
    teacher = [TeacherToken(text='', logprob=-1.0, start=0, end=6), TeacherToken(text='', logprob=-2.0, start=6, end=9)]
    assert groupwise_alignment(student, teacher) == [([0, 1], -1.0), ([2], -2.0)]

def test_opd_vram_reserves_dense_logits_unlike_fused_sft():
    """opd's gkd loss materializes dense logits (no fused CE), so its VRAM estimate must reserve the
    logits a >=3B SFT job fuses away — else a long-completion opd run is sized for a card that OOMs."""
    from flash.engine.vram import estimate_vram_gb
    kw = {'seq_len': 9216, 'max_tokens': 8192, 'vocab': 248320, 'lora_rank': 16}
    sft = estimate_vram_gb(4.0, 'sft', 'bf16', **kw)
    opd = estimate_vram_gb(4.0, 'opd', 'bf16', **kw)
    assert opd > sft + 10

def test_opd_vram_reserves_colocated_vllm_rollout_copy():
    """OPD student generation uses a resident vLLM engine, so VRAM includes a second weight/KV copy."""
    from flash.engine.vram import estimate_vram_gb
    kw = {'seq_len': 1536, 'max_tokens': 512, 'vocab': 248320, 'lora_rank': 16}
    grpo_without_vllm = estimate_vram_gb(4.0, 'grpo', 'bf16', use_vllm=False, **kw)
    opd_with_vllm = estimate_vram_gb(4.0, 'opd', 'bf16', use_vllm=True, **kw)
    opd_flag_ignored = estimate_vram_gb(4.0, 'opd', 'bf16', use_vllm=False, **kw)
    assert opd_with_vllm > grpo_without_vllm + 8.0
    assert opd_flag_ignored == opd_with_vllm
    assert estimate_vram_gb(4.0, 'opd', 'bf16', **kw) == opd_with_vllm

def test_opd_vram_sizes_rollout_kv_for_full_prompt_batch():
    from flash.engine.vram import estimate_vram_gb, opd_rollout_concurrency
    assert opd_rollout_concurrency(8, 3) == 24
    kw = {'seq_len': 8192, 'max_tokens': 512, 'vocab': 128000, 'lora_rank': 16}
    one_prompt = estimate_vram_gb(4.0, 'opd', 'bf16', batch_size=1, group_size=1, **kw)
    eight_prompts = estimate_vram_gb(4.0, 'opd', 'bf16', batch_size=8, group_size=1, **kw)
    assert eight_prompts > one_prompt + 20.0

def test_model_required_vram_uses_opd_group_default_not_grpo_default():
    from flash.engine.vram import model_required_vram_gb
    train = {'max_length': 8192, 'max_tokens': 512, 'batch_size': 8, 'lora_rank': 16}
    default_group = model_required_vram_gb('Qwen/Qwen3.5-4B', 'opd', train=train, headroom=1.0)
    explicit_opd_default = model_required_vram_gb('Qwen/Qwen3.5-4B', 'opd', train={**train, 'group_size': 1}, headroom=1.0)
    grpo_default_group = model_required_vram_gb('Qwen/Qwen3.5-4B', 'opd', train={**train, 'group_size': 8}, headroom=1.0)
    assert default_group == explicit_opd_default
    assert grpo_default_group > default_group

def test_opd_35b_vllm_rollout_routes_above_h200_to_b200():
    """35B OPD with colocated student vLLM routes above the old H200-sized OPD estimate."""
    from flash.engine.vram import model_required_vram_gb
    need = model_required_vram_gb('Qwen/Qwen3.6-35B-A3B', 'opd', train={'max_length': 1536, 'max_tokens': 512, 'batch_size': 1, 'group_size': 8, 'lora_rank': 16})
    assert 141 < need <= 180

def test_opd_35b_full_context_group1_fits_b200():
    """a full-context group-1 35b opd run fits the b200 with conservative kv sizing."""
    from flash.catalog import MODELS, vocab_size_for
    from flash.engine.vram import estimate_vram_gb, model_required_vram_gb
    from flash.providers.allocator import vram_headroom
    from flash.providers.base import cheapest_gpu
    moe = 'Qwen/Qwen3.6-35B-A3B'
    info = MODELS[moe]
    train = {'max_context_tokens': 4096, 'max_completion_tokens': 2048, 'batch_size': 8, 'group_size': 1, 'lora_rank': 32}
    need = model_required_vram_gb(moe, 'opd', train=train, headroom=vram_headroom())
    assert need <= 180
    assert cheapest_gpu(need) == 'B200'
    kw = {'seq_len': 4096, 'max_tokens': 2048, 'batch_size': 8, 'group_size': 1, 'lora_rank': 32, 'vocab': vocab_size_for(moe), 'active_params_b': info.active_params_b}
    fp8 = estimate_vram_gb(info.params_b, 'opd', 'bf16', fp8_kv=True, **kw)
    bf16 = estimate_vram_gb(info.params_b, 'opd', 'bf16', fp8_kv=False, **kw)
    assert fp8 < bf16
    hr = vram_headroom()
    assert fp8 * hr <= 180 < bf16 * hr

def test_opd_fp8_kv_gate_does_not_downroute_below_the_fp8_ceiling():
    """The fp8-KV discount must apply only when a run can ONLY land on a modern (cc >= 8.9) card. A
    smaller OPD run that fits the 80 GB A100 (sm80, no fp8) must keep its bf16 KV sizing and its A100
    route — never dropping onto a card that would not actually use fp8 (and would then OOM)."""
    from flash.engine.vram import model_required_vram_gb
    from flash.providers.base import cheapest_gpu, max_non_fp8_kv_vram_gb, supports_fp8_kv
    train = {'max_completion_tokens': 128, 'lora_rank': 32, 'lora_alpha': 64}
    need = model_required_vram_gb('Qwen/Qwen3.5-2B', 'opd', train=train, headroom=1.1)
    assert need <= max_non_fp8_kv_vram_gb()
    assert not supports_fp8_kv(cheapest_gpu(need))

def test_opd_oversized_reject_names_the_knobs_to_shrink():
    """When even the biggest GPU can't hold an OPD run, the reject must be actionable: it names that
    OPD is resident-only (trainer + colocated vLLM student = two weight copies + rollout KV) and the
    knobs that shrink it, not the opaque 'no GPU that big' message the raw cheapest_gpu emits."""
    from flash.providers.base import UnsupportedGpuError, provisional_gpu
    train = {'max_context_tokens': 4096, 'max_completion_tokens': 2048, 'batch_size': 8, 'group_size': 4}
    with pytest.raises(UnsupportedGpuError) as exc:
        provisional_gpu('Qwen/Qwen3.6-35B-A3B', 'opd', train=train)
    msg = str(exc.value)
    assert 'resident-only' in msg
    assert 'group_size' in msg
    assert 'batch_size' in msg
    assert 'max_completion_tokens' in msg

def test_opd_vram_keeps_chunked_text_peak_when_it_exceeds_dense_image_peak():
    """opd reserves the larger of one checkpointed text ce chunk and one dense image sample."""
    from flash.engine.vram import _OPD_CE_PEAK_BYTES_PER_LOGIT, OPD_CE_CHUNK_SIZE, estimate_vram_gb
    kw = {'seq_len': 1, 'max_tokens': OPD_CE_CHUNK_SIZE, 'lora_rank': 16, 'batch_size': 1, 'group_size': 1}
    v1, v2 = (100000, 248320)
    delta = estimate_vram_gb(4.0, 'opd', 'bf16', vocab=v2, **kw) - estimate_vram_gb(4.0, 'opd', 'bf16', vocab=v1, **kw)
    expected = OPD_CE_CHUNK_SIZE * (v2 - v1) * _OPD_CE_PEAK_BYTES_PER_LOGIT / 1000000000.0
    assert delta == pytest.approx(expected, rel=1e-09)

def test_opd_vram_dense_image_peak_grows_with_completion_budget():
    """the dense image fallback grows with the completion rows retained for its loss."""
    from flash.engine.vram import estimate_vram_gb
    kw = {'seq_len': 4096, 'vocab': 248320, 'lora_rank': 16}
    non_think = estimate_vram_gb(4.0, 'opd', 'bf16', thinking=False, **kw)
    think = estimate_vram_gb(4.0, 'opd', 'bf16', thinking=True, **kw)
    assert think > non_think

def test_opd_vram_scales_to_loss_microbatch_not_full_batch():
    """OPD's dense-logit loss budget tracks the worker's loss microbatch.

    It should grow from one to four samples for <=10B models, then stop at the loss microbatch cap
    instead of scaling with the full prompt batch. The 35B path remains serial by default.
    """
    from flash.engine.vram import estimate_vram_gb
    kw = {'seq_len': 1024, 'vocab': 248320, 'lora_rank': 16}
    opd_bs1 = estimate_vram_gb(4.0, 'opd', 'bf16', batch_size=1, group_size=1, **kw)
    opd_bs4 = estimate_vram_gb(4.0, 'opd', 'bf16', batch_size=4, group_size=1, **kw)
    opd_bs16 = estimate_vram_gb(4.0, 'opd', 'bf16', batch_size=16, group_size=1, **kw)
    assert opd_bs4 > opd_bs1
    assert opd_bs16 == opd_bs4
    kw_35b = {'seq_len': 1024, 'lora_rank': 16, 'group_size': 1}
    v1, v2 = (100000, 248320)
    opd_35b_delta_bs1 = estimate_vram_gb(35.0, 'opd', 'bf16', batch_size=1, vocab=v2, **kw_35b) - estimate_vram_gb(35.0, 'opd', 'bf16', batch_size=1, vocab=v1, **kw_35b)
    opd_35b_delta_bs16 = estimate_vram_gb(35.0, 'opd', 'bf16', batch_size=16, vocab=v2, **kw_35b) - estimate_vram_gb(35.0, 'opd', 'bf16', batch_size=16, vocab=v1, **kw_35b)
    assert opd_35b_delta_bs16 == pytest.approx(opd_35b_delta_bs1, rel=1e-09)
    sft_bs1 = estimate_vram_gb(4.0, 'sft', 'bf16', batch_size=1, seq_len=1024, vocab=1, lora_rank=16)
    sft_bs16 = estimate_vram_gb(4.0, 'sft', 'bf16', batch_size=16, seq_len=1024, vocab=1, lora_rank=16)
    assert sft_bs16 > sft_bs1

def test_opd_teacher_rate_matches_fireworks_glm5p2_input_price():
    """glm-5p2 (and the omitted-teacher default) price at Fireworks' $1.40/M input, not the old $0.90
    — opd echo-scoring bills input tokens from the submit-time quote."""
    from flash.cost.facts import teacher_price_per_1m
    assert teacher_price_per_1m('accounts/fireworks/models/glm-5p2')[0] == 1.4
    assert teacher_price_per_1m('')[0] == 1.4

def test_opd_teacher_price_table_covers_every_allowlisted_teacher():
    """Every allow-listed teacher is priced by its exact row (pricing routes through resolve_teacher
    over recipe.TEACHER_MODELS, so there is no unpriced teacher), the new teachers carry their own
    input prices (not silently GLM-priced), and an unknown teacher falls back to the default rate."""
    from flash.cost.facts import teacher_price_per_1m
    from flash.engine.recipe import TEACHER_MODELS
    for info in TEACHER_MODELS.values():
        assert teacher_price_per_1m(info.model_id) == info.usd_per_1m
    assert teacher_price_per_1m('accounts/fireworks/models/deepseek-v4-pro')[0] == 1.74
    assert teacher_price_per_1m('accounts/fireworks/models/kimi-k2p6')[0] == 0.95
    assert teacher_price_per_1m('accounts/fireworks/models/qwen3p7-max')[0] == 1.4
    assert teacher_price_per_1m('accounts/fireworks/models/minimax-m3')[0] == 1.4
    assert teacher_price_per_1m('accounts/fireworks/models/does-not-exist')[0] == 1.4

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
            capture['url'] = req.full_url
            capture['body'] = json.loads(req.data.decode())
        return _FakeResp(payload)
    monkeypatch.setattr(tm._ThreadLocalHttpsTransport, 'urlopen', fake_urlopen)

def test_teacher_score_returns_completion_region_with_rebased_offsets_and_logprobs(monkeypatch):
    payload = {'choices': [{'logprobs': {'tokens': ['P', ':', ' ', 'hi'], 'token_logprobs': [0.0, -1.0, -2.0, -0.5], 'text_offset': [0, 1, 2, 3]}}]}
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    toks = client.score('P: ', 'hi')
    assert len(toks) == 1
    assert toks[0].text == 'hi'
    assert toks[0].start == 0
    assert toks[0].logprob == -0.5
    assert capture['body']['max_tokens'] == 0
    assert capture['body']['echo'] is True
    assert capture['body']['logprobs'] == 1

def test_teacher_score_many_sends_prompt_list_and_maps_choice_indexes(monkeypatch):
    payload = {'choices': [{'index': 1, 'logprobs': {'tokens': ['Q', '2', 'B'], 'token_logprobs': [0.0, -1.0, -0.2], 'text_offset': [0, 1, 2]}}, {'index': 0, 'logprobs': {'tokens': ['Q', '1', 'A'], 'token_logprobs': [0.0, -1.0, -0.1], 'text_offset': [0, 1, 2]}}]}
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    out = client.score_many([('Q1', 'A'), ('Q2', 'B')])
    assert capture['body'] == {'model': 'glm', 'prompt': ['Q1A', 'Q2B'], 'max_tokens': 0, 'echo': True, 'logprobs': 1, 'temperature': 0}
    assert [[t.text for t in toks] for toks in out] == [['A'], ['B']]
    assert [out[0][0].logprob, out[1][0].logprob] == [-0.1, -0.2]

def test_teacher_score_many_multimodal_sends_nested_images_and_extracts_completion_suffix(monkeypatch):
    payload = {'choices': [{'index': 1, 'logprobs': {'tokens': ['User', ': ', '<expanded-image>', '\nAssistant:', ' blue'], 'token_logprobs': [None, -0.1, -0.2, -0.3, -0.7], 'text_offset': [0, 4, 6, 106, 118]}}, {'index': 0, 'logprobs': {'tokens': ['User', ': ', '<expanded-image>', '\nAssistant:', ' red'], 'token_logprobs': [None, -0.1, -0.2, -0.3, -0.4], 'text_offset': [0, 4, 4, 104, 116]}}]}
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient('k', 'https://api.example/v1', 'kimi')
    prompt = 'User: <|media_pad|>\nAssistant: '
    scored = client.score_many_multimodal([(prompt, 'red', ['data:image/png;base64,red']), (prompt, 'blue', ['data:image/png;base64,blue'])])
    assert capture['body'] == {'model': 'kimi', 'prompt': [prompt + 'red', prompt + 'blue'], 'images': [['data:image/png;base64,red'], ['data:image/png;base64,blue']], 'max_tokens': 0, 'echo': True, 'logprobs': 1, 'temperature': 0}
    assert [[token.text for token in tokens] for tokens in scored] == [[' red'], [' blue']]
    assert [(scored[0][0].start, scored[0][0].end), (scored[1][0].start, scored[1][0].end)] == [(0, 3), (0, 4)]
    assert [scored[0][0].logprob, scored[1][0].logprob] == [-0.4, -0.7]
    assert [scored[0].input_tokens, scored[1].input_tokens] == [5, 5]

def test_teacher_score_multimodal_single_request_uses_flat_image_list(monkeypatch):
    payload = {'choices': [{'logprobs': {'tokens': ['prompt', ' answer'], 'token_logprobs': [None, -0.2], 'text_offset': [0, 100]}}]}
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient('k', 'https://api.example/v1', 'kimi')
    client.score_many_multimodal([('prompt<|media_pad|>', 'answer', ['data:image/png;base64,image'])])
    assert capture['body']['prompt'] == 'prompt<|media_pad|>answer'
    assert capture['body']['images'] == ['data:image/png;base64,image']

def test_teacher_multimodal_echo_drops_trailing_zero_width_token(monkeypatch):
    payload = {'choices': [{'logprobs': {'tokens': ['prompt', ' red', ''], 'token_logprobs': [None, -0.4, -0.9], 'text_offset': [0, 100, 104]}}]}
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient('k', 'https://api.example/v1', 'kimi')
    scored = client.score_many_multimodal([('prompt<|media_pad|>', 'red', ['data:image/png;base64,image'])])
    assert [token.text for token in scored[0]] == [' red']
    assert (scored[0][0].start, scored[0][0].end) == (0, 3)
    assert scored[0][0].logprob == -0.4

@pytest.mark.parametrize(('tokens', 'logprobs', 'offsets', 'completion', 'message'), [(['p', ' x'], [None], [0, 100], 'x', 'length'), (['p', ' x'], [None, None], [0, 100], 'x', 'null'), (['p', ' x'], [None, float('nan')], [0, 100], 'x', 'non-finite'), (['p', ' x'], [None, 0.2], [0, 100], 'x', 'positive'), (['p', ' y'], [None, -0.2], [0, 100], 'x', 'exact completion suffix')])
def test_teacher_multimodal_echo_validator_rejects_bad_completion_contract(monkeypatch, tokens, logprobs, offsets, completion, message):
    from flash.engine.worker.teacher import TeacherError
    payload = {'choices': [{'logprobs': {'tokens': tokens, 'token_logprobs': logprobs, 'text_offset': offsets}}]}
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient('k', 'https://api.example/v1', 'kimi')
    with pytest.raises(TeacherError, match=message) as exc_info:
        client.score_many_multimodal([('prompt<|media_pad|>', completion, ['data:image/png;base64,image'])])
    assert exc_info.value.permanent is True

def test_teacher_transport_reuses_connection_and_reconnects_after_eof(monkeypatch):
    import http.client

    import flash.engine.worker.teacher as tm
    payload = {'choices': [{'logprobs': {'tokens': ['P', 'hi'], 'token_logprobs': [0.0, -0.5], 'text_offset': [0, 1]}}]}
    instances = []
    delays = []

    class _Socket:

        def settimeout(self, timeout):
            self.timeout = timeout

    class _Response:
        status = 200
        reason = 'OK'
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
                raise http.client.RemoteDisconnected('stale keep-alive socket')

        def getresponse(self):
            return _Response()

        def close(self):
            self.sock = None
    monkeypatch.setattr(tm.http.client, 'HTTPSConnection', _Connection)
    monkeypatch.setattr(tm.time, 'sleep', delays.append)
    client = TeacherClient('k', 'https://api.example/v1', 'glm', max_retries=2)
    client.score('P', 'hi')
    client.score('P', 'hi')
    client.score('P', 'hi')
    client.score('P', 'hi')
    assert len(instances) == 2
    assert instances[0].request_count == 3
    assert instances[1].request_count == 2
    assert delays == [2.0]

def test_teacher_score_keeps_boundary_crossing_token_clamped_to_completion(monkeypatch):
    payload = {'choices': [{'logprobs': {'tokens': ['P', ':', ' hi', '!'], 'token_logprobs': [0.0, -1.0, -0.5, -0.2], 'text_offset': [0, 1, 2, 5]}}]}
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    toks = client.score('P: ', 'hi!')
    assert [t.text for t in toks] == [' hi', '!']
    assert (toks[0].start, toks[0].end) == (0, 2)
    assert (toks[1].start, toks[1].end) == (2, 3)
    assert toks[0].logprob == -0.5

def test_teacher_score_raises_on_malformed_response(monkeypatch):
    from flash.engine.worker.teacher import TeacherError
    _mock_urlopen(monkeypatch, {'choices': [{'logprobs': {}}]})
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    with pytest.raises(TeacherError) as ei:
        client.score('', 'hi')
    assert ei.value.permanent is True

def test_teacher_score_treats_mismatched_array_lengths_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): a 200 response whose tokens / token_logprobs / text_offset
    arrays disagree in length would IndexError inside the per-token loop and escape as a generic
    (non-TeacherError) exception, so a teacher that consistently returns malformed arrays could burn
    every OPD step before the run fails with "no trained step". A length mismatch is a broken contract
    -> PERMANENT (abort now)."""
    from flash.engine.worker.teacher import TeacherError

    def _payload(tokens, logprobs, offsets):
        return {'choices': [{'logprobs': {'tokens': tokens, 'token_logprobs': logprobs, 'text_offset': offsets}}]}
    cases = [(['a', 'b', 'c'], [0.0, -1.0], [0, 1, 2]), (['a', 'b'], [0.0, -1.0, -2.0], [0, 1]), (['a', 'b'], [0.0, -1.0], [0, 1, 2])]
    for tokens, logprobs, offsets in cases:
        _mock_urlopen(monkeypatch, _payload(tokens, logprobs, offsets))
        client = TeacherClient('k', 'https://api.example/v1', 'glm')
        with pytest.raises(TeacherError) as ei:
            client.score('', ''.join(tokens))
        assert ei.value.permanent is True, f'{(tokens, logprobs, offsets)} must be PERMANENT'
        assert 'length' in str(ei.value).lower()

def test_teacher_score_rejects_null_logprob_on_completion_token_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): a null (None) realized logprob is legitimate ONLY for
    unscored PROMPT context. A None on a token that overlaps the COMPLETION (the ones score() keeps)
    means the teacher did not score it; coercing it to 0.0 (log-prob 1.0 == full confidence) would
    train the gkd loss on fabricated teacher confidence, so it must abort like the other contract
    violations. A prompt-context null (dropped anyway) must NOT trip it."""
    from flash.engine.worker.teacher import TeacherError
    bad = {'choices': [{'logprobs': {'tokens': ['P', 'hi'], 'token_logprobs': [0.0, None], 'text_offset': [0, 1]}}]}
    _mock_urlopen(monkeypatch, bad)
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    with pytest.raises(TeacherError) as ei:
        client.score('P', 'hi')
    assert ei.value.permanent is True
    assert 'null' in str(ei.value).lower()
    ok = {'choices': [{'logprobs': {'tokens': ['P', 'hi'], 'token_logprobs': [None, -0.5], 'text_offset': [0, 1]}}]}
    _mock_urlopen(monkeypatch, ok)
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    toks = client.score('P', 'hi')
    assert toks
    assert toks[0].logprob == -0.5

def test_teacher_4xx_is_permanent_but_5xx_is_transient(monkeypatch):
    import urllib.error

    import flash.engine.worker.teacher as tm
    from flash.engine.worker.teacher import TeacherError

    def raise_http(code):

        def fake_urlopen(_transport, req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, f'HTTP {code}', {}, None)
        monkeypatch.setattr(tm._ThreadLocalHttpsTransport, 'urlopen', fake_urlopen)
    client = TeacherClient('k', 'https://api.example/v1', 'glm', max_retries=1)
    raise_http(401)
    with pytest.raises(TeacherError) as ei:
        client.score('P', 'hi')
    assert ei.value.permanent is True
    raise_http(503)
    with pytest.raises(TeacherError) as ei:
        client.score('P', 'hi')
    assert ei.value.permanent is False

def test_teacher_http_error_diagnostic_omits_opaque_response_body(monkeypatch):
    import traceback
    import urllib.error

    import flash.engine.worker.teacher as tm
    from flash.engine.worker.teacher import TeacherError
    private = b'opaque-private-teacher-sentinel-91ad'

    def raise_http(_transport, req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, private.decode(), {}, io.BytesIO(private))
    monkeypatch.setattr(tm._ThreadLocalHttpsTransport, 'urlopen', raise_http)
    client = TeacherClient('k', 'https://api.example/v1', 'glm', max_retries=1)
    with pytest.raises(TeacherError) as exc_info:
        client.score('P', 'hi')
    detail = str(exc_info.value)
    formatted = ''.join(traceback.format_exception(exc_info.type, exc_info.value, exc_info.tb))
    assert exc_info.value.permanent is True
    assert 'teacher HTTP 403' in detail
    assert '/completions' in detail
    assert 'permanent' in detail
    assert private.decode() not in detail
    assert private.decode() not in formatted

def test_teacher_score_rejects_non_list_logprob_fields_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py:130): the length check assumes tokens/token_logprobs/
    text_offset are sequences. A malformed 200 with token_logprobs=null (or a scalar text_offset) makes
    len()/indexing raise TypeError OUTSIDE TeacherError, so a consistently malformed teacher could burn
    every OPD step. Non-list fields must raise a PERMANENT TeacherError up front."""
    from flash.engine.worker.teacher import TeacherError
    payload = {'choices': [{'logprobs': {'tokens': ['a', 'b'], 'token_logprobs': None, 'text_offset': [0, 1]}}]}
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    with pytest.raises(TeacherError) as ei:
        client.score('', 'ab')
    assert ei.value.permanent is True
    assert 'not all lists' in str(ei.value)

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
            return b'<html>502 Bad Gateway</html>'
    monkeypatch.setattr(tm._ThreadLocalHttpsTransport, 'urlopen', lambda _transport, req, timeout=None: _Resp())
    monkeypatch.setattr(tm.time, 'sleep', lambda *a, **k: None)
    client = TeacherClient('k', 'https://api.example/v1', 'glm', max_retries=2)
    with pytest.raises(TeacherError) as ei:
        client.score('P', 'hi')
    assert ei.value.permanent is False
    assert 'unparseable' in str(ei.value).lower()

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
            raise http.client.IncompleteRead(b'half', 100)
    monkeypatch.setattr(tm._ThreadLocalHttpsTransport, 'urlopen', lambda _transport, req, timeout=None: _Resp())
    monkeypatch.setattr(tm.time, 'sleep', lambda *a, **k: None)
    client = TeacherClient('k', 'https://api.example/v1', 'glm', max_retries=2)
    with pytest.raises(TeacherError) as ei:
        client.score('P', 'hi')
    assert ei.value.permanent is False
    assert 'truncated' in str(ei.value).lower()

def test_teacher_score_rejects_non_numeric_or_unordered_offsets_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): a malformed 200 can put a value in text_offset that passes
    the list/length guards yet corrupts the alignment: non-numeric (null/string) or out-of-order (as
    before), and also non-finite (int(NaN) RAISES outside TeacherError -> unclassified skip), fractional
    (int() silently truncates to a wrong char index), or out-of-[0, len(full)] (a span outside the
    completion region). All must be rejected up front as PERMANENT so the worker aborts, not skip-burns.
    full = 'P' + 'hi' = 'Phi', len 3."""
    from flash.engine.worker.teacher import TeacherError

    def _payload(offsets):
        return {'choices': [{'logprobs': {'tokens': ['P', 'hi'], 'token_logprobs': [0.0, -0.5], 'text_offset': offsets}}]}
    for bad, needle in (([0, None], 'non-numeric'), ([0, 'x'], 'non-numeric'), ([2, 1], 'non-decreasing'), ([0, 1.5], 'not an integer'), ([0, float('nan')], 'not finite'), ([0, 9], 'outside'), ([-1, 0], 'outside')):
        _mock_urlopen(monkeypatch, _payload(bad))
        client = TeacherClient('k', 'https://api.example/v1', 'glm')
        with pytest.raises(TeacherError) as ei:
            client.score('P', 'hi')
        assert ei.value.permanent is True, f'{bad!r} must be PERMANENT'
        assert needle in str(ei.value).lower(), f'{bad!r}: expected {needle!r} in {ei.value}'

def test_teacher_score_rejects_non_numeric_or_nonfinite_logprobs_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): token_logprobs[i] is coerced with float(...) below (None ->
    0.0 for a null realized logprob). A malformed 200 can still carry a non-numeric value (float() raises
    ValueError OUTSIDE TeacherError and can burn every OPD step) or a non-finite NaN/inf (feeds a
    poisoned gradient straight into the gkd loss). Both must be
    rejected up front as PERMANENT; a null logprob stays allowed (handled as 0.0)."""
    from flash.engine.worker.teacher import TeacherError

    def _payload(logprobs):
        return {'choices': [{'logprobs': {'tokens': ['P', 'hi'], 'token_logprobs': logprobs, 'text_offset': [0, 1]}}]}
    for bad, needle in (([0.0, 'x'], 'non-numeric'), ([0.0, float('nan')], 'non-finite'), ([0.0, float('inf')], 'non-finite')):
        _mock_urlopen(monkeypatch, _payload(bad))
        client = TeacherClient('k', 'https://api.example/v1', 'glm')
        with pytest.raises(TeacherError) as ei:
            client.score('P', 'hi')
        assert ei.value.permanent is True, f'{bad!r} must be PERMANENT'
        assert needle in str(ei.value).lower(), f'{bad!r}: expected {needle!r} in {ei.value}'
    _mock_urlopen(monkeypatch, _payload([None, -0.5]))
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    toks = client.score('P', 'hi')
    assert toks
    assert toks[0].logprob == -0.5

def test_teacher_score_rejects_positive_logprob_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): a log-probability cannot exceed 0. A malformed 200 with a
    POSITIVE token_logprob is a probability > 1; summed into teacher_logsum it poisons the reverse-KL
    coefficient with impossible teacher mass, so OPD would train on a bogus signal instead of aborting.
    Reject as PERMANENT like the other teacher-contract violations. A ~0 logprob (near-deterministic
    token) stays allowed via the small float-rounding tolerance."""
    from flash.engine.worker.teacher import TeacherError

    def _payload(logprobs):
        return {'choices': [{'logprobs': {'tokens': ['P', 'hi'], 'token_logprobs': logprobs, 'text_offset': [0, 1]}}]}
    _mock_urlopen(monkeypatch, _payload([0.0, 2.5]))
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    with pytest.raises(TeacherError) as ei:
        client.score('P', 'hi')
    assert ei.value.permanent is True
    assert 'positive' in str(ei.value).lower()
    _mock_urlopen(monkeypatch, _payload([None, 1e-09]))
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    toks = client.score('P', 'hi')
    assert toks
    assert abs(toks[0].logprob - 1e-09) < 1e-12

def test_teacher_score_rejects_truncated_echo_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): a malformed 200 with equal-length arrays can still OMIT a
    suffix of `full`. The final token's end falls back to len(full), so a truncated echo stretches the
    last returned token across text the teacher never scored (and, if that token sits in the prompt,
    drags it across the boundary into the completion) — a fabricated span. The echoed tokens must tile
    the whole input, so a last token whose own text ends short of len(full) is rejected as PERMANENT.
    prompt 'P' (plen 1) + 'hello' = 'Phello' (len 6); an echo of only ['P','h'] ends at char 2."""
    from flash.engine.worker.teacher import TeacherError
    payload = {'choices': [{'logprobs': {'tokens': ['P', 'h'], 'token_logprobs': [0.0, -0.5], 'text_offset': [0, 1]}}]}
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    with pytest.raises(TeacherError) as ei:
        client.score('P', 'hello')
    assert ei.value.permanent is True
    assert 'does not tile' in str(ei.value).lower()

def test_teacher_score_rejects_same_length_wrong_text_token_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): the tiling guard must compare the token TEXT to the echoed
    substring, not just its LENGTH. A malformed 200 echoing a same-length-but-different token over the
    right offsets (here 'XY' where full[1:3]=='hi') passes a length-only check yet trains the gkd loss
    on the wrong token's logprob. full 'P'+'hi'='Phi' (len 3); token 1 'XY' (len 2, == the span length)
    must still be rejected as PERMANENT because its text isn't the echoed substring."""
    from flash.engine.worker.teacher import TeacherError
    payload = {'choices': [{'logprobs': {'tokens': ['P', 'XY'], 'token_logprobs': [0.0, -0.5], 'text_offset': [0, 1]}}]}
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    with pytest.raises(TeacherError) as ei:
        client.score('P', 'hi')
    assert ei.value.permanent is True
    assert 'does not tile' in str(ei.value).lower()

def test_teacher_score_rejects_interior_tiling_gap_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): the coverage guard must validate EVERY token boundary, not
    only the final one. An INTERIOR gap/overlap — offsets[i+1] != offsets[i] + len(tokens[i]) — makes the
    emit loop use offsets[i+1] as token i's end, assigning token i's logprob to text the teacher never
    scored (a fabricated completion span when the gap straddles plen). full 'P'+'hiyo' = 'Phiyo' (len 5);
    a mid-sequence offset jump (token 1 'h' ends at char 2 but the next offset is 3) must be PERMANENT."""
    from flash.engine.worker.teacher import TeacherError
    payload = {'choices': [{'logprobs': {'tokens': ['P', 'h', 'yo'], 'token_logprobs': [0.0, -0.3, -0.5], 'text_offset': [0, 1, 3]}}]}
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    with pytest.raises(TeacherError) as ei:
        client.score('P', 'hiyo')
    assert ei.value.permanent is True
    assert 'does not tile' in str(ei.value).lower()

def test_teacher_score_rejects_echo_not_starting_at_offset_0_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): the tiling guard proves coverage of full[offsets[0]:] only —
    it never requires offsets[0] == 0. A malformed 200 that DROPS a prompt prefix and echoes a cleanly-
    tiling SUFFIX passes every offset/tiling check, but its completion logprobs were computed over a
    TRUNCATED prompt, so the gkd signal is scored against context the student never saw. full 'AB'+'cd' =
    'ABcd' (len 4); an echo of ['B','cd'] at offsets [1,2] tiles full[1:4] cleanly yet omits 'A' — the
    first offset is 1, not 0, and must be rejected as PERMANENT."""
    from flash.engine.worker.teacher import TeacherError
    payload = {'choices': [{'logprobs': {'tokens': ['B', 'cd'], 'token_logprobs': [0.0, -0.5], 'text_offset': [1, 2]}}]}
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    with pytest.raises(TeacherError) as ei:
        client.score('AB', 'cd')
    assert ei.value.permanent is True
    assert 'offset 0' in str(ei.value).lower()

def test_teacher_score_rejects_echo_with_no_completion_tokens_as_permanent(monkeypatch):
    """Regression (codex[bot], teacher.py): an echo that yields NO completion-region token for a
    non-empty completion (here the degenerate empty-arrays 200) scored nothing to distil; score() must
    reject it as PERMANENT instead of returning an empty list that then burns every OPD step on no
    signal before the generic no-trained-step failure."""
    from flash.engine.worker.teacher import TeacherError
    payload = {'choices': [{'logprobs': {'tokens': [], 'token_logprobs': [], 'text_offset': []}}]}
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient('k', 'https://api.example/v1', 'glm')
    with pytest.raises(TeacherError) as ei:
        client.score('P', 'hi')
    assert ei.value.permanent is True
    assert 'no completion-region tokens' in str(ei.value).lower()

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
            raise http.client.IncompleteRead(b'', 10)
    err = _BadBodyHTTPError('http://x', 503, 'Service Unavailable', {}, None)
    err.fp = object()

    def raise_503(_transport, req, timeout=None):
        raise err
    monkeypatch.setattr(tm._ThreadLocalHttpsTransport, 'urlopen', raise_503)
    monkeypatch.setattr(tm.time, 'sleep', lambda *a, **k: None)
    client = TeacherClient('k', 'https://api.example/v1', 'glm', max_retries=2)
    with pytest.raises(TeacherError) as ei:
        client.score('P', 'hi')
    assert ei.value.permanent is False
    assert '503' in str(ei.value)

def test_resolve_opd_knobs_rejects_zero_kl_penalty(monkeypatch):
    """Regression (codex[bot], opd.py:64): kl_penalty_coef scales the gkd objective, so an explicit 0
    (allowed by the shared schema for GRPO) makes every OPD backward a zero gradient while opt_steps
    still advances -> a fully-untrained adapter is published/charged. _resolve_opd_knobs must reject 0;
    omitting the field (None) still resolves to the positive recipe default."""
    from flash.engine.worker import opd as opd_mod

    class _Train:

        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, name):
            return None
    monkeypatch.setattr(opd_mod, '_w', SimpleNamespace(JOB_SPEC=SimpleNamespace(train=_Train(kl_penalty_coef=0.0)), THINKING=False))
    with pytest.raises(RuntimeError, match='kl_penalty_coef must be > 0'):
        opd_mod._resolve_opd_knobs()
    monkeypatch.setattr(opd_mod, '_w', SimpleNamespace(JOB_SPEC=SimpleNamespace(train=_Train(kl_penalty_coef=None)), THINKING=False))
    assert opd_mod._resolve_opd_knobs().kl_coef > 0.0

def test_resolve_opd_knobs_resolves_teacher_from_train(monkeypatch):
    """_resolve_opd_knobs defensively re-resolves [train].teacher_model at the worker's (tolerant)
    deserialization boundary: parse already canonicalized it to a Fireworks model id, but the worker
    still validates — accepting an alias or the model id — so the TeacherClient sends a supported model.
    An unset value keeps the default GLM 5.2 teacher; the shared base_url is unchanged; an unsupported
    teacher fails loudly on the worker."""
    from flash.engine.worker import opd as opd_mod

    class _Train:

        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, name):
            return None

    def _knobs(teacher):
        monkeypatch.setattr(opd_mod, '_w', SimpleNamespace(JOB_SPEC=SimpleNamespace(train=_Train(teacher_model=teacher)), THINKING=False))
        return opd_mod._resolve_opd_knobs()
    assert _knobs('kimi-k2.6').teacher_model == 'accounts/fireworks/models/kimi-k2p6'
    assert _knobs('deepseek-v4-pro').teacher_model == 'accounts/fireworks/models/deepseek-v4-pro'
    assert _knobs('').teacher_model == 'accounts/fireworks/models/glm-5p2'
    assert _knobs(None).teacher_model == 'accounts/fireworks/models/glm-5p2'
    assert _knobs('deepseek-v4-pro').teacher_base_url == opd_mod.RECIPE.opd.teacher_base_url
    with pytest.raises(RuntimeError, match='teacher_model'):
        _knobs('gpt-5.5')

def _dense_gkd_loss_from_logits_rows(rows, student_ids, groups, kl_coef=1.0):
    import torch
    import torch.nn.functional as F

    from flash.engine.worker import opd as opd_mod
    if not student_ids or not groups:
        return None
    prepared = groups if isinstance(groups, opd_mod._PreparedGkdGroups) else opd_mod._prepare_gkd_groups(groups)
    if prepared is None:
        return None
    rows = rows.float()
    token_ids = torch.tensor(student_ids, device=rows.device)
    logps = -F.cross_entropy(rows, token_ids, reduction='none')
    return opd_mod._gkd_loss_from_logps(logps, prepared, kl_coef=kl_coef)

def test_opd_loss_skips_empty_student_group_without_crashing():
    torch = pytest.importorskip('torch')
    rows = torch.zeros(1, 8, requires_grad=True)
    loss = _dense_gkd_loss_from_logits_rows(rows, [2], [([], -1.0), ([0], -2.0)], kl_coef=1.0)
    assert loss is not None
    loss.backward()
    assert rows.grad is not None
    assert rows.grad[0].abs().sum() > 0

def test_groupwise_alignment_emits_no_empty_student_group():
    student = _student([(2, 3), (3, 5)])
    teacher = _teacher([(0, 2), (2, 5)])
    groups = groupwise_alignment(student, teacher)
    assert all((s_idx for s_idx, _ in groups))
    assert [s_idx for s_idx, _ in groups] == [[0, 1]]

def test_teacher_client_requires_key():
    from flash.engine.worker.teacher import TeacherError
    with pytest.raises(TeacherError):
        TeacherClient('', 'https://api.example/v1', 'glm')

def test_opd_spec_json_round_trip():
    from flash.schema import spec_from_dict
    from flash.spec import JobSpec
    spec = spec_from_dict({'model': 'Qwen/Qwen3.5-4B', 'algorithm': 'opd', 'environment': {'id': 'github:owner/repo@main:env/environment.py'}, 'train': {'epochs': 25, 'max_examples': 8, 'batch_size': 8, 'hf_repo': 'owner/runs'}}, run_id='x')
    restored = JobSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.phase == 'opd'
    assert 'FIREWORKS_API_KEY' not in restored.environment.secrets

def test_opd_cost_is_step_priced_and_bills_teacher_tokens():
    from flash.cost.spec import estimate_for_spec, spec_steps
    from flash.schema import spec_from_dict
    spec = spec_from_dict({'model': 'Qwen/Qwen3.5-0.8B', 'algorithm': 'opd', 'environment': {'id': 'github:owner/repo@main:env/environment.py'}, 'train': {'epochs': 30, 'hf_repo': 'owner/runs'}}, run_id='x')
    assert spec_steps(spec) == 30
    est = estimate_for_spec(spec)
    assert est.method == 'opd'
    assert est.teacher_api_usd > 0.0
    assert est.total_usd == pytest.approx(est.billable_hours * est.gpu_hourly_usd)
    assert 'opd step' in ' '.join(est.notes)

def test_opd_loss_backpropagates_over_grouped_spans():
    torch = pytest.importorskip('torch')
    V = 8
    student_ids = [2, 3]
    rows = torch.zeros(len(student_ids), V, requires_grad=True)
    groups = [([0, 1], -1.5)]
    loss = _dense_gkd_loss_from_logits_rows(rows, student_ids, groups, kl_coef=1.0)
    assert loss is not None
    assert loss.requires_grad
    loss.backward()
    assert rows.grad[0].abs().sum() > 0
    assert rows.grad[1].abs().sum() > 0

def test_gkd_loss_from_logits_rows_matches_manual_logprob_math():
    torch = pytest.importorskip('torch')
    from flash.engine.worker import opd as opd_mod
    rows = torch.tensor([[0.2, -0.3, 0.7], [-0.5, 1.0, 0.1], [0.4, -0.2, 0.3]], dtype=torch.float32, requires_grad=True)
    student_ids = [2, 1, 0]
    groups = [([0, 1], -0.75), ([2], -0.25)]
    loss = _dense_gkd_loss_from_logits_rows(rows, student_ids, groups, kl_coef=0.5)
    manual_logps = rows.gather(1, torch.tensor(student_ids).unsqueeze(1)).squeeze(1) - torch.logsumexp(rows, dim=-1)
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
    prepared_loss = _dense_gkd_loss_from_logits_rows(rows2, student_ids, prepared, kl_coef=0.5)
    assert prepared_loss is not None
    torch.testing.assert_close(prepared_loss, expected.detach())

def test_opd_loss_none_without_groups_or_tokens():
    torch = pytest.importorskip('torch')
    rows = torch.zeros(2, 4, requires_grad=True)
    assert _dense_gkd_loss_from_logits_rows(rows, [2, 3], [], kl_coef=1.0) is None
    assert _dense_gkd_loss_from_logits_rows(rows[:0], [], [([0], -1.0)], kl_coef=1.0) is None

def test_opd_loss_coefficient_tracks_student_minus_teacher_logprob():
    torch = pytest.importorskip('torch')
    V = 8
    student_ids = [2]
    rows_hi = torch.zeros(1, V, requires_grad=True)
    rows_lo = torch.zeros(1, V, requires_grad=True)
    hi = _dense_gkd_loss_from_logits_rows(rows_hi, student_ids, [([0], -5.0)], kl_coef=1.0)
    lo = _dense_gkd_loss_from_logits_rows(rows_lo, student_ids, [([0], -0.5)], kl_coef=1.0)
    assert hi is not None
    assert lo is not None
    assert float(hi.detach()) < float(lo.detach())

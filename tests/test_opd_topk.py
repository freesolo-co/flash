from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.engine.worker.opd_topk import (
    CandidateGroup,
    CandidateRequest,
    CandidateTeacherScorer,
    TeacherInputMeter,
    candidate_set_cross_entropy,
    decode_candidate_groups,
    selected_prefix_indices,
)
from flash.engine.worker.teacher import TeacherError


class _BoundaryTokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        values = tuple(int(token_id) for token_id in ids)
        if values == (9,):
            return "x"
        if values == (9, 5):
            return "xy"
        mapping = {0: "", 2: "a", 3: "a", 4: "�", 5: " y", 9: "x"}
        return "".join(mapping.get(token_id, "z") for token_id in values)


def test_candidate_surfaces_exclude_eos_group_duplicates_and_reject_invalid():
    torch = pytest.importorskip("torch")
    logits = torch.full((10,), -20.0)
    logits[0] = 10.0
    logits[2] = 9.0
    logits[3] = 8.0
    logits[4] = 7.0
    logits[5] = 6.0

    groups, duplicates, invalid = decode_candidate_groups(
        _BoundaryTokenizer(), [9], logits, frozenset({0})
    )

    assert groups == (CandidateGroup("a", (2, 3)), CandidateGroup("y", (5,)))
    assert duplicates == 1
    assert invalid == 1
    assert all(0 not in group.token_ids for group in groups)


def test_candidate_surface_uses_prefix_boundary_not_token_alone():
    torch = pytest.importorskip("torch")
    logits = torch.full((10,), -20.0)
    logits[5] = 9.0
    logits[2] = 8.0
    logits[3] = 7.0
    logits[4] = 6.0

    groups, _duplicates, _invalid = decode_candidate_groups(
        _BoundaryTokenizer(), [9], logits, frozenset()
    )

    assert groups[0].surface == "y"
    assert _BoundaryTokenizer().decode([5], skip_special_tokens=True) == " y"


def test_candidate_selection_never_silently_lowers_fixed_k():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0])

    with pytest.raises(RuntimeError, match="requires 4 finite non-eos"):
        decode_candidate_groups(_BoundaryTokenizer(), [], logits, frozenset({0}))


def test_candidate_normalization_and_loss_have_gradients():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([2.0, 1.0, -0.5], requires_grad=True)
    groups = (CandidateGroup("a", (0, 1)), CandidateGroup("b", (2,)))

    loss = candidate_set_cross_entropy(logits, groups, [-0.2, -1.7])
    loss.backward()

    student_surface_logits = torch.stack(
        (torch.logsumexp(logits.detach()[:2], dim=0), logits.detach()[2])
    )
    expected = -(
        torch.softmax(torch.tensor([-0.2, -1.7]), dim=0)
        * torch.log_softmax(student_surface_logits, dim=0)
    ).sum()
    assert torch.allclose(loss.detach(), expected)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_candidate_teacher_scoring_batches_and_caches_exact_requests():
    calls = []

    class _Teacher:
        def score_many(self, items):
            calls.append(items)
            return [[SimpleNamespace(logprob=-0.25)] for _item in items]

    meter = TeacherInputMeter()
    meter.record_ordinary(100)
    scorer = CandidateTeacherScorer(_Teacher(), meter, {})
    requests = [
        CandidateRequest("ctx", "a", 10),
        CandidateRequest("ctx", "a", 10),
        CandidateRequest("ctx", "b", 12),
    ]

    first = scorer.score([tuple(requests)])
    second = scorer.score([tuple(requests)])

    assert calls == [[("ctx", "a"), ("ctx", "b")]]
    assert first.admitted_sets == (True,)
    assert first.requests == 2
    assert first.cache_hits == 0
    assert second.admitted_sets == (True,)
    assert second.requests == 0
    assert second.cache_hits == 2
    assert meter.candidate_tokens == 22


def test_candidate_teacher_scoring_rejects_entire_set_before_partial_charge():
    calls = []

    class _Teacher:
        def score_many(self, items):
            calls.append(items)
            return [[SimpleNamespace(logprob=-0.25)] for _item in items]

    meter = TeacherInputMeter()
    meter.record_ordinary(70)
    assert meter.reserve_candidate(40)
    scorer = CandidateTeacherScorer(_Teacher(), meter, {})
    request_set = tuple(CandidateRequest("ctx", surface, 10) for surface in "abcd")

    scored = scorer.score([request_set])

    assert scored.admitted_sets == (False,)
    assert scored.budget_exhausted
    assert scored.logprobs == {}
    assert scored.requests == 0
    assert calls == []
    assert meter.candidate_tokens == 40
    assert meter.cap_tokens - meter.total_tokens == 30


def test_candidate_teacher_scoring_reserves_only_exact_uncached_members():
    calls = []

    class _Teacher:
        def score_many(self, items):
            calls.append(items)
            return [[SimpleNamespace(logprob=-0.25)] for _item in items]

    meter = TeacherInputMeter()
    meter.record_ordinary(20)
    cache = {("ctx", "a"): -0.1}
    scorer = CandidateTeacherScorer(_Teacher(), meter, cache)
    request_set = (
        CandidateRequest("ctx", "a", 6),
        CandidateRequest("ctx", "b", 7),
        CandidateRequest("ctx", "c", 8),
    )

    first = scorer.score([request_set])
    charged = meter.candidate_tokens
    second = scorer.score([request_set])

    assert first.admitted_sets == (True,)
    assert first.cache_hits == 1
    assert first.requests == 2
    assert calls == [[("ctx", "b"), ("ctx", "c")]]
    assert charged == 15
    assert second.admitted_sets == (True,)
    assert second.cache_hits == 3
    assert second.requests == 0
    assert meter.candidate_tokens == charged


def test_candidate_teacher_scoring_rejects_invalid_responses():
    class _Teacher:
        def score_many(self, _items):
            return [[]]

    meter = TeacherInputMeter()
    meter.record_ordinary(100)
    scorer = CandidateTeacherScorer(_Teacher(), meter)

    with pytest.raises(TeacherError, match="no realized completion tokens") as exc:
        scorer.score([(CandidateRequest("ctx", "a", 10),)])
    assert exc.value.permanent


def test_invalid_candidate_batch_does_not_partially_populate_exact_cache():
    class _Teacher:
        def score_many(self, _items):
            return [[SimpleNamespace(logprob=-0.25)], []]

    meter = TeacherInputMeter()
    meter.record_ordinary(100)
    cache = {}
    scorer = CandidateTeacherScorer(_Teacher(), meter, cache)
    request_set = (
        CandidateRequest("ctx", "a", 10),
        CandidateRequest("ctx", "b", 10),
    )

    with pytest.raises(TeacherError, match="no realized completion tokens"):
        scorer.score([request_set])

    assert cache == {}
    assert meter.candidate_tokens == 20


def test_teacher_input_meter_refuses_request_before_crossing_hard_cap():
    meter = TeacherInputMeter()
    meter.record_ordinary(10)

    assert meter.reserve_candidate(10)
    assert meter.total_tokens == meter.cap_tokens == 20
    assert not meter.reserve_candidate(1)
    assert meter.total_tokens == 20
    assert meter.budget_exhausted


def test_selection_cadence_is_deterministic_and_positive():
    assert selected_prefix_indices(10, 4) == (4, 8)
    assert selected_prefix_indices(9, 3) == (3, 6)
    assert selected_prefix_indices(1, 100) == ()
    assert selected_prefix_indices(0, 4) == ()
    with pytest.raises(ValueError, match="positive"):
        selected_prefix_indices(10, 0)


def test_c13_cadence_larger_than_completion_does_no_candidate_work():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        eos_token_id = 5

        def decode(self, ids, skip_special_tokens=True):
            return "".join(str(int(token_id)) for token_id in ids)

    class _Teacher:
        def __init__(self):
            self.calls = []

        def score_many(self, items):
            self.calls.append(items)
            raise AssertionError("teacher must not be called")

    teacher = _Teacher()
    meter = TeacherInputMeter()
    meter.record_ordinary(10)
    outputs = opd_mod._candidate_objectives_for_chunk(
        torch.zeros((1, 2, 6), requires_grad=True),
        [
            SimpleNamespace(
                prompt_ids=[1],
                student_ids=[2],
                score_result=SimpleNamespace(teacher_prompt="teacher:"),
            )
        ],
        _Tok(),
        SimpleNamespace(topk_cadence=100),
        teacher,
        meter,
        {},
    )

    assert teacher.calls == []
    assert meter.candidate_tokens == 0
    assert outputs[0].term is None
    assert outputs[0].selected_prefixes == 0
    assert outputs[0].scored_prefixes == 0


def test_c13_candidate_set_admission_is_atomic_at_prefix_boundary():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        eos_token_id = 5

        def decode(self, ids, skip_special_tokens=True):
            mapping = {0: "d", 1: "p", 2: "a", 3: "b", 4: "c", 5: ""}
            return "".join(mapping[int(token_id)] for token_id in ids)

    class _Teacher:
        def __init__(self):
            self.calls = []

        def score_many(self, items):
            self.calls.append(items)
            return [[SimpleNamespace(logprob=-0.25)] for _item in items]

    logits = torch.zeros((1, 10, 6), requires_grad=True)
    logits.data[0, 8] = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, 9.0])
    teacher = _Teacher()
    meter = TeacherInputMeter()
    meter.record_ordinary(70)
    assert meter.reserve_candidate(40)
    outputs = opd_mod._candidate_objectives_for_chunk(
        logits,
        [
            SimpleNamespace(
                prompt_ids=[1] * 8,
                student_ids=[2, 3],
                score_result=SimpleNamespace(teacher_prompt="teacher:"),
            )
        ],
        _Tok(),
        SimpleNamespace(topk_cadence=1),
        teacher,
        meter,
        {},
    )

    assert teacher.calls == []
    assert meter.candidate_tokens == 40
    assert meter.cap_tokens - meter.total_tokens == 30
    assert outputs[0].term is None
    assert outputs[0].selected_prefixes == 1
    assert outputs[0].scored_prefixes == 0
    assert outputs[0].budget_exhausted


def test_c13_rejects_single_surface_prefix_before_metering_or_calls():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        eos_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            mapping = {0: "", 1: "a", 2: "a", 3: "", 4: "�", 9: "p"}
            return "".join(mapping.get(int(token_id), "z") for token_id in ids)

    class _Teacher:
        def __init__(self):
            self.calls = []

        def score_many(self, items):
            self.calls.append(items)
            raise AssertionError("teacher must not be called")

    logits = torch.full((1, 3, 10), -torch.inf, requires_grad=True)
    logits.data[0, 1, 1:5] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    teacher = _Teacher()
    meter = TeacherInputMeter()
    meter.record_ordinary(100)
    outputs = opd_mod._candidate_objectives_for_chunk(
        logits,
        [
            SimpleNamespace(
                prompt_ids=[9],
                student_ids=[9, 9],
                score_result=SimpleNamespace(teacher_prompt="teacher:"),
            )
        ],
        _Tok(),
        SimpleNamespace(topk_cadence=1),
        teacher,
        meter,
        {},
    )

    assert teacher.calls == []
    assert meter.candidate_tokens == 0
    assert outputs[0].term is None
    assert outputs[0].selected_prefixes == 1
    assert outputs[0].scored_prefixes == 0
    assert outputs[0].duplicate_surfaces == 1
    assert outputs[0].invalid_candidates == 2
    assert not outputs[0].budget_exhausted


@pytest.mark.parametrize("objective_ids", [(), ("c0",), ("c13",)])
def test_nonfiring_objectives_do_not_render_or_allocate_candidate_prompts(
    monkeypatch, objective_ids
):
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod
    from flash.engine.worker.tokenizer_align import TeacherToken

    class _Tok:
        pad_token_id = 0
        eos_token_id = 5

        def decode(self, ids, skip_special_tokens=True):
            mapping = {1: "p", 2: "a", 5: ""}
            return "".join(mapping.get(int(token_id), "d") for token_id in ids)

    class _Model:
        def __init__(self):
            self.weight = torch.tensor(
                [[0.0, 0.0, 2.0, 1.0, 0.5, 9.0], [0.0] * 6], requires_grad=True
            )
            self.config = SimpleNamespace(use_cache=True)

        def __call__(self, input_ids, **_kwargs):
            batch, length = input_ids.shape
            return SimpleNamespace(logits=self.weight[:length].unsqueeze(0).expand(batch, -1, -1))

        def parameters(self):
            return [self.weight]

        def train(self, mode=True):
            return self

    class _Teacher:
        def __init__(self):
            self.calls = []

        def score_many(self, items):
            self.calls.append(items)
            return [[TeacherToken(surface, -0.5, 0, len(surface))] for _context, surface in items]

    rendered = []

    def _render_teacher_prompt(prompt_messages, thinking_prefill=""):
        rendered.append((prompt_messages, thinking_prefill))
        return "teacher:"

    monkeypatch.setattr(opd_mod, "_teacher_prompt_text", _render_teacher_prompt)
    teacher = _Teacher()
    gen = opd_mod._GenResult(completion_ids=[2], completion_text="a", gen_tokens=1)
    pending = opd_mod._Pending(
        gen=gen,
        prompt_ids=[1],
        prompt_messages=[{"role": "user", "content": "p"}],
    )
    candidate_topk = objective_ids == ("c13",)
    retain_teacher_prompts = opd_mod._retain_candidate_teacher_prompt(
        [pending], cadence=100, candidate_topk=candidate_topk
    )
    score = opd_mod._score_many(
        teacher,
        [pending],
        thinking_prefill="",
        retain_teacher_prompts=retain_teacher_prompts,
    )[0]

    assert retain_teacher_prompts == (False,)
    assert len(rendered) == 1
    assert score.teacher_prompt == ""
    meter = TeacherInputMeter() if candidate_topk else None
    if meter is not None:
        meter.record_ordinary(2)
    results = opd_mod._resolve_samples_batched(
        _Model(),
        _Tok(),
        "cpu",
        [(gen, score, [1])],
        SimpleNamespace(
            kl_coef=1.0,
            eos_loss_coef=0.0,
            entropy_floor_coef=0.0,
            entropy_floor=0.0,
            stop_sequences=(),
            objective_ids=objective_ids,
            topk_cadence=100,
        ),
        microbatch=1,
        teacher=teacher if meter is not None else None,
        topk_meter=meter,
        topk_cache={} if meter is not None else None,
    )

    assert len(rendered) == 1
    assert len(teacher.calls) == 1
    assert results[0].loss is not None
    if meter is not None:
        assert meter.candidate_tokens == 0
        metrics = dict(results[0].objective_metrics)
        assert metrics["opd/objectives/c13/selected_prefixes"] == 0.0
        assert metrics["opd/objectives/c13/scored_prefixes"] == 0.0


def test_c13_prompt_retention_is_per_sample_for_mixed_multiturn_batch(monkeypatch):
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod
    from flash.engine.worker.tokenizer_align import TeacherToken

    class _Tok:
        eos_token_id = 5

        def decode(self, ids, skip_special_tokens=True):
            mapping = {1: "p", 2: "a", 3: "b", 5: ""}
            return "".join(mapping.get(int(token_id), "d") for token_id in ids)

    class _Teacher:
        def __init__(self):
            self.calls = []

        def score_many(self, items):
            self.calls.append(items)
            if len(self.calls) != 1:
                raise AssertionError("the non-firing sample must not trigger candidate scoring")
            return [[TeacherToken("a" * 101, -0.5, 0, 101)], []]

    rendered = []

    def _render_teacher_prompt(prompt_messages, thinking_prefill=""):
        name = prompt_messages[-1]["content"]
        rendered.append((name, thinking_prefill))
        return f"teacher:{name}:"

    monkeypatch.setattr(opd_mod, "_teacher_prompt_text", _render_teacher_prompt)
    long_gen = opd_mod._GenResult(
        completion_ids=[2] * 101, completion_text="a" * 101, gen_tokens=101
    )
    short_gen = opd_mod._GenResult(completion_ids=[3], completion_text="b", gen_tokens=1)
    pendings = [
        opd_mod._Pending(
            gen=long_gen,
            prompt_ids=[1],
            prompt_messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "prior"},
                {"role": "user", "content": "long"},
            ],
        ),
        opd_mod._Pending(
            gen=short_gen,
            prompt_ids=[1],
            prompt_messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "prior"},
                {"role": "user", "content": "short"},
            ],
        ),
    ]
    retain_teacher_prompts = opd_mod._retain_candidate_teacher_prompt(
        pendings, cadence=100, candidate_topk=True
    )
    scores = opd_mod._score_many(
        _teacher := _Teacher(),
        pendings,
        thinking_prefill="prefill",
        retain_teacher_prompts=retain_teacher_prompts,
    )

    assert retain_teacher_prompts == (True, False)
    assert rendered == [("long", "prefill"), ("short", "prefill")]
    assert scores[0].teacher_prompt == "teacher:long:"
    assert scores[1].teacher_prompt == ""
    assert len(scores[0].teacher_toks) == 1
    assert scores[1].teacher_toks == []
    assert _teacher.calls == [[("teacher:long:", "a" * 101), ("teacher:short:", "b")]]

    meter = TeacherInputMeter()
    meter.record_ordinary(2)
    outputs = opd_mod._candidate_objectives_for_chunk(
        torch.zeros((1, 2, 6), requires_grad=True),
        [
            SimpleNamespace(
                prompt_ids=[1],
                student_ids=[3],
                score_result=scores[1],
            )
        ],
        _Tok(),
        SimpleNamespace(topk_cadence=100),
        _teacher,
        meter,
        {},
    )

    assert len(_teacher.calls) == 1
    assert meter.candidate_tokens == 0
    assert outputs[0].selected_prefixes == 0
    assert outputs[0].scored_prefixes == 0


def test_c13_prompt_retention_stays_aligned_when_fallback_sample_fails(monkeypatch):
    from flash.engine.worker import opd as opd_mod

    class _Teacher:
        def __init__(self):
            self.calls = []

        def score(self, context, surface):
            self.calls.append((context, surface))
            if surface == "short":
                raise RuntimeError("sample failed")
            return [SimpleNamespace(logprob=-0.5)]

    monkeypatch.setattr(
        opd_mod,
        "_teacher_prompt_text",
        lambda prompt_messages, _thinking_prefill="": f"teacher:{prompt_messages[-1]['content']}:",
    )
    pendings = [
        opd_mod._Pending(
            gen=opd_mod._GenResult(completion_ids=[2], completion_text="short", gen_tokens=1),
            prompt_ids=[1],
            prompt_messages=[{"role": "user", "content": "short"}],
        ),
        opd_mod._Pending(
            gen=opd_mod._GenResult(
                completion_ids=[2] * 101, completion_text="long", gen_tokens=101
            ),
            prompt_ids=[1],
            prompt_messages=[{"role": "user", "content": "long"}],
        ),
    ]
    retain_teacher_prompts = opd_mod._retain_candidate_teacher_prompt(
        pendings, cadence=100, candidate_topk=True
    )
    teacher = _Teacher()
    scores = opd_mod._score_many(
        teacher,
        pendings,
        thinking_prefill="",
        max_attempts=1,
        retain_teacher_prompts=retain_teacher_prompts,
    )

    assert retain_teacher_prompts == (False, True)
    assert [score.status for score in scores] == ["error", "ok"]
    assert [score.teacher_prompt for score in scores] == ["", "teacher:long:"]
    assert teacher.calls == [("teacher:short:", "short"), ("teacher:long:", "long")]


def test_c13_integrates_candidate_teacher_scores_into_loss_gradient(monkeypatch):
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod
    from flash.engine.worker.tokenizer_align import TeacherToken

    class _Tok:
        pad_token_id = 0
        eos_token_id = 5

        def decode(self, ids, skip_special_tokens=True):
            mapping = {1: "p", 2: "a", 3: "b", 4: "c", 5: ""}
            return "".join(mapping.get(int(token_id), "d") for token_id in ids)

    class _Model:
        def __init__(self):
            self.weight = torch.tensor(
                [
                    [0.0, 0.0, 2.0, 1.0, 0.5, 9.0],
                    [0.0, 0.0, 2.0, 1.0, 0.5, 9.0],
                    [0.0] * 6,
                ],
                requires_grad=True,
            )
            self.config = SimpleNamespace(use_cache=True)

        def __call__(self, input_ids, **_kwargs):
            batch, length = input_ids.shape
            return SimpleNamespace(logits=self.weight[:length].unsqueeze(0).expand(batch, -1, -1))

        def parameters(self):
            return [self.weight]

        def train(self, mode=True):
            return self

    class _Teacher:
        def __init__(self):
            self.calls = []

        def score_many(self, items):
            self.calls.append(items)
            return [
                [TeacherToken(surface, -0.1 * (index + 1), 0, len(surface))]
                for index, (_context, surface) in enumerate(items)
            ]

    rendered = []

    def _render_teacher_prompt(prompt_messages, thinking_prefill=""):
        rendered.append((prompt_messages, thinking_prefill))
        return "teacher:"

    monkeypatch.setattr(opd_mod, "_teacher_prompt_text", _render_teacher_prompt)
    model = _Model()
    teacher = _Teacher()
    gen = opd_mod._GenResult(completion_ids=[2, 3], completion_text="ab", gen_tokens=2)
    pending = opd_mod._Pending(
        gen=gen,
        prompt_ids=[1],
        prompt_messages=[{"role": "user", "content": "p"}],
    )
    retain_teacher_prompts = opd_mod._retain_candidate_teacher_prompt(
        [pending], cadence=1, candidate_topk=True
    )
    score = opd_mod._score_many(
        teacher,
        [pending],
        thinking_prefill="",
        retain_teacher_prompts=retain_teacher_prompts,
    )[0]
    assert retain_teacher_prompts == (True,)
    assert len(rendered) == 1
    assert score.teacher_prompt == "teacher:"
    meter = TeacherInputMeter()
    meter.record_ordinary(20)
    knobs = SimpleNamespace(
        kl_coef=1.0,
        eos_loss_coef=0.0,
        entropy_floor_coef=0.0,
        entropy_floor=0.0,
        stop_sequences=(),
        objective_ids=("c13",),
        topk_cadence=1,
    )
    samples = [(gen, score, [1])]

    result = opd_mod._resolve_samples_batched(
        model,
        _Tok(),
        "cpu",
        samples,
        knobs,
        microbatch=1,
        teacher=teacher,
        topk_meter=meter,
        topk_cache={},
    )[0]
    result.loss.backward()

    assert len(rendered) == 1
    assert len(teacher.calls) == 2
    assert len(teacher.calls[1]) >= 2
    assert model.weight.grad is not None
    assert float(model.weight.grad.abs().sum()) > 0
    metrics = dict(result.objective_metrics)
    assert metrics["opd/objectives/c13/selected_prefixes"] == 1.0
    assert metrics["opd/objectives/c13/scored_prefixes"] == 1.0
    assert metrics["opd/objectives/c13/budget_exhausted"] == 0.0


def test_c13_requires_explicit_typed_cadence_and_round_trips():
    from flash.schema import spec_from_dict
    from flash.schema.fields import ConfigError
    from flash.spec import JobSpec

    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "opd",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "train": {"opd_objective_ids": ["c13"], "opd_topk_cadence": 4},
    }
    spec = spec_from_dict(raw, run_id="c13")
    assert spec.train.opd_topk_cadence == 4
    assert JobSpec.from_json(spec.to_json()).train.opd_topk_cadence == 4

    missing = {**raw, "train": {"opd_objective_ids": ["c13"]}}
    with pytest.raises(ConfigError, match="required"):
        spec_from_dict(missing, run_id="missing")
    with pytest.raises(ConfigError, match=r"positive integer|must be an integer"):
        spec_from_dict(
            {**raw, "train": {"opd_objective_ids": ["c13"], "opd_topk_cadence": True}},
            run_id="bool",
        )
    with pytest.raises(ConfigError, match="only valid"):
        spec_from_dict(
            {**raw, "train": {"opd_objective_ids": ["c0"], "opd_topk_cadence": 4}},
            run_id="c0",
        )

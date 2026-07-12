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

    first = scorer.score(requests)
    second = scorer.score(requests)

    assert calls == [[("ctx", "a"), ("ctx", "b")]]
    assert first.requests == 2
    assert first.cache_hits == 0
    assert second.requests == 0
    assert second.cache_hits == 2
    assert meter.candidate_tokens == 22


def test_candidate_teacher_scoring_rejects_invalid_responses():
    class _Teacher:
        def score_many(self, _items):
            return [[]]

    meter = TeacherInputMeter()
    meter.record_ordinary(100)
    scorer = CandidateTeacherScorer(_Teacher(), meter)

    with pytest.raises(TeacherError, match="no realized completion tokens") as exc:
        scorer.score([CandidateRequest("ctx", "a", 10)])
    assert exc.value.permanent


def test_teacher_input_meter_refuses_request_before_crossing_hard_cap():
    meter = TeacherInputMeter()
    meter.record_ordinary(10)

    assert meter.reserve_candidate(10)
    assert meter.total_tokens == meter.cap_tokens == 20
    assert not meter.reserve_candidate(1)
    assert meter.total_tokens == 20
    assert meter.budget_exhausted


def test_selection_cadence_is_deterministic_and_positive():
    assert selected_prefix_indices(10, 4) == (0, 4, 8)
    assert selected_prefix_indices(0, 4) == ()
    with pytest.raises(ValueError, match="positive"):
        selected_prefix_indices(10, 0)


def test_c13_integrates_candidate_teacher_scores_into_loss_gradient():
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
            return [[TeacherToken(surface, -0.1 * (index + 1), 0, len(surface))] for index, (_context, surface) in enumerate(items)]

    model = _Model()
    teacher = _Teacher()
    meter = TeacherInputMeter()
    meter.record_ordinary(10)
    knobs = SimpleNamespace(
        kl_coef=1.0,
        eos_loss_coef=0.0,
        entropy_floor_coef=0.0,
        entropy_floor=0.0,
        stop_sequences=(),
        objective_ids=("c13",),
        topk_cadence=1,
    )
    samples = [
        (
            opd_mod._GenResult(completion_ids=[2], completion_text="a", gen_tokens=1),
            opd_mod._ScoreResult(
                teacher_toks=[TeacherToken("a", -0.5, 0, 1)], status="ok"
            ),
            [1],
        )
    ]

    result = opd_mod._resolve_samples_batched(
        model,
        _Tok(),
        "cpu",
        samples,
        knobs,
        microbatch=1,
        teacher=teacher,
        candidate_teacher_prompts=["teacher:"],
        topk_meter=meter,
        topk_cache={},
    )[0]
    result.loss.backward()

    assert len(teacher.calls) == 1
    assert len(teacher.calls[0]) >= 2
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

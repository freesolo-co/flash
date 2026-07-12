from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.engine.worker.opd_listwise import (
    ContinuationRecord,
    ExactTeacherScoreCache,
    TeacherInputBudget,
    TeacherScoreCacheKey,
    build_continuation_record,
    detached_target_distribution,
    listwise_sequence_score_loss,
    pubmedqa_correctness,
    robust_within_prompt_normalize,
    select_teacher_candidates,
)
from flash.engine.worker.tokenizer_align import TeacherToken


def _record(
    index: int,
    *,
    prompt_key: str = "prompt",
    text: str | None = None,
    teacher_logprob: float | None = -1.0,
    correctness: float | None = None,
    boxed: float = 0.0,
    eos: float = 1.0,
    no_repetition: float = 1.0,
    no_length: float = 1.0,
    status: str = "ok",
):
    return ContinuationRecord(
        prompt_key=prompt_key,
        candidate_index=index,
        prompt_ids=(1, 2),
        completion_ids=(3,),
        completion_text=text if text is not None else f"candidate-{index}",
        teacher_logprob=teacher_logprob,
        correctness=correctness,
        boxed_format=boxed,
        terminal_eos=eos,
        no_repetition=no_repetition,
        no_length_termination=no_length,
        status=status,
    )


def test_continuation_records_preserve_prompt_grouping():
    records = (_record(0, prompt_key="a"), _record(1, prompt_key="a"), _record(0, prompt_key="b"))
    groups = {}
    for record in records:
        groups.setdefault(record.prompt_key, []).append(record)

    assert [record.candidate_index for record in groups["a"]] == [0, 1]
    assert [record.candidate_index for record in groups["b"]] == [0]


def test_deterministic_selection_prefers_unique_then_keeps_duplicates():
    duplicate_late = _record(2, text="same")
    records = (
        duplicate_late,
        _record(1, text="unique"),
        _record(0, text="same"),
    )

    selected = select_teacher_candidates(records, 2)

    assert [record.candidate_index for record in selected] == [0, 1]
    assert select_teacher_candidates(records, 2) == selected


def test_failed_candidates_are_not_selected():
    selected = select_teacher_candidates(
        (
            _record(0, status="error", teacher_logprob=None),
            _record(1),
            _record(2),
        ),
        2,
    )

    assert [record.candidate_index for record in selected] == [1, 2]


def test_local_utility_composes_available_correctness_and_format_features():
    record = _record(
        0,
        correctness=1.0,
        boxed=1.0,
        eos=1.0,
        no_repetition=0.0,
        no_length=1.0,
    )
    without_correctness = _record(1, correctness=None, boxed=1.0)

    assert record.local_utility == 4.0
    assert without_correctness.local_utility == 4.0


def test_pubmedqa_correctness_when_label_is_available():
    assert pubmedqa_correctness({"final_answer": "yes"}, "therefore \\boxed{yes}") == 1.0
    assert pubmedqa_correctness({"label": "no"}, "maybe") == 0.0
    assert pubmedqa_correctness({"question": "unknown"}, "yes") is None


def test_record_features_cover_box_eos_repetition_and_length():
    token = TeacherToken("yes", -0.25, 0, 3)
    record = build_continuation_record(
        prompt_key="p",
        candidate_index=0,
        prompt_ids=(1,),
        completion_ids=(2,),
        completion_text="\\boxed{yes}",
        teacher_tokens=(token,),
        example={"answer": "yes"},
        terminal_eos=True,
        length_terminated=False,
    )
    repeated = build_continuation_record(
        prompt_key="p",
        candidate_index=1,
        prompt_ids=(1,),
        completion_ids=(2,),
        completion_text="yes yes yes yes yes yes",
        teacher_tokens=(token,),
        terminal_eos=False,
        length_terminated=True,
    )

    assert record.teacher_logprob == -0.25
    assert record.local_utility == 5.0
    assert repeated.no_repetition == 0.0
    assert repeated.terminal_eos == 0.0
    assert repeated.no_length_termination == 0.0


def test_robust_normalization_ties_are_zero_and_target_is_uniform():
    torch = pytest.importorskip("torch")
    records = (_record(0), _record(1))

    assert robust_within_prompt_normalize((3.0, 3.0)) == (0.0, 0.0)
    target = detached_target_distribution(records)

    assert not target.requires_grad
    assert torch.allclose(target, torch.tensor([0.5, 0.5]))


def test_true_listwise_loss_has_cross_candidate_gradient():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([-2.0, -1.0], requires_grad=True)
    target = torch.tensor([0.8, 0.2], requires_grad=True)

    loss = listwise_sequence_score_loss(scores, target)
    loss.backward()

    expected = torch.softmax(scores.detach(), dim=0) - target.detach()
    assert torch.allclose(scores.grad, expected)
    assert target.grad is None
    assert scores.grad[0] == pytest.approx(-scores.grad[1])


def test_c14_resolver_groups_two_continuations_and_backpropagates_listwise_term():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "".join({2: "yes", 3: "no"}.get(int(token_id), "") for token_id in ids)

    class _Model:
        def __init__(self):
            self.weight = torch.zeros(2, 8, requires_grad=True)
            self.config = SimpleNamespace(use_cache=True)

        def __call__(self, input_ids, **_kwargs):
            batch, length = input_ids.shape
            return SimpleNamespace(logits=self.weight[:length].unsqueeze(0).expand(batch, -1, -1))

        def parameters(self):
            return [self.weight]

        def train(self, mode=True):
            return self

    model = _Model()
    samples = [
        opd_mod._ScoredContinuation(
            gen=opd_mod._GenResult(
                completion_ids=[2],
                completion_text="\\boxed{yes}",
                gen_tokens=1,
                terminal_eos=True,
            ),
            score=opd_mod._ScoreResult(
                teacher_toks=[TeacherToken("\\boxed{yes}", -0.2, 0, 11)], status="ok"
            ),
            prompt_ids=[1],
            example={"answer": "yes"},
            prompt_key="prompt",
            candidate_index=0,
        ),
        opd_mod._ScoredContinuation(
            gen=opd_mod._GenResult(
                completion_ids=[3], completion_text="no", gen_tokens=1, terminal_eos=True
            ),
            score=opd_mod._ScoreResult(
                teacher_toks=[TeacherToken("no", -1.0, 0, 2)], status="ok"
            ),
            prompt_ids=[1],
            example={"answer": "yes"},
            prompt_key="prompt",
            candidate_index=1,
        ),
    ]
    knobs = SimpleNamespace(
        kl_coef=1.0,
        eos_loss_coef=0.0,
        entropy_floor_coef=0.0,
        entropy_floor=0.0,
        stop_sequences=(),
        objective_ids=("c14",),
    )

    resolved = opd_mod._resolve_samples_batched(
        model,
        _Tok(),
        "cpu",
        samples,
        knobs,
        microbatch=1,
        backward_scale=0.5,
    )

    metrics = dict(resolved[0].objective_metrics)
    assert metrics["opd/objectives/c14/candidate_count"] == 2.0
    assert "opd/objectives/c14/target_entropy" in metrics
    assert resolved[1].objective_metrics == ()
    assert model.weight.grad is not None
    assert bool(torch.isfinite(model.weight.grad).all())
    assert not torch.equal(model.weight.grad, torch.zeros_like(model.weight.grad))


def test_c14_resolver_skips_prompt_when_one_candidate_fails():
    from flash.engine.worker import opd as opd_mod

    successful = opd_mod._ScoredContinuation(
        gen=opd_mod._GenResult(completion_ids=[2], completion_text="yes", gen_tokens=1),
        score=opd_mod._ScoreResult(
            teacher_toks=[TeacherToken("yes", -0.2, 0, 3)], status="ok"
        ),
        prompt_ids=[1],
        prompt_key="prompt",
        candidate_index=0,
    )
    failed = opd_mod._ScoredContinuation(
        gen=opd_mod._GenResult(completion_ids=[3], completion_text="no", gen_tokens=1),
        score=opd_mod._ScoreResult(status="error", error="failed"),
        prompt_ids=[1],
        prompt_key="prompt",
        candidate_index=1,
    )
    knobs = SimpleNamespace(
        kl_coef=1.0,
        eos_loss_coef=0.0,
        entropy_floor_coef=0.0,
        entropy_floor=0.0,
        stop_sequences=(),
        objective_ids=("c14",),
    )

    resolved = opd_mod._resolve_samples_batched(
        object(), object(), "cpu", [successful, failed], knobs, microbatch=1
    )

    assert resolved[0].loss is None
    assert resolved[0].skip_reason == "listwise_group_incomplete"
    assert resolved[1].loss is None
    assert resolved[1].skip_reason == "teacher_error"


def test_teacher_input_cap_refuses_before_crossing_and_reports_exhausted():
    budget = TeacherInputBudget(ordinary_run_tokens=5)

    assert budget.cap_tokens == 10
    assert budget.charge((6,))
    assert budget.used_tokens == 6
    assert not budget.charge((5,))
    assert budget.used_tokens == 6
    assert budget.budget_exhausted is True


def test_preflight_does_not_assume_cache_hits():
    budget = TeacherInputBudget(ordinary_run_tokens=2)
    cache = ExactTeacherScoreCache()
    key = TeacherScoreCacheKey("teacher", "prompt", "completion")
    cache.put(key, object())

    assert cache.get(key) is not None
    assert not budget.preflight((5,))
    assert budget.used_tokens == 0
    assert budget.budget_exhausted


def test_c14_exact_cache_deduplicates_identical_requests():
    from flash.engine.worker import opd as opd_mod

    class _Teacher:
        model = "teacher"

        def __init__(self):
            self.calls = []

        def score_many(self, prompts):
            self.calls.append(tuple(prompts))
            return [[TeacherToken("x", -0.5, 0, 1)] for _ in prompts]

    gen = opd_mod._GenResult(completion_ids=[3], completion_text="x", gen_tokens=1)
    pendings = [
        opd_mod._Pending(gen=gen, prompt_ids=[1, 2], prompt_messages=[]),
        opd_mod._Pending(gen=gen, prompt_ids=[1, 2], prompt_messages=[]),
    ]
    teacher = _Teacher()
    cache = ExactTeacherScoreCache()
    budget = TeacherInputBudget(ordinary_run_tokens=20)

    first = opd_mod._score_many_c14(
        teacher,
        pendings,
        thinking_prefill="",
        cache=cache,
        budget=budget,
    )
    second = opd_mod._score_many_c14(
        teacher,
        pendings,
        thinking_prefill="",
        cache=cache,
        budget=budget,
    )

    assert [result.status for result in first] == ["ok", "ok"]
    assert [result.input_tokens for result in first] == [3, 0]
    assert [result.status for result in second] == ["ok", "ok"]
    assert [result.input_tokens for result in second] == [0, 0]
    assert budget.used_tokens == 3
    assert len(teacher.calls) == 1
    assert len(teacher.calls[0]) == 1
    assert len(cache) == 1


def test_c14_registry_and_default_noop_contracts():
    from flash.engine.worker.opd_objectives import (
        OPD_OBJECTIVES,
        ObjectiveRequirements,
        ObjectiveView,
    )
    from flash.opd_objectives import normalize_opd_objective_ids

    c0 = OPD_OBJECTIVES.plan(("c0",))
    c14 = OPD_OBJECTIVES.plan(("c14",))

    assert normalize_opd_objective_ids(["c14"], algorithm="opd") == ("c14",)

    assert c0.requirements == ObjectiveRequirements()
    assert OPD_OBJECTIVES.evaluate(c0, ObjectiveView(), base_term=0.0).terms == ()
    assert c14.requirements == ObjectiveRequirements(
        student_logits=True,
        teacher_scores=True,
        continuation_records=True,
    )

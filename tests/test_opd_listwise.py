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
    has_repetition,
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


def _pubmed_example(answer: str):
    return {
        "pubid": "1",
        "question": "question",
        "context": {"contexts": ["study"]},
        "long_answer": "long answer",
        "final_decision": answer,
    }


def test_pubmedqa_correctness_is_schema_gated_and_uses_the_final_answer():
    assert pubmedqa_correctness(_pubmed_example("yes"), "therefore \\boxed{yes}") == 1.0
    assert pubmedqa_correctness(_pubmed_example("no"), "maybe") == 0.0
    assert pubmedqa_correctness(_pubmed_example("yes"), "No at first. Final answer: yes") == 1.0
    assert pubmedqa_correctness({"answer": "yes"}, "yes") is None
    assert pubmedqa_correctness({"label": "no", "question": "unrelated"}, "no") is None


def test_record_features_cover_box_eos_repetition_and_length():
    token = TeacherToken("yes", -0.25, 0, 3)
    record = build_continuation_record(
        prompt_key="p",
        candidate_index=0,
        prompt_ids=(1,),
        completion_ids=(2,),
        completion_text="\\boxed{yes}",
        teacher_tokens=(token,),
        example=_pubmed_example("yes"),
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


def test_repetition_detector_is_local_and_conservative_for_long_text():
    ordinary = " ".join(
        [
            "the study reports a clinically meaningful result",
            "the analysis compares the study result with prior evidence",
            "the conclusion explains why the result matters for care",
        ]
        * 12
    )
    assert not has_repetition(ordinary)
    assert has_repetition("the result is clear the result is clear")
    assert has_repetition("yes yes yes yes")


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
            self.max_batch = 0

        def __call__(self, input_ids, **_kwargs):
            batch, length = input_ids.shape
            self.max_batch = max(self.max_batch, batch)
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
            example=_pubmed_example("yes"),
            prompt_key="prompt",
            candidate_index=0,
        ),
        opd_mod._ScoredContinuation(
            gen=opd_mod._GenResult(
                completion_ids=[3], completion_text="no", gen_tokens=1, terminal_eos=True
            ),
            score=opd_mod._ScoreResult(teacher_toks=[TeacherToken("no", -1.0, 0, 2)], status="ok"),
            prompt_ids=[1],
            example=_pubmed_example("yes"),
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
    assert model.max_batch == 1
    assert model._flash_opd_c14_score_batches == 2
    assert model._flash_opd_full_logits_batches == 2


def test_c14_groups_interleaved_teacher_batches_by_prompt_identity():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return {2: "yes", 3: "no"}.get(int(ids[0]), "")

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

    def sample(prompt_key, candidate_index, token_id, logprob):
        text = {2: "yes", 3: "no"}[token_id]
        return opd_mod._ScoredContinuation(
            gen=opd_mod._GenResult(completion_ids=[token_id], completion_text=text, gen_tokens=1),
            score=opd_mod._ScoreResult(
                teacher_toks=[TeacherToken(text, logprob, 0, len(text))], status="ok"
            ),
            prompt_ids=[1],
            prompt_key=prompt_key,
            candidate_index=candidate_index,
        )

    # this ordering models two teacher futures that each hold half of both prompt pairs.
    samples = [
        sample("a", 0, 2, -0.2),
        sample("b", 0, 2, -0.3),
        sample("a", 1, 3, -1.0),
        sample("b", 1, 3, -1.1),
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
        _Model(), _Tok(), "cpu", samples, knobs, microbatch=2
    )

    metric_rows = [result for result in resolved if result.objective_metrics]
    assert len(metric_rows) == 2
    assert all(result.loss is not None for result in resolved)


def test_c14_incomplete_pair_keeps_successful_candidate_gkd():
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

    successful = opd_mod._ScoredContinuation(
        gen=opd_mod._GenResult(completion_ids=[2], completion_text="yes", gen_tokens=1),
        score=opd_mod._ScoreResult(teacher_toks=[TeacherToken("yes", -0.2, 0, 3)], status="ok"),
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
        _Model(), _Tok(), "cpu", [successful, failed], knobs, microbatch=1
    )

    assert resolved[0].loss is not None
    assert resolved[0].objective_metrics == ()
    assert resolved[1].loss is None
    assert resolved[1].skip_reason == "teacher_error"


def test_c14_ordinary_baseline_is_group_size_one_and_cap_is_two_equivalents():
    from flash.engine.worker import opd as opd_mod

    examples = [(None, None, [1, 2]), (None, None, [3])]
    ordinary = opd_mod._c14_planned_ordinary_input_tokens(
        examples, steps=2, prompts_per_step=2, max_completion=5
    )
    budget = TeacherInputBudget(ordinary)

    assert ordinary == (2 + 5) + (1 + 5) + (2 + 5) + (1 + 5)
    assert budget.cap_tokens == 2 * ordinary


def test_c14_budget_exhaustion_raises_immediately():
    from flash.engine.worker import opd as opd_mod

    with pytest.raises(RuntimeError, match="budget_exhausted"):
        opd_mod._raise_if_c14_budget_exhausted([opd_mod._ScoreResult(status="budget_exhausted")])


def test_c14_keeps_35b_loss_microbatch_serial(monkeypatch):
    from flash.engine import vram
    from flash.engine.worker import opd as opd_mod

    monkeypatch.setattr(vram, "resolve_params_b", lambda _model_id: 35.0)
    assert opd_mod._opd_loss_microbatch_size("large-model", 16) == 1


def test_terminal_eos_uses_preserved_vllm_cause_and_complete_eos_set():
    from flash.engine.worker import opd as opd_mod
    from flash.engine.worker.opd_vllm import OpdVllmOutput

    tok = SimpleNamespace(eos_token_id=2, decode=lambda ids, skip_special_tokens=True: "answer")
    knobs = SimpleNamespace(stop_sequences=())
    stripped = opd_mod._gen_from_vllm_output(
        OpdVllmOutput(token_ids=[8], text="answer", finish_reason="stop", stop_reason=None),
        tok,
        knobs,
        eos_ids={2, 7},
    )
    secondary = opd_mod._gen_from_vllm_output(
        OpdVllmOutput(token_ids=[8], text="answer", finish_reason="stop", stop_reason=7),
        tok,
        knobs,
        eos_ids={2, 7},
    )
    stopped_on_text = opd_mod._gen_from_vllm_output(
        OpdVllmOutput(token_ids=[8], text="answer", finish_reason="stop", stop_reason="<stop>"),
        tok,
        knobs,
        eos_ids={2, 7},
    )

    assert stripped.terminal_eos
    assert secondary.terminal_eos
    assert not stopped_on_text.terminal_eos


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


def test_exact_teacher_cache_is_bounded_lru_with_exact_keys():
    cache = ExactTeacherScoreCache(max_entries=2)
    first = TeacherScoreCacheKey("teacher", "prompt", "a")
    second = TeacherScoreCacheKey("teacher", "prompt", "b")
    third = TeacherScoreCacheKey("teacher", "prompt", "c")
    cache.put(first, "first")
    cache.put(second, "second")
    assert cache.get(first) == "first"
    cache.put(third, "third")

    assert cache.get(second) is None
    assert cache.get(first) == "first"
    assert cache.get(third) == "third"
    assert len(cache) == 2


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


def test_c14_conflicting_objectives_rejects_independent_policies():
    from flash.engine.worker import opd as opd_mod
    from flash.engine.worker.opd_objectives import OPD_OBJECTIVES

    # c07 and c12 own greedy-sidecar rollouts. c08 modifies the existing primary rollout and owns a
    # greedy sidecar. c13 owns an independent teacher-scoring and metering policy over top-k prefixes.
    assert opd_mod._c14_conflicting_objectives(OPD_OBJECTIVES.plan(("c12", "c14"))) == ("c12",)
    assert opd_mod._c14_conflicting_objectives(OPD_OBJECTIVES.plan(("c13", "c14"))) == ("c13",)
    assert opd_mod._c14_conflicting_objectives(OPD_OBJECTIVES.plan(("c07", "c14"))) == ("c07",)
    assert opd_mod._c14_conflicting_objectives(OPD_OBJECTIVES.plan(("c08", "c14"))) == ("c08",)
    # c14 on its own or with the c0 no-op stays compatible.
    assert opd_mod._c14_conflicting_objectives(OPD_OBJECTIVES.plan(("c14",))) == ()
    assert opd_mod._c14_conflicting_objectives(OPD_OBJECTIVES.plan(("c0", "c14"))) == ()


def test_c14_shared_conflict_id_set_matches_registry_requirements():
    from flash.engine.worker.opd_objectives import OPD_OBJECTIVES
    from flash.opd_objectives import OPD_C14_CONFLICTING_OBJECTIVE_IDS

    # the dependency-free conflict list used by schema/spec validators and the worker must match the
    # registry flags for independent sidecars, primary-rollout modification, or teacher scoring.
    registry = frozenset(
        definition.objective_id
        for definition in OPD_OBJECTIVES.definitions.values()
        if definition.requirements.greedy_sidecar
        or definition.requirements.sampled_primary
        or definition.requirements.candidate_topk
    )
    assert frozenset(OPD_C14_CONFLICTING_OBJECTIVE_IDS) == registry


def test_c14_conflict_message_lists_conflicts_and_does_not_claim_c14_must_be_alone():
    from flash.opd_objectives import opd_c14_conflict_message

    message = opd_c14_conflict_message(("c07", "c13"))
    assert "objective(s) that own an independent rollout or teacher-scoring policy" in message
    assert "c07" in message
    assert "c13" in message
    # it must not tell users c14 has to run alone; c14 composes with c0/c05/c06/c09/c10/c11.
    assert "on its own" not in message
    assert "may still be combined with" in message
    for compatible in ("c0", "c05", "c06", "c09", "c10", "c11"):
        assert compatible in message

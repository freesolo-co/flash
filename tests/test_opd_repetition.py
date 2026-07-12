from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.engine.worker.opd_repetition import (
    REPETITION_WEIGHT_MAX,
    REPETITION_WEIGHT_MIN,
    analyze_repetition,
    detect_repeated_ngram,
    detect_token_cycle,
    loop_closing_unlikelihood,
    normalize_repetition_weights,
)


def test_detects_short_token_cycle_and_marks_only_final_cycle():
    analysis = analyze_repetition([9, 8, 1, 2, 1, 2, 1, 2])

    assert detect_token_cycle([9, 8, 1, 2, 1, 2, 1, 2]).unit_length == 2
    assert analysis.has_loop
    assert analysis.closing_mask == (False, False, False, False, False, False, True, True)


def test_detects_repeated_nonconsecutive_ngram_suffix():
    ids = [4, 5, 6, 0, 4, 5, 6, 1, 4, 5, 6]
    match = detect_repeated_ngram(ids)
    analysis = analyze_repetition(ids)

    assert match is not None
    assert match.kind == "ngram"
    assert match.unit_length == 3
    assert match.repeats == 3
    assert match.occurrence_starts == (0, 4, 8)
    assert analysis.severity == pytest.approx(9 / 11)


def test_distant_common_trigrams_do_not_count_as_local_repetition():
    ids = list(range(210))
    for start in (10, 100, 207):
        ids[start : start + 3] = [901, 902, 903]

    assert detect_repeated_ngram(ids) is None
    assert not analyze_repetition(ids).has_loop


@pytest.mark.parametrize(
    "ids",
    [
        [],
        [1, 1, 1, 1, 1],
        [1, 2, 1, 2],
        [1, 2, 3, 1, 2, 3, 4],
        [1, 2, 3, 0, 1, 2, 3],
        [1, 2, 3, 4, 1, 2, 3, 4],
    ],
)
def test_conservative_safeguards_avoid_short_or_double_repeat_false_positives(ids):
    analysis = analyze_repetition(ids)

    assert not analysis.has_loop
    assert analysis.severity == 0.0


def test_unlikelihood_masks_forced_loop_closing_tokens():
    torch = pytest.importorskip("torch")
    rows = torch.zeros(8, 16, requires_grad=True)
    ids = [9, 8, 1, 2, 1, 2, 1, 2]

    term = loop_closing_unlikelihood(rows, ids, forced=(False,) * 7 + (True,))
    assert term is not None
    term.backward()

    nonzero_rows = rows.grad.abs().sum(dim=-1).nonzero().flatten().tolist()
    assert nonzero_rows == [6]


def test_forced_only_repetition_has_no_direct_or_weighting_signal():
    torch = pytest.importorskip("torch")
    ids = [9, 8, 3, 3, 3, 3, 3, 3]
    forced = (False, False, True, True, True, True, True, True)
    analysis = analyze_repetition(ids, forced=forced)
    rows = torch.zeros(len(ids), 16, requires_grad=True)

    assert not analysis.has_loop
    assert analysis.severity == 0.0
    assert normalize_repetition_weights([0.0, analysis.severity]) == pytest.approx((1.0, 1.0))
    assert loop_closing_unlikelihood(rows, ids, forced=forced, analysis=analysis) is None


def test_unlikelihood_is_noop_for_clean_output():
    torch = pytest.importorskip("torch")
    rows = torch.zeros(7, 16, requires_grad=True)

    assert loop_closing_unlikelihood(rows, [1, 2, 3, 4, 5, 6, 7]) is None
    assert rows.grad is None


def test_unlikelihood_has_finite_nonzero_gradients():
    torch = pytest.importorskip("torch")
    rows = torch.zeros(8, 16, requires_grad=True)
    ids = [9, 8, 1, 2, 1, 2, 1, 2]

    term = loop_closing_unlikelihood(rows, ids)
    term.backward()

    assert torch.isfinite(term)
    assert torch.isfinite(rows.grad).all()
    assert float(rows.grad.abs().sum()) > 0.0


def test_repetition_weights_are_bounded_normalized_and_ordered():
    weights = normalize_repetition_weights([0.0, 0.25, 1.0, 0.75])

    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert all(REPETITION_WEIGHT_MIN <= weight <= REPETITION_WEIGHT_MAX for weight in weights)
    assert weights[0] > weights[1] > weights[3] > weights[2]


def test_repetition_weights_empty_and_uniform_are_stable():
    assert normalize_repetition_weights([]) == ()
    assert normalize_repetition_weights([0.0, 0.0]) == pytest.approx((1.0, 1.0))


def test_greedy_sidecars_preserve_order_seed_and_truncated_tokens():
    from flash.engine.worker import opd as opd_mod

    calls = []

    class _Rollout:
        def generate(self, prompts, **kwargs):
            calls.append((prompts, kwargs))
            return [
                opd_mod.OpdVllmOutput([3, 3, 3, 3, 3, 3], "", finish_reason="length"),
                opd_mod.OpdVllmOutput([], "", finish_reason="stop"),
            ]

    prompts = [[1], [2]]
    sidecars = opd_mod._generate_greedy_sidecars(_Rollout(), prompts, max_tokens=6, seed=123)

    assert calls == [
        (
            prompts,
            {"max_tokens": 6, "temperature": 0.0, "seed": 123},
        )
    ]
    assert [sidecar.completion_ids for sidecar in sidecars] == [
        [3, 3, 3, 3, 3, 3],
        [],
    ]
    assert sidecars[0].truncated is True
    assert sidecars[0].repetition_analysis.has_loop
    assert sidecars[1].repetition_analysis.has_loop is False


def test_c08_primary_group_and_sidecars_are_accounted_separately():
    from flash.engine.worker import opd as opd_mod

    calls = []

    class _Rollout:
        def generate(self, prompts, **kwargs):
            calls.append((len(prompts), kwargs))
            return [
                opd_mod.OpdVllmOutput([index + 1], "x", finish_reason="stop")
                for index, _prompt in enumerate(prompts)
            ]

    rollout = _Rollout()
    knobs = SimpleNamespace(stop_sequences=())
    primary_prompts = [[1], [1], [2], [2]]
    sidecar_prompts = [[1], [2]]

    primary = opd_mod._generate_many_vllm(
        rollout,
        SimpleNamespace(decode=lambda ids, **_kwargs: "x"),
        primary_prompts,
        knobs,
        max_tokens=4,
        temperature=1.0,
        seed=50,
    )
    sidecars = opd_mod._generate_greedy_sidecars(
        rollout,
        sidecar_prompts,
        max_tokens=4,
        seed=60,
    )

    assert len(primary) == 4
    assert len(sidecars) == 2
    assert calls == [
        (4, {"max_tokens": 4, "temperature": 1.0, "seed": 50}),
        (2, {"max_tokens": 4, "temperature": 0.0, "seed": 60}),
    ]


def test_sidecar_unlikelihood_handles_empty_and_truncated_without_teacher():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Model:
        def __init__(self):
            self.weight = torch.zeros(16, 16, requires_grad=True)
            self.config = SimpleNamespace(use_cache=True)
            self.forward_calls = 0

        def __call__(self, input_ids, **_kwargs):
            self.forward_calls += 1
            batch, length = input_ids.shape
            return SimpleNamespace(logits=self.weight[:length].unsqueeze(0).expand(batch, -1, -1))

        def train(self):
            return self

    loop_ids = [3, 3, 3, 3, 3, 3]
    sidecars = [
        opd_mod._GenResult(
            completion_ids=loop_ids,
            truncated=True,
            repetition_analysis=analyze_repetition(loop_ids),
        ),
        opd_mod._GenResult(completion_ids=[], repetition_analysis=analyze_repetition([])),
    ]
    model = _Model()

    term, loops, closing = opd_mod._greedy_sidecar_unlikelihood(
        model,
        SimpleNamespace(pad_token_id=0),
        "cpu",
        [[1], [2]],
        sidecars,
        microbatch=2,
    )

    assert term is not None
    assert loops == 1
    assert closing == 1
    assert model.forward_calls == 1
    term.backward()
    assert model.weight.grad is not None


def test_sidecar_unlikelihood_scales_over_the_full_retained_batch():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    class _Model:
        def __init__(self):
            self.weight = torch.zeros(16, 16, requires_grad=True)
            self.config = SimpleNamespace(use_cache=True)

        def __call__(self, input_ids, **_kwargs):
            batch, length = input_ids.shape
            return SimpleNamespace(logits=self.weight[:length].unsqueeze(0).expand(batch, -1, -1))

        def train(self):
            return self

    loop_ids = [3, 3, 3, 3, 3, 3]
    loop = opd_mod._GenResult(
        completion_ids=loop_ids,
        repetition_analysis=analyze_repetition(loop_ids),
    )
    clean = opd_mod._GenResult(completion_ids=[], repetition_analysis=analyze_repetition([]))

    sparse_term, sparse_loops, _ = opd_mod._greedy_sidecar_unlikelihood(
        _Model(),
        SimpleNamespace(pad_token_id=0),
        "cpu",
        [[1]] * 32,
        [loop] + [clean] * 31,
        microbatch=32,
    )
    dense_term, dense_loops, _ = opd_mod._greedy_sidecar_unlikelihood(
        _Model(),
        SimpleNamespace(pad_token_id=0),
        "cpu",
        [[1]] * 32,
        [loop] * 32,
        microbatch=32,
    )

    assert sparse_loops == 1
    assert dense_loops == 32
    assert float(sparse_term.detach()) * 32 == pytest.approx(float(dense_term.detach()))


def test_c07_c09_plans_expose_reusable_rollout_requirements():
    from flash.engine.worker.opd_objectives import OPD_OBJECTIVES

    c07 = OPD_OBJECTIVES.plan(("c07",)).requirements
    c08 = OPD_OBJECTIVES.plan(("c08",)).requirements
    c09 = OPD_OBJECTIVES.plan(("c09",)).requirements

    assert c07.greedy_sidecar
    assert c08.greedy_sidecar
    assert c08.sampled_primary
    assert c09.student_logits
    assert c09.repetition_weighting


def _c09_test_runtime():
    torch = pytest.importorskip("torch")
    from flash.engine.worker.tokenizer_align import TeacherToken

    class _Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "".join("x" for _ in ids)

    class _Model:
        def __init__(self):
            self.weight = torch.zeros(32, 32, requires_grad=True)
            self.config = SimpleNamespace(use_cache=True)

        def __call__(self, input_ids, **_kwargs):
            batch, length = input_ids.shape
            return SimpleNamespace(logits=self.weight[:length].unsqueeze(0).expand(batch, -1, -1))

        def train(self):
            return self

    def _score(length):
        return TeacherToken("x" * length, -1.0, 0, length)

    knobs = SimpleNamespace(
        kl_coef=1.0,
        eos_loss_coef=0.0,
        entropy_floor_coef=0.0,
        entropy_floor=0.0,
        stop_sequences=(),
        objective_ids=("c09",),
    )
    return _Tok(), _Model, _score, knobs


def _objective_metric(result, name):
    return dict(result.objective_metrics)[f"opd/objectives/c09/{name}"]


def test_c09_normalizes_clean_and_repetitive_singleton_trajectories_together():
    from flash.engine.worker import opd as opd_mod

    tok, model_type, make_teacher_token, knobs = _c09_test_runtime()
    outputs = iter(
        (
            opd_mod.OpdVllmOutput([1, 2, 3, 4, 5, 6], "xxxxxx", finish_reason="stop"),
            opd_mod.OpdVllmOutput([3, 3, 3, 3, 3, 3], "xxxxxx", finish_reason="stop"),
        )
    )

    class _Rollout:
        def generate(self, prompts, **_kwargs):
            return [next(outputs) for _prompt in prompts]

    clean = opd_mod._generate_many_vllm(_Rollout(), tok, [[1]], knobs, max_tokens=6)[0]
    repetitive = opd_mod._generate_many_vllm(_Rollout(), tok, [[2]], knobs, max_tokens=6)[0]
    score = opd_mod._ScoreResult(teacher_toks=[make_teacher_token(6)], status="ok")

    model = model_type()
    results = opd_mod._resolve_samples_batched(
        model,
        tok,
        "cpu",
        [(clean, score, [1]), (repetitive, score, [2])],
        knobs,
        microbatch=1,
    )
    weights = [_objective_metric(result, "repetition_weight") for result in results]
    sum(result.loss for result in results).backward()

    assert model.weight.grad is not None
    assert weights == pytest.approx((1.25, 0.75))
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert _objective_metric(results[0], "repetition_severity") == 0.0
    assert _objective_metric(results[1], "repetition_severity") == 1.0


def test_c09_discarded_repetitive_sample_does_not_reweight_surviving_loss():
    from flash.engine.worker import opd as opd_mod

    tok, model_type, make_teacher_token, knobs = _c09_test_runtime()
    clean_ids = [1, 2, 3, 4, 5, 6]
    loop_ids = [3, 3, 3, 3, 3, 3]
    clean = opd_mod._GenResult(
        completion_ids=clean_ids,
        completion_text="xxxxxx",
        gen_tokens=6,
        repetition_analysis=analyze_repetition(clean_ids),
    )
    discarded = opd_mod._GenResult(
        completion_ids=loop_ids,
        completion_text="xxxxxx",
        gen_tokens=6,
        truncated=True,
        repetition_analysis=analyze_repetition(loop_ids),
    )
    score = opd_mod._ScoreResult(teacher_toks=[make_teacher_token(6)], status="ok")

    combined = opd_mod._resolve_samples_batched(
        model_type(),
        tok,
        "cpu",
        [(clean, score, [1]), (discarded, score, [2])],
        knobs,
        microbatch=2,
    )
    baseline = opd_mod._resolve_samples_batched(
        model_type(),
        tok,
        "cpu",
        [(clean, score, [1])],
        knobs,
        microbatch=1,
    )[0]

    assert combined[1].loss is None
    assert _objective_metric(combined[0], "repetition_weight") == 1.0
    assert float(combined[0].loss.detach()) == pytest.approx(float(baseline.loss.detach()))


def test_c09_forced_repetition_does_not_change_sequence_weights():
    from flash.engine.worker import opd as opd_mod

    tok, model_type, make_teacher_token, knobs = _c09_test_runtime()
    clean_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    forced_ids = [9, 8, 3, 3, 3, 3, 3, 3]
    forced = (False, False, True, True, True, True, True, True)
    samples = []
    for ids, forced_mask in ((clean_ids, ()), (forced_ids, forced)):
        gen = opd_mod._GenResult(
            completion_ids=ids,
            completion_text="x" * len(ids),
            gen_tokens=len(ids),
            forced=forced_mask,
            repetition_analysis=analyze_repetition(ids, forced=forced_mask),
        )
        score = opd_mod._ScoreResult(
            teacher_toks=[make_teacher_token(len(ids))],
            status="ok",
        )
        samples.append((gen, score, [1]))

    results = opd_mod._resolve_samples_batched(
        model_type(), tok, "cpu", samples, knobs, microbatch=2
    )

    assert [_objective_metric(result, "repetition_weight") for result in results] == pytest.approx(
        (1.0, 1.0)
    )
    assert _objective_metric(results[1], "repetition_severity") == 0.0
    assert _objective_metric(results[1], "loop_closing_tokens") == 0.0


def test_default_path_does_not_generate_sidecars_or_change_primary_kwargs(monkeypatch):
    from flash.engine.worker import opd as opd_mod

    calls = []

    class _Rollout:
        def generate(self, prompts, **kwargs):
            calls.append((prompts, kwargs))
            return [opd_mod.OpdVllmOutput([3], "x", finish_reason="stop")]

    knobs = SimpleNamespace(stop_sequences=())
    generated = opd_mod._generate_many_vllm(
        _Rollout(), SimpleNamespace(decode=lambda ids, **_kwargs: "x"), [[1]], knobs, max_tokens=4
    )

    assert len(generated) == 1
    assert calls == [([[1]], {"max_tokens": 4})]

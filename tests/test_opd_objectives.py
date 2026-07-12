from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from flash.engine.worker.opd_objectives import (
    OPD_OBJECTIVES,
    CvarEntropyConfig,
    ObjectiveConfig,
    ObjectiveDefinition,
    ObjectiveRegistry,
    ObjectiveRequirements,
    ObjectiveResult,
    ObjectiveView,
    PositionStatistics,
)


def _definition(objective_id, evaluate, requirements=None):
    return ObjectiveDefinition(
        objective_id=objective_id,
        requirements=requirements or ObjectiveRequirements(student_logits=True),
        evaluate=evaluate,
    )


def _planned_registry(evaluate, requirements=None):
    registry = ObjectiveRegistry((_definition("candidate", evaluate, requirements),))
    return registry, registry.plan(("candidate",))


def test_objective_contracts_are_immutable():
    requirements = ObjectiveRequirements(student_logits=True)
    view = ObjectiveView({"completion_logits": object()})
    result = ObjectiveResult(metrics={"value": 1.0})

    with pytest.raises(FrozenInstanceError):
        requirements.student_logits = False
    with pytest.raises(TypeError):
        view.values["other"] = object()
    with pytest.raises(TypeError):
        result.metrics["other"] = 2.0


def test_registry_rejects_duplicate_definitions_and_selections():
    definition = _definition("candidate", lambda _view: ObjectiveResult())
    with pytest.raises(ValueError, match="duplicate opd objective id"):
        ObjectiveRegistry((definition, definition))

    registry = ObjectiveRegistry((definition,))
    with pytest.raises(ValueError, match="duplicate opd objective id"):
        registry.resolve(("candidate", "candidate"))


def test_registry_rejects_unknown_objectives():
    with pytest.raises(ValueError, match="unknown opd objective id"):
        OPD_OBJECTIVES.resolve(("missing",))


def test_registry_plan_aggregates_supported_requirements():
    registry = ObjectiveRegistry(
        (
            _definition(
                "student",
                lambda _view: ObjectiveResult(),
                ObjectiveRequirements(student_logits=True),
            ),
            _definition(
                "teacher",
                lambda _view: ObjectiveResult(),
                ObjectiveRequirements(teacher_scores=True),
            ),
        )
    )
    plan = registry.plan(("student", "teacher"))

    assert plan.objective_ids == ("student", "teacher")
    assert plan.requirements == ObjectiveRequirements(student_logits=True, teacher_scores=True)
    assert not hasattr(plan.requirements, "extra_forwards")
    assert not hasattr(plan.requirements, "network_calls")


def test_registry_enforces_declared_view_requirements():
    registry, plan = _planned_registry(lambda _view: ObjectiveResult())
    with pytest.raises(RuntimeError, match="missing required value 'completion_logits'"):
        registry.evaluate(plan, ObjectiveView(), base_term=0.0)


def test_registry_namespaces_and_detaches_metrics():
    torch = pytest.importorskip("torch")
    metric = torch.tensor(2.5, requires_grad=True)
    registry, plan = _planned_registry(lambda _view: ObjectiveResult(metrics={"score": metric}))

    evaluated = registry.evaluate(
        plan,
        ObjectiveView({"completion_logits": torch.zeros(1, 2)}),
        base_term=torch.tensor(0.0),
    )

    assert evaluated.terms == ()
    assert evaluated.metrics == {"opd/objectives/candidate/score": 2.5}
    assert isinstance(evaluated.metrics["opd/objectives/candidate/score"], float)


@pytest.mark.parametrize("metric", ["2.5", True])
def test_registry_rejects_string_and_bool_metrics(metric):
    registry, plan = _planned_registry(lambda _view: ObjectiveResult(metrics={"score": metric}))

    with pytest.raises(RuntimeError, match="real numeric scalar"):
        registry.evaluate(
            plan,
            ObjectiveView({"completion_logits": object()}),
            base_term=0.0,
        )


@pytest.mark.parametrize(
    "term",
    ["1.0", True, 1 + 2j, float("nan"), float("inf")],
)
def test_registry_rejects_invalid_python_terms(term):
    torch = pytest.importorskip("torch")
    registry, plan = _planned_registry(lambda _view: ObjectiveResult(term=term))

    with pytest.raises(RuntimeError, match="term"):
        registry.evaluate(
            plan,
            ObjectiveView({"completion_logits": torch.zeros(1, 2)}),
            base_term=torch.tensor(0.0),
        )


def test_registry_rejects_invalid_tensor_terms():
    torch = pytest.importorskip("torch")
    invalid = (
        torch.tensor([1.0, 2.0]),
        torch.tensor(float("nan")),
        torch.tensor(1 + 2j),
        torch.tensor(True),
    )
    for term in invalid:
        registry, plan = _planned_registry(lambda _view, value=term: ObjectiveResult(term=value))
        with pytest.raises(RuntimeError, match="term"):
            registry.evaluate(
                plan,
                ObjectiveView({"completion_logits": torch.zeros(1, 2)}),
                base_term=torch.tensor(0.0),
            )


def test_registry_rejects_tensor_dtype_mismatch():
    torch = pytest.importorskip("torch")
    registry, plan = _planned_registry(
        lambda _view: ObjectiveResult(term=torch.tensor(1.0, dtype=torch.float64))
    )
    with pytest.raises(RuntimeError, match="dtype"):
        registry.evaluate(
            plan,
            ObjectiveView({"completion_logits": torch.zeros(1, 2)}),
            base_term=torch.tensor(0.0, dtype=torch.float32),
        )


def test_registry_rejects_tensor_device_mismatch_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("cuda is required for a real device mismatch")
    registry, plan = _planned_registry(
        lambda _view: ObjectiveResult(term=torch.tensor(1.0, device="cuda"))
    )
    with pytest.raises(RuntimeError, match="device"):
        registry.evaluate(
            plan,
            ObjectiveView({"completion_logits": torch.zeros(1, 2)}),
            base_term=torch.tensor(0.0, device="cpu"),
        )


def test_registry_accepts_and_normalizes_valid_real_terms():
    torch = pytest.importorskip("torch")
    base = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    scalar_registry, scalar_plan = _planned_registry(lambda _view: ObjectiveResult(term=1.25))
    scalar = scalar_registry.evaluate(
        scalar_plan,
        ObjectiveView({"completion_logits": torch.zeros(1, 2)}),
        base_term=base,
    ).terms[0]
    assert torch.is_tensor(scalar)
    assert scalar.shape == ()
    assert scalar.dtype == base.dtype
    assert scalar.device == base.device
    assert float(scalar) == 1.25

    tensor_term = torch.tensor([2.5], dtype=base.dtype, requires_grad=True)
    tensor_registry, tensor_plan = _planned_registry(
        lambda _view: ObjectiveResult(term=tensor_term)
    )
    normalized = tensor_registry.evaluate(
        tensor_plan,
        ObjectiveView({"completion_logits": torch.zeros(1, 2)}),
        base_term=base,
    ).terms[0]
    assert normalized.shape == ()
    assert normalized.data_ptr() == tensor_term.data_ptr()
    assert normalized.requires_grad


def test_worker_registry_exactly_matches_shared_objective_ids():
    from flash.opd_objectives import OPD_OBJECTIVE_IDS

    assert OPD_OBJECTIVES.objective_ids == OPD_OBJECTIVE_IDS


def test_c0_requires_no_extra_work_and_returns_no_additions():
    definition = OPD_OBJECTIVES.definitions["c0"]
    plan = OPD_OBJECTIVES.plan(("c0",))
    assert definition.requirements == ObjectiveRequirements()
    assert plan.requirements == ObjectiveRequirements()
    assert OPD_OBJECTIVES.evaluate(plan, ObjectiveView(), base_term=0.0).terms == ()
    assert OPD_OBJECTIVES.evaluate(plan, ObjectiveView(), base_term=0.0).metrics == {}


def _evaluate_c10(rows, *, fraction=0.5, floor=1.0, coef=1.0, forced=(), base_term=None):
    from flash.engine.worker import opd as opd_mod

    if base_term is None:
        base_term = rows.sum() * 0.0
    plan = OPD_OBJECTIVES.plan(("c10",))
    stats = opd_mod._completion_position_statistics(rows, forced)
    return OPD_OBJECTIVES.evaluate(
        plan,
        ObjectiveView(
            {
                "completion_logits": rows,
                "position_statistics": stats,
                "entropy_for_rows": lambda indices: opd_mod._chunked_row_entropies(
                    rows, indices, grad=True
                ),
                "cvar_entropy_config": CvarEntropyConfig(fraction=fraction, floor=floor, coef=coef),
                "base_term": base_term,
            }
        ),
        base_term=base_term,
    )


def test_c10_quantile_ties_include_every_row_at_threshold():
    torch = pytest.importorskip("torch")
    selected = []
    rows = torch.zeros(4, 3, requires_grad=True)
    stats = PositionStatistics(entropies=(0.1, 0.2, 0.2, 0.9), eligible_indices=(0, 1, 2, 3))
    plan = OPD_OBJECTIVES.plan(("c10",))
    evaluated = OPD_OBJECTIVES.evaluate(
        plan,
        ObjectiveView(
            {
                "completion_logits": rows,
                "position_statistics": stats,
                "entropy_for_rows": lambda indices: (
                    selected.extend(indices) or rows.sum() * 0.0 + 0.2
                ),
                "cvar_entropy_config": CvarEntropyConfig(fraction=0.5, floor=1.0, coef=1.0),
                "base_term": rows.sum(),
            }
        ),
        base_term=rows.sum(),
    )

    assert selected == [0, 1, 2]
    assert evaluated.metrics["opd/objectives/c10/threshold"] == pytest.approx(0.2)
    assert evaluated.metrics["opd/objectives/c10/selected_count"] == 3.0


def test_c10_short_sequence_selects_one_row():
    torch = pytest.importorskip("torch")
    rows = torch.tensor([[3.0, 0.0, 0.0]], requires_grad=True)

    evaluated = _evaluate_c10(rows, fraction=0.01, floor=1.0)

    assert evaluated.metrics["opd/objectives/c10/selected_count"] == 1.0
    assert len(evaluated.terms) == 1


def test_c10_fraction_one_matches_mean_entropy_floor():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    rows = torch.tensor([[4.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=True)
    expected, _ = opd_mod._entropy_floor_term(rows, coef=1.7, floor=1.2)
    evaluated = _evaluate_c10(rows, fraction=1.0, floor=1.2, coef=1.7)

    assert len(evaluated.terms) == 1
    assert torch.allclose(evaluated.terms[0], expected)


def test_c10_zero_selection_is_finite_and_inactive():
    torch = pytest.importorskip("torch")
    rows = torch.zeros(2, 4, requires_grad=True)

    evaluated = _evaluate_c10(rows, forced=(True, True), floor=10.0)

    assert evaluated.terms == ()
    assert evaluated.metrics["opd/objectives/c10/selected_count"] == 0.0
    assert evaluated.metrics["opd/objectives/c10/activation"] == 0.0
    assert all(math.isfinite(value) for value in evaluated.metrics.values())


def test_c10_excludes_grammar_forced_rows():
    torch = pytest.importorskip("torch")
    rows = torch.tensor([[8.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)

    evaluated = _evaluate_c10(rows, fraction=0.5, floor=2.0, forced=(True, False))

    assert evaluated.metrics["opd/objectives/c10/selected_count"] == 1.0
    assert evaluated.metrics["opd/objectives/c10/threshold"] == pytest.approx(math.log(3), rel=1e-5)
    assert evaluated.terms[0] is not None


def test_c10_gradient_step_raises_selected_tail_entropy():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    rows = torch.tensor([[5.0, 0.0, 0.0, 0.0]], requires_grad=True)
    before = opd_mod._chunked_row_entropies(rows, grad=False)[0]
    term = _evaluate_c10(rows, fraction=1.0, floor=1.0, coef=2.0).terms[0]
    term.backward()
    with torch.no_grad():
        rows -= 0.1 * rows.grad
    after = opd_mod._chunked_row_entropies(rows, grad=False)[0]

    assert after > before


def test_c10_reports_gradient_ratio_when_base_gradient_exists():
    torch = pytest.importorskip("torch")
    rows = torch.tensor([[4.0, 0.0, 0.0]], requires_grad=True)
    base_term = rows.square().mean()

    evaluated = _evaluate_c10(rows, fraction=1.0, floor=1.0, base_term=base_term)

    assert evaluated.metrics["opd/objectives/c10/gradient_ratio"] > 0.0


def test_c10_chunk_consistency(monkeypatch):
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    rows = torch.randn(7, 11, generator=torch.Generator().manual_seed(7), requires_grad=True)
    monkeypatch.setattr(opd_mod, "_ENTROPY_CHUNK_ROWS", 1)
    one = _evaluate_c10(rows, fraction=0.4, floor=10.0)
    monkeypatch.setattr(opd_mod, "_ENTROPY_CHUNK_ROWS", 128)
    full = _evaluate_c10(rows, fraction=0.4, floor=10.0)

    assert one.metrics == pytest.approx(full.metrics)
    assert torch.allclose(one.terms[0], full.terms[0])


def test_c10_nonfinite_rows_are_excluded_from_selection():
    torch = pytest.importorskip("torch")
    rows = torch.tensor([[float("nan"), 0.0], [2.0, 0.0]], requires_grad=True)

    evaluated = _evaluate_c10(rows, fraction=1.0, floor=1.0)

    assert evaluated.metrics["opd/objectives/c10/selected_count"] == 1.0
    assert math.isfinite(evaluated.metrics["opd/objectives/c10/threshold"])
    assert all(math.isfinite(value) for value in evaluated.metrics.values())


def test_c05_is_fixed_default_off_composition_without_teacher_boundary_assumption():
    definition = OPD_OBJECTIVES.definitions["c05"]

    assert OPD_OBJECTIVES.plan(()).definitions == ()
    assert definition.requirements == ObjectiveRequirements(
        student_logits=True, empty_rollouts=True
    )
    assert definition.config == ObjectiveConfig(
        entropy_floor=1.75,
        entropy_floor_coef=1.0,
        position0_eos_coef=1.0,
        recover_empty_rollouts=True,
        eos_in_rkl=False,
    )
    assert definition.config.teacher_boundary_probability is None


def test_objective_config_rejects_eos_in_rkl_without_teacher_boundary_probability():
    with pytest.raises(ValueError, match="requires teacher_boundary_probability"):
        ObjectiveConfig(eos_in_rkl=True)


def test_c05_multiple_eos_safety_reports_set_probability_rank_and_margin():
    torch = pytest.importorskip("torch")
    logits = torch.zeros(2, 8, requires_grad=True)
    logits.data[0, 3] = 2.0
    logits.data[0, 5] = 1.0
    plan = OPD_OBJECTIVES.plan(("c05",))
    result = OPD_OBJECTIVES.evaluate(
        plan,
        ObjectiveView(
            {
                "sample_logits": logits,
                "completion_logits": logits[1:2],
                "prompt_len": 1,
                "student_ids": (2,),
                "gkd_groups": object(),
                "eos_ids": frozenset({3, 5}),
                "stop_sequences": (),
                "empty_rollout": False,
                "termination_cause": "eos",
                "entropy": 2.0,
                "entropy_floor_active": False,
            }
        ),
        base_term=logits.sum() * 0.0,
    )

    expected_probability = float(torch.softmax(logits[0].detach(), dim=-1)[[3, 5]].sum())
    assert result.metrics["opd/objectives/c05/position0_eos_probability"] == pytest.approx(
        expected_probability
    )
    assert result.metrics["opd/objectives/c05/position0_eos_rank"] == 1.0
    assert result.metrics["opd/objectives/c05/position0_eos_margin"] == 2.0
    assert len(result.terms) == 1
    result.terms[0].backward()
    assert logits.grad[0, 3] > 0
    assert logits.grad[0, 5] > 0


def test_c05_stop_sequences_disable_position0_eos_safety():
    torch = pytest.importorskip("torch")
    logits = torch.zeros(1, 8, requires_grad=True)
    result = OPD_OBJECTIVES.evaluate(
        OPD_OBJECTIVES.plan(("c05",)),
        ObjectiveView(
            {
                "sample_logits": logits,
                "completion_logits": logits[:0],
                "prompt_len": 1,
                "student_ids": (),
                "gkd_groups": None,
                "eos_ids": frozenset({3}),
                "stop_sequences": ("</answer>",),
                "empty_rollout": True,
                "termination_cause": "stop_sequence",
                "entropy": None,
                "entropy_floor_active": False,
            }
        ),
        base_term=logits.sum() * 0.0,
    )

    assert result.terms == ()
    assert result.metrics == {}


def test_job_spec_parsing_does_not_import_worker_package():
    script = """
import sys
from flash.spec import JobSpec
assert "flash.engine.worker" not in sys.modules
default = JobSpec.from_json(JobSpec().to_json())
assert default.train.opd_objective_ids == ()
assert "flash.engine.worker" not in sys.modules
c0 = JobSpec.from_dict({"algorithm": "opd", "train": {"opd_objective_ids": ["c0"]}})
restored = JobSpec.from_json(c0.to_json())
assert restored.train.opd_objective_ids == ("c0",)
assert "flash.engine.worker" not in sys.modules
"""
    env = dict(os.environ)
    env["SEED"] = "not-an-int"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("objective_ids", "expected"),
    [
        (["c0"], ("c0",)),
        (("c0",), ("c0",)),
        (["c05"], ("c05",)),
        (["future-objective"], ("future-objective",)),
        (["c0", "c0"], ("c0", "c0")),
    ],
)
def test_job_spec_persisted_reads_preserve_typed_objective_ids(objective_ids, expected):
    from flash.spec import JobSpec

    persisted = {
        "algorithm": "grpo",
        "train": {"opd_objective_ids": objective_ids},
    }

    assert JobSpec.from_dict(persisted).train.opd_objective_ids == expected
    json_persisted = {**persisted, "train": {"opd_objective_ids": list(objective_ids)}}
    assert JobSpec.from_json(json.dumps(json_persisted)).train.opd_objective_ids == expected


@pytest.mark.parametrize(
    "objective_ids",
    ["c0", {"c0": True}, ["c0", 1]],
)
def test_job_spec_persisted_reads_reject_malformed_objective_ids(objective_ids):
    from flash.spec import JobSpec

    persisted = {"algorithm": "opd", "train": {"opd_objective_ids": objective_ids}}

    with pytest.raises(ValueError, match="opd_objective_ids"):
        JobSpec.from_dict(persisted)
    with pytest.raises(ValueError, match="opd_objective_ids"):
        JobSpec.from_json(json.dumps(persisted))


def test_opd_objective_ids_are_typed_opd_only_and_round_trip():
    from flash.schema import spec_from_dict
    from flash.schema.fields import ConfigError
    from flash.spec import JobSpec, TrainSpec

    assert TrainSpec().opd_objective_ids == ()
    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "opd",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "train": {"opd_objective_ids": ["c0"]},
    }
    spec = spec_from_dict(raw, run_id="objective-contract")
    assert spec.train.opd_objective_ids == ("c0",)
    assert JobSpec.from_json(spec.to_json()).train.opd_objective_ids == ("c0",)

    with pytest.raises(ConfigError, match="only valid when algorithm"):
        spec_from_dict({**raw, "algorithm": "grpo"}, run_id="wrong-algorithm")
    with pytest.raises(ConfigError, match="duplicate opd objective id"):
        spec_from_dict(
            {
                **raw,
                "train": {"opd_objective_ids": ["c0", "c0"]},
            },
            run_id="duplicate-objective",
        )
    with pytest.raises(ConfigError, match="unknown opd objective id"):
        spec_from_dict(
            {
                **raw,
                "train": {"opd_objective_ids": ["missing"]},
            },
            run_id="unknown-objective",
        )
    with pytest.raises(ConfigError, match="must be a list of strings"):
        spec_from_dict(
            {
                **raw,
                "train": {"opd_objective_ids": "c0"},
            },
            run_id="untyped-objective",
        )


def test_c10_config_is_strict_and_round_trips():
    from flash.schema import spec_from_dict
    from flash.schema.fields import ConfigError
    from flash.spec import JobSpec, TrainSpec

    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "opd",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "train": {
            "opd_objective_ids": ["c10"],
            "opd_cvar_entropy_fraction": 0.25,
            "opd_cvar_entropy_floor": 0.7,
            "opd_cvar_entropy_coef": 1.5,
        },
    }
    spec = spec_from_dict(raw, run_id="c10-config")
    restored = JobSpec.from_json(spec.to_json())

    assert TrainSpec().opd_cvar_entropy_fraction is None
    assert restored.train.opd_cvar_entropy_fraction == 0.25
    assert restored.train.opd_cvar_entropy_floor == 0.7
    assert restored.train.opd_cvar_entropy_coef == 1.5
    for value in (0.0, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ConfigError, match="opd_cvar_entropy_fraction"):
            spec_from_dict(
                {
                    **raw,
                    "train": {**raw["train"], "opd_cvar_entropy_fraction": value},
                },
                run_id="bad-c10-config",
            )


def test_selected_objective_receives_logits_and_detached_teacher_scores(monkeypatch):
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod
    from flash.engine.worker.tokenizer_align import TeacherToken

    class _Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "".join({2: "a"}.get(int(token_id), "x") for token_id in ids)

    class _TinyLm:
        def __init__(self):
            self.weight = torch.zeros(2, 8, requires_grad=True)
            self.config = SimpleNamespace(use_cache=True)
            self.forward_calls = 0

        def __call__(self, input_ids):
            self.forward_calls += 1
            batch, length = input_ids.shape
            return SimpleNamespace(logits=self.weight[:length].unsqueeze(0).expand(batch, -1, -1))

        def parameters(self):
            return [self.weight]

        def train(self, mode=True):
            return self

    captured = {}

    def _evaluate(view):
        captured.update(view.values)
        rows = view.require("completion_logits")
        scores = view.require("teacher_scores")
        return ObjectiveResult(
            term=rows.sum() * 0.0,
            metrics={"teacher_logprob": scores[0].logprob},
        )

    registry = ObjectiveRegistry(
        (
            _definition(
                "candidate",
                _evaluate,
                ObjectiveRequirements(student_logits=True, teacher_scores=True),
            ),
        )
    )
    monkeypatch.setattr(opd_mod, "OPD_OBJECTIVES", registry)
    teacher_token = TeacherToken("a", -0.5, 0, 1)
    samples = [
        (
            opd_mod._GenResult(completion_ids=[2], completion_text="a", gen_tokens=1),
            opd_mod._ScoreResult(teacher_toks=[teacher_token], status="ok"),
            [1],
        )
    ]
    model = _TinyLm()
    knobs = SimpleNamespace(
        kl_coef=1.0,
        eos_loss_coef=0.0,
        entropy_floor_coef=0.0,
        entropy_floor=0.0,
        stop_sequences=(),
        objective_ids=("candidate",),
    )

    result = opd_mod._resolve_samples_batched(model, _Tok(), "cpu", samples, knobs, microbatch=1)[0]

    assert captured["teacher_scores"] == (teacher_token,)
    assert captured["completion_logits"].requires_grad
    assert result.objective_metrics == (("opd/objectives/candidate/teacher_logprob", -0.5),)
    assert model.forward_calls == 1


def test_c0_golden_parity_has_identical_loss_gradient_and_forward_count():
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod
    from flash.engine.worker.tokenizer_align import TeacherToken

    class _Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "".join({2: "a"}.get(int(token_id), "x") for token_id in ids)

    class _TinyLm:
        def __init__(self):
            self.weight = torch.zeros(2, 8, requires_grad=True)
            self.config = SimpleNamespace(use_cache=True)
            self.forward_calls = 0

        def __call__(self, input_ids):
            self.forward_calls += 1
            batch, length = input_ids.shape
            return SimpleNamespace(logits=self.weight[:length].unsqueeze(0).expand(batch, -1, -1))

        def parameters(self):
            return [self.weight]

        def train(self, mode=True):
            return self

    samples = [
        (
            opd_mod._GenResult(completion_ids=[2], completion_text="a", gen_tokens=1),
            opd_mod._ScoreResult(teacher_toks=[TeacherToken("a", -0.5, 0, 1)], status="ok"),
            [1],
        )
    ]

    def _run(objective_ids):
        model = _TinyLm()
        knobs = SimpleNamespace(
            kl_coef=1.0,
            eos_loss_coef=0.0,
            entropy_floor_coef=0.0,
            entropy_floor=0.0,
            stop_sequences=(),
            objective_ids=objective_ids,
        )
        result = opd_mod._resolve_samples_batched(
            model, _Tok(), "cpu", samples, knobs, microbatch=1
        )[0]
        result.loss.backward()
        return result, model

    default, default_model = _run(())
    c0, c0_model = _run(("c0",))

    assert torch.equal(default.loss.detach(), c0.loss.detach())
    assert torch.equal(default_model.weight.grad, c0_model.weight.grad)
    assert default_model.forward_calls == c0_model.forward_calls == 1
    assert default.objective_metrics == c0.objective_metrics == ()

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from flash.engine.worker.opd_objectives import (
    OPD_OBJECTIVES,
    ObjectiveDefinition,
    ObjectiveRegistry,
    ObjectiveRequirements,
    ObjectiveResult,
    ObjectiveView,
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


def test_job_spec_persisted_reads_tolerate_future_objective_ids():
    from flash.spec import JobSpec

    persisted = {
        "algorithm": "grpo",
        "train": {"opd_objective_ids": ["future-objective", "future-objective"]},
    }

    restored = JobSpec.from_json(JobSpec.from_dict(persisted).to_json())

    assert restored.algorithm == "grpo"
    assert restored.train.opd_objective_ids == (
        "future-objective",
        "future-objective",
    )


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

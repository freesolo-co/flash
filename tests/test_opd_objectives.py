from __future__ import annotations

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


def _definition(objective_id, evaluate):
    return ObjectiveDefinition(
        objective_id=objective_id,
        requirements=ObjectiveRequirements(student_logits=True),
        evaluate=evaluate,
    )


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


def test_registry_enforces_declared_view_requirements():
    registry = ObjectiveRegistry((_definition("candidate", lambda _view: ObjectiveResult()),))
    with pytest.raises(RuntimeError, match="missing required value 'completion_logits'"):
        registry.evaluate(("candidate",), ObjectiveView())


def test_registry_namespaces_and_detaches_metrics():
    torch = pytest.importorskip("torch")
    metric = torch.tensor(2.5, requires_grad=True)
    registry = ObjectiveRegistry(
        (
            _definition(
                "candidate",
                lambda _view: ObjectiveResult(metrics={"score": metric}),
            ),
        )
    )

    evaluated = registry.evaluate(
        ("candidate",), ObjectiveView({"completion_logits": torch.zeros(1, 2)})
    )

    assert evaluated.terms == ()
    assert evaluated.metrics == {"opd/objectives/candidate/score": 2.5}
    assert isinstance(evaluated.metrics["opd/objectives/candidate/score"], float)


def test_registry_rejects_non_finite_terms():
    torch = pytest.importorskip("torch")
    registry = ObjectiveRegistry(
        (
            _definition(
                "candidate",
                lambda _view: ObjectiveResult(term=torch.tensor(float("nan"))),
            ),
        )
    )

    with pytest.raises(RuntimeError, match="non-finite term"):
        registry.evaluate(("candidate",), ObjectiveView({"completion_logits": torch.zeros(1, 2)}))


def test_c0_requires_no_extra_work_and_returns_no_additions():
    definition = OPD_OBJECTIVES.definitions["c0"]
    assert definition.requirements == ObjectiveRequirements()
    assert definition.requirements.extra_forwards == 0
    assert definition.requirements.network_calls == 0
    assert OPD_OBJECTIVES.evaluate(("c0",), ObjectiveView()).terms == ()
    assert OPD_OBJECTIVES.evaluate(("c0",), ObjectiveView()).metrics == {}


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

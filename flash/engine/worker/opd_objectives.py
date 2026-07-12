"""typed, closed objective registry for opd auxiliary loss terms."""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from flash.opd_objectives import OPD_OBJECTIVE_IDS, validate_opd_objective_id

_METRIC_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")


def _immutable_mapping(values: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True)
class ObjectiveRequirements:
    """runtime values needed by one objective without extra model or network calls."""

    student_logits: bool = False
    teacher_scores: bool = False


@dataclass(frozen=True)
class ObjectiveView:
    """immutable per-sample inputs exposed to objective implementations."""

    values: Mapping[str, Any] = field(default_factory=_immutable_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _immutable_mapping(self.values))

    def require(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise RuntimeError(f"opd objective view is missing required value {name!r}") from exc


@dataclass(frozen=True)
class ObjectiveResult:
    """one objective's optional scalar term and local detached metrics."""

    term: object | None = None
    metrics: Mapping[str, object] = field(default_factory=_immutable_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _immutable_mapping(self.metrics))


ObjectiveFunction = Callable[[ObjectiveView], ObjectiveResult]


@dataclass(frozen=True)
class ObjectiveDefinition:
    """closed registry entry for one objective id."""

    objective_id: str
    requirements: ObjectiveRequirements
    evaluate: ObjectiveFunction

    def __post_init__(self) -> None:
        validate_opd_objective_id(self.objective_id)


@dataclass(frozen=True)
class ObjectivePlan:
    """resolved definitions and their aggregate no-extra-work requirements."""

    definitions: tuple[ObjectiveDefinition, ...] = ()
    requirements: ObjectiveRequirements = field(default_factory=ObjectiveRequirements)

    @property
    def objective_ids(self) -> tuple[str, ...]:
        return tuple(definition.objective_id for definition in self.definitions)


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """immutable evaluated terms and fully namespaced detached metrics."""

    terms: tuple[object, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=_immutable_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "metrics", _immutable_mapping(self.metrics))


def _torch_module():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _detached_float(value: object, *, context: str) -> float:
    torch = _torch_module()
    if torch is not None and torch.is_tensor(value):
        detached = value.detach()
        if detached.numel() != 1:
            raise RuntimeError(f"{context} must be a scalar")
        if detached.dtype == torch.bool or torch.is_complex(detached):
            raise RuntimeError(f"{context} must be a real numeric scalar")
        number = float(detached.item())
    else:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise RuntimeError(f"{context} must be a real numeric scalar")
        number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{context} must be finite")
    return number


def _normalize_term(term: object, *, objective_id: str, base_term: object) -> object:
    context = f"opd objective {objective_id!r} term"
    torch = _torch_module()
    if torch is not None and torch.is_tensor(term):
        if term.numel() != 1:
            raise RuntimeError(f"{context} must be a one-element tensor")
        if term.dtype == torch.bool or torch.is_complex(term):
            raise RuntimeError(f"{context} must be a real tensor")
        if not bool(torch.isfinite(term.detach()).all().item()):
            raise RuntimeError(f"{context} must be finite")
        if not torch.is_tensor(base_term):
            raise RuntimeError(f"{context} requires a tensor base gkd loss")
        if term.device != base_term.device:
            raise RuntimeError(
                f"{context} device {term.device} does not match base gkd loss device "
                f"{base_term.device}"
            )
        if term.dtype != base_term.dtype:
            raise RuntimeError(
                f"{context} dtype {term.dtype} does not match base gkd loss dtype {base_term.dtype}"
            )
        return term.reshape(())
    if isinstance(term, bool) or not isinstance(term, numbers.Real):
        raise RuntimeError(f"{context} must be a real numeric scalar or one-element tensor")
    number = float(term)
    if not math.isfinite(number):
        raise RuntimeError(f"{context} must be finite")
    if torch is not None and torch.is_tensor(base_term):
        return base_term.new_tensor(number)
    return number


class ObjectiveRegistry:
    """immutable objective lookup, planning, and evaluation."""

    def __init__(self, definitions: Iterable[ObjectiveDefinition]) -> None:
        entries: dict[str, ObjectiveDefinition] = {}
        for definition in definitions:
            if definition.objective_id in entries:
                raise ValueError(f"duplicate opd objective id: {definition.objective_id}")
            entries[definition.objective_id] = definition
        self._definitions = MappingProxyType(entries)

    @property
    def definitions(self) -> Mapping[str, ObjectiveDefinition]:
        return self._definitions

    @property
    def objective_ids(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def resolve(self, objective_ids: Iterable[str]) -> tuple[ObjectiveDefinition, ...]:
        requested = tuple(objective_ids)
        seen: set[str] = set()
        duplicate: list[str] = []
        for objective_id in requested:
            validate_opd_objective_id(objective_id)
            if objective_id in seen and objective_id not in duplicate:
                duplicate.append(objective_id)
            seen.add(objective_id)
        if duplicate:
            raise ValueError(f"duplicate opd objective id(s): {', '.join(duplicate)}")
        unknown = [
            objective_id for objective_id in requested if objective_id not in self._definitions
        ]
        if unknown:
            allowed = ", ".join(self.objective_ids) or "none"
            raise ValueError(
                f"unknown opd objective id(s): {', '.join(unknown)}; allowed: {allowed}"
            )
        return tuple(self._definitions[objective_id] for objective_id in requested)

    def plan(self, objective_ids: Iterable[str]) -> ObjectivePlan:
        definitions = self.resolve(objective_ids)
        return ObjectivePlan(
            definitions=definitions,
            requirements=ObjectiveRequirements(
                student_logits=any(d.requirements.student_logits for d in definitions),
                teacher_scores=any(d.requirements.teacher_scores for d in definitions),
            ),
        )

    def evaluate(
        self,
        plan: ObjectivePlan,
        view: ObjectiveView,
        *,
        base_term: object,
    ) -> ObjectiveEvaluation:
        terms: list[object] = []
        metrics: dict[str, float] = {}
        for definition in plan.definitions:
            if definition.requirements.student_logits:
                view.require("completion_logits")
            if definition.requirements.teacher_scores:
                view.require("teacher_scores")
            result = definition.evaluate(view)
            if not isinstance(result, ObjectiveResult):
                raise TypeError(
                    f"opd objective {definition.objective_id!r} must return ObjectiveResult"
                )
            if result.term is not None:
                terms.append(
                    _normalize_term(
                        result.term,
                        objective_id=definition.objective_id,
                        base_term=base_term,
                    )
                )
            for name, value in result.metrics.items():
                if not _METRIC_NAME_RE.fullmatch(name):
                    raise ValueError(
                        f"opd objective {definition.objective_id!r} metric name must match "
                        f"[a-z][a-z0-9_.-]*, got {name!r}"
                    )
                key = f"opd/objectives/{definition.objective_id}/{name}"
                if key in metrics:
                    raise ValueError(f"duplicate opd objective metric: {key}")
                metrics[key] = _detached_float(value, context=f"opd objective metric {key!r}")
        return ObjectiveEvaluation(terms=tuple(terms), metrics=metrics)


def _c0(_view: ObjectiveView) -> ObjectiveResult:
    return ObjectiveResult()


OPD_OBJECTIVES = ObjectiveRegistry(
    (
        ObjectiveDefinition(
            objective_id="c0",
            requirements=ObjectiveRequirements(),
            evaluate=_c0,
        ),
    )
)

if OPD_OBJECTIVES.objective_ids != OPD_OBJECTIVE_IDS:
    raise RuntimeError(
        "worker opd objective definitions do not match shared objective identifiers: "
        f"{OPD_OBJECTIVES.objective_ids!r} != {OPD_OBJECTIVE_IDS!r}"
    )

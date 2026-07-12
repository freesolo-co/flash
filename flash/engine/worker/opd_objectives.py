"""typed, closed objective registry for opd auxiliary loss terms."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

_OBJECTIVE_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_METRIC_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")


def _immutable_mapping(values: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True)
class ObjectiveRequirements:
    """declared runtime work needed by one objective."""

    student_logits: bool = False
    teacher_scores: bool = False
    extra_forwards: int = 0
    network_calls: int = 0

    def __post_init__(self) -> None:
        if self.extra_forwards < 0 or self.network_calls < 0:
            raise ValueError("objective requirements cannot contain negative work counts")


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
        if not _OBJECTIVE_ID_RE.fullmatch(self.objective_id):
            raise ValueError(
                f"objective id must match [a-z][a-z0-9_.-]*, got {self.objective_id!r}"
            )


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """immutable evaluated terms and fully namespaced detached metrics."""

    terms: tuple[object, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=_immutable_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "metrics", _immutable_mapping(self.metrics))


def _detached_float(value: object, *, context: str) -> float:
    detached = value.detach() if hasattr(value, "detach") else value
    if hasattr(detached, "numel") and int(detached.numel()) != 1:
        raise RuntimeError(f"{context} must be a scalar")
    if hasattr(detached, "item"):
        detached = detached.item()
    try:
        number = float(detached)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be numeric") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{context} must be finite")
    return number


def _require_finite_term(term: object, *, objective_id: str) -> None:
    detached = term.detach() if hasattr(term, "detach") else term
    if hasattr(detached, "numel") and int(detached.numel()) != 1:
        raise RuntimeError(f"opd objective {objective_id!r} returned a non-scalar term")
    if hasattr(detached, "isfinite"):
        finite = detached.isfinite()
        if hasattr(finite, "all"):
            finite = finite.all()
        if hasattr(finite, "item"):
            finite = finite.item()
        if not bool(finite):
            raise RuntimeError(f"opd objective {objective_id!r} returned a non-finite term")
        return
    _detached_float(term, context=f"opd objective {objective_id!r} term")


class ObjectiveRegistry:
    """immutable objective lookup and evaluator with strict id validation."""

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

    def evaluate(self, objective_ids: Iterable[str], view: ObjectiveView) -> ObjectiveEvaluation:
        terms: list[object] = []
        metrics: dict[str, float] = {}
        for definition in self.resolve(objective_ids):
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
                _require_finite_term(result.term, objective_id=definition.objective_id)
                terms.append(result.term)
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

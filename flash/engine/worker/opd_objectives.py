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
    position_statistics: bool = False


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


@dataclass(frozen=True)
class PositionStatistics:
    """detached, objective-neutral completion-row statistics."""

    entropies: tuple[float, ...] = ()
    eligible_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class CvarEntropyConfig:
    """strict runtime config for the c10 lower-tail entropy objective."""

    fraction: float
    floor: float
    coef: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.fraction) or not 0.0 < self.fraction <= 1.0:
            raise ValueError("c10 cvar entropy fraction must be finite and in (0, 1]")
        if not math.isfinite(self.floor) or self.floor < 0.0:
            raise ValueError("c10 cvar entropy floor must be finite and >= 0")
        if not math.isfinite(self.coef) or self.coef < 0.0:
            raise ValueError("c10 cvar entropy coef must be finite and >= 0")


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
                position_statistics=any(d.requirements.position_statistics for d in definitions),
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
            if definition.requirements.position_statistics:
                view.require("position_statistics")
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


def _c10_gradient_ratio(term, base_term, rows) -> float | None:
    torch = _torch_module()
    if torch is None or not all(torch.is_tensor(value) for value in (term, base_term, rows)):
        return None
    try:
        tail_grad = torch.autograd.grad(term, rows, retain_graph=True, allow_unused=True)[0]
        base_grad = torch.autograd.grad(base_term, rows, retain_graph=True, allow_unused=True)[0]
    except RuntimeError:
        return None
    if tail_grad is None or base_grad is None:
        return None
    denominator = float(base_grad.detach().float().norm())
    numerator = float(tail_grad.detach().float().norm())
    if denominator <= 0.0 or not math.isfinite(denominator) or not math.isfinite(numerator):
        return None
    return numerator / denominator


def _c10(view: ObjectiveView) -> ObjectiveResult:
    stats = view.require("position_statistics")
    config = view.require("cvar_entropy_config")
    if not isinstance(stats, PositionStatistics):
        raise TypeError("c10 position_statistics must be PositionStatistics")
    if not isinstance(config, CvarEntropyConfig):
        raise TypeError("c10 cvar_entropy_config must be CvarEntropyConfig")

    finite = [
        (index, stats.entropies[index])
        for index in stats.eligible_indices
        if index < len(stats.entropies) and math.isfinite(stats.entropies[index])
    ]
    metrics: dict[str, object] = {
        "tail_fraction": config.fraction,
        "threshold": 0.0,
        "selected_count": 0.0,
        "mean_entropy": 0.0,
        "tail_entropy": 0.0,
        "activation": 0.0,
    }
    if not finite:
        return ObjectiveResult(metrics=metrics)

    ordered = sorted(entropy for _, entropy in finite)
    count = max(1, math.ceil(config.fraction * len(ordered)))
    threshold = ordered[count - 1]
    selected = tuple(index for index, entropy in finite if entropy <= threshold)
    tail_entropy_detached = sum(stats.entropies[index] for index in selected) / len(selected)
    metrics.update(
        threshold=threshold,
        selected_count=float(len(selected)),
        mean_entropy=sum(entropy for _, entropy in finite) / len(finite),
        tail_entropy=tail_entropy_detached,
    )
    if config.coef <= 0.0 or tail_entropy_detached >= config.floor:
        return ObjectiveResult(metrics=metrics)

    entropy_for_rows = view.require("entropy_for_rows")
    tail_entropy = entropy_for_rows(selected)
    if tail_entropy is None:
        return ObjectiveResult(metrics=metrics)
    term = config.coef * (config.floor - tail_entropy)
    metrics["activation"] = 1.0
    ratio = _c10_gradient_ratio(
        term,
        view.values.get("base_term"),
        view.values.get("completion_logits"),
    )
    if ratio is not None:
        metrics["gradient_ratio"] = ratio
    return ObjectiveResult(term=term, metrics=metrics)


OPD_OBJECTIVES = ObjectiveRegistry(
    (
        ObjectiveDefinition(
            objective_id="c0",
            requirements=ObjectiveRequirements(),
            evaluate=_c0,
        ),
        ObjectiveDefinition(
            objective_id="c10",
            requirements=ObjectiveRequirements(student_logits=True, position_statistics=True),
            evaluate=_c10,
        ),
    )
)

if OPD_OBJECTIVES.objective_ids != OPD_OBJECTIVE_IDS:
    raise RuntimeError(
        "worker opd objective definitions do not match shared objective identifiers: "
        f"{OPD_OBJECTIVES.objective_ids!r} != {OPD_OBJECTIVE_IDS!r}"
    )

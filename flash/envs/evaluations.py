"""Flash-native held-out evaluation suites for Freesolo environment packages.

An environment package may define ``evaluations.py`` beside ``environment.py``.
The sidecar exports ``load_evaluations(...)`` or a module-level ``EVALUATIONS``
list/tuple of suites. Flash resolves the whole environment package first, so the
same contract works for local paths, managed hub slugs, and GitHub references.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Protocol, runtime_checkable

_DEFAULT_EVALUATIONS_PATH = "evaluations.py"
_DEFAULT_EVALUATIONS_FACTORY = "load_evaluations"


@dataclass(frozen=True)
class EvalCase:
    """One held-out prompt and its optional gold answer."""

    input: str
    expected: object | None = None
    id: str | None = None
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input, str) or not self.input.strip():
            raise ValueError("EvalCase.input must be a non-empty string")
        if self.id is not None and (not isinstance(self.id, str) or not self.id.strip()):
            raise ValueError("EvalCase.id must be a non-empty string when provided")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("EvalCase.metadata must be a dict when provided")


@dataclass(frozen=True)
class EvalResult:
    """Normalized result for one evaluation case."""

    case_id: str
    passed: bool
    score: float
    response: str
    reason: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("EvalResult.case_id must be a non-empty string")
        if not isinstance(self.passed, bool):
            raise TypeError("EvalResult.passed must be a bool")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("EvalResult.score must be a finite float")
        if not math.isfinite(float(self.score)):
            raise ValueError(f"EvalResult.score must be finite, got {self.score!r}")
        object.__setattr__(self, "score", float(self.score))
        if not isinstance(self.response, str):
            raise TypeError("EvalResult.response must be a string")
        for field_name in ("reason", "error"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"EvalResult.{field_name} must be a string when provided")


@dataclass(frozen=True)
class EvalSuiteReport:
    """Aggregate metrics for one evaluation suite."""

    name: str
    results: tuple[EvalResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("EvalSuiteReport.name must be a non-empty string")
        if not isinstance(self.results, tuple) or not all(
            isinstance(result, EvalResult) for result in self.results
        ):
            raise TypeError("EvalSuiteReport.results must be a tuple of EvalResult objects")

    @property
    def scored(self) -> tuple[EvalResult, ...]:
        """Results carrying a real measurement; a generation or scoring error is not one."""
        return tuple(result for result in self.results if result.error is None)

    @property
    def errors(self) -> int:
        return len(self.results) - len(self.scored)

    @property
    def passed(self) -> int:
        return sum(result.passed for result in self.scored)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        # a case that never reached the model is missing data, not a zero the model earned.
        # averaging it in would report a healthy suite behind a broken deployment as low quality.
        scored = self.scored
        return sum(result.passed for result in scored) / len(scored) if scored else 0.0

    @property
    def mean_score(self) -> float:
        scored = self.scored
        return sum(result.score for result in scored) / len(scored) if scored else 0.0


@runtime_checkable
class EvalSuite(Protocol):
    """Authoring contract for one named held-out evaluation suite."""

    name: str

    def cases(self) -> list[EvalCase]: ...

    def score(self, case: EvalCase, response: str) -> EvalResult | float | bool: ...


class BaseEvalSuite:
    """Convenience base with expected-answer substring scoring."""

    name = "evaluation"

    def cases(self) -> list[EvalCase]:
        raise NotImplementedError

    def score(self, case: EvalCase, response: str) -> EvalResult:
        # `case.expected or ""` would erase a falsy-but-real gold answer: expected=0 and
        # expected=False are legitimate answers no response could ever match. Only an
        # absent expected means "no gold to compare against".
        gold = "" if case.expected is None else str(case.expected).strip()
        # `"" in x` is always true, so missing gold must not grade every response correct.
        passed = bool(gold) and gold in (response or "")
        return EvalResult(
            case_id=case.id or "case",
            passed=passed,
            score=1.0 if passed else 0.0,
            response=response,
        )


def normalize_eval_result(
    case: EvalCase,
    response: str,
    result: EvalResult | float | bool,
    *,
    case_id: str | None = None,
) -> EvalResult:
    """Normalize a suite score return and validate its finite result contract."""
    resolved_case_id = case_id or case.id or "case"
    if isinstance(result, EvalResult):
        updates = {}
        if result.case_id != resolved_case_id:
            updates["case_id"] = resolved_case_id
        if result.response != response:
            updates["response"] = response
        return replace(result, **updates) if updates else result
    if isinstance(result, bool):
        return EvalResult(
            case_id=resolved_case_id,
            passed=result,
            score=1.0 if result else 0.0,
            response=response,
        )
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        score = float(result)
        if not math.isfinite(score):
            raise ValueError(f"evaluation score must be finite, got {score!r}")
        return EvalResult(
            case_id=resolved_case_id,
            passed=score > 0.0,
            score=score,
            response=response,
        )
    raise TypeError(
        f"evaluation score must return EvalResult, float, or bool; got {type(result).__name__}"
    )


def validate_evaluation_cases(suite: EvalSuite, *, source: str | Path) -> list[EvalCase]:
    """Load and validate one suite's cases with source-aware errors."""
    try:
        cases = suite.cases()
    except Exception as exc:
        raise RuntimeError(
            f"{source}: evaluation suite {suite.name!r} cases() failed: "
            f"{str(exc) or exc.__class__.__name__}"
        ) from exc
    if not isinstance(cases, list):
        raise TypeError(
            f"{source}: evaluation suite {suite.name!r} cases() must return "
            f"list[EvalCase], got {type(cases).__name__}"
        )
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, EvalCase):
            raise TypeError(
                f"{source}: evaluation suite {suite.name!r} case {index} must be "
                f"an EvalCase, got {type(case).__name__}"
            )
    return cases


def _resolve_environment_entrypoint(env_ref_or_path: str | Path) -> Path:
    from flash.envs.loader import _DEFAULT_ENVIRONMENT_PATH, _resolve_environment_reference

    raw = str(env_ref_or_path)
    local = Path(raw).expanduser()
    if local.exists():
        resolved = local.resolve()
    else:
        resolved_reference = _resolve_environment_reference(raw)
        resolved = Path(resolved_reference)
    if resolved.is_dir():
        resolved = resolved / _DEFAULT_ENVIRONMENT_PATH
    if not resolved.is_file():
        raise FileNotFoundError(
            f"could not locate environment entrypoint for {raw!r}; expected "
            f"{_DEFAULT_ENVIRONMENT_PATH} in the resolved environment package"
        )
    return resolved


def _evaluation_path(env_ref_or_path: str | Path) -> Path:
    return _resolve_environment_entrypoint(env_ref_or_path).parent / _DEFAULT_EVALUATIONS_PATH


def has_evaluations(env_ref_or_path: str | Path) -> bool:
    """Return whether the resolved environment package contains evaluations.py."""
    return _evaluation_path(env_ref_or_path).is_file()


def _import_evaluations_module(module_path: Path) -> ModuleType:
    module_name = f"_flash_env_evaluations_{module_path.stem}_{abs(hash(module_path))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import evaluation module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    # the package dir stays on sys.path after this returns: a sidecar may import its local
    # helpers lazily inside cases() or score(), which run long after the module is loaded.
    # removing it here would break those sidecars. the membership check keeps repeated loads
    # from growing sys.path without bound.
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    # a sibling a sidecar imports is cached under its plain name, so a second package importing
    # its own `helper` found the FIRST package's module already in sys.modules and reused it --
    # silently running the wrong cases and the wrong scoring logic, with no import error to see
    # (codex[bot]). dropping the siblings this load introduced makes the next package import its
    # own; anything already imported before this call is left alone, since it is not ours to evict.
    before = set(sys.modules)
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        _forget_sidecar_siblings(module_dir, before)
        raise
    _forget_sidecar_siblings(module_dir, before)
    return module


def _forget_sidecar_siblings(module_dir: str, before: set[str]) -> None:
    """Drop modules this sidecar load imported from its own package directory.

    Only modules that (a) were absent before the load and (b) resolve to a file inside this
    package directory. A stdlib or third-party module the sidecar imported first stays cached:
    evicting it would re-execute unrelated code for every later load."""
    directory = Path(module_dir).resolve()
    for name in set(sys.modules) - before:
        origin = getattr(getattr(sys.modules[name], "__spec__", None), "origin", None)
        if not origin:
            continue
        try:
            resolved = Path(origin).resolve()
        except OSError:
            continue
        if resolved.parent == directory:
            sys.modules.pop(name, None)


def _call_factory(factory, kwargs: dict[str, object]) -> object:
    signature = inspect.signature(factory)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    has_positional_only = any(
        parameter.kind == inspect.Parameter.POSITIONAL_ONLY
        for parameter in signature.parameters.values()
    )
    # **kwargs alone means every name is accepted, so pass them through. but `**kwargs` does not
    # make a positional-only parameter keyword-passable: `load_evaluations(environment, /, **opts)`
    # put environment into opts and then raised for the missing positional, so a valid factory
    # failed to load at all. fall through to the positional handling below when both are present.
    if accepts_var_kwargs and not has_positional_only:
        return factory(**kwargs)
    # a positional-only parameter is in signature.parameters but cannot be passed by name, so
    # matching on membership alone would raise TypeError for `load_evaluations(environment, /)`.
    # such a factory declared the argument, so pass it positionally rather than dropping it and
    # handing the suite environment=None -- which downgrades a real scorer to substring matching.
    #
    # a positional-only parameter we have no value for still has to be filled, otherwise every
    # parameter after it shifts left. use its default; a required one cannot be satisfied at all,
    # so stop and let the factory raise its own TypeError rather than passing a wrong argument.
    positional: list[object] = []
    from_kwargs: list[bool] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind != inspect.Parameter.POSITIONAL_ONLY:
            break
        if name in kwargs:
            positional.append(kwargs[name])
            from_kwargs.append(True)
            continue
        if parameter.default is inspect.Parameter.empty:
            break
        positional.append(parameter.default)
        from_kwargs.append(False)
    # trailing arguments filled from their own defaults carry no information, so drop them and
    # let the factory apply those defaults itself.
    while from_kwargs and not from_kwargs[-1]:
        positional.pop()
        from_kwargs.pop()
    consumed = list(signature.parameters)[: len(positional)]
    filtered_kwargs = {
        name: value
        for name, value in kwargs.items()
        # a factory with **kwargs accepts any name, so only drop what a positional slot already got.
        if (accepts_var_kwargs or name in signature.parameters) and name not in consumed
    }
    return factory(*positional, **filtered_kwargs)


def _validate_suites(value: object, *, source: Path) -> list[EvalSuite]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"{source}: expected {_DEFAULT_EVALUATIONS_FACTORY}() or EVALUATIONS "
            f"to provide a list/tuple of evaluation suites, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(
            f"{source}: expected at least one evaluation suite from "
            f"{_DEFAULT_EVALUATIONS_FACTORY}() or EVALUATIONS"
        )
    suites: list[EvalSuite] = []
    names: set[str] = set()
    for index, suite in enumerate(value, start=1):
        name = getattr(suite, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise TypeError(
                f"{source}: evaluation suite {index} must define a non-empty string name"
            )
        if name in names:
            raise ValueError(f"{source}: duplicate evaluation suite name {name!r}")
        if not callable(getattr(suite, "cases", None)) or not callable(
            getattr(suite, "score", None)
        ):
            raise TypeError(
                f"{source}: evaluation suite {name!r} must define cases() and score(case, response)"
            )
        names.add(name)
        suites.append(suite)
    return suites


def load_evaluation_suites(
    env_ref_or_path: str | Path, *, environment: object | None = None
) -> list[EvalSuite]:
    """Load evaluations.py beside a resolved environment entrypoint."""
    source = _evaluation_path(env_ref_or_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"{source}: no {_DEFAULT_EVALUATIONS_PATH} sidecar found beside the environment entrypoint"
        )
    try:
        module = _import_evaluations_module(source)
    except Exception as exc:
        raise RuntimeError(
            f"failed to import evaluation sidecar {source}: {str(exc) or exc.__class__.__name__}"
        ) from exc

    factory = getattr(module, _DEFAULT_EVALUATIONS_FACTORY, None)
    try:
        if factory is not None:
            if not callable(factory):
                raise TypeError(f"{source}: {_DEFAULT_EVALUATIONS_FACTORY} must be callable")
            loaded = _call_factory(factory, {"environment": environment})
        elif hasattr(module, "EVALUATIONS"):
            loaded = module.EVALUATIONS
        else:
            raise AttributeError(
                f"{source}: define {_DEFAULT_EVALUATIONS_FACTORY}() or a module-level EVALUATIONS list/tuple"
            )
        return _validate_suites(loaded, source=source)
    except Exception as exc:
        if isinstance(exc, (AttributeError, TypeError, ValueError, RuntimeError)) and str(
            exc
        ).startswith(str(source)):
            raise
        raise RuntimeError(
            f"{source}: could not load evaluation suites: {str(exc) or exc.__class__.__name__}"
        ) from exc


__all__ = [
    "_DEFAULT_EVALUATIONS_PATH",
    "BaseEvalSuite",
    "EvalCase",
    "EvalResult",
    "EvalSuite",
    "EvalSuiteReport",
    "has_evaluations",
    "load_evaluation_suites",
    "normalize_eval_result",
    "validate_evaluation_cases",
]

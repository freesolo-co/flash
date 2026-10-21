"""load flash-native held-out suites from an environment package.

packages expose ``load_evaluations(...)`` or ``EVALUATIONS`` beside ``environment.py``. resolving the
whole package keeps the contract identical for local paths, hub slugs, and github references.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
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
        # an error means the case was never graded, so it cannot also have passed. the report
        # already excludes it from the metrics and fails the command, but the console labelled it
        # PASS and the upload recorded `success: true` beside the error -- three readings of one
        # case that disagree. reject the state rather than pick a winner downstream.
        if self.error and self.passed:
            raise ValueError(
                f"EvalResult.passed must be False when an error is reported, "
                f"got error={self.error!r}"
            )


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
    """Authoring contract for a suite that grades one model response."""

    @property
    def name(self) -> str: ...

    def cases(self) -> list[EvalCase]: ...

    def score(self, case: EvalCase, response: str) -> EvalResult | float | bool: ...


class EpisodeEvalSuite(Protocol):
    """Suite contract for full-episode play at one generation per turn.

    ``grades_episodes`` must be true at runtime. The finished transcript is passed in ``state``.
    """

    @property
    def name(self) -> str: ...

    grades_episodes: bool

    def cases(self) -> list[EvalCase]: ...

    def score(self, case: EvalCase, response: str, state: dict) -> EvalResult | float | bool: ...


EvaluationSuite = EvalSuite | EpisodeEvalSuite


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


def _episode_grading_enabled(suite: EvaluationSuite) -> bool:
    """Return the validated full-episode grading opt-in."""
    value = getattr(suite, "grades_episodes", False)
    if not isinstance(value, bool):
        name = getattr(suite, "name", "evaluation")
        raise TypeError(
            f"suite {name!r} grades_episodes must be a bool, got {type(value).__name__}"
        )
    return value


def validate_evaluation_cases(suite: EvaluationSuite, *, source: str | Path) -> list[EvalCase]:
    """Load and validate one suite's cases with source-aware errors."""
    _episode_grading_enabled(suite)
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
    from flash.envs.loading.loader import _DEFAULT_ENVIRONMENT_PATH, _resolve_environment_reference

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
    # bare sibling imports are global in sys.modules, so another package can silently reuse the first
    # package's helper. remove only modules introduced from this package, including lazy imports made
    # during cases() and score(); preserve modules that predated the scope.
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        with _sidecar_scope(str(module_path.parent)):
            spec.loader.exec_module(module)
    except BaseException:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise
    return module


# Each sidecar directory's own modules, parked here while it is out of scope. They are removed from
# sys.modules so the next environment's `helper` resolves to its own file, and put back on re-entry
# so a helper's module-level state -- a counter, a client, a loaded judge model -- survives from one
# case to the next instead of re-executing for every callback.
_PARKED_SIDECAR_MODULES: dict[str, dict[str, ModuleType]] = {}


def _forget_sidecar_siblings(module_dir: str, before: set[str]) -> dict[str, ModuleType]:
    """remove and return modules introduced from this package directory.

    include nested descendants such as ``graders.rules``; matching only immediate children leaves
    stale scoring modules cached. preserve stdlib/third-party modules that existed before the load.
    """
    directory = Path(module_dir).resolve()
    taken: dict[str, ModuleType] = {}
    for name in set(sys.modules) - before:
        module = sys.modules[name]
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if origin:
            if _is_under(origin, directory):
                taken[name] = sys.modules.pop(name)
            continue
        # a namespace package (a sibling dir with no __init__.py) has no origin at all, so an
        # origin-only test leaves it cached. its __path__ recomputes from sys.path, so imports
        # through it still resolve correctly, but the module object itself would outlive the
        # load that created it.
        search = getattr(module, "__path__", None)
        if search is not None and any(_is_under(entry, directory) for entry in search):
            taken[name] = sys.modules.pop(name)
    return taken


def _is_under(path: str, directory: Path) -> bool:
    """Whether `path` names a file or directory inside `directory`."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    return directory in resolved.parents


def _sidecar_module_names(directory: Path) -> set[str]:
    """return top-level import names this package directory can supply.

    include identifier-named namespace-package directories even without ``__init__.py`` (pep 420),
    otherwise nested helpers can remain cached across environments. exclude non-identifiers and
    ``__pycache__`` because this directory cannot intentionally import them as helpers.
    """
    names: set[str] = set()
    try:
        entries = list(directory.iterdir())
    except OSError:
        return names
    for entry in entries:
        if entry.is_symlink():
            continue
        if entry.is_file() and entry.suffix == ".py":
            names.add(entry.stem)
        elif entry.is_dir() and entry.name.isidentifier() and entry.name != "__pycache__":
            names.add(entry.name)
    return names


def _shadowed_cached_modules(directory: Path) -> dict[str, ModuleType]:
    """temporarily displace cached modules whose names this directory owns.

    sys.modules precedes sys.path, so cached ``helper`` or ``graders.rules`` from another environment
    would silently win. displace the package and its submodules only when they resolve elsewhere;
    modules already loaded from this directory are correct.
    """
    shadowed: dict[str, ModuleType] = {}
    for name in _sidecar_module_names(directory):
        prefix = name + "."
        for cached_name in [name, *(n for n in sys.modules if n.startswith(prefix))]:
            cached = sys.modules.get(cached_name)
            if cached is None:
                continue
            origin = getattr(getattr(cached, "__spec__", None), "origin", None)
            if origin and _is_under(origin, directory):
                continue
            shadowed[cached_name] = cached
    return shadowed


@contextmanager
def _sidecar_scope(module_dir: str) -> Iterator[None]:
    """scope bare imports to ``module_dir`` and restore them afterward.

    cover lazy imports inside cases() and score(), not only module loading. park this sidecar's modules
    on exit so another package cannot reuse them, then restore the same objects on re-entry to avoid
    reinitializing counters, models, or connections.
    """
    resolved = Path(module_dir).resolve()
    directory = str(resolved)
    before = set(sys.modules)
    # evict first, so `before` records the name as absent and the exit path parks whatever the
    # sidecar imported in its place before the original is restored.
    shadowed = _shadowed_cached_modules(resolved)
    for name in shadowed:
        del sys.modules[name]
    before -= set(shadowed)
    # this directory's own modules from a previous scope, resuming with the state they built up.
    # they were absent from sys.modules until now, so `before` correctly excludes them and the
    # exit path parks them again.
    sys.modules.update(_PARKED_SIDECAR_MODULES.pop(directory, {}))
    sys.path.insert(0, directory)
    try:
        yield
    finally:
        # a sidecar that rewrote sys.path itself may already have removed it; nothing to undo.
        with suppress(ValueError):
            sys.path.remove(directory)
        parked = _forget_sidecar_siblings(directory, before)
        if parked:
            _PARKED_SIDECAR_MODULES[directory] = parked
        # restore unconditionally: the displaced module belongs to the rest of the process, and
        # a sidecar that never imported the name must not leave it evicted either.
        sys.modules.update(shadowed)


class _ScopedSuite:
    """One suite bound to the package directory its sidecar was loaded from.

    Delegates rather than subclasses: a suite is any object with `name`, `cases()`, and
    `score()`, so there is no base class to extend and user state must stay on the user's
    own object."""

    __slots__ = ("_module_dir", "_suite")

    def __init__(self, suite: EvaluationSuite, module_dir: str) -> None:
        self._suite = suite
        self._module_dir = module_dir

    @property
    def name(self) -> str:
        return self._suite.name

    def cases(self) -> list[EvalCase]:
        with _sidecar_scope(self._module_dir):
            return self._suite.cases()

    @property
    def score(self):
        """The suite's own scorer, run inside the sidecar scope and reporting its own signature.

        A fixed `(case, response)` method here erased the episode state an episode-grading suite
        is meant to receive. The caller inspects THIS object's `score` to decide whether and how
        to pass state, so a real sidecar -- always wrapped by the loader -- looked like a
        two-argument scorer and was never handed the transcript, while `grades_episodes` still
        forwarded through `__getattr__` and promised otherwise.

        Forwarding `*args`/`**kwargs` alone is not enough: that signature reports as accepting
        state for EVERY suite, so a genuine two-argument scorer would be called with a third
        argument it cannot take. Copying `__signature__` from the inner scorer keeps inspection
        honest in both directions.
        """
        inner = self._suite.score
        module_dir = self._module_dir

        def scoped_score(*args, **kwargs):
            with _sidecar_scope(module_dir):
                return inner(*args, **kwargs)

        with suppress(TypeError, ValueError):
            scoped_score.__signature__ = inspect.signature(inner)
        return scoped_score

    def __getattr__(self, name: str):
        # tests and callers reach through to suite attributes (`suite.environment`), and an
        # unknown name must raise the suite's own AttributeError, not this wrapper's.
        return getattr(self._suite, name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._suite!r})"


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
    # positional-only ``environment`` must be passed positionally, not omitted. fill earlier
    # positional-only parameters from defaults so arguments do not shift; if a required value is
    # unavailable, let the factory raise its own TypeError.
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


def _validate_suites(value: object, *, source: Path) -> list[EvaluationSuite]:
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
    suites: list[EvaluationSuite] = []
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
) -> list[EvaluationSuite]:
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
    module_dir = str(source.parent)
    try:
        if factory is not None:
            if not callable(factory):
                raise TypeError(f"{source}: {_DEFAULT_EVALUATIONS_FACTORY} must be callable")
            # the factory is user code that may import its own helpers too, so it runs in the
            # same scope its cases() and score() will later run in.
            with _sidecar_scope(module_dir):
                loaded = _call_factory(factory, {"environment": environment})
        elif hasattr(module, "EVALUATIONS"):
            loaded = module.EVALUATIONS
        else:
            raise AttributeError(
                f"{source}: define {_DEFAULT_EVALUATIONS_FACTORY}() or a module-level EVALUATIONS list/tuple"
            )
        # bound to this package before returning: cases() and score() run after this call, when
        # nothing else would keep a sibling import resolving to the package it belongs to.
        return [
            _ScopedSuite(suite, module_dir) for suite in _validate_suites(loaded, source=source)
        ]
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
    "EpisodeEvalSuite",
    "EvalCase",
    "EvalResult",
    "EvalSuite",
    "EvalSuiteReport",
    "EvaluationSuite",
    "_evaluation_path",
    "has_evaluations",
    "load_evaluation_suites",
    "normalize_eval_result",
    "validate_evaluation_cases",
]

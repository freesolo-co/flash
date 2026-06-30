"""Training-specific environment loader parameter handling."""

from __future__ import annotations

import ast
import os
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FLASH_TRAIN_MAX_EXAMPLES = "__flash_train_max_examples"
_LOADER_CAP_PARAMS = ("max_examples", "limit")
_LOADER_CAP_PARAM_SET = frozenset(_LOADER_CAP_PARAMS)
_MAX_REEXPORT_DEPTH = 8


@dataclass(frozen=True)
class _LoaderSignature:
    keyword_params: frozenset[str]
    positional_only_params: frozenset[str]
    accepts_kwargs: bool

    @property
    def declares_cap(self) -> bool:
        return bool(self.keyword_params & _LOADER_CAP_PARAM_SET)


_EMPTY_SIGNATURE = _LoaderSignature(frozenset(), frozenset(), False)


def training_env_params(spec: Any) -> dict[str, Any]:
    """Environment-loader params for a training run.

    ``[train].max_examples`` caps SFT after the worker shuffles examples, but some
    environments also have a loader-side cap with a small default. Carry an internal marker
    so the adapter can clear that loader cap before the worker calls ``env.dataset()``.
    """
    params = dict(getattr(getattr(spec, "environment", None), "params", None) or {})
    train = getattr(spec, "train", None)
    if train is None:
        return params
    if _has_loader_cap(params):
        return params

    algorithm = str(getattr(spec, "algorithm", "") or "").lower()
    if algorithm != "sft":
        return params

    params[_FLASH_TRAIN_MAX_EXAMPLES] = None
    return params


def apply_training_max_examples(reference: str, params: dict[str, Any]) -> dict[str, Any]:
    """Clear loader-side dataset caps when SFT intends Flash to own max_examples."""
    if _FLASH_TRAIN_MAX_EXAMPLES not in params:
        return params

    params.pop(_FLASH_TRAIN_MAX_EXAMPLES)
    if _has_loader_cap(params):
        return params

    ref_path = Path(reference)
    signature = _load_environment_signature(ref_path) if ref_path.is_file() else _EMPTY_SIGNATURE
    for name in _LOADER_CAP_PARAMS:
        if name in signature.keyword_params:
            params[name] = None
    if not signature.declares_cap and signature.accepts_kwargs:
        for name in _LOADER_CAP_PARAMS:
            if name not in signature.positional_only_params:
                params[name] = None
    return params


def _has_loader_cap(params: dict[str, Any]) -> bool:
    return any(name in params for name in _LOADER_CAP_PARAMS)


def _resolved_import_path(base: Path, module: str | None, level: int) -> Path | None:
    if module is None:
        return None
    parts = module.split(".")
    if any(not part.isidentifier() for part in parts):
        return None

    root = base.parent
    for _ in range(max(level - 1, 0)):
        root = root.parent

    candidate = root.joinpath(*parts)
    file_candidate = candidate.with_suffix(".py")
    if file_candidate.is_file():
        return file_candidate
    init_candidate = candidate / "__init__.py"
    if init_candidate.is_file():
        return init_candidate
    return None


def _load_environment_signature(
    path: Path,
    target_name: str = "load_environment",
    seen: set[Path] | None = None,
) -> _LoaderSignature:
    if seen is None:
        seen = set()
    resolved_path = path.resolve()
    if resolved_path in seen or len(seen) >= _MAX_REEXPORT_DEPTH:
        return _EMPTY_SIGNATURE
    seen.add(resolved_path)
    try:
        with tokenize.open(path) as source:
            tree = ast.parse(source.read(), filename=os.fspath(path))
    except Exception:
        return _EMPTY_SIGNATURE

    signature: _LoaderSignature | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == target_name:
            signature = _function_signature(node)
        elif isinstance(node, ast.ImportFrom):
            imported_signature = _imported_signature(path, node, target_name, seen)
            if imported_signature is not None:
                signature = imported_signature
    return signature or _EMPTY_SIGNATURE


def _function_signature(node: ast.FunctionDef) -> _LoaderSignature:
    args = node.args
    keyword_names = {
        arg.arg
        for arg in (
            *args.args,
            *args.kwonlyargs,
        )
    }
    positional_only_names = {arg.arg for arg in args.posonlyargs}
    return _LoaderSignature(
        frozenset(keyword_names),
        frozenset(positional_only_names),
        args.kwarg is not None,
    )


def _imported_signature(
    base: Path,
    node: ast.ImportFrom,
    target_name: str,
    seen: set[Path],
) -> _LoaderSignature | None:
    for alias in node.names:
        if (alias.asname or alias.name) != target_name:
            continue
        import_path = _resolved_import_path(base, node.module, node.level)
        if import_path is None:
            return None
        return _load_environment_signature(import_path, alias.name, seen)
    return None

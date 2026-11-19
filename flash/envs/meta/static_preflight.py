"""Submit-time environment checks that read the package source without executing it.

The control plane never imports ``environment.py``: it is user code and the plane is shared. So
these checks read the AST and compare it with the Freesolo SDK installed here, which is the same
pinned SDK the worker image installs. Each finding is a load failure the worker would otherwise
hit only after a GPU has been rented and cold-started. The checks are deliberately one-sided:
anything not provable from the source (guarded imports, bases imported from other packages,
dynamic attributes) is skipped, so a passing preflight proves nothing but a failing one is
certain.
"""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path

_MAX_LOCAL_MODULES = 64
_MAX_FINDINGS = 8
_SDK_ROOT = "freesolo"


class EnvironmentPreflightError(ValueError):
    """the packaged environment cannot load on the worker, proven from its source."""


@dataclass(frozen=True)
class StaticEnvironmentFacts:
    """what the source proves about the env class ``load_environment()`` builds."""

    # the class declares ``image_observations = True``, which makes a GRPO run multimodal
    # (flash/envs/loading/adapter.py reads exactly this class attribute).
    image_observations: bool = False


def _local_module_path(base_dir: Path, name: str) -> Path | None:
    head = name.split(".", 1)[0]
    for candidate in (base_dir / f"{head}.py", base_dir / head / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _parse(path: Path, base_dir: Path) -> ast.Module:
    try:
        return ast.parse(path.read_bytes(), filename=str(path))
    except SyntaxError as exc:
        raise EnvironmentPreflightError(
            f"{path.relative_to(base_dir)} line {exc.lineno}: SyntaxError: {exc.msg}"
        ) from None


def _module_names(tree: ast.Module) -> list[tuple[ast.stmt, str, list[str]]]:
    """top-level absolute imports only: these run unconditionally when the module is imported."""
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend((node, alias.name, []) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node, node.module, [alias.name for alias in node.names]))
    return found


def _sdk_import_error(module: str, names: list[str]) -> str | None:
    try:
        imported = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == module or module.startswith(exc.name + ".")):
            return f"ModuleNotFoundError: No module named {exc.name!r}"
        return None
    except Exception:
        return None
    for name in names:
        if name == "*" or hasattr(imported, name):
            continue
        try:
            importlib.import_module(f"{module}.{name}")
        except ImportError:
            return f"ImportError: cannot import name {name!r} from {module!r}"
    return None


def _sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version(_SDK_ROOT)
    except Exception:
        return "unknown"


def _may_bind(tree: ast.Module, name: str) -> bool:
    """whether a package ``__init__`` could bind ``name`` itself (conservatively yes)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (
            node.id in {"globals", "setattr", "__getattr__"}
            or (node.id == name and isinstance(node.ctx, ast.Store))
        ):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in {name, "__getattr__"}:
                return True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if alias.name == "*" or bound == name:
                    return True
    return False


def _imported_submodules(init: Path, names: list[str], base_dir: Path) -> list[Path]:
    """submodules ``from pkg import name`` provably imports.

    Python imports ``pkg/name.py`` only when ``pkg/__init__.py`` has not already bound ``name``,
    so a submodule the package could shadow is never checked.
    """
    subs = [init.parent / f"{name}.py" for name in names]
    subs = [sub for sub in subs if sub.is_file()]
    if not subs:
        return []
    tree = _parse(init, base_dir)
    return [sub for sub in subs if not _may_bind(tree, sub.stem)]


def _import_findings(
    entrypoint: Path, base_dir: Path, *, check_sdk: bool
) -> tuple[list[str], dict[Path, ast.Module]]:
    """walk environment.py and the sibling modules it imports at top level."""
    findings: list[str] = []
    trees: dict[Path, ast.Module] = {}
    pending = [entrypoint]
    while pending and len(trees) < _MAX_LOCAL_MODULES:
        path = pending.pop()
        if path in trees:
            continue
        trees[path] = tree = _parse(path, base_dir)
        for node, module, names in _module_names(tree):
            local = _local_module_path(base_dir, module)
            if local is not None:
                pending.append(local)
                if isinstance(node, ast.ImportFrom) and local.name == "__init__.py":
                    pending.extend(_imported_submodules(local, names, base_dir))
                continue
            if not check_sdk or module.split(".", 1)[0] != _SDK_ROOT:
                continue
            error = _sdk_import_error(module, names)
            if error is not None:
                findings.append(f"{path.relative_to(base_dir)} line {node.lineno}: {error}")
    return findings, trees


class _ClassResolver:
    """resolve class bases in environment.py to local classes or Freesolo SDK classes."""

    def __init__(self, tree: ast.Module):
        self.classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        self.names: dict[str, tuple[str, str]] = {}
        self.modules: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for alias in node.names:
                    self.names[alias.asname or alias.name] = (node.module, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.modules[alias.asname] = alias.name
                    else:
                        root = alias.name.split(".", 1)[0]
                        self.modules[root] = root

    def _sdk_object(self, expr: ast.expr) -> object | None:
        parts: list[str] = []
        while isinstance(expr, ast.Attribute):
            parts.insert(0, expr.attr)
            expr = expr.value
        if not isinstance(expr, ast.Name):
            return None
        if expr.id in self.names and not parts:
            module, attr = self.names[expr.id]
        elif expr.id in self.modules and parts:
            module, attr = ".".join([self.modules[expr.id], *parts[:-1]]), parts[-1]
        else:
            return None
        if module.split(".", 1)[0] != _SDK_ROOT:
            return None
        try:
            return getattr(importlib.import_module(module), attr, None)
        except Exception:
            return None

    def chain(self, name: str) -> tuple[list[ast.ClassDef], list[type]] | None:
        """local classes and SDK bases behind ``name``, or None when any base is unknown."""
        local: list[ast.ClassDef] = []
        sdk: list[type] = []
        pending = [name]
        while pending:
            node = self.classes.get(pending.pop())
            if node is None or node in local or node.keywords or node.decorator_list:
                return None
            local.append(node)
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in self.classes:
                    pending.append(base.id)
                    continue
                resolved = self._sdk_object(base)
                if not isinstance(resolved, type):
                    return None
                sdk.append(resolved)
        return local, sdk


def _class_level_names(node: ast.ClassDef) -> set[str] | None:
    """names a class body binds, or None when a statement could bind names conditionally."""
    names: set[str] = set()
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            names.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
        elif not isinstance(stmt, (ast.Expr, ast.Pass)):
            return None
    return names


def _may_return(stmt: ast.stmt) -> bool:
    """whether ``stmt`` contains a ``return`` of the enclosing function (nested defs excluded)."""
    pending: list[ast.AST] = [stmt]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Return):
            return True
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            pending.extend(ast.iter_child_nodes(node))
    return False


def _local_call(expr: ast.expr | None, resolver: _ClassResolver) -> str | None:
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id in resolver.classes
    ):
        return expr.func.id
    return None


def _built_class(tree: ast.Module, resolver: _ClassResolver) -> str | None:
    """the local class ``load_environment()`` provably constructs and returns, else None.

    Only straight-line code counts: the first top-level ``return Cls(...)`` (or ``env =
    Cls(...)`` then ``return env``) with no earlier statement that could return instead.
    Anything conditional or unresolved abstains, since an unselected branch never runs.
    """
    func = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_environment"),
        None,
    )
    bound: dict[str, str] = {}
    for stmt in func.body if func is not None else []:
        if isinstance(stmt, ast.Return):
            if isinstance(stmt.value, ast.Name):
                return bound.get(stmt.value.id)
            return _local_call(stmt.value, resolver)
        if _may_return(stmt):
            return None
        if isinstance(stmt, ast.Assign):
            built = _local_call(stmt.value, resolver)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    bound.pop(target.id, None)
                    if built is not None and len(stmt.targets) == 1:
                        bound[target.id] = built
    return None


def _assigns_dataset_attribute(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == "dataset"
        and isinstance(node.ctx, ast.Store)
        for node in ast.walk(tree)
    )


def _attribute_binding(node: ast.ClassDef, attr: str) -> ast.stmt | None:
    """the last statement in a class body that binds ``attr`` (the one Python keeps)."""
    found = None
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [stmt.name]
        elif isinstance(stmt, ast.Assign):
            names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            names = [stmt.target.id] if isinstance(stmt.target, ast.Name) else []
        else:
            continue
        if attr in names:
            found = stmt
    return found


def _effective_binding(
    node: ast.ClassDef, resolver: _ClassResolver, attr: str
) -> tuple[ast.ClassDef, ast.stmt] | None:
    """the class-level binding of ``attr`` that Python's MRO resolves for ``node``, or None.

    Only a single local inheritance line is followed (the SDK bases bind neither ``dataset``
    nor ``image_observations``); multiple local bases or a conditionally bound class body
    abstain with None, as does an attribute no local class binds.
    """
    while _class_level_names(node) is not None:
        stmt = _attribute_binding(node, attr)
        if stmt is not None:
            return node, stmt
        bases = [b.id for b in node.bases if isinstance(b, ast.Name) and b.id in resolver.classes]
        if len(bases) != 1:
            return None
        node = resolver.classes[bases[0]]
    return None


def _declares_image_observations(node: ast.ClassDef, resolver: _ClassResolver) -> bool:
    binding = _effective_binding(node, resolver, "image_observations")
    if binding is None or not isinstance(binding[1], (ast.Assign, ast.AnnAssign)):
        return False
    value = binding[1].value
    return isinstance(value, ast.Constant) and value.value is True


def _dataset_method_findings(node: ast.ClassDef, resolver: _ClassResolver) -> list[str]:
    binding = _effective_binding(node, resolver, "dataset")
    if binding is None:
        return []
    owner, stmt = binding
    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) or stmt.decorator_list:
        return []
    message = (
        f"environment.py line {stmt.lineno}: {owner.name}.dataset is a method, but Flash reads "
        "`dataset` as the row sequence itself (iterating it raises TypeError: 'method' object "
        "is not iterable). Assign the rows instead, e.g. "
        "`self.dataset = load_task_examples(path)`, or decorate it with @property"
    )
    return [message]


def _class_findings(
    name: str,
    local: list[ast.ClassDef],
    sdk: list[type],
    resolver: _ClassResolver,
    *,
    skip_dataset_check: bool,
) -> list[str]:
    from freesolo.environments import EnvironmentMultiTurn, EnvironmentSingleTurn

    findings: list[str] = []
    bound = [_class_level_names(node) for node in local]
    defined = None if None in bound else set().union(*bound)
    sdk = list(dict.fromkeys(sdk))
    if len(sdk) == 1 and defined is not None:
        if not issubclass(sdk[0], (EnvironmentSingleTurn, EnvironmentMultiTurn)):
            findings.append(
                f"environment class {name!r} subclasses {sdk[0].__name__}; load_environment() "
                "must return a subclass of freesolo.environments.EnvironmentSingleTurn or "
                "EnvironmentMultiTurn"
            )
        missing = sorted(set(getattr(sdk[0], "__abstractmethods__", ())) - defined)
        if missing:
            findings.append(
                f"TypeError: Can't instantiate abstract class {name} without an implementation "
                f"for abstract method(s) {', '.join(repr(m) for m in missing)} (required by "
                f"{sdk[0].__name__})"
            )
    if not skip_dataset_check:
        findings.extend(_dataset_method_findings(local[0], resolver))
    return findings


def check_environment_package(
    entrypoint: Path, *, check_sdk_imports: bool = True, reads_env_dataset: bool = True
) -> StaticEnvironmentFacts:
    """raise EnvironmentPreflightError for load failures provable from the package source.

    ``reads_env_dataset`` is False when the run's explicit records or non-default dataset_path
    take precedence, so the worker never iterates the environment's own ``dataset``.
    """
    base_dir = entrypoint.parent
    findings, trees = _import_findings(entrypoint, base_dir, check_sdk=check_sdk_imports)
    tree = trees[entrypoint]
    facts = StaticEnvironmentFacts()
    # an import failure stops the module before any class exists, so report only that.
    check_classes = check_sdk_imports and not findings
    resolver = _ClassResolver(tree)
    skip_dataset_check = not reads_env_dataset or _assigns_dataset_attribute(tree)
    name = _built_class(tree, resolver)
    resolved = resolver.chain(name) if name is not None else None
    if resolved is not None and resolved[1]:
        local, sdk = resolved
        if _declares_image_observations(local[0], resolver):
            facts = StaticEnvironmentFacts(image_observations=True)
        if check_classes:
            findings.extend(
                _class_findings(name, local, sdk, resolver, skip_dataset_check=skip_dataset_check)
            )
    if findings:
        shown = findings[:_MAX_FINDINGS]
        more = len(findings) - len(shown)
        raise EnvironmentPreflightError(
            "environment package would fail to load on the training worker (checked before any "
            f"GPU is allocated, against freesolo {_sdk_version()}, the SDK the worker runs): "
            + "; ".join(shown)
            + (f"; and {more} more" if more > 0 else "")
            + ". Run `flash env test <env dir>` to load it exactly as the worker does."
        )
    return facts

"""Static import-closure walk for `flash env push`.

Packaging an environment means shipping the entrypoint plus everything it imports from
inside the environment root. That resolution is a self-contained AST walk: find the
imports (including `importlib.import_module("...")` calls made through a local alias),
follow them transitively, and yield the sidecar files that belong in the upload.

Split out of `flash.cli.commands.env.push` to keep that module under the file-size limit.
"""

from __future__ import annotations

import ast
import os
from collections.abc import Iterator
from pathlib import Path

from flash.cli.commands.env.ops.push import (
    _ENV_EVALUATIONS_SIDECAR,
    _decode_env_entrypoint_source,
    _ignore_env_push_path,
    _raise_walk_error,
)

# the functions that import a module named at runtime. one definition, because the binding walk
# and the call-site matcher below must agree on what counts as one.
_DYNAMIC_IMPORT_FUNCTIONS = frozenset({"import_module", "__import__"})


def _dynamic_import_callees(tree) -> frozenset[str]:
    """Return aliases that call ``importlib.import_module`` or ``__import__``.
    Follow plain and annotated assignment chains in either order. Unsupported binding forms remain
    excluded; missing an alias can omit a sibling and fail after publishing.
    """
    from collections import deque

    names = set(_DYNAMIC_IMPORT_FUNCTIONS)
    # names bound from another name, keyed by the name they read: `b = a` waits under "a" until
    # "a" is known to be an importer, whether that happens before or after this line was parsed.
    pending: dict[str, list[str]] = {}
    newly_known: deque[str] = deque()

    def learn(name: str) -> None:
        """Record a name as an importer, and queue it so anything waiting on it can resolve."""
        if name not in names:
            names.add(name)
            newly_known.append(name)

    def bind(targets, value) -> None:
        if isinstance(value, ast.Attribute):
            # `importlib.import_module`, under any alias of the module itself. Matched on the
            # canonical attribute name only, exactly as `_dynamic_import_name` matches the call
            # site, so the two cannot disagree about what counts as a dynamic import.
            if value.attr not in _DYNAMIC_IMPORT_FUNCTIONS:
                return
            source = None  # bound straight from the importer, so it needs nothing else first
        elif isinstance(value, ast.Name):
            source = value.id  # bound from another name, which may not be known yet
        else:
            return
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if source is None:
                learn(target.id)
            else:
                pending.setdefault(source, []).append(target.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    learn(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            bind(node.targets, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind([node.target], node.value)

    # seeded with the canonical names too, so `load = import_module` resolves like any other link.
    newly_known.extend(_DYNAMIC_IMPORT_FUNCTIONS)
    while newly_known:
        # each binding is released exactly once, when the name it reads becomes known, so the
        # whole walk costs one visit per binding however the chain is ordered in the source.
        for target_id in pending.pop(newly_known.popleft(), ()):
            learn(target_id)
    return frozenset(names)


def _dynamic_import_name(node, callees: frozenset[str]) -> str | None:
    """Return the module named by a literal dynamic import call, if any.

    Literal siblings must be packaged to avoid post-publish ModuleNotFoundError; computed names are
    left to runtime rather than guessed.
    """

    func = node.func
    if isinstance(func, ast.Attribute):
        # `importlib.import_module(...)`, under any alias of the module itself. Attribute access is
        # matched on the canonical name only: a module that happens to bind `load` at top level
        # does not make an unrelated `obj.load("x")` a dynamic import.
        matched = func.attr in _DYNAMIC_IMPORT_FUNCTIONS
    else:
        matched = getattr(func, "id", None) in callees
    if not matched:
        return None
    # both accept the module as the keyword `name`, and that form imports identically. reading only
    # the positional argument left `import_module(name="judge")` out of the archive, so the suite
    # that passed locally -- its directory is importable there -- raised ModuleNotFoundError on its
    # first published case.
    first = (
        node.args[0]
        if node.args
        else next((kw.value for kw in node.keywords if kw.arg == "name"), None)
    )
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return None
    # `import_module(".helper", package=...)` is relative; the sidecar directory is not a package,
    # so only an absolute name can name a sibling this push would carry.
    return first.value.split(".", 1)[0] or None


def _imported_module_names(tree, *, relative_names_are_siblings: bool = True) -> set[str]:
    """Return top-level sibling names imported by statement or literal call.

    Resolve relative names as root siblings only for files at the env root; nested package relatives
    are already handled by the package walk.
    """

    names: set[str] = set()
    callees = _dynamic_import_callees(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    names.add(node.module.split(".", 1)[0])
            elif node.level == 1 and relative_names_are_siblings:
                # level-one relative imports name siblings. include both ``module`` and bare
                # ``from . import x`` names so loose-module fallbacks are packaged.
                if node.module:
                    names.add(node.module.split(".", 1)[0])
                else:
                    names.update(alias.name.split(".", 1)[0] for alias in node.names)
            # level >= 2 climbs above the package this push carries, so it can name no sibling.
        elif isinstance(node, ast.Call):
            dynamic = _dynamic_import_name(node, callees)
            if dynamic:
                names.add(dynamic)
    return names


def _read_env_module_source(path: Path) -> str:
    """Read one packaged Python file under the interpreter's own encoding contract.

    the publish path decodes the entrypoint with `tokenize.detect_encoding`, so a file carrying a
    non-utf-8 encoding cookie is publishable. reading it as utf-8 here raises instead, and every
    caller below treats that as "no imports" -- so the sibling helpers never enter the archive and
    `env push` exits 0 with an environment that only fails once a gpu has been rented. one decoder
    for both paths is what keeps "publishable" and "walked for imports" the same set of files.
    """
    return _decode_env_entrypoint_source(path.read_bytes())[0]


def _helper_imports(helper: Path, *, env_root: Path) -> set[str]:
    """Return imports needed by a packaged helper.

    Ship unparseable helpers verbatim. Only root helpers resolve relative names against the env
    root; the package walk already carries nested siblings.
    """

    if helper.suffix != ".py":
        return set()
    try:
        return _imported_module_names(
            ast.parse(_read_env_module_source(helper), filename=str(helper)),
            relative_names_are_siblings=helper.parent == env_root,
        )
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()


def _iter_import_closure(
    seed_modules: set[str], *, env_root: Path, entrypoint: Path, yielded: set[Path]
) -> Iterator[tuple[Path, Path]]:
    """Yield imported sibling helpers and their transitive sibling imports.

    This is an import closure, not a general dependency resolver. Sharing ``yielded`` prevents
    duplicate members from being charged twice by ``_check_env_push_limits``.
    """

    pending = sorted(seed_modules)
    visited: set[str] = set()
    while pending:
        module_name = pending.pop()
        if module_name in visited:
            continue
        visited.add(module_name)
        # mirror python import precedence: a regular package beats a same-named module, while a bare
        # PEP 420 namespace directory is only a fallback. choosing the wrong side omits the imported
        # helper and fails after publishing.
        package = env_root / module_name
        helper = env_root / f"{module_name}.py"
        ships_package = (
            package.is_dir()
            and not package.is_symlink()
            and ((package / "__init__.py").is_file() or not helper.is_file())
        )
        if ships_package and not _ignore_env_push_path(
            package, env_root=env_root, entrypoint=entrypoint
        ):
            for root, dirs, files in os.walk(
                package, topdown=True, followlinks=False, onerror=_raise_walk_error
            ):
                root_path = Path(root)
                dirs[:] = sorted(
                    name
                    for name in dirs
                    if not _ignore_env_push_path(
                        root_path / name, env_root=env_root, entrypoint=entrypoint
                    )
                )
                for name in sorted(files):
                    child = root_path / name
                    if child in yielded or _ignore_env_push_path(
                        child, env_root=env_root, entrypoint=entrypoint
                    ):
                        continue
                    yielded.add(child)
                    pending.extend(_helper_imports(child, env_root=env_root))
                    yield child, child.relative_to(env_root)
            continue
        if (
            helper.is_file()
            and helper != entrypoint
            and helper not in yielded
            and not _ignore_env_push_path(helper, env_root=env_root, entrypoint=entrypoint)
        ):
            yielded.add(helper)
            pending.extend(_helper_imports(helper, env_root=env_root))
            yield helper, helper.relative_to(env_root)


def _iter_env_sidecar_files(
    env_root: Path, *, entrypoint: Path, include_full_tree: bool
) -> Iterator[tuple[Path, Path]]:
    """Yield publishable sidecar files and their package-relative paths."""

    roots = [env_root]
    # shared with the dataset walk below: the import closure can reach a helper package that IS
    # `dataset/` (or lives under it), and yielding it twice makes `_check_env_push_limits` charge
    # those bytes and members twice -- rejecting a tree that is actually under the limit. the copy
    # pass is idempotent, so only the limit check ever saw the double.
    yielded: set[Path] = set()
    if not include_full_tree:
        import ast

        # a single-file push uses bounded sidecar selections. root *.toml and recursive
        # configs/**/*.toml are handled explicitly below rather than widening to neighbouring trees.
        # ship user-authored docs so the synthesized stub readme does not replace real guidance.
        for doc_name in ("README.md", "TRAINING.md"):
            doc = env_root / doc_name
            if doc.is_file() and not _ignore_env_push_path(
                doc, env_root=env_root, entrypoint=entrypoint
            ):
                yielded.add(doc)
                yield doc, doc.relative_to(env_root)
        # root config files and the conventional configs tree are runtime inputs, not a request to
        # widen an exact-file push to unrelated neighbouring trees. apply the shared ignore policy
        # and yielded set so the limit and copy passes select the identical safe members.
        for config in sorted(env_root.iterdir()):
            if (
                config.is_file()
                and config.suffix.lower() == ".toml"
                and config not in yielded
                and not _ignore_env_push_path(config, env_root=env_root, entrypoint=entrypoint)
            ):
                yielded.add(config)
                yield config, config.relative_to(env_root)
        configs = env_root / "configs"
        if configs.is_dir() and not _ignore_env_push_path(
            configs, env_root=env_root, entrypoint=entrypoint
        ):
            for root, dirs, files in os.walk(
                configs, topdown=True, followlinks=False, onerror=_raise_walk_error
            ):
                root_path = Path(root)
                dirs[:] = sorted(
                    name
                    for name in dirs
                    if not _ignore_env_push_path(
                        root_path / name, env_root=env_root, entrypoint=entrypoint
                    )
                )
                for name in sorted(files):
                    config = root_path / name
                    if (
                        config.suffix.lower() != ".toml"
                        or config in yielded
                        or _ignore_env_push_path(config, env_root=env_root, entrypoint=entrypoint)
                    ):
                        continue
                    yielded.add(config)
                    yield config, config.relative_to(env_root)
        # the entrypoint's OWN imports ship, exactly as its evaluation sidecar's do. seeding the
        # closure only from evaluations.py published an environment whose entrypoint imported
        # siblings without them: `env push` exits 0 and prints an id, so nothing surfaces until a
        # GPU has been rented and the worker imports the module. parsed with the same reader as
        # every other module here, so a lazily imported helper (inside load_environment) counts.
        try:
            entry_tree = ast.parse(_read_env_module_source(entrypoint), filename=str(entrypoint))
        except (OSError, SyntaxError, UnicodeDecodeError):
            # the entrypoint is validated for syntax on its own path, and a push that cannot read
            # it fails there with a better message. an unreadable file here just means no closure.
            entry_tree = None
        if entry_tree is not None:
            yield from _iter_import_closure(
                _imported_module_names(entry_tree),
                env_root=env_root,
                entrypoint=entrypoint,
                yielded=yielded,
            )
        # the eval sidecar ships with its entrypoint. `env eval` loads evaluations.py next to an
        # exact .py target, so omitting it here would publish an environment whose suite passed
        # locally and is simply absent once pushed.
        sidecar = env_root / _ENV_EVALUATIONS_SIDECAR
        if (
            sidecar.is_file()
            and sidecar != entrypoint
            and not _ignore_env_push_path(sidecar, env_root=env_root, entrypoint=entrypoint)
        ):
            # the entrypoint closure above shares `yielded` and runs first, so an entrypoint that
            # imports `evaluations` already shipped it. yielding it a second time made
            # `_check_env_push_limits` charge those bytes and that member twice, rejecting a tree
            # that is actually under the limit. only the yield is skipped: the syntax check and
            # import walk below must still run either way.
            if sidecar not in yielded:
                yielded.add(sidecar)
                yield sidecar, sidecar.relative_to(env_root)
            try:
                tree = ast.parse(_read_env_module_source(sidecar), filename=str(sidecar))
            except SyntaxError as exc:
                raise ValueError(
                    f"{sidecar}: invalid evaluation sidecar syntax: {exc.msg}"
                ) from exc
            # walk the whole tree, not just tree.body: the loader deliberately keeps the package
            # dir on sys.path so a sidecar can import its helpers lazily inside cases() or score()
            # (flash/envs/meta/evaluations.py). scanning only top-level nodes skips exactly that
            # pattern, so `env test` passes locally and the pushed environment is missing the
            # helper the sidecar imports the moment it grades a case.
            yield from _iter_import_closure(
                _imported_module_names(tree),
                env_root=env_root,
                entrypoint=entrypoint,
                yielded=yielded,
            )
        roots = [
            env_root / name
            for name in ("dataset", "datasets")
            if (env_root / name).is_dir() and not (env_root / name).is_symlink()
        ]

    for walk_root in roots:
        for root, dirs, files in os.walk(
            walk_root, topdown=True, followlinks=False, onerror=_raise_walk_error
        ):
            root_path = Path(root)
            dirs[:] = sorted(
                name
                for name in dirs
                if not _ignore_env_push_path(
                    root_path / name, env_root=env_root, entrypoint=entrypoint
                )
            )
            for name in sorted(files):
                child = root_path / name
                if child in yielded or _ignore_env_push_path(
                    child, env_root=env_root, entrypoint=entrypoint
                ):
                    continue
                yielded.add(child)
                yield child, child.relative_to(env_root)

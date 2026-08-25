"""Enforce the tracked ``flash/`` source layout contract.

Every directory may contain at most ten tracked files. A directory with tracked subdirectories may
contain only semantic package markers, never implementation files. Reading Git's index instead of
the working tree keeps generated caches, ignored files, and local build artifacts out of the decision.

Usage: python scripts/check_source_layout.py [root]
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

FILE_MAX = 10
GATED_PACKAGE = PurePosixPath("flash")
PACKAGE_MARKERS = frozenset({"__init__.py", "__main__.py", "py.typed"})


def tracked_source_files(root: Path) -> tuple[PurePosixPath, ...]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", str(GATED_PACKAGE)],
        check=True,
        capture_output=True,
    )
    return tuple(PurePosixPath(value.decode()) for value in result.stdout.split(b"\0") if value)


def structural_violations(
    files: tuple[PurePosixPath, ...],
) -> tuple[list[tuple[PurePosixPath, int]], list[tuple[PurePosixPath, tuple[str, ...]]]]:
    files_by_directory: dict[PurePosixPath, list[str]] = defaultdict(list)
    subdirectories: dict[PurePosixPath, set[str]] = defaultdict(set)
    directories = {GATED_PACKAGE}

    for path in files:
        if not path.is_relative_to(GATED_PACKAGE):
            continue
        files_by_directory[path.parent].append(path.name)
        parent = path.parent
        directories.add(parent)
        while parent != GATED_PACKAGE:
            child = parent
            parent = parent.parent
            subdirectories[parent].add(child.name)
            directories.add(parent)

    oversized = sorted(
        (directory, len(files_by_directory[directory]))
        for directory in directories
        if len(files_by_directory[directory]) > FILE_MAX
    )
    mixed = sorted(
        (
            directory,
            tuple(
                sorted(
                    name for name in files_by_directory[directory] if name not in PACKAGE_MARKERS
                )
            ),
        )
        for directory in directories
        if subdirectories[directory]
        and any(name not in PACKAGE_MARKERS for name in files_by_directory[directory])
    )
    return oversized, mixed


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_import(node: ast.stmt) -> bool:
    return isinstance(node, (ast.Import, ast.ImportFrom))


def _has_postponed_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _is_safe_annotation(node: ast.expr, *, postponed: bool) -> bool:
    if postponed:
        return True
    return isinstance(node, ast.Name) or (
        isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def _is_simple_metadata(node: ast.stmt, *, postponed_annotations: bool) -> bool:
    if isinstance(node, ast.Assign):
        if not all(isinstance(target, ast.Name) for target in node.targets):
            return False
        value = node.value
    elif isinstance(node, ast.AnnAssign):
        if not isinstance(node.target, ast.Name) or not _is_safe_annotation(
            node.annotation, postponed=postponed_annotations
        ):
            return False
        value = node.value
    else:
        return False
    if value is None:
        return True
    try:
        ast.literal_eval(value)
    except (ValueError, TypeError):
        return False
    return True


def _is_import_error_type(node: ast.expr | None) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {"ImportError", "ModuleNotFoundError"}
    if isinstance(node, ast.Tuple):
        return bool(node.elts) and all(_is_import_error_type(item) for item in node.elts)
    return False


def _is_import_fallback(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Try) or node.finalbody or not node.handlers:
        return False
    if not all(_is_import_error_type(handler.type) for handler in node.handlers):
        return False
    blocks = [node.body, node.orelse, *(handler.body for handler in node.handlers)]
    return all(
        all(_is_import(item) or isinstance(item, ast.Pass) for item in block) for block in blocks
    )


def _contains_definition(tree: ast.AST) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )


def _is_name_guard(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _is_imported_call(node: ast.expr, imported: set[str]) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in imported
        and not node.args
        and not node.keywords
    )


def _is_imported_main_call(
    node: ast.stmt, imported: set[str], *, system_exit_shadowed: bool
) -> bool:
    if isinstance(node, ast.Expr):
        return _is_imported_call(node.value, imported)
    if system_exit_shadowed or not isinstance(node, ast.Raise) or node.cause is not None:
        return False
    exc = node.exc
    return (
        isinstance(exc, ast.Call)
        and isinstance(exc.func, ast.Name)
        and exc.func.id == "SystemExit"
        and len(exc.args) == 1
        and not exc.keywords
        and _is_imported_call(exc.args[0], imported)
    )


def marker_violation(path: PurePosixPath, source: str) -> str | None:
    if path.name == "py.typed":
        if not source.strip() or source in {"partial", "partial\n"}:
            return None
        return "py.typed must be empty or contain exactly 'partial'"
    if path.name not in {"__init__.py", "__main__.py"}:
        return None
    try:
        compile(source, str(path), "exec", dont_inherit=True)
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return f"cannot parse marker: {exc.msg} at line {exc.lineno}"
    if _contains_definition(tree):
        return "marker contains a function, async function, or class definition"
    if path.name == "__init__.py":
        postponed_annotations = _has_postponed_annotations(tree)
        allowed = (_is_docstring, _is_import, _is_import_fallback)
        if all(
            any(check(node) for check in allowed)
            or _is_simple_metadata(node, postponed_annotations=postponed_annotations)
            for node in tree.body
        ):
            return None
        return "__init__.py contains executable implementation"

    imported: set[str] = set()
    system_exit_shadowed = False
    body = list(tree.body)
    if body and _is_docstring(body[0]):
        body.pop(0)
    while body and _is_import(body[0]):
        node = body.pop(0)
        local_names = {alias.asname or alias.name.split(".")[0] for alias in node.names}
        system_exit_shadowed |= "SystemExit" in local_names
        if isinstance(node, ast.ImportFrom):
            system_exit_shadowed |= "*" in local_names
            imported.update(local_names)
    if len(body) != 1 or not isinstance(body[0], ast.If) or not _is_name_guard(body[0].test):
        return "__main__.py must contain only imports and one __name__ launcher guard"
    guard = body[0]
    if (
        guard.orelse
        or len(guard.body) != 1
        or not _is_imported_main_call(
            guard.body[0], imported, system_exit_shadowed=system_exit_shadowed
        )
    ):
        return "__main__.py launcher must invoke an imported callable or raise SystemExit from it"
    return None


def semantic_marker_violations(
    root: Path, files: tuple[PurePosixPath, ...]
) -> list[tuple[PurePosixPath, str]]:
    tracked = set(files)
    directories_with_children = {
        path.parent
        for path in files
        for other in files
        if other.parent != path.parent and other.is_relative_to(path.parent)
    }
    out = []
    for path in sorted(tracked):
        if path.name not in PACKAGE_MARKERS or path.parent not in directories_with_children:
            continue
        violation = marker_violation(path, (root / path).read_text())
        if violation:
            out.append((path, violation))
    return out


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    files = tracked_source_files(root)
    if not files:
        print("no tracked source files under flash/")
        return 1
    oversized, mixed = structural_violations(files)
    markers = semantic_marker_violations(root, files)
    if not oversized and not mixed and not markers:
        return 0

    if oversized:
        print(f"{len(oversized)} source director(ies) exceed the {FILE_MAX}-file limit:")
        for directory, count in oversized:
            print(f"  {count:>3}  {directory}")
    if mixed:
        if oversized:
            print()
        print(f"{len(mixed)} source director(ies) mix subdirectories with implementation files:")
        for directory, names in mixed:
            print(f"  {directory}: {', '.join(names)}")
    if markers:
        if oversized or mixed:
            print()
        print(f"{len(markers)} package marker(s) contain implementation:")
        for path, detail in markers:
            print(f"  {path}: {detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

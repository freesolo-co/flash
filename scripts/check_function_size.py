"""Fail if any production function is longer than the function-size limit.

Ruff's own length rules are opt-in per-construct and do not cover plain functions, so this runs
alongside `ruff check` in CI. Only `flash/` is gated, matching `check_file_size.py`.

The file-size gate alone does not catch this: a 700-line function sits comfortably inside a
900-line module, so the file passes while the function nobody can hold in their head does not.
Length is the proxy -- a function past the limit has stopped being one thing, and its branches
stop being reviewed because reaching them means paging in everything above.

The fix is to extract a cohesive phase into a helper, never to reflow or to merge lines. When the
enclosing module has no room left under the file-size limit, the helper belongs in a sibling
module that the parent re-exports.

nested functions are measured independently and excluded from their owner's count. nested class
bodies remain charged to the owner, while methods are independently measured and excluded.

Usage: python scripts/check_function_size.py [root]
"""

from __future__ import annotations

import ast
import os
import sys

FUNCTION_MAX = 150
GATED_PACKAGE = "flash"
SKIP_DIRS = {"__pycache__", ".git", ".venv", "build", ".ruff_cache"}


def _walk_defs(node: ast.AST, prefix: str = ""):
    """Yield (qualified_name, def_node) for every function defined under `node`."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            yield from _walk_defs(child, f"{prefix}{child.name}.")
        elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            qualname = f"{prefix}{child.name}"
            yield qualname, child
            yield from _walk_defs(child, f"{qualname}.")
        else:
            yield from _walk_defs(child, prefix)


def _nested_definition_spans(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[int, int]]:
    """nested function spans owned by ``node``.

    a nested function is excluded as one span and checked independently. class bodies remain charged
    to the owner, while traversal continues through them so their methods are found and excluded.
    """
    spans: list[tuple[int, int]] = []
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            start = min(
                (decorator.lineno for decorator in child.decorator_list), default=child.lineno
            )
            spans.append((start, child.end_lineno))
        else:
            stack.extend(ast.iter_child_nodes(child))
    return spans


def _length(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """lines owned by this function, excluding nested function spans.

    decorators are excluded because ``node.lineno`` points at the ``def``.
    """
    total = node.end_lineno - node.lineno + 1
    return total - sum(end - start + 1 for start, end in _nested_definition_spans(node))


def _all_oversized(root: str) -> list[tuple[int, str, str, int]]:
    """Every production function over the exclusive line limit."""
    found = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, GATED_PACKAGE)):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding="utf-8") as f:
                source = f.read()
            # a syntax error is ruff's to report, not this gate's; skipping keeps one failure mode
            # from being reported twice with different wording.
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            rel = os.path.relpath(path, root)
            for qualname, node in _walk_defs(tree):
                lines = _length(node)
                if lines > FUNCTION_MAX:
                    found.append((lines, rel, qualname, node.lineno))
    found.sort(reverse=True)
    return found


def oversized(root: str) -> list[tuple[int, str, str, int]]:
    return _all_oversized(root)


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    found = oversized(root)
    if not found:
        return 0
    print(f"{len(found)} function(s) over the {FUNCTION_MAX}-line limit:")
    for lines, rel, qualname, lineno in found:
        print(f"  {lines:>5}  {rel}:{lineno}  {qualname}")
    print(
        "\nExtract a cohesive phase into a helper; put it in a sibling module when the file is full."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

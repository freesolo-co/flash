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

Nested definitions are measured on their own as well as inside their parent, so a fat closure is
reported in its own right; shortening it shortens both.

Usage: python scripts/check_function_size.py [root]
"""

from __future__ import annotations

import ast
import os
import sys

FUNCTION_MAX = 150
GATED_PACKAGE = "flash"
SKIP_DIRS = {"__pycache__", ".git", ".venv", "build", ".ruff_cache"}

# Functions whose source is extracted and shipped elsewhere to run standalone, so a helper they
# call is simply absent at the far end. Splitting one of these passes every local test and then
# raises NameError on the worker, which is why they are listed rather than left to a judgement
# call. Each entry needs the mechanism that ships it, so the exemption can be rechecked when that
# mechanism changes. Keep this list short: it is an escape hatch for a transport boundary, never
# for a function that is merely hard to split.
SOURCE_SHIPPED = {
    # `docker/make_rp_handler.py` AST-extracts this at image build time and writes it into
    # `/rp_handler.py`, which the worker image runs as its CMD with nothing else from flash on the
    # path. `tests/test_flash_worker.py` pins the same contract from the other side: every name it
    # uses must be imported inside its own body.
    "flash/providers/runpod/serverless/endpoints.py::_train_body",
}


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


def _length(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Lines from `def` through the final body line.

    Decorators are excluded: `node.lineno` points at the `def`, so a stack of decorators does not
    count against the body they wrap.
    """
    return node.end_lineno - node.lineno + 1


def _all_oversized(root: str) -> list[tuple[int, str, str, int]]:
    """Every function over the limit, before SOURCE_SHIPPED is applied."""
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
    return [
        entry for entry in _all_oversized(root) if f"{entry[1]}::{entry[2]}" not in SOURCE_SHIPPED
    ]


def stale_exemptions(root: str) -> list[str]:
    """Entries in SOURCE_SHIPPED that no longer name an oversized function.

    An exemption list rots silently: the function gets split, renamed, or deleted, and the entry
    stays behind exempting nothing. Reporting those keeps the list honest, so it can only shrink
    by being noticed.
    """
    live = set()
    for _lines, rel, qualname, _lineno in _all_oversized(root):
        live.add(f"{rel}::{qualname}")
    return sorted(SOURCE_SHIPPED - live)


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    stale = stale_exemptions(root)
    found = oversized(root)
    if stale:
        print(f"{len(stale)} stale entr(ies) in SOURCE_SHIPPED -- no longer oversized:")
        for entry in stale:
            print(f"    {entry}")
        print("\nRemove the entry: the exemption is no longer carrying anything.")
        return 1
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

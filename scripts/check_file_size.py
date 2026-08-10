"""Fail if any production module is longer than the file-size limit.

Ruff has no file-length rule, so this runs alongside `ruff check` in CI. Only `flash/` is gated:
the test suite has its own long files and they are deliberately out of scope here.

A module over the limit is not a style problem, it is a review problem -- nobody reads a
2000-line file end to end, so its invariants stop being checked. The fix is always to move a
cohesive group of definitions into a sibling module, never to reflow the code that is there.

Usage: python scripts/check_file_size.py [root]
"""

from __future__ import annotations

import os
import sys

FILE_MAX = 1000
GATED_PACKAGE = "flash"
SKIP_DIRS = {"__pycache__", ".git", ".venv", "build", ".ruff_cache"}


def oversized(root: str) -> list[tuple[int, str]]:
    found = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, GATED_PACKAGE)):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding="utf-8") as f:
                lines = sum(1 for _ in f)
            if lines > FILE_MAX:
                found.append((lines, os.path.relpath(path, root)))
    found.sort(reverse=True)
    return found


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    found = oversized(root)
    if not found:
        return 0
    print(f"{len(found)} file(s) over the {FILE_MAX}-line limit:")
    for lines, rel in found:
        print(f"  {lines:>5}  {rel}")
    print("\nSplit a cohesive group of definitions into a sibling module and re-export it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

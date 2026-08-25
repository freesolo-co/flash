"""Enforce the tracked ``flash/`` source layout contract.

Every directory may contain at most ten tracked files. A directory with tracked subdirectories may
contain only Python package markers, never implementation files. Reading Git's index instead of the
working tree keeps generated caches, ignored files, and local build artifacts out of the decision.

Usage: python scripts/check_source_layout.py [root]
"""

from __future__ import annotations

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


def violations(
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


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    oversized, mixed = violations(tracked_source_files(root))
    if not oversized and not mixed:
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

import subprocess
import sys
from pathlib import PurePosixPath

from scripts.check_source_layout import (
    PACKAGE_MARKERS,
    marker_violation,
    structural_violations,
    tracked_source_files,
)


def _paths(*values: str) -> tuple[PurePosixPath, ...]:
    return tuple(PurePosixPath(value) for value in values)


def test_ten_files_in_one_directory_are_allowed() -> None:
    files = tuple(PurePosixPath(f"flash/group/module_{index}.py") for index in range(10))

    assert structural_violations(files) == ([], [])


def test_eleven_files_in_one_directory_are_rejected() -> None:
    files = tuple(PurePosixPath(f"flash/group/module_{index}.py") for index in range(11))

    oversized, mixed = structural_violations(files)

    assert oversized == [(PurePosixPath("flash/group"), 11)]
    assert mixed == []


def test_package_markers_can_share_a_directory_with_subdirectories() -> None:
    files = _paths(
        *(f"flash/package/{marker}" for marker in sorted(PACKAGE_MARKERS)),
        "flash/package/child/module.py",
    )

    assert structural_violations(files) == ([], [])


def test_implementation_file_cannot_share_a_directory_with_subdirectories() -> None:
    files = _paths(
        "flash/package/__init__.py",
        "flash/package/runtime.py",
        "flash/package/child/module.py",
    )

    oversized, mixed = structural_violations(files)

    assert oversized == []
    assert mixed == [(PurePosixPath("flash/package"), ("runtime.py",))]


def test_paths_outside_flash_do_not_affect_the_contract() -> None:
    files = _paths(
        *(f"tests/test_{index}.py" for index in range(20)),
        "flash/package/module.py",
    )

    assert structural_violations(files) == ([], [])


def test_checker_reads_only_git_tracked_files(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    package = tmp_path / "flash" / "package"
    package.mkdir(parents=True)
    tracked = package / "tracked.py"
    tracked.write_text("")
    subprocess.run(["git", "-C", str(tmp_path), "add", "flash/package/tracked.py"], check=True)
    (package / "ignored.py").write_text("")
    (tmp_path / ".gitignore").write_text("flash/package/ignored.py\n")

    assert tracked_source_files(tmp_path) == (PurePosixPath("flash/package/tracked.py"),)


def test_init_rejects_hidden_implementation() -> None:
    detail = marker_violation(
        PurePosixPath("flash/package/__init__.py"), "def hidden():\n    pass\n"
    )

    assert detail == "marker contains a function, async function, or class definition"


def test_main_rejects_non_thin_launcher() -> None:
    detail = marker_violation(
        PurePosixPath("flash/package/__main__.py"),
        "from .cli import main\nvalue = prepare()\nif __name__ == '__main__':\n    main()\n",
    )

    assert detail == "__main__.py must contain only imports and one __name__ launcher guard"


def test_main_allows_thin_system_exit_launcher() -> None:
    detail = marker_violation(
        PurePosixPath("flash/package/__main__.py"),
        '"""launcher."""\nfrom .cli import main\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n',
    )

    assert detail is None


def test_main_rejects_raising_the_imported_callable_directly() -> None:
    detail = marker_violation(
        PurePosixPath("flash/package/__main__.py"),
        'from .cli import main\n\nif __name__ == "__main__":\n    raise main()\n',
    )

    assert (
        detail
        == "__main__.py launcher must invoke an imported callable or raise SystemExit from it"
    )


def test_main_rejects_calling_an_imported_module() -> None:
    detail = marker_violation(
        PurePosixPath("flash/package/__main__.py"),
        'import flash.cli.parsing.main as main\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n',
    )

    assert (
        detail
        == "__main__.py launcher must invoke an imported callable or raise SystemExit from it"
    )


def test_main_rejects_shadowed_system_exit() -> None:
    detail = marker_violation(
        PurePosixPath("flash/package/__main__.py"),
        "from helpers import SystemExit, main\n\n"
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    )

    assert (
        detail
        == "__main__.py launcher must invoke an imported callable or raise SystemExit from it"
    )


def test_main_rejects_wildcard_import_before_system_exit() -> None:
    detail = marker_violation(
        PurePosixPath("flash/package/__main__.py"),
        "from helpers import *\nfrom .cli import main\n\n"
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    )

    assert (
        detail
        == "__main__.py launcher must invoke an imported callable or raise SystemExit from it"
    )


def test_py_typed_rejects_unrecognized_content() -> None:
    detail = marker_violation(PurePosixPath("flash/package/py.typed"), "complete\n")

    assert detail == "py.typed must be empty or contain exactly 'partial'"


def test_py_typed_rejects_whitespace_padded_partial() -> None:
    detail = marker_violation(PurePosixPath("flash/package/py.typed"), "  partial  \n")

    assert detail == "py.typed must be empty or contain exactly 'partial'"


def test_init_rejects_executable_annotation_without_future_annotations() -> None:
    detail = marker_violation(
        PurePosixPath("flash/package/__init__.py"),
        "from attacker import Hook\nVALUE: Hook[42] = 1\n",
    )

    assert detail == "__init__.py contains executable implementation"


def test_init_allows_deferred_metadata_annotation() -> None:
    detail = marker_violation(
        PurePosixPath("flash/package/__init__.py"),
        "from __future__ import annotations\nVALUE: dict[str, int] = {}\n",
    )

    assert detail is None


def test_init_rejects_misplaced_future_annotations_import() -> None:
    detail = marker_violation(
        PurePosixPath("flash/package/__init__.py"),
        "VALUE = 1\nfrom __future__ import annotations\nOTHER: Hook[42] = 1\n",
    )

    assert (
        detail
        == "cannot parse marker: from __future__ imports must occur at the beginning of the file at line 2"
    )


def test_init_rejects_executable_import_fallback_handler() -> None:
    detail = marker_violation(
        PurePosixPath("flash/package/__init__.py"),
        "try:\n    import optional\nexcept build_handler():\n    pass\n",
    )

    assert detail == "__init__.py contains executable implementation"


def test_init_allows_static_import_error_fallback() -> None:
    detail = marker_violation(
        PurePosixPath("flash/package/__init__.py"),
        "try:\n    import optional\nexcept (ImportError, ModuleNotFoundError):\n    pass\n",
    )

    assert detail is None


def test_marker_parse_failure_is_reported() -> None:
    detail = marker_violation(PurePosixPath("flash/package/__init__.py"), "if:\n")

    assert detail == "cannot parse marker: invalid syntax at line 1"


def test_checker_rejects_empty_tracked_source_set(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)

    result = subprocess.run(
        [sys.executable, "scripts/check_source_layout.py", str(tmp_path)],
        cwd=PurePosixPath(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == "no tracked source files under flash/\n"

import subprocess
from pathlib import PurePosixPath

from scripts.check_source_layout import PACKAGE_MARKERS, tracked_source_files, violations


def _paths(*values: str) -> tuple[PurePosixPath, ...]:
    return tuple(PurePosixPath(value) for value in values)


def test_ten_files_in_one_directory_are_allowed() -> None:
    files = tuple(PurePosixPath(f"flash/group/module_{index}.py") for index in range(10))

    assert violations(files) == ([], [])


def test_eleven_files_in_one_directory_are_rejected() -> None:
    files = tuple(PurePosixPath(f"flash/group/module_{index}.py") for index in range(11))

    oversized, mixed = violations(files)

    assert oversized == [(PurePosixPath("flash/group"), 11)]
    assert mixed == []


def test_package_markers_can_share_a_directory_with_subdirectories() -> None:
    files = _paths(
        *(f"flash/package/{marker}" for marker in sorted(PACKAGE_MARKERS)),
        "flash/package/child/module.py",
    )

    assert violations(files) == ([], [])


def test_implementation_file_cannot_share_a_directory_with_subdirectories() -> None:
    files = _paths(
        "flash/package/__init__.py",
        "flash/package/runtime.py",
        "flash/package/child/module.py",
    )

    oversized, mixed = violations(files)

    assert oversized == []
    assert mixed == [(PurePosixPath("flash/package"), ("runtime.py",))]


def test_paths_outside_flash_do_not_affect_the_contract() -> None:
    files = _paths(
        *(f"tests/test_{index}.py" for index in range(20)),
        "flash/package/module.py",
    )

    assert violations(files) == ([], [])


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

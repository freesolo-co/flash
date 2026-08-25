"""Hermetic rollback coverage for environment destination replacement helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import flash.envs.loading.pull as pull


def _source(tmp_path: Path, files: dict[str, str]) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for name, text in files.items():
        (source / name).write_text(text)
    return source


def test_cwd_inside_returns_false_when_path_resolution_fails(monkeypatch, tmp_path) -> None:
    """Destination safety probing must fail closed without crashing on filesystem resolution errors."""
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(Path, "resolve", lambda self: (_ for _ in ()).throw(OSError("gone")))

    assert pull._cwd_is_inside(tmp_path) is False


def test_populate_empty_dir_rejects_destination_changes_after_staging(
    tmp_path, monkeypatch
) -> None:
    """A concurrent destination writer must abort population and leave its file untouched."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    real_copytree = pull.shutil.copytree

    def copytree(src, staging):
        result = real_copytree(src, staging)
        (dest / "concurrent.txt").write_text("keep")
        return result

    monkeypatch.setattr(pull.shutil, "copytree", copytree)

    with pytest.raises(FileExistsError, match="already exists"):
        pull._populate_empty_dir(source, dest)

    assert (dest / "concurrent.txt").read_text() == "keep"
    assert not (dest / "environment.py").exists()


def test_populate_empty_dir_rolls_back_files_moved_before_a_conflict(tmp_path, monkeypatch) -> None:
    """A mid-move conflict must remove already-moved files while preserving the conflicting writer."""
    source = _source(tmp_path, {"a.txt": "a", "b.txt": "b"})
    dest = tmp_path / "dest"
    dest.mkdir()
    real_replace = os.replace
    calls = []

    def replace(src, target):
        calls.append(Path(target).name)
        if Path(target).name == "a.txt":
            real_replace(src, target)
            (dest / "b.txt").write_text("concurrent")
            return
        real_replace(src, target)

    monkeypatch.setattr(pull.os, "replace", replace)

    with pytest.raises(FileExistsError, match=r"b\.txt"):
        pull._populate_empty_dir(source, dest)

    assert calls == ["a.txt"]
    assert not (dest / "a.txt").exists()
    assert (dest / "b.txt").read_text() == "concurrent"


def test_replace_with_tree_restores_original_after_replace_failure(tmp_path, monkeypatch) -> None:
    """A failed staged replacement must restore the original destination exactly."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    real_replace = os.replace
    calls = 0

    def replace(src, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("replacement denied")
        real_replace(src, target)

    monkeypatch.setattr(pull.os, "replace", replace)

    with pytest.raises(OSError, match="replacement denied"):
        pull._replace_with_tree(source, dest)

    assert (dest / "old.txt").read_text() == "old"
    assert not (dest / "environment.py").exists()


def test_replace_with_tree_retains_backup_when_restore_also_fails(tmp_path, monkeypatch) -> None:
    """A double replacement failure must retain the original tree in the documented backup path."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    real_replace = os.replace
    calls = 0

    def replace(src, target):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError(f"replace failure {calls}")
        real_replace(src, target)

    monkeypatch.setattr(pull.os, "replace", replace)

    with pytest.raises(RuntimeError, match="original retained at"):
        pull._replace_with_tree(source, dest)

    backups = list(tmp_path.glob(".flash-env-pull-*/dest.old/old.txt"))
    assert len(backups) == 1
    assert backups[0].read_text() == "old"
    assert not dest.exists()


def test_replace_with_tree_installs_a_new_destination(tmp_path) -> None:
    """A missing destination must receive the staged source without creating a backup."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "new-dest"

    pull._replace_with_tree(source, dest)

    assert (dest / "environment.py").read_text() == "new"


def test_copy_environment_source_refuses_current_working_directory(tmp_path, monkeypatch) -> None:
    """The copy helper must retain its own current-directory guard even after preflight."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    monkeypatch.setattr(pull, "ensure_environment_pull_destination_available", lambda *a, **k: dest)
    monkeypatch.setattr(pull, "_cwd_is_inside", lambda path: True)

    with pytest.raises(RuntimeError, match="current working directory"):
        pull._copy_environment_source(source, dest, overwrite=True)

    assert (dest / "old.txt").read_text() == "old"


def test_copy_environment_source_rejects_existing_destination_without_overwrite(
    tmp_path, monkeypatch
) -> None:
    """The copy helper must not replace an occupied destination unless overwrite is explicit."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.write_text("old")
    monkeypatch.setattr(pull, "ensure_environment_pull_destination_available", lambda *a, **k: dest)

    with pytest.raises(FileExistsError, match="already exists"):
        pull._copy_environment_source(source, dest, overwrite=False)

    assert dest.read_text() == "old"

"""privacy-safe packaged serving startup instrumentation."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.serve.app import progress


def _root_usage(monkeypatch) -> None:
    monkeypatch.setattr(
        progress.shutil,
        "disk_usage",
        lambda target: SimpleNamespace(
            total=10 * progress._MIB + 17,
            used=4 * progress._MIB + 23,
            free=6 * progress._MIB - 6,
        ),
    )


def test_filesystem_usage_emits_root_mib_and_nested_cache_logical_bytes(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cache_root = tmp_path / "private-customer-cache-sentinel"
    nested = cache_root / "models" / "nested"
    nested.mkdir(parents=True)
    (cache_root / "manifest.json").write_bytes(b"manifest")
    (nested / "weights.bin").write_bytes(b"weight-bytes")
    _root_usage(monkeypatch)
    monkeypatch.setattr(progress, "_STARTED_AT", 10.0)
    monkeypatch.setattr(progress.time, "perf_counter", lambda: 11.25)

    assert progress.boot_elapsed_seconds() == 1.25
    progress.emit_filesystem_usage("cache-prepared", cache_root)

    output = capsys.readouterr().out
    assert output == (
        "flash-serving boot elapsed=1.250s phase=filesystem-usage "
        'stage="cache-prepared" root_total_mib="10" root_used_mib="4" '
        'root_free_mib="5" cache_logical_bytes="20"\n'
    )
    assert str(cache_root) not in output


def test_cache_tree_does_not_follow_symlinks_outside_root(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cache_root = tmp_path / "private-customer-cache-sentinel"
    cache_root.mkdir()
    (cache_root / "inside.bin").write_bytes(b"inside")
    outside = tmp_path / "outside-secret-sentinel"
    outside.mkdir()
    (outside / "large-secret.bin").write_bytes(b"secret" * 100)
    (cache_root / "outside-file").symlink_to(outside / "large-secret.bin")
    (cache_root / "outside-directory").symlink_to(outside, target_is_directory=True)
    _root_usage(monkeypatch)

    progress.emit_filesystem_usage("engine-constructed", cache_root)

    output = capsys.readouterr().out
    assert 'cache_logical_bytes="6"' in output
    assert str(cache_root) not in output
    assert str(outside) not in output
    assert "large-secret.bin" not in output


def test_cache_tree_does_not_double_count_hard_links(monkeypatch, tmp_path: Path, capsys) -> None:
    cache_root = tmp_path / "private-customer-cache-sentinel"
    cache_root.mkdir()
    original = cache_root / "weights.bin"
    original.write_bytes(b"weight-bytes")
    os.link(original, cache_root / "weights-copy.bin")
    _root_usage(monkeypatch)

    progress.emit_filesystem_usage("serving-ready", cache_root)

    output = capsys.readouterr().out
    assert 'cache_logical_bytes="12"' in output
    assert str(cache_root) not in output


def test_cache_scan_closes_directory_fds_on_unexpected_recursive_failure(monkeypatch) -> None:
    root_fd = 10
    child_fd = 11
    closed: list[int] = []

    class ChildDirectory:
        name = "child"

        def stat(self, *, follow_symlinks: bool):
            assert follow_symlinks is False
            return SimpleNamespace(st_mode=stat.S_IFDIR)

    class RootEntries:
        def __enter__(self):
            return iter((ChildDirectory(),))

        def __exit__(self, *_args):
            return None

    def open_directory(path, flags, *, dir_fd=None):
        expected_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        assert flags == expected_flags
        if dir_fd is None:
            assert path == "/cache"
            return root_fd
        assert path == "child"
        assert dir_fd == root_fd
        return child_fd

    def scan_directory(directory_fd):
        if directory_fd == root_fd:
            return RootEntries()
        assert directory_fd == child_fd
        raise RuntimeError("recursive failure sentinel")

    monkeypatch.setattr(progress.os, "open", open_directory)
    monkeypatch.setattr(progress.os, "scandir", scan_directory)
    monkeypatch.setattr(progress.os, "close", closed.append)

    with pytest.raises(RuntimeError, match="recursive failure sentinel"):
        progress._cache_tree_logical_bytes("/cache")

    assert closed == [child_fd, root_fd]


def test_cache_scan_oserror_is_partial_and_does_not_leak_or_fail_startup(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cache_root = tmp_path / "private-customer-cache-sentinel"
    cache_root.mkdir()
    secret_error = f"secret failure for {cache_root / 'vanished.bin'}"

    class VanishedEntry:
        name = "vanished.bin"

        def stat(self, *, follow_symlinks: bool):
            assert follow_symlinks is False
            raise OSError(secret_error)

    class Entries:
        def __enter__(self):
            return iter((VanishedEntry(),))

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(progress.os, "scandir", lambda _directory_fd: Entries())
    _root_usage(monkeypatch)

    progress.emit_filesystem_usage("serving-ready", cache_root)

    output = capsys.readouterr().out
    assert 'cache_logical_bytes="0" cache_status="partial"' in output
    assert str(cache_root) not in output
    assert "vanished.bin" not in output
    assert "secret failure" not in output


def test_filesystem_usage_reports_only_allowlisted_unavailable_status(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cache_root = tmp_path / "private-customer-cache-sentinel"
    secret_error = f"secret failure for {cache_root}"

    def unavailable(*_args, **_kwargs):
        raise OSError(secret_error)

    monkeypatch.setattr(progress.shutil, "disk_usage", unavailable)
    monkeypatch.setattr(progress.os, "open", unavailable)
    monkeypatch.setattr(progress, "_STARTED_AT", 20.0)
    monkeypatch.setattr(progress.time, "perf_counter", lambda: 20.5)

    progress.emit_filesystem_usage("serving-ready", cache_root)

    output = capsys.readouterr().out
    assert output == (
        "flash-serving boot elapsed=0.500s phase=filesystem-usage "
        'stage="serving-ready" root_status="unavailable" cache_status="unavailable"\n'
    )
    assert str(cache_root) not in output
    assert "secret failure" not in output

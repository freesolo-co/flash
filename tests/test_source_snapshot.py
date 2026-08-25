from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import pytest

import flash.snapshot.archive as source_snapshot
from flash.snapshot.archive import (
    SourceSnapshotDescriptor,
    SourceSnapshotError,
    attempt_materialization_path,
    build_source_archive,
    descriptor_for_archive,
    materialize_verified_archive_file,
    parse_descriptor,
    read_verified_archive,
)

REVISION = "a" * 40


def _package(tmp_path: Path) -> Path:
    package = tmp_path / "flash"
    package.mkdir()
    (package / "__init__.py").write_text("value = 1\n")
    (package / "worker.py").write_text("def run():\n    return 1\n")
    hidden = package / ".cache"
    hidden.mkdir()
    (hidden / "ignored.py").write_text("ignored = True\n")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ignored")
    return package


def _archive(
    entries: list[tuple[str, bytes, int | None]],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression, compresslevel=9) as target:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (mode if mode is not None else stat.S_IFREG | 0o644) << 16
            info.compress_type = compression
            target.writestr(info, data)
    return output.getvalue()


def _payload(archive: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        return {info.filename: source.read(info) for info in source.infolist()}


def test_default_archive_root_is_the_flash_package() -> None:
    payload = _payload(build_source_archive())
    assert "flash/__init__.py" in payload
    assert "flash/cli/__init__.py" in payload
    assert "flash/providers/__init__.py" in payload
    assert "flash/snapshot/archive.py" in payload
    assert len(payload) > 100


def test_archive_is_deterministic_and_contains_only_canonical_package_files(tmp_path: Path) -> None:
    package = _package(tmp_path)
    first = build_source_archive(package_dir=package)
    second = build_source_archive(package_dir=package)
    assert first == second

    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        infos = archive.infolist()
        assert [info.filename for info in infos] == ["flash/__init__.py", "flash/worker.py"]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in infos)
        assert all(stat.S_ISREG(info.external_attr >> 16) for info in infos)
    assert read_verified_archive(first, descriptor_for_archive(first, REVISION)) == {
        "flash/__init__.py": b"value = 1\n",
        "flash/worker.py": b"def run():\n    return 1\n",
    }


@pytest.mark.parametrize("mutation", ["same_length", "truncated"])
def test_external_digest_rejects_byte_mutation(tmp_path: Path, mutation: str) -> None:
    archive = build_source_archive(package_dir=_package(tmp_path))
    descriptor = descriptor_for_archive(archive, REVISION)
    if mutation == "same_length":
        changed = bytearray(archive)
        changed[len(changed) // 2] ^= 1
        sabotaged = bytes(changed)
    else:
        sabotaged = archive[:-1]
    with pytest.raises(SourceSnapshotError):
        read_verified_archive(sabotaged, descriptor)


def test_empty_archive_is_rejected() -> None:
    archive = _archive([])
    with pytest.raises(SourceSnapshotError, match="no members"):
        read_verified_archive(archive, descriptor_for_archive(archive, REVISION))


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("../escape.py", None),
        ("/absolute.py", None),
        ("flash\\alias.py", None),
        ("outside.py", None),
        ("flash/dir/", stat.S_IFDIR | 0o755),
        ("flash/link.py", stat.S_IFLNK | 0o777),
        ("flash/device", stat.S_IFCHR | 0o600),
    ],
)
def test_unsafe_outside_directory_symlink_and_nonregular_members_are_rejected(
    name: str,
    mode: int | None,
) -> None:
    archive = _archive([("flash/worker.py", b"ok", None), (name, b"target", mode)])
    with pytest.raises(SourceSnapshotError):
        read_verified_archive(archive, descriptor_for_archive(archive, REVISION))


def test_duplicate_archive_member_is_rejected() -> None:
    archive = _archive(
        [
            ("flash/worker.py", b"first", None),
            ("flash/worker.py", b"second", None),
        ]
    )
    with pytest.raises(SourceSnapshotError, match="duplicate"):
        read_verified_archive(archive, descriptor_for_archive(archive, REVISION))


def test_member_crc_is_verified() -> None:
    archive = _archive(
        [("flash/worker.py", b"payload", None)],
        compression=zipfile.ZIP_STORED,
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        info = source.infolist()[0]
        data_offset = info.header_offset + len(info.FileHeader())
    corrupted = bytearray(archive)
    corrupted[data_offset] ^= 1
    corrupted_archive = bytes(corrupted)
    with pytest.raises(SourceSnapshotError, match="readable zip"):
        read_verified_archive(
            corrupted_archive,
            descriptor_for_archive(corrupted_archive, REVISION),
        )


def test_typed_descriptors_use_strict_validation() -> None:
    invalid = SourceSnapshotDescriptor(
        archive_path="source/not-the-digest/flash-source.zip",
        sha256="a" * 64,
        size=123,
        revision="b" * 40,
    )
    with pytest.raises(SourceSnapshotError, match="archive path"):
        parse_descriptor(invalid)


def test_materialization_path_binds_run_and_attempt() -> None:
    first = attempt_materialization_path("/runcode", "run-a", 0)
    second = attempt_materialization_path("/runcode", "run-b", 0)
    assert first == Path("/runcode/run-a-attempt-0")
    assert second == Path("/runcode/run-b-attempt-0")
    assert first != second

    for run_id in ("../escape", "/absolute", "run/alias", ""):
        with pytest.raises(SourceSnapshotError, match="run_id"):
            attempt_materialization_path("/runcode", run_id, 0)
    for attempt in (True, -1, "0"):
        with pytest.raises(SourceSnapshotError, match="attempt"):
            attempt_materialization_path("/runcode", "run-a", attempt)


def test_file_materialization_verifies_once_and_leaves_no_partial_tree(
    monkeypatch,
    tmp_path: Path,
) -> None:
    archive = build_source_archive(package_dir=_package(tmp_path))
    descriptor = descriptor_for_archive(archive, REVISION)
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(archive)
    destination = tmp_path / "materialized"
    verify_calls = 0
    original_verify = source_snapshot.read_verified_archive

    def counted_verify(data, parsed_descriptor):
        nonlocal verify_calls
        verify_calls += 1
        return original_verify(data, parsed_descriptor)

    monkeypatch.setattr(source_snapshot, "read_verified_archive", counted_verify)
    materialize_verified_archive_file(archive_path, descriptor, destination)
    assert verify_calls == 1
    assert (destination / "flash" / "worker.py").read_text().startswith("def run")

    broken_path = tmp_path / "broken.zip"
    broken_path.write_bytes(archive + b"extra")
    broken_destination = tmp_path / "broken"
    with pytest.raises(SourceSnapshotError, match="size mismatch"):
        materialize_verified_archive_file(broken_path, descriptor, broken_destination)
    assert not broken_destination.exists()
    assert not list(tmp_path.glob(".flash-source-*"))

from __future__ import annotations

import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path

import pytest

from flash.source_snapshot import (
    MANIFEST_NAME,
    SourceSnapshotError,
    attempt_materialization_path,
    build_source_archive,
    descriptor_for_archive,
    materialize_verified_archive,
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


def _payload(archive: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        return {info.filename: source.read(info.filename) for info in source.infolist()}


def _archive(entries: list[tuple[str, bytes, int | None]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (mode if mode is not None else stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(info, data)
    return output.getvalue()


def _repack(payload: dict[str, bytes]) -> bytes:
    return _archive([(name, data, None) for name, data in sorted(payload.items())])


def test_archive_is_deterministic_and_manifest_is_canonical(tmp_path: Path) -> None:
    package = _package(tmp_path)
    first = build_source_archive(package_dir=package)
    second = build_source_archive(package_dir=package)
    assert first == second

    payload = _payload(first)
    manifest = json.loads(payload[MANIFEST_NAME])
    assert [member["path"] for member in manifest["members"]] == [
        "flash/__init__.py",
        "flash/worker.py",
    ]
    assert payload[MANIFEST_NAME].endswith(b"\n")


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


def test_member_set_rejects_missing_and_extra(tmp_path: Path) -> None:
    archive = build_source_archive(package_dir=_package(tmp_path))
    payload = _payload(archive)
    descriptor = descriptor_for_archive(archive, REVISION)

    missing = dict(payload)
    missing.pop("flash/worker.py")
    missing_archive = _repack(missing)
    with pytest.raises(SourceSnapshotError):
        read_verified_archive(
            missing_archive,
            descriptor_for_archive(missing_archive, REVISION),
        )

    extra = dict(payload)
    extra["flash/extra.py"] = b"extra = True\n"
    extra_archive = _repack(extra)
    with pytest.raises(SourceSnapshotError):
        read_verified_archive(extra_archive, descriptor_for_archive(extra_archive, REVISION))

    assert descriptor.sha256 == hashlib.sha256(archive).hexdigest()


def test_consistent_archive_and_manifest_replacement_fails_external_digest(tmp_path: Path) -> None:
    archive = build_source_archive(package_dir=_package(tmp_path))
    descriptor = descriptor_for_archive(archive, REVISION)
    payload = _payload(archive)
    replacement = b"value = 2\n"
    payload["flash/__init__.py"] = replacement
    manifest = json.loads(payload[MANIFEST_NAME])
    for member in manifest["members"]:
        if member["path"] == "flash/__init__.py":
            member["size"] = len(replacement)
            member["sha256"] = hashlib.sha256(replacement).hexdigest()
    payload[MANIFEST_NAME] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    replaced = _repack(payload)
    with pytest.raises(SourceSnapshotError, match="digest mismatch"):
        read_verified_archive(replaced, descriptor)


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("../escape.py", None),
        ("/absolute.py", None),
        ("flash\\alias.py", None),
        ("flash/dir/", stat.S_IFDIR | 0o755),
        ("flash/link.py", stat.S_IFLNK | 0o777),
    ],
)
def test_unsafe_directory_and_symlink_members_are_rejected(
    tmp_path: Path, name: str, mode: int | None
) -> None:
    archive = build_source_archive(package_dir=_package(tmp_path))
    payload = _payload(archive)
    entries = [(member, data, None) for member, data in sorted(payload.items())]
    entries.append((name, b"target", mode))
    sabotaged = _archive(entries)
    with pytest.raises(SourceSnapshotError):
        read_verified_archive(sabotaged, descriptor_for_archive(sabotaged, REVISION))


def test_duplicate_archive_member_is_rejected(tmp_path: Path) -> None:
    archive = build_source_archive(package_dir=_package(tmp_path))
    payload = _payload(archive)
    entries = [(member, data, None) for member, data in sorted(payload.items())]
    entries.append(("flash/worker.py", payload["flash/worker.py"], None))
    duplicate = _archive(entries)
    with pytest.raises(SourceSnapshotError, match="duplicate"):
        read_verified_archive(duplicate, descriptor_for_archive(duplicate, REVISION))


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


def test_materialization_is_atomic_and_leaves_no_partial_tree(tmp_path: Path) -> None:
    archive = build_source_archive(package_dir=_package(tmp_path))
    descriptor = descriptor_for_archive(archive, REVISION)
    destination = tmp_path / "materialized"
    materialize_verified_archive(archive, descriptor, destination)
    assert (destination / "flash" / "worker.py").read_text().startswith("def run")

    broken_destination = tmp_path / "broken"
    with pytest.raises(SourceSnapshotError):
        materialize_verified_archive(archive[:-1], descriptor, broken_destination)
    assert not broken_destination.exists()
    assert not list(tmp_path.glob(".flash-source-*"))

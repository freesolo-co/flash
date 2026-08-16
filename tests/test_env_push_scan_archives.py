"""Archive member expansion in the `flash env push` credential scan.

A deflated zip member or a tar entry does not contain its credential anywhere in the archive's own
bytes, so scanning the container literally cannot see it. These cover the walk that opens each
member, the bounds that stop a hostile archive, and the controls that keep an ordinary dataset
publishable.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import tarfile
import zipfile

from flash.envscan.secrets import _Unscannable, credential_in_file

FREESOLO_KEY = "fslo_" + "A1bCdEfGhIjKlMnOpQrS"
SECRET_LINE = f"export FREESOLO_API_KEY={FREESOLO_KEY}"


def _zip(tmp_path, name: str, members: dict[str, bytes]):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for member, payload in members.items():
            archive.writestr(member, payload)
    path = tmp_path / name
    path.write_bytes(buffer.getvalue())
    return path


def _tar(tmp_path, name: str, members: dict[str, bytes]):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for member, payload in members.items():
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    path = tmp_path / name
    path.write_bytes(buffer.getvalue())
    return path


def test_a_zip_member_is_expanded(tmp_path):
    path = _zip(tmp_path, "bundle.zip", {"env.sh": SECRET_LINE.encode()})
    assert credential_in_file(path) == "a Freesolo API key"


def test_a_tar_member_is_expanded(tmp_path):
    path = _tar(tmp_path, "bundle.tar", {"env.sh": SECRET_LINE.encode()})
    assert credential_in_file(path) == "a Freesolo API key"


def test_a_nested_container_is_reached(tmp_path):
    """Stopping at one level treated an inner member's bytes as final content, so a zip holding a
    gzipped shard -- an ordinary way to ship a dataset -- hid a key one layer further in."""
    path = _zip(tmp_path, "nested.zip", {"shard.gz": gzip.compress(SECRET_LINE.encode())})
    assert credential_in_file(path) == "a Freesolo API key"


def test_a_member_name_is_scanned(tmp_path):
    """A file whose name is the key leaks it through the member list even when its contents are
    empty, and the published repo shows that name in its tree forever."""
    path = _zip(tmp_path, "named.zip", {f"{FREESOLO_KEY}.txt": b""})
    assert credential_in_file(path) == "a Freesolo API key"


def test_a_later_member_is_still_reached(tmp_path):
    """Each member is guarded separately. Wrapping the whole loop meant one unreadable entry
    abandoned every entry behind it, so a bad member at the top hid a real key further down."""
    path = _zip(
        tmp_path,
        "many.zip",
        {"a.txt": b"ordinary", "b.bin": b"\x00\x01\x02", "env.sh": SECRET_LINE.encode()},
    )
    assert credential_in_file(path) == "a Freesolo API key"


def test_a_clean_archive_still_publishes(tmp_path):
    path = _zip(tmp_path, "clean.zip", {"train.py": b"import torch\n", "data.txt": b"rows\n"})
    assert credential_in_file(path) is None


def test_a_clean_tar_still_publishes(tmp_path):
    path = _tar(tmp_path, "clean.tar", {"train.py": b"import torch\n"})
    assert credential_in_file(path) is None


def test_a_corrupt_archive_does_not_crash_the_publish(tmp_path):
    """A half-written shard in a dataset directory is ordinary, and crashing on it would be a worse
    bug than the hole being closed."""
    path = tmp_path / "truncated.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("env.sh", "ordinary content")
    path.write_bytes(buffer.getvalue()[:20])
    # refusing is acceptable; crashing the publish is not
    with contextlib.suppress(_Unscannable):
        credential_in_file(path)


def test_an_unreadable_tar_remainder_does_not_crash_the_publish(tmp_path, monkeypatch):
    """A concatenated tar's remainder is read separately, and that read can fail.

    `_read_at` answers None when it cannot read, which the zip prefix and suffix paths both guard
    against. The tar remainder path called `.startswith` on it directly, so an i/o failure raised
    `AttributeError` out of a validation check and aborted the publish, where every other
    unreadable region is a refusal.
    """
    from flash.envscan import archive as env_archive

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("a.txt")
        info.size = 5
        archive.addfile(info, io.BytesIO(b"hello"))
    path = tmp_path / "concat.tar"
    path.write_bytes(buffer.getvalue() + b"TRAILING" * 128)

    real = env_archive._read_at

    def flaky(source, start, size):
        return None if start > 0 else real(source, start, size)

    monkeypatch.setattr(env_archive, "_read_at", flaky)
    # refusing is acceptable; an AttributeError out of a validation check is not
    with contextlib.suppress(_Unscannable):
        credential_in_file(path)

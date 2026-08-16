"""Structurally recognized containers the scan cannot expand, refused rather than published.

Parquet, HDF5, Arrow IPC, Git packs and RPMs store their values in framed or columnar layouts, so a
credential inside one appears nowhere in the file as contiguous bytes. None has a standard-library
reader, and adding PyArrow or h5py would put a compiled dependency in the base CLI. Recognizing
them structurally and refusing is the honest answer: unverifiable is not clean.

The refusals must be structural, never name-based, or an ordinary file with a misleading extension
becomes unpublishable.
"""

from __future__ import annotations

import io
import shutil
import subprocess

import pytest

from flash.envscan.opaque import _BUNDLE_PREREQUISITE, opaque_format
from flash.envscan.secrets import _Unscannable, credential_in_file

FREESOLO_KEY = "fslo_" + "A1bCdEfGhIjKlMnOpQrS"


def _rpm_lead(name: bytes = b"example-1.0-1.x86_64") -> bytes:
    """A structurally valid 96-byte RPM lead."""
    lead = bytearray(96)
    lead[0:4] = b"\xed\xab\xee\xdb"
    lead[4:6] = b"\x03\x00"
    lead[6:8] = (0).to_bytes(2, "big")
    lead[8:10] = (1).to_bytes(2, "big")
    lead[10:76] = name.ljust(66, b"\x00")
    lead[76:78] = (1).to_bytes(2, "big")
    lead[78:80] = (5).to_bytes(2, "big")
    return bytes(lead)


@pytest.mark.parametrize(
    ("name", "data", "expected"),
    [
        ("shard.parquet", b"PAR1" + b"\x00" * 64 + b"PAR1", "Parquet"),
        ("store.h5", b"\x89HDF\r\n\x1a\n" + b"\x00" * 128, "HDF5"),
        ("pkg.rpm", _rpm_lead() + b"\x00" * 128, "RPM"),
    ],
)
def test_a_recognised_opaque_container_is_refused(tmp_path, name, data, expected):
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises(_Unscannable) as caught:
        credential_in_file(path)
    assert expected in str(caught.value)


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("notes.parquet", b"ordinary notes, not a parquet file\n"),
        ("notes.h5", b"just ordinary notes about hdf5\n"),
        ("notes.rpm", b"a changelog mentioning rpm packaging\n"),
        ("readme.md", b"we store data in ARROW1 format sometimes\n"),
        ("short.bin", b"\x89HDF" + b"not really hdf5\n"),
        ("truncated.rpm", _rpm_lead()[:40]),
    ],
)
def test_a_name_or_partial_signature_alone_does_not_refuse(tmp_path, name, data):
    """The refusal is structural. Naming a file `.parquet`, or a chance short-magic collision, is
    not evidence, and refusing on it would make ordinary files unpublishable."""
    path = tmp_path / name
    path.write_bytes(data)
    assert credential_in_file(path) is None


def test_a_credential_beside_an_invalid_signature_is_still_found(tmp_path):
    """A file that only looks like an opaque container still gets the literal scan."""
    path = tmp_path / "notes.h5"
    path.write_bytes(b"\x89HDF not a real superblock\n" + FREESOLO_KEY.encode())
    assert credential_in_file(path) == "a Freesolo API key"


def test_the_detector_reports_the_format_it_proved():
    """`opaque_format` owns the formats that need structural validation to recognise at all.

    Parquet is not one of them: its anchored `PAR1` magic is decided by the format tables, so it is
    already refused without this module. What lives here is the set whose signature alone is not
    proof - HDF5 superblock offsets, Arrow framing, pack checksums, the RPM lead.
    """
    assert opaque_format(b"\x89HDF\r\n\x1a\n" + b"\x00" * 128) == "HDF5"
    assert opaque_format(_rpm_lead() + b"\x00" * 128) == "RPM"
    assert opaque_format(b"ordinary bytes") is None


@pytest.mark.parametrize("column", ["large_string", "large_binary"])
def test_arrow_types_above_the_original_bound_are_recognised(tmp_path, column):
    """The Arrow `Type` union runs to 26, not 18.

    `LargeBinary` (19) and `LargeUtf8` (20) are what pandas and polars emit for ordinary wide
    columns. Rejecting their discriminant made a structurally valid Arrow file stop matching as
    opaque, so the refusal never fired and columnar values stayed invisible to the literal scan.

    Flat columns only, deliberately: a nested type's footer takes a different validation path that
    this finding does not reach, so including one would prove something other than the bound.
    """
    pa = pytest.importorskip("pyarrow", reason="arrow fixtures need the writer")
    arrow_type = {"large_string": pa.large_string(), "large_binary": pa.large_binary()}[column]
    # ordinary values, no credential: what is under test is that the file is recognized as a
    # layout the scan cannot read. a key placed in the column would be found by the literal scan
    # anyway, since Arrow stores short strings inline, and would pass whatever the bound was.
    values = ["ordinary", "values"] if column == "large_string" else [b"ordinary", b"values"]

    batch = pa.record_batch([pa.array(values, type=arrow_type)], names=["value"])
    sink = io.BytesIO()
    with pa.ipc.new_file(sink, batch.schema) as writer:
        writer.write_batch(batch)

    assert opaque_format(sink.getvalue()) == "Arrow IPC"

    path = tmp_path / "shard.arrow"
    path.write_bytes(sink.getvalue())
    with pytest.raises(_Unscannable) as caught:
        credential_in_file(path)
    assert "Arrow IPC" in str(caught.value)


def _git_bundle(tmp_path, *args: str) -> bytes:
    """A bundle written by git itself, not hand-assembled to match the parser."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*command: str) -> None:
        subprocess.run(command, cwd=repo, check=True, capture_output=True)

    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.invalid")
    run("git", "config", "user.name", "Test")
    (repo / "a.txt").write_text("hello\n")
    run("git", "add", "a.txt")
    run("git", "commit", "-q", "-m", "init")
    out = tmp_path / "out.bundle"
    run("git", "bundle", "create", str(out), *args)
    return out.read_bytes()


@pytest.mark.parametrize("selector", [["--all"], ["HEAD"]])
def test_a_real_git_bundle_is_recognised(tmp_path, selector):
    """`git bundle create --all` writes a bare `HEAD` line beside the `refs/...` ones.

    Per gitformat-bundle a reference is `obj-id SP refname`, and refname is unqualified, so
    requiring a `refs/` prefix made an ordinary bundle stop matching. It then fell through to the
    literal scan, where a pack's contents are deflated and a credential inside is not contiguous.
    """
    if shutil.which("git") is None:
        pytest.skip("git is needed to write a real bundle")
    assert opaque_format(_git_bundle(tmp_path, *selector)) == "Git bundle"


def test_a_bundle_prerequisite_without_a_comment_is_accepted():
    """`prerequisite = "-" obj-id SP comment` with `comment = *CHAR`, so the comment may be
    empty. Demanding a non-empty one rejected a valid header the same way."""
    oid = b"3c49ea908b80928b8b72559f408aacaa2ce399d3"
    assert _BUNDLE_PREREQUISITE.fullmatch(b"-" + oid) is not None
    assert _BUNDLE_PREREQUISITE.fullmatch(b"-" + oid + b" ") is not None
    assert _BUNDLE_PREREQUISITE.fullmatch(b"-" + oid + b" a comment") is not None


def test_bundle_prose_is_not_a_bundle(tmp_path):
    """The recognition stays structural: a document quoting a bundle header is not one."""
    path = tmp_path / "notes.md"
    path.write_bytes(b"a bundle line looks like 3c49ea90 refs/heads/main\n")
    assert credential_in_file(path) is None

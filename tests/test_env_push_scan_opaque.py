"""Structurally recognized containers the scan cannot expand, refused rather than published.

Parquet, HDF5, Arrow IPC, Git packs and RPMs store their values in framed or columnar layouts, so a
credential inside one appears nowhere in the file as contiguous bytes. None has a standard-library
reader, and adding PyArrow or h5py would put a compiled dependency in the base CLI. Recognizing
them structurally and refusing is the honest answer: unverifiable is not clean.

The refusals must be structural, never name-based, or an ordinary file with a misleading extension
becomes unpublishable.
"""

from __future__ import annotations

import pytest

from flash.env_opaque import opaque_format
from flash.env_secrets import _Unscannable, credential_in_file

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

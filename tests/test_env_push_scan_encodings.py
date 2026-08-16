"""Reconstruction of credentials that are present but not contiguous.

A key written as base64, in UTF-16, split across adjacent source literals, or wrapped over several
lines is fully present in the file, yet it matches nothing when the bytes are read literally. These
cover the decodings that rebuild the value, and the controls that stop the rebuild from inventing
one that was never there.

The controls carry most of the weight here. Resolving an escape a syntax does not interpret, or
joining lines that only happen to be the same width, turns ordinary prose into an unpublishable
file.
"""

from __future__ import annotations

import base64
import gzip

import pytest

from flash.env_secrets import credential_in_file

FREESOLO_KEY = "fslo_" + "A1bCdEfGhIjKlMnOpQrS"
ASSIGNMENT = f"FREESOLO_API_KEY={FREESOLO_KEY}"


def _write(tmp_path, name: str, data: bytes | str):
    path = tmp_path / name
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


def test_a_base64_value_is_decoded(tmp_path):
    path = _write(tmp_path, "config.txt", base64.b64encode(ASSIGNMENT.encode()))
    assert credential_in_file(path) == "a Freesolo API key"


def test_a_base64url_encoded_compressed_value_is_decoded(tmp_path):
    """Encoding chains compose. Stopping after one decode published the key at two layers."""
    path = _write(
        tmp_path, "blob.txt", base64.urlsafe_b64encode(gzip.compress(ASSIGNMENT.encode()))
    )
    assert credential_in_file(path) == "a Freesolo API key"


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_a_wide_encoded_value_is_read(tmp_path, encoding):
    """A PowerShell profile is ordinarily UTF-16. Every byte of the key is present, separated by
    the encoding's nul padding, so a literal scan sees nothing."""
    path = _write(tmp_path, "profile.ps1", f"$env:{ASSIGNMENT}".encode(encoding))
    assert credential_in_file(path) == "a Freesolo API key"


def test_adjacent_python_literals_are_joined(tmp_path):
    path = _write(tmp_path, "settings.py", f'KEY = "fslo_" "{FREESOLO_KEY[5:]}"\n')
    assert credential_in_file(path) == "a Freesolo API key"


def test_a_shell_continuation_is_joined(tmp_path):
    path = _write(tmp_path, "setup.sh", f"export KEY=fslo_\\\n{FREESOLO_KEY[5:]}\n")
    assert credential_in_file(path) == "a Freesolo API key"


def test_a_python_escape_is_resolved(tmp_path):
    """Python interprets `\\x66` as `f`, so the literal bytes never spell the prefix."""
    path = _write(tmp_path, "settings.py", f'KEY = "\\x66slo_{FREESOLO_KEY[5:]}"\n')
    assert credential_in_file(path) == "a Freesolo API key"


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("secret.yml", "secret: |\n  {wrapped}\n"),
        ("secret.yaml", "token: {first}\n    {rest}\n"),
    ],
)
def test_a_wrapped_base64_value_is_joined(tmp_path, name, body):
    """A long value wrapped to a column is one value, and each line alone decodes to nothing."""
    encoded = base64.b64encode(ASSIGNMENT.encode()).decode()
    parts = [encoded[at : at + 20] for at in range(0, len(encoded), 20)]
    text = body.format(wrapped="\n  ".join(parts), first=parts[0], rest="\n    ".join(parts[1:]))
    assert credential_in_file(_write(tmp_path, name, text)) == "a Freesolo API key"


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("note.yml", r"note: 'a \x66slo_ literal backslash'" + "\n"),
        ("note.toml", "note = '''\nnot \\x66slo_ a key\n'''\n"),
    ],
)
def test_an_uninterpreted_escape_is_not_resolved(tmp_path, name, body):
    """YAML single quotes and TOML literal strings never interpret escapes, so resolving one here
    invents a credential the file does not contain and blocks an ordinary publish."""
    assert credential_in_file(_write(tmp_path, name, body)) is None


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("notes.txt", "\n".join(["aGVsbG8gdGhlcmUgZnJp"] * 4) + "\n"),
        ("readme.md", "\n".join(["the quick brown fox ju"] * 4) + "\n"),
    ],
)
def test_same_width_lines_alone_are_not_joined(tmp_path, name, body):
    """Joining is anchored on syntax that proves the lines are one value. Width alone is not
    evidence, and treating it as evidence rewrites ordinary prose before scanning it."""
    assert credential_in_file(_write(tmp_path, name, body)) is None


def test_ordinary_encoded_binary_still_publishes(tmp_path):
    """An embedded binary value is ordinary and must stay publishable. What is decoded is scanned
    like any other content, so ordinary bytes decode to ordinary bytes and the publish proceeds."""
    payload = base64.b64encode(bytes(range(256)) * 8)
    assert credential_in_file(_write(tmp_path, "weights.txt", payload)) is None


def test_an_encoded_container_is_carried_into_the_container_scan(tmp_path):
    """Decoding is not the end of the walk. A base64 value that decodes to a container is handed
    to the container path, which reaches its contents or refuses. Treating the decode itself as
    the answer published every key one layer inside an encoded archive."""
    payload = base64.b64encode(gzip.compress(ASSIGNMENT.encode()))
    assert credential_in_file(_write(tmp_path, "shard.b64", payload)) == "a Freesolo API key"


def test_prose_naming_the_variable_still_publishes(tmp_path):
    assert credential_in_file(_write(tmp_path, "README.md", "set FREESOLO_API_KEY\n")) is None

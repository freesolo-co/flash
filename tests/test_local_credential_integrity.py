"""Credential updates must preserve the old file and the exact supplied secret."""

import json
import os
import stat

import pytest

from flash._internal import fileio
from flash.client.runtime_secrets import runtime_secrets_from_local_env


def test_failed_serialization_preserves_saved_login(tmp_path):
    path = tmp_path / "config.json"
    original = '{"api_key": "previous-login"}'
    path.write_text(original)

    with pytest.raises(TypeError):
        fileio.secure_json_write(path, {"api_key": "new-login", "invalid": object()})

    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]


def test_failed_flush_preserves_saved_login(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    original = '{"api_key": "previous-login"}'
    path.write_text(original)

    def fail_sync(_fd):
        raise OSError("storage unavailable")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="storage unavailable"):
        fileio.secure_json_write(path, {"api_key": "new-login"})

    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]


def test_replacing_permissive_file_never_writes_new_secret_into_old_inode(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"api_key": "previous-login"}')
    path.chmod(0o644)
    with path.open() as original_reader:
        fileio.secure_json_write(path, {"api_key": "new-login"})
        assert "new-login" not in original_reader.read()
    assert json.loads(path.read_text()) == {"api_key": "new-login"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_credential_write_refuses_symlink_without_touching_target(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("original")
    path = tmp_path / "config.json"
    path.symlink_to(target)
    with pytest.raises(OSError, match="credential symlink"):
        fileio.secure_json_write(path, {"api_key": "new-login"})
    assert path.is_symlink()
    assert target.read_text() == "original"


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ('"token\'"', "token'"),
        ("'\"token\"'", '"token"'),
        ("token'", "token'"),
        ('token"', 'token"'),
        ("\"'token'\"", "'token'"),
        ('" token "', " token "),
        ("plain-token", "plain-token"),
    ],
)
def test_env_file_preserves_secret_contents(tmp_path, monkeypatch, literal, expected):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIT_SECRET", raising=False)
    (tmp_path / ".env").write_text(f"AUDIT_SECRET={literal}\n")
    assert runtime_secrets_from_local_env(keys=["AUDIT_SECRET"])["AUDIT_SECRET"] == expected

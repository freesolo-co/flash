"""Large native checkpoint transport and pinned-revision recovery evidence."""

import json
import shutil
from pathlib import Path

import pytest

from flash.engine.support import checkpoint_transport as transport


@pytest.fixture
def staged(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "PART_BYTES", 7)
    source = tmp_path / "source"
    (source / "actor").mkdir(parents=True)
    (source / "actor/model.pt").write_bytes(b"0123456789abcdefghijklmn")
    (source / "state.json").write_bytes(b"{}")
    return source


def test_transport_roundtrip_preserves_native_bytes_and_removes_staging(staged, tmp_path):
    with transport.upload_directory(str(staged)) as upload:
        path = Path(upload)
        assert (path / "state.json").stat().st_ino == (staged / "state.json").stat().st_ino
        manifest = transport.parse_manifest((path / transport.MANIFEST).read_bytes())
        assert len(manifest["actor/model.pt"]["parts"]) == 4
        assert all(
            p.stat().st_size <= 7 for p in (path / transport.PARTS).rglob("*") if p.is_file()
        )
        restored = tmp_path / "restored"
        shutil.copytree(upload, restored)
    assert not path.exists()
    transport.restore_directory(str(restored))
    assert (restored / "actor/model.pt").read_bytes() == (staged / "actor/model.pt").read_bytes()
    assert (restored / "state.json").read_bytes() == b"{}"
    assert not (restored / transport.PARTS).exists()
    transport.restore_directory(str(restored))  # native checkpoints remain readable


@pytest.mark.parametrize("corruption", ["checksum", "missing", "traversal", "duplicate", "length"])
def test_corrupt_transport_never_becomes_a_native_checkpoint(staged, corruption):
    with transport.upload_directory(str(staged)) as upload:
        root = Path(upload)
        path = root / transport.MANIFEST
        manifest = json.loads(path.read_text())
        entry = manifest["files"]["actor/model.pt"]
        part = entry["parts"][0]
        if corruption == "checksum":
            (root / part["path"]).write_bytes(b"X" * part["size"])
        elif corruption == "missing":
            (root / part["path"]).unlink()
        elif corruption == "traversal":
            manifest["files"]["../outside.pt"] = manifest["files"].pop("actor/model.pt")
        elif corruption == "duplicate":
            entry["parts"][1] = part
        else:
            entry["size"] += 1
        path.write_text(json.dumps(manifest))
        with pytest.raises((ValueError, FileNotFoundError)):
            transport.restore_directory(upload)
        assert not (root / "actor/model.pt").exists()

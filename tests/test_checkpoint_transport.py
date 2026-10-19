"""Large native checkpoint transport and pinned-revision recovery evidence."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.engine.support import checkpoint_transport as transport
from flash.providers.artifacts import hf


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


def test_remote_evidence_requires_every_part_at_the_same_revision(staged, monkeypatch):
    with transport.upload_directory(str(staged)) as upload:
        root = Path(upload)
        prefix = "opd/run/checkpoint/checkpoint-1"
        paths = [
            f"{prefix}/{p.relative_to(root).as_posix()}" for p in root.rglob("*") if p.is_file()
        ]
        raw = (root / transport.MANIFEST).read_bytes()

        def download(repo, path, *, revision, token):
            assert (repo, path, revision, token) == (
                "owner/repo",
                f"{prefix}/{transport.MANIFEST}",
                "pinned",
                "token",
            )
            return raw

        def infos(**kwargs):
            assert kwargs["revision"] == "pinned"
            return [
                SimpleNamespace(path=p, size=(root / p.removeprefix(prefix + "/")).stat().st_size)
                for p in kwargs["paths"]
            ]

        monkeypatch.setattr(hf, "_download_pinned_hf_bytes", download)
        api = SimpleNamespace(get_paths_info=infos)
        kwargs = {
            "hf_repo": "owner/repo",
            "prefix": prefix,
            "paths": paths,
            "revision": "pinned",
            "token": "token",
        }
        assert hf._transported_checkpoint_files(api, **kwargs) == {"actor/model.pt": 24}
        api.get_paths_info = lambda **kwargs: infos(**kwargs)[:-1]
        with pytest.raises(ValueError, match="length mismatch"):
            hf._transported_checkpoint_files(api, **kwargs)
        api.get_paths_info = infos
        kwargs["paths"] = [p for p in paths if not p.endswith("00000")]
        with pytest.raises(ValueError, match="missing"):
            hf._transported_checkpoint_files(api, **kwargs)


def test_opd_native_manifest_accepts_complete_transported_model_only(tmp_path, monkeypatch):
    from flash.teacher.retry_contract import OPD_RESUME_STATE_REQUIRED_FILES

    monkeypatch.setattr(transport, "PART_BYTES", 128)
    source = tmp_path / "native"
    actor = source / "actor"
    actor.mkdir(parents=True)
    (actor / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 2, "world_size": 1}))
    for kind in ("model", "optim", "extra_state"):
        (actor / f"{kind}_world_size_1_rank_0.pt").write_bytes(
            b"x" * (500 if kind == "model" else 10)
        )
    for name in (*OPD_RESUME_STATE_REQUIRED_FILES, "adapter_model.safetensors"):
        (source / name).write_bytes(b"{}")
    with transport.upload_directory(str(source)) as upload:
        root = Path(upload)
        prefix = "opd/run/checkpoint/checkpoint-1"
        paths = [
            f"{prefix}/{p.relative_to(root).as_posix()}" for p in root.rglob("*") if p.is_file()
        ]

        def infos(**kwargs):
            assert kwargs["revision"] == "pinned"
            return [
                SimpleNamespace(path=p, size=(root / p.removeprefix(prefix + "/")).stat().st_size)
                for p in kwargs["paths"]
            ]

        def download(_repo, path, **kwargs):
            assert kwargs["revision"] == "pinned"
            return (root / path.removeprefix(prefix + "/")).read_bytes()

        monkeypatch.setattr(hf, "_download_pinned_hf_bytes", download)
        api = SimpleNamespace(list_repo_files=lambda **_: paths, get_paths_info=infos)
        kwargs = {
            "hf_repo": "owner/repo",
            "phase": "opd",
            "run_id": "run",
            "revision": "pinned",
            "token": "token",
        }
        selected = hf._opd_resume_checkpoint(api, **kwargs)
        assert selected is not None
        assert selected.inspection.loadable
        paths[:] = [p for p in paths if not p.endswith("00000")]
        assert hf._opd_resume_checkpoint(api, **kwargs) is None

"""Exercise the bake lifecycle with real files and a fake GPU subprocess."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def entry_module():
    spec = importlib.util.spec_from_file_location(
        "bake_entry", Path(__file__).parents[1] / "docker/bake_sandbox_entry.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exported_copy_replaces_originals_and_preserves_absolute_references(tmp_path, monkeypatch):
    entry = entry_module()
    cache = tmp_path / "cache"
    library = cache / "flashinfer/.cache/kernel.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"compiled")
    manifest = cache / "group.json"
    manifest.write_text(json.dumps({"child_paths": {"kernel": str(library)}}))
    remove = entry.shutil.rmtree
    removed = []

    def remove_originals(path, **kwargs):
        removed.append(path)
        remove(path, **kwargs)
        assert not library.exists()

    monkeypatch.setattr(entry.shutil, "rmtree", remove_originals)
    entry._replace_cache_with_exported_copy(str(cache), str(tmp_path / "export.tar.gz"))
    assert removed == [str(cache)]
    assert library.read_bytes() == b"compiled"
    assert (
        Path(json.loads(manifest.read_text())["child_paths"]["kernel"]).read_bytes() == b"compiled"
    )


@pytest.mark.parametrize(("warm_rc", "replay_rc"), [(0, 0), (1, 0), (0, 1)])
def test_bake_clears_inherited_cache_and_replays_export(monkeypatch, tmp_path, warm_rc, replay_rc):
    import tarfile

    import huggingface_hub

    entry = entry_module()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "stale.cubin").write_bytes(b"stale")
    (cache / "cache_reload.json").write_text('{"passed": true}')
    monkeypatch.setattr(entry, "CACHE_DIR", str(cache))
    monkeypatch.setattr(entry, "OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(entry, "CACHE_ARCHIVE", str(tmp_path / "kernel_cache.tar.gz"))
    trainer = tmp_path / "trainer-python"
    trainer.touch()
    monkeypatch.setenv("FLASH_VERL_PYTHON", str(trainer))
    monkeypatch.setenv("BAKE_HF_REPO", "test/bake")
    monkeypatch.setenv("HF_TOKEN", "test")
    calls = []
    uploaded = {}
    actual_replace_cache = entry._replace_cache_with_exported_copy

    def replace_cache_with_exported_copy(cache_dir, archive_path):
        calls.append("replace_cache_with_exported_copy")
        actual_replace_cache(cache_dir, archive_path)

    monkeypatch.setattr(
        entry, "_replace_cache_with_exported_copy", replace_cache_with_exported_copy
    )

    def run(cmd, **kwargs):
        assert cmd[0] == str(trainer)
        assert cmd[cmd.index("--out") + 1] == str(cache)
        replay = "--verify-cache" in cmd
        calls.append("replay" if replay else "warm")
        assert not (cache / "stale.cubin").exists()
        if not replay:
            assert not (cache / "cache_reload.json").exists()
            (cache / "kernel.cubin").write_bytes(b"compiled")
            return warm_rc
        assert calls == ["warm", "replace_cache_with_exported_copy", "replay"]
        assert (cache / "kernel.cubin").read_bytes() == b"compiled"
        return replay_rc

    monkeypatch.setattr(entry.subprocess, "call", run)

    def upload(**kwargs):
        source = kwargs["path_or_fileobj"]
        if hasattr(source, "getvalue"):
            uploaded[kwargs["path_in_repo"]] = source.getvalue()
        elif source.endswith(".tar.gz"):
            with tarfile.open(source) as archive:
                assert archive.extractfile("./kernel.cubin").read() == b"compiled"

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(upload_file=upload))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: None)
    expected = warm_rc or replay_rc
    assert entry.main() == expected
    validation = json.loads((cache / "cache_validation.json").read_text())
    assert validation.get("exported_cache_replayed", False) == (warm_rc == 0)
    assert uploaded["out/STATUS"].startswith(f"rc={expected}\n".encode())

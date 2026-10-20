import importlib.util
from pathlib import Path

import pytest


def bake_module():
    spec = importlib.util.spec_from_file_location(
        "bake", Path(__file__).parents[1] / "docker/bake_kernel_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_baked_arch_has_a_modal_gpu():
    from flash.providers._lifecycle.net.worker import BAKED_PER_SM_ARCHES
    from flash.providers.core.base import GPU_CLASSES

    mapping = bake_module().GPU_BY_SM
    assert set(mapping) == BAKED_PER_SM_ARCHES
    for sm, request in mapping.items():
        assert any(
            g.sm == sm and g.modal_name.rstrip("!") == request.rstrip("!") for g in GPU_CLASSES
        )


def test_missing_bake_artifact_fails_verification(tmp_path):
    assert bake_module()._verify(str(tmp_path), "sm90") != 0


@pytest.mark.parametrize(
    "artifact", ["triton/kernel.cubin", "triton/kernel.ptx", "inductor/kernel.so"]
)
def test_verify_accepts_real_raw_artifacts(tmp_path, artifact):
    path = tmp_path / artifact
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"compiled")
    (tmp_path / "mega_cache.json").write_text('{"sm": "sm100"}')
    assert bake_module()._verify(str(tmp_path), "sm100") == 0
    assert bake_module()._verify(str(tmp_path), "sm90") == 1


def test_verify_rejects_metadata_and_lock_files_without_kernels(tmp_path):
    (tmp_path / "triton").mkdir()
    (tmp_path / "triton" / "kernel.lock").write_text("lock")
    (tmp_path / "mega_cache.json").write_text('{"sm": "sm100"}')
    assert bake_module()._verify(str(tmp_path), "sm100") == 1


def test_verify_rejects_legacy_blob_without_raw_kernels(tmp_path):
    (tmp_path / "mega_cache.bin").write_bytes(b"legacy-blob")
    (tmp_path / "mega_cache.json").write_text('{"sm": "sm100"}')
    assert bake_module()._verify(str(tmp_path), "sm100") == 1


@pytest.mark.parametrize("upload_fails", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cleanup_covers_input_upload_and_never_overrides_result(
    monkeypatch, tmp_path, upload_fails, cleanup_fails
):
    import sys
    from types import SimpleNamespace

    import huggingface_hub

    from flash.providers.modal import api

    bake = bake_module()
    deleted = []

    def delete(**kwargs):
        deleted.append(kwargs["repo_id"])
        if cleanup_fails:
            raise RuntimeError("cleanup outage")

    hf = SimpleNamespace(
        create_repo=lambda **kw: None,
        update_repo_settings=lambda **kw: None,
        upload_folder=lambda **kw: None,
        delete_repo=delete,
    )

    def upload(**kwargs):
        if upload_fails:
            raise RuntimeError("input upload outage")

    hf.upload_folder = upload
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: hf)
    monkeypatch.setenv("HF_TOKEN", "test")
    monkeypatch.setattr(api, "call", lambda *a, **kw: None)
    monkeypatch.setattr(bake, "_status_present", lambda *a, **kw: True)
    monkeypatch.setattr(bake, "_verify", lambda *a: 0)

    def download(**kwargs):
        (Path(kwargs["local_dir"]) / "out").mkdir()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bake",
            "--arch",
            "blackwell",
            "--sm",
            "sm100",
            "--image",
            "test/image",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    if upload_fails:
        with pytest.raises(RuntimeError, match="input upload outage"):
            bake.main()
    else:
        assert bake.main() == 0
    assert len(deleted) == 1


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_cache_archive_rejects_escaping_paths(tmp_path, name):
    import io
    import tarfile

    path = tmp_path / "cache.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo(name)
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="invalid kernel cache"):
        bake_module()._extract_cache_archive(path, tmp_path / "out")


def test_cache_archive_preserves_hidden_backend_directories(tmp_path):
    import io
    import tarfile

    path = tmp_path / "cache.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("flashinfer/.cache/kernel.so")
        entry.size = 8
        archive.addfile(entry, io.BytesIO(b"compiled"))
    bake_module()._extract_cache_archive(path, tmp_path / "out")
    assert (tmp_path / "out/flashinfer/.cache/kernel.so").read_bytes() == b"compiled"

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

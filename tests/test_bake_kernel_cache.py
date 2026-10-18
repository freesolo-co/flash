import importlib.util
import json
from pathlib import Path


def bake_module():
    spec = importlib.util.spec_from_file_location(
        "bake", Path(__file__).parents[1] / "docker/bake_kernel_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_baked_arch_has_a_modal_gpu():
    from flash.providers._lifecycle.worker import BAKED_PER_SM_ARCHES
    from flash.providers.base import GPU_CLASSES

    mapping = bake_module().GPU_BY_SM
    assert set(mapping) == BAKED_PER_SM_ARCHES
    for sm, request in mapping.items():
        assert any(
            g.sm == sm and g.modal_name.rstrip("!") == request.rstrip("!") for g in GPU_CLASSES
        )


def test_missing_bake_artifact_fails_verification(tmp_path):
    assert bake_module()._verify(str(tmp_path), "sm90") != 0


def test_wrong_bake_arch_fails_verification(tmp_path):
    (tmp_path / "mega_cache.bin").write_bytes(b"cache")
    (tmp_path / "mega_cache.json").write_text(json.dumps({"arch": "sm80"}))
    assert bake_module()._verify(str(tmp_path), "sm90") != 0

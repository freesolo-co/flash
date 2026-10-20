import json
import os
import sys
import types

import pytest

from flash.engine.worker.runtime import kernel_warmup


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_capability(index):
        return (8, 9)

    @staticmethod
    def get_device_name(index):
        return "fake gpu"


class _FakeTorch:
    __version__ = "test"
    cuda = _FakeCuda()
    version = types.SimpleNamespace(cuda="12.8")


def test_kernel_warmup_out_overrides_existing_cache_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", "/old/triton")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/old/inductor")

    kernel_warmup._point_backends_at(str(tmp_path))

    assert str(tmp_path / "triton") == os.environ["TRITON_CACHE_DIR"]
    assert str(tmp_path / "inductor") == os.environ["TORCHINDUCTOR_CACHE_DIR"]


def test_kernel_warmup_writes_arch_metadata(tmp_path):
    assert kernel_warmup.save_cache_metadata(
        _FakeTorch,
        str(tmp_path),
        requested_arch="8.9",
        warmed=3,
    )

    meta = json.loads((tmp_path / kernel_warmup.CACHE_META_FILENAME).read_text())
    assert meta["sm"] == "sm89"
    assert meta["requested_arch"] == "8.9"
    assert meta["warmed_groups"] == 3


def test_worker_skips_baked_cache_when_arch_undetermined(monkeypatch, tmp_path):
    (tmp_path / "triton").mkdir()
    (tmp_path / "triton" / "gdn.cubin").write_bytes(b"compiled")
    (tmp_path / kernel_warmup.CACHE_META_FILENAME).write_text('{"sm": "sm89"}')
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))

    assert kernel_warmup.configure_kernel_cache() is False
    assert os.environ["TRITON_CACHE_DIR"] != str(tmp_path / "triton")


def test_warmup_saves_real_raw_kernels_without_a_compile_graph(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_warmup, "_require_gpu", lambda: _FakeTorch)
    monkeypatch.setattr(kernel_warmup, "warm_flash_attn", lambda torch: True)

    def warm_gdn(torch):
        artifact = tmp_path / "triton" / "gdn.cubin"
        artifact.write_bytes(b"compiled-gdn")
        return True

    monkeypatch.setattr(kernel_warmup, "warm_fla_gdn", warm_gdn)
    assert kernel_warmup.warmup(str(tmp_path), arch="8.9") == 0
    assert not (tmp_path / "mega_cache.bin").exists()
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    assert kernel_warmup.configure_kernel_cache() is True
    assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "triton")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "inductor")


def test_warmup_rejects_empty_cache_even_if_smoke_tests_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_warmup, "_require_gpu", lambda: _FakeTorch)
    monkeypatch.setattr(kernel_warmup, "warm_flash_attn", lambda torch: True)
    monkeypatch.setattr(kernel_warmup, "warm_fla_gdn", lambda torch: True)
    assert kernel_warmup.warmup(str(tmp_path)) == 1


def test_raw_cache_arch_mismatch_redirects_to_fresh_cache(monkeypatch, tmp_path):
    (tmp_path / "triton").mkdir()
    (tmp_path / "triton" / "gdn.cubin").write_bytes(b"wrong-arch")
    (tmp_path / kernel_warmup.CACHE_META_FILENAME).write_text('{"sm": "sm90"}')
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    assert kernel_warmup.configure_kernel_cache() is False
    assert os.environ["TRITON_CACHE_DIR"] != str(tmp_path / "triton")


def test_unbaked_runtime_jit_cache_is_preserved_without_loading_torch(monkeypatch, tmp_path):
    triton = tmp_path / "triton"
    triton.mkdir()
    (triton / "runtime.cubin").write_bytes(b"compiled-on-this-worker")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(triton))
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", None)
    assert kernel_warmup.configure_kernel_cache() is False
    assert os.environ["TRITON_CACHE_DIR"] == str(triton)


def test_worker_rejects_unreadable_bake_metadata(monkeypatch, tmp_path):
    (tmp_path / "triton").mkdir()
    (tmp_path / "triton" / "gdn.cubin").write_bytes(b"compiled")
    (tmp_path / kernel_warmup.CACHE_META_FILENAME).write_text("invalid json")
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)

    assert kernel_warmup.configure_kernel_cache() is False
    assert os.environ["TRITON_CACHE_DIR"] != str(tmp_path / "triton")


@pytest.mark.parametrize(
    ("field", "value"), [("torch", "old"), ("cuda", "11.8"), ("python", "3.9")]
)
def test_baked_runtime_mismatch_redirects_every_backend(monkeypatch, tmp_path, field, value):
    kernel_warmup._point_backends_at(str(tmp_path))
    (tmp_path / "tilelang" / "kernel.so").write_bytes(b"compiled")
    kernel_warmup.save_cache_metadata(_FakeTorch, str(tmp_path), requested_arch="8.9", warmed=2)
    path = tmp_path / kernel_warmup.CACHE_META_FILENAME
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    assert not kernel_warmup.configure_kernel_cache()
    for name in kernel_warmup.KERNEL_CACHE_ENV_SUBDIRS:
        assert not os.environ[name].startswith(str(tmp_path))


def test_tilelang_artifacts_are_part_of_bake(tmp_path):
    cache = tmp_path / "tilelang"
    cache.mkdir()
    (cache / "kernel.so").write_bytes(b"compiled")
    assert kernel_warmup.has_raw_kernel_cache(str(tmp_path))

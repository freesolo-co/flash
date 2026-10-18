import json
import os
import sys
import types

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
    compiler = types.SimpleNamespace(load_cache_artifacts=lambda blob: None)


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

    meta = json.loads((tmp_path / kernel_warmup.MEGA_CACHE_META_FILENAME).read_text())
    assert meta["sm"] == "sm89"
    assert meta["requested_arch"] == "8.9"
    assert meta["warmed_groups"] == 3


def test_worker_skips_baked_cache_when_arch_mismatches(monkeypatch, tmp_path):
    cache = tmp_path / kernel_warmup.MEGA_CACHE_FILENAME
    meta = tmp_path / kernel_warmup.MEGA_CACHE_META_FILENAME
    cache.write_bytes(b"cache")
    meta.write_text(json.dumps({"sm": "sm90"}))
    loaded = {"called": False}

    fake_torch = types.SimpleNamespace(
        cuda=_FakeCuda(),
        compiler=types.SimpleNamespace(
            load_cache_artifacts=lambda blob: loaded.__setitem__("called", True)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))

    assert kernel_warmup.load_mega_cache() is False
    assert loaded["called"] is False


def test_worker_skips_baked_cache_when_arch_undetermined(monkeypatch, tmp_path):
    cache = tmp_path / kernel_warmup.MEGA_CACHE_FILENAME
    meta = tmp_path / kernel_warmup.MEGA_CACHE_META_FILENAME
    cache.write_bytes(b"cache")
    # metadata arch would match a real sm89 worker, but this worker can't determine its own arch
    # (no CUDA) -> we must NOT load a blob we cannot verify; fall back to JIT.
    meta.write_text(json.dumps({"sm": "sm89"}))
    loaded = {"called": False}

    class _NoCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def get_device_capability(index):
            raise RuntimeError("no cuda")

        @staticmethod
        def get_device_name(index):
            return "none"

    fake_torch = types.SimpleNamespace(
        cuda=_NoCuda(),
        compiler=types.SimpleNamespace(
            load_cache_artifacts=lambda blob: loaded.__setitem__("called", True)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))

    assert kernel_warmup.load_mega_cache() is False
    assert loaded["called"] is False


def test_warmup_saves_real_raw_kernels_without_a_compile_graph(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_warmup, "_require_gpu", lambda: _FakeTorch)
    monkeypatch.setattr(kernel_warmup, "warm_flash_attn", lambda torch: True)

    def warm_gdn(torch):
        artifact = tmp_path / "triton" / "gdn.cubin"
        artifact.write_bytes(b"compiled-gdn")
        return True

    monkeypatch.setattr(kernel_warmup, "warm_fla_gdn", warm_gdn)
    monkeypatch.setattr(kernel_warmup, "save_mega_cache", lambda *_: False)
    assert kernel_warmup.warmup(str(tmp_path), arch="8.9") == 0
    assert not (tmp_path / kernel_warmup.MEGA_CACHE_FILENAME).exists()
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    assert kernel_warmup.load_mega_cache() is True


def test_warmup_rejects_empty_cache_even_if_smoke_tests_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_warmup, "_require_gpu", lambda: _FakeTorch)
    monkeypatch.setattr(kernel_warmup, "warm_flash_attn", lambda torch: True)
    monkeypatch.setattr(kernel_warmup, "warm_fla_gdn", lambda torch: True)
    monkeypatch.setattr(kernel_warmup, "save_mega_cache", lambda *_: False)
    assert kernel_warmup.warmup(str(tmp_path)) == 1


def test_raw_cache_arch_mismatch_redirects_to_fresh_cache(monkeypatch, tmp_path):
    (tmp_path / "triton").mkdir()
    (tmp_path / "triton" / "gdn.cubin").write_bytes(b"wrong-arch")
    (tmp_path / kernel_warmup.MEGA_CACHE_META_FILENAME).write_text('{"sm": "sm90"}')
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    assert kernel_warmup.load_mega_cache() is False
    assert os.environ["TRITON_CACHE_DIR"] != str(tmp_path / "triton")


def test_unbaked_runtime_jit_cache_is_preserved_without_loading_torch(monkeypatch, tmp_path):
    triton = tmp_path / "triton"
    triton.mkdir()
    (triton / "runtime.cubin").write_bytes(b"compiled-on-this-worker")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(triton))
    monkeypatch.setattr(kernel_warmup, "DEFAULT_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", None)
    assert kernel_warmup.load_mega_cache() is False
    assert os.environ["TRITON_CACHE_DIR"] == str(triton)

import json
import os
import sys
import types

from flash.engine.worker import kernel_warmup


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
    import flash.engine.worker as worker

    cache = tmp_path / "mega_cache.bin"
    meta = tmp_path / "mega_cache.json"
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
    monkeypatch.setattr(worker, "_KERNEL_CACHE_FILE", str(cache))
    monkeypatch.setattr(worker, "_KERNEL_CACHE_META_FILE", str(meta))

    assert worker._load_kernel_cache_if_present() is False
    assert loaded["called"] is False


def test_worker_skips_baked_cache_when_arch_undetermined(monkeypatch, tmp_path):
    import flash.engine.worker as worker

    cache = tmp_path / "mega_cache.bin"
    meta = tmp_path / "mega_cache.json"
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
    monkeypatch.setattr(worker, "_KERNEL_CACHE_FILE", str(cache))
    monkeypatch.setattr(worker, "_KERNEL_CACHE_META_FILE", str(meta))

    assert worker._load_kernel_cache_if_present() is False
    assert loaded["called"] is False

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


def test_warm_chalk_kernels_calls_default_standalone_installers(monkeypatch):
    calls = []

    def _install(module_name, attr, result=False):
        mod = types.ModuleType(module_name)

        def installer(*args):
            calls.append((module_name, attr, args))
            return result

        setattr(mod, attr, installer)
        monkeypatch.setitem(sys.modules, module_name, mod)

    for module_name, attr, result in (
        ("chalk.ops.rope", "install_qwen35_rope", True),
        ("chalk.ops.rmsnorm", "install_qwen35_rmsnorm", False),
        ("chalk.ops.swiglu", "install_qwen35_swiglu", False),
        ("chalk.ops.flce", "install_qwen35_flce", False),
        ("chalk.ops.lora", "install_fused_lora_delta", False),
        ("chalk.ops.qkv", "install_qwen35_qknorm_rope", False),
        ("chalk.ops.embedding", "install_qwen35_fused_embedding", False),
        ("chalk.ops.gdn", "install_qwen35_gdn", False),
    ):
        _install(module_name, attr, result)

    assert kernel_warmup.warm_chalk_kernels() is True
    assert calls == [
        ("chalk.ops.rope", "install_qwen35_rope", ()),
        ("chalk.ops.rmsnorm", "install_qwen35_rmsnorm", ()),
        ("chalk.ops.swiglu", "install_qwen35_swiglu", ()),
        ("chalk.ops.flce", "install_qwen35_flce", (None,)),
        ("chalk.ops.lora", "install_fused_lora_delta", ()),
        ("chalk.ops.qkv", "install_qwen35_qknorm_rope", (None,)),
        ("chalk.ops.embedding", "install_qwen35_fused_embedding", (None,)),
        ("chalk.ops.gdn", "install_qwen35_gdn", ()),
    ]

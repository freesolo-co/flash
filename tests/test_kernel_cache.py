import contextlib
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


def test_bake_uses_the_native_training_backend(monkeypatch):
    implementation = object()
    monkeypatch.setitem(
        sys.modules,
        "fla.ops.gated_delta_rule",
        types.SimpleNamespace(chunk_gated_delta_rule=implementation),
    )
    assert kernel_warmup._training_gdn_kernel() is implementation


@pytest.mark.parametrize("finite", [True, False])
def test_flashinfer_prefill_bake_executes_native_kernel(monkeypatch, finite):
    calls = []

    def prefill(**kwargs):
        assert kwargs["output_final_state"] is True
        assert kwargs["initial_state"] is not None
        assert kwargs["cu_seqlens"] is not None
        calls.append("prefill")
        return object(), object()

    # The replacement kernel has no get_gdn_prefill_module or C++ AOT library.
    module = types.SimpleNamespace(chunk_gated_delta_rule=prefill)
    monkeypatch.setitem(sys.modules, "flashinfer.gdn_prefill", module)
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(synchronize=lambda: calls.append("sync")),
        zeros=lambda *a, **k: object(),
        ones=lambda *a, **k: object(),
        full=lambda *a, **k: object(),
        tensor=lambda *a, **k: object(),
        bfloat16="bfloat16",
        float32="float32",
        int64="int64",
        isfinite=lambda _: types.SimpleNamespace(all=lambda: finite),
    )
    assert kernel_warmup.warm_flashinfer_gdn(torch) is finite
    assert calls == ["prefill", "sync"]


@pytest.mark.parametrize("verify_cache", [False, True])
def test_hopper_bake_requires_rollout_prefill_coverage(monkeypatch, tmp_path, verify_cache):
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(get_device_capability=lambda _: (9, 0)),
    )
    monkeypatch.setattr(kernel_warmup, "_require_gpu", lambda: torch)
    monkeypatch.setattr(kernel_warmup, "warm_flash_attn", lambda _: True)
    monkeypatch.setattr(kernel_warmup, "warm_fla_gdn", lambda _: True)
    monkeypatch.setattr(kernel_warmup, "warm_flashinfer_gdn", lambda _: False)
    monkeypatch.setattr(kernel_warmup, "has_raw_kernel_cache", lambda _: True)
    monkeypatch.setattr(kernel_warmup, "save_cache_metadata", lambda *a, **k: True)
    monkeypatch.setattr(kernel_warmup, "_cached_kernels_only", lambda _: contextlib.nullcontext([]))
    (tmp_path / "mega_cache.json").write_text('{"sm": "sm90"}')
    assert kernel_warmup.warmup(str(tmp_path), verify_cache=verify_cache) == 1


@pytest.mark.parametrize("backend", [None, "triton", "tilelang", "flashinfer", "torch"])
def test_reload_rejects_compilation_even_if_the_backend_swallows_the_error(
    monkeypatch, tmp_path, backend
):
    class Source:
        def make_ir(self):
            pytest.fail("unexpected compilation")

    class TileKernel:
        def _compile_and_create_adapter(self):
            pytest.fail("unexpected compilation")

    class FlashKernel:
        def build(self):
            pytest.fail("unexpected compilation")

    cpp = types.SimpleNamespace(_run_ninja_build=lambda: pytest.fail("unexpected compilation"))
    monkeypatch.setitem(sys.modules, "torch.utils", types.SimpleNamespace(cpp_extension=cpp))
    monkeypatch.setitem(
        sys.modules,
        "triton.compiler.compiler",
        types.SimpleNamespace(ASTSource=Source, IRSource=Source),
    )
    monkeypatch.setitem(sys.modules, "tilelang.jit", types.SimpleNamespace(JITKernel=TileKernel))
    monkeypatch.setitem(
        sys.modules, "flashinfer.jit.core", types.SimpleNamespace(JitSpec=FlashKernel)
    )
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(get_device_capability=lambda _: (9, 0))
    )
    monkeypatch.setattr(kernel_warmup, "_require_gpu", lambda: torch)
    original = Source.make_ir

    def warm(_torch):
        if backend is not None:
            calls = {
                "triton": Source().make_ir,
                "tilelang": TileKernel()._compile_and_create_adapter,
                "flashinfer": FlashKernel().build,
                "torch": cpp._run_ninja_build,
            }
            # A backend fallback must not turn a cache miss into a passing bake.
            with contextlib.suppress(RuntimeError):
                calls[backend]()
        return True

    monkeypatch.setattr(kernel_warmup, "warm_flash_attn", lambda _: True)
    monkeypatch.setattr(kernel_warmup, "warm_fla_gdn", warm)
    monkeypatch.setattr(kernel_warmup, "warm_flashinfer_gdn", lambda _: True)
    monkeypatch.setattr(
        kernel_warmup, "save_cache_metadata", lambda *a, **k: pytest.fail("overwrote bake metadata")
    )
    (tmp_path / "mega_cache.json").write_text('{"sm": "sm90"}')
    (tmp_path / "triton").mkdir()
    (tmp_path / "triton/kernel.cubin").write_bytes(b"compiled")
    assert kernel_warmup.warmup(str(tmp_path), verify_cache=True) == int(backend is not None)
    proof = json.loads((tmp_path / "cache_reload.json").read_text())
    assert proof == {
        "cached_groups_compilation_forbidden": True,
        "cached_groups": ["flash_attn", "fla_gdn"],
        "startup_jit_groups": ["flashinfer_gdn"],
        "compile_attempts": int(backend is not None),
        "passed": backend is None,
    }
    assert Source.make_ir is original

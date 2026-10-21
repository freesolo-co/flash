"""The baked library must survive copied file mtimes and reject incompatible workers."""

import json
import sys
from types import SimpleNamespace

import pytest

from flash.engine.worker.train.core.child import flashinfer_cache as cache


@pytest.fixture
def baked(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHINFER_CACHE_DIR", str(tmp_path / "cache"))
    torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(get_device_capability=lambda _: (9, 0)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    env = SimpleNamespace(
        flashinfer_version="0.6.6",
        FLASHINFER_AOT_DIR=tmp_path / "builtin",
        FLASHINFER_JIT_DIR=tmp_path / "jit",
    )
    source = env.FLASHINFER_JIT_DIR / cache.MODULE / f"{cache.MODULE}.so"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"compiled-fixture")
    cache.publish(env, torch)
    return env, tmp_path / "cache" / "aot"


def test_portable_library_uses_aot_without_ninja_or_source_files(baked):
    env, root = baked
    import shutil

    shutil.rmtree(env.FLASHINFER_JIT_DIR)
    assert cache.activate(env)
    assert root == env.FLASHINFER_AOT_DIR
    assert (root / cache.MODULE / f"{cache.MODULE}.so").read_bytes() == b"compiled-fixture"


@pytest.mark.parametrize("key", ["format", "flashinfer", "torch", "cuda", "python", "sm"])
def test_mismatched_library_identity_keeps_normal_jit(baked, key):
    env, root = baked
    path = root / "manifest.json"
    meta = json.loads(path.read_text())
    meta[key] = "wrong"
    path.write_text(json.dumps(meta))
    original = env.FLASHINFER_AOT_DIR
    assert not cache.activate(env)
    assert original == env.FLASHINFER_AOT_DIR


def test_missing_library_and_corrupt_manifest_fall_back(baked):
    env, root = baked
    (root / cache.MODULE / f"{cache.MODULE}.so").unlink()
    assert not cache.activate(env)
    (root / "manifest.json").write_text("invalid")
    assert not cache.activate(env)


def test_existing_broader_aot_cache_is_not_hidden(baked):
    env, _ = baked
    other = env.FLASHINFER_AOT_DIR / "other" / "other.so"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"existing")
    assert not cache.activate(env)


def test_bootstrap_registers_without_importing_flashinfer(monkeypatch):
    monkeypatch.delitem(sys.modules, "flashinfer.jit.env", raising=False)
    registered = []
    runtime = SimpleNamespace(
        _DEFERRED_FINDER=SimpleNamespace(register=lambda *a: registered.append(a))
    )
    cache.install(runtime)
    assert registered == [("flashinfer.jit.env", cache.activate)]

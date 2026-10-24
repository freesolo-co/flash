"""Reject a different engine or binary even if the source checkout is pinned."""

import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from flash.serving.deepinfer import provenance


@pytest.fixture
def loaded(monkeypatch, tmp_path):
    root = tmp_path / "DeepInfer"
    root.mkdir()
    binary = root / "_C.so"
    binary.write_bytes(b"test native binary")
    native = ModuleType("deepinfer._C")
    native.__file__ = str(binary)
    package = ModuleType("deepinfer")
    package._C = native
    engine_module = ModuleType("deepinfer.engine.llm_engine")
    engine_module.__file__ = str(root / "llm_engine.py")
    engine = type("LLMEngine", (), {"__module__": engine_module.__name__})()
    for name, module in (
        ("deepinfer", package),
        ("deepinfer._C", native),
        (engine_module.__name__, engine_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(get_device_name=lambda _: "NVIDIA B200"),
            __version__="test",
            version=SimpleNamespace(cuda="13.2"),
        ),
    )
    monkeypatch.setattr(provenance, "RUNTIME_ROOT", str(root))
    monkeypatch.setattr(
        provenance.subprocess, "check_output", lambda *a, **k: provenance.RUNTIME_REVISION
    )
    return SimpleNamespace(llm_engine=engine), native, engine_module


def test_proof_hashes_the_loaded_binary(loaded):
    llm, _, _ = loaded
    proof = provenance.loaded_runtime(llm, 3)
    assert proof["rank"] == 3
    assert proof["native_sha256"] == hashlib.sha256(b"test native binary").hexdigest()
    assert proof["gpu"] == "NVIDIA B200"


@pytest.mark.parametrize("mismatch", ["revision", "engine", "binary"])
def test_wrong_runtime_fails_closed(loaded, monkeypatch, tmp_path, mismatch):
    llm, native, engine = loaded
    if mismatch == "revision":
        monkeypatch.setattr(provenance.subprocess, "check_output", lambda *a, **k: "wrong")
    elif mismatch == "engine":
        engine.__file__ = str(tmp_path / "replacement.py")
    else:
        native.__file__ = str(tmp_path / "replacement.so")
    with pytest.raises(RuntimeError, match="DeepInfer"):
        provenance.loaded_runtime(llm, 0)

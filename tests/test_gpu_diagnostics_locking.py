from __future__ import annotations

import sys
import threading
import types

import pytest

from flash.engine.worker.perf import diagnostics


@pytest.fixture
def fast_nvidia(monkeypatch):
    monkeypatch.setattr(
        diagnostics, "_query_nvidia_gpu", lambda: {"gpu_util_pct": 0, "device_name": "FAKE-GPU"}
    )
    monkeypatch.setattr(diagnostics, "_query_nvidia_processes", list)


def _install_blocking_torch(monkeypatch, gate: threading.Event) -> None:
    def blocking_mem_get_info():
        gate.wait(timeout=30.0)
        return (1 << 30, 1 << 34)

    torch = types.ModuleType("torch")
    torch.__version__ = "2.10.0-fake"
    torch.version = types.SimpleNamespace(cuda="12.8")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda i=0: "FAKE-GPU",
        mem_get_info=blocking_mem_get_info,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def _run_async(fn):
    state: dict = {}

    def target():
        state["value"] = fn()
        state["done"] = True

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, state


def test_torch_diagnostics_wait_for_a_locked_cuda_runtime(monkeypatch, fast_nvidia):
    gate = threading.Event()
    _install_blocking_torch(monkeypatch, gate)

    thread, state = _run_async(lambda: diagnostics.gpu_diagnostics(include_torch=True))
    thread.join(timeout=1.5)
    assert not state.get("done")

    gate.set()
    thread.join(timeout=5.0)
    assert state.get("done")


def test_nvidia_diagnostics_remain_responsive_while_cuda_is_locked(monkeypatch, fast_nvidia):
    gate = threading.Event()
    _install_blocking_torch(monkeypatch, gate)

    thread, state = _run_async(lambda: diagnostics.gpu_diagnostics(include_torch=False))
    thread.join(timeout=2.0)

    assert state.get("done")
    result = state["value"]
    assert result.get("gpu_util_pct") == 0
    assert "torch" not in result
    assert "torch_memory_free_gb" not in result

"""The custom backend returns cached CUDA blocks to the driver before an operation when free
memory is low, so allocations outside PyTorch's cache (cuBLAS, Triton) can still succeed.

A fake torch module stands in for CUDA, so these run on CPU-only CI without torch installed.
"""

import sys
import types

from flash.custom import backend

GIB = 1024**3


def fake_cuda(monkeypatch, *, available=True, free=1 * GIB, total=80 * GIB, fail=False):
    calls = []

    def mem_get_info(device=0):
        if fail:
            raise RuntimeError("driver unavailable")
        return free, total

    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: 1,
        mem_get_info=mem_get_info,
        empty_cache=lambda: calls.append("empty_cache"),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    return calls


def test_low_free_memory_releases_the_cache(monkeypatch):
    calls = fake_cuda(monkeypatch, free=1 * GIB)
    assert backend._ensure_cuda_headroom() is True
    assert calls == ["empty_cache"]


def test_enough_free_memory_keeps_the_cache(monkeypatch):
    calls = fake_cuda(monkeypatch, free=backend.CUDA_HEADROOM_BYTES)
    assert backend._ensure_cuda_headroom() is False
    assert calls == []


def test_low_free_memory_on_second_gpu_releases_cache(monkeypatch):
    calls = fake_cuda(monkeypatch)
    cuda = sys.modules["torch"].cuda
    cuda.device_count = lambda: 2
    cuda.mem_get_info = lambda device: ((20 if device == 0 else 1) * GIB, 80 * GIB)
    assert backend._ensure_cuda_headroom() is True
    assert calls == ["empty_cache"]


def test_no_cuda_is_a_no_op(monkeypatch):
    calls = fake_cuda(monkeypatch, available=False)
    assert backend._ensure_cuda_headroom() is False
    assert calls == []


def test_driver_errors_never_fail_an_operation(monkeypatch):
    calls = fake_cuda(monkeypatch, fail=True)
    assert backend._ensure_cuda_headroom() is False
    assert calls == []


def test_every_operation_checks_headroom_first(monkeypatch):
    order = []
    monkeypatch.setattr(backend, "_ensure_cuda_headroom", lambda: order.append("headroom"))
    fake = types.SimpleNamespace(forward=lambda **kw: order.append(("forward", kw)) or {"ok": True})
    assert backend.TorchBackend.execute(fake, "forward", {"data": []}) == {"ok": True}
    assert order == ["headroom", ("forward", {"data": []})]

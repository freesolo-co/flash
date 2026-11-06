"""Custom layer sharding must preserve per-device admission and whole-allocation billing."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flash.custom import model_parallel
from flash.custom.billing import estimate
from flash.custom.placement import per_device_vram_gb
from flash.providers.core import allocator
from flash.providers.core.base import Candidate, UnsupportedGpuError, provisional_gpu
from flash.schema import ConfigError, spec_from_dict

MODEL = "Qwen/Qwen3.6-35B-A3B"
TRAIN = {"lora_rank": 64, "max_context_tokens": 8192}


def spec(gpu):
    return spec_from_dict(
        {
            "model": MODEL,
            "algorithm": "custom",
            "custom": {"file": "a.py"},
            "train": TRAIN,
            "gpu": gpu,
        }
    )


def test_two_gpu_shape_parses_but_single_card_and_small_pairs_do_not():
    assert spec({"type": "H200", "count": 2}).gpu.count == 2
    assert provisional_gpu(MODEL, "custom", train=TRAIN, gpu_count=2) == "H200"
    for shape in ({}, {"type": "H200"}, {"type": "A100 SXM", "count": 2}):
        with pytest.raises(ConfigError):
            spec(shape)


def test_per_device_floor_does_not_divide_logits_by_gpu_count():
    one = per_device_vram_gb(MODEL, TRAIN, 1)
    two = per_device_vram_gb(MODEL, TRAIN, 2)
    assert one / 2 < two < one
    long = {**TRAIN, "max_context_tokens": 16384}
    # The full logits/activation growth remains on EACH device.
    assert per_device_vram_gb(MODEL, long, 2) - two >= per_device_vram_gb(MODEL, long, 1) - one


def test_allocator_filters_pooled_fit_that_fails_per_device_floor(monkeypatch):
    candidates = [Candidate("modal", "H200", 4.0, 141, gpu_count=2)]
    monkeypatch.setattr(allocator, "available_providers", lambda: ("modal",))
    monkeypatch.setattr(
        allocator,
        "get_provider",
        lambda _: SimpleNamespace(live_candidates=lambda *args: candidates),
    )
    run = spec({"count": 2})
    chosen = allocator.allocate(MODEL, "custom", train=run.train, max_gpu_count=2)
    assert chosen.gpu_count == 2
    quote = estimate(run, allocation=chosen)
    assert quote.gpu_count == 2
    assert quote.total_usd == 8.0  # one-hour cap, two $4/hour cards
    # Combined capacity clears 187GB, but each card misses the 121GB layer-placement floor.
    candidates[:] = [Candidate("modal", "H200", 4.0, 120, gpu_count=2)]
    with pytest.raises(UnsupportedGpuError):
        allocator.allocate(MODEL, "custom", train=run.train, max_gpu_count=2)


def test_loader_reserves_only_requested_cuda_devices(monkeypatch):
    cuda = SimpleNamespace(
        device_count=lambda: 4,
        get_device_properties=lambda _: SimpleNamespace(total_memory=141 * 10**9),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    placement = model_parallel.load_placement(MODEL, 64, 8192, 2)
    assert placement["device_map"] == "balanced"
    assert set(placement["max_memory"]) == {0, 1}
    cuda.device_count = lambda: 1
    with pytest.raises(ValueError, match="visible GPUs"):
        model_parallel.load_placement(MODEL, 64, 8192, 2)
    cuda.device_count = lambda: 2
    cuda.get_device_properties = lambda _: SimpleNamespace(total_memory=80 * 10**9)
    with pytest.raises(ValueError, match="on each GPU"):
        model_parallel.load_placement(MODEL, 64, 8192, 2)


def test_worker_passes_allocated_gpu_count_to_loader(monkeypatch):
    from flash.custom import worker

    context = SimpleNamespace(config=SimpleNamespace(spec=spec({"type": "H200", "count": 2})))
    load = Mock(side_effect=RuntimeError("stop before GPU loading"))
    monkeypatch.setattr(worker, "RunCheckpoints", Mock())
    monkeypatch.setattr(worker, "heartbeat", Mock())
    monkeypatch.setattr(worker, "liveness_heartbeat", lambda *a, **k: nullcontext())
    monkeypatch.setattr(worker, "load_backend", load)
    with pytest.raises(RuntimeError, match="stop before GPU loading"):
        worker.run(context=context)
    assert load.call_args.kwargs["gpu_count"] == 2


@pytest.mark.parametrize("devices", [("cuda:0", "cpu"), ("cuda:0",)])
def test_loader_rejects_offload_and_unused_allocated_gpu(devices):
    class Device:
        def __init__(self, name):
            self.type = name.split(":")[0]

    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=Device(d)) for d in devices])
    with pytest.raises(ValueError, match=r"GPU-resident|every allocated GPU"):
        model_parallel.validate_placement(model, 2)


@pytest.mark.parametrize(
    ("states", "count", "expected"),
    [
        (["a", "b"], 2, [("a", 0), ("b", 1)]),
        (["a", "b"], 1, [("a", 0)]),
        (["a"], 2, [("a", 0), ("a", 1)]),
    ],
)
def test_resume_rng_handles_changed_gpu_shape(monkeypatch, states, count, expected):
    calls = []
    cuda = SimpleNamespace(
        device_count=lambda: count,
        set_rng_state=lambda state, device: calls.append((state, device)),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    model_parallel.restore_cuda_rng(states)
    assert calls == expected

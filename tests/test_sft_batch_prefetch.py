"""Packed batches retain exact ordering while keeping a bounded epoch-to-epoch producer pool."""

import sys
from types import ModuleType

import pytest

from flash.engine.worker.train.sft.child.plugin import _install_exact_dataloaders


@pytest.mark.parametrize("workers", [0, 2, 4])
def test_prefetch_pool_is_bounded_and_zero_workers_supported(monkeypatch, workers):
    calls = []

    class Sampler:
        def __init__(self, *args, **kwargs):
            pass

    class Loader:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

    sampling = ModuleType("torch.utils.data.distributed")
    sampling.DistributedSampler = Sampler
    loading = ModuleType("torchdata.stateful_dataloader")
    loading.StatefulDataLoader = Loader
    monkeypatch.setitem(sys.modules, sampling.__name__, sampling)
    monkeypatch.setitem(sys.modules, loading.__name__, loading)
    _install_exact_dataloaders()
    Loader(num_workers=workers, batch_size=8, drop_last=True)
    assert calls[0]["drop_last"] is False
    assert calls[0]["batch_size"] == 8
    if workers:
        assert calls[0]["persistent_workers"] is True
        assert calls[0]["prefetch_factor"] == 2
    else:
        assert "persistent_workers" not in calls[0]
        assert "prefetch_factor" not in calls[0]

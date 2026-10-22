"""Packed batches retain ordering with a bounded, persistent producer pool."""

from types import SimpleNamespace

import pytest

from flash.engine.worker.train.core.child.options import RuntimeOptions
from tests.test_verl_child_plugins import _sft_recipe


@pytest.mark.parametrize("workers", [0, 2, 4])
def test_prefetch_pool_is_bounded_and_zero_workers_supported(monkeypatch, workers):
    pytest.importorskip("torchdata")
    recipe = _sft_recipe(monkeypatch)
    trainer = recipe.FlashSFTTrainer(
        SimpleNamespace(
            model=SimpleNamespace(runtime_options=RuntimeOptions("sft", 9, "ready").encode()),
            data=SimpleNamespace(train_batch_size=4, pad_mode="right", num_workers=workers),
        )
    )
    loader = trainer.train_dataloader
    assert loader.drop_last is False
    assert loader.batch_size == 4
    assert loader.num_workers == workers
    assert loader.persistent_workers is bool(workers)
    assert loader.prefetch_factor == (2 if workers else None)

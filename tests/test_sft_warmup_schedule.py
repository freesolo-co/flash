"""warmup must permit the first required checkpoint to contain a learned delta."""

import pytest

from flash.engine.worker.train.sft.child.schedule import linear_warmup_factor


@pytest.mark.parametrize(("warmup", "total"), [(0, 2), (1, 34), (6, 200), (30, 1000)])
def test_first_update_has_positive_lr_and_decay_keeps_existing_indices(warmup, total):
    assert 0 < linear_warmup_factor(0, warmup=warmup, total=total) <= 1
    for step in range(1, total + 2):
        expected = (
            step / max(1, warmup)
            if step < warmup
            else max(0.0, (total - step) / max(1, total - warmup))
        )
        assert linear_warmup_factor(step, warmup=warmup, total=total) == expected
    assert linear_warmup_factor(total, warmup=warmup, total=total) == 0

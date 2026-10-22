"""partial SFT billing preserves the exact profile's token invariants."""

import math
from dataclasses import replace

import pytest

from flash.cost.analytical import estimate_cost
from flash.cost.types import RunConfig
from flash.runner.accounting.costs import cancelled_charge_usd, charge_usd_for_spec
from flash.runner.lifecycle.state import RunStatus


@pytest.mark.parametrize("supervised", [1, 156, 480])
def test_partial_profile_remains_priceable_and_quote_bounded(monkeypatch, supervised):
    from flash.cost import spec as cost_spec

    cfg = RunConfig(
        "Qwen/Qwen3.5-9B",
        "sft",
        12,
        seq_len=512,
        batch_size=1,
        lora_rank=8,
        train_tokens=480,
        supervised_train_tokens=supervised,
        sft_packing_mode="packed",
        sft_packed_blocks=4,
        sft_retained_examples=4,
        save_at_steps=(1, 12),
        gpu_type="H100",
        gpu_count=1,
        provider="modal",
    )
    monkeypatch.setattr(cost_spec, "run_config_from_spec", lambda spec: cfg)
    spec = object()
    status = RunStatus(
        run_id="partial-profile", state="cancelled", spec={}, estimated_cost_usd=0.07
    )
    charges = []
    for completed in (0, 1, 6, 12, 20):
        charge = cancelled_charge_usd(status, spec, steps=completed, fallback=math.nan)
        assert math.isfinite(charge)
        assert 0 <= charge <= 0.07
        charges.append(charge)
    assert charges == sorted(charges)
    assert charges[0] == 0
    assert 0 < charges[1] < 0.07
    assert charges[-1] == charges[-2] == 0.07
    expected = replace(
        cfg,
        steps=1,
        train_tokens=40,
        supervised_train_tokens=max(1, supervised // 12),
        save_at_steps=(1,),
    )
    assert charge_usd_for_spec(spec, steps=1, fallback=math.nan) == float(
        estimate_cost(expected).total_usd
    )
    assert cfg.train_tokens == 480
    assert cfg.supervised_train_tokens == supervised

"""Customer charges retain the chosen Spot tariff through completion and recovery."""

from dataclasses import replace

import pytest

from flash.core.spec import GpuSpec
from flash.cost.currency import usd_amount
from flash.cost.spec import estimate_for_spec
from flash.runner.accounting import costs
from flash.runner.lifecycle.state import RunStatus
from tests.test_charge_pricing import _spec, _write_terminal_steps


def spot_spec(gpu="H100", *, legacy_preference=False):
    choice = {"providers": ("spot",)} if legacy_preference else {"spot": True}
    return replace(_spec(), gpu=GpuSpec(type=gpu, count=1, **choice))


@pytest.mark.parametrize(("gpu", "hourly"), [("H100", 1.0), ("H200", 1.0), ("B200", 4.0)])
@pytest.mark.parametrize("provider_hint", ["", "spot", "modal"])
@pytest.mark.parametrize("legacy_preference", [False, True])
def test_spot_charge_uses_spot_tariff_even_with_missing_or_stale_provider(
    gpu, hourly, provider_hint, legacy_preference
):
    spec = spot_spec(gpu, legacy_preference=legacy_preference)
    quote = estimate_for_spec(spec)
    assert quote.provider == "spot"
    assert quote.gpu_hourly_usd == hourly
    charge = costs.charge_usd_for_spec(
        spec, provider=provider_hint, gpu_type=gpu, gpu_count=1, fallback=float("nan")
    )
    assert charge == pytest.approx(quote.total_usd)
    regular = replace(spec, gpu=GpuSpec(type=gpu, count=1, provider="modal"))
    assert charge < costs.charge_usd_for_spec(regular)


@pytest.mark.parametrize("gpu", ["H100", "H200", "B200"])
@pytest.mark.parametrize("completed_steps", [0, 10, 20])
def test_cancelled_spot_charge_prorates_the_accepted_spot_quote(gpu, completed_steps):
    spec = spot_spec(gpu)
    full = costs.charge_usd_for_spec(spec)
    quote = usd_amount(full)
    status = RunStatus(
        run_id=spec.run_id, state="cancelled", spec=spec.to_dict(), estimated_cost_usd=quote
    )
    # Recovery may retain the GPU shape after the provider field or entire handle is lost.
    rented = {"allocated_gpu": gpu, "allocated_gpu_count": 1}
    actual = costs.cancelled_charge_usd(status, spec, steps=completed_steps, rented_remote=rented)
    partial = costs.charge_usd_for_spec(spec, steps=completed_steps)
    assert actual == pytest.approx(quote * partial / full)
    assert 0 <= actual <= quote


@pytest.mark.parametrize("gpu", ["H100", "H200", "B200"])
def test_completed_spot_recovery_charges_quote_not_expensive_worker_wall(
    tmp_path, monkeypatch, gpu
):
    spec = spot_spec(gpu)
    quote = usd_amount(estimate_for_spec(spec).total_usd)
    status = RunStatus(
        run_id=spec.run_id, state="done", spec=spec.to_dict(), estimated_cost_usd=quote
    )
    _write_terminal_steps(
        tmp_path,
        monkeypatch,
        spec,
        20,
        wall_seconds=100_000,
        allocated_gpu=gpu,
        allocated_gpu_count=1,
    )
    assert costs._status_estimated_charge(status, spec, fallback=999.0) == quote


def test_missing_provider_preserves_eight_gpu_spot_billing_shape():
    spec = replace(spot_spec(), gpu=GpuSpec(spot=True, type="H100", count=8))
    allocation = costs._pinned_offline_allocation("spot", "H100", 8)
    expected = estimate_for_spec(spec, allocation=allocation)
    charge = costs.charge_usd_for_spec(spec, gpu_type="H100", gpu_count=8, fallback=float("nan"))
    assert expected.gpu_count == 8
    assert expected.gpu_hourly_usd == 1
    assert charge == pytest.approx(expected.total_usd)

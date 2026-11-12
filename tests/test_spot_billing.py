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


@pytest.mark.parametrize(("gpu", "hourly"), [("H100", 2.0), ("H200", 2.5)])
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


@pytest.mark.parametrize("gpu", ["H100", "H200"])
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


@pytest.mark.parametrize("gpu", ["H100", "H200"])
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
    assert expected.gpu_hourly_usd == 2.0
    assert charge == pytest.approx(expected.total_usd)


# --- quote the costliest interchangeable class, bill the class actually rented -----------------


def _remote(rented, *, quoted="H100", count=1):
    remote = {"provider": "spot", "gpu": rented, "gpu_count": count}
    if quoted:
        remote["quoted_gpu"] = quoted
    return remote


def test_quote_rate_covers_every_class_the_run_may_land_on():
    from flash.providers.mithril.pricing import quote_rate

    assert quote_rate("H100", 1) == 2.5  # an H100 run may launch on an H200
    assert quote_rate("H200", 1) == 2.5


def test_allocator_quotes_spot_at_the_quote_rate():
    from flash.providers.core.allocator import _cheapest_allocation
    from flash.providers.core.base import Candidate

    allocation = _cheapest_allocation(
        [Candidate("spot", "H100", 1.4, 80, 1)], need=47, cost_per_step=None, provider_rank={}
    )
    assert (allocation.gpu, allocation.hourly_usd) == ("H100", 2.5)


@pytest.mark.parametrize(
    ("remote", "factor"),
    [
        (_remote("H100"), 0.8),  # quoted at $2.50, ran on an H100 at $2.00
        (_remote("H200"), 1.0),  # quoted at $2.50, ran on an H200 at $2.50
        (_remote("H200", quoted="H200"), 1.0),
        (_remote("H100", quoted=""), 1.0),  # handle from before: quoted at its own class rate
        ({"provider": "modal", "gpu": "H100", "quoted_gpu": "H100"}, 1.0),
    ],
)
def test_billed_rate_factor_is_rented_rate_over_quoted_rate(remote, factor):
    from flash.providers.mithril.pricing import billed_rate_factor

    assert costs._billed_rate_factor(remote, None) == pytest.approx(factor)
    if remote.get("quoted_gpu"):
        assert billed_rate_factor(remote) == pytest.approx(factor)


@pytest.mark.parametrize(("rented", "expected"), [("H100", 8.0), ("H200", 10.0), (None, 10.0)])
def test_completed_spot_run_is_charged_for_the_class_it_rented(
    tmp_path, monkeypatch, rented, expected
):
    from flash.runner.lifecycle import state

    monkeypatch.setattr(state, "artifacts_dir", lambda spec: str(tmp_path))  # no metrics.json
    remote = _remote(rented) if rented else None
    status = RunStatus(
        run_id="r", state="done", spec=spot_spec().to_dict(), estimated_cost_usd=10.0, remote=remote
    )
    assert costs._status_estimated_charge(status, spot_spec()) == pytest.approx(expected)


def test_custom_spot_session_is_charged_for_the_class_it_rented(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from flash.custom.billing import session_charge
    from flash.runner.lifecycle import state

    monkeypatch.setattr(state, "artifacts_dir", lambda spec: str(tmp_path))
    (tmp_path / "metrics.json").write_text(json.dumps({"session_seconds": 1800.0}))
    spec = SimpleNamespace(custom={"max_wall_seconds": 3600})
    status = SimpleNamespace(
        last_heartbeat={}, estimated_cost_usd=20.0, remote=_remote("H100"), billing_remote=None
    )
    assert session_charge(status, spec) == pytest.approx(20.0 * 0.5 * 0.8)
    status.remote = _remote("H200")
    assert session_charge(status, spec) == pytest.approx(20.0 * 0.5)


def test_spot_handle_keeps_the_quoted_class_and_reads_older_handles():
    from flash.providers.mithril.handles import SpotJobHandle

    fields = {
        "instance_id": "bid_x",
        "gpu": "H200",
        "hourly_usd": 2.5,
        "attempt": 0,
        "started_ts": 1.0,
        "name": "n",
        "run_id": "r",
        "hf_repo": "o/r",
        "phase": "sft",
        "scope": "s",
        "gpu_count": 1,
    }
    handle = SpotJobHandle(**fields, quoted_gpu="H100")
    assert SpotJobHandle.from_dict(handle.to_dict()).quoted_gpu == "H100"
    legacy = SpotJobHandle(**fields).to_dict()
    assert "quoted_gpu" not in legacy
    assert SpotJobHandle.from_dict(legacy).quoted_gpu == ""


@pytest.mark.parametrize(
    ("chosen", "remote", "expected"),
    [
        (("spot", "H100"), {"gpu": "H200"}, ("H200", 141)),  # substituted: record what ran
        (("spot", "H100"), {"gpu": "H100"}, ("H100", 80)),
        (("spot", "H100"), {}, ("H100", 80)),
        (("modal", "H100"), {"gpu": "H200"}, ("H100", 80)),  # only Spot substitutes
    ],
)
def test_supervisor_records_the_class_actually_rented(chosen, remote, expected):
    from flash.providers.core.base import Candidate
    from flash.runner.supervise.execution import _rented_candidate

    rented = _rented_candidate(Candidate(chosen[0], chosen[1], 2.5, 80, 1), remote)
    assert (rented.gpu, rented.vram_gb) == expected

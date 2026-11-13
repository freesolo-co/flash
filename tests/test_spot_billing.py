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


@pytest.mark.parametrize(("gpu", "hourly"), [("H100", 2.1), ("H200", 2.6)])
@pytest.mark.parametrize("provider_hint", ["", "spot", "modal"])
@pytest.mark.parametrize("legacy_preference", [False, True])
def test_spot_charge_uses_spot_tariff_even_with_missing_or_stale_provider(
    gpu, hourly, provider_hint, legacy_preference
):
    spec = spot_spec(gpu, legacy_preference=legacy_preference)
    quote = estimate_for_spec(spec)
    assert quote.provider == "spot"
    # the quote covers the costliest class the run may take (an H100 run may launch on an H200)
    assert quote.gpu_hourly_usd == 2.6
    charge = costs.charge_usd_for_spec(
        spec, provider=provider_hint, gpu_type=gpu, gpu_count=1, fallback=float("nan")
    )
    # repriced on the rented class's own rate: exactly the quote scaled to that class
    assert charge == pytest.approx(quote.total_usd * hourly / 2.6)
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
    assert expected.gpu_hourly_usd == 2.1
    assert charge == pytest.approx(expected.total_usd)


# --- quote at the top Spot rate, bill each stretch at the class it actually ran on -------------


def _status(spec, remote_gpu, **fields):
    """A run submitted under the current tariff, the way submit records it."""
    from flash.providers.mithril.pricing import tariff

    remote = {"provider": "spot", "gpu": remote_gpu} if remote_gpu else None
    fields.setdefault("submitted_spot_rates", tariff())
    return RunStatus(run_id="r", state="running", spec=spec.to_dict(), remote=remote, **fields)


def _beat(status, hb):
    """Feed one heartbeat through the usage ledger the way record_heartbeat does."""
    from flash.runner.accounting.spot_usage import record

    prev = status.last_heartbeat
    same = isinstance(prev, dict) and prev.get("attempt") == hb.get("attempt")
    record(status, prev, hb, same_attempt=same)
    status.last_heartbeat = hb


def test_every_spot_quote_uses_the_top_spot_rate():
    from flash.providers.core.allocator import _cheapest_allocation
    from flash.providers.core.base import Candidate
    from flash.providers.mithril.pricing import quote_rate

    assert quote_rate() == 2.6
    allocation = _cheapest_allocation(
        [Candidate("spot", "H100", 1.4, 80, 1)], need=47, cost_per_step=None, provider_rank={}
    )
    assert (allocation.gpu, allocation.hourly_usd) == ("H100", 2.6)


@pytest.mark.parametrize(
    ("usage", "factor"),
    [
        ({"H100": 10}, 2.1 / 2.6),  # all on H100: $2.10 of the $2.60 quote rate
        ({"H200": 10}, 1.0),
        ({"H100": 5, "H200": 5}, (2.1 + 2.6) / (2 * 2.6)),  # half and half
        (None, 1.0),  # no ledger (runs launched before this accounting): the quote
    ],
)
def test_rate_factor_weights_usage_by_class(usage, factor):
    from types import SimpleNamespace

    from flash.providers.mithril.pricing import tariff
    from flash.runner.accounting.spot_usage import rate_factor

    hb = {"spot_usage_by_gpu": usage} if usage is not None else {}
    status = SimpleNamespace(last_heartbeat=hb, submitted_spot_rates=tariff())
    assert rate_factor(status) == pytest.approx(factor)


@pytest.mark.parametrize(
    ("usage", "factor"),
    [
        ({"H100": 10}, 0.8),  # $2.00 of the $2.50 quote rate the run accepted
        ({"H200": 10}, 1.0),
        ({"H100": 5, "H200": 5}, 0.9),
    ],
)
def test_runs_quoted_before_rates_were_recorded_settle_on_the_legacy_tariff(usage, factor):
    """A run submitted under the old $2.00/$2.50 tariff is never repriced by a later rate change:
    an H100-only run with an accepted $2.50 quote still settles at $2.00, not $2.50 x 2.10/2.60."""
    from types import SimpleNamespace

    from flash.runner.accounting.spot_usage import rate_factor

    legacy = SimpleNamespace(last_heartbeat={"spot_usage_by_gpu": usage})  # no recorded rates
    assert rate_factor(legacy) == pytest.approx(factor)
    assert 2.5 * rate_factor(
        SimpleNamespace(last_heartbeat={"spot_usage_by_gpu": {"H100": 1}})
    ) == (pytest.approx(2.0))


def test_recorded_submit_rates_survive_a_later_rate_change(monkeypatch):
    from decimal import Decimal
    from types import SimpleNamespace

    from flash.providers.mithril import pricing
    from flash.runner.accounting.spot_usage import rate_factor

    status = SimpleNamespace(
        last_heartbeat={"spot_usage_by_gpu": {"H100": 10}},
        submitted_spot_rates={"H100": 2.1, "H200": 2.6},
    )
    monkeypatch.setattr(
        pricing,
        "SPOT_GPUS",
        {"H100": pricing.SpotGpu(Decimal("3.00")), "H200": pricing.SpotGpu(Decimal("4.00"))},
    )
    assert rate_factor(status) == pytest.approx(2.1 / 2.6)


def test_steps_are_credited_to_the_class_that_reached_them_once():
    spec = spot_spec()
    status = _status(spec, "H100")
    for step in (1, 2, 3, 4):
        _beat(status, {"attempt": 0, "stage": "sft_step", "step": step})
    status.remote = {"provider": "spot", "gpu": "H200"}  # preempted; replacement is an H200
    for step in (3, 4, 5, 6):  # resumed from step 2: steps 3-4 repeat and are not billed twice
        _beat(status, {"attempt": 1, "stage": "sft_step", "step": step})
    _beat(status, {"attempt": 1, "stage": "checkpoint_deployable"})  # no step: totals carry
    assert status.last_heartbeat["spot_usage_by_gpu"] == {"H100": 4.0, "H200": 2.0}


def test_custom_session_mixing_classes_is_billed_per_attempt(tmp_path, monkeypatch):
    """Review case: a 1 h $2.60 quote, 900 s on H100 then 900 s on H200 charges $1.175."""
    import json
    from types import SimpleNamespace

    from flash.custom.billing import session_charge
    from flash.runner.lifecycle import state

    custom = replace(spot_spec(), algorithm="custom")
    status = _status(custom, "H100")
    _beat(status, {"attempt": 0, "stage": "custom_step", "session_seconds": 900.0})
    status.remote = {"provider": "spot", "gpu": "H200"}
    _beat(status, {"attempt": 1, "stage": "custom_step", "session_seconds": 900.0})
    assert status.last_heartbeat["spot_usage_by_gpu"] == {"H100": 900.0, "H200": 900.0}

    monkeypatch.setattr(state, "artifacts_dir", lambda spec: str(tmp_path))
    (tmp_path / "metrics.json").write_text(json.dumps({"session_seconds": 900.0}))
    status.last_heartbeat["session_seconds_prior"] = 900.0
    status.estimated_cost_usd = 2.6
    charged = session_charge(status, SimpleNamespace(custom={"max_wall_seconds": 3600}))
    # 1,800 s of a 3,600 s cap, priced (900 x $2.10 + 900 x $2.60) / (1,800 x $2.60)
    assert charged == pytest.approx(2.6 * 0.5 * (2.1 + 2.6) / (2 * 2.6))  # $1.175


@pytest.mark.parametrize(
    ("usage", "hourly"), [({"H100": 20}, 2.1), ({"H200": 20}, 2.6), (None, 2.6)]
)
def test_completed_spot_run_is_charged_for_the_classes_it_rented(
    tmp_path, monkeypatch, usage, hourly
):
    from flash.runner.lifecycle import state

    monkeypatch.setattr(state, "artifacts_dir", lambda spec: str(tmp_path))  # no metrics.json
    spec = spot_spec("H100")
    quote = estimate_for_spec(spec).total_usd  # what prepare_job persists
    hb = {"spot_usage_by_gpu": usage} if usage else {}
    status = _status(spec, "H100", estimated_cost_usd=quote, last_heartbeat=hb)
    charged = costs._status_estimated_charge(status, spec)
    assert charged == pytest.approx(quote * hourly / 2.6)
    assert charged <= quote


def test_allocation_admits_a_pinned_class_even_when_only_another_is_cheap(monkeypatch):
    """Review case: allocation never reads live prices, so an H100-pinned run is admitted and
    the queue can then take an H200 that is below its cap."""
    from flash.providers.core.base import AllocationConstraints
    from flash.providers.core.registry import get_provider
    from flash.providers.mithril import prices

    monkeypatch.setattr(prices, "affordable_offers", lambda **kw: pytest.fail("no live prices"))
    candidates = get_provider("spot").live_candidates(
        47, AllocationConstraints(max_gpu_count=1, gpu_type="H100")
    )
    assert [(c.gpu, c.gpu_count, c.hourly_usd) for c in candidates] == [("H100", 1, 2.6)]


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


def test_submit_records_the_spot_tariff_only_for_spot_runs():
    from flash.providers.mithril.pricing import tariff
    from flash.runner.lifecycle.submit import _spot_tariff

    assert _spot_tariff(spot_spec("H100")) == tariff() == {"H100": 2.1, "H200": 2.6}
    regular = replace(spot_spec(), gpu=GpuSpec(type="H100", count=1, provider="modal"))
    assert _spot_tariff(regular) is None

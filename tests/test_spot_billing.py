"""Customer charges retain the chosen Spot tariff through completion and recovery."""

from dataclasses import replace

import pytest

from flash.core.spec import GpuSpec
from flash.runner.lifecycle.state import RunStatus
from tests.test_charge_pricing import _spec


def spot_spec(gpu="H100", *, legacy_preference=False):
    choice = {"providers": ("spot",)} if legacy_preference else {"spot": True}
    return replace(_spec(), gpu=GpuSpec(type=gpu, count=1, **choice))


# --- quote at the top Spot rate, bill each stretch at the class it actually ran on -------------


def _status(spec, remote_gpu, **fields):
    remote = {"provider": "spot", "gpu": remote_gpu} if remote_gpu else None
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

    assert quote_rate() == 2.5
    allocation = _cheapest_allocation(
        [Candidate("spot", "H100", 1.4, 80, 1)], need=47, cost_per_step=None, provider_rank={}
    )
    assert (allocation.gpu, allocation.hourly_usd) == ("H100", 2.5)


@pytest.mark.parametrize(
    ("usage", "factor"),
    [
        ({"H100": 10}, 0.8),  # all on H100: $2.00 of the $2.50 quote rate
        ({"H200": 10}, 1.0),
        ({"H100": 5, "H200": 5}, 0.9),  # half and half
        (None, 1.0),  # no ledger (runs launched before this accounting): the quote
    ],
)
def test_rate_factor_weights_usage_by_class(usage, factor):
    from types import SimpleNamespace

    from flash.runner.accounting.spot_usage import rate_factor

    hb = {"spot_usage_by_gpu": usage} if usage is not None else {}
    assert rate_factor(SimpleNamespace(last_heartbeat=hb)) == pytest.approx(factor)


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
    """Review case: a 1 h $2.50 quote, 900 s on H100 then 900 s on H200 charges $1.125."""
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
    status.estimated_cost_usd = 2.5
    charged = session_charge(status, SimpleNamespace(custom={"max_wall_seconds": 3600}))
    # 1,800 s of a 3,600 s cap, priced (900 x $2.00 + 900 x $2.50) / (1,800 x $2.50) = 0.9
    assert charged == pytest.approx(2.5 * 0.5 * 0.9)  # $1.125


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
    assert [(c.gpu, c.gpu_count, c.hourly_usd) for c in candidates] == [("H100", 1, 2.5)]


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

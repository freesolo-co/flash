"""Realized provider-cost reconciliation: pure cost shaping (RunPod / Vast), the provider
dispatch, and the reconcile selection/report logic. Offline -- the provider HTTP calls and the
backend POST are stubbed, so nothing touches the network."""

from __future__ import annotations

from flash import runner
from flash.providers import realized
from flash.providers.runpod.cost import shape_endpoint_cost
from flash.providers.vast.cost import shape_instance_cost
from flash.server import reconcile


# --------------------------------------------------------------------------- RunPod shaping
def test_runpod_shape_sums_amount_and_filters_by_endpoint():
    rows = [
        {"endpointId": "ep-1", "amount": 4.5, "timeBilledMs": 1_800_000},
        {"endpointId": "ep-1", "amount": 1.25, "timeBilledMs": 600_000},
        {"endpointId": "ep-other", "amount": 99.0, "timeBilledMs": 1_000},  # different run
    ]
    rc = shape_endpoint_cost(rows, endpoint_id="ep-1")
    assert rc.provider == "runpod"
    assert rc.realized_usd == 5.75
    assert rc.by_resource == {"gpu": 5.75}
    assert rc.wall_seconds == 2400.0  # (1_800_000 + 600_000) ms
    assert rc.source == {"endpoint_id": "ep-1"}


def test_runpod_shape_empty_is_zero():
    rc = shape_endpoint_cost([], endpoint_id="ep-1")
    assert rc.realized_usd == 0.0
    assert rc.wall_seconds is None


# --------------------------------------------------------------------------- Vast shaping
def test_vast_shape_matches_instance_and_itemizes():
    rows = [
        {
            "source": "instance-12345",
            "amount": 9.87,
            "items": [
                {"type": "gpu", "amount": 9.5},
                {"type": "disk", "amount": 0.3},
                {"type": "bwd", "amount": 0.07},
            ],
        },
        {"source": "instance-99999", "amount": 50.0, "items": []},  # different run
    ]
    rc = shape_instance_cost(rows, instance_id=12345)
    assert rc.provider == "vast"
    assert rc.realized_usd == 9.87
    assert rc.by_resource == {"gpu": 9.5, "disk": 0.3, "bwd": 0.07}  # captures storage + bandwidth
    assert rc.source == {"instance_id": 12345}


def test_vast_shape_unitemized_falls_back_to_gpu():
    rows = [{"source": "instance-7", "amount": 2.0}]
    rc = shape_instance_cost(rows, instance_id=7)
    assert rc.realized_usd == 2.0
    assert rc.by_resource == {"gpu": 2.0}


# --------------------------------------------------------------------------- provider dispatch
def test_dispatch_runpod(monkeypatch):
    from flash.providers.runpod import api

    monkeypatch.setattr(
        api, "billing_endpoints", lambda **kw: [{"endpointId": "ep-1", "amount": 3.0}]
    )
    rc = realized.realized_cost_for_remote(
        {"provider": "runpod", "endpoint_id": "ep-1"}, start=0, end=100
    )
    assert rc is not None
    assert rc.provider == "runpod"
    assert rc.realized_usd == 3.0


def test_dispatch_vast(monkeypatch):
    from flash.providers.vast import api

    monkeypatch.setattr(api, "get_charges", lambda **kw: [{"source": "instance-5", "amount": 1.5}])
    rc = realized.realized_cost_for_remote({"provider": "vast", "instance_id": 5}, start=0, end=100)
    assert rc is not None
    assert rc.provider == "vast"
    assert rc.realized_usd == 1.5


def test_dispatch_none_when_no_handle_or_unknown_provider():
    assert realized.realized_cost_for_remote(None, start=0, end=1) is None
    assert realized.realized_cost_for_remote({}, start=0, end=1) is None
    assert realized.realized_cost_for_remote({"provider": "gcp"}, start=0, end=1) is None
    # runpod handle with no endpoint id -> nothing to query.
    assert realized.realized_cost_for_remote({"provider": "runpod"}, start=0, end=1) is None


# --------------------------------------------------------------------------- reconcile selection
def _status(**kw) -> runner.RunStatus:
    base = {"run_id": "r1", "state": "done", "spec": {}, "created_at": 0.0, "updated_at": 0.0}
    base.update(kw)
    return runner.RunStatus(**base)


def test_due_requires_billable_terminal_settled_unreconciled_with_handle():
    now = 1_000_000.0
    settled = now - 7200  # 2h ago (past the 1h settle delay, within the 7d window)
    handle = {"provider": "runpod", "endpoint_id": "ep-1"}

    assert reconcile._due(_status(state="done", updated_at=settled, remote=handle), now)
    # deployed runs have FINISHED training (cost is final) -> still reconciled
    assert reconcile._due(_status(state="deployed", updated_at=settled, remote=handle), now)
    # not terminal yet
    assert not reconcile._due(_status(state="running", updated_at=settled, remote=handle), now)
    # dry_run spent no GPU
    assert not reconcile._due(_status(state="dry_run", updated_at=settled, remote=handle), now)
    # too fresh (billing hasn't settled)
    assert not reconcile._due(_status(state="done", updated_at=now - 60, remote=handle), now)
    # too old (aged out of the window)
    assert not reconcile._due(_status(state="done", updated_at=now - 8 * 86400, remote=handle), now)
    # already reconciled
    assert not reconcile._due(
        _status(state="done", updated_at=settled, remote=handle, reconciled_at=now - 1), now
    )
    # no provider handle to attribute cost
    assert not reconcile._due(_status(state="done", updated_at=settled, remote=None), now)


def test_reconcile_run_reports_and_persists(monkeypatch):
    now = 1_000_000.0
    status = _status(
        run_id="r-pos",
        updated_at=now - 7200,
        remote={"provider": "vast", "instance_id": 5, "allocated_gpu": "RTX 5090"},
    )
    monkeypatch.setattr(
        reconcile,
        "realized_cost_for_remote",
        lambda remote, **kw: realized.RealizedCost(
            provider="vast", realized_usd=4.2, by_resource={"gpu": 4.0, "disk": 0.2}
        ),
    )
    posted: dict = {}
    monkeypatch.setattr(reconcile, "_report", lambda body: posted.update(body) or True)
    updates: dict = {}
    monkeypatch.setattr(
        runner,
        "record_realized_cost",
        lambda run_id, **kw: updates.update(run_id=run_id, **kw),
    )

    assert reconcile.reconcile_run(status, now=now) is True
    assert posted["runId"] == "r-pos"
    assert posted["realizedCostUsd"] == 4.2
    assert posted["provider"] == "vast"
    assert posted["gpu"] == "RTX 5090"
    assert posted["costBasis"] == "realized"
    # persisted locally via the cost-only writer (never touches state) with the realized figure
    # + a reconcile marker.
    assert updates["run_id"] == "r-pos"
    assert "state" not in updates
    assert updates["realized_cost_usd"] == 4.2
    assert updates["reconciled_at"] == now


def test_reconcile_run_skips_zero_and_unreported(monkeypatch):
    now = 1_000_000.0
    status = _status(updated_at=now - 7200, remote={"provider": "runpod", "endpoint_id": "e"})

    # zero realized cost (invoice not settled) -> not reconciled, retried later
    monkeypatch.setattr(
        reconcile,
        "realized_cost_for_remote",
        lambda remote, **kw: realized.RealizedCost(provider="runpod", realized_usd=0.0),
    )
    monkeypatch.setattr(reconcile, "_report", lambda body: True)
    assert reconcile.reconcile_run(status, now=now) is False

    # positive cost but the report POST fails -> not marked reconciled
    monkeypatch.setattr(
        reconcile,
        "realized_cost_for_remote",
        lambda remote, **kw: realized.RealizedCost(provider="runpod", realized_usd=1.0),
    )
    monkeypatch.setattr(reconcile, "_report", lambda body: False)
    assert reconcile.reconcile_run(status, now=now) is False


def test_reconcile_run_does_not_revert_status_advanced_after_snapshot(tmp_path, monkeypatch):
    # Regression (HIGH): reconcile_run holds a status SNAPSHOT taken when the run was `done`,
    # but the run can advance to `deployed` while provider billing is pulled. Persisting must
    # update ONLY the cost columns and keep the run's CURRENT state -- it must NOT write the
    # stale `done` back and revert the live deployment (the terminal-sticky CAS wouldn't catch
    # it, since `deployed` is non-terminal). Exercises the real on-disk record_realized_cost.
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path))
    now = 1_000_000.0
    snapshot = _status(
        run_id="r-adv",
        state="done",  # state as seen when the reconcile snapshot was taken
        updated_at=now - 7200,
        remote={"provider": "runpod", "endpoint_id": "ep-1"},
    )
    # The run has since advanced to `deployed` on disk (serving stood up on the finished run).
    runner._save_status(
        runner.RunStatus(
            run_id="r-adv",
            state="deployed",
            spec={},
            created_at=0.0,
            updated_at=now - 100,
            remote={"provider": "runpod", "endpoint_id": "ep-1"},
            deployment={"state": "active"},
        )
    )
    monkeypatch.setattr(
        reconcile,
        "realized_cost_for_remote",
        lambda remote, **kw: realized.RealizedCost(provider="runpod", realized_usd=3.3),
    )
    monkeypatch.setattr(reconcile, "_report", lambda body: True)

    assert reconcile.reconcile_run(snapshot, now=now) is True
    persisted = runner.get_status("r-adv")
    # Status preserved (NOT reverted to the stale `done`), cost fields written.
    assert persisted.state == "deployed"
    assert persisted.deployment == {"state": "active"}
    assert persisted.realized_cost_usd == 3.3
    assert persisted.reconciled_at == now


def test_reconcile_once_disabled_without_internal_key(monkeypatch):
    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    monkeypatch.setattr(runner, "list_runs", lambda: [_status(updated_at=0)])
    assert reconcile.reconcile_once(now=1_000_000.0) == 0


def test_reconcile_once_sweeps_due_runs(monkeypatch):
    now = 1_000_000.0
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "k")
    due = _status(
        run_id="due", updated_at=now - 7200, remote={"provider": "runpod", "endpoint_id": "e"}
    )
    not_due = _status(
        run_id="fresh", updated_at=now - 60, remote={"provider": "runpod", "endpoint_id": "e"}
    )
    monkeypatch.setattr(runner, "list_runs", lambda: [due, not_due])
    seen: list[str] = []

    def fake_reconcile_run(status, *, now):
        seen.append(status.run_id)
        return True

    monkeypatch.setattr(reconcile, "reconcile_run", fake_reconcile_run)
    assert reconcile.reconcile_once(now=now) == 1
    assert seen == ["due"]

"""Reconciliation reports the provider bill net of the worker-measured startup.

The quote bills training wall only (cold start is reported, never charged), so the realized figure
the accuracy view compares it against must drop startup too. See reconcile._exclude_startup.
"""

from __future__ import annotations

import json

import pytest

from flash import runner
from flash.providers import realized
from flash.server.domain import reconcile


def _status(**kw) -> runner.RunStatus:
    base = {"run_id": "r1", "state": "done", "spec": {}, "created_at": 0.0, "updated_at": 0.0}
    base.update(kw)
    if "finished_at" not in base:
        base["finished_at"] = base["updated_at"]
    return runner.RunStatus(**base)


def _write_metrics(tmp_path, **fields) -> str:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metrics.json").write_text(json.dumps(fields))
    return str(artifacts)


def _capture_reconcile(monkeypatch, realized_cost):
    posted: dict = {}
    updates: dict = {}
    monkeypatch.setattr(reconcile, "realized_cost_for_remote", lambda remote, **kw: realized_cost)
    monkeypatch.setattr(reconcile, "_report", lambda body: posted.update(body) or True)
    monkeypatch.setattr(
        runner,
        "record_realized_cost",
        lambda run_id, **kw: updates.update(run_id=run_id, **kw),
    )
    return posted, updates


def test_reconcile_strips_measured_startup_from_the_provider_bill(tmp_path, monkeypatch):
    # 1200s billed, 300s of measured setup -> three quarters of the $4 bill is training
    now = 1_000_000.0
    status = _status(
        run_id="r-setup",
        updated_at=now - 7200,
        remote={"provider": "runpod", "endpoint_id": "ep-5", "allocated_gpu": "RTX 5090"},
        artifacts_dir=_write_metrics(tmp_path, setup_seconds=300.0, wall_seconds=850.0),
    )
    posted, updates = _capture_reconcile(
        monkeypatch,
        realized.RealizedCost(
            provider="runpod",
            realized_usd=4.0,
            by_resource={"gpu": 4.0},
            wall_seconds=1200.0,
            source={"endpoint_id": "ep-5"},
        ),
    )

    assert reconcile.reconcile_run(status, now=now) is True
    assert posted["realizedCostUsd"] == 3.0
    assert posted["costByResource"] == {"gpu": 3.0}
    assert posted["wallSeconds"] == 900.0
    assert posted["costBasis"] == "realized"
    # the gross bill and the inputs to the proration are preserved for audit
    assert posted["source"] == {
        "endpoint_id": "ep-5",
        "grossRealizedUsd": 4.0,
        "providerWallSeconds": 1200.0,
        "setupSeconds": 300.0,
        "startupExcluded": True,
    }
    assert updates["realized_cost_usd"] == 3.0
    assert updates["reconciled_at"] == now


def test_reconcile_prorates_from_instance_lifetime_when_the_pull_has_no_wall(tmp_path, monkeypatch):
    # runpod rows without timeBilledMs carry no wall; fall back to started_ts -> finished_at
    now = 1_000_000.0
    finished = now - 7200
    status = _status(
        run_id="r-nowall",
        updated_at=finished,
        finished_at=finished,
        remote={"provider": "runpod", "endpoint_id": "ep-5", "started_ts": finished - 1000.0},
        artifacts_dir=_write_metrics(tmp_path, setup_seconds=250.0),
    )
    posted, _updates = _capture_reconcile(
        monkeypatch,
        realized.RealizedCost(provider="runpod", realized_usd=2.0, by_resource={"gpu": 2.0}),
    )

    assert reconcile.reconcile_run(status, now=now) is True
    assert posted["realizedCostUsd"] == 1.5
    assert posted["wallSeconds"] == 750.0
    assert posted["source"]["providerWallSeconds"] == 1000.0
    assert posted["source"]["startupExcluded"] is True


def test_reconcile_reports_the_gross_bill_when_startup_is_unmeasured(tmp_path, monkeypatch):
    # no metrics.json (pre-instrumentation record, or a run that never reached training): report
    # the bill as pulled and say so, rather than guessing a startup to subtract
    now = 1_000_000.0
    status = _status(
        run_id="r-nometrics",
        updated_at=now - 7200,
        remote={"provider": "runpod", "endpoint_id": "ep-5"},
        artifacts_dir=str(tmp_path / "missing"),
    )
    posted, updates = _capture_reconcile(
        monkeypatch,
        realized.RealizedCost(
            provider="runpod", realized_usd=4.2, by_resource={"gpu": 4.2}, wall_seconds=1200.0
        ),
    )

    assert reconcile.reconcile_run(status, now=now) is True
    assert posted["realizedCostUsd"] == 4.2
    assert posted["wallSeconds"] == 1200.0
    assert posted["source"]["startupExcluded"] is False
    assert posted["source"]["setupSeconds"] is None
    assert posted["source"]["grossRealizedUsd"] == 4.2
    assert updates["realized_cost_usd"] == 4.2


@pytest.mark.parametrize("bad_setup", [-1.0, "300", True, float("nan")])
def test_reconcile_ignores_an_unusable_setup_stamp(tmp_path, monkeypatch, bad_setup):
    now = 1_000_000.0
    status = _status(
        run_id="r-badsetup",
        updated_at=now - 7200,
        remote={"provider": "runpod", "endpoint_id": "ep-5"},
        artifacts_dir=_write_metrics(tmp_path, setup_seconds=bad_setup),
    )
    posted, _updates = _capture_reconcile(
        monkeypatch,
        realized.RealizedCost(
            provider="runpod", realized_usd=4.2, by_resource={"gpu": 4.2}, wall_seconds=1200.0
        ),
    )

    assert reconcile.reconcile_run(status, now=now) is True
    assert posted["realizedCostUsd"] == 4.2
    assert posted["source"]["startupExcluded"] is False


def test_reconcile_clamps_startup_to_the_wall_and_still_reconciles(tmp_path, monkeypatch):
    # a cold start longer than the whole billed wall nets to $0 training cost. that is reported
    # (and the run marked reconciled) rather than left to retry forever: the provider did bill,
    # and the quote for a run like this is exactly the miss the accuracy view should surface
    now = 1_000_000.0
    status = _status(
        run_id="r-clamp",
        updated_at=now - 7200,
        remote={"provider": "vast", "instance_id": "i-1"},
        artifacts_dir=_write_metrics(tmp_path, setup_seconds=5000.0),
    )
    posted, updates = _capture_reconcile(
        monkeypatch,
        realized.RealizedCost(
            provider="vast", realized_usd=1.0, by_resource={"gpu": 1.0}, wall_seconds=1200.0
        ),
    )

    assert reconcile.reconcile_run(status, now=now) is True
    assert posted["realizedCostUsd"] == 0.0
    assert posted["wallSeconds"] == 0.0
    assert posted["source"]["grossRealizedUsd"] == 1.0
    assert posted["source"]["setupSeconds"] == 5000.0
    assert updates["realized_cost_usd"] == 0.0
    assert updates["reconciled_at"] == now


def test_reconcile_still_skips_a_zero_provider_bill(monkeypatch):
    # the startup step runs only after a positive pull; a zero pull stays unreconciled to retry
    now = 1_000_000.0
    status = _status(
        run_id="r-zero", updated_at=now - 7200, remote={"provider": "runpod", "endpoint_id": "ep"}
    )
    posted, updates = _capture_reconcile(
        monkeypatch,
        realized.RealizedCost(provider="runpod", realized_usd=0.0, wall_seconds=1200.0),
    )

    assert reconcile.reconcile_run(status, now=now) is False
    assert posted == {}
    assert updates == {}

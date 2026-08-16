"""The durable attempt-resource ledger: every paid resource a run created, and how it settles.

Offline -- no provider or backend calls; teardown and billing are stubbed.
"""

from __future__ import annotations

import json

import pytest

from flash import runner
from flash.providers import realized
from flash.server.domain import reconcile

RUNPOD_A = {
    "provider": "runpod",
    "endpoint_id": "ep-a",
    "endpoint_name": "flash-r1-a0",
    "key_fingerprint": "rpk-0123456789ab",
    "job_id": "job-a",
    "attempt": 0,
    "started_ts": 1000.0,
    "allocated_gpu": "B200",
    "allocated_gpu_count": 2,
}
RUNPOD_B = {**RUNPOD_A, "endpoint_id": "ep-b", "job_id": "job-b", "attempt": 1}


def _seed_run(tmp_path, monkeypatch, run_id="r1", **kw) -> runner.RunStatus:
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(runner, "_report_status", lambda status: None)
    base = {
        "run_id": run_id,
        "state": "running",
        "spec": {},
        "created_at": 1000.0,
        "updated_at": 1000.0,
    }
    base.update(kw)
    status = runner.RunStatus(**base)
    runner._save_status(status, _run_deadline_at=100_000.0, _next_attempt=0)
    return status


def _raw(run_id="r1") -> dict:
    with open(runner.runs_file_path(run_id, ".json")) as f:
        return json.load(f)


def _ledger(run_id="r1") -> list[dict]:
    return _raw(run_id).get("attempt_resources", [])


# --------------------------------------------------------------------------- identity
def test_paid_resource_key_excludes_attempt_and_job_so_one_endpoint_bills_once():
    """The billing key is the RESOURCE, not the attempt that touched it.

    `_remote_resource_identity` (used for active-handle compare-and-set) includes attempt and
    job_id; keying the ledger that way would bill one endpoint once per attempt that referenced it.
    """
    retry_view = {**RUNPOD_A, "attempt": 3, "job_id": "job-later"}
    assert runner._attempt_resource_key(RUNPOD_A) == runner._attempt_resource_key(retry_view)
    assert runner._attempt_resource_key(RUNPOD_A) != runner._attempt_resource_key(RUNPOD_B)
    # the CAS identity deliberately still separates them -- it answers a different question.
    assert runner._remote_resource_identity(RUNPOD_A) != runner._remote_resource_identity(
        retry_view
    )


def test_paid_resource_key_is_none_for_handles_naming_nothing_billable():
    assert runner._attempt_resource_key(None) is None
    assert runner._attempt_resource_key({}) is None
    assert runner._attempt_resource_key({"provider": "gcp", "endpoint_id": "x"}) is None
    assert runner._attempt_resource_key({"provider": "runpod"}) is None  # no endpoint
    assert runner._attempt_resource_key({"provider": "lambda"}) is None  # no instance


# --------------------------------------------------------------------------- durability
def test_ledger_is_private_to_raw_json_and_never_reaches_run_status(tmp_path, monkeypatch):
    """The ledger is operator accounting, so it must not appear on the public run surface."""
    status = _seed_run(tmp_path, monkeypatch)
    assert runner._update("r1", "running", remote=dict(RUNPOD_A)) is True

    assert len(_ledger()) == 1  # present in the durable record
    assert "attempt_resources" in _raw()
    reloaded = runner.get_status("r1")
    assert not hasattr(reloaded, "attempt_resources")
    assert "attempt_resources" not in reloaded.to_dict()
    assert status.run_id == "r1"


def test_ledger_survives_later_status_writes(tmp_path, monkeypatch):
    """Carry-forward: an unrelated status write must not drop the private key."""
    _seed_run(tmp_path, monkeypatch)
    runner._update("r1", "running", remote=dict(RUNPOD_A))
    runner._update("r1", "done")

    assert len(_ledger()) == 1
    assert _ledger()[0]["identity"]["endpoint_id"] == "ep-a"


def test_ledger_outlives_confirmed_cleanup_queue_removal(tmp_path, monkeypatch):
    """cleanup_remotes is a work queue that empties on confirmed deletion; the ledger is history.

    This is exactly why the queue cannot double as an accounting record: by the time billing
    settles (an hour later) the entry is normally gone.
    """
    _seed_run(tmp_path, monkeypatch)
    runner._update("r1", "running", remote=dict(RUNPOD_A))
    runner._record_cleanup_remote("r1", dict(RUNPOD_A))
    assert _raw().get("cleanup_remotes")

    runner._compare_and_remove_cleanup_remote("r1", dict(RUNPOD_A))
    assert not _raw().get("cleanup_remotes")  # queue drained
    assert len(_ledger()) == 1  # history kept


# --------------------------------------------------------------------------- upsert semantics
def test_republishing_the_same_resource_is_idempotent_and_enriches(tmp_path, monkeypatch):
    """Recovery re-persists an adopted handle; that is one resource, not two.

    The replay may carry locator fields the first write lacked (job_id lands after submission), so
    it enriches -- but it must never reset the launch time, which is when billing started.
    """
    _seed_run(tmp_path, monkeypatch)
    first = {**RUNPOD_A, "job_id": None, "started_ts": 1000.0}
    runner._update("r1", "running", remote=first)
    later = {**RUNPOD_A, "job_id": "job-a", "started_ts": 5000.0}
    runner._update("r1", "running", remote=later)

    records = _ledger()
    assert len(records) == 1
    assert records[0]["identity"]["job_id"] == "job-a"  # enriched
    assert records[0]["launched_at"] == 1000.0  # NOT moved forward


def test_same_paid_resource_under_a_different_attempt_is_rejected(tmp_path, monkeypatch):
    """One resource cannot belong to two attempts: that is the double-billing shape."""
    _seed_run(tmp_path, monkeypatch)
    runner._record_launched_attempt_resource("r1", dict(RUNPOD_A))
    stolen = {**RUNPOD_A, "attempt": 1}

    with pytest.raises(RuntimeError, match="different attempt"):
        runner._record_launched_attempt_resource("r1", stolen)
    assert len(_ledger()) == 1


def test_each_retry_resource_gets_its_own_record(tmp_path, monkeypatch):
    _seed_run(tmp_path, monkeypatch)
    runner._update("r1", "running", remote=dict(RUNPOD_A))
    runner._update("r1", "running", remote=dict(RUNPOD_B))

    assert [r["identity"]["endpoint_id"] for r in _ledger()] == ["ep-a", "ep-b"]
    assert [r["attempt"] for r in _ledger()] == [0, 1]


def test_malformed_stored_ledger_fails_closed(tmp_path, monkeypatch):
    """Corrupt financial history must never be silently dropped."""
    _seed_run(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError):
        runner._attempt_resources_from_raw({"attempt_resources": "not-a-list"})
    with pytest.raises(RuntimeError):
        runner._attempt_resources_from_raw({"attempt_resources": [{"identity": {}}]})
    dupe = {"identity": dict(RUNPOD_A)}
    with pytest.raises(RuntimeError, match="duplicate"):
        runner._attempt_resources_from_raw({"attempt_resources": [dupe, dict(dupe)]})
    # a missing key is a LEGACY run, not corruption
    assert runner._attempt_resources_from_raw({}) == []


# --------------------------------------------------------------------------- lifecycle stamps
def test_teardown_and_deletion_stamps_are_write_once(tmp_path, monkeypatch):
    """A retried drain must not move a billing boundary that was already observed."""
    _seed_run(tmp_path, monkeypatch)
    runner._update("r1", "running", remote=dict(RUNPOD_A))

    runner._record_attempt_resource_teardown_requested("r1", dict(RUNPOD_A))
    first_request = _ledger()[0]["teardown_requested_at"]
    runner._record_attempt_resource_deletion_confirmed("r1", dict(RUNPOD_A))
    first_delete = _ledger()[0]["deletion_confirmed_at"]
    assert first_request
    assert first_delete

    runner._record_attempt_resource_teardown_requested("r1", dict(RUNPOD_A))
    runner._record_attempt_resource_deletion_confirmed("r1", dict(RUNPOD_A))
    assert _ledger()[0]["teardown_requested_at"] == first_request
    assert _ledger()[0]["deletion_confirmed_at"] == first_delete
    # confirmed deletion is when the provider stops charging, so it closes the billing clock
    assert _ledger()[0]["terminal_at"] == first_delete


def test_cancel_marks_open_resources_without_closing_their_billing_clock(tmp_path, monkeypatch):
    """Cancellation records a disposition; only CONFIRMED deletion may set terminal_at.

    An endpoint whose deletion was never confirmed may still be running, so cancel must not declare
    its billing over.
    """
    _seed_run(tmp_path, monkeypatch)
    runner._update("r1", "running", remote=dict(RUNPOD_A))
    runner._update("r1", "running", remote=dict(RUNPOD_B))
    runner._record_attempt_resource_outcome("r1", dict(RUNPOD_A), "failed")

    runner._mark_open_attempt_resources_cancelled("r1")
    by_id = {r["identity"]["endpoint_id"]: r for r in _ledger()}
    assert by_id["ep-a"]["outcome"] == "failed"  # already dispositioned, untouched
    assert by_id["ep-b"]["outcome"] == "cancelled"
    assert by_id["ep-b"]["terminal_at"] is None  # NOT closed by cancellation alone


def test_unconfirmed_runpod_teardown_does_not_record_deletion(tmp_path, monkeypatch):
    """The `False` teardown branch means the endpoint may still bill; it must not look deleted."""
    from flash.runner.supervise import lifecycle

    _seed_run(tmp_path, monkeypatch)
    runner._update("r1", "running", remote=dict(RUNPOD_A))

    class _Provider:
        def cancel(self, handle):
            return None

        def destroy(self, handle):
            raise RuntimeError("endpoint deletion refused")

    monkeypatch.setattr("flash.providers.get_provider", lambda name: _Provider())
    # patched where it is RESOLVED: _strict_teardown_handle calls the name in its defining module.
    monkeypatch.setattr(
        "flash.runner.supervise.recovery._worker_provably_gone", lambda run_id, handle: True
    )

    assert lifecycle._strict_teardown_handle(dict(RUNPOD_A), "r1") is False
    record = _ledger()[0]
    assert record["teardown_requested_at"]  # we DID ask
    assert record["deletion_confirmed_at"] is None  # but it was never confirmed
    assert record["terminal_at"] is None


# --------------------------------------------------------------------------- settlement
def _settled_status(tmp_path, monkeypatch, records, *, now, run_id="r-settle", **kw):
    terminal = kw.pop("finished_at", now - 7200.0)
    status = _seed_run(
        tmp_path,
        monkeypatch,
        run_id=run_id,
        state="done",
        updated_at=terminal,
        finished_at=terminal,
        **kw,
    )
    raw = _raw(run_id)
    raw["attempt_resources"] = records
    with open(runner.runs_file_path(run_id, ".json"), "w") as f:
        json.dump(raw, f, indent=2, sort_keys=True)
    return runner.get_status(run_id)


def _record(remote, **kw) -> dict:
    base = {
        "attempt": remote.get("attempt", 0),
        "provider": remote["provider"],
        "identity": dict(remote),
        "allocation": {"gpu": "B200", "gpu_count": 1},
        "accepted_rate_usd_hr": 4.0,
        "launched_at": 1000.0,
        "outcome": "succeeded",
        "outcome_at": None,
        "teardown_requested_at": None,
        "deletion_confirmed_at": None,
        "terminal_at": None,
        "realized_usage": None,
    }
    base.update(kw)
    return base


def test_partial_settlement_reports_now_and_tops_up_later(tmp_path, monkeypatch):
    """An unsettled resource must not close the run at a total that omits it.

    ep-a settled an hour ago; ep-b was torn down seconds ago and its invoice has not landed. The
    settled subset is reported immediately, but `reconciled_at` stays unset so the next sweep can
    restate the total upward once ep-b settles.
    """
    now = 1_000_000.0
    status = _settled_status(
        tmp_path,
        monkeypatch,
        [
            _record(RUNPOD_A, terminal_at=now - 7200.0),
            _record(RUNPOD_B, terminal_at=now - 60.0),  # too fresh to have settled
        ],
        now=now,
    )
    costs = {"ep-a": 1.0, "ep-b": 2.0}
    monkeypatch.setattr(
        reconcile,
        "realized_cost_for_remote",
        lambda remote, **kw: realized.RealizedCost(
            provider="runpod",
            realized_usd=costs[remote["endpoint_id"]],
            by_resource={"gpu": costs[remote["endpoint_id"]]},
        ),
    )
    posted: list[dict] = []
    monkeypatch.setattr(reconcile, "_report", lambda body: posted.append(body) or True)

    assert reconcile.reconcile_run(status, now=now) is True
    assert posted[-1]["realizedCostUsd"] == 1.0  # only the settled half
    persisted = runner.get_status("r-settle")
    assert persisted.realized_cost_usd == 1.0
    assert persisted.reconciled_at is None  # left OPEN for the top-up
    assert reconcile._due(persisted, now) is True

    # ep-b has now settled: the later sweep restates the full total and closes the run.
    later = now + 7200.0
    assert reconcile.reconcile_run(runner.get_status("r-settle"), now=later) is True
    assert posted[-1]["realizedCostUsd"] == 3.0
    closed = runner.get_status("r-settle")
    assert closed.realized_cost_usd == 3.0  # restated, not double-counted to 4.0
    assert closed.reconciled_at == later
    assert reconcile._due(closed, later) is False


def test_restatement_never_lowers_an_already_reported_total(tmp_path, monkeypatch):
    """A transient provider omission must not erase COGS that was already reported."""
    now = 1_000_000.0
    _settled_status(
        tmp_path,
        monkeypatch,
        [_record(RUNPOD_A, terminal_at=now - 7200.0)],
        now=now,
        run_id="r-mono",
    )
    runner.record_partial_realized_cost("r-mono", realized_cost_usd=5.0)
    runner.record_partial_realized_cost("r-mono", realized_cost_usd=1.0)
    assert runner.get_status("r-mono").realized_cost_usd == 5.0
    assert runner.get_status("r-mono").reconciled_at is None


def test_partial_write_never_reopens_a_closed_run(tmp_path, monkeypatch):
    """`reconciled_at` is authoritative: a late partial must not resurrect a closed run."""
    now = 1_000_000.0
    _settled_status(
        tmp_path,
        monkeypatch,
        [_record(RUNPOD_A, terminal_at=now - 7200.0)],
        now=now,
        run_id="r-closed",
    )
    runner.record_realized_cost("r-closed", realized_cost_usd=9.0, reconciled_at=now)
    runner.record_partial_realized_cost("r-closed", realized_cost_usd=99.0)
    closed = runner.get_status("r-closed")
    assert closed.realized_cost_usd == 9.0
    assert closed.reconciled_at == now


def test_window_close_finalizes_a_still_partial_run(tmp_path, monkeypatch):
    """Past the 7-day window nothing more will be billed, so the partial total becomes final."""
    now = 1_000_000.0
    old = now - 8 * 86400.0
    status = _settled_status(
        tmp_path,
        monkeypatch,
        [
            _record(RUNPOD_A, terminal_at=old),
            _record(RUNPOD_B, terminal_at=now - 60.0),
        ],
        now=now,
        run_id="r-window",
        finished_at=old,
    )
    monkeypatch.setattr(
        reconcile,
        "realized_cost_for_remote",
        lambda remote, **kw: realized.RealizedCost(provider="runpod", realized_usd=1.0),
    )
    monkeypatch.setattr(reconcile, "_report", lambda body: True)

    assert reconcile._due(status, now) is False  # aged out of the sweep
    assert reconcile.reconcile_run(status, now=now) is True
    assert runner.get_status("r-window").reconciled_at == now  # finalized, not left open forever


def test_report_marks_provider_and_gpu_mixed_only_when_genuinely_cross_provider(
    tmp_path, monkeypatch
):
    now = 1_000_000.0
    instance = {"provider": "lambda", "instance_id": "i-1", "hourly_usd": 2.0}
    status = _settled_status(
        tmp_path,
        monkeypatch,
        [
            _record(RUNPOD_A, terminal_at=now - 7200.0),
            _record(
                instance,
                terminal_at=now - 7200.0,
                allocation={"gpu": "H100", "gpu_count": 1},
            ),
        ],
        now=now,
        run_id="r-mixed",
    )
    monkeypatch.setattr(
        reconcile,
        "realized_cost_for_remote",
        lambda remote, **kw: realized.RealizedCost(
            provider=remote["provider"], realized_usd=1.0, by_resource={"gpu": 1.0}
        ),
    )
    posted: dict = {}
    monkeypatch.setattr(reconcile, "_report", lambda body: posted.update(body) or True)

    assert reconcile.reconcile_run(status, now=now) is True
    assert posted["provider"] == "mixed"
    assert posted["gpu"] == "mixed"
    assert posted["realizedCostUsd"] == 2.0
    assert posted["costByResource"] == {"gpu": 2.0}
    audit = posted["source"]["attemptResources"]
    assert {a["provider"] for a in audit} == {"runpod", "lambda"}


def test_legacy_run_without_a_ledger_keeps_single_handle_behavior(tmp_path, monkeypatch):
    """Pre-ledger runs are reconstructed from what survives, never backfilled with invented spend."""
    now = 1_000_000.0
    terminal = now - 7200.0
    status = _seed_run(
        tmp_path,
        monkeypatch,
        run_id="r-legacy",
        state="done",
        updated_at=terminal,
        finished_at=terminal,
        remote=dict(RUNPOD_A),
    )
    assert "attempt_resources" not in _raw("r-legacy")

    records = runner.attempt_resources_for_status(status)
    assert [r["identity"]["endpoint_id"] for r in records] == ["ep-a"]
    assert records[0]["terminal_at"] == terminal  # the only boundary a legacy run ever had

    monkeypatch.setattr(
        reconcile,
        "realized_cost_for_remote",
        lambda remote, **kw: realized.RealizedCost(provider="runpod", realized_usd=2.5),
    )
    posted: dict = {}
    monkeypatch.setattr(reconcile, "_report", lambda body: posted.update(body) or True)

    assert reconcile._due(status, now) is True
    assert reconcile.reconcile_run(status, now=now) is True
    assert posted["realizedCostUsd"] == 2.5
    assert runner.get_status("r-legacy").reconciled_at == now

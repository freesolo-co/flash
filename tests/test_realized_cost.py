"""Realized provider-cost reconciliation: RunPod shaping, dispatch, and reporting.

Offline -- the provider HTTP calls and backend POST are stubbed, so nothing touches the network.
"""

from __future__ import annotations

import pytest

import flash.runner.accounting.costs as runner_costs
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
from flash.providers.core import realized
from flash.providers.runpod.client.cost import shape_endpoint_cost
from flash.server.domain.ops import reconcile


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


# --------------------------------------------------------------------------- provider dispatch
def test_dispatch_runpod(monkeypatch):
    from flash.providers.runpod.client import api

    monkeypatch.setattr(
        api, "billing_endpoints", lambda **kw: [{"endpointId": "ep-1", "amount": 3.0}]
    )
    rc = realized.realized_cost_for_remote(
        {"provider": "runpod", "endpoint_id": "ep-1"}, start=0, end=100
    )
    assert rc is not None
    assert rc.provider == "runpod"
    assert rc.realized_usd == 3.0


def test_dispatch_none_when_no_handle_or_unknown_provider():
    assert realized.realized_cost_for_remote(None, start=0, end=1) is None
    assert realized.realized_cost_for_remote({}, start=0, end=1) is None
    assert realized.realized_cost_for_remote({"provider": "gcp"}, start=0, end=1) is None
    # runpod handle with no endpoint id -> nothing to query.
    assert realized.realized_cost_for_remote({"provider": "runpod"}, start=0, end=1) is None


# --------------------------------------------------------------------------- reconcile selection
def _status(**kw) -> runner_state.RunStatus:
    base = {"run_id": "r1", "state": "done", "spec": {}, "created_at": 0.0, "updated_at": 0.0}
    base.update(kw)
    if "finished_at" not in base:
        base["finished_at"] = base["updated_at"]
    return runner_state.RunStatus(**base)


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
    # a confirmed teardown moves the handle out of the active remote while retaining exact COGS evidence.
    assert reconcile._due(
        _status(
            state="done",
            updated_at=settled,
            remote=None,
            realized_cost_remote=handle,
        ),
        now,
    )
    # no provider handle to attribute cost
    assert not reconcile._due(_status(state="done", updated_at=settled, remote=None), now)


def test_due_anchors_settle_and_window_to_finished_at_not_bumped_updated_at():
    """_due bases the settle delay and the 7-day window on the frozen finished_at (teardown), not
    the mutable updated_at that deploy / late heartbeat move past teardown. So a run finished long
    enough ago is due even if updated_at was just bumped, and one finished outside the window is
    NOT resurrected by a recent bump."""
    now = 1_000_000.0
    handle = {"provider": "lambda", "instance_id": "i-1", "hourly_usd": 1.29}

    # deployed run: updated_at bumped to the deploy time 1 min ago (would look "too fresh" under
    # the old rule), but it finished training 2h ago -> past the settle delay -> DUE.
    assert reconcile._due(
        _status(state="deployed", updated_at=now - 60, finished_at=now - 7200, remote=handle), now
    )
    # finished 8 days ago but bumped 1 day ago: old rule (updated_at) would reconcile it; the
    # window must bound by finish time -> NOT due.
    assert not reconcile._due(
        _status(state="done", updated_at=now - 86400, finished_at=now - 8 * 86400, remote=handle),
        now,
    )
    # finished only 1 min ago -> still within the settle delay -> NOT due (even if updated_at is
    # older from some earlier write).
    assert not reconcile._due(
        _status(state="done", updated_at=now - 7200, finished_at=now - 60, remote=handle), now
    )


def test_reconcile_run_reports_and_persists(monkeypatch):
    now = 1_000_000.0
    status = _status(
        run_id="r-pos",
        updated_at=now - 7200,
        remote={"provider": "runpod", "endpoint_id": "ep-5", "allocated_gpu": "RTX 5090"},
    )
    monkeypatch.setattr(
        reconcile,
        "realized_cost_for_remote",
        lambda remote, **kw: realized.RealizedCost(
            provider="runpod", realized_usd=4.2, by_resource={"gpu": 4.2}
        ),
    )
    posted: dict = {}
    monkeypatch.setattr(reconcile, "_report", lambda body: posted.update(body) or True)
    updates: dict = {}
    monkeypatch.setattr(
        runner_costs,
        "record_realized_cost",
        lambda run_id, **kw: updates.update(run_id=run_id, **kw),
    )

    assert reconcile.reconcile_run(status, now=now) is True
    assert posted["runId"] == "r-pos"
    assert posted["realizedCostUsd"] == 4.2
    assert posted["provider"] == "runpod"
    assert posted["gpu"] == "RTX 5090"
    assert posted["costBasis"] == "realized"
    # persisted locally via the cost-only writer (never touches state) with the realized figure
    # + a reconcile marker.
    assert updates["run_id"] == "r-pos"
    assert "state" not in updates
    assert updates["realized_cost_usd"] == 4.2
    assert updates["reconciled_at"] == now


def test_instance_realized_cost_bills_launch_to_run_end_not_padded_end():
    """Instance providers bill flat $/hr over launch->run_end; the settle-padded billing `end`
    (used only for RunPod's invoice query) must NOT inflate their wall."""
    remote = {"provider": "lambda", "instance_id": "i-1", "hourly_usd": 2.0, "started_ts": 1000.0}
    rc = realized.realized_cost_for_remote(
        remote,
        start=1000.0,
        end=1_000_000.0,
        run_end=4600.0,  # 1h of wall, end padded way past
    )
    assert rc is not None
    assert rc.provider == "lambda"
    assert rc.wall_seconds == 3600.0  # 4600 - 1000, NOT the padded end
    assert rc.realized_usd == 2.0  # 1h x $2/hr


@pytest.mark.parametrize(
    "started_ts",
    [pytest.param(None, id="missing"), 0.0, -1.0, float("inf"), float("nan"), True],
)
def test_instance_realized_cost_rejects_invalid_launch_timestamp(started_ts):
    remote = {"provider": "lambda", "instance_id": "i-1", "hourly_usd": 1.29}
    if started_ts is not None:
        remote["started_ts"] = started_ts

    assert realized.realized_cost_for_remote(remote, start=100.0, end=4600.0) is None


def test_reconcile_leaves_invalid_instance_launch_unsettled_and_due(monkeypatch):
    now = 1_000_000.0
    status = _status(
        run_id="r-invalid-launch",
        created_at=now - 10_000.0,
        updated_at=now - 7200.0,
        finished_at=now - 7200.0,
        remote={"provider": "lambda", "instance_id": "i-1", "hourly_usd": 1.29},
    )
    monkeypatch.setattr(
        reconcile,
        "_report",
        lambda _body: pytest.fail("unattributable cost must not be reported"),
    )
    monkeypatch.setattr(
        runner_costs,
        "record_realized_cost",
        lambda *_args, **_kwargs: pytest.fail("unattributable cost must remain unsettled"),
    )

    assert reconcile.reconcile_run(status, now=now) is False
    assert status.reconciled_at is None
    assert reconcile._due(status, now) is True


def test_reconcile_uses_finished_at_not_deploy_bumped_updated_at_for_instance(monkeypatch):
    """A Lambda run deployed AFTER completion has updated_at moved to the deploy time;
    reconciliation must pass the FROZEN training-teardown (finished_at) as the instance run_end,
    not that later deploy time, or it over-reports COGS (flat $/hr from launch until deployment)."""
    now = 1_000_000.0
    teardown = now - 7200.0  # training finished (and instance torn down) 2h ago
    deploy_t = now - 600.0  # deployed 10 min ago -> updated_at bumped to here
    captured: dict = {}

    def fake_realized(remote, *, start, end, run_end=None):
        captured.update(start=start, end=end, run_end=run_end)
        return realized.RealizedCost(provider="lambda", realized_usd=1.0)

    monkeypatch.setattr(reconcile, "realized_cost_for_remote", fake_realized)
    monkeypatch.setattr(reconcile, "_report", lambda body: True)
    monkeypatch.setattr(runner_costs, "record_realized_cost", lambda run_id, **kw: None)

    status = _status(
        state="deployed",
        updated_at=deploy_t,
        finished_at=teardown,
        remote={
            "provider": "lambda",
            "instance_id": "i-1",
            "hourly_usd": 1.29,
            "started_ts": now - 10800,
        },
    )
    assert reconcile.reconcile_run(status, now=now) is True
    assert captured["run_end"] == teardown  # frozen training end, not the deploy bump
    assert captured["run_end"] != deploy_t


def test_reconcile_rejects_a_run_without_finished_at(monkeypatch):
    monkeypatch.setattr(
        reconcile,
        "realized_cost_for_remote",
        lambda *args, **kwargs: pytest.fail("provider billing must not run without finished_at"),
    )
    status = _status(
        state="done",
        updated_at=1_000_000.0,
        finished_at=None,
        remote={
            "provider": "lambda",
            "instance_id": "i-1",
            "hourly_usd": 1.29,
            "started_ts": 999_000.0,
        },
    )

    with pytest.raises(ValueError, match="missing finished_at"):
        reconcile.reconcile_run(status, now=1_000_000.0)


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
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    now = 1_000_000.0
    created_at = now - 10_000.0
    from flash.schema import spec_from_dict

    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
            "train": {"epochs": 1, "max_examples": 1},
            "gpu": {},
        },
        run_id="r-adv",
    )
    remote = {
        "provider": "runpod",
        "endpoint_id": "ep-1",
        "endpoint_name": "flash-r-adv-a0",
        "key_fingerprint": "rpk-" + "0" * 64,
        "job_id": "job-1",
        "attempt": 0,
        "started_ts": created_at + 100.0,
    }
    snapshot = runner_state.RunStatus(
        run_id="r-adv",
        state="done",
        spec=spec.to_dict(),
        created_at=created_at,
        updated_at=now - 7200,
        finished_at=now - 7200,
        remote=remote,
    )
    # the run has since advanced to `deployed` on disk (serving stood up on the finished run).
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="r-adv",
            state="deployed",
            spec=spec.to_dict(),
            created_at=created_at,
            updated_at=now - 100,
            finished_at=now - 7200,
            remote=remote,
            deployment={"state": "active"},
        ),
        _run_deadline_at=created_at + float(spec.gpu.max_wall_seconds),
        _next_attempt=1,
    )
    monkeypatch.setattr(
        reconcile,
        "realized_cost_for_remote",
        lambda remote, **kw: realized.RealizedCost(provider="runpod", realized_usd=3.3),
    )
    monkeypatch.setattr(reconcile, "_report", lambda body: True)

    assert reconcile.reconcile_run(snapshot, now=now) is True
    persisted = runner_status.get_status("r-adv")
    # Status preserved (NOT reverted to the stale `done`), cost fields written.
    assert persisted.state == "deployed"
    assert persisted.deployment == {"state": "active"}
    assert persisted.realized_cost_usd == 3.3
    assert persisted.reconciled_at == now
    assert persisted.realized_cost_remote is None


def test_reconciled_cost_retains_charge_attribution_until_billing_completes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    remote = {
        "provider": "runpod",
        "endpoint_id": "ep-charge-pending",
        "allocated_gpu": "A100 PCIe",
        "allocated_gpu_count": 4,
    }
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="charge-pending",
            state="done",
            spec={"algorithm": "sft", "model": "m"},
            billing_context={"org_id": "org-1"},
            billing_state="failed",
            realized_cost_remote=remote,
        )
    )

    runner_costs.record_realized_cost("charge-pending", realized_cost_usd=3.5, reconciled_at=100.0)

    pending = runner_status.get_status("charge-pending")
    assert pending.realized_cost_remote == remote
    runner_costs.record_billing_state(
        "charge-pending",
        billing_state="charged",
        billing_error=None,
        billing_charge={"amountCents": 125},
    )
    charged = runner_status.get_status("charge-pending")
    assert charged.realized_cost_remote is None
    assert charged.realized_cost_usd == 3.5


def test_reconcile_once_disabled_without_internal_key(monkeypatch):
    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    monkeypatch.setattr(runner_status, "list_runs", lambda: [_status(updated_at=0)])
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
    monkeypatch.setattr(runner_status, "list_runs", lambda: [due, not_due])
    seen: list[str] = []

    def fake_reconcile_run(status, *, now):
        seen.append(status.run_id)
        return True

    monkeypatch.setattr(reconcile, "reconcile_run", fake_reconcile_run)
    assert reconcile.reconcile_once(now=now) == 1
    assert seen == ["due"]


# ----------------------------------------------------------- _reconcile_cost_loop cancel/error
# The background loop must (a) re-raise asyncio.CancelledError so the lifespan's task.cancel()
# can stop it at shutdown (no stall), and (b) swallow a real Exception from a sweep and continue
# to the next cycle — the same contract the sibling reaper/sweep loops uphold. #191 follow-up.
class _StopLoop(Exception):
    """Sentinel raised from the patched asyncio.sleep to terminate the otherwise-infinite loop."""


def _run_reconcile_loop_once(monkeypatch, reconcile_once_impl):
    """Drive _reconcile_cost_loop through exactly ONE sweep, then break out via the next sleep.

    The loop awaits asyncio.sleep BEFORE the sweep, so the first sleep returns and the second
    raises _StopLoop to end the loop deterministically without a 3600s wait. reconcile_once is a
    SYNC callable run via asyncio.to_thread(...), so the stub is a plain sync function."""
    import asyncio

    from flash.server.asgi import app as server_app

    # reconcile_once is imported function-locally inside the loop
    # (`from flash.server.domain.ops.reconcile import reconcile_once`), so patch it at the source module.
    monkeypatch.setattr(reconcile, "reconcile_once", reconcile_once_impl)

    calls = {"sleep": 0}

    async def fake_sleep(_interval):
        calls["sleep"] += 1
        if calls["sleep"] >= 2:  # 1st sleep: enter the loop body; 2nd: break out post-sweep
            raise _StopLoop

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return asyncio.run(server_app._reconcile_cost_loop()), calls


def test_reconcile_cost_loop_swallows_sweep_error_and_continues(monkeypatch):
    import pytest

    swept = {"n": 0}

    def boom_once(*a, **k):
        swept["n"] += 1
        raise RuntimeError("provider billing API blip")

    # The RuntimeError must NOT escape; the loop reaches the next sleep, which breaks it (_StopLoop).
    with pytest.raises(_StopLoop):
        _run_reconcile_loop_once(monkeypatch, boom_once)
    assert swept["n"] == 1  # the sweep ran once and its failure was swallowed (loop kept going)


def test_reconcile_cost_loop_reraises_cancelled(monkeypatch):
    import asyncio

    import pytest

    def cancel_once(*a, **k):
        raise asyncio.CancelledError

    # A cancel surfacing during the sweep must propagate (re-raised), NOT be swallowed as a
    # generic Exception — otherwise shutdown would stall until the next cancellation point.
    with pytest.raises(asyncio.CancelledError):
        _run_reconcile_loop_once(monkeypatch, cancel_once)

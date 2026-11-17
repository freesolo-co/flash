"""Queued-run cancellation, cancel-vs-bid races and Spot queue visibility; no paid resources."""

from __future__ import annotations

import io
import time

import pytest

import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.supervise.deploy as runner_deploy
from flash.core.spec import GpuSpec, JobSpec
from flash.providers.mithril import api, client, jobs, queue, queue_board
from flash.providers.mithril.prices import Offer
from flash.runner.lifecycle.startup import record_startup_stage
from flash.runner.supervise.errors import _RunCancelled
from tests.test_mithril_api import launch, owned_bid

RUN_ID = "flash-queue-test"


@pytest.fixture(autouse=True)
def runs_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("FLASH_MITHRIL_API_KEY", "test-secret")
    monkeypatch.setenv("FLASH_MITHRIL_PROJECT_ID", "proj_test")
    monkeypatch.delenv("FLASH_MITHRIL_IMAGE_VERSION", raising=False)
    monkeypatch.delenv("FLASH_MITHRIL_SSH_KEY_IDS", raising=False)
    monkeypatch.setattr(api, "CREATE_RECHECK_SECONDS", 0)
    return tmp_path


def _spec(run_id: str = RUN_ID) -> JobSpec:
    return JobSpec(
        run_id=run_id,
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(type="H100", provider="spot"),
    )


def _save_run(state: str = "queued", run_id: str = RUN_ID) -> None:
    runner_state._save_status(
        runner_state.RunStatus(run_id=run_id, state=state, spec=_spec(run_id).to_dict())
    )


def _cancel_on_disk(run_id: str = RUN_ID) -> None:
    runner_status._update(run_id, "cancelled", cancel_reason="test cancel")


# --- fix 1: a queued run's cancel is immediate -------------------------------------------------


def test_queued_cancel_is_persisted_before_any_provider_cleanup(monkeypatch):
    from flash.runner.supervise import recovery

    _save_run("queued")
    seen = []

    def slow_provider_sweep(_spec):
        # the provider sweep is where minutes went; the state must already be terminal here.
        seen.append(runner_status.get_status(RUN_ID).state)

    monkeypatch.setattr(recovery, "_gc_run_sandboxes", slow_provider_sweep)
    status = runner_deploy.cancel_run(RUN_ID)
    assert seen == ["cancelled"]
    assert status.state == "cancelled"
    assert status.to_dict()["state"] == "cancelled"
    assert status.cancel_reason.startswith("cancelled by user request at ")


def test_queued_cancel_is_visible_while_cancel_waits_on_a_held_lock(monkeypatch):
    import threading

    from flash.runner.supervise import recovery
    from flash.server.platform.locks import _deploy_lock

    _save_run("queued")
    monkeypatch.setattr(recovery, "_gc_run_sandboxes", lambda _spec: None)
    lock = _deploy_lock(RUN_ID)
    lock.acquire()
    result = {}
    worker = threading.Thread(target=lambda: result.update(status=runner_deploy.cancel_run(RUN_ID)))
    try:
        worker.start()
        deadline = time.monotonic() + 10
        while runner_status.get_status(RUN_ID).state != "cancelled":
            assert time.monotonic() < deadline, "cancel was not visible while the lock was held"
            time.sleep(0.01)
        assert worker.is_alive()  # still waiting on the lock, yet already cancelled
    finally:
        lock.release()
        worker.join(10)
    assert result["status"].state == "cancelled"


def test_run_cleanup_inspects_only_that_runs_bids(monkeypatch):
    """Listing every project bid and its instances made one run's cancel take minutes."""
    bids = [
        owned_bid(fid=f"bid_other{index}", status="Terminated") for index in range(50)
    ]  # owned_bid labels them flash-run=test
    mine = owned_bid(fid="bid_mine", status="Allocated")
    mine_labels = {"flash-run": RUN_ID}
    inspected = []

    def ownership(bid):
        return {"flash-run": RUN_ID} if bid["fid"] == "bid_mine" else {"flash-run": "test"}

    def instances(bid_id, **_kw):
        inspected.append(bid_id)
        return []

    monkeypatch.setattr(api, "pages", lambda *a, **kw: [*bids, mine])
    monkeypatch.setattr(api, "_ownership", ownership)
    monkeypatch.setattr(api, "_instances", instances)
    found = api.list_instances(run_id=RUN_ID)
    assert [item["metadata"]["id"] for item in found] == ["bid_mine"]
    assert found[0]["metadata"]["labels"] == mine_labels
    assert inspected == []  # no per-bid lookups for other runs' terminated bids

    terminated = []
    monkeypatch.setattr(api, "terminate", terminated.append)
    from flash.providers.core.registry import get_provider

    get_provider("spot").terminate_run(RUN_ID)
    assert terminated == ["bid_mine"]


# --- fix 2: no bid once a run is cancelled -----------------------------------------------------


def _submit(monkeypatch, *, create):
    from flash import runtime_capsule

    offer = Offer("H100", 1, "it_h100", 1.0, 1500, "us-central2-b")
    monkeypatch.setattr(queue, "wait_for_price", lambda *a, **kw: offer)
    monkeypatch.setattr(jobs, "scope_fingerprint", lambda: "scope")
    monkeypatch.setattr(jobs, "build_payload", lambda *a, **kw: {"payload": True})
    monkeypatch.setattr(jobs, "worker_image_for_gpu", lambda gpu: "worker")
    monkeypatch.setattr(runtime_capsule, "build_capsule", lambda *_a: (b"capsule", None))
    monkeypatch.setattr(api, "create", create)
    terminated = []
    monkeypatch.setattr(api, "terminate", terminated.append)
    handles = []
    log = io.StringIO()
    with pytest.raises(_RunCancelled):
        jobs.submit_attempt(
            _spec(),
            log=log,
            on_handle=handles.append,
            deadline_at=time.time() + 3600,
        )
    return terminated, handles, log.getvalue()


def test_cancel_during_payload_build_places_no_bid(monkeypatch):
    """Regression: flash-1794908910-4696240d logged its cancel, then still bid and booted."""
    _save_run("provisioning")
    from flash import runtime_capsule

    def capsule_while_cancelled(*_a):
        _cancel_on_disk()
        return b"capsule", None

    def create(**_kw):
        pytest.fail("a cancelled run must never place a Spot bid")

    monkeypatch.setattr(runtime_capsule, "build_capsule", capsule_while_cancelled)
    offer = Offer("H100", 1, "it_h100", 1.0, 1500, "us-central2-b")
    monkeypatch.setattr(queue, "wait_for_price", lambda *a, **kw: offer)
    monkeypatch.setattr(jobs, "scope_fingerprint", lambda: "scope")
    monkeypatch.setattr(jobs, "build_payload", lambda *a, **kw: {"payload": True})
    monkeypatch.setattr(jobs, "worker_image_for_gpu", lambda gpu: "worker")
    monkeypatch.setattr(api, "create", create)
    with pytest.raises(_RunCancelled):
        jobs.submit_attempt(_spec(), log=io.StringIO(), deadline_at=time.time() + 3600)


def test_bid_created_while_cancel_lands_is_terminated_and_never_persisted(monkeypatch):
    _save_run("provisioning")

    def create(**_kw):
        _cancel_on_disk()  # the cancel lands while the create request is in flight
        return "bid_late"

    terminated, handles, _ = _submit(monkeypatch, create=create)
    assert terminated == ["bid_late"]
    assert handles == []
    assert runner_status.get_status(RUN_ID).state == "cancelled"


def test_queue_never_returns_an_offer_once_cancelled(monkeypatch):
    _save_run("queued")
    offer = Offer("H100", 1, "it_h100", 1.0, 1500, "us-central2-b")

    def offers_while_cancelled(**_kw):
        _cancel_on_disk()
        return [offer]

    monkeypatch.setattr(queue, "affordable_offers", offers_while_cancelled)
    monkeypatch.setattr(queue, "quota_available", lambda *a, **kw: True)
    with pytest.raises(_RunCancelled):
        queue.wait_for_price("H100", RUN_ID, deadline_at=time.time() + 600, log=io.StringIO())
    assert runner_status.get_status(RUN_ID).state == "cancelled"


def test_handle_persistence_refuses_a_cancelled_run():
    """The last line of defence: a bid whose handle arrives after the cancel is never adopted."""
    from flash.runner.lifecycle import attempts

    _save_run("provisioning")
    claim = attempts.reserve_verified_attempt_launch(RUN_ID)
    assert claim is not None
    _cancel_on_disk()
    assert not attempts.persist_claimed_remote(RUN_ID, claim, {"provider": "spot"})
    assert runner_status.get_status(RUN_ID).remote is None


# --- fix 3: queue reason, position and ETA ------------------------------------------------------


def _wait_once(monkeypatch, *, offers=(), quota=True, run_id=RUN_ID):
    """Run one queue pass, then cancel during the pause so the call returns."""
    now = [10_000.0]
    monkeypatch.setattr(queue.time, "time", lambda: now[0])
    monkeypatch.setattr(queue_board.time, "time", lambda: now[0])

    def sleep(seconds):
        now[0] += seconds
        _cancel_on_disk(run_id)

    monkeypatch.setattr(queue.time, "sleep", sleep)
    monkeypatch.setattr(queue, "affordable_offers", lambda **kw: list(offers))
    monkeypatch.setattr(queue, "quota_available", lambda *a, **kw: quota)
    seen = {}

    real_record = record_startup_stage

    def record(run, stage, **kw):
        real_record(run, stage, **kw)
        seen["startup"] = runner_status.get_status(run).to_dict()["startup"]

    import flash.runner.lifecycle.startup as startup

    monkeypatch.setattr(startup, "record_startup_stage", record)
    log = io.StringIO()
    with pytest.raises(_RunCancelled):
        queue.wait_for_price(("H100", "H200"), run_id, deadline_at=now[0] + 3600, log=log)
    return log.getvalue(), seen["startup"]


def test_quota_refusal_is_the_queue_reason_with_position_and_unknown_eta(monkeypatch):
    _save_run("queued")
    now = [9_500.0]
    monkeypatch.setattr(queue_board.time, "time", lambda: now[0])
    queue_board.join("flash-earlier", ("H100",))  # an earlier H100 run is ahead of us
    queue_board.join("flash-h200-only", ("H200",))  # overlaps via H200
    queue_board.join("flash-other-class", ("A100",))  # does not compete
    queue_board.record_quota_refusal("H100")
    text, startup = _wait_once(monkeypatch)

    assert "Spot queued (GPU quota full: Mithril refused a Flash bid for GPU quota" in text
    assert "position 3 of 3 among Flash Spot runs waiting for these GPUs" in text
    assert "ETA unknown" in text
    queue_view = startup["queue"]
    assert queue_view["reason"] == "gpu_quota_full"
    assert queue_view["position"] == 3
    assert queue_view["waiting"] == 3
    assert queue_view["eta_seconds"] is None
    assert queue_view["eta"].startswith("ETA unknown")
    assert startup["stage"] == "waiting_for_gpu"
    assert "GPU quota full; position 3 of 3; ETA unknown" in startup["detail"]
    # still queued: no cold-start expectation yet
    assert "typical_cold_start" not in startup
    # the cancel removed the run from the board
    assert queue_board.view(RUN_ID) is None


def test_no_affordable_offer_is_reported_as_price_cap(monkeypatch):
    _save_run("queued")
    text, startup = _wait_once(monkeypatch)
    assert "Spot queued (no GPU below the price cap:" in text
    assert "position 1 of 1" in text
    assert startup["queue"]["reason"] == "no_gpu_below_price_cap"


def test_affordable_offer_without_project_quota_is_reported_as_quota(monkeypatch):
    _save_run("queued")
    offer = Offer("H100", 1, "it_h100", 1.0, 1500, "us-central2-b")
    text, startup = _wait_once(monkeypatch, offers=[offer], quota=False)
    assert "Spot queued (GPU quota full: project Spot quota" in text
    assert startup["queue"]["reason"] == "gpu_quota_full"


def test_eta_comes_from_recent_allocations_and_stays_coarse(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(queue_board.time, "time", lambda: now[0])
    for run_id, joined_at in (("a", 0.0), ("b", 1.0), ("me", 2.0), ("later", 3.0)):
        now[0] = joined_at
        queue_board.join(run_id, ("H100",))
    now[0] = 100.0
    queue_board.leave("a", allocated=True)
    now[0] = 1800.0
    queue_board.leave("b", allocated=True)
    queue_board.leave("later", allocated=False)  # a cancel is not an allocation
    now[0] = 3700.0
    view = queue_board.join("me", ("H100",))
    assert (view.position, view.waiting, view.recent_allocations) == (1, 1, 2)
    # 2 allocations over the last 3600 s, position 1 -> 1800 s
    assert view.eta_seconds() == pytest.approx(1800.0)
    assert view.eta_short() == "ETA roughly 30 min"
    assert "rough: 2 queued Flash Spot runs got a GPU in the last 60 min" in view.eta_text()
    public = view.public("no_gpu_below_price_cap", "detail")
    assert public["eta_seconds"] == 1800


def test_stale_waiters_do_not_hold_positions(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(queue_board.time, "time", lambda: now[0])
    queue_board.join("crashed", ("H100",))
    now[0] = queue_board.STALE_AFTER_S + 10
    view = queue_board.join("me", ("H100",))
    assert (view.position, view.waiting) == (1, 1)


def test_cold_start_hint_only_after_a_gpu_is_assigned():
    from flash.cli.ui.render import run_status

    _save_run("queued")
    record_startup_stage(
        RUN_ID,
        "waiting_for_gpu",
        extra="Spot: GPU quota full; position 2 of 4; ETA unknown",
        queue={"reason": "gpu_quota_full", "reason_detail": "x", "eta": "ETA unknown"},
    )
    queued = runner_status.get_status(RUN_ID).to_dict()
    assert "typical_cold_start" not in queued["startup"]
    panel = run_status(queued)
    assert "position 2 of 4" in panel
    assert "cold start typically" not in panel
    assert "ETA unknown" in panel

    record_startup_stage(RUN_ID, "allocating_gpu")
    assert "typical_cold_start" not in runner_status.get_status(RUN_ID).to_dict()["startup"]

    record_startup_stage(RUN_ID, "booting_instance", attempt=0)
    booting = runner_status.get_status(RUN_ID).to_dict()
    assert booting["startup"]["typical_cold_start"] == "10-12 minutes"
    assert "queue" not in booting["startup"]
    assert "cold start typically takes 10-12 minutes" in run_status(booting)


def test_accepted_bid_leaves_the_queue_and_counts_as_an_allocation(monkeypatch):
    _save_run("provisioning")
    queue_board.join(RUN_ID, ("H100",))
    from flash import runtime_capsule

    offer = Offer("H100", 1, "it_h100", 1.0, 1500, "us-central2-b")
    monkeypatch.setattr(queue, "wait_for_price", lambda *a, **kw: offer)
    monkeypatch.setattr(jobs, "scope_fingerprint", lambda: "scope")
    monkeypatch.setattr(jobs, "build_payload", lambda *a, **kw: {"payload": True})
    monkeypatch.setattr(jobs, "worker_image_for_gpu", lambda gpu: "worker")
    monkeypatch.setattr(runtime_capsule, "build_capsule", lambda *_a: (b"capsule", None))
    monkeypatch.setattr(api, "create", lambda **kw: "bid_ok")
    monkeypatch.setattr(api, "terminate", lambda _bid: None)

    class _Stop(Exception):
        pass

    def on_handle(_handle):
        raise _Stop

    log = io.StringIO()
    with pytest.raises(_Stop):
        jobs.submit_attempt(_spec(), log=log, on_handle=on_handle, deadline_at=time.time() + 3600)
    assert "Spot: Mithril accepted the bid for 1x H100 in us-central2-b (bid_ok)" in log.getvalue()
    assert queue_board.view(RUN_ID) is None
    assert queue_board._load(queue_board._path())["allocations"]


def test_quota_refusal_is_recorded_and_logged_as_a_requeue(monkeypatch):
    from flash.providers.core.base import CapacityUnavailableError
    from flash.runner.supervise import execution

    monkeypatch.setattr(api, "pages", lambda *a, **kw: [])

    def refuse(*args, **kwargs):
        raise client.MithrilApiError(
            status=422,
            reason="Self-serve GPU quota exceeded in cluster us-central2-b: using 16 of 16",
        )

    monkeypatch.setattr(api, "call", refuse)
    with pytest.raises(CapacityUnavailableError) as error:
        launch(monkeypatch)
    assert "quota" in error.value.cause.lower()
    assert queue_board._load(queue_board._path())["quota_refusals"].keys() == {"H200"}

    queue_board.join("flash-ahead", ("H100",))
    queue_board.join(RUN_ID, ("H100", "H200"))
    detail = f"provider refused the create for capacity (CapacityUnavailableError after {error.value.cause})"
    line = execution._spot_requeue_line(RUN_ID, 3, detail)
    assert line.startswith(
        "Spot bid 3: provider refused (quota full); back in the queue at position 2 of 2; "
        "nothing was provisioned or billed"
    )
    assert "failed" not in line.splitlines()[0]

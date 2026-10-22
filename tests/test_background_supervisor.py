"""Concurrency contracts for application-owned training and recovery work."""

import threading

import pytest

from flash._internal.background import (
    Limits,
    Retry,
    Supervisor,
    TaskAdmissionError,
    current_supervisor,
    pause,
)


def test_delayed_retry_releases_worker_but_retains_admission():
    owner = Supervisor({"recovery": Limits(1, 2)})
    observed = threading.Event()
    next_task = threading.Event()

    def observe():
        observed.set()
        return Retry(3600)

    try:
        assert owner.submit("retry", "recovery", observe)
        assert observed.wait(1)
        assert not owner.submit("retry", "recovery", lambda: pytest.fail("duplicate"))
        assert owner.submit("next", "recovery", next_task.set)
        assert next_task.wait(1), "a scheduled retry must release the only recovery worker"
        assert owner.contains("retry")
    finally:
        assert owner.close(1)


def test_delayed_work_and_reservations_share_a_finite_bound():
    owner = Supervisor({"training": Limits(1, 2)})
    try:
        with owner.reserve("reserved", "training"):
            owner.submit("delayed", "training", lambda: None, delay=3600)
            with pytest.raises(TaskAdmissionError, match="capacity"):
                owner.submit("overflow", "training", lambda: None)
        assert not owner.contains("reserved")
        assert owner.cancel("delayed")
        assert not owner.contains("delayed")
    finally:
        assert owner.close(1)


def test_cleanup_and_recovery_have_capacity_while_training_is_full():
    owner = Supervisor({lane: Limits(1, 1) for lane in ("training", "cleanup", "recovery")})
    running = threading.Event()
    cleaned = threading.Event()
    recovered = threading.Event()

    def monitor():
        running.set()
        pause(3600)

    try:
        owner.submit("monitor", "training", monitor)
        assert running.wait(1)
        with pytest.raises(TaskAdmissionError):
            owner.submit("extra", "training", lambda: None)
        owner.submit("cleanup", "cleanup", cleaned.set)
        owner.submit("recover", "recovery", recovered.set)
        assert cleaned.wait(1)
        assert recovered.wait(1)
    finally:
        assert owner.close(1), "shutdown must interrupt cooperative monitoring waits"


def test_cancel_stops_active_task_and_discards_queued_work():
    owner = Supervisor({"training": Limits(1, 2)})
    running = threading.Event()
    release = threading.Event()
    queued_ran = threading.Event()

    def monitor():
        running.set()
        release.wait(2)

    try:
        owner.submit("active", "training", monitor)
        assert running.wait(1)
        owner.submit("queued", "training", queued_ran.set)
        assert owner.cancel("queued")
        assert owner.cancel("active")
        assert not owner.close(0), "blocking provider calls may outlive the shutdown budget"
        with pytest.raises(TaskAdmissionError, match="shutting down"):
            owner.submit("new", "training", lambda: None)
        release.set()
        assert owner.close(1)
        assert not queued_ran.is_set()
    finally:
        release.set()
        owner.close(1)


def test_reservation_only_dispatches_after_persistence_and_releases_on_error():
    owner = Supervisor({"training": Limits(1, 1)})
    ran = threading.Event()
    try:
        with (
            pytest.raises(ValueError, match="persistence failed"),
            owner.reserve("failed", "training"),
        ):
            raise ValueError("persistence failed")
        assert not owner.contains("failed")
        with owner.reserve("accepted", "training") as dispatch:
            assert not ran.is_set()
            dispatch(ran.set)
        assert ran.wait(1)
    finally:
        assert owner.close(1)


def test_shutdown_invalidates_undispatched_reservation():
    owner = Supervisor({"training": Limits(1, 1)})
    with owner.reserve("run", "training") as dispatch:
        assert owner.close()
        with pytest.raises(TaskAdmissionError):
            dispatch(lambda: pytest.fail("dispatch after shutdown"))


def test_independent_owners_are_bound_inside_their_tasks():
    owners = [Supervisor({"recovery": Limits(1, 1)}) for _ in range(2)]
    seen = []
    events = [threading.Event(), threading.Event()]
    try:
        for owner, event in zip(owners, events, strict=True):

            def observe(event=event):
                seen.append(current_supervisor())
                event.set()

            owner.submit("same-key", "recovery", observe)
        assert all(event.wait(1) for event in events)
        assert set(seen) == set(owners)
    finally:
        for owner in owners:
            assert owner.close(1)


def test_submission_rejects_capacity_before_writing_run(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec
    from flash.runner.lifecycle import state
    from flash.runner.lifecycle.submit import PreparedJob, submit_job

    monkeypatch.setattr(state, "RUNS_DIR", str(tmp_path))
    spec = JobSpec(run_id="no-capacity", model="Qwen/Qwen3.5-9B", algorithm="sft")
    owner = Supervisor({"training": Limits(1, 1)})
    try:
        with owner.reserve("busy", "training"), pytest.raises(TaskAdmissionError):
            submit_job(
                spec,
                background=True,
                supervisor=owner,
                prepared_job=PreparedJob(spec, spec, 1.0),
            )
        assert not (tmp_path / "no-capacity.json").exists()
    finally:
        assert owner.close(1)


def test_worker_start_failure_preserves_admitted_work(monkeypatch):
    owner = Supervisor({"recovery": Limits(1, 1)})
    ran = threading.Event()
    original_start = threading.Thread.start
    failures = []

    def start(thread):
        if thread.name == "flash-recovery" and not failures:
            failures.append(True)
            raise RuntimeError("temporary thread creation failure")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        owner.submit("recover", "recovery", ran.set)
        assert ran.wait(3)
        assert failures == [True]
    finally:
        assert owner.close(1)


def test_queued_run_requiring_request_secrets_cannot_recover_without_them(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec
    from flash.runner.lifecycle import state, status
    from flash.server.platform import runtime

    monkeypatch.setattr(state, "RUNS_DIR", str(tmp_path))
    spec = JobSpec(run_id="secret-queued", model="Qwen/Qwen3.5-9B", algorithm="sft")
    state._save_status(
        state.RunStatus(
            run_id=spec.run_id,
            state="queued",
            spec=spec.to_dict(),
            requires_runtime_secrets=True,
        )
    )
    stored = status.get_status(spec.run_id)
    assert stored.requires_runtime_secrets
    assert "requires_runtime_secrets" not in stored.to_dict()
    with pytest.raises(ValueError, match="cannot restore request runtime secrets"):
        status.reallocation_spec_from_status(stored)
    assert not runtime._enqueue_resubmit(spec)
    assert status.get_status(spec.run_id).state == "failed"


def test_user_cancellation_discards_followups_but_keeps_cleanup(monkeypatch):
    from types import SimpleNamespace

    from flash.server.routes import runs

    owner = Supervisor()
    status = SimpleNamespace(state="cancelled", to_dict=lambda: {"state": "cancelled"})
    monkeypatch.setattr(runs, "require_owned_run", lambda *_args: status)
    monkeypatch.setattr(runs, "cancel_run", lambda _run_id: status)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(training_jobs=owner)))
    try:
        for kind in ("run", "retry", "attach", "cleanup"):
            owner.submit(
                (kind, "one"),
                "cleanup" if kind == "cleanup" else "recovery",
                lambda: None,
                delay=3600,
            )
        assert runs.cancel("one", request, {}) == {"state": "cancelled"}
        assert all(not owner.contains((kind, "one")) for kind in ("run", "retry", "attach"))
        assert owner.contains(("cleanup", "one"))
    finally:
        assert owner.close(1)


def test_recovered_launch_rechecks_provider_after_waiting_for_capacity(monkeypatch):
    from types import SimpleNamespace

    from flash.runner.lifecycle import deadlines
    from flash.server.platform import runtime

    spec = SimpleNamespace(run_id="delayed-launch")
    monkeypatch.setattr(
        runtime,
        "get_status",
        lambda _run: SimpleNamespace(
            state="provisioning", remote=None, cleanup_confirmed_remote=None
        ),
    )
    monkeypatch.setattr(deadlines, "_remaining_run_wall_seconds", lambda _run: 30.0)
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: False)
    monkeypatch.setattr(
        runtime,
        "_start_handleless_resubmit",
        lambda *_args: pytest.fail("must not reserve a replacement over a live worker"),
    )
    result = runtime._resubmit_task(spec)
    assert isinstance(result, Retry)
    assert result.delay == 30.0


def test_delayed_queued_resubmission_uses_current_state_on_every_retry(monkeypatch):
    from types import SimpleNamespace

    from flash.runner.lifecycle import deadlines
    from flash.server.platform import runtime

    spec = SimpleNamespace(run_id="changed-while-queued")
    status = SimpleNamespace(
        state="queued", remote=None, cleanup_confirmed_remote=None, requires_runtime_secrets=False
    )
    monkeypatch.setattr(runtime, "get_status", lambda _: status)
    monkeypatch.setattr(
        "flash.runner.lifecycle.status.source_snapshot_from_status", lambda *a, **kw: {}
    )
    monkeypatch.setattr(deadlines, "_remaining_run_wall_seconds", lambda _: 30.0)
    monkeypatch.setattr(runtime, "_recovery_wall_deadline_is_open", lambda _: True)
    callbacks = []
    monkeypatch.setattr(
        current_supervisor(), "submit", lambda key, lane, target: callbacks.append(target) or True
    )
    assert runtime._enqueue_resubmit(spec)
    status.state = "provisioning"
    checks = []
    clear = [False]
    monkeypatch.setattr(
        runtime, "_confirm_run_clear", lambda _: checks.append(status.state) or clear[0]
    )
    reservations = []
    monkeypatch.setattr(
        runtime,
        "_start_handleless_resubmit",
        lambda _, expected: reservations.append(expected) or True,
    )
    assert isinstance(callbacks[0](), Retry)
    assert checks == ["provisioning"]
    assert reservations == []
    status.state = "running"
    clear[0] = True
    assert callbacks[0]() is None
    assert checks == ["provisioning", "running"]
    assert reservations == ["running"]


def test_cleanup_overflow_is_rediscovered_from_persisted_records(monkeypatch):
    from flash.server.platform import runtime

    owner = Supervisor({"cleanup": Limits(1, 1)})
    cleaned = threading.Event()
    monkeypatch.setattr(runtime, "_cleanup_pending", lambda run_id: run_id == "needs-cleanup")
    monkeypatch.setattr(runtime, "_drain_cleanup_remotes_bg", lambda _run: cleaned.set())
    try:
        with owner.bind(), owner.reserve("busy", "cleanup"):
            runtime._schedule_cleanup("terminal-without-resources")
            runtime._schedule_cleanup("needs-cleanup")
            assert not owner.contains(("cleanup", "needs-cleanup"))
        with owner.bind():
            runtime._schedule_cleanup("needs-cleanup")
        assert cleaned.wait(1)
    finally:
        assert owner.close(1)


def test_queued_replacement_does_not_reap_a_newly_adopted_worker(monkeypatch):
    from types import SimpleNamespace

    from flash.server.platform import runtime

    spec = SimpleNamespace(run_id="adopted-while-queued")
    monkeypatch.setattr(
        runtime,
        "get_status",
        lambda _run: SimpleNamespace(
            state="running",
            remote={"instance_id": "new-worker"},
            cleanup_confirmed_remote=None,
        ),
    )
    monkeypatch.setattr(
        runtime, "_confirm_run_clear", lambda _spec: pytest.fail("must not reap an adopted worker")
    )
    assert runtime._resubmit_task(spec) is None

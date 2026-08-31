"""Which control-plane process may recover the managed-teacher ledger, and when.

Ledger recovery rewrites every live request, so it is safe only while no process is serving. Every
contract here is about that ownership boundary rather than about broker request handling, and each
one needs real separate processes to mean anything -- an in-process test shares one lease table and
cannot express turnover at all. That is why they live apart from ``test_teacher_broker``.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import pathlib
import sqlite3

import pytest

from flash.server.domain.teacher import broker as teacher_broker
from flash.server.platform import db
from tests._helpers.teacher_broker import PLANE_API_KEY, _body, _issue, _limits, _response
from tests._helpers.teacher_broker import broker_db as broker_db


def _configure_broker_lifespan(db_path):
    from fastapi.testclient import TestClient

    import flash.server.asgi.app as app_mod
    from flash.providers.core import preflight
    from flash.runner.lifecycle import reporting
    from flash.server.billing import retry as billing_retry
    from flash.server.domain.ops import reconcile, repo_cleanup
    from flash.server.routes import serving

    db.DB_PATH = db_path
    preflight.check_run_preflight = lambda: None
    reporting._open_status_reporter = lambda: None
    reporting._shutdown_status_reporter = lambda *_args, **_kwargs: None
    billing_retry.charge_retry_enabled = lambda: False
    reconcile.reconcile_enabled = lambda: False
    repo_cleanup.repo_cleanup_enabled = lambda: False
    app_mod._instance_providers_configured = lambda: False
    app_mod._open_deployment_jobs = lambda: None
    app_mod._wait_for_deployment_jobs = lambda _timeout: True
    app_mod.recover_runs = lambda: None
    serving.recover_deployments = lambda: 0
    serving.replay_status_reports = lambda _stop: 0
    return app_mod, TestClient


def _hold_broker_lifespan(db_path, entered, release, errors):
    try:
        app_mod, test_client = _configure_broker_lifespan(db_path)
        with test_client(app_mod.create_app()):
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("broker lifespan release timed out")
    except BaseException as exc:
        errors.put(repr(exc))
        raise


def _hold_broker_lifespan_observing_recovery(
    db_path,
    recovery_attempted,
    entered,
    release,
    errors,
):
    try:
        app_mod, test_client = _configure_broker_lifespan(db_path)
        original_lease = app_mod._teacher_recovery_lease

        @contextlib.contextmanager
        def observed_lease():
            recovery_attempted.set()
            with original_lease():
                yield

        app_mod._teacher_recovery_lease = observed_lease
        with test_client(app_mod.create_app()):
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("broker lifespan release timed out")
    except BaseException as exc:
        errors.put(repr(exc))
        raise


def _run_live_teacher_request(
    db_path,
    capability_token,
    request_id,
    provider_entered,
    provider_release,
    request_completed,
    errors,
):
    try:
        app_mod, test_client = _configure_broker_lifespan(db_path)

        def dispatch(*_args):
            provider_entered.set()
            if not provider_release.wait(timeout=10):
                raise TimeoutError("provider release timed out")
            return 200, _response()

        teacher_broker._require_current_attempt = lambda _capability: None
        teacher_broker._provider_post = dispatch
        os.environ["PARASAIL_API_KEY"] = PLANE_API_KEY
        with test_client(app_mod.create_app()) as client:
            response = client.post(
                "/v1/teacher/completions",
                content=_body(),
                headers={
                    "authorization": f"Bearer {capability_token}",
                    "content-type": "application/json",
                    "x-flash-teacher-request-id": request_id,
                },
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"teacher request failed: {response.status_code} {response.text}"
                )
            request_completed.set()
    except BaseException as exc:
        errors.put(repr(exc))
        raise


def _join_broker_process(process, errors):
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail(f"broker process {process.name} did not exit")
    if process.exitcode != 0:
        detail = errors.get(timeout=2) if not errors.empty() else "no child error reported"
        pytest.fail(f"broker process {process.name} exited {process.exitcode}: {detail}")


def _teacher_request_accounting(db_path, request_id):
    connection = sqlite3.connect(db_path)
    row = connection.execute(
        "SELECT state FROM teacher_score_requests WHERE request_id = ?", (request_id,)
    ).fetchone()
    in_flight = connection.execute("SELECT in_flight FROM teacher_capabilities").fetchone()[0]
    connection.close()
    return row[0], in_flight


def _seed_started_request(db_path, request_id):
    db.DB_PATH = str(db_path)
    owner = db.ensure_internal_key(f"owner-{request_id}")
    db.record_run("run-1", owner["id"])
    token = _issue(limits=_limits(max_upstream_attempts=2))
    reservation = db.reserve_teacher_request(
        token=token,
        request_id=request_id,
        request_fingerprint="a" * 64,
        request_bytes=10,
        score_items=1,
        expected_run_id="run-1",
        expected_attempt=2,
    )
    db.mark_teacher_request_started(reservation["capability"]["id"], request_id)


def test_second_lifespan_does_not_recover_a_live_broker_request(broker_db):
    pytest.importorskip("fastapi")
    context = multiprocessing.get_context("spawn")
    token = _issue(limits=_limits(max_concurrency=1))
    provider_entered = context.Event()
    provider_release = context.Event()
    request_completed = context.Event()
    second_recovery_attempted = context.Event()
    second_entered = context.Event()
    second_release = context.Event()
    errors = context.Queue()
    first = context.Process(
        target=_run_live_teacher_request,
        args=(
            str(broker_db),
            token,
            "request-live-ownership-01",
            provider_entered,
            provider_release,
            request_completed,
            errors,
        ),
        name="live-teacher-broker",
    )
    second = context.Process(
        target=_hold_broker_lifespan_observing_recovery,
        args=(
            str(broker_db),
            second_recovery_attempted,
            second_entered,
            second_release,
            errors,
        ),
        name="second-control-plane",
    )

    first.start()
    try:
        assert provider_entered.wait(timeout=10)
        second.start()
        assert second_recovery_attempted.wait(timeout=10)
        assert not second_entered.wait(timeout=0.5)
        assert _teacher_request_accounting(broker_db, "request-live-ownership-01") == (
            "started",
            1,
        )
        provider_release.set()
        assert request_completed.wait(timeout=10)
        assert second_entered.wait(timeout=10)
    finally:
        provider_release.set()
        second_release.set()
        if first.pid is not None:
            _join_broker_process(first, errors)
        if second.pid is not None:
            _join_broker_process(second, errors)

    assert _teacher_request_accounting(broker_db, "request-live-ownership-01") == (
        "succeeded",
        0,
    )


def test_broker_recovery_lease_is_scoped_to_database_path(tmp_path):
    pytest.importorskip("fastapi")
    context = multiprocessing.get_context("spawn")
    first_db = tmp_path / "first" / "server.db"
    second_db = tmp_path / "second" / "server.db"
    _seed_started_request(first_db, "request-first-stale-001")
    _seed_started_request(second_db, "request-second-stale-01")
    first_entered = context.Event()
    second_entered = context.Event()
    release = context.Event()
    errors = context.Queue()
    first = context.Process(
        target=_hold_broker_lifespan,
        args=(str(first_db), first_entered, release, errors),
        name="first-ledger-broker",
    )
    second = context.Process(
        target=_hold_broker_lifespan,
        args=(str(second_db), second_entered, release, errors),
        name="second-ledger-broker",
    )

    first.start()
    second.start()
    try:
        assert first_entered.wait(timeout=10)
        assert second_entered.wait(timeout=10)
        assert _teacher_request_accounting(first_db, "request-first-stale-001") == (
            "outcome_unknown",
            0,
        )
        assert _teacher_request_accounting(second_db, "request-second-stale-01") == (
            "outcome_unknown",
            0,
        )
    finally:
        release.set()
        _join_broker_process(first, errors)
        _join_broker_process(second, errors)


def test_fresh_broker_process_recovers_stale_teacher_request(tmp_path):
    pytest.importorskip("fastapi")
    context = multiprocessing.get_context("spawn")
    broker_path = tmp_path / "server.db"
    _seed_started_request(broker_path, "request-restart-stale-1")
    entered = context.Event()
    release = context.Event()
    errors = context.Queue()
    process = context.Process(
        target=_hold_broker_lifespan,
        args=(str(broker_path), entered, release, errors),
        name="restarted-broker",
    )

    process.start()
    try:
        assert entered.wait(timeout=10)
        assert _teacher_request_accounting(broker_path, "request-restart-stale-1") == (
            "outcome_unknown",
            0,
        )
    finally:
        release.set()
        _join_broker_process(process, errors)


def _observe_ledger_state_during_run_recovery(db_path, request_id, observed_path, errors):
    """Record the ledger state visible to `recover_runs`, which resubmits OPD work."""
    try:
        app_mod, test_client = _configure_broker_lifespan(db_path)

        def observe_during_run_recovery():
            state, _in_flight = _teacher_request_accounting(db_path, request_id)
            pathlib.Path(observed_path).write_text(state, encoding="utf-8")

        app_mod.recover_runs = observe_during_run_recovery
        with test_client(app_mod.create_app()):
            pass
    except BaseException as exc:
        errors.put(repr(exc))
        raise


def test_ledger_recovery_precedes_run_recovery(tmp_path):
    """`recover_runs` resubmits OPD runs, so the ledger must be settled before it runs."""
    pytest.importorskip("fastapi")
    context = multiprocessing.get_context("spawn")
    broker_path = tmp_path / "server.db"
    _seed_started_request(broker_path, "request-ordering-stale-1")
    observed = tmp_path / "observed-state"
    errors = context.Queue()
    process = context.Process(
        target=_observe_ledger_state_during_run_recovery,
        args=(str(broker_path), "request-ordering-stale-1", str(observed), errors),
        name="ordering-broker",
    )

    process.start()
    _join_broker_process(process, errors)
    assert observed.read_text(encoding="utf-8") == "outcome_unknown"


def _crash_during_live_teacher_request(db_path, capability_token, provider_entered):
    app_mod, test_client = _configure_broker_lifespan(db_path)

    def crash(*_args):
        provider_entered.set()
        os._exit(17)

    teacher_broker._require_current_attempt = lambda _capability: None
    teacher_broker._provider_post = crash
    os.environ["PARASAIL_API_KEY"] = PLANE_API_KEY
    with test_client(app_mod.create_app()) as client:
        client.post(
            "/v1/teacher/completions",
            content=_body(),
            headers={
                "authorization": f"Bearer {capability_token}",
                "content-type": "application/json",
                "x-flash-teacher-request-id": "request-crashed-owner-001",
            },
        )


def test_replacement_recovers_a_crashed_broker_while_a_sibling_survives(broker_db):
    pytest.importorskip("fastapi")
    context = multiprocessing.get_context("spawn")
    token = _issue(limits=_limits(max_concurrency=1))
    survivor_entered = context.Event()
    survivor_release = context.Event()
    provider_entered = context.Event()
    replacement_entered = context.Event()
    replacement_release = context.Event()
    errors = context.Queue()
    survivor = context.Process(
        target=_hold_broker_lifespan,
        args=(str(broker_db), survivor_entered, survivor_release, errors),
        name="surviving-broker",
    )
    crashed = context.Process(
        target=_crash_during_live_teacher_request,
        args=(str(broker_db), token, provider_entered),
        name="crashed-broker",
    )
    replacement = context.Process(
        target=_hold_broker_lifespan,
        args=(str(broker_db), replacement_entered, replacement_release, errors),
        name="replacement-broker",
    )

    survivor.start()
    try:
        assert survivor_entered.wait(timeout=10)
        crashed.start()
        assert provider_entered.wait(timeout=10)
        crashed.join(timeout=5)
        assert crashed.exitcode == 17
        replacement.start()
        assert replacement_entered.wait(timeout=10)
        assert _teacher_request_accounting(broker_db, "request-crashed-owner-001") == (
            "outcome_unknown",
            0,
        )
    finally:
        survivor_release.set()
        replacement_release.set()
        if crashed.is_alive():
            crashed.terminate()
            crashed.join(timeout=5)
        if survivor.pid is not None:
            _join_broker_process(survivor, errors)
        if replacement.pid is not None:
            _join_broker_process(replacement, errors)


def _fail_first_recovery(db_path, recovery_entered, recovery_release, errors):
    try:
        app_mod, test_client = _configure_broker_lifespan(db_path)

        def fail_recovery():
            recovery_entered.set()
            if not recovery_release.wait(timeout=10):
                raise TimeoutError("failed recovery release timed out")
            raise RuntimeError("recovery owner failed")

        db.recover_teacher_request_ledger = fail_recovery
        with (
            pytest.raises(RuntimeError, match="recovery owner failed"),
            test_client(app_mod.create_app()),
        ):
            pass
    except BaseException as exc:
        errors.put(repr(exc))
        raise


def test_contender_retries_recovery_after_the_first_owner_fails(tmp_path):
    pytest.importorskip("fastapi")
    context = multiprocessing.get_context("spawn")
    broker_path = tmp_path / "server.db"
    _seed_started_request(broker_path, "request-failed-owner-01")
    recovery_entered = context.Event()
    recovery_release = context.Event()
    replacement_recovery_attempted = context.Event()
    replacement_entered = context.Event()
    replacement_release = context.Event()
    errors = context.Queue()
    failed_owner = context.Process(
        target=_fail_first_recovery,
        args=(str(broker_path), recovery_entered, recovery_release, errors),
        name="failed-recovery-owner",
    )
    replacement = context.Process(
        target=_hold_broker_lifespan_observing_recovery,
        args=(
            str(broker_path),
            replacement_recovery_attempted,
            replacement_entered,
            replacement_release,
            errors,
        ),
        name="recovery-contender",
    )

    failed_owner.start()
    try:
        assert recovery_entered.wait(timeout=10)
        replacement.start()
        assert replacement_recovery_attempted.wait(timeout=10)
        assert not replacement_entered.wait(timeout=0.5)
        recovery_release.set()
        _join_broker_process(failed_owner, errors)
        assert replacement_entered.wait(timeout=10)
        assert _teacher_request_accounting(broker_path, "request-failed-owner-01") == (
            "outcome_unknown",
            0,
        )
    finally:
        recovery_release.set()
        replacement_release.set()
        if failed_owner.is_alive():
            failed_owner.terminate()
            failed_owner.join(timeout=5)
        if replacement.pid is not None:
            _join_broker_process(replacement, errors)


def test_lease_failure_precedes_lifespan_resource_creation(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import flash.server.asgi.app as app_mod
    from flash.providers.core import preflight

    monkeypatch.setattr(preflight, "check_run_preflight", lambda: None)

    monkeypatch.setattr(
        app_mod,
        "_teacher_recovery_lease",
        lambda: (_ for _ in ()).throw(PermissionError("lease denied")),
    )
    monkeypatch.setattr(
        app_mod,
        "_open_deployment_jobs",
        lambda: pytest.fail("deployment jobs opened before the broker lease"),
    )
    monkeypatch.setattr(
        app_mod,
        "recover_runs",
        lambda: pytest.fail("run recovery started before the broker lease"),
    )

    with pytest.raises(PermissionError, match="lease denied"), TestClient(app_mod.create_app()):
        pass


def test_later_startup_failure_releases_lease_and_unwinds_resources(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import flash.server.asgi.app as app_mod
    from flash.providers.core import preflight
    from flash.runner.lifecycle import reporting

    events = []
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setattr(preflight, "check_run_preflight", lambda: events.append("preflight"))

    @contextlib.contextmanager
    def recovery_lease():
        events.append("enter_recovery_lease")
        try:
            yield
        finally:
            events.append("release_recovery_lease")

    monkeypatch.setattr(app_mod, "_teacher_recovery_lease", recovery_lease)
    monkeypatch.setattr(db, "recover_teacher_request_ledger", lambda: events.append("recover"))
    monkeypatch.setattr(app_mod, "_open_deployment_jobs", lambda: events.append("open_jobs"))

    def fail_reporter():
        events.append("open_reporter")
        raise RuntimeError("reporter startup failed")

    monkeypatch.setattr(reporting, "_open_status_reporter", fail_reporter)
    monkeypatch.setattr(
        app_mod,
        "_wait_for_deployment_jobs",
        lambda _timeout: events.append("close_jobs") or True,
    )
    with (
        pytest.raises(RuntimeError, match="reporter startup failed"),
        TestClient(app_mod.create_app()),
    ):
        pass

    assert events == [
        "preflight",
        "enter_recovery_lease",
        "recover",
        "release_recovery_lease",
        "open_jobs",
        "open_reporter",
        "close_jobs",
    ]


def _hold_recovery_lease(db_path, entered, release, errors):
    try:
        from flash.server.platform import db as db_mod
        from flash.server.platform.locks import _teacher_recovery_lease

        db_mod.DB_PATH = db_path
        with _teacher_recovery_lease():
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("recovery lease release timed out")
    except BaseException as exc:
        errors.put(repr(exc))
        raise


def _hold_serving_lease(db_path, entered, release, errors):
    try:
        from flash.server.platform import db as db_mod
        from flash.server.platform.locks import _teacher_serving_lease

        db_mod.DB_PATH = db_path
        with _teacher_serving_lease():
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("serving lease release timed out")
    except BaseException as exc:
        errors.put(repr(exc))
        raise


def test_broker_recovery_lease_excludes_an_aliased_database_path(tmp_path):
    """Two names for one ledger must share a lease, not produce independent ones."""
    context = multiprocessing.get_context("spawn")
    real = tmp_path / "server.db"
    real.write_bytes(b"")
    alias = tmp_path / "alias.db"
    alias.symlink_to(real)
    recovery_entered = context.Event()
    recovery_release = context.Event()
    serving_entered = context.Event()
    serving_release = context.Event()
    errors = context.Queue()
    recovery = context.Process(
        target=_hold_recovery_lease,
        args=(str(real), recovery_entered, recovery_release, errors),
        name="aliased-recovery-owner",
    )
    serving = context.Process(
        target=_hold_serving_lease,
        args=(str(alias), serving_entered, serving_release, errors),
        name="aliased-serving-owner",
    )

    recovery.start()
    try:
        assert recovery_entered.wait(timeout=10)
        serving.start()
        assert not serving_entered.wait(timeout=0.5)
        recovery_release.set()
        assert serving_entered.wait(timeout=10)
    finally:
        recovery_release.set()
        serving_release.set()
        if recovery.pid is not None:
            _join_broker_process(recovery, errors)
        if serving.pid is not None:
            _join_broker_process(serving, errors)

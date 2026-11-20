"""A custom run whose controller is lost (not its algorithm) fails as infrastructure.

Five custom Spot runs from five projects failed together on 2026-11-19 at 21:55-21:59 UTC: the
control plane was OOM-killed and restarted, so the controllers (which attach through it) could
not reach their sessions for longer than idle_seconds. The worker ended each session as "custom
controller disconnected or timed out", which was classified as the run's own failure and never
retried, and the supervised path then persisted only "RuntimeError: run failed".
"""

from types import SimpleNamespace

import pytest


def _session(**fields):
    defaults = {"attached_at": 1.0, "failed": False, "idle_seconds": 120}
    return SimpleNamespace(**{**defaults, **fields})


def test_lost_controller_is_a_retriable_infrastructure_fault():
    from flash.custom.worker import unfinished_session_error
    from flash.engine.worker.entry.worker import _worker_failure_flags
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    error = unfinished_session_error(_session(), "ended", None)
    assert isinstance(error, RetriableInfraError)
    assert "custom controller disconnected or timed out" in str(error)
    assert "120s" in str(error)
    # the worker marks the attempt retriable, so Spot resumes from the newest checkpoint
    assert _worker_failure_flags(error) == {"retriable": True, "oom": False}

    saved = unfinished_session_error(_session(), "ended", "run/step-3")
    assert isinstance(saved, RetriableInfraError)
    assert "saved checkpoint run/step-3" in str(saved)


@pytest.mark.parametrize(
    ("fields", "reason", "match"),
    [
        ({"attached_at": None}, "ended", "no custom controller attached"),
        ({"failed": True}, "ended", "session operation failed"),
        ({}, "wall", "wall-clock cap"),
    ],
)
def test_algorithm_side_endings_stay_job_failures(fields, reason, match):
    from flash.custom.worker import unfinished_session_error
    from flash.engine.worker.entry.worker import _worker_failure_flags

    error = unfinished_session_error(_session(**fields), reason, None)
    assert match in str(error)
    assert _worker_failure_flags(error)["retriable"] is False


def test_spot_custom_run_retries_a_lost_controller_and_classifies_it_infra():
    from flash.providers.core.base import Candidate
    from flash.runner.supervise.retry_decision import (
        FailureObservation,
        RetryState,
        failure_class,
        transition_failure,
    )
    from flash.schema import spec_from_dict

    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "custom",
            "custom": {"file": "algorithm.py", "idle_seconds": 30},
            "gpu": {"provider": "spot", "spot": True},
        }
    )
    state = RetryState.initial_for_spec(spec)
    chosen = Candidate("spot", "H100", 1.0, 80)
    state, plan = transition_failure(
        state, FailureObservation("worker_retriable", chosen=chosen), attempt=0
    )
    assert plan.retry
    assert "resume from last checkpoint" in plan.action
    assert failure_class("worker_retriable") == "infra"


def test_spot_address_lookup_outage_is_retryable_not_a_500(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from flash.providers.mithril import api, handles
    from flash.providers.mithril.client import MithrilApiError
    from flash.server.platform.deps import require_key
    from flash.server.routes import custom

    remote = {
        "provider": "spot",
        "instance_id": "bid_test",
        "name": "flash-spot-run-2",
        "run_id": "spot-run",
        "hf_repo": "owner/repo",
        "phase": "custom",
        "gpu_count": 1,
        "scope": "scope",
        "gpu": "H100",
        "hourly_usd": 1.0,
        "attempt": 2,
        "started_ts": 1.0,
    }
    app = FastAPI()
    app.include_router(custom.router)
    app.dependency_overrides[require_key] = lambda: {"id": 1}
    status = SimpleNamespace(spec={"algorithm": "custom"}, state="running", remote=remote)
    monkeypatch.setattr(custom, "require_owned_run", lambda run_id, key: status)
    monkeypatch.setattr(handles, "require_scope", lambda handle: None)

    def outage(instance_id):
        raise MithrilApiError(status=503)

    monkeypatch.setattr(api, "public_ip", outage)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/v1/runs/spot-run/custom")
    # controllers retry 503 (and 425); a 500 made them treat the session as gone
    assert response.status_code == 503


def test_failed_run_error_keeps_the_last_attempt_detail():
    from flash.runner.supervise.errors import RunAttemptsFailed
    from flash.runner.supervise.lifecycle import _terminal_failure_detail

    detail = (
        "worker_retriable: RetriableInfraError: custom controller disconnected or timed out "
        "(no request reached the session for 30s); resume from the last durable checkpoint"
    )
    error = RunAttemptsFailed("flash-1", detail)
    assert isinstance(error, RuntimeError)
    assert _terminal_failure_detail(error) == detail
    # any other exception still keeps only its type: its text may carry internal payloads
    assert _terminal_failure_detail(ValueError("secret body")) == "ValueError: run failed"


def test_unreconciled_provider_create_is_infra_but_never_retried():
    from flash.providers.core.base import Candidate
    from flash.runner.supervise.retry_decision import (
        UNRECONCILED_CREATE_FAILURE,
        FailureObservation,
        RetryState,
        failure_class,
        transition_failure,
    )

    chosen = Candidate("spot", "H200", 3.2, 141)
    _, plan = transition_failure(
        RetryState(5, 0, 0),
        FailureObservation(UNRECONCILED_CREATE_FAILURE, chosen=chosen),
        attempt=7,
    )
    assert not plan.retry
    assert failure_class(UNRECONCILED_CREATE_FAILURE) == "infra"

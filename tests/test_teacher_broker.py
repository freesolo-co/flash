"""Focused managed-teacher broker security and lifecycle contracts."""

from __future__ import annotations

import io
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from flash.server import db, teacher_broker
from flash.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec


def _limits(**updates):
    values = {
        "max_requests": 4,
        "max_score_items": 8,
        "max_request_bytes": teacher_broker.MAX_REQUEST_BODY_BYTES,
        "max_response_bytes": teacher_broker.MAX_RESPONSE_BODY_BYTES,
        "max_concurrency": 2,
        "max_upstream_attempts": 1,
        "max_request_tokens": 128,
        "max_total_tokens": 512,
    }
    values.update(updates)
    return values


@pytest.fixture
def broker_db(monkeypatch, tmp_path):
    path = tmp_path / "server.db"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    owner = db.ensure_internal_key("test-owner-key")
    db.record_run("run-1", owner["id"])
    return path


def _issue(*, now=None, expires_at=None, limits=None):
    current = time.time() if now is None else float(now)
    return db.issue_teacher_capability(
        run_id="run-1",
        attempt=2,
        teacher_alias="glm-5.2",
        provider=teacher_broker.FIREWORKS_PROVIDER,
        model="accounts/fireworks/models/glm-5p2",
        scoring_mode=teacher_broker.FIREWORKS_SCORING_MODE,
        expires_at=expires_at if expires_at is not None else current + 600,
        limits=limits or _limits(),
        now=current,
    )


def _body(*, prompt="questionanswer", model="accounts/fireworks/models/glm-5p2", **extra):
    value = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
        "temperature": 0,
    }
    value.update(extra)
    return json.dumps(value, separators=(",", ":")).encode()


def _response(*, token="questionanswer"):
    return json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "logprobs": {
                        "tokens": [token],
                        "token_logprobs": [-0.1],
                        "text_offset": [0],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        }
    ).encode()


def _service_ready(monkeypatch):
    monkeypatch.setattr(teacher_broker, "_require_current_attempt", lambda _capability: None)
    monkeypatch.setenv("FIREWORKS_API_KEY", "control-plane-only-canary")


def test_capability_persists_only_sha256_and_has_256_bits_of_entropy(broker_db):
    token = _issue()
    assert len(token) >= 43

    connection = sqlite3.connect(broker_db)
    row = connection.execute(
        "SELECT token_hash FROM teacher_capabilities WHERE run_id = 'run-1'"
    ).fetchone()
    dump = "\n".join(connection.iterdump())
    connection.close()

    assert row is not None
    assert len(row[0]) == 64
    assert token not in dump
    assert "control-plane-only-canary" not in dump


def test_capability_is_bound_to_attempt_and_new_attempt_revokes_old(broker_db):
    first = _issue()
    second = db.issue_teacher_capability(
        run_id="run-1",
        attempt=3,
        teacher_alias="glm-5.2",
        provider=teacher_broker.FIREWORKS_PROVIDER,
        model="accounts/fireworks/models/glm-5p2",
        scoring_mode=teacher_broker.FIREWORKS_SCORING_MODE,
        expires_at=time.time() + 600,
        limits=_limits(),
    )

    assert db.teacher_capability_binding(first)["revoked_at"] is not None
    assert db.teacher_capability_binding(second)["attempt"] == 3


def test_exact_24_hour_capability_deadline_is_accepted(broker_db, monkeypatch):
    monkeypatch.setenv("FLASH_TEACHER_BROKER_URL", "https://broker.example")
    monkeypatch.setenv("FIREWORKS_API_KEY", "control-plane-only-canary")
    issued_at = 1_000.0
    deadline_at = issued_at + teacher_broker.MAX_CAPABILITY_LIFETIME_S
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-1",
    )

    _broker_url, token = teacher_broker.issue_teacher_capability(
        spec,
        attempt=2,
        deadline_at=deadline_at,
        now=issued_at,
    )

    assert db.teacher_capability_binding(token)["expires_at"] == deadline_at


def test_48_hour_opd_wall_is_rejected_before_allocation(monkeypatch):
    import flash.providers.allocator as allocator
    from flash.runner import lifecycle

    monkeypatch.setenv("FLASH_TEACHER_BROKER_URL", "https://broker.example")
    monkeypatch.setenv("FIREWORKS_API_KEY", "control-plane-only-canary")
    monkeypatch.setattr(
        lifecycle,
        "_pin_environment_for_run",
        lambda spec, _log, *, attempt_started: spec,
    )

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("allocation must not run for an opd wall over 24 hours")

    monkeypatch.setattr(allocator, "allocate", unexpected_allocation)
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        gpu=GpuSpec(max_wall_seconds=48 * 60 * 60),
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-over-capability-wall",
    )

    with pytest.raises(ValueError, match="24 hours or less"):
        lifecycle._submit_seed_supervised(spec, 42, io.StringIO())


def test_deadline_contract_over_24_hours_is_rejected(monkeypatch):
    monkeypatch.setenv("FLASH_TEACHER_BROKER_URL", "https://broker.example")
    monkeypatch.setenv("FIREWORKS_API_KEY", "control-plane-only-canary")
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-over-capability-deadline",
    )
    now = 1_000.0

    with pytest.raises(RuntimeError, match="24-hour run deadline"):
        teacher_broker.require_teacher_broker_configuration(
            spec,
            deadline_at=now + teacher_broker.MAX_CAPABILITY_LIFETIME_S + 1,
            now=now,
        )


def test_wrong_expired_and_revoked_capabilities_fail_closed(broker_db):
    with pytest.raises(db.TeacherLedgerError, match="invalid_capability"):
        db.teacher_capability_binding("not-a-capability")

    expired = _issue(now=10, expires_at=11)
    with pytest.raises(db.TeacherLedgerError, match="expired_capability"):
        db.reserve_teacher_request(
            token=expired,
            request_id="request-expired-0001",
            request_fingerprint="f" * 64,
            request_bytes=10,
            score_items=1,
            expected_run_id="run-1",
            expected_attempt=2,
            now=12,
        )

    db.delete_run("run-1")
    owner = db.ensure_internal_key("test-owner-key")
    db.record_run("run-1", owner["id"])
    revoked = _issue()
    db.revoke_teacher_capability(revoked)
    with pytest.raises(db.TeacherLedgerError, match="revoked_capability"):
        db.reserve_teacher_request(
            token=revoked,
            request_id="request-revoked-0001",
            request_fingerprint="f" * 64,
            request_bytes=10,
            score_items=1,
            expected_run_id="run-1",
            expected_attempt=2,
        )


def test_quota_and_concurrency_admission_use_immediate_transactions(broker_db):
    token = _issue(limits=_limits(max_requests=2, max_score_items=2, max_concurrency=1))
    first = db.reserve_teacher_request(
        token=token,
        request_id="request-quota-000001",
        request_fingerprint="a" * 64,
        request_bytes=10,
        score_items=1,
        expected_run_id="run-1",
        expected_attempt=2,
    )
    with pytest.raises(db.TeacherLedgerError, match="broker_busy"):
        db.reserve_teacher_request(
            token=token,
            request_id="request-quota-000002",
            request_fingerprint="b" * 64,
            request_bytes=10,
            score_items=1,
            expected_run_id="run-1",
            expected_attempt=2,
        )
    db.retry_teacher_request_before_dispatch(
        first["capability"]["id"], "request-quota-000001", error_class="local_busy"
    )
    db.reserve_teacher_request(
        token=token,
        request_id="request-quota-000002",
        request_fingerprint="b" * 64,
        request_bytes=10,
        score_items=1,
        expected_run_id="run-1",
        expected_attempt=2,
    )
    with pytest.raises(db.TeacherLedgerError, match="request_quota_exhausted"):
        db.reserve_teacher_request(
            token=token,
            request_id="request-quota-000003",
            request_fingerprint="c" * 64,
            request_bytes=10,
            score_items=1,
            expected_run_id="run-1",
            expected_attempt=2,
        )


def test_total_token_quota_is_reserved_before_dispatch_and_never_overshoots(broker_db):
    token = _issue(
        limits=_limits(
            max_requests=4,
            max_score_items=4,
            max_concurrency=3,
            max_request_tokens=64,
            max_total_tokens=128,
        )
    )
    first = db.reserve_teacher_request(
        token=token,
        request_id="request-tokens-000001",
        request_fingerprint="a" * 64,
        request_bytes=10,
        score_items=1,
        expected_run_id="run-1",
        expected_attempt=2,
    )
    second = db.reserve_teacher_request(
        token=token,
        request_id="request-tokens-000002",
        request_fingerprint="b" * 64,
        request_bytes=10,
        score_items=1,
        expected_run_id="run-1",
        expected_attempt=2,
    )
    assert db.teacher_capability_usage(token)["capability"]["token_count"] == 128
    with pytest.raises(db.TeacherLedgerError, match="token_quota_exhausted"):
        db.reserve_teacher_request(
            token=token,
            request_id="request-tokens-000003",
            request_fingerprint="c" * 64,
            request_bytes=10,
            score_items=1,
            expected_run_id="run-1",
            expected_attempt=2,
        )

    db.retry_teacher_request_before_dispatch(
        first["capability"]["id"], "request-tokens-000001", error_class="local_busy"
    )
    assert db.teacher_capability_usage(token)["capability"]["token_count"] == 64
    db.mark_teacher_request_started(second["capability"]["id"], "request-tokens-000002")
    db.complete_teacher_request(
        second["capability"]["id"],
        "request-tokens-000002",
        state="succeeded",
        input_tokens=10,
    )
    assert db.teacher_capability_usage(token)["capability"]["token_count"] == 10
    db.reserve_teacher_request(
        token=token,
        request_id="request-tokens-000003",
        request_fingerprint="c" * 64,
        request_bytes=10,
        score_items=1,
        expected_run_id="run-1",
        expected_attempt=2,
    )
    assert db.teacher_capability_usage(token)["capability"]["token_count"] == 74


def test_eight_item_batch_uses_aggregate_token_limit_across_ledger_states(broker_db):
    token = _issue(
        limits=_limits(
            max_requests=1,
            max_score_items=8,
            max_concurrency=1,
            max_request_tokens=2,
            max_total_tokens=16,
        )
    )
    request_id = "request-eight-item-001"
    reservation = db.reserve_teacher_request(
        token=token,
        request_id=request_id,
        request_fingerprint="a" * 64,
        request_bytes=10,
        score_items=8,
        expected_run_id="run-1",
        expected_attempt=2,
    )
    capability_id = reservation["capability"]["id"]
    assert db.teacher_capability_usage(token)["capability"]["token_count"] == 16

    db.retry_teacher_request_before_dispatch(capability_id, request_id, error_class="local_busy")
    assert db.teacher_capability_usage(token)["capability"]["token_count"] == 0
    db.reserve_teacher_request(
        token=token,
        request_id=request_id,
        request_fingerprint="a" * 64,
        request_bytes=10,
        score_items=8,
        expected_run_id="run-1",
        expected_attempt=2,
    )
    assert db.teacher_capability_usage(token)["capability"]["token_count"] == 16

    assert db.recover_teacher_request_ledger() == {"retryable": 1, "outcome_unknown": 0}
    assert db.teacher_capability_usage(token)["capability"]["token_count"] == 0
    db.reserve_teacher_request(
        token=token,
        request_id=request_id,
        request_fingerprint="a" * 64,
        request_bytes=10,
        score_items=8,
        expected_run_id="run-1",
        expected_attempt=2,
    )
    db.mark_teacher_request_started(capability_id, request_id)
    db.complete_teacher_request(
        capability_id,
        request_id,
        state="succeeded",
        input_tokens=16,
    )
    assert db.teacher_capability_usage(token)["capability"]["token_count"] == 16

    choices = [
        {
            "index": index,
            "logprobs": {
                "tokens": ["a", "b"],
                "token_logprobs": [-0.1, -0.1],
                "text_offset": [0, 1],
            },
        }
        for index in range(8)
    ]
    capability = db.teacher_capability_binding(token)
    exact = json.dumps(
        {
            "choices": choices,
            "usage": {"prompt_tokens": 16, "completion_tokens": 0, "total_tokens": 16},
        }
    ).encode()
    assert (
        teacher_broker._validated_fireworks_response(
            exact, score_items=8, capability=capability
        ).input_tokens
        == 16
    )
    over_limit = json.dumps(
        {
            "choices": choices,
            "usage": {"prompt_tokens": 17, "completion_tokens": 0, "total_tokens": 17},
        }
    ).encode()
    with pytest.raises(teacher_broker.TeacherBrokerError, match="provider_token_limit_exceeded"):
        teacher_broker._validated_fireworks_response(
            over_limit,
            score_items=8,
            capability=capability,
        )


def test_stale_started_request_becomes_outcome_unknown_and_never_redispatches(broker_db):
    token = _issue()
    reservation = db.reserve_teacher_request(
        token=token,
        request_id="request-stale-000001",
        request_fingerprint="a" * 64,
        request_bytes=10,
        score_items=1,
        expected_run_id="run-1",
        expected_attempt=2,
    )
    db.mark_teacher_request_started(reservation["capability"]["id"], "request-stale-000001")
    db.reserve_teacher_request(
        token=token,
        request_id="request-reserved-0001",
        request_fingerprint="b" * 64,
        request_bytes=10,
        score_items=1,
        expected_run_id="run-1",
        expected_attempt=2,
    )

    recovered = db.recover_teacher_request_ledger()

    assert recovered == {"retryable": 1, "outcome_unknown": 1}
    assert db.teacher_capability_usage(token)["capability"]["token_count"] == 128
    with pytest.raises(db.TeacherLedgerError, match="outcome_unknown"):
        db.reserve_teacher_request(
            token=token,
            request_id="request-stale-000001",
            request_fingerprint="a" * 64,
            request_bytes=10,
            score_items=1,
            expected_run_id="run-1",
            expected_attempt=2,
        )


def test_closed_fireworks_contract_rejects_extra_fields_model_changes_and_bad_images(broker_db):
    token = _issue()
    capability = db.teacher_capability_binding(token)

    with pytest.raises(teacher_broker.TeacherBrokerError, match="extra_request_fields"):
        teacher_broker.validate_completion_request(
            teacher_broker.parse_strict_json(_body(destination="https://attacker.invalid")),
            capability,
        )
    with pytest.raises(teacher_broker.TeacherBrokerError, match="model_scope_mismatch"):
        teacher_broker.validate_completion_request(
            teacher_broker.parse_strict_json(_body(model="accounts/fireworks/models/other")),
            capability,
        )
    with pytest.raises(teacher_broker.TeacherBrokerError, match="teacher_does_not_support_images"):
        teacher_broker.validate_completion_request(
            teacher_broker.parse_strict_json(_body(images=["data:image/png;base64,YQ=="])),
            capability,
        )
    with pytest.raises(teacher_broker.TeacherBrokerError, match="invalid_prompt_batch"):
        teacher_broker.validate_completion_request(
            teacher_broker.parse_strict_json(_body(prompt=["x"] * 9)),
            capability,
        )
    kimi = {
        **capability,
        "teacher_alias": "kimi-k2.6",
        "model": "accounts/fireworks/models/kimi-k2p6",
    }
    valid_image = teacher_broker.validate_completion_request(
        teacher_broker.parse_strict_json(
            _body(
                model="accounts/fireworks/models/kimi-k2p6",
                images=["data:image/png;base64,YQ=="],
            )
        ),
        kimi,
    )
    assert valid_image.multimodal is True
    assert valid_image.score_items == 1


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers():
    with pytest.raises(teacher_broker.TeacherBrokerError, match="duplicate_json_key"):
        teacher_broker.parse_strict_json(b'{"model":"a","model":"b"}')
    with pytest.raises(teacher_broker.TeacherBrokerError, match="non_finite_number"):
        teacher_broker.parse_strict_json(b'{"temperature":NaN}')


def test_request_fingerprint_is_capability_keyed_hmac(broker_db):
    import hashlib
    import hmac

    token = _issue()
    canonical = teacher_broker.validate_completion_request(
        teacher_broker.parse_strict_json(_body()), db.teacher_capability_binding(token)
    ).canonical_body
    fingerprint = teacher_broker.request_fingerprint(token, canonical)

    assert fingerprint == hmac.new(token.encode(), canonical, hashlib.sha256).hexdigest()
    assert fingerprint != hashlib.sha256(canonical).hexdigest()


def test_concurrent_duplicate_request_id_dispatches_upstream_once(broker_db, monkeypatch):
    _service_ready(monkeypatch)
    token = _issue()
    entered = threading.Event()
    release = threading.Event()
    dispatches = []

    def dispatch(_body, _api_key, _timeout):
        dispatches.append(1)
        entered.set()
        release.wait(timeout=5)
        return 200, _response()

    monkeypatch.setattr(teacher_broker, "_fireworks_post", dispatch)

    def call():
        return teacher_broker.complete_fireworks_request(
            capability_token=token,
            request_id="request-concurrent-001",
            raw_body=_body(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(call)
        assert entered.wait(timeout=5)
        second = executor.submit(call)
        with pytest.raises(teacher_broker.TeacherBrokerError) as error:
            second.result(timeout=5)
        assert error.value.code == "request_in_progress"
        release.set()
        assert first.result(timeout=5)["choices"]

    assert len(dispatches) == 1


def test_fireworks_text_batch_is_one_upstream_call_and_counts_each_score_item(
    broker_db, monkeypatch
):
    _service_ready(monkeypatch)
    token = _issue()
    dispatches = []
    response = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "logprobs": {"tokens": ["a"], "token_logprobs": [-0.1], "text_offset": [0]},
                },
                {
                    "index": 1,
                    "logprobs": {"tokens": ["b"], "token_logprobs": [-0.2], "text_offset": [0]},
                },
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2},
        }
    ).encode()

    def dispatch(*_args):
        dispatches.append(1)
        return 200, response

    monkeypatch.setattr(teacher_broker, "_fireworks_post", dispatch)
    teacher_broker.complete_fireworks_request(
        capability_token=token,
        request_id="request-batch-0000001",
        raw_body=_body(prompt=["a", "b"]),
    )

    usage = db.teacher_capability_usage(token)["capability"]
    assert len(dispatches) == 1
    assert usage["request_count"] == 1
    assert usage["score_item_count"] == 2


def test_changed_body_and_success_replay_are_permanent_without_second_dispatch(
    broker_db, monkeypatch
):
    _service_ready(monkeypatch)
    token = _issue()
    dispatches = []
    monkeypatch.setattr(
        teacher_broker,
        "_fireworks_post",
        lambda *_args: (dispatches.append(1) or 200, _response()),
    )
    request_id = "request-replay-000001"

    teacher_broker.complete_fireworks_request(
        capability_token=token, request_id=request_id, raw_body=_body()
    )
    with pytest.raises(teacher_broker.TeacherBrokerError) as changed:
        teacher_broker.complete_fireworks_request(
            capability_token=token,
            request_id=request_id,
            raw_body=_body(prompt="changed"),
        )
    with pytest.raises(teacher_broker.TeacherBrokerError) as replay:
        teacher_broker.complete_fireworks_request(
            capability_token=token, request_id=request_id, raw_body=_body()
        )

    assert changed.value.code == "request_body_changed"
    assert replay.value.code == "replay_unavailable"
    assert len(dispatches) == 1


def test_provider_error_body_is_suppressed_from_response_and_sqlite(broker_db, monkeypatch):
    _service_ready(monkeypatch)
    token = _issue()
    private_canary = "private-provider-error-body-canary"
    dispatches = []

    def reject(*_args):
        dispatches.append(1)
        return 500, private_canary.encode()

    monkeypatch.setattr(teacher_broker, "_fireworks_post", reject)

    with pytest.raises(teacher_broker.TeacherBrokerError) as error:
        teacher_broker.complete_fireworks_request(
            capability_token=token,
            request_id="request-private-00001",
            raw_body=_body(),
        )

    connection = sqlite3.connect(broker_db)
    dump = "\n".join(connection.iterdump())
    connection.close()
    assert error.value.payload() == {
        "error": {
            "code": "provider_rejected",
            "classification": "permanent",
            "request_id": "request-private-00001",
        }
    }
    assert error.value.retryable is False
    with pytest.raises(teacher_broker.TeacherBrokerError) as duplicate:
        teacher_broker.complete_fireworks_request(
            capability_token=token,
            request_id="request-private-00001",
            raw_body=_body(),
        )
    assert duplicate.value.code == "provider_rejected"
    assert duplicate.value.retryable is False
    assert len(dispatches) == 1
    assert private_canary not in dump
    assert "control-plane-only-canary" not in dump
    assert token not in dump


def test_worker_reuses_one_logical_request_id_across_transport_retries(monkeypatch):
    import urllib.error

    from flash.engine.worker import teacher as worker_teacher

    payload = json.loads(_response())
    request_ids = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def urlopen(_transport, request, timeout=None):
        request_ids.append(dict(request.header_items())["X-flash-teacher-request-id"])
        if len(request_ids) == 1:
            raise urllib.error.URLError("broker unavailable")
        return Response()

    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _seconds: None)
    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "accounts/fireworks/models/glm-5p2"
    )

    assert client.score("question", "answer")
    assert len(request_ids) == 2
    assert len(set(request_ids)) == 1


def test_worker_retries_predispatch_http_error_with_same_request_id(monkeypatch):
    import urllib.error

    from flash.engine.worker import teacher as worker_teacher

    request_ids = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return _response()

    def urlopen(_transport, request, timeout=None):
        request_ids.append(dict(request.header_items())["X-flash-teacher-request-id"])
        if len(request_ids) == 1:
            body = json.dumps(
                {
                    "error": {
                        "code": "provider_unavailable",
                        "classification": "transient",
                    }
                }
            ).encode()
            raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {}, io.BytesIO(body))
        return Response()

    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _seconds: None)
    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "accounts/fireworks/models/glm-5p2"
    )

    assert client.score("question", "answer")
    assert len(request_ids) == 2
    assert len(set(request_ids)) == 1


def test_worker_does_not_retry_postdispatch_http_error(monkeypatch):
    import urllib.error

    from flash.engine.worker import teacher as worker_teacher

    request_ids = []

    def urlopen(_transport, request, timeout=None):
        request_ids.append(dict(request.header_items())["X-flash-teacher-request-id"])
        body = json.dumps(
            {
                "error": {
                    "code": "provider_rejected",
                    "classification": "transient",
                }
            }
        ).encode()
        raise urllib.error.HTTPError(request.full_url, 502, "rejected", {}, io.BytesIO(body))

    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _seconds: None)
    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "accounts/fireworks/models/glm-5p2"
    )

    with pytest.raises(worker_teacher.TeacherError) as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert "provider_rejected (permanent)" in str(error.value)
    assert len(request_ids) == 1


def test_operation_specific_route_requires_bearer_json_and_request_id(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from flash.server.routes import teacher as teacher_route

    captured = {}

    def complete(**kwargs):
        captured.update(kwargs)
        return {"choices": []}

    monkeypatch.setattr(teacher_route, "complete_fireworks_request", complete)
    app = fastapi.FastAPI()
    app.include_router(teacher_route.router)
    client = TestClient(app)

    response = client.post("/v1/teacher/completions", content=b"{}")
    assert response.status_code == 415
    response = client.post(
        "/v1/teacher/completions",
        content=b"{}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 401
    response = client.post(
        "/v1/teacher/completions",
        content=b"{}",
        headers={
            "Authorization": "Bearer capability-value",
            "Content-Type": "application/json",
            "X-Flash-Teacher-Request-Id": "request-route-000001",
        },
    )
    assert response.status_code == 200
    assert captured == {
        "capability_token": "capability-value",
        "request_id": "request-route-000001",
        "raw_body": b"{}",
    }


def test_current_nonterminal_attempt_is_checked_on_every_admission(monkeypatch):
    capability = {
        "run_id": "run-1",
        "attempt": 2,
        "teacher_alias": "glm-5.2",
        "provider": teacher_broker.FIREWORKS_PROVIDER,
        "model": "accounts/fireworks/models/glm-5p2",
        "scoring_mode": teacher_broker.FIREWORKS_SCORING_MODE,
    }
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(teacher_model="accounts/fireworks/models/glm-5p2"),
        run_id="run-1",
    )
    import flash.runner as runner

    monkeypatch.setattr(runner, "get_status", lambda _run_id: SimpleNamespace(state="running"))
    monkeypatch.setattr(runner, "_latest_reserved_attempt", lambda _run_id: 2)
    monkeypatch.setattr(runner, "_internal_spec_from_status", lambda _status: spec)
    teacher_broker._require_current_attempt(capability)

    monkeypatch.setattr(runner, "_latest_reserved_attempt", lambda _run_id: 3)
    with pytest.raises(teacher_broker.TeacherBrokerError, match="attempt_replaced"):
        teacher_broker._require_current_attempt(capability)

    monkeypatch.setattr(runner, "get_status", lambda _run_id: SimpleNamespace(state="cancelled"))
    with pytest.raises(teacher_broker.TeacherBrokerError, match="run_not_active"):
        teacher_broker._require_current_attempt(capability)


def test_only_predispatch_failure_can_retry_same_logical_request(broker_db, monkeypatch):
    _service_ready(monkeypatch)
    token = _issue()
    monkeypatch.delenv("FIREWORKS_API_KEY")
    request_id = "request-predispatch-01"

    with pytest.raises(teacher_broker.TeacherBrokerError) as missing:
        teacher_broker.complete_fireworks_request(
            capability_token=token, request_id=request_id, raw_body=_body()
        )
    assert missing.value.code == "provider_unavailable"
    assert missing.value.retryable is True

    monkeypatch.setenv("FIREWORKS_API_KEY", "control-plane-only-canary")
    monkeypatch.setattr(teacher_broker, "_fireworks_post", lambda *_args: (200, _response()))
    assert teacher_broker.complete_fireworks_request(
        capability_token=token, request_id=request_id, raw_body=_body()
    )["choices"]


@pytest.mark.parametrize("missing", ["FLASH_TEACHER_BROKER_URL", "FIREWORKS_API_KEY"])
def test_missing_broker_configuration_fails_before_allocation(monkeypatch, missing):
    import flash.providers.allocator as allocator
    from flash.runner import lifecycle

    monkeypatch.setenv("FLASH_TEACHER_BROKER_URL", "https://broker.example")
    monkeypatch.setenv("FIREWORKS_API_KEY", "control-plane-only-canary")
    monkeypatch.delenv(missing, raising=False)
    monkeypatch.setattr(
        lifecycle,
        "_pin_environment_for_run",
        lambda spec, _log, *, attempt_started: spec,
    )

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("allocation must not run without broker configuration")

    monkeypatch.setattr(allocator, "allocate", unexpected_allocation)
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-preflight",
    )

    with pytest.raises(RuntimeError):
        lifecycle._submit_seed_supervised(spec, 42, io.StringIO())


def test_failed_submission_scope_revokes_attempt_capability(broker_db, monkeypatch):
    monkeypatch.setenv("FLASH_TEACHER_BROKER_URL", "https://broker.example")
    monkeypatch.setenv("FIREWORKS_API_KEY", "control-plane-only-canary")
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-1",
    )

    captured = {}

    def fail_submission():
        with teacher_broker.teacher_attempt_transport(
            spec, attempt=2, deadline_at=time.time() + 600
        ) as runtime:
            captured["capability"] = runtime["FLASH_TEACHER_CAPABILITY"]
            raise RuntimeError("submission failed")

    with pytest.raises(RuntimeError, match="submission failed"):
        fail_submission()

    assert db.teacher_capability_binding(captured["capability"])["revoked_at"] is not None


def test_old_attempt_context_exit_does_not_revoke_new_attempt_token(broker_db, monkeypatch):
    monkeypatch.setenv("FLASH_TEACHER_BROKER_URL", "https://broker.example")
    monkeypatch.setenv("FIREWORKS_API_KEY", "control-plane-only-canary")
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-1",
    )

    with teacher_broker.teacher_attempt_transport(
        spec,
        attempt=2,
        deadline_at=time.time() + 600,
    ) as runtime:
        old_token = runtime["FLASH_TEACHER_CAPABILITY"]
        new_token = db.issue_teacher_capability(
            run_id="run-1",
            attempt=3,
            teacher_alias="glm-5.2",
            provider=teacher_broker.FIREWORKS_PROVIDER,
            model="accounts/fireworks/models/glm-5p2",
            scoring_mode=teacher_broker.FIREWORKS_SCORING_MODE,
            expires_at=time.time() + 600,
            limits=_limits(),
        )

    assert db.teacher_capability_binding(old_token)["revoked_at"] is not None
    assert db.teacher_capability_binding(new_token)["revoked_at"] is None


def test_cancellation_fences_teacher_capabilities_before_lifecycle_work(monkeypatch):
    import flash.runner as runner
    from flash.runner.deploy import cancel_run

    events = []
    status = SimpleNamespace(
        state="running",
        deployment=None,
        remote=None,
        billing_context=None,
        spec={},
    )
    monkeypatch.setattr(
        db,
        "revoke_teacher_capabilities_for_run",
        lambda run_id: events.append(("revoke", run_id)) or 1,
    )
    monkeypatch.setattr(
        runner,
        "get_status",
        lambda run_id: events.append(("status", run_id)) or status,
    )
    monkeypatch.setattr(runner, "effective_spec_from_status", lambda _status: None)
    monkeypatch.setattr(runner, "mark_deployment_undeployed", lambda _run_id: None)

    def mark_cancelled(_run_id, state, **_updates):
        status.state = state
        return status

    monkeypatch.setattr(runner, "_update", mark_cancelled)

    assert cancel_run("run-1") is status
    assert status.state == "cancelled"
    assert events[0] == ("revoke", "run-1")
    assert [event for event in events if event[0] == "revoke"] == [
        ("revoke", "run-1"),
        ("revoke", "run-1"),
    ]


def test_runpod_lambda_and_vast_payloads_never_expose_provider_credentials(monkeypatch):
    from flash.providers._worker import build_worker_env
    from flash.providers.base import PollResult
    from flash.providers.lambdalabs.jobs.builders import build_payload as build_lambda_payload
    from flash.providers.runpod import jobs as runpod_jobs
    from flash.providers.vast.jobs.builders import build_payload as build_vast_payload

    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        environment=EnvironmentSpec(id="org/env"),
        train=TrainSpec(max_examples=8, hf_repo="owner/runs"),
        run_id="run-payload",
    )
    runtime = {
        "FIREWORKS_API_KEY": "fireworks-worker-canary",
        "PARASAIL_API_KEY": "parasail-worker-canary",
        "FLASH_TEACHER_BROKER_URL": "https://broker.example",
        "FLASH_TEACHER_CAPABILITY": "capability-worker-canary",
    }
    monkeypatch.setenv("FIREWORKS_API_KEY", "fireworks-control-plane-canary")
    monkeypatch.setenv("PARASAIL_API_KEY", "parasail-control-plane-canary")
    deadline = time.time() + 3600

    lambda_payload = build_lambda_payload(
        spec, 42, 0, runtime_secrets=runtime, deadline_at=deadline
    )
    vast_payload = build_vast_payload(spec, 42, 0, runtime_secrets=runtime, deadline_at=deadline)
    captured = {}
    monkeypatch.setattr(
        runpod_jobs,
        "deploy_train_endpoint",
        lambda *_args, **_kwargs: ("endpoint-id", "endpoint-name", "key-fingerprint"),
    )

    def submit_job(_endpoint, payload, **_kwargs):
        captured["payload"] = payload
        return "job-id"

    monkeypatch.setattr(runpod_jobs.runpod_api, "submit_job", submit_job)
    monkeypatch.setattr(
        runpod_jobs,
        "poll_job",
        lambda *_args, **_kwargs: PollResult(True, metrics={}),
    )
    runpod_jobs.submit_run(
        spec,
        42,
        attempt=0,
        runtime_secrets=runtime,
        deadline_at=deadline,
    )

    serialized = json.dumps(
        {"runpod": captured["payload"], "lambda": lambda_payload, "vast": vast_payload}
    )
    assert "fireworks-worker-canary" not in serialized
    assert "parasail-worker-canary" not in serialized
    assert "fireworks-control-plane-canary" not in serialized
    assert "parasail-control-plane-canary" not in serialized
    assert "FIREWORKS_API_KEY" not in serialized
    assert "PARASAIL_API_KEY" not in serialized
    assert serialized.count("capability-worker-canary") == 3

    invalid_spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        environment=EnvironmentSpec(
            id="org/env", secrets=("FIREWORKS_API_KEY", "PARASAIL_API_KEY")
        ),
        train=TrainSpec(max_examples=8, hf_repo="owner/runs"),
        run_id="run-invalid-payload",
    )
    with pytest.raises(ValueError, match="managed teacher credential names"):
        build_worker_env(invalid_spec, 42, runtime_secrets=runtime)


def test_capability_policy_is_run_bounded_and_rejects_excessive_shapes(monkeypatch):
    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1, batch_size=8, group_size=1),
        run_id="run-1",
    )
    limits = teacher_broker.capability_limits_for_spec(spec)
    assert limits["max_score_items"] == 64
    assert limits["max_requests"] == 64
    assert limits["max_upstream_attempts"] == 1

    monkeypatch.setattr(teacher_broker, "MAX_TOTAL_SCORE_ITEMS", 4)
    with pytest.raises(ValueError, match="score-item limit"):
        teacher_broker.capability_limits_for_spec(spec)

"""Focused managed-teacher broker security and lifecycle contracts."""

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import flash.runner.lifecycle.attempts as runner_attempts
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.supervise.lifecycle as runner_lifecycle
from flash.core.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec
from flash.server.domain.teacher import broker as teacher_broker
from flash.server.platform import db
from tests._helpers.source_snapshot import valid_source_snapshot

_SOURCE_SNAPSHOT = valid_source_snapshot()


@pytest.fixture(autouse=True)
def stub_worker_teacher_tokenizer(monkeypatch):
    from flash.engine.worker.teacher import client as worker_teacher
    from flash.engine.worker.teacher.encoding import EncodedTeacherToken

    class Tokenizer:
        def encode(self, text):
            return [EncodedTeacherToken(7, 0, len(text))]

    monkeypatch.setattr(worker_teacher, "load_teacher_tokenizer", lambda _model: Tokenizer())


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
        provider=teacher_broker.PARASAIL_PROVIDER,
        model="parasail-glm-52",
        scoring_mode=teacher_broker.PARASAIL_SCORING_MODE,
        expires_at=expires_at if expires_at is not None else current + 600,
        limits=limits or _limits(),
        now=current,
    )


def _body(*, prompt="questionanswer", model="parasail-glm-52", **extra):
    value = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "echo": True,
        "logprobs": 1,
        "prompt_logprobs": 1,
        "return_token_ids": True,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
    }
    value.update(extra)
    return json.dumps(value, separators=(",", ":")).encode()


def _response(*, token="questionanswer"):
    return json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "prompt_token_ids": [7],
                    "token_ids": [8],
                    "prompt_logprobs": [{"7": {"logprob": -0.1, "decoded_token": token}}],
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


def _chat_body(*, model="parasail-glm-52", messages=None, **extra):
    value = {
        "model": model,
        "messages": messages
        or [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                    },
                    {"type": "text", "text": "question"},
                ],
            },
            {"role": "assistant", "content": "answer"},
        ],
        "max_tokens": 1,
        "temperature": 0,
        "seed": 0,
        "prompt_logprobs": 1,
        "return_token_ids": True,
    }
    value.update(extra)
    return json.dumps(value, separators=(",", ":")).encode()


def _chat_response():
    return json.dumps(
        {
            "choices": [{"index": 0, "token_ids": [8], "message": {"role": "assistant"}}],
            "prompt_token_ids": [151655, 7],
            "prompt_logprobs": [None, {"7": {"logprob": -0.1}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
    ).encode()


def _service_ready(monkeypatch):
    monkeypatch.setattr(teacher_broker, "_require_current_attempt", lambda _capability: None)
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")


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
        provider=teacher_broker.PARASAIL_PROVIDER,
        model="parasail-glm-52",
        scoring_mode=teacher_broker.PARASAIL_SCORING_MODE,
        expires_at=time.time() + 600,
        limits=_limits(),
    )

    assert db.teacher_capability_binding(first)["revoked_at"] is not None
    assert db.teacher_capability_binding(second)["attempt"] == 3


def test_exact_24_hour_capability_deadline_is_accepted(broker_db, monkeypatch):
    monkeypatch.setenv("FLASH_PUBLIC_URL", "https://broker.example")
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")
    issued_at = 1_000.0
    deadline_at = issued_at + teacher_broker.MAX_CAPABILITY_LIFETIME_S
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
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
    import flash.providers.core.allocator as allocator
    from flash.runner.supervise import lifecycle

    monkeypatch.setenv("FLASH_PUBLIC_URL", "https://broker.example")
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("allocation must not run for an opd wall over 24 hours")

    monkeypatch.setattr(allocator, "allocate", unexpected_allocation)
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        gpu=GpuSpec(max_wall_seconds=48 * 60 * 60),
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-over-capability-wall",
    )

    with pytest.raises(ValueError, match="24 hours or less"):
        lifecycle._run_attempts_supervised(spec, io.StringIO())


def test_broker_accepts_every_catalog_teacher(monkeypatch):
    # the broker keeps its own alias set, so a catalog addition that is not mirrored here
    # would be rejected at submit despite resolving fine.
    from flash.engine.plan.recipe import TEACHER_MODELS

    monkeypatch.setenv("FLASH_PUBLIC_URL", "https://broker.example")
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")
    for alias in TEACHER_MODELS:
        spec = JobSpec(
            model="Qwen/Qwen3.5-9B",
            algorithm="opd",
            train=TrainSpec(max_examples=8, max_steps=1, teacher_model=alias),
            run_id=f"run-teacher-{alias}",
        )
        assert teacher_broker.require_teacher_broker_configuration(spec) == "https://broker.example"


def test_deadline_contract_over_24_hours_is_rejected(monkeypatch):
    monkeypatch.setenv("FLASH_PUBLIC_URL", "https://broker.example")
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
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
    assert db.teacher_capability_binding(token)["token_count"] == 128
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
    assert db.teacher_capability_binding(token)["token_count"] == 64
    db.mark_teacher_request_started(second["capability"]["id"], "request-tokens-000002")
    db.complete_teacher_request(
        second["capability"]["id"],
        "request-tokens-000002",
        state="succeeded",
        input_tokens=10,
        response_body=b"{}",
    )
    assert db.teacher_capability_binding(token)["token_count"] == 10
    db.reserve_teacher_request(
        token=token,
        request_id="request-tokens-000003",
        request_fingerprint="c" * 64,
        request_bytes=10,
        score_items=1,
        expected_run_id="run-1",
        expected_attempt=2,
    )
    assert db.teacher_capability_binding(token)["token_count"] == 74


def test_stale_started_request_becomes_terminal_after_broker_recovery(broker_db):
    token = _issue(limits=_limits(max_upstream_attempts=2))
    reservation = db.reserve_teacher_request(
        token=token,
        request_id="request-stale-000001",
        request_fingerprint="a" * 64,
        request_bytes=10,
        score_items=1,
        expected_run_id="run-1",
        expected_attempt=2,
    )
    capability_id = reservation["capability"]["id"]
    db.mark_teacher_request_started(capability_id, "request-stale-000001")
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
    assert db.teacher_capability_binding(token)["token_count"] == 128
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
    assert db.recover_teacher_request_ledger() == {"retryable": 0, "outcome_unknown": 0}
    connection = sqlite3.connect(broker_db)
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT state, upstream_attempt_count FROM teacher_score_requests "
        "WHERE request_id = 'request-stale-000001'"
    ).fetchone()
    in_flight = connection.execute("SELECT in_flight FROM teacher_capabilities").fetchone()[0]
    connection.close()
    assert (row["state"], row["upstream_attempt_count"]) == ("outcome_unknown", 1)
    assert in_flight == 0
    assert db.teacher_capability_binding(token)["token_count"] == 128


def test_closed_parasail_contract_rejects_extra_fields_model_changes_and_batches(broker_db):
    token = _issue()
    capability = db.teacher_capability_binding(token)

    with pytest.raises(teacher_broker.TeacherBrokerError, match="extra_request_fields"):
        teacher_broker.validate_completion_request(
            teacher_broker.parse_strict_json(_body(destination="https://attacker.invalid")),
            capability,
        )
    with pytest.raises(teacher_broker.TeacherBrokerError, match="model_scope_mismatch"):
        teacher_broker.validate_completion_request(
            teacher_broker.parse_strict_json(_body(model="parasail-other")),
            capability,
        )
    with pytest.raises(teacher_broker.TeacherBrokerError, match="extra_request_fields"):
        teacher_broker.validate_completion_request(
            teacher_broker.parse_strict_json(_body(images=["data:image/png;base64,YQ=="])),
            capability,
        )
    with pytest.raises(teacher_broker.TeacherBrokerError, match="unsupported_scoring_parameters"):
        teacher_broker.validate_completion_request(
            teacher_broker.parse_strict_json(_body(prompt=["a", "b"])),
            capability,
        )
    request = teacher_broker.validate_completion_request(
        teacher_broker.parse_strict_json(_body()),
        capability,
    )
    assert request.score_items == 1


def test_chat_scoring_contract_accepts_exact_shape_and_rejects_unknown_input(broker_db):
    token = _issue()
    capability = db.teacher_capability_binding(token)

    request = teacher_broker.validate_chat_completion_request(
        teacher_broker.parse_strict_json(_chat_body()),
        capability,
    )
    assert request.score_items == 1

    with pytest.raises(teacher_broker.TeacherBrokerError, match="extra_request_fields"):
        teacher_broker.validate_chat_completion_request(
            teacher_broker.parse_strict_json(_chat_body(echo=True)),
            capability,
        )
    with pytest.raises(
        teacher_broker.TeacherBrokerError,
        match="unknown_chat_content_block_type",
    ):
        teacher_broker.validate_chat_completion_request(
            teacher_broker.parse_strict_json(
                _chat_body(
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "audio", "audio": "data:audio/wav;base64,YQ=="}],
                        }
                    ]
                )
            ),
            capability,
        )


def test_chat_scoring_contract_enforces_capability_request_bytes(broker_db):
    token = _issue()
    capability = dict(db.teacher_capability_binding(token))
    capability["max_request_bytes"] = 64

    with pytest.raises(teacher_broker.TeacherBrokerError, match="request_too_large"):
        teacher_broker.validate_chat_completion_request(
            teacher_broker.parse_strict_json(_chat_body()),
            capability,
        )


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


def test_chat_request_forwards_to_parasail_chat_route_and_preserves_top_level_scores(
    broker_db, monkeypatch
):
    _service_ready(monkeypatch)
    token = _issue()
    captured = {}

    def dispatch(body, api_key, timeout):
        captured.update(
            {
                "body": json.loads(body),
                "api_key": api_key,
                "timeout": timeout,
            }
        )
        return 200, _chat_response()

    monkeypatch.setattr(teacher_broker, "_provider_chat_post", dispatch)

    response = teacher_broker.complete_teacher_chat_request(
        capability_token=token,
        request_id="request-chat-00000001",
        raw_body=_chat_body(),
    )

    assert captured["body"] == json.loads(_chat_body())
    assert captured["api_key"] == "control-plane-only-canary"
    assert captured["timeout"] > 0
    assert response["prompt_token_ids"] == [151655, 7]
    assert response["prompt_logprobs"][1]["7"]["logprob"] == -0.1
    assert "prompt_token_ids" not in response["choices"][0]


def test_provider_chat_post_uses_parasail_chat_completions_path(monkeypatch):
    captured = {}

    def post(path, body, api_key, timeout):
        captured.update({"path": path, "body": body, "api_key": api_key, "timeout": timeout})
        return 200, b"{}"

    monkeypatch.setattr(teacher_broker, "_provider_post_path", post)

    assert teacher_broker._provider_chat_post(b"body", "key", 3.0) == (200, b"{}")
    assert captured == {
        "path": "/v1/chat/completions",
        "body": b"body",
        "api_key": "key",
        "timeout": 3.0,
    }


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

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)

    def call():
        return teacher_broker.complete_teacher_request(
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
        assert error.value.retryable is True
        release.set()
        assert first.result(timeout=5)["choices"]

    assert len(dispatches) == 1


def test_success_replays_after_response_loss_and_changed_body_stays_rejected(
    broker_db, monkeypatch
):
    _service_ready(monkeypatch)
    token = _issue()
    dispatches = []
    monkeypatch.setattr(
        teacher_broker,
        "_provider_post",
        lambda *_args: (dispatches.append(1) or 200, _response()),
    )
    request_id = "request-replay-000001"

    first = teacher_broker.complete_teacher_request(
        capability_token=token, request_id=request_id, raw_body=_body()
    )
    replay = teacher_broker.complete_teacher_request(
        capability_token=token, request_id=request_id, raw_body=_body()
    )
    with pytest.raises(teacher_broker.TeacherBrokerError) as changed:
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id=request_id,
            raw_body=_body(prompt="changed"),
        )
    assert replay == first
    assert changed.value.code == "request_body_changed"
    assert len(dispatches) == 1

    connection = sqlite3.connect(broker_db)
    stored = connection.execute(
        "SELECT response_body FROM teacher_score_requests WHERE request_id = ?",
        (request_id,),
    ).fetchone()[0]
    connection.close()
    assert json.loads(stored) == first
    assert len(stored) <= teacher_broker.MAX_RESPONSE_BODY_BYTES
    assert b"control-plane-only-canary" not in stored
    assert token.encode() not in stored


def test_provider_error_body_is_suppressed_from_response_and_sqlite(broker_db, monkeypatch):
    _service_ready(monkeypatch)
    token = _issue()
    private_canary = "private-provider-error-body-canary"
    dispatches = []

    def reject(*_args):
        dispatches.append(1)
        return 400, private_canary.encode()

    monkeypatch.setattr(teacher_broker, "_provider_post", reject)

    with pytest.raises(teacher_broker.TeacherBrokerError) as error:
        teacher_broker.complete_teacher_request(
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
            "provider_status": 400,
        }
    }
    assert error.value.retryable is False
    with pytest.raises(teacher_broker.TeacherBrokerError) as duplicate:
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id="request-private-00001",
            raw_body=_body(),
        )
    assert duplicate.value.code == "provider_rejected"
    assert duplicate.value.retryable is False
    assert duplicate.value.payload()["error"]["provider_status"] == 400
    assert len(dispatches) == 1
    assert private_canary not in dump
    assert "control-plane-only-canary" not in dump
    assert token not in dump


@pytest.mark.parametrize("provider_status", [True, "invalid", 0, -1, 600, 700])
def test_malformed_stored_provider_status_is_omitted_on_replay(
    broker_db, monkeypatch, provider_status
):
    _service_ready(monkeypatch)
    token = _issue()
    private_canary = "private-provider-error-body-canary"
    dispatches = []

    def reject(*_args):
        dispatches.append(1)
        return 403, private_canary.encode()

    monkeypatch.setattr(teacher_broker, "_provider_post", reject)
    request_id = "request-malformed-status-01"

    with pytest.raises(teacher_broker.TeacherBrokerError):
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id=request_id,
            raw_body=_body(),
        )

    connection = sqlite3.connect(broker_db)
    connection.execute(
        "UPDATE teacher_score_requests SET provider_status = ? WHERE request_id = ?",
        (provider_status, request_id),
    )
    connection.commit()
    connection.close()

    with pytest.raises(teacher_broker.TeacherBrokerError) as replay:
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id=request_id,
            raw_body=_body(),
        )

    payload = replay.value.payload()
    assert replay.value.retryable is False
    assert replay.value.provider_status is None
    assert payload["error"]["code"] == "provider_rejected"
    assert payload["error"]["classification"] == "permanent"
    assert "provider_status" not in payload["error"]
    assert private_canary not in json.dumps(payload)
    assert len(dispatches) == 1


def test_provider_429_is_readmitted_and_dispatches_again(broker_db, monkeypatch):
    """a conventional 429 proves rejection before execution and is safe to redispatch."""
    status = 429
    _service_ready(monkeypatch)
    token = _issue(limits=_limits(max_upstream_attempts=teacher_broker.MAX_UPSTREAM_ATTEMPTS))
    outcomes = [(status, b"upstream"), (200, _response())]
    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return outcomes[len(dispatches) - 1]

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)

    with pytest.raises(teacher_broker.TeacherBrokerError) as error:
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id="request-shed-00000001",
            raw_body=_body(),
        )
    assert error.value.code == "provider_rejected"
    assert error.value.retryable is True
    assert error.value.payload()["error"]["classification"] == "transient"

    replayed = teacher_broker.complete_teacher_request(
        capability_token=token,
        request_id="request-shed-00000001",
        raw_body=_body(),
    )

    assert replayed["choices"]
    assert len(dispatches) == 2
    connection = sqlite3.connect(broker_db)
    row = connection.execute(
        "SELECT state, upstream_attempt_count, input_tokens, output_tokens "
        "FROM teacher_score_requests WHERE request_id = 'request-shed-00000001'"
    ).fetchone()
    token_count = connection.execute("SELECT token_count FROM teacher_capabilities").fetchone()[0]
    connection.close()
    assert row == ("succeeded", 2, 1, 1)
    # the rejected 429 did no provider work, so the single accepted call is the only billable one.
    assert token_count == 2


@pytest.mark.parametrize("status", [408, 500, 502, 503])
def test_ambiguous_provider_status_is_terminal_without_redispatch(broker_db, monkeypatch, status):
    _service_ready(monkeypatch)
    token = _issue(limits=_limits(max_upstream_attempts=teacher_broker.MAX_UPSTREAM_ATTEMPTS))
    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return status, b"upstream"

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)
    request_id = "request-ambiguous-0001"

    with pytest.raises(teacher_broker.TeacherBrokerError) as first:
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id=request_id,
            raw_body=_body(),
        )
    assert first.value.code == "outcome_unknown"
    assert first.value.retryable is False

    with pytest.raises(teacher_broker.TeacherBrokerError) as replay:
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id=request_id,
            raw_body=_body(),
        )
    assert replay.value.code == "outcome_unknown"
    assert replay.value.retryable is False
    assert len(dispatches) == 1

    connection = sqlite3.connect(broker_db)
    row = connection.execute(
        "SELECT state, upstream_attempt_count, provider_status, error_class "
        "FROM teacher_score_requests WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    connection.close()
    assert row == ("outcome_unknown", 1, status, "permanent")


def test_validated_response_with_failed_ledger_settlement_never_redispatches(
    broker_db, monkeypatch
):
    _service_ready(monkeypatch)
    token = _issue(limits=_limits(max_upstream_attempts=teacher_broker.MAX_UPSTREAM_ATTEMPTS))
    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return 200, _response()

    real_complete = db.complete_teacher_request

    def fail_success_completion(capability_id, request_id, **kwargs):
        if kwargs["state"] == "succeeded":
            raise sqlite3.OperationalError("ledger write failed after provider success")
        return real_complete(capability_id, request_id, **kwargs)

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)
    monkeypatch.setattr(db, "complete_teacher_request", fail_success_completion)
    request_id = "request-settlement-0001"

    with pytest.raises(teacher_broker.TeacherBrokerError) as first:
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id=request_id,
            raw_body=_body(),
        )
    assert first.value.code == "outcome_unknown"
    assert first.value.retryable is True

    with pytest.raises(teacher_broker.TeacherBrokerError) as replay:
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id=request_id,
            raw_body=_body(),
        )
    assert replay.value.code == "outcome_unknown"
    assert replay.value.retryable is False
    assert len(dispatches) == 1

    connection = sqlite3.connect(broker_db)
    row = connection.execute(
        "SELECT state, upstream_attempt_count, error_class FROM teacher_score_requests "
        "WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    connection.close()
    assert row == ("outcome_unknown", 1, "ledger_completion_failed")


def test_worker_retries_committed_success_settlement_exception_without_redispatch(
    broker_db, monkeypatch
):
    import urllib.error

    from flash.engine.worker.teacher import client as worker_teacher

    _service_ready(monkeypatch)
    token = _issue(limits=_limits(max_upstream_attempts=teacher_broker.MAX_UPSTREAM_ATTEMPTS))
    dispatches = []
    request_ids = []

    def dispatch(*_args):
        dispatches.append(1)
        return 200, _response()

    real_complete = db.complete_teacher_request
    raised_after_commit = False

    def commit_then_raise(capability_id, request_id, **kwargs):
        nonlocal raised_after_commit
        result = real_complete(capability_id, request_id, **kwargs)
        if kwargs["state"] == "succeeded" and not raised_after_commit:
            raised_after_commit = True
            raise sqlite3.OperationalError("ledger response lost after commit")
        return result

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.body

    def urlopen(_transport, request, timeout=None):
        headers = {name.lower(): value for name, value in request.header_items()}
        request_id = headers["x-flash-teacher-request-id"]
        request_ids.append(request_id)
        capability_token = headers["authorization"].removeprefix("Bearer ")
        try:
            result = teacher_broker.complete_teacher_request(
                capability_token=capability_token,
                request_id=request_id,
                raw_body=request.data,
            )
        except teacher_broker.TeacherBrokerError as error:
            raise urllib.error.HTTPError(
                request.full_url,
                error.status_code,
                error.code,
                {},
                io.BytesIO(json.dumps(error.payload()).encode()),
            ) from error
        return Response(json.dumps(result).encode())

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)
    monkeypatch.setattr(db, "complete_teacher_request", commit_then_raise)
    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _seconds: None)
    client = worker_teacher.TeacherClient(token, "https://broker.example", "parasail-glm-52")

    assert client.score("question", "answer")
    assert raised_after_commit is True
    assert len(request_ids) == 2
    assert len(set(request_ids)) == 1
    assert len(dispatches) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_genuine_4xx_provider_rejects_stay_terminal(broker_db, monkeypatch, status):
    """a non-429 4xx means the request or credential itself is rejected; retrying cannot help."""
    _service_ready(monkeypatch)
    token = _issue(limits=_limits(max_upstream_attempts=teacher_broker.MAX_UPSTREAM_ATTEMPTS))
    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return status, b"upstream"

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)

    with pytest.raises(teacher_broker.TeacherBrokerError) as error:
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id="request-perm-00000001",
            raw_body=_body(),
        )
    assert error.value.code == "provider_rejected"
    assert error.value.retryable is False
    assert error.value.payload()["error"]["classification"] == "permanent"

    with pytest.raises(teacher_broker.TeacherBrokerError) as duplicate:
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id="request-perm-00000001",
            raw_body=_body(),
        )
    assert duplicate.value.code == "provider_rejected"
    assert duplicate.value.retryable is False
    assert len(dispatches) == 1


def test_upstream_attempts_are_bounded_even_for_transient_rejects(broker_db, monkeypatch):
    _service_ready(monkeypatch)
    token = _issue(limits=_limits(max_upstream_attempts=2))
    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return 429, b"rate limited"

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)

    for _attempt in range(2):
        with pytest.raises(teacher_broker.TeacherBrokerError) as error:
            teacher_broker.complete_teacher_request(
                capability_token=token,
                request_id="request-bound-0000001",
                raw_body=_body(),
            )
        assert error.value.code == "provider_rejected"
        assert error.value.retryable is True

    with pytest.raises(teacher_broker.TeacherBrokerError) as exhausted:
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id="request-bound-0000001",
            raw_body=_body(),
        )
    assert exhausted.value.code == "upstream_attempt_quota_exhausted"
    assert exhausted.value.retryable is False
    assert len(dispatches) == 2

    # the refusal happens at readmission, so the row keeps its terminal state and its held
    # reservation instead of bouncing to 'retryable' and staying readmissible forever.
    with pytest.raises(teacher_broker.TeacherBrokerError) as repeated:
        teacher_broker.complete_teacher_request(
            capability_token=token,
            request_id="request-bound-0000001",
            raw_body=_body(),
        )
    assert repeated.value.code == "upstream_attempt_quota_exhausted"
    assert len(dispatches) == 2
    connection = sqlite3.connect(broker_db)
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT state, error_class, upstream_attempt_count FROM teacher_score_requests "
        "WHERE request_id = 'request-bound-0000001'"
    ).fetchone()
    in_flight = connection.execute("SELECT in_flight FROM teacher_capabilities").fetchone()[0]
    connection.close()
    assert (row["state"], row["error_class"]) == ("provider_rejected", "transient")
    assert row["upstream_attempt_count"] == 2
    assert in_flight == 0


def test_worker_default_timeout_exceeds_broker_provider_ceiling():
    from flash.engine.worker.teacher import client as worker_teacher

    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "parasail-glm-52"
    )

    assert worker_teacher._BROKER_PROVIDER_TIMEOUT_CEILING_S == 90.0
    assert client.timeout == worker_teacher._DEFAULT_TEACHER_TIMEOUT_S
    assert client.timeout > worker_teacher._BROKER_PROVIDER_TIMEOUT_CEILING_S


def test_worker_reuses_one_logical_request_id_across_transport_retries(monkeypatch):
    import urllib.error

    from flash.engine.worker.teacher import client as worker_teacher

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
        "capability-value", "https://broker.example", "parasail-glm-52"
    )

    assert client.score("question", "answer")
    assert len(request_ids) == 2
    assert len(set(request_ids)) == 1


@pytest.mark.parametrize("body", [b"", b"<html>bad gateway</html>", b'{"error":'])
def test_worker_keeps_unstructured_502_terminal(monkeypatch, body):
    import urllib.error

    from flash.engine.worker.teacher import client as worker_teacher

    request_ids = []

    def urlopen(_transport, request, timeout=None):
        request_ids.append(dict(request.header_items())["X-flash-teacher-request-id"])
        raise urllib.error.HTTPError(request.full_url, 502, "bad gateway", {}, io.BytesIO(body))

    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _seconds: None)
    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "parasail-glm-52"
    )

    with pytest.raises(worker_teacher.TeacherError) as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert len(request_ids) == 1


@pytest.mark.parametrize("status", [429])
def test_worker_retries_unstructured_proven_pre_dispatch_http_status(monkeypatch, status):
    import urllib.error

    from flash.engine.worker.teacher import client as worker_teacher

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
            raise urllib.error.HTTPError(
                request.full_url, status, "broker failure", {}, io.BytesIO(b"")
            )
        return Response()

    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _seconds: None)
    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "parasail-glm-52"
    )

    assert client.score("question", "answer")
    assert len(request_ids) == 2
    assert len(set(request_ids)) == 1


@pytest.mark.parametrize("status", [408, 409, 500, 502, 599])
def test_worker_keeps_unstructured_post_dispatch_ambiguity_terminal(monkeypatch, status):
    import urllib.error

    from flash.engine.worker.teacher import client as worker_teacher

    request_ids = []

    def urlopen(_transport, request, timeout=None):
        request_ids.append(dict(request.header_items())["X-flash-teacher-request-id"])
        raise urllib.error.HTTPError(
            request.full_url, status, "broker failure", {}, io.BytesIO(b"")
        )

    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _seconds: None)
    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "parasail-glm-52"
    )

    with pytest.raises(worker_teacher.TeacherError) as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert len(request_ids) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 413, 415, 422])
def test_worker_keeps_unstructured_permanent_4xx_terminal(monkeypatch, status):
    import urllib.error

    from flash.engine.worker.teacher import client as worker_teacher

    request_ids = []

    def urlopen(_transport, request, timeout=None):
        request_ids.append(dict(request.header_items())["X-flash-teacher-request-id"])
        raise urllib.error.HTTPError(
            request.full_url, status, "broker rejection", {}, io.BytesIO(b"<html>rejected</html>")
        )

    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _seconds: None)
    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "parasail-glm-52"
    )

    with pytest.raises(worker_teacher.TeacherError) as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert "broker_http_error (permanent)" in str(error.value)
    assert len(request_ids) == 1


def test_worker_retries_body_ingress_timeout_with_same_request_id(monkeypatch):
    import urllib.error

    from flash.engine.worker.teacher import client as worker_teacher

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
                        "code": "body_ingress_timeout",
                        "classification": "transient",
                    }
                }
            ).encode()
            raise urllib.error.HTTPError(request.full_url, 408, "timeout", {}, io.BytesIO(body))
        return Response()

    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _seconds: None)
    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "parasail-glm-52"
    )

    assert client.score("question", "answer")
    assert len(request_ids) == 2
    assert len(set(request_ids)) == 1


def test_worker_retries_request_in_progress_with_same_request_id(monkeypatch):
    import urllib.error

    from flash.engine.worker.teacher import client as worker_teacher

    request_ids = []
    sleeps = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return _response()

    def urlopen(_transport, request, timeout=None):
        request_ids.append(dict(request.header_items())["X-flash-teacher-request-id"])
        if len(request_ids) < 3:
            body = json.dumps(
                {
                    "error": {
                        "code": "request_in_progress",
                        "classification": "transient",
                    }
                }
            ).encode()
            raise urllib.error.HTTPError(request.full_url, 409, "conflict", {}, io.BytesIO(body))
        return Response()

    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    monkeypatch.setattr(worker_teacher.time, "sleep", sleeps.append)
    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "parasail-glm-52"
    )

    assert client.score("question", "answer")
    assert request_ids
    assert len(set(request_ids)) == 1
    assert len(request_ids) == 3
    assert sleeps == [2.0, 4.0]


@pytest.mark.parametrize("classification", ["permanent", None, "unknown", {"bad": "shape"}])
def test_worker_fails_closed_on_nontransient_broker_classification(monkeypatch, classification):
    import urllib.error

    from flash.engine.worker.teacher import client as worker_teacher

    request_ids = []

    def urlopen(_transport, request, timeout=None):
        request_ids.append(dict(request.header_items())["X-flash-teacher-request-id"])
        error_payload = {"code": "provider_rejected"}
        if classification is not None:
            error_payload["classification"] = classification
        body = json.dumps({"error": error_payload}).encode()
        raise urllib.error.HTTPError(request.full_url, 502, "rejected", {}, io.BytesIO(body))

    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _seconds: None)
    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "parasail-glm-52"
    )

    with pytest.raises(worker_teacher.TeacherError) as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert error.value.provider_status is None
    assert "provider_rejected (permanent)" in str(error.value)
    assert "provider_status=" not in str(error.value)
    assert len(request_ids) == 1


def test_opd_batch_error_preserves_provider_status():
    from flash.engine.worker.teacher.client import TeacherError
    from flash.engine.worker.train.opd.bridging.batching import _teacher_batch_error

    wrapped = _teacher_batch_error(
        TeacherError("rate limited", permanent=False, provider_status=429)
    )

    assert isinstance(wrapped, TeacherError)
    assert wrapped.permanent is False
    assert wrapped.provider_status == 429


def test_worker_renders_provider_status_without_leaking_broker_payload(monkeypatch):
    import urllib.error

    from flash.engine.worker.teacher import client as worker_teacher

    private_canary = "private-provider-error-body-canary"

    def urlopen(_transport, request, timeout=None):
        body = json.dumps(
            {
                "error": {
                    "code": "provider_rejected",
                    "classification": "permanent",
                    "provider_status": 403,
                    "provider_text": private_canary,
                }
            }
        ).encode()
        raise urllib.error.HTTPError(request.full_url, 502, "rejected", {}, io.BytesIO(body))

    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "parasail-glm-52"
    )

    with pytest.raises(worker_teacher.TeacherError) as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert error.value.provider_status == 403
    assert "provider_rejected provider_status=403 (permanent)" in str(error.value)
    assert private_canary not in str(error.value)


@pytest.mark.parametrize("provider_status", [True, "403", 0, -1, 600, 700])
def test_worker_omits_malformed_provider_status(monkeypatch, provider_status):
    import urllib.error

    from flash.engine.worker.teacher import client as worker_teacher

    body = json.dumps(
        {
            "error": {
                "code": "provider_rejected",
                "classification": "permanent",
                "provider_status": provider_status,
            }
        }
    ).encode()

    def urlopen(_transport, request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 502, "rejected", {}, io.BytesIO(body))

    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    client = worker_teacher.TeacherClient(
        "capability-value", "https://broker.example", "parasail-glm-52"
    )

    with pytest.raises(worker_teacher.TeacherError) as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert error.value.provider_status is None
    assert "provider_rejected (permanent)" in str(error.value)
    assert "provider_status=" not in str(error.value)


def test_operation_specific_route_requires_bearer_json_and_request_id(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from flash.server.routes import teacher as teacher_route

    captured = {}

    def complete(**kwargs):
        captured.update(kwargs)
        return {"choices": []}

    monkeypatch.setattr(
        teacher_route,
        "authenticate_teacher_capability",
        lambda **kwargs: (
            kwargs["request_id"],
            kwargs["capability_token"],
            {"expires_at": time.time() + 60},
        ),
    )
    monkeypatch.setattr(teacher_route, "complete_teacher_request", complete)
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
        "raw_body": bytearray(b"{}"),
    }
    assert isinstance(captured["raw_body"], bytearray)


@pytest.mark.parametrize(
    ("provider_status", "expected_status"),
    [
        (403, 403),
        (None, None),
        (True, None),
        ("403", None),
        (0, None),
        (-1, None),
        (600, None),
        (700, None),
    ],
)
def test_broker_error_serializes_only_valid_provider_status(provider_status, expected_status):
    error = teacher_broker.TeacherBrokerError(
        "provider_rejected",
        status_code=502,
        request_id="request-status-boundary-01",
        provider_status=provider_status,
    )

    payload = error.payload()["error"]
    if expected_status is None:
        assert error.provider_status is None
        assert "provider_status" not in payload
    else:
        assert error.provider_status == expected_status
        assert payload["provider_status"] == expected_status


def test_teacher_route_propagates_provider_status_through_dispatch_and_replay(
    broker_db, monkeypatch
):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from flash.server.routes import teacher as teacher_route

    _service_ready(monkeypatch)
    token = _issue()
    request_id = "request-route-replay-001"
    private_canary = "private-provider-error-body-canary"
    dispatches = []

    def reject(*_args):
        dispatches.append(1)
        return 403, private_canary.encode()

    monkeypatch.setattr(teacher_broker, "_provider_post", reject)
    app = fastapi.FastAPI()
    app.include_router(teacher_route.router)
    client = TestClient(app)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Flash-Teacher-Request-Id": request_id,
    }
    expected = {
        "error": {
            "code": "provider_rejected",
            "classification": "permanent",
            "request_id": request_id,
            "provider_status": 403,
        }
    }

    first = client.post("/v1/teacher/completions", content=_body(), headers=headers)
    replay = client.post("/v1/teacher/completions", content=_body(), headers=headers)

    assert first.status_code == 502
    assert replay.status_code == 409
    assert first.json() == expected
    assert replay.json() == expected
    assert private_canary not in first.text
    assert private_canary not in replay.text
    assert len(dispatches) == 1
    connection = sqlite3.connect(broker_db)
    stored = connection.execute(
        "SELECT state, provider_status, error_class FROM teacher_score_requests "
        "WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    connection.close()
    assert stored == ("provider_rejected", 403, "permanent")


def test_chat_route_mirrors_completion_route_auth_and_request_binding(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from flash.server.routes import teacher as teacher_route

    captured = {}

    monkeypatch.setattr(
        teacher_route,
        "authenticate_teacher_capability",
        lambda **kwargs: (
            kwargs["request_id"],
            kwargs["capability_token"],
            {"expires_at": time.time() + 60},
        ),
    )
    monkeypatch.setattr(
        teacher_route,
        "complete_teacher_chat_request",
        lambda **kwargs: captured.update(kwargs) or {"choices": []},
    )
    app = fastapi.FastAPI()
    app.include_router(teacher_route.router)
    client = TestClient(app)

    response = client.post(
        "/v1/teacher/chat_completions",
        content=b"{}",
        headers={
            "Authorization": "Bearer capability-value",
            "Content-Type": "application/json",
            "X-Flash-Teacher-Request-Id": "request-chat-route-0001",
        },
    )

    assert response.status_code == 200
    assert captured == {
        "capability_token": "capability-value",
        "request_id": "request-chat-route-0001",
        "raw_body": bytearray(b"{}"),
    }


def test_bounded_body_accepts_exact_limit_without_copy_and_rejects_next_chunk(monkeypatch):
    from flash.server.routes import teacher as teacher_route

    monkeypatch.setattr(teacher_route, "MAX_REQUEST_BODY_BYTES", 4)

    class Request:
        def __init__(self, chunks):
            self.headers = {}
            self._chunks = chunks

        async def stream(self):
            for chunk in self._chunks:
                yield chunk

    body = asyncio.run(teacher_route._bounded_body(Request([b"ab", b"cd"])))
    assert type(body) is bytearray
    assert body == b"abcd"

    with pytest.raises(teacher_broker.TeacherBrokerError, match="request_too_large"):
        asyncio.run(teacher_route._bounded_body(Request([b"ab", b"cd", b"e"])))


def test_strict_json_accepts_bytearray_without_converting_input():
    raw = bytearray(b'{"model":"parasail-glm-52"}')

    assert teacher_broker.parse_strict_json(raw) == {"model": "parasail-glm-52"}
    assert type(raw) is bytearray


@pytest.mark.parametrize("state", ["invalid", "revoked"])
def test_route_rejects_inactive_capability_before_consuming_stream(broker_db, state):
    from flash.server.routes import teacher as teacher_route

    token = "a" * 43
    if state == "revoked":
        token = _issue()
        db.revoke_teacher_capability(token)
    consumed = False

    class Request:
        def __init__(self):
            self.headers = {
                "authorization": f"Bearer {token}",
                "content-type": "application/json",
                "x-flash-teacher-request-id": f"request-route-{state}-001",
            }

        async def stream(self):
            nonlocal consumed
            consumed = True
            yield b"{}"

    response = asyncio.run(teacher_route.teacher_completions(Request()))

    assert response.status_code == 401
    assert consumed is False
    assert json.loads(response.body)["error"]["code"] == f"{state}_capability"


def test_route_bounds_concurrent_body_readers(monkeypatch):
    from flash.server.routes import teacher as teacher_route

    active = 0
    max_active = 0

    class Request:
        def __init__(self, index):
            self.headers = {
                "authorization": f"Bearer {'a' * 43}",
                "content-type": "application/json",
                "x-flash-teacher-request-id": f"request-ingress-{index:04d}",
            }

        async def stream(self):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.01)
                yield b"{}"
            finally:
                active -= 1

    async def exercise():
        monkeypatch.setattr(teacher_route, "_BODY_INGRESS_SEMAPHORE", asyncio.Semaphore(2))
        monkeypatch.setattr(
            teacher_route,
            "authenticate_teacher_capability",
            lambda **kwargs: (
                kwargs["request_id"],
                kwargs["capability_token"],
                {"expires_at": time.time() + 60},
            ),
        )
        monkeypatch.setattr(
            teacher_route,
            "complete_teacher_request",
            lambda **_kwargs: {"choices": []},
        )
        return await asyncio.gather(
            *(teacher_route.teacher_completions(Request(index)) for index in range(6))
        )

    responses = asyncio.run(exercise())

    assert all(response.status_code == 200 for response in responses)
    assert max_active == 2


def test_stalled_capability_bodies_time_out_without_blocking_another_capability(monkeypatch):
    from flash.server.routes import teacher as teacher_route

    active_stalled = 0
    stalled_ready = asyncio.Event()

    class Request:
        def __init__(self, token, request_id, *, stalled):
            self.headers = {
                "authorization": f"Bearer {token}",
                "content-type": "application/json",
                "x-flash-teacher-request-id": request_id,
            }
            self.stalled = stalled

        async def stream(self):
            nonlocal active_stalled
            if self.stalled:
                active_stalled += 1
                if active_stalled == 2:
                    stalled_ready.set()
                await asyncio.Event().wait()
            yield b"{}"

    async def exercise():
        monkeypatch.setattr(teacher_route, "MAX_BODY_INGRESS_SECONDS", 0.05)
        monkeypatch.setattr(teacher_route, "MAX_CONCURRENT_BODY_READERS_PER_CAPABILITY", 2)
        monkeypatch.setattr(teacher_route, "_BODY_INGRESS_SEMAPHORE", asyncio.Semaphore(3))
        teacher_route._CAPABILITY_INGRESS.clear()
        monkeypatch.setattr(
            teacher_route,
            "authenticate_teacher_capability",
            lambda **kwargs: (
                kwargs["request_id"],
                kwargs["capability_token"],
                {"expires_at": time.time() + 60},
            ),
        )
        monkeypatch.setattr(
            teacher_route,
            "complete_teacher_request",
            lambda **_kwargs: {"choices": []},
        )
        stalled = [
            asyncio.create_task(
                teacher_route.teacher_completions(
                    Request("a" * 43, f"request-stalled-a-{index:03d}", stalled=True)
                )
            )
            for index in range(2)
        ]
        await asyncio.wait_for(stalled_ready.wait(), timeout=1.0)
        other = await asyncio.wait_for(
            teacher_route.teacher_completions(
                Request("b" * 43, "request-service-b-001", stalled=False)
            ),
            timeout=0.2,
        )
        timed_out = await asyncio.gather(*stalled)
        recovered = await teacher_route.teacher_completions(
            Request("a" * 43, "request-recovered-a-001", stalled=False)
        )
        return other, timed_out, recovered

    other, timed_out, recovered = asyncio.run(exercise())

    assert other.status_code == 200
    # 429, not 408: the ingress timeout is retryable, and a retryable broker failure has to say so
    # in the status line because an intermediary can replace the body. a bare 408 is unusable for
    # that -- a proxy emits it for its own post-dispatch timeout -- so the concurrency-limit
    # condition reports as 429. see BODY_INDEPENDENT_TRANSIENT_STATUSES.
    assert [response.status_code for response in timed_out] == [429, 429]
    assert all(
        json.loads(response.body)["error"]["code"] == "body_ingress_timeout"
        for response in timed_out
    )
    assert recovered.status_code == 200
    assert teacher_route._CAPABILITY_INGRESS == {}


def test_current_nonterminal_attempt_is_checked_on_every_admission(monkeypatch):
    capability = {
        "run_id": "run-1",
        "attempt": 2,
        "teacher_alias": "glm-5.2",
        "provider": teacher_broker.PARASAIL_PROVIDER,
        "model": "parasail-glm-52",
        "scoring_mode": teacher_broker.PARASAIL_SCORING_MODE,
    }
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        train=TrainSpec(teacher_model="glm-5.2"),
        run_id="run-1",
    )

    monkeypatch.setattr(
        runner_status, "get_status", lambda _run_id: SimpleNamespace(state="running")
    )
    monkeypatch.setattr(runner_attempts, "latest_reserved_attempt", lambda _run_id: 2)
    monkeypatch.setattr(runner_state, "_internal_spec_from_status", lambda _status: spec)
    teacher_broker._require_current_attempt(capability)

    monkeypatch.setattr(runner_attempts, "latest_reserved_attempt", lambda _run_id: 3)
    with pytest.raises(teacher_broker.TeacherBrokerError, match="attempt_replaced"):
        teacher_broker._require_current_attempt(capability)

    monkeypatch.setattr(
        runner_status, "get_status", lambda _run_id: SimpleNamespace(state="cancelled")
    )
    with pytest.raises(teacher_broker.TeacherBrokerError, match="run_not_active"):
        teacher_broker._require_current_attempt(capability)


def test_only_predispatch_failure_can_retry_same_logical_request(broker_db, monkeypatch):
    _service_ready(monkeypatch)
    token = _issue()
    monkeypatch.delenv("PARASAIL_API_KEY")
    request_id = "request-predispatch-01"

    with pytest.raises(teacher_broker.TeacherBrokerError) as missing:
        teacher_broker.complete_teacher_request(
            capability_token=token, request_id=request_id, raw_body=_body()
        )
    assert missing.value.code == "provider_unavailable"
    assert missing.value.retryable is True

    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")
    monkeypatch.setattr(teacher_broker, "_provider_post", lambda *_args: (200, _response()))
    assert teacher_broker.complete_teacher_request(
        capability_token=token, request_id=request_id, raw_body=_body()
    )["choices"]


@pytest.mark.parametrize(
    "url",
    [
        "http://flash.example.com",
        "https://user:pass@flash.example.com",
        "https://flash.example.com?query=1",
        "https://flash.example.com#fragment",
        "https:///missing-host",
    ],
)
def test_public_url_requires_a_canonical_worker_reachable_https_origin(url):
    with pytest.raises(RuntimeError, match="FLASH_PUBLIC_URL"):
        teacher_broker.validate_public_url(url)

    assert teacher_broker.validate_public_url("https://flash.example.com/") == (
        "https://flash.example.com"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://plane.example:bad",
        "https://plane.example:99999",
        "https://plane.example:0",
        "https://plane.example:00",
    ],
)
def test_public_url_rejects_an_unusable_port_before_allocation(url):
    """urlsplit yields a hostname for these, so only forcing .port catches them here.

    a malformed or out-of-range port raises on attribute access, which without this gate would
    surface on the worker after the gpu is allocated. a zero port is worse: it is in range, so it
    parses cleanly, but it is falsy, so the `parsed.port or 443` the worker opens the connection
    with silently dials 443 instead. that one never fails at all, it just talks to the wrong port.
    """
    with pytest.raises(RuntimeError, match="valid port"):
        teacher_broker.validate_public_url(url)

    assert teacher_broker.validate_public_url("https://plane.example:8443") == (
        "https://plane.example:8443"
    )
    # a port-less origin is legitimate and must not be caught by the zero check: .port is None.
    assert teacher_broker.validate_public_url("https://plane.example") == "https://plane.example"


def test_public_url_does_not_fall_back_to_the_cli_api_url(monkeypatch):
    """FLASH_API_URL is where the cli dials in, which a rented worker may not resolve.

    falling back would defer that mismatch to the worker, after the gpu is allocated.
    """
    monkeypatch.delenv("FLASH_PUBLIC_URL", raising=False)
    monkeypatch.setenv("FLASH_API_URL", "https://flash.example.com")

    with pytest.raises(RuntimeError, match="FLASH_PUBLIC_URL"):
        teacher_broker.resolve_public_url()


def test_legacy_control_panel_url_does_not_configure_the_plane(monkeypatch):
    """the renamed FLASH_CONTROL_PANEL_URL must not keep working as an alias."""
    monkeypatch.delenv("FLASH_PUBLIC_URL", raising=False)
    monkeypatch.setenv("FLASH_CONTROL_PANEL_URL", "https://stale.example")

    with pytest.raises(RuntimeError, match="FLASH_PUBLIC_URL"):
        teacher_broker.resolve_public_url()


def test_legacy_teacher_broker_url_does_not_configure_the_plane(monkeypatch):
    monkeypatch.delenv("FLASH_PUBLIC_URL", raising=False)
    monkeypatch.setenv("FLASH_TEACHER_BROKER_URL", "https://broker.example")
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-legacy-control-panel-url",
    )

    with pytest.raises(RuntimeError, match="FLASH_PUBLIC_URL"):
        teacher_broker.require_teacher_broker_configuration(spec)


@pytest.mark.parametrize("missing", ["FLASH_PUBLIC_URL", "PARASAIL_API_KEY"])
def test_missing_broker_configuration_fails_before_allocation(monkeypatch, missing):
    import flash.providers.core.allocator as allocator
    from flash.runner.supervise import lifecycle

    monkeypatch.setenv("FLASH_PUBLIC_URL", "https://broker.example")
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")
    monkeypatch.delenv(missing, raising=False)

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("allocation must not run without broker configuration")

    monkeypatch.setattr(allocator, "allocate", unexpected_allocation)
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-preflight",
    )

    with pytest.raises(RuntimeError):
        lifecycle._run_attempts_supervised(spec, io.StringIO())


def test_failed_submission_scope_revokes_attempt_capability(broker_db, monkeypatch):
    monkeypatch.setenv("FLASH_PUBLIC_URL", "https://broker.example")
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
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
    monkeypatch.setenv("FLASH_PUBLIC_URL", "https://broker.example")
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
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
            provider=teacher_broker.PARASAIL_PROVIDER,
            model="parasail-glm-52",
            scoring_mode=teacher_broker.PARASAIL_SCORING_MODE,
            expires_at=time.time() + 600,
            limits=_limits(),
        )

    assert db.teacher_capability_binding(old_token)["revoked_at"] is not None
    assert db.teacher_capability_binding(new_token)["revoked_at"] is None


def test_cancellation_fences_teacher_capabilities_before_lifecycle_work(monkeypatch):
    from flash.runner.supervise.deploy import cancel_run

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
        runner_status,
        "get_status",
        lambda run_id: events.append(("status", run_id)) or status,
    )
    monkeypatch.setattr(runner_status, "effective_spec_from_status", lambda _status: None)

    def mark_cancelled(_run_id, state, **_updates):
        status.state = state
        return status

    monkeypatch.setattr(runner_status, "_update", mark_cancelled)

    assert cancel_run("run-1") is status
    assert status.state == "cancelled"
    assert events[0] == ("revoke", "run-1")
    assert [event for event in events if event[0] == "revoke"] == [
        ("revoke", "run-1"),
        ("revoke", "run-1"),
    ]


def test_runpod_lambda_and_vast_payloads_never_expose_provider_credentials(monkeypatch):
    from flash.providers._lifecycle.net.worker import build_worker_env
    from flash.providers.core.base import PollResult
    from flash.providers.lambda_.jobs.builders import build_payload as build_lambda_payload
    from flash.providers.runpod.execution import job_execution as runpod_job_execution
    from flash.providers.runpod.execution import polling as runpod_polling
    from flash.providers.vast.jobs.builders import build_payload as build_vast_payload

    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        environment=EnvironmentSpec(id="org/env"),
        train=TrainSpec(max_examples=8, hf_repo="owner/runs"),
        run_id="run-payload",
    )
    runtime = {
        "PARASAIL_API_KEY": "parasail-worker-canary",
        "FLASH_PUBLIC_URL": "https://broker.example",
        "FLASH_TEACHER_CAPABILITY": "capability-worker-canary",
    }
    monkeypatch.setenv("PARASAIL_API_KEY", "parasail-control-plane-canary")
    deadline = time.time() + 3600

    lambda_payload = build_lambda_payload(
        spec,
        0,
        runtime_secrets=runtime,
        source_snapshot=_SOURCE_SNAPSHOT,
        deadline_at=deadline,
    )
    vast_payload = build_vast_payload(
        spec,
        0,
        runtime_secrets=runtime,
        source_snapshot=_SOURCE_SNAPSHOT,
        deadline_at=deadline,
    )
    captured = {}
    monkeypatch.setattr(
        runpod_job_execution,
        "deploy_train_endpoint",
        lambda *_args, **_kwargs: ("endpoint-id", "endpoint-name", "key-fingerprint"),
    )

    def submit_job(_endpoint, payload, **_kwargs):
        captured["payload"] = payload
        return "job-id"

    monkeypatch.setattr(runpod_job_execution.runpod_api, "submit_job", submit_job)
    monkeypatch.setattr(
        runpod_polling,
        "poll_job",
        lambda *_args, **_kwargs: PollResult(True, metrics={}),
    )
    runpod_job_execution.submit_attempt(
        spec,
        attempt=0,
        runtime_secrets=runtime,
        source_snapshot=_SOURCE_SNAPSHOT,
        deadline_at=deadline,
    )

    serialized = json.dumps(
        {"runpod": captured["payload"], "lambda": lambda_payload, "vast": vast_payload}
    )
    assert "parasail-worker-canary" not in serialized
    assert "parasail-control-plane-canary" not in serialized
    assert "PARASAIL_API_KEY" not in serialized
    assert serialized.count("capability-worker-canary") == 3

    invalid_spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        environment=EnvironmentSpec(id="org/env", secrets=("PARASAIL_API_KEY",)),
        train=TrainSpec(max_examples=8, hf_repo="owner/runs"),
        run_id="run-invalid-payload",
    )
    with pytest.raises(ValueError, match="managed teacher credential names"):
        build_worker_env(invalid_spec, runtime_secrets=runtime)


def test_capability_policy_is_run_bounded_and_rejects_excessive_shapes(monkeypatch):
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1, batch_size=8, group_size=1),
        run_id="run-1",
    )
    limits = teacher_broker.capability_limits_for_spec(spec)
    assert limits["max_score_items"] == 24
    assert limits["max_requests"] == 24
    assert limits["max_upstream_attempts"] == 4
    assert limits["max_concurrency"] == teacher_broker.OPD_TEACHER_SCORING_CONCURRENCY

    monkeypatch.setattr(teacher_broker, "MAX_TOTAL_SCORE_ITEMS", 4)
    with pytest.raises(ValueError, match="score-item limit"):
        teacher_broker.capability_limits_for_spec(spec)


@pytest.mark.parametrize(("max_turns", "multiplier"), [(24, 72), (64, 192)])
def test_multiturn_capability_quota_covers_turns_and_no_signal_attempts(max_turns, multiplier):
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        environment=EnvironmentSpec(params={"multi_turn": True, "max_turns": max_turns}),
        train=TrainSpec(max_examples=8, max_steps=1, batch_size=8, group_size=1),
        run_id=f"run-turns-{max_turns}",
    )

    limits = teacher_broker.capability_limits_for_spec(spec)

    expected_requests = 8 * multiplier
    assert limits["max_requests"] == expected_requests
    assert limits["max_score_items"] == expected_requests
    assert limits["max_requests"] <= teacher_broker.MAX_TOTAL_REQUESTS
    assert limits["max_score_items"] <= teacher_broker.MAX_TOTAL_SCORE_ITEMS


def test_unknown_external_environment_uses_identical_ceiling_for_quota_and_cost():
    from flash.cost.spec import runconfig_from_spec
    from flash.teacher.limits import opd_teacher_request_multiplier

    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        environment=EnvironmentSpec(id="owner/external"),
        train=TrainSpec(max_examples=8, max_steps=1, batch_size=8, group_size=1),
        run_id="run-external-unknown-turns",
    )

    limits = teacher_broker.capability_limits_for_spec(spec)
    config = runconfig_from_spec(spec)
    cost_multiplier = opd_teacher_request_multiplier(
        multi_turn=config.opd_multi_turn,
        max_turns=config.opd_max_turns,
    )

    assert cost_multiplier == 192
    assert limits["max_requests"] == 8 * cost_multiplier
    assert limits["max_score_items"] == 8 * cost_multiplier


@pytest.mark.parametrize("missing", ["FLASH_PUBLIC_URL", "PARASAIL_API_KEY"])
def test_broker_gate_reason_survives_into_the_persisted_run_error(tmp_path, monkeypatch, missing):
    """A plane misconfiguration must be readable from the failed run, not flattened to "run failed".

    Every opd run failed ~2.7s after creation with a bare `RuntimeError: run failed`, because the
    terminal handler recorded only the exception TYPE. Both a missing plane credential and a bad
    spec landed on that identical string, so the run itself could not say which had happened.
    """
    import flash.providers.core.allocator as allocator
    from flash.runner.lifecycle.state import RunStatus
    from flash.runner.supervise import lifecycle

    monkeypatch.setenv("FLASH_PUBLIC_URL", "https://broker.example")
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")
    monkeypatch.delenv(missing, raising=False)
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    (tmp_path / "runs").mkdir()

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("allocation must not run without broker configuration")

    monkeypatch.setattr(allocator, "allocate", unexpected_allocation)
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-gate-detail",
    )
    runner_state._save_status(RunStatus(run_id="run-gate-detail", state="queued", spec={}))

    # catch the BASE type on purpose: against the unfixed handler this test must reach the
    # assertions below and fail on the flattened message, not stop early on a missing symbol.
    with pytest.raises(RuntimeError):
        lifecycle._run_job_inner(spec, str(tmp_path / "log.txt"), lambda *_a, **_k: None)

    status = runner_status.get_status("run-gate-detail")
    assert status.state == "failed"
    # the specific env var is what tells an operator which side is misconfigured
    assert missing in (status.error or "")
    assert (status.error or "") != "RuntimeError: run failed"


def test_unrelated_run_failures_keep_their_message_redacted(tmp_path, monkeypatch):
    """The gate's messages are authored for the submitter; an arbitrary exception's are not.

    `RunStatus.error` is user-visible, so preserving `str(exc)` wholesale would publish internal
    storage paths and upstream bodies. Only the typed configuration gate opts back in.
    """
    from flash.runner.lifecycle.state import RunStatus
    from flash.runner.supervise import lifecycle

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    (tmp_path / "runs").mkdir()
    raw = "/internal/artifacts/secret-path exploded"

    def boom(*_args, **_kwargs):
        raise RuntimeError(raw)

    monkeypatch.setattr(runner_lifecycle, "_run_training", boom)
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-generic-failure",
    )
    runner_state._save_status(RunStatus(run_id="run-generic-failure", state="queued", spec={}))

    with pytest.raises(RuntimeError):
        lifecycle._run_job_inner(spec, str(tmp_path / "log.txt"), lambda *_a, **_k: None)

    status = runner_status.get_status("run-generic-failure")
    assert status.error == "RuntimeError: run failed"
    assert raw not in (status.error or "")


@pytest.mark.parametrize("missing", ["FLASH_PUBLIC_URL", "PARASAIL_API_KEY"])
def test_dry_run_rejects_an_unservable_opd_spec_before_creating_a_run(
    tmp_path, monkeypatch, missing
):
    """The gate ran only before allocation, so `--dry-run` validated clean and left a run record.

    Hoisting it into submit-time validation means the misconfiguration is reported by the preview
    that exists to catch exactly this, and no run row is written for work that cannot start.
    """
    from flash.runner.lifecycle.submit import submit_job

    monkeypatch.setenv("FLASH_PUBLIC_URL", "https://broker.example")
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")
    monkeypatch.delenv(missing, raising=False)
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    (tmp_path / "runs").mkdir()
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-dry-gate",
    )

    # base type again: an unfixed submit path raises nothing at all here, and the assertion that
    # matters is that the dry run refused and wrote no record.
    with pytest.raises(RuntimeError, match=missing):
        submit_job(spec, dry_run=True)

    assert not (tmp_path / "runs" / "run-dry-gate.json").exists()


def test_dry_run_still_previews_a_servable_opd_spec(tmp_path, monkeypatch):
    """The hoisted gate must not reject a correctly configured plane: paired control for the above."""
    from flash.runner.lifecycle.submit import submit_job

    monkeypatch.setenv("FLASH_PUBLIC_URL", "https://broker.example")
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    (tmp_path / "runs").mkdir()
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        train=TrainSpec(max_examples=8, max_steps=1),
        run_id="run-dry-ok",
    )

    assert submit_job(spec, dry_run=True).state == "dry_run"


def test_preflight_is_a_noop_for_non_opd_algorithms(monkeypatch):
    """sft/grpo have no managed teacher, so the hoisted gate must not gate them on plane env."""
    monkeypatch.delenv("FLASH_PUBLIC_URL", raising=False)
    monkeypatch.delenv("PARASAIL_API_KEY", raising=False)
    for algorithm in ("sft", "grpo"):
        spec = JobSpec(
            model="Qwen/Qwen3.5-9B",
            algorithm=algorithm,
            train=TrainSpec(max_examples=8, max_steps=1),
            run_id=f"run-{algorithm}-noop",
        )
        assert teacher_broker.preflight_validate_managed_teacher(spec) is None

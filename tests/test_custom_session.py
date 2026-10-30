"""CPU-only protocol tests: retries, stale weights, masks, and poisoned state."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from flash.custom.server import create_app
from flash.custom.session import SessionConflict, TrainingSession
from flash.custom.types import Datum, parse_batch


class CounterBackend:
    def __init__(self, tmp_path):
        self.value = 0
        self.checkpoint_dir = tmp_path

    def info(self):
        return {"value": self.value}

    def validate(self, operation, arguments):
        if operation not in {"optim_step", "close", "crash"}:
            raise ValueError("unknown operation")
        return arguments

    def execute(self, operation, arguments):
        self.value += 1
        if operation == "crash":
            raise RuntimeError("partial GPU mutation")
        return {"metrics": {"value": self.value}}


def request(session, operation="optim_step", **changes):
    return {
        "session_id": session.id,
        "sequence": session.sequence + 1,
        "model_version": session.version,
        "operation": operation,
        "arguments": {},
        **changes,
    }


def test_concurrent_duplicate_optimizer_request_applies_once(tmp_path):
    backend = CounterBackend(tmp_path)
    session = TrainingSession(backend)
    payload = request(session)
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(session.execute, [payload] * 4))
    assert replies == [replies[0]] * 4
    assert backend.value == 1
    assert session.step == 1
    assert session.version == 1


def test_stale_changed_and_expired_sequences_never_mutate(tmp_path):
    session = TrainingSession(CounterBackend(tmp_path))
    first = request(session)
    session.execute(first)
    changed = copy.deepcopy(first)
    changed["arguments"] = {"learning_rate": 0.9}
    with pytest.raises(SessionConflict, match="different request"):
        session.execute(changed)
    with pytest.raises(SessionConflict, match="stale model version"):
        session.execute(request(session, model_version=0))
    session.execute(request(session))
    with pytest.raises(SessionConflict, match="expired"):
        session.execute(first)
    with pytest.raises(SessionConflict, match="session changed"):
        session.execute(request(session, session_id="previous-process"))
    assert session.backend.value == 2


def test_partial_failure_is_never_replayed(tmp_path):
    session = TrainingSession(CounterBackend(tmp_path))
    payload = request(session, operation="crash")
    with pytest.raises(RuntimeError, match="partial GPU"):
        session.execute(payload)
    with pytest.raises(SessionConflict, match="failed"):
        session.execute(payload)
    assert session.failed
    assert session.backend.value == 1
    assert session.events[-1]["error"] == "RuntimeError: partial GPU mutation"


def test_step_cap_close_and_idle_state(tmp_path):
    session = TrainingSession(CounterBackend(tmp_path), max_steps=1, idle_seconds=30)
    session.execute(request(session))
    with pytest.raises(ValueError, match="limit"):
        session.execute(request(session))
    session.last_activity -= 31
    assert session.expired()
    session.active_exports = 1
    assert not session.expired()
    session.active_exports = 0
    with session.lock:
        assert not session.expired()
    session.execute(request(session, operation="close"))
    with pytest.raises(SessionConflict, match="closed"):
        session.execute(request(session))


@pytest.mark.parametrize(
    ("tokens", "weights"),
    [
        ([1], []),
        ([1, True], [1]),
        ([1, -1], [1]),
        ([1, 100], [1]),
        ([1, 2], [1, 1]),
        ([1, 2], [float("nan")]),
        ([1, 2], None),
    ],
)
def test_invalid_token_alignment_rejected(tokens, weights):
    with pytest.raises(ValueError, match=r"datum|token|weights|weight|forward_backward"):
        parse_batch(
            [Datum(tokens, weights).to_dict()],
            vocab_size=100,
            max_tokens=10,
            max_batch=2,
            weighted=True,
        )


def test_signed_weights_preserved_without_implicit_normalization():
    rows = parse_batch(
        [{"tokens": [1, 2, 3], "weights": [0, -0.5]}],
        vocab_size=100,
        max_tokens=10,
        max_batch=2,
        weighted=True,
    )
    assert rows[0].weights == [0.0, -0.5]


def test_http_auth_validation_logs_and_retry(tmp_path):
    session = TrainingSession(CounterBackend(tmp_path))
    client = TestClient(create_app(session, "s" * 40))
    assert client.get("/v1/custom/status").status_code == 401
    headers = {"Authorization": "Bearer " + "s" * 40}
    payload = request(session)
    first = client.post("/v1/custom/operation", headers=headers, json=payload)
    assert first.status_code == 200
    assert client.post("/v1/custom/operation", headers=headers, json=payload).json() == first.json()
    bad = client.post(
        "/v1/custom/operation", headers=headers, json=request(session, model_version=0)
    )
    assert bad.status_code == 409
    assert bad.json()["detail"]["code"] == "custom_rejected"
    assert (
        client.get("/v1/custom/logs", headers=headers).json()["events"][0]["operation"]
        == "optim_step"
    )
    assert (
        client.post(
            "/v1/custom/operation", headers=headers, content="x" * (4 * 1024 * 1024 + 1)
        ).status_code
        == 413
    )
    assert session.step == 1

from __future__ import annotations

import copy
import io
import json
import multiprocessing
import threading
import uuid
from pathlib import Path

import pytest

from flash.serve.deploy import ServingError
from flash.serve.model_checkpoints import (
    FLASH_INTERPOLATED_CHECKPOINT_NAMESPACE,
    build_checkpoint_outbox,
    checkpoint_deployment_token,
    checkpoint_registration_payload,
    validate_active_checkpoint_response,
)

_CONTRACT_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "flash_checkpoint_contract_v1.json"


def _contract_fixture() -> dict:
    return json.loads(_CONTRACT_FIXTURE_PATH.read_text())


def _intent(run_id: str) -> dict:
    return {
        "schema": "flash.interpolated_checkpoint_intent",
        "version": 1,
        "model_id": run_id,
        "base_model": "Qwen/Qwen3.5-4B",
        "model_repo_id": f"Freesolo-Co/flash-checkpoint-{run_id}",
        "model_revision": "d" * 40,
        "tokenizer_repo_id": None,
        "tokenizer_revision": None,
        "thinking": False,
        "structured_outputs": None,
        "private": True,
        "metadata": {
            "schema": "flash.model-interpolation",
            "version": 1,
            "canonical_model": "Qwen/Qwen3.5-4B",
            "formula": "W=(1-alpha)*W_base+alpha*W_instruct",
            "alpha": 0.5,
            "parents": {
                "base": {
                    "model": "Qwen/Qwen3.5-4B-Base",
                    "requested_revision": "a" * 40,
                    "commit": "a" * 40,
                    "config_fingerprint": "1" * 64,
                },
                "instruct": {
                    "model": "Qwen/Qwen3.5-4B",
                    "requested_revision": "b" * 40,
                    "commit": "b" * 40,
                    "config_fingerprint": "2" * 64,
                },
            },
            "tokenizer_config_source": "instruct",
            "interpolation_output_fingerprint": "f" * 64,
            "interpolation_tree_fingerprint": "3" * 64,
            "trained_tree_fingerprint": "e" * 64,
        },
        "output_fingerprint": "e" * 64,
        "interpolation_output_fingerprint": "f" * 64,
    }


def _ready(checkpoint: dict) -> dict:
    return {
        "protocol_version": 1,
        "schema": "flash.interpolated_checkpoint_intent",
        "version": checkpoint["version"],
        "deployment_token": checkpoint["deployment_token"],
        "payload_hash": checkpoint["payload_hash"],
        "model_id": checkpoint["model_id"],
        "base_model": checkpoint["base_model"],
        "model_repo_id": checkpoint["model_repo_id"],
        "model_revision": checkpoint["model_revision"],
        "tokenizer_repo_id": checkpoint["tokenizer_repo_id"],
        "tokenizer_revision": checkpoint["tokenizer_revision"],
        "thinking": checkpoint["thinking"],
        "structured_outputs": checkpoint["structured_outputs"],
        "private": True,
        "metadata": checkpoint["metadata"],
        "output_fingerprint": checkpoint["output_fingerprint"],
        "interpolation_output_fingerprint": checkpoint[
            "interpolation_output_fingerprint"
        ],
        "status": "ready",
        "activated_at": "2026-07-11T00:00:00Z",
    }


class _Response:
    def __init__(self, status_code: int, body: dict | None = None):
        self.status_code = status_code
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def _record_synced_mirror(runner, run_id: str, *, mirrored_at: float | None = None):
    checkpoint = runner.get_status(run_id).checkpoint
    return runner.record_checkpoint_mirror(
        run_id,
        synced=True,
        mirrored_at=mirrored_at,
        expected_deployment_token=checkpoint["deployment_token"],
        expected_activation_state=checkpoint["activation_state"],
        expected_activation_updated_at=checkpoint["activation_updated_at"],
    )


def _stored_run(monkeypatch, tmp_path, run_id="flash-activation"):
    import flash.runner as runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(run_id=run_id, model="Qwen/Qwen3.5-4B", algorithm="grpo")
    checkpoint = build_checkpoint_outbox(_intent(run_id), run_id=run_id, now=1.0)
    status = runner.RunStatus(run_id=run_id, state="done", spec=spec.to_dict(), checkpoint=checkpoint)
    runner._save_status(status)
    result_dir = tmp_path / "results" / "runpod" / "rl" / run_id
    result_dir.mkdir(parents=True)
    (result_dir / "metrics.json").write_text(json.dumps({"notes": {}}))
    return runner, checkpoint


def test_flash_checkpoint_contract_v1_fixture_is_stable():
    from flash.runner import RunStatus
    from flash.server.run_registry import _checkpoint_from_status

    fixture = _contract_fixture()
    intent = fixture["intent"]
    checkpoint = build_checkpoint_outbox(intent, run_id=intent["model_id"], now=1.0)
    registration = checkpoint_registration_payload(checkpoint)

    assert checkpoint["payload_hash"] == fixture["payload_hash"]
    assert checkpoint["deployment_token"] == fixture["deployment_token"]
    assert registration == fixture["registration_payload"]
    response_identity = {
        key: value for key, value in registration.items() if key != "expected_active_token"
    }
    assert fixture["active_response"] == {
        **response_identity,
        "status": "ready",
        "activated_at": "2026-07-11T00:00:00Z",
    }
    assert validate_active_checkpoint_response(fixture["active_response"], checkpoint) == fixture[
        "active_response"
    ]
    assert _checkpoint_from_status(
        RunStatus(run_id=intent["model_id"], state="done", spec={}, checkpoint=checkpoint)
    ) == fixture["run_mirror_checkpoint"]
    assert checkpoint_deployment_token(fixture["payload_hash"]) == str(
        uuid.uuid5(FLASH_INTERPOLATED_CHECKPOINT_NAMESPACE, fixture["payload_hash"])
    )


def test_production_shaped_active_response_without_expected_active_token_is_accepted():
    fixture = _contract_fixture()
    intent = fixture["intent"]
    checkpoint = build_checkpoint_outbox(intent, run_id=intent["model_id"], now=1.0)

    assert "expected_active_token" not in fixture["active_response"]
    assert validate_active_checkpoint_response(fixture["active_response"], checkpoint) == fixture[
        "active_response"
    ]


@pytest.mark.parametrize(
    "field",
    [
        "protocol_version",
        "schema",
        "version",
        "deployment_token",
        "payload_hash",
        "model_id",
        "base_model",
        "model_repo_id",
        "model_revision",
        "tokenizer_repo_id",
        "tokenizer_revision",
        "thinking",
        "structured_outputs",
        "private",
        "metadata",
        "output_fingerprint",
        "interpolation_output_fingerprint",
        "status",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "mismatch"])
def test_active_checkpoint_response_requires_exact_canonical_identity(field, mutation):
    fixture = _contract_fixture()
    intent = fixture["intent"]
    checkpoint = build_checkpoint_outbox(intent, run_id=intent["model_id"], now=1.0)
    response = copy.deepcopy(fixture["active_response"])
    if mutation == "missing":
        del response[field]
    else:
        response[field] = "mismatch"

    with pytest.raises(ServingError, match=field) as excinfo:
        validate_active_checkpoint_response(response, checkpoint)

    assert excinfo.value.status_code == 409


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("metadata", "alpha"), True, "alpha"),
        (("metadata", "parents", "base", "model"), "other/model", "parent pair"),
        (("metadata", "trained_tree_fingerprint"), "0" * 64, "does not match"),
    ],
)
def test_checkpoint_intent_rejects_invalid_metadata(path, value, match):
    intent = copy.deepcopy(_intent("metadata-validation"))
    target = intent
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        build_checkpoint_outbox(intent, run_id="metadata-validation", now=1.0)


def test_activation_response_loss_recovers_by_exact_get(monkeypatch, tmp_path):
    from flash.server import checkpoint_retry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return _Response(200, _ready(checkpoint))

    monkeypatch.setattr(checkpoint_retry, "checkpoint_request", request)
    monkeypatch.setattr(checkpoint_retry, "reconcile_backend_mirrors", lambda run_id: True)

    assert checkpoint_retry.reconcile_checkpoint(checkpoint["model_id"]) is True
    assert [method for method, _, _ in calls] == ["GET"]
    assert runner.get_status(checkpoint["model_id"]).checkpoint["activation_state"] == "active"


def test_restart_from_activating_reuses_token(monkeypatch, tmp_path):
    from flash.server import checkpoint_retry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    runner.record_checkpoint_state(
        checkpoint["model_id"],
        deployment_token=checkpoint["deployment_token"],
        activation_state="activating",
        increment_attempts=True,
    )
    seen_tokens = []

    def request(method, path, **kwargs):
        if method == "GET" and not seen_tokens:
            seen_tokens.append(kwargs["params"]["expected_deployment_token"])
            return _Response(404)
        if method == "POST":
            seen_tokens.append(kwargs["json"]["deployment_token"])
            return _Response(201, _ready(checkpoint))
        return _Response(200, _ready(checkpoint))

    monkeypatch.setattr(checkpoint_retry, "checkpoint_request", request)
    monkeypatch.setattr(checkpoint_retry, "reconcile_backend_mirrors", lambda run_id: True)
    checkpoint_retry.reconcile_checkpoint(checkpoint["model_id"])
    assert seen_tokens == [checkpoint["deployment_token"], checkpoint["deployment_token"]]


def test_active_checkpoint_cannot_regress(monkeypatch, tmp_path):
    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    runner.record_checkpoint_state(
        checkpoint["model_id"],
        deployment_token=checkpoint["deployment_token"],
        activation_state="active",
    )
    with pytest.raises(ValueError, match="not allowed"):
        runner.record_checkpoint_state(
            checkpoint["model_id"],
            deployment_token=checkpoint["deployment_token"],
            activation_state="retry_wait",
        )


def test_permanent_conflict_becomes_failed(monkeypatch, tmp_path):
    from flash.server import checkpoint_retry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    responses = iter([_Response(404), _Response(409, {"detail": "occupied"})])
    monkeypatch.setattr(checkpoint_retry, "checkpoint_request", lambda *a, **k: next(responses))
    checkpoint_retry.reconcile_checkpoint(checkpoint["model_id"])
    assert runner.get_status(checkpoint["model_id"]).checkpoint["activation_state"] == "failed"


def test_unprocessable_registration_becomes_failed(monkeypatch, tmp_path):
    from flash.server import checkpoint_retry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    responses = iter([_Response(404), _Response(422, {"detail": "invalid contract"})])
    monkeypatch.setattr(checkpoint_retry, "checkpoint_request", lambda *a, **k: next(responses))

    checkpoint_retry.reconcile_checkpoint(checkpoint["model_id"])

    stored = runner.get_status(checkpoint["model_id"]).checkpoint
    assert stored["activation_state"] == "failed"
    assert "POST HTTP 422" in stored["activation_error"]


def test_token_specific_readback_conflict_becomes_failed(monkeypatch, tmp_path):
    from flash.server import checkpoint_retry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    responses = iter(
        [
            _Response(404),
            _Response(201, _ready(checkpoint)),
            _Response(409, {"detail": "deployment token mismatch"}),
        ]
    )
    monkeypatch.setattr(checkpoint_retry, "checkpoint_request", lambda *a, **k: next(responses))

    checkpoint_retry.reconcile_checkpoint(checkpoint["model_id"])

    stored = runner.get_status(checkpoint["model_id"]).checkpoint
    assert stored["activation_state"] == "failed"
    assert stored["activation_error"] == "readback HTTP 409"


def test_transient_retry_is_bounded_and_has_no_limit(monkeypatch, tmp_path):
    from flash.server import checkpoint_retry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    monkeypatch.setattr(checkpoint_retry, "checkpoint_request", lambda *a, **k: _Response(503))
    checkpoint_retry.reconcile_checkpoint(checkpoint["model_id"])
    stored = runner.get_status(checkpoint["model_id"]).checkpoint
    assert stored["activation_state"] == "retry_wait"
    assert 0 < stored["activation_next_retry_at"] - stored["activation_updated_at"] <= 900
    assert checkpoint_retry.retry_delay_seconds(checkpoint["deployment_token"], 1000) <= 900


def test_explicit_deactivation_is_token_qualified_and_confirmed(monkeypatch, tmp_path):
    from flash.server import checkpoint_retry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    runner.record_checkpoint_state(
        checkpoint["model_id"],
        deployment_token=checkpoint["deployment_token"],
        activation_state="active",
    )
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return _Response(200) if method == "DELETE" else _Response(404)

    monkeypatch.setattr(checkpoint_retry, "checkpoint_request", request)
    monkeypatch.setattr(checkpoint_retry, "reconcile_backend_mirrors", lambda run_id: True)
    assert checkpoint_retry.deactivate_checkpoint(checkpoint["model_id"]) is True
    assert [method for method, _, _ in calls] == ["DELETE", "GET"]
    assert calls[0][2]["params"] == {
        "expected_deployment_token": checkpoint["deployment_token"]
    }
    stored = runner.get_status(checkpoint["model_id"]).checkpoint
    assert stored["activation_state"] == "disabled"


def test_cross_process_checkpoint_updates_are_serialized(monkeypatch, tmp_path):
    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    runner.record_checkpoint_state(
        checkpoint["model_id"],
        deployment_token=checkpoint["deployment_token"],
        activation_state="activating",
    )

    def increment_attempts():
        for _ in range(20):
            runner.record_checkpoint_state(
                checkpoint["model_id"],
                deployment_token=checkpoint["deployment_token"],
                activation_state="activating",
                increment_attempts=True,
            )

    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=increment_attempts) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    stored = runner.get_status(checkpoint["model_id"]).checkpoint
    assert stored["activation_attempts"] == 40


def test_every_activation_transition_resets_backend_mirror_atomically(monkeypatch, tmp_path):
    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    transitions = ("activating", "retry_wait", "activating", "active", "disabled")

    for activation_state in transitions:
        _record_synced_mirror(runner, checkpoint["model_id"], mirrored_at=2.0)
        assert runner.record_checkpoint_state(
            checkpoint["model_id"],
            deployment_token=checkpoint["deployment_token"],
            activation_state=activation_state,
        )
        stored = runner.get_status(checkpoint["model_id"]).checkpoint
        assert stored["backend_mirror_state"] == "pending"
        assert stored["backend_mirror_error"] is None
        assert stored["backend_mirrored_at"] is None


def test_checkpoint_cannot_be_removed_after_persistence(monkeypatch, tmp_path):
    runner, checkpoint = _stored_run(monkeypatch, tmp_path)

    def remove(status):
        status.checkpoint = None
        return True

    with pytest.raises(ValueError, match="cannot be removed"):
        runner._mutate_status(checkpoint["model_id"], remove)
    assert runner.get_status(checkpoint["model_id"]).checkpoint == checkpoint


def test_backend_mirror_failure_does_not_change_activation(monkeypatch, tmp_path):
    from flash.server import checkpoint_retry, run_registry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    runner.record_checkpoint_state(
        checkpoint["model_id"],
        deployment_token=checkpoint["deployment_token"],
        activation_state="active",
    )
    monkeypatch.setattr(run_registry, "record_training_run", lambda **kwargs: False)
    monkeypatch.setattr(run_registry, "record_training_checkpoint", lambda **kwargs: False)
    assert checkpoint_retry.reconcile_backend_mirrors(checkpoint["model_id"]) is False
    stored = runner.get_status(checkpoint["model_id"]).checkpoint
    assert stored["activation_state"] == "active"
    assert stored["backend_mirror_state"] == "pending"


def test_mixed_backend_replicas_cannot_sync_when_run_mirror_ignores_checkpoint(
    monkeypatch, tmp_path
):
    from urllib.parse import urlparse

    from flash.server import checkpoint_retry, run_registry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    status = runner.get_status(checkpoint["model_id"])
    status.platform_context = {"org_id": "org-1"}
    runner._save_status(status)
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-test")
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://backend.test")
    seen = []

    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    def urlopen(request, timeout=None):
        path = urlparse(request.full_url).path
        seen.append(path)
        if path == "/api/flash/runs/internal":
            return Response({})
        if path == "/api/flash/runs/checkpoints/internal":
            return Response({"checkpointAccepted": True})
        raise AssertionError(f"unexpected backend path: {path}")

    monkeypatch.setattr(run_registry.urllib.request, "urlopen", urlopen)

    assert checkpoint_retry.reconcile_backend_mirrors(checkpoint["model_id"]) is False
    assert seen == [
        "/api/flash/runs/internal",
        "/api/flash/runs/checkpoints/internal",
    ]
    stored = runner.get_status(checkpoint["model_id"]).checkpoint
    assert stored["backend_mirror_state"] == "pending"
    assert "not accepted" in stored["backend_mirror_error"]


def test_stale_backend_mirror_cannot_sync_newer_activation_snapshot(monkeypatch, tmp_path):
    from flash.server import checkpoint_retry, run_registry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    request_started = threading.Event()
    release_response = threading.Event()
    result = []

    def record_run(*, status):
        assert status.checkpoint["activation_state"] == "pending"
        request_started.set()
        assert release_response.wait(timeout=5)
        return True

    monkeypatch.setattr(run_registry, "record_training_run", record_run)
    monkeypatch.setattr(run_registry, "record_training_checkpoint", lambda **kwargs: True)

    thread = threading.Thread(
        target=lambda: result.append(checkpoint_retry.reconcile_backend_mirrors(checkpoint["model_id"]))
    )
    thread.start()
    assert request_started.wait(timeout=5)
    assert runner.record_checkpoint_state(
        checkpoint["model_id"],
        deployment_token=checkpoint["deployment_token"],
        activation_state="active",
    )
    release_response.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result == [False]
    stored = runner.get_status(checkpoint["model_id"]).checkpoint
    assert stored["activation_state"] == "active"
    assert stored["backend_mirror_state"] == "pending"
    assert stored["backend_mirror_error"] is None
    assert stored["backend_mirrored_at"] is None


def test_repeated_transient_sweeps_keep_retrying(monkeypatch, tmp_path):
    from flash.server import checkpoint_retry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(checkpoint_retry, "_RETRY_BASE_SECONDS", 0.0)
    monkeypatch.setattr(
        checkpoint_retry,
        "checkpoint_request",
        lambda *a, **k: (calls.append((a, k)), _Response(503))[1],
    )
    monkeypatch.setattr(checkpoint_retry, "reconcile_backend_mirrors", lambda run_id: False)

    for _ in range(3):
        assert checkpoint_retry.reconcile_checkpoints_once() == 0

    stored = runner.get_status(checkpoint["model_id"]).checkpoint
    assert stored["activation_state"] == "retry_wait"
    assert stored["activation_attempts"] == 3
    assert len(calls) == 3


@pytest.mark.parametrize("activation_state", ["failed", "disabled"])
def test_sweep_retries_unsynced_mirror_for_terminal_activation_states(
    monkeypatch, tmp_path, activation_state
):
    from flash.server import checkpoint_retry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    if activation_state == "failed":
        runner.record_checkpoint_state(
            checkpoint["model_id"],
            deployment_token=checkpoint["deployment_token"],
            activation_state="failed",
        )
    else:
        runner.record_checkpoint_state(
            checkpoint["model_id"],
            deployment_token=checkpoint["deployment_token"],
            activation_state="active",
        )
        runner.record_checkpoint_state(
            checkpoint["model_id"],
            deployment_token=checkpoint["deployment_token"],
            activation_state="disabled",
        )
    mirrored = []
    monkeypatch.setattr(checkpoint_retry, "reconcile_backend_mirrors", mirrored.append)
    monkeypatch.setattr(
        checkpoint_retry,
        "checkpoint_request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("serving must not be called")),
    )

    assert checkpoint_retry.reconcile_checkpoints_once() == 0
    assert mirrored == [checkpoint["model_id"]]


def test_converged_sweep_does_not_call_serving(monkeypatch, tmp_path):
    from flash.server import checkpoint_retry

    runner, checkpoint = _stored_run(monkeypatch, tmp_path)
    runner.record_checkpoint_state(
        checkpoint["model_id"],
        deployment_token=checkpoint["deployment_token"],
        activation_state="active",
    )
    _record_synced_mirror(runner, checkpoint["model_id"])
    monkeypatch.setattr(
        checkpoint_retry,
        "checkpoint_request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("serving must not be called")),
    )
    assert checkpoint_retry.reconcile_checkpoints_once() == 0


def test_status_write_failure_and_cancelled_race_make_zero_serving_calls(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.runner import lifecycle
    from flash.server import checkpoint_retry
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(run_id="ordinary", model="Qwen/Qwen3.5-4B", algorithm="grpo")
    runner._save_status(runner.RunStatus(run_id=spec.run_id, state="cancelled", spec=spec.to_dict()))
    calls = []
    monkeypatch.setattr(checkpoint_retry, "request_checkpoint_reconciliation", calls.append)
    assert lifecycle._complete_run(
        spec, {"wall_seconds": 1.0, "notes": {}}, prior_cost=0.0, log=io.StringIO()
    ) is False
    assert calls == []

    status = runner.get_status(spec.run_id)
    status.state = "running"
    runner._save_status(status)
    monkeypatch.setattr(runner, "_update", lambda *a, **k: False)
    assert lifecycle._complete_run(
        spec, {"wall_seconds": 1.0, "notes": {}}, prior_cost=0.0, log=io.StringIO()
    ) is False
    assert calls == []

    def fail_write(*args, **kwargs):
        raise OSError("status persistence failed")

    monkeypatch.setattr(runner, "_update", fail_write)
    with pytest.raises(OSError, match="status persistence failed"):
        lifecycle._complete_run(
            spec, {"wall_seconds": 1.0, "notes": {}}, prior_cost=0.0, log=io.StringIO()
        )
    assert calls == []


def test_normal_completion_and_attach_run_persist_equivalent_checkpoints(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.providers.base import PollResult
    from flash.runner import lifecycle
    from flash.runner.deploy import attach_run
    from flash.server import checkpoint_retry
    from flash.spec import JobSpec, ModelInterpolationSpec

    spec = JobSpec(
        run_id="completion-equivalence",
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        model_initialization=ModelInterpolationSpec(
            base_model="Qwen/Qwen3.5-4B-Base",
            instruct_model="Qwen/Qwen3.5-4B",
            alpha=0.5,
            base_revision="a" * 40,
            instruct_revision="b" * 40,
        ),
        checkpoint_protocol="control_plane_v1",
    )
    metrics = {
        "wall_seconds": 1.0,
        "cost_usd": 0.01,
        "notes": {"interpolated_checkpoint_intent": _intent(spec.run_id)},
    }
    monkeypatch.setattr(lifecycle, "_charge_completed_run_best_effort", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_register_checkpoints_best_effort", lambda *a, **k: None)
    monkeypatch.setattr(checkpoint_retry, "request_checkpoint_reconciliation", lambda run_id: None)
    monkeypatch.setattr(checkpoint_retry, "reconcile_backend_mirrors", lambda run_id: False)
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda spec: None)

    def complete(root, *, attached):
        monkeypatch.setattr(runner, "RUNS_DIR", str(root / "runs"))
        monkeypatch.setattr(runner, "RESULTS_DIR", str(root / "results"))
        status = runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            remote={"provider": "fake"} if attached else None,
        )
        runner._save_status(status)
        if attached:
            provider = type(
                "Provider",
                (),
                {"poll": lambda self, *a, **k: PollResult(ok=True, metrics=copy.deepcopy(metrics))},
            )()
            monkeypatch.setattr("flash.providers.get_provider", lambda name: provider)
            result = attach_run(spec.run_id, log_stream=io.StringIO())
        else:
            assert lifecycle._complete_run(
                spec,
                copy.deepcopy(metrics),
                prior_cost=0.0,
                log=io.StringIO(),
            )
            result = runner.get_status(spec.run_id)
        checkpoint = dict(result.checkpoint)
        checkpoint.pop("activation_updated_at")
        return checkpoint

    normal = complete(tmp_path / "normal", attached=False)
    attached = complete(tmp_path / "attached", attached=True)
    assert attached == normal

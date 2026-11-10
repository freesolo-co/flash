"""Custom training on Nebius Spot: direct pinned-TLS sessions, automatic saves, preemption recovery."""

import itertools
import json
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flash.client.http import ClientError
from flash.schema import ConfigError, spec_from_dict


@pytest.fixture(autouse=True)
def internal_key(monkeypatch):
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "test-internal-key")


def custom_raw(**gpu):
    return {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "custom",
        "project": "11111111-1111-4111-8111-111111111111",
        "custom": {"file": "algorithm.py"},
        "train": {"max_steps": 6},
        "gpu": {"type": "H100", **gpu},
    }


# --- placement and schema -------------------------------------------------------------------


def test_custom_spot_spec_parses_with_automatic_saves_on_one_gpu():
    spec = spec_from_dict(custom_raw(spot=True))
    assert spec.gpu.provider == "spot"
    assert spec.train.save_every is None  # Spot recovery saves are automatic and time-based


def test_custom_spot_rejects_manual_interval_and_two_gpus():
    raw = custom_raw(spot=True)
    raw["train"]["save_every"] = 3
    with pytest.raises((ConfigError, ValueError), match="time-based and automatic"):
        spec_from_dict(raw)
    with pytest.raises(ConfigError, match="exactly one GPU"):
        spec_from_dict(custom_raw(spot=True, count=2))


def test_custom_gpu_shape_keeps_modal_default_and_allows_one_spot_gpu():
    from flash.providers.core.base import custom_gpu_shape

    assert custom_gpu_shape("Qwen/Qwen3.5-9B", "", ("H100",), None)[0] == "modal"
    assert custom_gpu_shape("Qwen/Qwen3.5-9B", "modal", ("H100",), 2)[3] == 2
    assert custom_gpu_shape("Qwen/Qwen3.5-9B", "spot", ("H100",), None) == ("spot", "H100", (), 1)
    with pytest.raises(ValueError, match="exactly one GPU"):
        custom_gpu_shape("Qwen/Qwen3.5-9B", "spot", ("H100",), 2)
    with pytest.raises(ValueError, match="managed Modal GPUs or one Spot GPU"):
        custom_gpu_shape("Qwen/Qwen3.5-9B", "lambda", ("H100",), 1)


# --- per-attempt TLS identity ---------------------------------------------------------------


def test_session_tls_is_deterministic_per_attempt():
    from flash.custom.managed import session_tls

    first = session_tls("run-a", 0)
    assert first == session_tls("run-a", 0)  # reconciled relaunches serve the same identity
    assert first["cert"] != session_tls("run-a", 1)["cert"]
    assert first["cert"] != session_tls("run-b", 0)["cert"]
    assert "PRIVATE KEY" in first["key"]
    assert "CERTIFICATE" in first["cert"]


class _Status(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"session_id": "s1", "step": 4, "sequence": 9, "model_version": 2})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def tls_status_server(tmp_path):
    from flash.custom.managed import session_tls

    identity = session_tls("run-a", 3)
    (tmp_path / "cert.pem").write_text(identity["cert"])
    (tmp_path / "key.pem").write_text(identity["key"])
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tmp_path / "cert.pem", tmp_path / "key.pem")
    server = HTTPServer(("127.0.0.1", 0), _Status)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"https://127.0.0.1:{server.server_address[1]}", identity
    server.shutdown()


def test_training_client_pins_the_attempt_certificate(tls_status_server):
    from flash.custom.client import TrainingClient
    from flash.custom.managed import session_tls

    url, identity = tls_status_server
    client = TrainingClient(url, api_key="token", timeout=10, tls_cert=identity["cert"])
    assert (client.session_id, client.step) == ("s1", 4)
    # another attempt's (or anyone else's) certificate is not trusted.
    with pytest.raises(ClientError):
        TrainingClient(url, api_key="token", timeout=10, tls_cert=session_tls("run-a", 4)["cert"])
    # and no public-CA fallback without a pin.
    with pytest.raises(ClientError):
        TrainingClient(url, api_key="token", timeout=10)


# --- Nebius launch and connection -----------------------------------------------------------


def _payload(algorithm):
    return {"job_spec_json": json.dumps({"algorithm": algorithm}), "run_id": "r", "attempt": 0}


def test_cloud_init_publishes_the_session_port_only_for_custom():
    from flash.providers.nebius import api

    deadline = time.time() + 3600
    custom = api.cloud_init(_payload("custom"), {"url": "x"}, "img", deadline_at=deadline)
    sft = api.cloud_init(_payload("sft"), {"url": "x"}, "img", deadline_at=deadline)
    assert '"-p", "8000:8000"' in custom
    assert '"-p"' not in sft


def test_public_ip_reads_only_a_global_address(monkeypatch):
    from flash.providers.nebius import api

    def instances(address):
        return [
            {
                "metadata": {"id": "computeinstance-a"},
                "status": {
                    "state": "RUNNING",
                    "network_interfaces": [{"public_ip_address": {"address": address}}],
                },
            }
        ]

    monkeypatch.setattr(api, "list_instances", lambda **_: instances("8.8.4.4/32"))
    assert api.public_ip("computeinstance-a") == "8.8.4.4"
    assert api.public_ip("computeinstance-other") is None
    monkeypatch.setattr(api, "list_instances", lambda **_: instances("10.0.0.5/32"))
    assert api.public_ip("computeinstance-a") is None
    monkeypatch.setattr(api, "list_instances", lambda **_: instances(""))
    assert api.public_ip("computeinstance-a") is None


def _spot_remote(run_id="spot-run", attempt=2):
    return {
        "provider": "spot",
        "instance_id": "computeinstance-a",
        "name": "flash-spot-run-2",
        "run_id": run_id,
        "hf_repo": "owner/repo",
        "phase": "custom",
        "gpu_count": 1,
        "scope": "scope",
        "gpu": "H100",
        "hourly_usd": 1.0,
        "attempt": attempt,
        "started_ts": 1.0,
    }


def _route_client(monkeypatch, remote, *, address="8.8.4.4"):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from flash.providers.nebius import api, handles
    from flash.server.platform.deps import require_key
    from flash.server.routes import custom

    app = FastAPI()
    app.include_router(custom.router)
    app.dependency_overrides[require_key] = lambda: {"id": 1}
    status = SimpleNamespace(spec={"algorithm": "custom"}, state="running", remote=remote)
    monkeypatch.setattr(custom, "require_owned_run", lambda run_id, key: status)
    monkeypatch.setattr(handles, "require_scope", lambda handle: None)
    monkeypatch.setattr(api, "public_ip", lambda instance_id: address)
    return TestClient(app)


def test_spot_connection_returns_direct_url_token_and_pinned_certificate(monkeypatch):
    from flash.custom.managed import session_tls, session_token

    response = _route_client(monkeypatch, _spot_remote()).get("/v1/runs/spot-run/custom")
    assert response.status_code == 200
    body = response.json()
    assert body["url"] == "https://8.8.4.4:8000"
    assert body["api_key"] == session_token("spot-run", 2)
    assert body["tls_cert"] == session_tls("spot-run", 2)["cert"]
    assert "key" not in body
    assert "PRIVATE" not in json.dumps(body)


def test_spot_connection_waits_for_an_address_and_rejects_foreign_attempts(monkeypatch):
    client = _route_client(monkeypatch, _spot_remote(), address=None)
    assert client.get("/v1/runs/spot-run/custom").status_code == 425
    foreign = _route_client(monkeypatch, _spot_remote(run_id="other-run"))
    assert foreign.get("/v1/runs/spot-run/custom").status_code == 409


# --- worker saves ---------------------------------------------------------------------------


def test_spot_worker_saves_on_the_recovery_clock_and_modal_on_its_interval():
    from flash.core.spec import TrainSpec
    from flash.custom.worker import checkpoint_due
    from flash.engine.worker.train.core.child.checkpoint_schedule import CheckpointClock

    now = [0.0]
    clock = CheckpointClock(clock=lambda: now[0], target_seconds=600)
    train = TrainSpec(save_every=None, save_at_steps=[7], max_steps=10)
    saves = []
    for step in range(1, 11):
        now[0] += 250  # 250 s per step: save before a third step would exceed 600 s
        if checkpoint_due(train, clock, step, 10):
            saves.append(step)
            clock.saved()
    assert saves[0] == 1
    assert 7 in saves
    assert saves[-1] == 10
    assert all(b - a <= 2 for a, b in itertools.pairwise(saves))
    modal = TrainSpec(save_every=4, save_at_steps=[3], max_steps=10)
    assert [s for s in range(1, 11) if checkpoint_due(modal, None, s, 10)] == [3, 4, 8, 10]


# --- controller recovery --------------------------------------------------------------------


def test_replacement_session_only_reconnects_to_a_new_attempt():
    from flash.custom.controller import replacement_session

    previous = SimpleNamespace(session_id="old")
    same = SimpleNamespace(session_id="old")
    new = SimpleNamespace(session_id="new")
    assert replacement_session(None, "r", previous, timeout=1, connect=lambda *a, **k: same) is None
    assert replacement_session(None, "r", previous, timeout=1, connect=lambda *a, **k: new) is new
    ended = Mock(side_effect=ClientError("custom run ended"))
    assert replacement_session(None, "r", previous, timeout=1, connect=ended) is None


# --- billing across attempts ----------------------------------------------------------------


def test_checkpoint_heartbeat_keeps_the_attempts_session_time(monkeypatch, tmp_path):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = spec_from_dict(custom_raw(spot=True))
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id, state="running", spec=spec.to_dict(), created_at=time.time()
        )
    )
    beat = runner_status.record_heartbeat
    beat(spec.run_id, {"stage": "custom_step", "attempt": 0, "session_seconds": 900.0})
    beat(spec.run_id, {"stage": "checkpoint_deployable", "attempt": 0})  # no session_seconds
    beat(spec.run_id, {"stage": "custom_model_load", "attempt": 1})
    status = runner_status.get_status(spec.run_id)
    assert status.last_heartbeat["session_seconds_prior"] == pytest.approx(900.0)


def test_transient_checkpoint_discovery_failure_retries_the_attempt():
    from flash.custom.worker import _latest_checkpoint_or_retry
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    def timeout(spec):
        raise TimeoutError("storage timed out")

    def broken(spec):
        raise ValueError("not a storage error")

    with pytest.raises(RetriableInfraError, match="transiently"):
        _latest_checkpoint_or_retry(timeout, None)
    with pytest.raises(ValueError, match="not a storage error"):
        _latest_checkpoint_or_retry(broken, None)
    assert _latest_checkpoint_or_retry(lambda spec: None, None) is None


def test_custom_session_time_accumulates_across_spot_attempts(monkeypatch, tmp_path):
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.lifecycle.status as runner_status
    from flash.custom.billing import session_charge

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = spec_from_dict(custom_raw(spot=True))
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id, state="running", spec=spec.to_dict(), created_at=time.time()
        )
    )
    beat = runner_status.record_heartbeat
    beat(spec.run_id, {"stage": "custom_step", "attempt": 0, "session_seconds": 900.0})
    beat(spec.run_id, {"stage": "custom_waiting", "attempt": 1, "session_seconds": 0})
    beat(spec.run_id, {"stage": "custom_step", "attempt": 1, "session_seconds": 300.0})
    beat(spec.run_id, {"stage": "custom_waiting", "attempt": 2, "session_seconds": 0})
    status = runner_status.get_status(spec.run_id)
    assert status.last_heartbeat["session_seconds_prior"] == pytest.approx(1200.0)

    monkeypatch.setattr(runner_state, "artifacts_dir", lambda _: str(tmp_path))
    (tmp_path / "metrics.json").write_text(json.dumps({"session_seconds": 600.0}))
    status.estimated_cost_usd = 36.0
    wall = spec.custom["max_wall_seconds"]
    assert session_charge(status, spec) == pytest.approx(36.0 * min(1.0, 1800.0 / wall))


def test_spot_first_connection_waits_for_the_run_deadline_not_twenty_minutes():
    from flash.custom.controller import first_connect_timeout

    custom = {"file": "a.py", "max_wall_seconds": 7200}
    modal = spec_from_dict({**custom_raw(), "custom": custom})
    spot = spec_from_dict({**custom_raw(spot=True), "custom": custom})
    assert first_connect_timeout(modal) == 1200
    assert first_connect_timeout(spot) == 7200

"""Managed custom config, async ordering and durable publication contracts."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flash.custom.async_client import AsyncTrainingClient
from flash.schema import ConfigError, spec_from_dict


def custom_raw():
    return {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "custom",
        "project": "11111111-1111-4111-8111-111111111111",
        "custom": {"file": "algorithm.py"},
        "train": {"max_steps": 6},
    }


def test_custom_allocation_uses_shared_capacity_lookup_and_memory_fit(monkeypatch):
    from flash.cost.spec import estimate_for_spec
    from flash.providers.core import allocator
    from flash.providers.core.base import Candidate, UnsupportedGpuError

    seen = []

    def candidates(need, constraints):
        seen.append((need, constraints))
        return [
            # cheapest, but too small for Qwen3.5-9B custom training.
            Candidate("modal", "L4", 0.5, 24),
            Candidate("modal", "A100 SXM", 1.0, 80),
            Candidate("modal", "H100", 4.0, 80),
        ]

    monkeypatch.setattr(allocator, "available_providers", lambda: ("modal",))
    provider = SimpleNamespace(live_candidates=candidates)
    monkeypatch.setattr(allocator, "get_provider", lambda name: provider)
    spec = spec_from_dict(custom_raw())
    assert spec.gpu.type == ""  # no authored GPU pin; shared sizing limits the pool.
    result = allocator.allocate(spec.model, spec.algorithm, train=spec.train)
    assert result.gpu == "A100 SXM"
    assert result.hourly_usd == 1.0
    assert seen[0][1].max_gpu_count == 1
    assert seen[0][1].gpu_type == ""  # no synthetic pin from a per-model GPU list
    assert estimate_for_spec(spec, allocation=result).total_usd == 1.0
    provider.live_candidates = lambda *a: []
    with pytest.raises(UnsupportedGpuError):
        allocator.allocate(spec.model, spec.algorithm, train=spec.train)


def test_custom_config_uses_common_gpu_syntax_and_lora_validation():
    raw = custom_raw()
    raw["gpu"] = {"type": ["H100"], "provider": " MODAL "}
    spec = spec_from_dict(raw)
    assert spec.gpu.type == "H100"
    assert spec.gpu.provider == "modal"
    raw["train"]["lora_rank"] = 0
    with pytest.raises(ConfigError, match="lora_rank"):
        spec_from_dict(raw)


def test_custom_config_roundtrip_and_default_specs_keep_their_bytes():
    from flash.client.specs import spec_payload
    from flash.core.spec import JobSpec

    spec = spec_from_dict(custom_raw())
    assert spec.algorithm == "custom"
    assert not spec.environment.id
    assert spec.gpu.max_retries == 0
    assert spec_from_dict(spec_payload(spec)).to_dict() == spec.to_dict()
    assert JobSpec.from_dict(spec.to_internal_dict()) == spec
    assert "custom" not in JobSpec().to_internal_dict()
    assert "custom" not in JobSpec().to_dict()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("custom", "resolved_resume", {"repo": "someone/else"}),
        ("custom", "file", ""),
        ("custom", "resume_from", "other/final"),
        ("custom", "max_wall_seconds", True),
        ("train", "learning_rate", 0.01),
        ("train", "max_steps", 0),
        ("gpu", "count", 2),
        ("gpu", "type", "A10G"),
    ],
)
def test_custom_rejects_unsupported_config_before_allocating(section, key, value):
    raw = custom_raw()
    raw.setdefault(section, {})[key] = value
    with pytest.raises(ConfigError):
        spec_from_dict(raw)


def test_custom_resume_resolution_is_never_public():
    from dataclasses import replace

    from flash.core.spec import JobSpec

    spec = spec_from_dict(custom_raw())
    spec = replace(spec, custom={**spec.custom, "resolved_resume": {"repo": "private/repo"}})
    assert "resolved_resume" not in spec.to_dict()["custom"]
    assert JobSpec.from_dict(spec.to_internal_dict()).custom["resolved_resume"] == {
        "repo": "private/repo"
    }


def test_async_queue_is_nonblocking_and_cancellation_does_not_overtake_update():
    started, release = threading.Event(), threading.Event()
    calls = []

    class Client:
        def optim_step(self, **kwargs):
            calls.append("update-start")
            started.set()
            release.wait(5)
            calls.append("update-end")
            return {"step": 1}

        def forward(self, data):
            calls.append("forward")
            return {"version": 1}

        def close(self):
            calls.append("close")

    async def scenario():
        client = AsyncTrainingClient(Client())
        update = asyncio.create_task(client.optim_step(learning_rate=0.1))
        assert await asyncio.to_thread(started.wait, 2)
        update.cancel()
        with pytest.raises(asyncio.CancelledError):
            await update
        forward = asyncio.create_task(client.forward([]))
        await asyncio.sleep(0)
        assert calls == ["update-start"]
        release.set()
        assert await forward == {"version": 1}
        await client.close()
        assert calls == ["update-start", "update-end", "forward", "close"]

    asyncio.run(scenario())


def test_connection_capability_requires_run_ownership_before_provider_lookup(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from flash.providers.modal import api
    from flash.server.platform import db
    from flash.server.platform.deps import require_key
    from flash.server.routes.custom import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_key] = lambda: {"id": 2}
    monkeypatch.setattr(db, "run_owner", lambda run_id: 1)
    lookup = Mock(side_effect=AssertionError("must not disclose another run's tunnel"))
    monkeypatch.setattr(api, "custom_tunnel", lookup)
    assert TestClient(app).get("/v1/runs/private-run/custom").status_code == 404
    lookup.assert_not_called()


def test_resume_checks_source_access_before_reading_artifacts(monkeypatch):
    from flash.custom.managed import resolve_resume
    from flash.runner.lifecycle import preparation, status

    raw = custom_raw()
    raw["custom"]["resume_from"] = "private-run/step-2"
    spec = spec_from_dict(raw)
    monkeypatch.setattr(status, "get_status", lambda _: SimpleNamespace())
    monkeypatch.setattr(preparation, "_warmstart_source_is_authorized", lambda *a, **k: False)
    with pytest.raises(ValueError, match="not accessible"):
        resolve_resume(spec, owner_org_id="another-org", owner_key_id=2)


def test_custom_billing_uses_session_time_including_zero_update_waiting(tmp_path, monkeypatch):
    from flash.custom.billing import session_charge
    from flash.runner.accounting.costs import cancelled_charge_usd
    from flash.runner.lifecycle import state

    spec = spec_from_dict(custom_raw())
    monkeypatch.setattr(state, "artifacts_dir", lambda _: str(tmp_path))
    status = SimpleNamespace(
        last_heartbeat={"session_started_at": 1000},
        updated_at=1100,
        estimated_cost_usd=36.0,
    )
    assert cancelled_charge_usd(status, spec, steps=0) == pytest.approx(1.0)
    status.finished_at = 1050
    status.updated_at = 9000
    assert cancelled_charge_usd(status, spec, steps=0) == pytest.approx(0.5)
    (tmp_path / "metrics.json").write_text('{"session_seconds": 50}')
    assert session_charge(status, spec) == pytest.approx(0.5)
    (tmp_path / "metrics.json").write_text('{"session_seconds": 99999}')
    assert session_charge(status, spec) == 36.0


def test_async_ambiguous_failure_blocks_queued_updates_until_explicit_retry():
    class Client:
        pending = False
        updates = 0

        def optim_step(self, **kwargs):
            if self.pending:
                raise RuntimeError("outcome unknown")
            self.pending = True
            self.updates += 1
            raise TimeoutError("response lost")

        def retry_pending(self):
            self.pending = False
            return {"step": self.updates}

        def close(self):
            pass

    async def scenario():
        raw = Client()
        async with AsyncTrainingClient(raw) as client:
            with pytest.raises(TimeoutError):
                await client.optim_step(learning_rate=0.1)
            with pytest.raises(RuntimeError, match="unknown"):
                await client.optim_step(learning_rate=0.1)
            assert await client.retry_pending() == {"step": 1}
            assert raw.updates == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("http_status", [None, 400, 500, 502, 503])
def test_submission_error_reconciles_exact_controller_without_a_second_create(http_status):
    from flash.client.http import ApiError, ClientError
    from flash.custom.controller import submit_custom

    api = Mock(timeout=60)
    captured = []

    def created_but_lost(payload, **kwargs):
        captured.append(payload)
        if http_status is not None:
            raise ApiError(http_status, "submission response failed")
        raise ClientError("response lost")

    api.create_run.side_effect = created_but_lost
    api._request.side_effect = lambda *a, **k: {
        "runs": [
            {"run_id": "someone-else", "spec": {"custom": {"controller_id": "other"}}},
            {"run_id": "created-once", "spec": captured[0]},
        ]
    }
    result = submit_custom(api, custom_raw())
    assert result["run_id"] == "created-once"
    assert api.create_run.call_count == 1
    assert api.timeout == 300


def test_explicit_not_created_submission_error_does_not_reconcile():
    from flash.client.http import ApiError
    from flash.custom.controller import submit_custom

    api = Mock(timeout=60)
    error = ApiError(400, "invalid config", submission_outcome="not-created")
    api.create_run.side_effect = error
    with pytest.raises(ApiError) as caught:
        submit_custom(api, custom_raw())
    assert caught.value is error
    api._request.assert_not_called()
    assert api.create_run.call_count == 1


@pytest.mark.parametrize("outcome", [None, "not-created"])
def test_submission_outcome_survives_http_error_translation(monkeypatch, outcome):
    import io
    from email.message import Message
    from urllib.error import HTTPError

    from flash.client import http

    headers = Message()
    if outcome:
        headers["X-Flash-Submission-Outcome"] = outcome

    def fail(*args, **kwargs):
        raise HTTPError(
            "https://flash.example/v1/runs",
            503,
            "unavailable",
            headers,
            io.BytesIO(b'{"detail": "unavailable"}'),
        )

    monkeypatch.setattr(http, "_urlopen_no_redirect", fail)
    with pytest.raises(http.ApiError) as caught:
        http.ApiClient("https://flash.example")._request("POST", "/v1/runs", {})
    assert caught.value.submission_outcome == outcome


def test_custom_adapter_uses_existing_checkpoint_listing_and_storage_reference():
    from dataclasses import replace

    from flash.runner.results.checkpoints import _checkpoints_from_files
    from flash.schema import parse_adapter_storage_ref

    spec = spec_from_dict(custom_raw(), run_id="custom-test")
    spec = replace(spec, train=replace(spec.train, hf_repo="owner/private-artifacts"))
    prefix = "custom/custom-test/checkpoints/step-2"
    rows = _checkpoints_from_files(
        spec,
        [
            f"{prefix}/adapter/adapter_config.json",
            f"{prefix}/adapter/adapter_model.safetensors",
            f"{prefix}/state.pt",
            f"{prefix}/metadata.json",
        ],
    )
    assert rows[0]["step"] == 2
    assert rows[0]["adapter_prefix"] == prefix
    assert parse_adapter_storage_ref(f"owner/private-artifacts:{prefix}") == (
        "owner/private-artifacts",
        prefix,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_train_command_runs_local_sync_or_async_file(tmp_path, monkeypatch, asynchronous):
    from flash.cli.commands.ops.train import cmd_train
    from flash.custom import controller

    keyword = "async " if asynchronous else ""
    await_keyword = "await " if asynchronous else ""
    (tmp_path / "algorithm.py").write_text(
        f"{keyword}def main(client):\n    {await_keyword}client.optim_step(learning_rate=0.01)\n"
    )
    config = tmp_path / "run.toml"
    config.write_text(
        'project = "11111111-1111-4111-8111-111111111111"\n'
        'model = "Qwen/Qwen3.5-9B"\nalgorithm = "custom"\n[custom]\nfile = "algorithm.py"\n'
    )
    api = Mock(timeout=60)
    api.create_run.return_value = {"run_id": "new-custom-run"}
    api.health.return_value = {"capabilities": ["custom_training_v1"]}
    client = Mock()
    monkeypatch.setattr(controller, "client_from_config", lambda: api)
    monkeypatch.setattr(controller, "connect", lambda *a, **k: client)
    monkeypatch.setattr("flash.cli.commands.ops.runs._follow_run", lambda *a: 0)
    args = SimpleNamespace(
        config=str(config),
        cost=False,
        dry_run=False,
        background=False,
        overrides=[],
        extra_configs=[],
    )
    assert cmd_train(args) == 0
    client.optim_step.assert_called_once_with(learning_rate=0.01)
    client.close.assert_called_once()
    api.cancel_run.assert_not_called()
    assert api.create_run.call_args.kwargs["client_train_schema"]["fields"]


def test_custom_preview_never_executes_user_code(tmp_path, monkeypatch):
    from flash.custom import controller

    (tmp_path / "algorithm.py").write_text(
        'raise AssertionError("must not execute during a quote")'
    )
    api = Mock(timeout=60)
    api.health.return_value = {"capabilities": ["custom_training_v1"]}
    api.create_run.return_value = {"state": "dry_run", "estimated_cost_usd": 4.0}
    monkeypatch.setattr(controller, "client_from_config", lambda: api)
    args = SimpleNamespace(
        config=str(tmp_path / "run.toml"), cost=True, dry_run=False, background=False
    )
    assert controller.run_custom(args, spec_from_dict(custom_raw()), set()) == 0
    assert api.create_run.call_args.kwargs["dry_run"] is True


def test_custom_rejects_old_deployed_server_before_importing_script_or_submitting(
    tmp_path, monkeypatch
):
    from flash.client.http import ClientError
    from flash.custom import controller

    (tmp_path / "algorithm.py").write_text('raise AssertionError("must not execute")')
    api = Mock(timeout=60)
    api.health.return_value = {"capabilities": []}
    monkeypatch.setattr(controller, "client_from_config", lambda: api)
    args = SimpleNamespace(config=str(tmp_path / "run.toml"), cost=False, dry_run=False)
    with pytest.raises(ClientError, match="Deploy a server release"):
        controller.run_custom(args, spec_from_dict(custom_raw()), set())
    api.create_run.assert_not_called()


def test_every_catalog_model_supports_custom_training(monkeypatch):
    from dataclasses import replace

    from flash.core.catalog import MODELS, public_model_rows, validate_model_for_algorithm

    model = replace(MODELS["Qwen/Qwen3.5-9B"], id="test/new-catalog-model")
    monkeypatch.setitem(MODELS, model.id, model)
    models = [row["id"] for row in public_model_rows() if "custom" in row["algos"]]
    assert set(models) == set(MODELS)
    for model_id, info in MODELS.items():
        assert validate_model_for_algorithm(model_id, "custom") == info

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import flash.providers.preflight as preflight_mod
import flash.server.app as app_mod
import flash.server.billing_retry as billing_retry_mod
import flash.server.db as db_mod
import flash.server.reconcile as reconcile_mod


def _forbidden(name: str):
    def fail(*args, **kwargs):
        raise AssertionError(f"local control plane called {name}")

    return fail


def test_local_control_plane_skips_automatic_operations_and_keeps_routes(
    monkeypatch, tmp_path
) -> None:
    calls: list[str] = []
    submitted: list[dict] = []

    monkeypatch.setenv("FLASH_LOCAL_CONTROL_PLANE", "1")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-a,rp-b")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "local-internal")
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setattr(preflight_mod, "check_run_preflight", lambda: calls.append("preflight"))
    monkeypatch.setattr(app_mod, "recover_runs", _forbidden("recover_runs"))
    monkeypatch.setattr(app_mod, "_charge_retry_startup", _forbidden("startup charge retry"))
    monkeypatch.setattr(app_mod, "_reconcile_cost_loop", _forbidden("cost reconcile"))
    monkeypatch.setattr(app_mod, "_charge_retry_loop", _forbidden("periodic charge retry"))
    monkeypatch.setattr(app_mod, "_reap_idle_endpoints_loop", _forbidden("runpod reaper"))
    monkeypatch.setattr(app_mod, "_sweep_orphan_instances_loop", _forbidden("lambda sweeper"))
    monkeypatch.setattr(
        app_mod, "_instance_providers_configured", _forbidden("provider discovery")
    )
    monkeypatch.setattr(
        billing_retry_mod, "charge_retry_enabled", _forbidden("charge retry gate")
    )
    monkeypatch.setattr(reconcile_mod, "reconcile_enabled", _forbidden("reconcile gate"))

    import flash.providers.runpod.train.endpoints as endpoint_mod

    monkeypatch.setattr(
        endpoint_mod, "reconcile_endpoint_slots", _forbidden("endpoint slot reconcile")
    )

    def submit_job(spec, **kwargs):
        submitted.append(kwargs)
        return SimpleNamespace(
            to_dict=lambda: {
                "run_id": spec.run_id,
                "state": "queued",
                "spec": spec.to_dict(),
            }
        )

    monkeypatch.setattr(app_mod, "submit_job", submit_job)

    with TestClient(app_mod.create_app()) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200
        response = client.post(
            "/v1/runs",
            headers={"Authorization": "Bearer local-internal"},
            json={
                "dry_run": True,
                "spec": {
                    "model": "Qwen/Qwen3.5-4B",
                    "algorithm": "grpo",
                    "environment": {"id": "freesolo/gsm8k"},
                    "train": {"max_steps": 1, "hf_repo": "org/test-runs"},
                    "gpu": {"type": "RTX 5090"},
                },
            },
        )

    assert response.status_code == 200, response.text
    assert len(submitted) == 1
    assert submitted[0]["dry_run"] is True
    assert submitted[0]["background"] is True
    assert calls == ["preflight"]


@pytest.mark.parametrize("local_value", [None, "true"])
def test_automatic_operations_remain_enabled_outside_exact_local_mode(
    monkeypatch, local_value
) -> None:
    calls: list[str] = []

    if local_value is None:
        monkeypatch.delenv("FLASH_LOCAL_CONTROL_PLANE", raising=False)
    else:
        monkeypatch.setenv("FLASH_LOCAL_CONTROL_PLANE", local_value)
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-a,rp-b")
    monkeypatch.setattr(preflight_mod, "check_run_preflight", lambda: calls.append("preflight"))
    monkeypatch.setattr(app_mod, "recover_runs", lambda: calls.append("recover_runs"))
    monkeypatch.setattr(billing_retry_mod, "charge_retry_enabled", lambda: True)
    monkeypatch.setattr(reconcile_mod, "reconcile_enabled", lambda: True)
    monkeypatch.setattr(app_mod, "_instance_providers_configured", lambda: True)

    import flash.providers.runpod.train.endpoints as endpoint_mod

    monkeypatch.setattr(
        endpoint_mod,
        "reconcile_endpoint_slots",
        lambda: calls.append("endpoint slot reconcile"),
    )

    def background(name: str):
        calls.append(name)

        async def wait_forever() -> None:
            await asyncio.Event().wait()

        return wait_forever()

    monkeypatch.setattr(
        app_mod, "_charge_retry_startup", lambda: background("startup charge retry")
    )
    monkeypatch.setattr(app_mod, "_reconcile_cost_loop", lambda: background("cost reconcile"))
    monkeypatch.setattr(
        app_mod, "_charge_retry_loop", lambda: background("periodic charge retry")
    )
    monkeypatch.setattr(
        app_mod, "_reap_idle_endpoints_loop", lambda: background("runpod reaper")
    )
    monkeypatch.setattr(
        app_mod, "_sweep_orphan_instances_loop", lambda: background("lambda sweeper")
    )

    with TestClient(app_mod.create_app()) as client:
        assert client.get("/v1/health").status_code == 200

    assert calls == [
        "preflight",
        "recover_runs",
        "startup charge retry",
        "endpoint slot reconcile",
        "cost reconcile",
        "periodic charge retry",
        "runpod reaper",
        "lambda sweeper",
    ]

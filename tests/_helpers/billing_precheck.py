"""Shared fixtures for hosted billing precheck tests."""

from __future__ import annotations

import importlib
import json
from contextlib import contextmanager
from typing import Any


class PrecheckResponse:
    def __init__(self, status: int, body: object) -> None:
        self.status = status
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def precheck_http_error(status: int, *, detail: str = "private upstream detail"):
    from flash.server.billing import charges as billing

    disposition = billing.classify_precheck_http_status(status)
    return billing.PrecheckError(
        source=billing.PrecheckFailureSource.HTTP,
        retry=(
            billing.PrecheckRetryDisposition.FAIL_OPEN
            if disposition is billing.PrecheckHttpDisposition.FAIL_OPEN
            else billing.PrecheckRetryDisposition.BLOCK
        ),
        status_code=status,
        public_detail=detail if status == 402 else None,
        private_detail=detail,
    )


def sabotage_submission_boundaries(monkeypatch) -> list[str]:
    import flash.providers.core.allocator as allocator
    import flash.providers.lambda_.client.api as lambda_api
    import flash.providers.runpod.client.api as runpod_api
    import flash.providers.vast.client.api as vast_api
    import flash.runner.lifecycle.state as runner_state
    import flash.server.asgi.app as app_mod
    import flash.server.platform.db as db_mod

    events: list[str] = []

    def block(name):
        def blocked(*_args, **_kwargs):
            events.append(name)
            raise AssertionError(f"{name} must not run")

        return blocked

    monkeypatch.setattr(db_mod, "record_run", block("ownership"))
    monkeypatch.setattr(db_mod, "delete_run", block("ownership cleanup"))
    monkeypatch.setattr(runner_state, "_save_status", block("persistence"))
    monkeypatch.setattr(app_mod, "submit_job", block("submission"))
    monkeypatch.setattr(allocator, "allocate", block("allocation"))
    monkeypatch.setattr(runpod_api, "submit_job", block("runpod create"))
    monkeypatch.setattr(lambda_api, "launch_instance", block("lambda create"))
    monkeypatch.setattr(vast_api, "create_instance", block("vast create"))
    return events


@contextmanager
def billing_api(
    tmp_path,
    monkeypatch,
    *,
    source_snapshot: dict[str, Any],
    user_prefix: str,
    identity_for_token,
):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test,rp-test-2")
    monkeypatch.setenv("LAMBDA_API_KEY", "lam-test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")

    import flash.providers._lifecycle.net.worker as provider_worker
    import flash.providers.runpod.client.auth as runpod_keys
    import flash.runner.lifecycle.state as runner_state
    import flash.runner.supervise.lifecycle as runner_lifecycle
    import flash.server.platform.auth as auth_mod
    import flash.server.platform.db as db_mod

    runpod_keys.reset()
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(provider_worker, "publish_source_snapshot", lambda _repo: source_snapshot)
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setattr(runner_lifecycle, "_run_job", lambda *a, **k: None)

    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)
    from fastapi.testclient import TestClient

    import flash.server.billing.charges as billing_mod
    import flash.server.domain.registry.environment_registry as environment_registry_mod
    import flash.server.domain.registry.projects as projects_mod
    import flash.server.domain.registry.runs as runs
    from flash.providers.core import registry as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", list, raising=False)
    monkeypatch.setattr(
        projects_mod,
        "require_project_access",
        lambda *, project_id, **_kwargs: project_id,
    )
    monkeypatch.setattr(
        environment_registry_mod,
        "require_environment_project",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(runs, "_post", lambda *a, **k: False, raising=False)
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token.startswith(user_prefix))
    monkeypatch.setattr(auth_mod, "_cached_identity", identity_for_token)
    monkeypatch.setattr(billing_mod, "precheck_training_run", lambda **k: {"ok": True})
    with TestClient(app_mod.create_app()) as client:
        monkeypatch.delenv("GITHUB_TOKEN")
        yield client

"""Control-plane API: freesolo bearer auth, multi-tenant isolation (CPU-only).

User auth is freesolo API keys only (no native key system). Tests run offline: the `api` fixture
monkeypatches ``auth._freesolo_verify`` to accept any token shaped like a freesolo user key, so each
distinct token resolves to its own run-ownership identity via ``db.ensure_external_key``.
"""

from __future__ import annotations

import importlib
import itertools
import json
import os
import sqlite3
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

import flash.core.spec as runner_spec
import flash.runner.accounting.reconciliation as runner_reconciliation
import flash.runner.lifecycle.preparation as runner_preparation
import flash.runner.lifecycle.reporting as runner_reporting
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
import flash.runner.results.verified_revisions as runner_verified_revisions
import flash.runner.supervise.lifecycle as runner_lifecycle
import flash.runner.supervise.recovery as runner_recovery
import flash.runner.supervise.transitions as runner_transitions
from flash.server.platform import db as _db_mod
from tests._helpers.chat_provenance import managed_chat_result as _managed_chat_result
from tests._helpers.source_snapshot import valid_source_snapshot

_SOURCE_SNAPSHOT = valid_source_snapshot()

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def _env_package_b64() -> str:
    """A real base64 `.tar.gz` holding `environment.py`.

    The route validates and decodes the package before it decides anything else, so a publish
    test cannot pass a placeholder string: stubbing `publish_package` skips the upload, not the
    input contract. Built here rather than hardcoded so it stays a genuinely valid archive.
    """
    import base64
    import gzip
    import io
    import tarfile

    # mtime=0 throughout: gzip embeds a timestamp, so the default makes this bytes-unstable
    # between processes. Not a test id today, but a payload that differs per xdist worker is a
    # trap waiting for whoever parametrizes over it.
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        body = b"# test environment\n"
        info = tarfile.TarInfo("environment.py")
        info.size = len(body)
        info.mtime = 0
        tar.addfile(info, io.BytesIO(body))
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    return base64.b64encode(buf.getvalue()).decode()


ENV_PACKAGE_B64 = _env_package_b64()


def stub_publish_package(slug: str, *, record: list | None = None):
    """Return a publish stub that records the route contract when requested."""

    def _publish(*, package_b64, name, key, project_slug):
        if record is not None:
            record.append(
                {
                    "package_b64": package_b64,
                    "name": name,
                    "key": key,
                    "project_slug": project_slug,
                }
            )
        return slug

    return _publish


SPEC = {
    "model": "Qwen/Qwen3.5-9B",
    "project": "11111111-1111-4111-8111-111111111111",
    "algorithm": "grpo",
    # A hub slug, because this fixture drives the HOSTED api: the managed plane accepts
    # `namespace/project/name` only, and a `github:` ref is refused at submit.
    "environment": {"id": "acme/checkout-bot/gsm8k"},
    "train": {"epochs": 1, "max_examples": 1},
    "gpu": {},
}

# Tokens shaped like a verified freesolo user key. The fixture's stub verify accepts any
# token with this prefix, so each distinct one is a distinct authenticated user.
_USER_PREFIX = "fslo-user-"
_counter = itertools.count()


def _bearer(key: str) -> dict:
    headers = {"Authorization": f"Bearer {key}"}
    if key.startswith("fslo-internal"):
        headers["X-Freesolo-Org-Id"] = "org-test"
    return headers


def _login() -> str:
    """A fresh, distinct freesolo user token (accepted by the fixture's stub verify)."""
    return f"{_USER_PREFIX}{next(_counter)}"


def _identity_for_token(token: str) -> dict[str, str]:
    if not token.startswith(_USER_PREFIX):
        return {}
    suffix = token.removeprefix(_USER_PREFIX)
    return {
        "email": f"user-{suffix}@example.com",
        "key_prefix": "fslo_test",
        "org_id": f"org-{suffix}",
        "org_slug": f"org-{suffix}",
    }


@pytest.fixture
def api(tmp_path, monkeypatch):
    # Full operator config so the app's startup preflight passes (>= 2 RunPod accounts + Lambda +
    # the shared tokens + the internal key); see tests/test_preflight.py for the gate.
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test,rp-test-2")
    monkeypatch.setenv("LAMBDA_API_KEY", "lam-test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv("FLASH_DEPLOY_SYNC", "1")
    # runpod.auth caches the parsed pool on first read; reset so the startup preflight reads THIS
    # RUNPOD_API_KEY (the autouse _offline fixture also resets, but make the fixture self-contained).
    import flash.providers.runpod.client.auth as runpod_keys

    runpod_keys.reset()
    import flash.server.domain.registry.environment_registry as environment_registry_mod
    import flash.server.domain.registry.projects as projects_mod
    import flash.server.platform.auth as auth_mod
    import flash.server.platform.db as db_mod

    # The storage roots are fixed constants (not env-configurable); redirect them to tmp for
    # test isolation by patching the module attributes after reload.
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **_k: {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]},
    )
    # The new preflight requires the Lambda key above, which also makes
    # `configured_providers()` treat it as live -- so the startup lifespan's `recover_runs()`
    # and the orphan-sweep loop would dispatch real `sweep_orphans()` (Lambda list calls) and
    # break test hermeticity. These API tests don't exercise orphan reaping, so stub the
    # provider set to empty: preflight still passes on the keys, but startup stays CPU-only
    # with no network.
    import flash.server.domain.registry.runs as runs
    from flash.providers.core import registry as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", list, raising=False)
    # The dummy FREESOLO_INTERNAL_KEY also enables the best-effort backend reporting path: a dry-run
    # /v1/runs submit carries an org_id, so runner_submit.submit_job() -> _report_status() ->
    # runs._post() would urllib-POST the real backend (or wait out its 10s timeout). Stub the
    # single network choke-point so these offline tests stay hermetic (same as the billing fixture).
    monkeypatch.setattr(runs, "_post", lambda *a, **k: False, raising=False)
    # Offline auth: a token is a valid freesolo USER key iff it has the test prefix. This stub
    # replaces the real network verify.
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token.startswith(_USER_PREFIX))
    monkeypatch.setattr(auth_mod, "_cached_identity", _identity_for_token)

    def validate_project(*, project_id, key, authorization, org_id=None):
        assert isinstance(project_id, str)
        assert project_id.strip()
        assert str(authorization or "").startswith("Bearer ")
        if key.get("auth_kind") == "internal":
            assert org_id == "org-test"
        return project_id.strip()

    monkeypatch.setattr(projects_mod, "require_project_access", validate_project)
    monkeypatch.setattr(
        projects_mod,
        "require_project_access_slug",
        lambda **kwargs: (validate_project(**kwargs), "checkout-bot"),
    )
    monkeypatch.setattr(
        environment_registry_mod,
        "require_environment_project",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        environment_registry_mod,
        "record_published_environment",
        lambda **_kwargs: True,
    )
    with TestClient(app_mod.create_app()) as client:
        # the fake token exists only to satisfy startup preflight. leaving it live for requests makes
        # submit-time environment pinning call the real GitHub API with `ghp-test`; tokenless planes
        # deliberately defer that work to the worker. tests that exercise a token set their own.
        monkeypatch.delenv("GITHUB_TOKEN")
        yield client


def test_me(api):
    key = _login()
    me = api.get("/v1/me", headers=_bearer(key))
    assert me.status_code == 200
    # A verified freesolo user key resolves to the Freesolo identity returned by verify.
    assert me.json()["email"] == f"user-{key.removeprefix(_USER_PREFIX)}@example.com"
    assert me.json()["key_prefix"] == "fslo_test"
    assert me.json()["org_slug"] == f"org-{key.removeprefix(_USER_PREFIX)}"


def test_requests_without_key_are_rejected(api):
    assert api.get("/v1/runs").status_code == 401
    # A token that doesn't verify with freesolo is rejected.
    assert api.get("/v1/runs", headers=_bearer("not-a-freesolo-key")).status_code == 401
    assert api.get("/v1/models", headers=_bearer("nope")).status_code == 401
    health = api.get("/v1/health")
    assert health.status_code == 200  # health stays open
    assert health.json()["capabilities"] == []


def test_project_validation_blocks_before_run_preparation(api, monkeypatch) -> None:
    from fastapi import HTTPException

    import flash.server.domain.registry.projects as projects_mod
    import flash.server.routes.runs as runs_route

    monkeypatch.setattr(
        projects_mod,
        "require_project_access",
        lambda **_kwargs: (_ for _ in ()).throw(
            HTTPException(status_code=403, detail="project denied")
        ),
    )
    monkeypatch.setattr(
        runs_route._app,
        "prepare_job",
        lambda *_args, **_kwargs: pytest.fail("project validation must run before preparation"),
    )
    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": SPEC},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "project denied"


def test_environment_project_validation_blocks_before_run_preparation(api, monkeypatch) -> None:
    from fastapi import HTTPException

    import flash.server.domain.registry.environment_registry as registry
    import flash.server.routes.runs as runs_route

    def reject_environment(**kwargs):
        assert kwargs["slug"] == "acme/checkout-bot/my-env"
        assert kwargs["repair_missing"] is True
        raise HTTPException(status_code=409, detail="flash environment belongs to another project")

    monkeypatch.setattr(registry, "require_environment_project", reject_environment)
    monkeypatch.setattr(
        runs_route._app,
        "prepare_job",
        lambda *_args, **_kwargs: pytest.fail(
            "environment project validation must run before preparation"
        ),
    )
    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": {**SPEC, "environment": {"id": "acme/checkout-bot/my-env"}}},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "flash environment belongs to another project"


@pytest.mark.parametrize(
    "environment_id",
    [
        # a repo Freesolo has no relationship with
        "github:acme/envs@main:gsm8k/environment.py",
        "https://github.com/acme/envs/tree/main/gsm8k",
        # the hub's OWN repo, spelled the long way: still refused, so there is exactly one
        # accepted spelling of a hub environment rather than two that must agree.
        "github:freesolo-co/environment-hub@main:acme/checkout-bot/my-env/environment.py",
        "github:FREESOLO-CO/ENVIRONMENT-HUB@main:acme/checkout-bot/my-env/environment.py",
        # references that used to fail closed further downstream as "malformed hub reference";
        # they are now refused earlier, by form, which subsumes that check.
        "github:freesolo-co/environment-hub@dev:acme/checkout-bot/my-env/environment.py",
        "github:freesolo-co/environment-hub@main:acme/checkout-bot/my-env/other.py",
    ],
)
def test_a_github_environment_is_refused_before_validation_or_preparation(
    api, monkeypatch, environment_id
) -> None:
    """The hosted plane runs hub environments only, and refuses the rest at the submit boundary.

    Refusing early is the point: a GitHub ref must not reach project-ownership validation (which
    is meaningless for a repo the plane does not own) nor job preparation (which would fetch and
    run the code). Both are asserted by failing the test if either is called.
    """
    import flash.server.domain.registry.environment_registry as registry
    import flash.server.routes.runs as runs_route

    monkeypatch.setattr(
        registry,
        "require_environment_project",
        lambda **_kwargs: pytest.fail("a github environment must not reach project validation"),
    )
    monkeypatch.setattr(
        runs_route._app,
        "prepare_job",
        lambda *_args, **_kwargs: pytest.fail("a github environment must block preparation"),
    )

    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": {**SPEC, "environment": {"id": environment_id}}},
    )

    assert response.status_code == 400, response.text
    assert "hub" in response.json()["detail"]


def test_a_hub_environment_still_reaches_preparation(api, monkeypatch) -> None:
    """The counterpart: the refusal above is about the FORM, not the parser blocking every run."""
    import flash.server.domain.registry.environment_registry as registry

    seen: list[str] = []
    monkeypatch.setattr(
        registry,
        "require_environment_project",
        lambda **kwargs: seen.append(kwargs["slug"]),
    )

    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": SPEC, "dry_run": True},
    )

    assert response.status_code == 200, response.text
    assert seen == ["acme/checkout-bot/gsm8k"]


@pytest.mark.parametrize(
    "environment_id", ["Acme/Checkout-Bot/My-Env", " acme/checkout-bot/my-env "]
)
def test_run_rejects_noncanonical_managed_environment_before_registry(
    api, monkeypatch, environment_id
) -> None:
    import flash.server.domain.registry.environment_registry as registry
    import flash.server.routes.runs as runs_route

    monkeypatch.setattr(
        registry,
        "require_environment_project",
        lambda **_kwargs: pytest.fail(
            "noncanonical environment must not reach registry validation"
        ),
    )
    monkeypatch.setattr(
        runs_route._app,
        "prepare_job",
        lambda *_args, **_kwargs: pytest.fail("noncanonical environment must block preparation"),
    )

    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": {**SPEC, "environment": {"id": environment_id}}},
    )

    assert response.status_code == 400
    assert "invalid env id segment" in response.json()["detail"]


def test_project_validation_blocks_before_environment_publication(api, monkeypatch) -> None:
    from fastapi import HTTPException

    import flash.server.domain.registry.envs as envs_mod
    import flash.server.domain.registry.projects as projects_mod

    monkeypatch.setattr(
        projects_mod,
        "require_project_access_slug",
        lambda **_kwargs: (_ for _ in ()).throw(
            HTTPException(status_code=403, detail="project denied")
        ),
    )
    monkeypatch.setattr(
        envs_mod,
        "publish_package",
        lambda **_kwargs: pytest.fail("project validation must run before publication"),
    )
    response = api.post(
        "/v1/envs",
        headers=_bearer(_login()),
        json={
            "name": "env",
            "package_b64": ENV_PACKAGE_B64,
            "project_id": "11111111-1111-4111-8111-111111111111",
        },
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "project denied"


def test_canonical_slug_resolution_blocks_before_environment_publication(api, monkeypatch) -> None:
    import flash.server.domain.registry.environment_registry as registry_mod
    import flash.server.domain.registry.envs as envs_mod
    import flash.server.domain.registry.projects as projects_mod

    importlib.reload(projects_mod)
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://freesolo.test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-test")
    project_id = "11111111-1111-4111-8111-111111111111"
    key = _login()
    org_id = f"org-{key.removeprefix(_USER_PREFIX)}"
    validation_calls: list[str] = []
    publish_events: list[str] = []

    class _Response:
        status = 200

        def __init__(self, body: dict):
            self._body = json.dumps(body).encode()

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout=None):
        method = request.get_method()
        validation_calls.append(method)
        if method == "GET":
            return _Response({"id": project_id, "name": "Foo Bar"})
        body = json.loads(request.data)
        assert body == {"orgId": org_id, "projectId": project_id}
        return _Response({"ok": True, **body})

    monkeypatch.setattr(projects_mod.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(
        envs_mod,
        "publish_package",
        lambda **_kwargs: publish_events.append("published"),
    )
    monkeypatch.setattr(
        registry_mod,
        "record_published_environment",
        lambda **_kwargs: publish_events.append("associated"),
    )

    response = api.post(
        "/v1/envs",
        headers=_bearer(key),
        json={"name": "env", "package_b64": ENV_PACKAGE_B64, "project_id": project_id},
    )

    assert publish_events == []
    assert response.status_code == 502
    assert "canonical project slug" in response.json()["detail"]
    assert validation_calls == ["GET", "POST"]


def _install_real_internal_project_validation(monkeypatch):
    import flash.server.domain.registry.projects as projects_mod

    importlib.reload(projects_mod)
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://api.freesolo.co")
    requests: list[dict] = []

    class _Response:
        status = 200

        def __init__(self, body: dict):
            self._body = json.dumps(body).encode()

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout=None):
        body = json.loads(request.data)
        requests.append(
            {
                "method": request.get_method(),
                "url": request.full_url,
                "authorization": request.get_header("Authorization"),
                "body": body,
            }
        )
        return _Response({"ok": True, "projectSlug": "checkout-bot", **body})

    monkeypatch.setattr(projects_mod.urllib.request, "urlopen", urlopen)
    return requests


_INTERNAL_PROJECT_VALIDATE_URL = "https://api.freesolo.co/api/flash/projects/validate/internal"


def _assert_internal_project_request(requests: list[dict]) -> None:
    """Exactly one PROJECT validation call, with the internal key and the run's org.

    Filtered to that endpoint rather than asserting on the whole list: a hub environment also
    triggers an ``environments/use/internal`` call, which is correct and is covered by the
    environment tests. Asserting the full list here would make this project-scoped test fail
    whenever an unrelated internal call is added.
    """
    project_requests = [r for r in requests if r["url"] == _INTERNAL_PROJECT_VALIDATE_URL]
    assert project_requests == [
        {
            "method": "POST",
            "url": _INTERNAL_PROJECT_VALIDATE_URL,
            "authorization": "Bearer fslo-internal-test",
            "body": {"orgId": "org-test", "projectId": SPEC["project"]},
        }
    ]


def test_internal_run_uses_internal_project_validation_endpoint(api, monkeypatch) -> None:
    requests = _install_real_internal_project_validation(monkeypatch)

    response = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": SPEC, "dry_run": True},
    )

    assert response.status_code == 200, response.text
    _assert_internal_project_request(requests)


def test_internal_publish_uses_internal_project_validation_endpoint(api, monkeypatch) -> None:
    import flash.server.domain.registry.environment_registry as registry
    import flash.server.domain.registry.envs as envs_mod

    requests = _install_real_internal_project_validation(monkeypatch)
    monkeypatch.setattr(
        envs_mod, "publish_package", stub_publish_package("org-test/checkout-bot/env")
    )
    monkeypatch.setattr(registry, "record_published_environment", lambda **_kwargs: True)

    response = api.post(
        "/v1/envs",
        headers=_bearer("fslo-internal-test"),
        json={"name": "env", "package_b64": ENV_PACKAGE_B64, "project_id": SPEC["project"]},
    )

    assert response.status_code == 200, response.text
    _assert_internal_project_request(requests)


def test_internal_delete_uses_internal_project_validation_endpoint(api, monkeypatch) -> None:
    import flash.server.domain.registry.environment_registry as registry
    import flash.server.domain.registry.envs as envs_mod

    requests = _install_real_internal_project_validation(monkeypatch)
    monkeypatch.setattr(envs_mod, "delete_package", lambda **_kwargs: True)
    monkeypatch.setattr(registry, "record_deleted_environment", lambda **_kwargs: True)

    response = api.delete(
        "/v1/envs/org-test/checkout-bot/env",
        headers={
            **_bearer("fslo-internal-test"),
            "X-Freesolo-Project-Id": SPEC["project"],
        },
    )

    assert response.status_code == 200, response.text
    _assert_internal_project_request(requests)


def test_dry_run_reports_schema_agreement_without_persisting_it(api) -> None:
    from flash.schema import train_schema_metadata

    metadata = {
        "version": "0.2.56",
        "fields": train_schema_metadata(),
        "authored_keys": sorted(SPEC["train"]),
    }
    response = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": SPEC, "dry_run": True, "client_train_schema": metadata},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["train_schema_compatibility"] == {
        "status": "agreement",
        "client_only": [],
        "server_only": [],
        "introduced_in_differences": [],
    }
    status = api.get(f"/v1/runs/{body['run_id']}", headers=_bearer("fslo-internal-test")).json()
    assert "train_schema_compatibility" not in status


def test_dry_run_schema_disagreement_is_diagnostic_only(api) -> None:
    from flash.schema import train_schema_metadata

    fields = train_schema_metadata()
    fields.pop("teacher_model")
    fields.pop("structured_outputs")
    fields["epochs"] = "0.2.1"
    metadata = {
        "version": "0.2.55",
        "fields": fields,
        "authored_keys": sorted(SPEC["train"]),
    }
    response = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": SPEC, "dry_run": True, "client_train_schema": metadata},
    )

    assert response.status_code == 200, response.text
    compatibility = response.json()["train_schema_compatibility"]
    assert compatibility["status"] == "disagreement"
    assert compatibility["client_only"] == []
    assert compatibility["server_only"] == ["structured_outputs", "teacher_model"]
    assert compatibility["introduced_in_differences"] == [
        {"key": "epochs", "client": "0.2.1", "server": "0.2.0"}
    ]


def test_missing_or_malformed_schema_metadata_does_not_change_parser_acceptance(api) -> None:
    payloads = [
        {"spec": SPEC, "dry_run": True},
        {
            "spec": SPEC,
            "dry_run": True,
            "client_train_schema": {
                "version": "0.2.56",
                "fields": ["not", "a", "mapping"],
                "authored_keys": sorted(SPEC["train"]),
            },
        },
    ]

    for payload in payloads:
        response = api.post("/v1/runs", headers=_bearer("fslo-internal-test"), json=payload)
        assert response.status_code == 200, response.text
        assert "train_schema_compatibility" not in response.json()


def test_dry_run_rejects_semantically_invalid_thinking_json_schema(api, monkeypatch) -> None:
    import flash.server.routes.runs as runs_route

    monkeypatch.setattr(
        runs_route._app,
        "submit_job",
        lambda *_a, **_k: pytest.fail("invalid serving config must fail before submission"),
    )
    spec = {
        **SPEC,
        "thinking": True,
        "train": {
            **SPEC["train"],
            "structured_outputs": {"json": {"type": 7}},
        },
    }

    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": spec, "dry_run": True},
    )

    assert response.status_code == 400
    assert "train.structured_outputs JSON schema is invalid" in response.json()["detail"]


def test_dry_run_rejects_structured_completion_budget_above_serving_capacity(api) -> None:
    spec = {
        **SPEC,
        "thinking": True,
        "train": {
            # the context must still leave a prompt budget, or the grpo spec-parse guard rejects it
            # first and this test would stop covering the serving-capacity message.
            **SPEC["train"],
            "max_context_tokens": 32769,
            "max_completion_tokens": 32513,
            "structured_outputs": {"choice": ["4"]},
        },
    }

    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": spec, "dry_run": True},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "effective budget (32513) cannot fit" in detail
    assert "serving max_model_len=32768" in detail
    assert "lower train.max_completion_tokens to <= 32512" in detail


def test_valid_thinking_structured_run_passes_dry_run(api) -> None:
    spec = {
        **SPEC,
        "thinking": True,
        "train": {
            **SPEC["train"],
            "structured_outputs": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    }

    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": spec, "dry_run": True},
    )

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "dry_run"


@pytest.mark.parametrize(
    ("structured_outputs", "message"),
    [
        ({"regex": "["}, "regex is invalid"),
        (
            {"json": {"$ref": "https://example.invalid/schema.json"}},
            "external schema retrieval is unsupported",
        ),
        (
            {
                "json": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "$dynamicRef": "https://example.invalid/schema.json#answer",
                }
            },
            "external schema retrieval is unsupported",
        ),
        (
            {
                "json": {
                    "$schema": "https://json-schema.org/draft/2019-09/schema",
                    "$recursiveRef": "https://example.invalid/schema.json#",
                }
            },
            "external schema retrieval is unsupported",
        ),
        (
            {"json_object": True, "whitespace_pattern": "["},
            "whitespace_pattern is invalid",
        ),
    ],
)
def test_dry_run_rejects_invalid_structured_serving_constraints(
    api, monkeypatch, structured_outputs, message
) -> None:
    import flash.server.routes.runs as runs_route

    monkeypatch.setattr(
        runs_route._app,
        "submit_job",
        lambda *_a, **_k: pytest.fail("invalid serving config must fail before submission"),
    )
    spec = {
        **SPEC,
        "thinking": True,
        "train": {**SPEC["train"], "structured_outputs": structured_outputs},
    }

    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": spec, "dry_run": True},
    )

    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_warmstart_dry_run_preserves_serving_preflight_error(api) -> None:
    spec = {
        **SPEC,
        "thinking": True,
        "train": {
            **SPEC["train"],
            "init_from_adapter": "source-run/final",
            "structured_outputs": {"json": {"type": 7}},
        },
    }

    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": spec, "dry_run": True},
    )

    assert response.status_code == 400
    assert "train.structured_outputs JSON schema is invalid" in response.json()["detail"]
    assert "could not be prepared" not in response.json()["detail"]


def test_warmstart_dry_run_preserves_context_preflight_error(api) -> None:
    # a warm-start run whose non-structured context preflight fails must surface the SPECIFIC
    # context error, not the generic warm-start "could not be prepared" mask (the context guard
    # runs ahead of adapter resolution, so its ValueError must propagate like structured runs do)
    spec = {
        "project": "11111111-1111-4111-8111-111111111111",
        **SPEC,
        "model": "Qwen/Qwen3.5-9B",
        "train": {
            **SPEC["train"],
            "init_from_adapter": "source-run/final",
            "max_completion_tokens": 34000,
        },
    }

    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": spec, "dry_run": True},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "could not be prepared" not in detail
    assert "serving max_model_len" in detail


@pytest.mark.parametrize(
    "structured_outputs",
    [
        {"regex": r"^[0-9]+$", "whitespace_pattern": r"\s*"},
        {
            "json": {
                "$defs": {"answer": {"type": "string"}},
                "type": "object",
                "properties": {"answer": {"$ref": "#/$defs/answer"}},
            }
        },
        {"json": {"const": {"$ref": "https://example.invalid/is-instance-data"}}},
        {
            "json": {
                "$schema": "http://json-schema.org/draft-04/schema#",
                "type": "object",
            }
        },
    ],
)
def test_dry_run_accepts_valid_regex_and_local_ref_constraints(api, structured_outputs) -> None:
    spec = {
        **SPEC,
        "thinking": True,
        "train": {**SPEC["train"], "structured_outputs": structured_outputs},
    }

    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": spec, "dry_run": True},
    )

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "dry_run"


@pytest.mark.parametrize(
    ("max_completion_tokens", "status_code"),
    [(32000, 400), (2950, 200)],
)
def test_opd_structured_dry_run_checks_rollout_context_before_allocation(
    api, monkeypatch, tmp_path, max_completion_tokens, status_code
) -> None:
    import flash.engine.worker.train.opd.orchestration.validation as opd_validation
    import flash.envs.loading.loader as envs_loader
    import flash.schema as schema
    import flash.server.routes.runs as runs_route
    from tests._helpers.teacher import configure_managed_teacher

    # an opd submission is now validated against the plane's managed-teacher configuration before
    # the run record; this test is about rollout context ordering, so give it a configured plane.
    configure_managed_teacher(monkeypatch)
    monkeypatch.setattr(schema, "provisional_gpu", lambda *_a, **_k: "B200")
    # offline: the structured-OPD preflight resolves model metadata over the network -- geometry
    # (model_info + a config.json download) and list_repo_files, to detect a mistral tokenizer.
    # Without this the dry run reached hf.co and passed only on a connected runner. The values are
    # not what this test asserts on (it checks that an invalid context is rejected BEFORE
    # allocation), so stub the resolver itself rather than pin geometry numbers that read as
    # meaningful; vocab comes from the catalog entry for this model.
    monkeypatch.setattr(
        opd_validation,
        "_resolve_structured_model_metadata",
        lambda *_a, **_k: (151936, ("config.json", "tokenizer.json")),
        raising=False,
    )
    # offline: the valid-context path pins the github env ref to a sha; stub it so the
    # test never makes a real github request (the api fixture only sets a fake token)
    monkeypatch.setattr(envs_loader, "_resolve_ref_sha", lambda *_a, **_k: "0" * 40)
    # the valid-context path also runs the image-opd preflight, which resolves the env
    # reference to inspect its dataset for images. point it at an empty local dir so the
    # preflight finds no packaged dataset and returns without a real github request.
    _offline_env_dir = tmp_path / "env"
    _offline_env_dir.mkdir()
    (_offline_env_dir / "environment.py").write_text("")
    monkeypatch.setattr(
        envs_loader,
        "_resolve_environment_reference",
        lambda *_a, **_k: str(_offline_env_dir / "environment.py"),
    )
    if status_code == 400:
        monkeypatch.setattr(
            runs_route._app,
            "submit_job",
            lambda *_a, **_k: pytest.fail("invalid context must fail before allocation"),
        )
    spec = {
        "project": "11111111-1111-4111-8111-111111111111",
        **SPEC,
        "model": "Qwen/Qwen3.6-35B-A3B",
        "algorithm": "opd",
        "thinking": False,
        "train": {
            **SPEC["train"],
            # the 35b is a gdn hybrid, so opd sizes it with a bf16 kv cache (the worker refuses fp8
            # for them). training the routed experts puts this past every single card at any batch
            # size, so the run is pinned to two below; otherwise the valid-context case would fail
            # allocation instead of exercising the ordering this test is about.
            "prompts_per_step": 4,
            # both cards only JOIN a step wide enough to give each one a share, and the RETAINED pool
            # bounds that width, not the requested batch: `SPEC["train"]` retains one row, which caps
            # the priced step at one prompt however many this asks for, so the second rank holds no
            # share, contributes no memory, and the 213 GB need is refused on one 180 GB card. two
            # retained rows are what make the pin above real.
            "max_examples": 2,
            "max_completion_tokens": max_completion_tokens,
            "structured_outputs": {"choice": ["4"]},
        },
        "gpu": {"count": 2},
    }

    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": spec, "dry_run": True},
    )

    assert response.status_code == status_code, response.text
    if status_code == 400:
        detail = response.json()["detail"]
        assert "OPD rollout prompt+completion)=33024" in detail
        assert "serving max_model_len=32768" in detail
    else:
        assert response.json()["state"] == "dry_run"


def test_grpo_rollout_shape_rejects_before_secrets_persistence_or_submission(
    api, monkeypatch
) -> None:
    import flash.server.routes.runs as runs_route

    monkeypatch.setattr(
        runs_route, "_runtime_secrets", lambda *_a, **_k: pytest.fail("secrets inspected")
    )
    monkeypatch.setattr(runs_route.db, "record_run", lambda *_a, **_k: pytest.fail("run persisted"))
    monkeypatch.setattr(
        runs_route._app, "submit_job", lambda *_a, **_k: pytest.fail("job submitted")
    )
    spec = {
        **SPEC,
        "train": {
            **SPEC["train"],
            "prompts_per_step": 65,
            "group_size": 8,
        },
    }
    response = api.post(
        "/v1/runs",
        headers=_bearer(_login()),
        json={"spec": spec, "dry_run": False},
    )
    assert response.status_code == 400
    assert "prompts_per_step * train.group_size must be <= 512" in response.json()["detail"]


def test_unknown_authored_train_key_enriches_parser_rejection_once(api, monkeypatch) -> None:
    import flash.server.routes.runs as runs_route
    from flash.schema import train_schema_metadata

    original_parse = runs_route._parse_spec
    calls = 0

    def counted_parse(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(runs_route, "_parse_spec", counted_parse)
    monkeypatch.setattr(
        runs_route, "_runtime_secrets", lambda *_a, **_k: pytest.fail("secrets inspected")
    )
    monkeypatch.setattr(runs_route.db, "record_run", lambda *_a, **_k: pytest.fail("run persisted"))
    monkeypatch.setattr(
        runs_route._app, "submit_job", lambda *_a, **_k: pytest.fail("job submitted")
    )
    fields = train_schema_metadata()
    fields["future_knob"] = "0.3.0"
    spec = {**SPEC, "train": {**SPEC["train"], "future_knob": 1}}
    response = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={
            "spec": spec,
            "dry_run": True,
            "client_train_schema": {
                "version": "0.3.0",
                "fields": fields,
                "authored_keys": sorted(spec["train"]),
            },
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "unknown key(s): future_knob" in detail
    assert "future_knob (minimum released Flash version 0.3.0)" in detail
    assert "client/server [train] schemas disagree" in detail
    assert calls == 1
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


def test_malformed_schema_metadata_does_not_enrich_parser_rejection(api) -> None:
    spec = {**SPEC, "train": {**SPEC["train"], "future_knob": 1}}
    response = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={
            "spec": spec,
            "dry_run": True,
            "client_train_schema": {
                "version": "0.3.0",
                "fields": {"future_knob": "0.3.0"},
                "authored_keys": ["future_knob", "future_knob"],
            },
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "unknown key(s): future_knob" in detail
    assert "minimum released Flash version" not in detail
    assert "schemas disagree" not in detail


def test_internal_key_authenticates_as_service_identity(api, monkeypatch):
    # With FREESOLO_INTERNAL_KEY configured, the shared internal key works as a bearer and
    # owns the runs it submits — the freesolo SDK authenticates with the same credential the
    # platform uses. It is matched BEFORE the freesolo user-key verify, so it never hits the
    # backend.
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-secret")
    r = api.post(
        "/v1/runs",
        json={"spec": SPEC, "dry_run": True},
        headers=_bearer("fslo-internal-secret"),
    )
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    # owns its run (run_owner resolves to the provisioned service identity)
    assert api.get(f"/v1/runs/{run_id}", headers=_bearer("fslo-internal-secret")).status_code == 200
    # a token that is neither the internal key nor a verified freesolo key is rejected
    assert api.get(f"/v1/runs/{run_id}", headers=_bearer("wrong-internal")).status_code == 401
    # the internal key is stored hashed, like any other (never persisted in the clear)
    with sqlite3.connect(_db_mod.db_path()) as conn:
        prefixes = [row[0] for row in conn.execute("SELECT key_prefix FROM api_keys").fetchall()]
    assert "internal" in prefixes


def test_internal_key_rejected_when_unconfigured(api):
    # Without FREESOLO_INTERNAL_KEY set, the would-be internal key is just an unknown token
    # that doesn't verify with freesolo and gets 401 — no implicit acceptance.
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-secret")).status_code == 401


def test_freesolo_user_key_authenticates(api, monkeypatch):
    # A user who `flash login`s with a freesolo key sends it as the bearer. With the token
    # verified by the backend it authenticates and resolves to a stable per-token identity
    # (its own run-ownership row).
    import flash.server.platform.auth as auth_mod

    auth_mod._verify_cache.clear()
    calls = {"n": 0}

    def fake_verify(token):
        calls["n"] += 1
        return token == "fslo-user-good"

    monkeypatch.setattr(auth_mod, "_freesolo_verify", fake_verify)
    monkeypatch.setattr(
        auth_mod,
        "_cached_identity",
        lambda token: (
            {"email": "user-good@example.com", "key_prefix": "fslo_good", "org_slug": "acme"}
            if token == "fslo-user-good"
            else {}
        ),
    )

    row = auth_mod.authenticate("Bearer fslo-user-good")
    assert row is not None
    assert row["email"] == "user-good@example.com"
    # An unverified token returns None (401).
    assert auth_mod.authenticate("Bearer fslo-user-bad") is None
    # The same key resolves to the same identity across requests (stable per-token row).
    again = auth_mod.authenticate("Bearer fslo-user-good")
    assert again["id"] == row["id"]


def test_freesolo_user_key_without_org_slug_is_rejected(api, monkeypatch):
    # A verified external key must include an org slug. Do not fall back to email or
    # token-derived namespaces for env publishing.
    import flash.server.platform.auth as auth_mod

    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: True)
    monkeypatch.setattr(
        auth_mod,
        "_cached_identity",
        lambda token: {"email": "user@example.com", "key_prefix": "fslo_noorg"},
    )

    assert auth_mod.authenticate("Bearer fslo-no-org") is None


def test_freesolo_user_key_without_email_authenticates_with_org_slug(api, monkeypatch):
    import flash.server.platform.auth as auth_mod

    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: True)
    monkeypatch.setattr(
        auth_mod,
        "_cached_identity",
        lambda token: {"key_prefix": "fslo_noemail", "org_slug": "acme", "org_id": "org-acme"},
    )

    row = auth_mod.authenticate("Bearer fslo-no-email")
    assert row is not None
    assert row["org_slug"] == "acme"
    assert not row.get("email")


def test_invalid_external_email_is_not_persisted(api, monkeypatch):
    import flash.server.platform.auth as auth_mod

    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: True)
    monkeypatch.setattr(
        auth_mod,
        "_cached_identity",
        lambda token: {
            "email": "not-an-email",
            "key_prefix": "fslo_invalidemail",
            "org_slug": "acme",
            "org_id": "org-acme",
        },
    )
    captured = {}
    ensure_external_key = auth_mod.db.ensure_external_key

    def capture_ensure_external_key(token, *, key_prefix=None, email=None):
        captured["email"] = email
        return ensure_external_key(token, key_prefix=key_prefix, email=email)

    monkeypatch.setattr(auth_mod.db, "ensure_external_key", capture_ensure_external_key)

    assert auth_mod.authenticate("Bearer fslo-invalid-email") is not None
    assert captured["email"] is None


def test_create_run_rejects_authored_warmstart_rank_before_prepare_or_persist(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    calls = {"prepare": 0, "persist": 0}

    def unexpected_prepare(*args, **kwargs):
        calls["prepare"] += 1
        raise AssertionError("prepare_job must not run")

    def unexpected_persist(*args, **kwargs):
        calls["persist"] += 1
        raise AssertionError("record_run must not run")

    monkeypatch.setattr(app_mod, "prepare_job", unexpected_prepare)
    monkeypatch.setattr(app_mod.db, "record_run", unexpected_persist)
    spec = {
        **SPEC,
        "train": {
            **SPEC["train"],
            "init_from_adapter": "source-run/final",
            "lora_rank": 32,
        },
    }

    resp = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": spec},
    )

    assert resp.status_code == 400
    assert (
        resp.json()["detail"]
        == "train.lora_rank cannot be set with train.init_from_adapter because source adapter "
        "rank metadata is authoritative"
    )
    assert calls == {"prepare": 0, "persist": 0}
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


def test_warmstart_dry_run_persists_source_adapter_alpha(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    def prepare(spec, **_kwargs):
        resolved = replace(spec, train=replace(spec.train, lora_alpha=32))
        return runner_submit.PreparedJob(
            public_spec=resolved,
            worker_spec=resolved,
            estimated_cost_usd=1.25,
            adapter_identity=None,
        )

    monkeypatch.setattr(app_mod, "prepare_job", prepare)
    spec = {
        **SPEC,
        "train": {**SPEC["train"], "init_from_adapter": "source-run/final"},
    }

    resp = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": spec, "dry_run": True},
    )

    assert resp.status_code == 200, resp.text
    # a warm start cannot author alpha, so it is stripped from the public spec; the resolved
    # warm-start source alpha is persisted in the internal worker-spec carrier.
    assert "lora_alpha" not in resp.json()["spec"]["train"]
    status = runner_status.get_status(resp.json()["run_id"])
    assert status.effective_preparation["worker_spec"]["train"]["lora_alpha"] == 32


def test_warmstart_accepts_normalized_default_alpha_without_authored_metadata(api, monkeypatch):
    import flash.server.asgi.app as app_mod
    from flash.core.spec import JobSpec

    def prepare(spec, **_kwargs):
        resolved = replace(spec, train=replace(spec.train, lora_alpha=32))
        return runner_submit.PreparedJob(
            public_spec=resolved,
            worker_spec=resolved,
            estimated_cost_usd=1.25,
            adapter_identity=None,
        )

    monkeypatch.setattr(app_mod, "prepare_job", prepare)
    normalized = JobSpec.from_dict(
        {
            "project": "11111111-1111-4111-8111-111111111111",
            **SPEC,
            "train": {**SPEC["train"], "init_from_adapter": "source-run/final"},
        }
    ).to_dict()
    # a warm start cannot author alpha, so a normalized warm-start spec omits it.
    assert "lora_alpha" not in normalized["train"]

    resp = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": normalized, "dry_run": True},
    )

    assert resp.status_code == 200, resp.text
    assert "lora_alpha" not in resp.json()["spec"]["train"]
    status = runner_status.get_status(resp.json()["run_id"])
    assert status.effective_preparation["worker_spec"]["train"]["lora_alpha"] == 32


def test_warmstart_rejects_explicit_conflicting_alpha(api, monkeypatch):
    import flash.server.asgi.app as app_mod
    from flash.schema import train_schema_metadata

    def prepare(spec, **_kwargs):
        resolved = replace(spec, train=replace(spec.train, lora_alpha=32))
        return runner_submit.PreparedJob(
            public_spec=resolved,
            worker_spec=resolved,
            estimated_cost_usd=1.25,
            adapter_identity=None,
        )

    monkeypatch.setattr(app_mod, "prepare_job", prepare)
    spec = {
        **SPEC,
        "train": {
            **SPEC["train"],
            "init_from_adapter": "source-run/final",
            "lora_alpha": 64,
        },
    }

    resp = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={
            "spec": spec,
            "dry_run": True,
            "client_train_schema": {
                "version": "0.2.56",
                "fields": train_schema_metadata(),
                "authored_keys": ["init_from_adapter", "lora_alpha"],
            },
        },
    )

    assert resp.status_code == 400
    # lora_alpha is authorable again, but not alongside init_from_adapter: the source adapter's
    # alpha is authoritative, so the parser refuses the pair rather than silently overriding it.
    assert "train.lora_alpha cannot be set with train.init_from_adapter" in resp.json()["detail"]
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


def test_create_run_preflights_init_adapter_rank_before_submit(api, monkeypatch):
    import flash.adapters.lora_rank as rank_mod
    import flash.runner.results.checkpoints as checkpoints
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-9B",
            "project": "11111111-1111-4111-8111-111111111111",
            "algorithm": "sft",
            "train": {"epochs": 1, "hf_repo": "Freesolo-Co/source"},
        }
    )
    runner_state._save_status(
        runner_state.RunStatus(run_id="source-run", state="done", spec=source.to_dict())
    )
    monkeypatch.setattr(checkpoints, "adapter_artifact_exists", lambda spec, *, step: True)
    monkeypatch.setattr(
        rank_mod,
        "load_hf_adapter_config",
        lambda *a, **k: {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3.5-9B",
            "r": 96,
            "lora_alpha": 192,
        },
    )

    spec = {
        **SPEC,
        "train": {
            **SPEC["train"],
            "init_from_adapter": "source-run/final",
        },
    }

    resp = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": spec},
    )

    assert resp.status_code == 400
    assert "source 'source-run/final' could not be prepared" in resp.text
    assert "rank 96" not in resp.text
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


def test_create_run_dry_run_still_preflights_init_adapter_rank(api, monkeypatch):
    # A dry-run is a faithful server-side preview: it runs the SAME warm-start rank preflight as a
    # real submit, so a rank-mismatched adapter is rejected at --dry-run (400) instead of being
    # silently accepted and only failing at live submit. A rejected dry-run leaves no run behind.
    import flash.adapters.lora_rank as rank_mod
    import flash.runner.results.checkpoints as checkpoints
    from flash.core.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-9B",
            "project": "11111111-1111-4111-8111-111111111111",
            "algorithm": "sft",
            "train": {"epochs": 1, "hf_repo": "Freesolo-Co/source"},
        }
    )
    runner_state._save_status(
        runner_state.RunStatus(run_id="source-run", state="done", spec=source.to_dict())
    )
    monkeypatch.setattr(checkpoints, "adapter_artifact_exists", lambda spec, *, step: True)
    monkeypatch.setattr(
        rank_mod,
        "load_hf_adapter_config",
        lambda *a, **k: {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3.5-9B",
            "r": 96,
            "lora_alpha": 192,
        },
    )

    spec = {
        **SPEC,
        "train": {
            **SPEC["train"],
            "init_from_adapter": "source-run/final",
        },
    }

    resp = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": spec, "dry_run": True},
    )

    assert resp.status_code == 400
    assert "source 'source-run/final' could not be prepared" in resp.text
    assert "rank 96" not in resp.text
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


def test_create_run_redacts_internal_warmstart_preparation_error(api, monkeypatch):
    # fail inside adapter resolution itself, which is where an internal storage ref can come from.
    # stubbing the whole of prepare_job instead would assert something broader than this test's
    # name: that EVERY submit failure is redacted for a warm-start run, including gpu sizing and
    # budget, which fail identically for the non-warm-start runs that never redacted them.

    internal_ref = "private-owner/private-repo:sft/source-run/checkpoints/step-20"

    def _boom(spec, **kwargs):
        raise RuntimeError(f"failed to read {internal_ref}")

    monkeypatch.setattr(runner_preparation, "_prepare_init_from_adapter_inner", _boom)
    spec = {
        **SPEC,
        "train": {**SPEC["train"], "init_from_adapter": "source-run/step-20"},
    }

    resp = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": spec},
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "source-run/step-20" in detail
    assert "private-owner" not in resp.text
    assert "private-repo" not in resp.text
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


def test_sft_missing_dataset_is_a_400_with_packaging_remediation(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    monkeypatch.setattr(
        app_mod,
        "prepare_job",
        lambda *a, **k: (_ for _ in ()).throw(
            runner_preparation.WorkloadProfileUnavailable(
                "environment package has no readable dataset for split 'train'. "
                "Add dataset/train.jsonl to the environment package."
            )
        ),
    )

    response = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": SPEC, "dry_run": True},
    )

    assert response.status_code == 400
    assert "no readable dataset" in response.json()["detail"]
    assert "dataset/train.jsonl" in response.json()["detail"]
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


def test_create_run_does_not_blame_the_adapter_for_an_unrelated_failure(api, monkeypatch):
    # the companion direction to the redaction above. a failure raised OUTSIDE adapter resolution
    # keeps its own message, so a warm-start run told to check its adapter really has an adapter
    # problem. the previous broad except rewrote every prepare_job failure into the adapter
    # message, sending users to re-verify a healthy adapter while the real cause never arrived.
    import flash.server.asgi.app as app_mod

    monkeypatch.setattr(
        app_mod,
        "prepare_job",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("no configured provider can provision")),
    )
    spec = {
        **SPEC,
        "train": {**SPEC["train"], "init_from_adapter": "source-run/step-20"},
    }

    resp = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": spec},
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "no configured provider can provision" in detail
    assert "could not be prepared" not in detail
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


def test_create_run_keeps_ownership_when_submit_fails_after_persisting_status(api, monkeypatch):
    """A submit that dies after saving status leaves a run the owner can still see and cancel.

    ``submit_job`` persists ``RunStatus`` and then keeps working (dispatch, provisioning), so a
    failure past that point leaves a real run behind: the charge sweep and recovery both walk the
    status files, not the db. Deleting the ownership row there would strand it: 404 on status,
    logs and cancel for the only key entitled to it, while the provider footprint lives on.
    """
    import flash.server.asgi.app as app_mod
    from flash.server.platform import db

    submitted = []

    def submit(spec, **_kwargs):
        submitted.append(spec.run_id)
        # mirror the real ordering: status lands first, the rest of the launch can still blow up.
        runner_state._save_status(
            runner_state.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
        )
        raise RuntimeError("provisioning died after status was written")

    monkeypatch.setattr(app_mod, "submit_job", submit)

    key = _login()
    resp = api.post("/v1/runs", headers=_bearer(key), json={"spec": SPEC})

    assert resp.status_code == 400, resp.text
    run_id = submitted[0]
    # the owner keeps its handle on the run the failed submit left behind.
    assert db.run_owner(run_id) is not None
    assert api.get(f"/v1/runs/{run_id}", headers=_bearer(key)).status_code == 200
    assert api.get(f"/v1/runs/{run_id}/logs", headers=_bearer(key)).status_code == 200
    assert [r["run_id"] for r in api.get("/v1/runs", headers=_bearer(key)).json()["runs"]] == [
        run_id
    ]
    # and can still drive it to a terminal state itself.
    cancelled = api.post(f"/v1/runs/{run_id}/cancel", headers=_bearer(key))
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["state"] == "cancelled"


def test_create_run_deletes_the_row_when_submit_fails_before_persisting_status(api, monkeypatch):
    # the other half of the guard: no status file means the launch left nothing behind, so the
    # ownership row is pure debris and must go rather than wedge the id forever.
    import flash.server.asgi.app as app_mod
    from flash.server.platform import db

    submitted = []

    def submit(spec, **_kwargs):
        submitted.append(spec.run_id)
        raise RuntimeError("provisioning died before status was written")

    monkeypatch.setattr(app_mod, "submit_job", submit)

    key = _login()
    resp = api.post("/v1/runs", headers=_bearer(key), json={"spec": SPEC})

    assert resp.status_code == 400, resp.text
    run_id = submitted[0]
    assert db.run_owner(run_id) is None
    assert api.get(f"/v1/runs/{run_id}", headers=_bearer(key)).status_code == 404
    assert api.get("/v1/runs", headers=_bearer(key)).json()["runs"] == []


def _persist_queued_then_raise(app_mod, monkeypatch, submitted):
    """Monkeypatch submit_job to mirror its real failure ordering: status lands, then it dies."""

    def submit(spec, **_kwargs):
        submitted.append(spec.run_id)
        runner_state._save_status(
            runner_state.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
        )
        raise RuntimeError("provisioning died after status was written")

    monkeypatch.setattr(app_mod, "submit_job", submit)


def _classified_resubmits() -> list[str]:
    """Run ids startup recovery would resubmit right now."""
    from flash.server.platform import runtime

    active: set[str] = set()
    known: set[str] = set()
    resubmit: list = []
    runtime._classify_recoverable_runs(active, known, resubmit)
    return [spec.run_id for spec, _state in resubmit]


def test_create_run_dry_run_failure_leaves_no_recoverable_run(api, monkeypatch):
    """A dry-run submit that dies after persisting `queued` must not be retained.

    submit_job persists the status as `queued` and only later flips it to `dry_run`, so a failure
    in between leaves a queued record. Retaining its ownership row would hand it to startup
    recovery, which resubmits every owned queued run as a REAL job - provisioning a gpu the user
    explicitly asked never to rent. The row is dropped instead, exactly as before the guard.
    """
    import flash.server.asgi.app as app_mod
    from flash.server.platform import db

    submitted: list[str] = []
    _persist_queued_then_raise(app_mod, monkeypatch, submitted)

    key = _login()
    resp = api.post("/v1/runs", headers=_bearer(key), json={"spec": SPEC, "dry_run": True})

    assert resp.status_code == 400, resp.text
    run_id = submitted[0]
    assert db.run_owner(run_id) is None
    assert api.get(f"/v1/runs/{run_id}", headers=_bearer(key)).status_code == 404
    # recovery walks the ownership rows: with the row gone the queued record is unreachable.
    assert run_id not in _classified_resubmits()


def test_create_run_retained_secretful_run_fails_instead_of_recovering(api, monkeypatch):
    """A retained run whose runtime secrets were never dispatched must not silently recover.

    the secrets live only in the request and are deliberately excluded from the persisted spec, so
    recovery would resubmit the run without them: it would train with missing credentials and
    silently change behavior. the guard fails the run loudly instead; the owner keeps the row and
    the error, and recovery skips terminal runs.
    """
    import flash.server.asgi.app as app_mod
    from flash.server.platform import db

    submitted: list[str] = []
    _persist_queued_then_raise(app_mod, monkeypatch, submitted)

    key = _login()
    resp = api.post(
        "/v1/runs",
        headers=_bearer(key),
        json={"spec": SPEC, "runtime_secrets": {"WANDB_API_KEY": "user-wandb-key"}},
    )

    assert resp.status_code == 400, resp.text
    run_id = submitted[0]
    # the owner keeps the run and a loud, actionable error.
    assert db.run_owner(run_id) is not None
    body = api.get(f"/v1/runs/{run_id}", headers=_bearer(key)).json()
    assert body["state"] == "failed"
    assert "runtime secrets" in body["error"]
    # and recovery classifies nothing to resubmit for it.
    assert run_id not in _classified_resubmits()


def test_create_run_secretful_run_dropped_when_terminalization_fails(api, monkeypatch):
    """If the compensating terminal write RAISES, the ownership row must go.

    that `_update` is the only thing keeping startup recovery away from a queued run whose
    secrets were never dispatched. a full or read-only status store makes it raise, and
    swallowing that would leave the run both recoverable and secretless. an orphaned 404 is the
    lesser harm, so the row is dropped instead.
    """
    import flash.server.asgi.app as app_mod
    from flash.server.platform import db

    submitted: list[str] = []
    _persist_queued_then_raise(app_mod, monkeypatch, submitted)

    def boom(*_args, **_kwargs):
        raise OSError("[Errno 28] No space left on device")

    monkeypatch.setattr(runner_status, "_update", boom)

    key = _login()
    resp = api.post(
        "/v1/runs",
        headers=_bearer(key),
        json={"spec": SPEC, "runtime_secrets": {"WANDB_API_KEY": "user-wandb-key"}},
    )

    assert resp.status_code == 400, resp.text
    run_id = submitted[0]
    # the queued status record survives on disk, so only the dropped row keeps recovery off it.
    assert runner_status.get_status(run_id).state == "queued"
    assert db.run_owner(run_id) is None
    assert run_id not in _classified_resubmits()


def test_create_run_secretful_run_kept_when_status_read_fails(api, monkeypatch):
    """A terminal write that returned must not be second-guessed by a failing status read.

    `_update` returning without raising already proves the run is terminal (True applied the write,
    a sticky False means it was already terminal). a transient read error afterwards says nothing
    about that, so treating it as a failed terminalization would delete the ownership row of a
    correctly failed run - orphaning it for its owner and throwing away the error just persisted.
    """
    import flash.server.asgi.app as app_mod
    from flash.server.platform import db

    submitted: list[str] = []
    _persist_queued_then_raise(app_mod, monkeypatch, submitted)

    # break status reads only once the terminal write itself has landed (`_update` reads the record
    # to apply it), so this is exactly "the write succeeded, the read after it did not".
    real_get_status, real_update = runner_status.get_status, runner_status._update
    reading_fails = {"on": False}

    def flaky(run_id, *args, **kwargs):
        if reading_fails["on"]:
            raise OSError("[Errno 5] Input/output error")
        return real_get_status(run_id, *args, **kwargs)

    def update_then_break_reads(*args, **kwargs):
        applied = real_update(*args, **kwargs)
        reading_fails["on"] = True
        return applied

    monkeypatch.setattr(runner_status, "get_status", flaky)
    monkeypatch.setattr(runner_status, "_update", update_then_break_reads)

    key = _login()
    resp = api.post(
        "/v1/runs",
        headers=_bearer(key),
        json={"spec": SPEC, "runtime_secrets": {"WANDB_API_KEY": "user-wandb-key"}},
    )
    reading_fails["on"] = False

    assert resp.status_code == 400, resp.text
    run_id = submitted[0]
    # the terminal write landed, so the owner keeps the row and the actionable error.
    assert db.run_owner(run_id) is not None
    body = api.get(f"/v1/runs/{run_id}", headers=_bearer(key)).json()
    assert body["state"] == "failed"
    assert "runtime secrets" in body["error"]
    # and the run is terminal, so recovery still has nothing to resubmit.
    assert run_id not in _classified_resubmits()


def test_create_run_retained_run_records_managed_environment_use(api, monkeypatch):
    # a retained run stays live and can recover into real training, so it must carry the same
    # managed-environment association a successful submission records.
    import flash.server.asgi.app as app_mod
    import flash.server.domain.registry.environment_registry as registry
    from flash.server.platform import db

    calls: list[dict] = []
    monkeypatch.setattr(
        registry,
        "record_environment_use",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    submitted: list[str] = []
    _persist_queued_then_raise(app_mod, monkeypatch, submitted)
    spec = {**SPEC, "environment": {"id": "acme/checkout-bot/my-env"}}

    key = _login()
    resp = api.post("/v1/runs", headers=_bearer(key), json={"spec": spec})

    assert resp.status_code == 400, resp.text
    run_id = submitted[0]
    assert db.run_owner(run_id) is not None
    assert calls
    assert calls[0]["slug"] == "acme/checkout-bot/my-env"
    assert calls[0]["run_id"] == run_id


def test_freesolo_user_key_disabled_is_401_not_500(api, monkeypatch):
    # A freesolo key that verifies with the backend but whose db row was disabled (revoked)
    # must be rejected as 401 (authenticate -> None), not raise a 500.
    import sqlite3

    import flash.server.platform.auth as auth_mod
    import flash.server.platform.db as db_mod

    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: True)
    monkeypatch.setattr(
        auth_mod,
        "_cached_identity",
        lambda token: {
            "email": "revoked@example.com",
            "key_prefix": "fslo_revoked",
            "org_slug": "acme",
        },
    )

    assert auth_mod.authenticate("Bearer fslo-revoked") is not None  # provisioned on first use
    with sqlite3.connect(db_mod.db_path()) as conn:
        conn.execute(
            "UPDATE api_keys SET disabled = 1 WHERE key_hash = ?",
            (db_mod.hash_key("fslo-revoked"),),
        )
    # Verified by freesolo but disabled in the db -> None (401), never a raised 500.
    assert auth_mod.authenticate("Bearer fslo-revoked") is None


def test_freesolo_verify_does_not_cache_network_errors(monkeypatch):
    # A transient network error must NOT be cached as a rejection, or a valid key would be
    # locked out for the whole TTL. The next call (backend recovered) must succeed.
    import urllib.error

    import flash.server.platform.auth as auth_mod

    # Use the real _freesolo_verify (not the fixture stub) and let it touch the (patched) net.
    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    state = {"fail": True}

    def flaky(req, timeout=None):
        if state["fail"]:
            raise urllib.error.URLError("connection timed out")
        return _Resp()

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", flaky)
    assert auth_mod._freesolo_verify("tok") is False  # transient failure
    assert "tok" not in auth_mod._verify_cache  # NOT cached
    state["fail"] = False
    assert auth_mod._freesolo_verify("tok") is True  # recovers immediately


def test_freesolo_verify_5xx_transient_but_4xx_cached(monkeypatch):
    # A backend 5xx/429 is a transient hiccup (urllib raises HTTPError for these too): it must
    # NOT be cached, so a valid key recovers immediately. A definitive 4xx (401/403) IS cached.
    import urllib.error

    import flash.server.platform.auth as auth_mod

    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    state = {"code": 503}

    def responder(req, timeout=None):
        code = state["code"]
        if code == 200:
            return _Resp()
        raise urllib.error.HTTPError(req.full_url, code, "err", {}, None)

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", responder)
    # 5xx -> transient, not cached
    assert auth_mod._freesolo_verify("tok") is False
    assert "tok" not in auth_mod._verify_cache
    # 429 -> transient, not cached
    state["code"] = 429
    assert auth_mod._freesolo_verify("tok") is False
    assert "tok" not in auth_mod._verify_cache
    # backend recovers -> immediately verified (no stale negative cached)
    state["code"] = 200
    assert auth_mod._freesolo_verify("tok") is True

    # a definitive 401 IS cached as a rejection (no repeated backend round-trips)
    auth_mod._verify_cache.clear()
    state["code"] = 401
    assert auth_mod._freesolo_verify("bad") is False
    assert auth_mod._verify_cache.get("bad", (None,))[0] is False


def test_freesolo_verify_discards_identity_on_exit_http_error(monkeypatch):
    import io
    import urllib.error

    import flash.server.platform.auth as auth_mod

    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()

    class _Resp:
        status = 200

        def read(self):
            return b'{"email":"user@example.com","org_id":"org-1","org_slug":"acme"}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            raise urllib.error.HTTPError(
                "https://api.freesolo.co/api/auth/verify",
                401,
                "unauthorized",
                {},
                io.BytesIO(b'{"detail":"unauthorized"}'),
            )

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", lambda req, timeout=None: _Resp())

    assert auth_mod._freesolo_verify("tok") is False
    verified, identity, _expires_at = auth_mod._verify_cache["tok"]
    assert verified is False
    assert identity == {}


def test_freesolo_verify_negative_short_ttl_positive_long_ttl(monkeypatch):
    # A negative verdict (a 401 may be a TRANSIENT backend auth-lookup outage, not a real
    # rejection) gets the short negative TTL so a valid key isn't locked out for 5 minutes;
    # a positive keeps the long TTL.
    import urllib.error

    import flash.server.platform.auth as auth_mod

    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    state = {"code": 401}

    def responder(req, timeout=None):
        code = state["code"]
        if code == 200:
            return _Resp()
        raise urllib.error.HTTPError(req.full_url, code, "err", {}, None)

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", responder)

    # Negative (401) -> cached with the SHORT negative TTL.
    now = time.time()
    assert auth_mod._freesolo_verify("neg") is False
    neg_exp = auth_mod._verify_cache["neg"][2]
    assert neg_exp <= now + auth_mod._VERIFY_CACHE_NEG_TTL_S + 1
    # ...and definitely shorter than the long TTL.
    assert neg_exp < now + auth_mod._VERIFY_CACHE_TTL_S

    # Positive -> cached with the LONG TTL.
    auth_mod._verify_cache.clear()
    state["code"] = 200
    now = time.time()
    assert auth_mod._freesolo_verify("pos") is True
    pos_exp = auth_mod._verify_cache["pos"][2]
    assert pos_exp > now + auth_mod._VERIFY_CACHE_NEG_TTL_S + 1
    assert pos_exp <= now + auth_mod._VERIFY_CACHE_TTL_S + 1

    # The negative entry expires after the short TTL: simulate the clock advancing past
    # _VERIFY_CACHE_NEG_TTL_S (but not the long TTL) and confirm the negative is treated as
    # expired while a same-age positive would still be live.
    auth_mod._verify_cache.clear()
    base = time.time()
    auth_mod._verify_cache["neg"] = (False, {}, base + auth_mod._VERIFY_CACHE_NEG_TTL_S)
    auth_mod._verify_cache["pos"] = (True, {}, base + auth_mod._VERIFY_CACHE_TTL_S)
    later = base + auth_mod._VERIFY_CACHE_NEG_TTL_S + 1.0  # past neg TTL, well under pos TTL
    assert auth_mod._verify_cache["neg"][2] <= later  # negative entry has expired
    assert auth_mod._verify_cache["pos"][2] > later  # positive entry is still live


def test_freesolo_verify_rejects_oversized_token(monkeypatch):
    # An oversized bearer must be rejected before it touches the cache or the network, so it
    # can't bloat _verify_cache (keyed by the raw token) or send a huge Authorization header.
    import flash.server.platform.auth as auth_mod

    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()

    def boom(*a, **k):
        raise AssertionError("oversized token must not reach the network")

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", boom)
    huge = "x" * (auth_mod._MAX_TOKEN_LEN + 1)
    assert auth_mod._freesolo_verify(huge) is False
    assert huge not in auth_mod._verify_cache


def test_freesolo_user_key_unverified_when_backend_unreachable(api, monkeypatch):
    # When the backend verify can't be reached (the offline test harness makes urlopen fail),
    # _freesolo_verify returns False and authenticate yields None — an unverifiable key is
    # never admitted.
    import urllib.error

    import flash.server.platform.auth as auth_mod

    auth_mod._verify_cache.clear()
    # Drop the fixture's stub so the real _freesolo_verify runs, and make the backend
    # unreachable (offline): the verify can't be reached.
    importlib.reload(auth_mod)
    monkeypatch.setattr(
        auth_mod.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    assert auth_mod.authenticate("Bearer unknown-token") is None


def test_freesolo_verify_cache_prevents_second_call(monkeypatch):
    # The in-process cache means a second authenticate for the same token doesn't re-hit the
    # backend within the TTL (positives and negatives are both cached).
    import flash.server.platform.auth as auth_mod

    # Use the real _freesolo_verify (not the fixture stub) and let it touch the (patched) net.
    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1

        class _Resp:
            status = 200

            def read(self):
                return (
                    b'{"email":"cached@example.com","key_prefix":"fslo_cached","org_slug":"acme"}'
                )

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", fake_urlopen)

    first = auth_mod.authenticate("Bearer fslo-cached")
    second = auth_mod.authenticate("Bearer fslo-cached")
    assert first is not None
    assert second is not None
    assert calls["n"] == 1  # second authenticate served from cache, no second backend call


def test_freesolo_verify_cache_is_bounded_and_prunes_expired(monkeypatch):
    # The verify cache keys by the raw bearer token, so a stream of distinct tokens could
    # grow it without bound. Each write prunes expired entries and caps the cache size.
    import time

    import flash.server.platform.auth as auth_mod

    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", lambda req, timeout=None: _Resp())

    # An already-expired entry must be removed on the next write (no longer reachable).
    auth_mod._verify_cache["stale"] = (True, {}, time.time() - 1)
    auth_mod._freesolo_verify("fresh-token")
    assert "stale" not in auth_mod._verify_cache
    assert "fresh-token" in auth_mod._verify_cache

    # Verifying many distinct (live) tokens never grows the cache past the cap.
    monkeypatch.setattr(auth_mod, "_VERIFY_CACHE_MAX", 8)
    auth_mod._verify_cache.clear()
    for i in range(50):
        auth_mod._freesolo_verify(f"tok-{i}")
        assert len(auth_mod._verify_cache) <= auth_mod._VERIFY_CACHE_MAX
    assert len(auth_mod._verify_cache) <= auth_mod._VERIFY_CACHE_MAX
    auth_mod._verify_cache.clear()


def test_keys_are_hashed_at_rest(api):
    key = _login()
    # Authenticate once so the key's row is provisioned.
    assert api.get("/v1/me", headers=_bearer(key)).status_code == 200
    with sqlite3.connect(_db_mod.db_path()) as conn:
        rows = conn.execute("SELECT key_hash, key_prefix FROM api_keys").fetchall()
    assert rows
    for key_hash, _prefix in rows:
        assert key_hash != key
        assert len(key_hash) == 64  # sha256 hex
    with open(_db_mod.db_path(), "rb") as f:
        raw = f.read()
    assert key.encode() not in raw


def test_run_lifecycle_and_tenant_isolation(api):
    key_a, key_b = _login(), _login()
    created = api.post("/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key_a))
    assert created.status_code == 200, created.text
    run_id = created.json()["run_id"]
    assert created.json()["state"] == "dry_run"

    # Owner sees it (status, list); the other tenant gets 404s and an empty list.
    assert api.get(f"/v1/runs/{run_id}", headers=_bearer(key_a)).status_code == 200
    assert [r["run_id"] for r in api.get("/v1/runs", headers=_bearer(key_a)).json()["runs"]] == [
        run_id
    ]
    assert api.get(f"/v1/runs/{run_id}", headers=_bearer(key_b)).status_code == 404
    assert api.get("/v1/runs", headers=_bearer(key_b)).json()["runs"] == []
    assert api.post(f"/v1/runs/{run_id}/cancel", headers=_bearer(key_b)).status_code == 404
    assert api.get(f"/v1/runs/{run_id}/logs", headers=_bearer(key_b)).status_code == 404


def test_runtime_secret_validation_and_non_persistence(api):
    key = _login()
    bad = api.post(
        "/v1/runs",
        json={
            "spec": SPEC,
            "dry_run": True,
            "runtime_secrets": {"RUNPOD_API_KEY": "must-stay-platform-side"},
        },
        headers=_bearer(key),
    )
    assert bad.status_code == 400
    assert "unsupported runtime secret" in bad.json()["detail"]

    created = api.post(
        "/v1/runs",
        json={
            "spec": SPEC,
            "dry_run": True,
            "runtime_secrets": {"WANDB_API_KEY": "user-wandb-key"},
        },
        headers=_bearer(key),
    )
    assert created.status_code == 200, created.text
    body = created.json()
    dumped = json.dumps(body)
    assert "user-wandb-key" not in dumped
    assert "runtime_secrets" not in dumped

    env_secret_spec = {
        **SPEC,
        "environment": {
            **SPEC["environment"],
            "secrets": ["SERPAPI_API_KEY"],
        },
    }
    created = api.post(
        "/v1/runs",
        json={
            "spec": env_secret_spec,
            "dry_run": True,
            "runtime_secrets": {"SERPAPI_API_KEY": "serp-user-key"},
        },
        headers=_bearer(key),
    )
    assert created.status_code == 200, created.text
    body = created.json()
    dumped = json.dumps(body)
    assert "serp-user-key" not in dumped
    assert "runtime_secrets" not in dumped
    assert body["spec"]["environment"]["secrets"] == ["SERPAPI_API_KEY"]

    for dry_run in (False, True):
        missing = api.post(
            "/v1/runs",
            json={"spec": env_secret_spec, "dry_run": dry_run, "runtime_secrets": {}},
            headers=_bearer(key),
        )
        assert missing.status_code == 400
        assert (
            missing.json()["detail"]
            == "missing runtime secret(s) required by [environment] secrets: SERPAPI_API_KEY"
        )


def test_logs_offset_paging(api):
    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    log_path = os.path.join(runner_state.RUNS_DIR, f"{run_id}.log")
    with open(log_path, "w") as f:
        f.write("line one\n")
    page = api.get(f"/v1/runs/{run_id}/logs", headers=_bearer(key)).json()
    assert page["logs"] == "line one\n"
    assert page["state"] == "dry_run"
    with open(log_path, "a") as f:
        f.write("line two\n")
    page2 = api.get(f"/v1/runs/{run_id}/logs?offset={page['offset']}", headers=_bearer(key)).json()
    assert page2["logs"] == "line two\n"


def test_worker_output_route(api, monkeypatch):
    # /worker surfaces the train-subprocess stdout/traceback from the run's HF repo (operator
    # token, server-side). Best-effort: no artifacts -> empty dict; present -> passed through.
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]

    # Stub _worker_artifacts to {} BEFORE the first request: the real impl would hf_hub_download from
    # the run's HF repo (slow/flaky network in a unit test). This keeps the "no artifacts -> {}"
    # assertion fully offline/deterministic.
    monkeypatch.setattr(app_mod, "_worker_artifacts", lambda spec: {})
    empty = api.get(f"/v1/runs/{run_id}/worker", headers=_bearer(key)).json()
    assert empty["run_id"] == run_id
    assert empty["worker"] == {}

    monkeypatch.setattr(
        app_mod, "_worker_artifacts", lambda spec: {"console_sft.txt": "real worker stdout\n"}
    )
    got = api.get(f"/v1/runs/{run_id}/worker", headers=_bearer(key)).json()
    assert got["worker"] == {"console_sft.txt": "real worker stdout\n"}

    # Another user can't read it (same ownership gate as /logs).
    other = _login()
    assert api.get(f"/v1/runs/{run_id}/worker", headers=_bearer(other)).status_code == 404


def test_internal_key_reads_logs_and_worker_with_matching_org(api, monkeypatch):
    # The platform web proxy has no per-run API key, so it reads a user's run logs with the shared
    # internal key plus the run's org in X-Freesolo-Org-Id. The header is honored ONLY for the
    # internal key and ONLY when it matches the run's persisted org. All failures are 404 so run
    # existence is never leaked, and a rejected request must not trigger the expensive worker fetch.
    import flash.server.asgi.app as app_mod

    owner = _login()
    org = f"org-{owner.removeprefix(_USER_PREFIX)}"
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(owner)
    ).json()["run_id"]
    with open(os.path.join(runner_state.RUNS_DIR, f"{run_id}.log"), "w") as f:
        f.write("orchestrator line\n")

    worker_calls = {"n": 0}

    def counting_worker(spec):
        worker_calls["n"] += 1
        return {"console_sft.txt": "worker stdout\n"}

    monkeypatch.setattr(app_mod, "_worker_artifacts", counting_worker)

    internal = _bearer("fslo-internal-test")
    matched = {**internal, "X-Freesolo-Org-Id": org}

    logs = api.get(f"/v1/runs/{run_id}/logs", headers=matched)
    assert logs.status_code == 200, logs.text
    assert logs.json()["logs"] == "orchestrator line\n"
    worker = api.get(f"/v1/runs/{run_id}/worker", headers=matched)
    assert worker.status_code == 200, worker.text
    assert worker.json()["worker"] == {"console_sft.txt": "worker stdout\n"}
    assert worker_calls["n"] == 1

    # wrong org, empty/whitespace org, or no header at all -> 404, and none of these rejected
    # /worker requests reach the expensive artifact fetch (the call count stays at 1).
    for headers in (
        {**internal, "X-Freesolo-Org-Id": "org-someone-else"},
        {**internal, "X-Freesolo-Org-Id": ""},
        {**internal, "X-Freesolo-Org-Id": "   "},
        internal,
    ):
        assert api.get(f"/v1/runs/{run_id}/logs", headers=headers).status_code == 404
        assert api.get(f"/v1/runs/{run_id}/worker", headers=headers).status_code == 404
    assert worker_calls["n"] == 1


def test_org_header_is_ignored_for_non_internal_keys(api):
    # A non-owner user key can't read another tenant's logs even if it forges the org header: the
    # header is consulted only for the internal service key, so the owner check still governs users.
    owner = _login()
    org = f"org-{owner.removeprefix(_USER_PREFIX)}"
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(owner)
    ).json()["run_id"]

    intruder = _login()
    forged = {**_bearer(intruder), "X-Freesolo-Org-Id": org}
    assert api.get(f"/v1/runs/{run_id}/logs", headers=forged).status_code == 404
    assert api.get(f"/v1/runs/{run_id}/worker", headers=forged).status_code == 404

    # the real owner still reads its own run with no org header (unchanged CLI path).
    assert api.get(f"/v1/runs/{run_id}/logs", headers=_bearer(owner)).status_code == 200


def test_internal_org_header_does_not_widen_owner_only_endpoints(api):
    # readable_run only governs the read-only /logs and /worker endpoints. /status, /cancel, and
    # /checkpoints stay owner-only via owned_run, so the internal key plus a matching org header
    # must NOT grant access to them.
    owner = _login()
    org = f"org-{owner.removeprefix(_USER_PREFIX)}"
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(owner)
    ).json()["run_id"]

    matched = {**_bearer("fslo-internal-test"), "X-Freesolo-Org-Id": org}
    assert api.get(f"/v1/runs/{run_id}", headers=matched).status_code == 404
    assert api.post(f"/v1/runs/{run_id}/cancel", headers=matched).status_code == 404
    assert api.get(f"/v1/runs/{run_id}/checkpoints", headers=matched).status_code == 404


def test_internal_key_cannot_read_run_without_persisted_org(api, monkeypatch):
    # A run with no persisted org context cannot be matched by the internal path even with a
    # non-empty header: _status_org_id is empty, so it never equals a real org and fails closed.
    import flash.server.asgi.app as app_mod

    owner = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(owner)
    ).json()["run_id"]

    real_get_status = app_mod.get_status

    def stripped(rid):
        status = real_get_status(rid)
        if rid == run_id:
            status.platform_context = None
            status.billing_context = None
        return status

    monkeypatch.setattr(app_mod, "get_status", stripped)

    internal = _bearer("fslo-internal-test")
    for org in ("org-anything", f"org-{owner.removeprefix(_USER_PREFIX)}"):
        headers = {**internal, "X-Freesolo-Org-Id": org}
        assert api.get(f"/v1/runs/{run_id}/logs", headers=headers).status_code == 404
        assert api.get(f"/v1/runs/{run_id}/worker", headers=headers).status_code == 404


def test_latest_error_artifact_name_picks_highest_attempt(monkeypatch):
    """The logs fetcher resolves the newest attempt-scoped error file, so a retried-then-failed run
    surfaces the FINAL attempt's traceback, not attempt0's stale one."""
    import huggingface_hub

    from flash.server.platform.runtime import _latest_error_artifact_name

    prefix = "sft/run-1/seed0"
    listed = [
        f"{prefix}/console_sft.txt",
        f"{prefix}/error_sft_attempt0.txt",
        f"{prefix}/error_sft_attempt2.txt",
        f"{prefix}/error_sft_attempt1.txt",
        f"{prefix}/heartbeat.json",
        "other/run/error_sft_attempt9.txt",  # different prefix -> ignored
    ]

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            return listed

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    assert _latest_error_artifact_name("org/repo", prefix, "sft") == "error_sft_attempt2.txt"


def test_latest_error_artifact_name_defaults_when_unlistable(monkeypatch):
    """If the repo can't be listed, fall back to attempt0 rather than failing the logs fetch."""
    import huggingface_hub

    from flash.server.platform.runtime import _latest_error_artifact_name

    class _BoomApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            raise RuntimeError("HF down")

    monkeypatch.setattr(huggingface_hub, "HfApi", _BoomApi)
    assert _latest_error_artifact_name("org/repo", "rl/r/seed0", "rl") == "error_rl_attempt0.txt"


def test_worker_artifacts_fetches_console_and_latest_attempt_error(monkeypatch, tmp_path):
    """The fetcher pulls the worker console plus the NEWEST attempt-scoped error file
    (error_<phase>_attempt<N>.txt) — on a retried run only the highest attempt is the real crash."""
    import types

    import huggingface_hub

    from flash.server.platform.runtime import _worker_artifacts

    spec = types.SimpleNamespace(
        phase="rl",
        run_id="r1",
        train=types.SimpleNamespace(hf_repo="org/repo"),
    )
    content = {
        "rl/r1/console_rl.txt": "worker console\n",
        "rl/r1/error_rl_attempt0.txt": "stale first-attempt traceback\n",
        "rl/r1/error_rl_attempt1.txt": "TRACEBACK latest\n",
    }

    def fake_dl(repo_id, repo_type, filename, token=None, force_download=False):
        if filename not in content:
            raise FileNotFoundError(filename)
        p = tmp_path / filename.replace("/", "_")
        p.write_text(content[filename])
        return str(p)

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            return list(content)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    out = _worker_artifacts(spec)
    assert out["console_rl.txt"] == "worker console\n"
    # The newest attempt's traceback surfaces; the superseded attempt0 is not fetched.
    assert out["error_rl_attempt1.txt"] == "TRACEBACK latest\n"
    assert "error_rl_attempt0.txt" not in out


def test_worker_artifacts_surfaces_the_ray_failure_logs(monkeypatch, tmp_path):
    """The ray collector's whole purpose is defeated if nothing fetches what it uploads.

    A raylet death shows up in the traceback only as its downstream symptom ("Failed to register
    worker to Raylet: ... Attempt-scoped like the traceback beside it: on a retry only the highest
    attempt reproduced the failure, and a stale one would misdirect the diagnosis it exists to give.
    """
    import types

    import huggingface_hub

    from flash.server.platform.runtime import _worker_artifacts

    spec = types.SimpleNamespace(
        phase="rl",
        run_id="r1",
        train=types.SimpleNamespace(hf_repo="org/repo"),
    )
    content = {
        "rl/r1/console_rl.txt": "worker console\n",
        "rl/r1/error_rl_attempt1.txt": "Failed to register worker to Raylet: ... End of file\n",
        "rl/r1/raylogs_rl_attempt0.txt": "stale first-attempt ray logs\n",
        "rl/r1/raylogs_rl_attempt1.txt": "RAYLET died: /tmp/ray/session_x/logs/raylet.err\n",
    }

    def fake_dl(repo_id, repo_type, filename, token=None, force_download=False):
        if filename not in content:
            raise FileNotFoundError(filename)
        p = tmp_path / filename.replace("/", "_")
        p.write_text(content[filename])
        return str(p)

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            return list(content)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    out = _worker_artifacts(spec)
    assert out["raylogs_rl_attempt1.txt"] == (
        "RAYLET died: /tmp/ray/session_x/logs/raylet.err\n"
    ), "the ray failure logs never reach the user, so a raylet death stays undiagnosable"
    assert "raylogs_rl_attempt0.txt" not in out, "a superseded attempt's ray logs must not surface"
    # the artifacts it already carried are unaffected.
    assert out["console_rl.txt"] == "worker console\n"
    assert "End of file" in out["error_rl_attempt1.txt"]


def test_worker_artifacts_does_not_pair_a_prior_attempts_ray_logs_with_this_traceback(
    monkeypatch, tmp_path
):
    """Ray logs must describe the SAME attempt as the traceback they appear beside.

    They are uploaded only when ray actually failed, so on a retried run the newest raylogs and the
    newest traceback can belong to different attempts: attempt 0 dies to a raylet, attempt 1 fails
    for an unrelated reason and uploads none.
    """
    import types

    import huggingface_hub

    from flash.server.platform.runtime import _worker_artifacts

    spec = types.SimpleNamespace(
        phase="rl", run_id="r1", train=types.SimpleNamespace(hf_repo="org/repo")
    )
    content = {
        "rl/r1/console_rl.txt": "worker console\n",
        "rl/r1/error_rl_attempt1.txt": "ValueError: dataset row 3 is malformed\n",
        "rl/r1/raylogs_rl_attempt0.txt": "RAYLET died in the FIRST attempt\n",
    }

    def fake_dl(repo_id, repo_type, filename, token=None, force_download=False):
        if filename not in content:
            raise FileNotFoundError(filename)
        p = tmp_path / filename.replace("/", "_")
        p.write_text(content[filename])
        return str(p)

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            return list(content)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    out = _worker_artifacts(spec)
    assert not [k for k in out if k.startswith("raylogs")], (
        "a prior attempt's ray logs were paired with this attempt's unrelated traceback"
    )
    # the real evidence for this attempt is untouched -- suppressing the mismatch must not cost the
    # traceback that actually explains the failure.
    assert out["error_rl_attempt1.txt"] == "ValueError: dataset row 3 is malformed\n"
    assert out["console_rl.txt"] == "worker console\n"


def test_worker_artifacts_skips_ray_logs_when_the_traceback_is_unscoped(monkeypatch, tmp_path):
    # a legacy unscoped traceback names no attempt, so there is nothing to pin raylogs to. guessing
    # the newest would reintroduce exactly the mismatch above, on the one artifact shape that cannot
    # be checked.
    import types

    import huggingface_hub

    from flash.server.platform.runtime import _worker_artifacts

    spec = types.SimpleNamespace(
        phase="rl", run_id="r1", train=types.SimpleNamespace(hf_repo="org/repo")
    )
    content = {
        "rl/r1/error_rl.txt": "legacy unscoped traceback\n",
        "rl/r1/raylogs_rl_attempt0.txt": "ray logs from some attempt\n",
    }

    def fake_dl(repo_id, repo_type, filename, token=None, force_download=False):
        if filename not in content:
            raise FileNotFoundError(filename)
        p = tmp_path / filename.replace("/", "_")
        p.write_text(content[filename])
        return str(p)

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            return list(content)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    out = _worker_artifacts(spec)
    assert not [k for k in out if k.startswith("raylogs")]
    assert out["error_rl.txt"] == "legacy unscoped traceback\n"


def test_ray_log_name_is_built_the_same_way_the_worker_builds_it():
    # the control plane derives this name; the worker writes it. they are in different processes with
    # no shared constant, so pin the two spellings against each other rather than against a literal.
    from flash.engine.worker.io.hf import ray_log_artifact_name
    from flash.server.platform.runtime import _ray_log_name_for_attempt

    for phase in ("rl", "sft", "opd"):
        for attempt in (0, 1, 7):
            assert _ray_log_name_for_attempt(
                phase, f"error_{phase}_attempt{attempt}.txt"
            ) == ray_log_artifact_name(phase, attempt)


def test_worker_artifacts_is_unaffected_when_ray_never_failed(monkeypatch, tmp_path):
    """The ray artifact is uploaded only when ray actually failed, so it is usually absent.

    Its absence must stay a non-event: the fetch is best-effort per file, and a missing raylogs
    file cannot suppress the console and traceback that every failure has.
    """
    import types

    import huggingface_hub

    from flash.server.platform.runtime import _worker_artifacts

    spec = types.SimpleNamespace(
        phase="sft",
        run_id="r2",
        train=types.SimpleNamespace(hf_repo="org/repo"),
    )
    content = {
        "sft/r2/console_sft.txt": "worker console\n",
        "sft/r2/error_sft_attempt0.txt": "ordinary traceback\n",
    }

    def fake_dl(repo_id, repo_type, filename, token=None, force_download=False):
        if filename not in content:
            raise FileNotFoundError(filename)
        p = tmp_path / filename.replace("/", "_")
        p.write_text(content[filename])
        return str(p)

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            return list(content)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    out = _worker_artifacts(spec)
    assert out["console_sft.txt"] == "worker console\n"
    assert out["error_sft_attempt0.txt"] == "ordinary traceback\n"
    assert not [k for k in out if k.startswith("raylogs")]


def test_worker_artifacts_prefers_latest_attempt_console(monkeypatch, tmp_path):
    """When console output is attempt-scoped, /worker should show the current attempt's tail."""
    import types

    import huggingface_hub

    from flash.server.platform.runtime import _worker_artifacts

    spec = types.SimpleNamespace(
        phase="rl",
        run_id="r1",
        train=types.SimpleNamespace(hf_repo="org/repo"),
    )
    content = {
        "rl/r1/console_rl_attempt0.txt": "stale console\n",
        "rl/r1/console_rl_attempt2.txt": "current console\n",
        "rl/r1/error_rl_attempt2.txt": "current traceback\n",
    }

    def fake_dl(repo_id, repo_type, filename, token=None, force_download=False):
        if filename not in content:
            raise FileNotFoundError(filename)
        p = tmp_path / filename.replace("/", "_")
        p.write_text(content[filename])
        return str(p)

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            return list(content)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    out = _worker_artifacts(spec)
    assert out["console_rl_attempt2.txt"] == "current console\n"
    assert out["error_rl_attempt2.txt"] == "current traceback\n"
    assert "console_rl_attempt0.txt" not in out


def test_worker_artifacts_keep_previous_attempt_evidence_until_the_retry_uploads(
    monkeypatch, tmp_path
):
    """A live retry may not have uploaded any attempt-1 artifact yet.

    The highest uploaded attempt is then attempt 0. Keep its console, traceback, and matching ray logs
    so the CLI can label the historical evidence instead of hiding the OOM that caused the retry.
    """
    import types

    import huggingface_hub

    from flash.server.platform.runtime import _worker_artifacts

    spec = types.SimpleNamespace(
        phase="rl",
        run_id="r1",
        train=types.SimpleNamespace(hf_repo="org/repo"),
    )
    content = {
        "rl/r1/console_rl_attempt0.txt": "HEARTBEAT attempt=0 device=NVIDIA H200\n",
        "rl/r1/error_rl_attempt0.txt": "torch.OutOfMemoryError: CUDA OOM\n",
        "rl/r1/raylogs_rl_attempt0.txt": "raylet exited after OOM\n",
    }

    def fake_dl(repo_id, repo_type, filename, token=None, force_download=False):
        if filename not in content:
            raise FileNotFoundError(filename)
        path = tmp_path / filename.replace("/", "_")
        path.write_text(content[filename])
        return str(path)

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            return list(content)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    assert _worker_artifacts(spec) == {
        "console_rl_attempt0.txt": "HEARTBEAT attempt=0 device=NVIDIA H200\n",
        "error_rl_attempt0.txt": "torch.OutOfMemoryError: CUDA OOM\n",
        "raylogs_rl_attempt0.txt": "raylet exited after OOM\n",
    }


def test_worker_artifacts_surface_the_terminal_console_tail_not_only_the_snapshot(
    monkeypatch, tmp_path
):
    """The crash lands in the canonical console, which the attempt ranking scores -1.

    The periodic snapshot is written mid-run to the attempt-scoped name; the terminal tail is written
    at teardown to ``console_<phase>.txt``. They are separate destinations on purpose, so the newest
    NAME is not the newest CONTENT: ranking by attempt picks the scoped snapshot and stops, hiding
    the tail that actually holds the traceback. Fetch both so the failure is reachable.
    """
    import types

    import huggingface_hub

    from flash.server.platform.runtime import _worker_artifacts

    spec = types.SimpleNamespace(
        phase="rl",
        run_id="r1",
        train=types.SimpleNamespace(hf_repo="org/repo"),
    )
    content = {
        # mid-run snapshot: healthy, and the highest-ranked name.
        "rl/r1/console_rl_attempt0.txt": "HEARTBEAT attempt=0 step=3\n",
        # terminal tail: the same stream, 56k further on, ending in the crash.
        "rl/r1/console_rl.txt": (
            "HEARTBEAT attempt=0 step=3\nCUDA error: an illegal memory access\n"
        ),
    }

    def fake_dl(repo_id, repo_type, filename, token=None, force_download=False):
        if filename not in content:
            raise FileNotFoundError(filename)
        path = tmp_path / filename.replace("/", "_")
        path.write_text(content[filename])
        return str(path)

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            return list(content)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    out = _worker_artifacts(spec)
    assert "illegal memory access" in out["console_rl.txt"]
    # the snapshot stays: on a retry whose terminal upload never ran, the canonical name can belong
    # to an OLDER attempt, and the scoped one is then the only evidence for the live attempt.
    assert out["console_rl_attempt0.txt"] == "HEARTBEAT attempt=0 step=3\n"


def test_local_env_path_rejected(api):
    # Managed runs accept Freesolo environment ids; local [environment] paths are rejected.
    key = _login()
    bad = {**SPEC, "environment": {"id": "custom", "path": "/home/user/env.py"}}
    r = api.post("/v1/runs", json={"spec": bad, "dry_run": True}, headers=_bearer(key))
    assert r.status_code == 400
    assert "not supported on the managed service" in r.json()["detail"]


def test_bad_spec_is_400(api):
    key = _login()
    r = api.post("/v1/runs", json={"spec": {"algorithm": "grpo"}}, headers=_bearer(key))
    assert r.status_code == 400
    assert "model" in r.json()["detail"]


def test_non_object_spec_fields_get_targeted_400(api):
    # A falsy NON-object JSON value (""/0/[]/false) for spec / spec.environment / runtime_secrets
    # must 400 with the intended "must be a JSON object" message, not get coerced to {} (which
    # would surface a misleading downstream error like "config must set [environment] id").
    key = _login()
    for bad_spec in ("", 0, [], False):
        r = api.post("/v1/runs", json={"spec": bad_spec}, headers=_bearer(key))
        assert r.status_code == 400, (bad_spec, r.text)
        assert "spec must be a JSON object" in r.json()["detail"], (bad_spec, r.text)

    for bad_env in ("", 0, [], False):
        r = api.post(
            "/v1/runs",
            json={"spec": {**SPEC, "environment": bad_env}},
            headers=_bearer(key),
        )
        assert r.status_code == 400, (bad_env, r.text)
        assert "spec.environment must be a JSON object" in r.json()["detail"], (bad_env, r.text)

    for bad_secrets in ("", 0, [], False):
        r = api.post(
            "/v1/runs",
            json={"spec": SPEC, "dry_run": True, "runtime_secrets": bad_secrets},
            headers=_bearer(key),
        )
        assert r.status_code == 400, (bad_secrets, r.text)
        assert "runtime_secrets must be a JSON object" in r.json()["detail"], (bad_secrets, r.text)


def test_create_run_rejects_top_level_and_gpu_typos_as_400(api):
    key = _login()
    for spec, expected in (
        ({**SPEC, "model_revison": "main"}, "model_revison"),
        ({**SPEC, "gpu": {"exact_typ": "H100"}}, "exact_typ"),
    ):
        response = api.post(
            "/v1/runs",
            json={"spec": spec, "dry_run": True},
            headers=_bearer(key),
        )
        assert response.status_code == 400, response.text
        assert expected in response.json()["detail"]


def test_deploy_dry_run(api):
    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    dep = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"dry_run": True, "checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )
    assert dep.status_code == 200, dep.text
    assert dep.json()["state"] == "dry_run"
    assert "mode" not in dep.json()
    # Dry-run deploys never show up as active deployments.
    assert api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"] == []


def test_user_key_undeploy_returns_public_persisted_deployment(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "deployed")
    revision = f"{run_id}/final"
    status = runner_status.get_status(run_id)
    status.deployment = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "checkpoint_id": revision,
        "previous_deployment": {"state": "ready", "endpoint_name": "https://old.example"},
        "verification_generation": 7,
    }
    runner_state._save_status(status)
    monkeypatch.setattr(
        app_mod,
        "undeploy_adapter",
        lambda target, **_: {
            "checkpoint_id": target,
            "disabled_checkpoints": [target],
            "serving_deregistered": True,
        },
    )

    response = api.delete(
        f"/v1/runs/{run_id}/deploy?checkpoint_id={revision}", headers=_bearer(key)
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "undeployed"
    assert body["checkpoint_id"] == revision
    assert body["openai_model"] == revision
    assert body["run_id"] == run_id
    assert body["disabled_checkpoints"] == [revision]
    assert body["serving_deregistered"] is True
    persisted = runner_status.get_status(run_id)
    assert persisted.state == "done"
    assert persisted.deployment["state"] == "undeployed"
    for field in ("disabled_checkpoints", "serving_deregistered"):
        body.pop(field)
    assert body == api.get(f"/v1/runs/{run_id}/deploy", headers=_bearer(key)).json()


def test_sibling_undeploy_returns_exact_removed_checkpoint(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "deployed")
    current = f"{run_id}/final"
    sibling = f"{run_id}/step-20"
    status = runner_status.get_status(run_id)
    status.deployment = {
        "state": "ready",
        "checkpoint_id": current,
        "checkpoint_step": None,
        "verified_at": 123.0,
    }
    runner_state._save_status(status)
    for checkpoint_id in (current, sibling):
        runner_verified_revisions.add_verified_checkpoint(
            run_id,
            checkpoint_id,
            expected_generation=runner_verified_revisions.verified_checkpoint_generation(run_id),
        )
    monkeypatch.setattr(
        app_mod,
        "undeploy_adapter",
        lambda target, **_: {
            "checkpoint_id": target,
            "disabled_checkpoints": [target],
            "serving_deregistered": False,
        },
    )

    response = api.delete(f"/v1/runs/{run_id}/deploy?checkpoint_id={sibling}", headers=_bearer(key))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {
        "run_id": run_id,
        "checkpoint_step": 20,
        "checkpoint_id": sibling,
        "state": "undeployed",
        "verified_at": None,
        "openai_model": sibling,
        "disabled_checkpoints": [sibling],
        "serving_deregistered": False,
    }
    persisted = runner_status.get_status(run_id)
    assert persisted.state == "deployed"
    assert persisted.deployment["state"] == "ready"
    assert persisted.deployment["checkpoint_id"] == current


@pytest.mark.parametrize(
    "retired_model",
    ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"],
)
def test_hosted_undeploy_preserves_historical_removed_model_cleanup(
    api, monkeypatch, retired_model
):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "deployed")
    status = runner_status.get_status(run_id)
    status.spec = {**status.spec, "model": retired_model}
    status.deployment = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "checkpoint_id": f"{run_id}/final",
    }
    runner_state._save_status(status)
    calls = []
    monkeypatch.setattr(app_mod, "undeploy_adapter", lambda target, **_: calls.append(target) or {})

    response = api.delete(
        f"/v1/runs/{run_id}/deploy?checkpoint_id={run_id}/final", headers=_bearer(key)
    )

    assert response.status_code == 200, response.text
    assert calls == [f"{run_id}/final"]
    assert runner_status.get_status(run_id).deployment["state"] == "undeployed"


def test_internal_org_undeploy_returns_public_persisted_deployment(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    owner = _login()
    org = f"org-{owner.removeprefix(_USER_PREFIX)}"
    project = "22222222-2222-4222-8222-222222222222"
    run_id = api.post(
        "/v1/runs",
        json={"spec": {**SPEC, "project": project}, "dry_run": True},
        headers=_bearer(owner),
    ).json()["run_id"]
    revision = f"{run_id}/step-40"
    status = runner_status.get_status(run_id)
    status.state = "deployed"
    status.deployment = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "checkpoint_id": revision,
        "checkpoint_step": 40,
        "verified_at": 123.0,
        "previous_deployment": {"state": "ready", "endpoint_name": "https://old.example"},
        "verification_generation": 9,
    }
    runner_state._save_status(status)
    monkeypatch.setattr(
        app_mod,
        "undeploy_adapter",
        lambda target, **_: {
            "checkpoint_id": target,
            "disabled_checkpoints": [target],
            "serving_deregistered": True,
        },
    )
    headers = {
        **_bearer("fslo-internal-test"),
        "X-Freesolo-Org-Id": org,
        "X-Freesolo-Project-Id": project,
    }

    response = api.delete(f"/v1/runs/{run_id}/deploy?checkpoint_id={revision}", headers=headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "undeployed"
    assert body["checkpoint_id"] == revision
    assert body["checkpoint_step"] == 40
    assert body["verified_at"] == 123.0
    assert body["run_id"] == run_id
    assert body["openai_model"] == revision
    assert body["disabled_checkpoints"] == [revision]
    assert body["serving_deregistered"] is True
    persisted = runner_status.get_status(run_id)
    assert persisted.state == "done"
    assert persisted.deployment["state"] == "undeployed"
    for field in ("disabled_checkpoints", "serving_deregistered"):
        body.pop(field)
    assert body == api.get(f"/v1/runs/{run_id}/deploy", headers=headers).json()


def test_internal_deployment_management_rejects_malformed_persisted_projects(api):

    owner = _login()
    org = f"org-{owner.removeprefix(_USER_PREFIX)}"
    project = "22222222-2222-4222-8222-222222222222"
    run_id = api.post(
        "/v1/runs",
        json={"spec": {**SPEC, "project": project}, "dry_run": True},
        headers=_bearer(owner),
    ).json()["run_id"]
    internal = {
        **_bearer("fslo-internal-test"),
        "X-Freesolo-Org-Id": org,
        "X-Freesolo-Project-Id": project,
    }
    unknown = {"detail": f"unknown run_id: {run_id}"}
    missing = object()

    for persisted_project in (
        123,
        [project],
        {"id": project},
        None,
        "",
        "   ",
        "project-wrong",
        missing,
    ):
        status = runner_status.get_status(run_id)
        if persisted_project is missing:
            status.spec.pop("project", None)
        else:
            status.spec["project"] = persisted_project
        runner_state._save_status(status)

        responses = (
            api.get(f"/v1/runs/{run_id}/deploy", headers=internal),
            api.post(
                f"/v1/runs/{run_id}/deploy",
                json={"dry_run": True, "checkpoint_id": f"{run_id}/final"},
                headers=internal,
            ),
            api.delete(f"/v1/runs/{run_id}/deploy?checkpoint_id={run_id}/final", headers=internal),
        )
        for response in responses:
            assert response.status_code == 404
            assert response.json() == unknown


def test_internal_deployment_management_rejects_missing_run_org(api):

    owner = _login()
    project = "22222222-2222-4222-8222-222222222222"
    run_id = api.post(
        "/v1/runs",
        json={"spec": {**SPEC, "project": project}, "dry_run": True},
        headers=_bearer(owner),
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.platform_context = None
    status.billing_context = None
    runner_state._save_status(status)

    internal = {
        **_bearer("fslo-internal-test"),
        "X-Freesolo-Org-Id": "org-anything",
        "X-Freesolo-Project-Id": project,
    }
    assert api.get(f"/v1/runs/{run_id}/deploy", headers=internal).status_code == 404
    assert (
        api.post(
            f"/v1/runs/{run_id}/deploy",
            json={"dry_run": True, "checkpoint_id": f"{run_id}/final"},
            headers=internal,
        ).status_code
        == 404
    )
    assert (
        api.delete(
            f"/v1/runs/{run_id}/deploy?checkpoint_id={run_id}/final", headers=internal
        ).status_code
        == 404
    )
    assert api.get(f"/v1/runs/{run_id}/deploy", headers=_bearer(owner)).status_code == 200


def test_internal_owned_run_still_requires_matching_org_for_deployment_management(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    internal = _bearer("fslo-internal-test")
    project = "33333333-3333-4333-8333-333333333333"
    run_id = api.post(
        "/v1/runs",
        json={"spec": {**SPEC, "project": project}, "dry_run": True},
        headers=internal,
    ).json()["run_id"]
    internal_key = _db_mod.lookup_key("fslo-internal-test")
    assert internal_key is not None
    assert _db_mod.run_owner(run_id) == internal_key["id"]

    org = "org-internal-owner"
    status = runner_status.get_status(run_id)
    status.platform_context = {"org_id": org}
    status.billing_context = None
    runner_state._save_status(status)

    with open(os.path.join(runner_state.RUNS_DIR, f"{run_id}.log"), "w") as f:
        f.write("internal owner log\n")
    monkeypatch.setattr(
        app_mod,
        "_worker_artifacts",
        lambda _spec: {"console_sft.txt": "internal owner worker\n"},
    )
    assert api.get(f"/v1/runs/{run_id}/logs", headers=internal).status_code == 200
    assert api.get(f"/v1/runs/{run_id}/worker", headers=internal).status_code == 200

    calls = {"deploy": 0, "undeploy": 0}

    def fake_deploy(**kwargs):
        calls["deploy"] += 1
        return _FakeDeployment(kwargs["adapter_prefix"])

    def fake_undeploy(target, **_):
        calls["undeploy"] += 1
        return {"run_id": target}

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(app_mod, "undeploy_adapter", fake_undeploy)

    for headers in (
        internal,
        {
            **internal,
            "X-Freesolo-Org-Id": "org-wrong",
            "X-Freesolo-Project-Id": project,
        },
        {**internal, "X-Freesolo-Org-Id": org},
        {
            **internal,
            "X-Freesolo-Org-Id": org,
            "X-Freesolo-Project-Id": "project-wrong",
        },
    ):
        assert api.get(f"/v1/runs/{run_id}/deploy", headers=headers).status_code == 404
        assert (
            api.post(
                f"/v1/runs/{run_id}/deploy",
                json={"dry_run": True, "checkpoint_id": f"{run_id}/final"},
                headers=headers,
            ).status_code
            == 404
        )
        assert (
            api.delete(
                f"/v1/runs/{run_id}/deploy?checkpoint_id={run_id}/final", headers=headers
            ).status_code
            == 404
        )
    assert calls == {"deploy": 0, "undeploy": 0}

    matched = {
        **internal,
        "X-Freesolo-Org-Id": org,
        "X-Freesolo-Project-Id": project,
    }
    assert api.get(f"/v1/runs/{run_id}/deploy", headers=matched).status_code == 200
    assert (
        api.post(
            f"/v1/runs/{run_id}/deploy",
            json={"dry_run": True, "checkpoint_id": f"{run_id}/final"},
            headers=matched,
        ).status_code
        == 200
    )
    assert (
        api.delete(
            f"/v1/runs/{run_id}/deploy?checkpoint_id={run_id}/final", headers=matched
        ).status_code
        == 200
    )
    assert calls == {"deploy": 1, "undeploy": 1}


def test_deploy_allows_runner_assigned_revision_pin(api):
    """An SFT run pinned by the runner stays deployable."""
    from flash.core.spec import JobSpec

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    # reproduce the shape a REAL auto-pinned submit persists, which is asymmetric: submit stores
    # `spec=public_spec.to_dict()`, and to_dict() strips a runner-assigned pin along with its
    # marker, so the public half carries neither and only the worker half does. Writing either onto
    # status.spec would be a shape production cannot produce, and it would paper over
    # `_validate_effective_spec` rejecting the real one -- a 409 that leaves auto-pinned runs just
    # as undeployable as the 400 did.
    assert "model_revision_auto" not in status.spec, status.spec
    assert "model_revision_force_pin" not in status.spec, status.spec
    assert not status.spec.get("model_revision"), status.spec
    snapshot = status.effective_preparation
    assert isinstance(snapshot, dict), snapshot
    assert snapshot["worker_spec"]["model_revision_force_pin"] is False
    snapshot["worker_spec"]["model_revision"] = "a" * 40
    snapshot["worker_spec"]["model_revision_auto"] = True
    # re-digest the way submit does. the marker is a privilege input the deploy guard reads, so it
    # is bound to the digest; a fixture that skipped this would be forging one, which is what
    # `test_deploy_rejects_a_forged_auto_pin_marker` covers.
    snapshot["preparation_digest"] = runner_preparation._preparation_digest(
        JobSpec.from_dict(status.spec),
        JobSpec.from_dict(snapshot["worker_spec"]),
        snapshot.get("adapter_identity"),
    )
    runner_state._save_status(status)

    response = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"dry_run": True, "checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )

    # a coherent auto-pinned run deploys, so assert the success itself rather than the absence of
    # one particular rejection
    assert response.status_code == 200, response.json()
    assert "revision-pinned" not in json.dumps(response.json())


def test_deploy_rejects_a_forged_auto_pin_marker(api):
    """A worker-only pin written without re-digesting fails integrity validation."""

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    snapshot = status.effective_preparation
    assert isinstance(snapshot, dict), snapshot
    digest_before = snapshot["preparation_digest"]
    snapshot["worker_spec"]["model_revision"] = "a" * 40
    snapshot["worker_spec"]["model_revision_auto"] = True
    assert snapshot["preparation_digest"] == digest_before  # forged: no re-digest
    runner_state._save_status(status)

    response = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"dry_run": True, "checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )

    assert response.status_code == 409, response.json()
    assert "integrity validation" in response.json()["detail"]


def test_public_run_routes_redact_private_deployment_fields(api, monkeypatch):
    import flash.serve.deployment.deploy as deploy_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    revision = f"{run_id}/final"
    status.deployment = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "openai_base_url": "https://serve.example/v1",
        "previous_deployment": {"state": "ready", "endpoint_name": "https://old.example"},
        "checkpoint_id": revision,
    }
    runner_state._save_status(status)
    runner_verified_revisions.add_verified_checkpoint(
        run_id,
        revision,
        expected_generation=runner_verified_revisions.verified_checkpoint_generation(run_id),
    )

    responses = [
        api.get(f"/v1/runs/{run_id}", headers=_bearer(key)).json(),
        api.get("/v1/runs", headers=_bearer(key)).json()["runs"][0],
        api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"][0],
    ]
    for body in responses:
        deployment = body["deployment"]
        assert deployment["openai_base_url"] == "https://serve.example/v1"
        assert "previous_deployment" not in deployment

    persisted = runner_status.get_status(run_id).deployment
    assert persisted["previous_deployment"]["endpoint_name"] == "https://old.example"
    assert persisted["openai_base_url"] == "https://serve.example/v1"
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset({revision})

    monkeypatch.setattr(deploy_mod, "undeploy_adapter", lambda target, **_: [target])
    cancelled = api.post(f"/v1/runs/{run_id}/cancel", headers=_bearer(key))

    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["deployment"]["state"] == "undeployed"
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset()
    assert api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"] == []


def test_deploy_uses_effective_warmstart_rank(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    public_spec = {**status.spec, "train": {**status.spec["train"]}}
    public_spec["train"].update({"init_from_adapter": "source-run/final", "lora_rank": 8})
    worker_spec = {**public_spec, "train": {**public_spec["train"]}}
    worker_spec["train"].update(
        {
            "init_from_adapter": "private-owner/private-repo:sft/source-run",
            "init_from_adapter_revision": "a" * 40,
            "lora_rank": 64,
        }
    )
    identity = {"digest": "immutable-v1"}
    status.spec = public_spec
    status.effective_preparation = {
        "worker_spec": worker_spec,
        "adapter_identity": identity,
        "version": 1,
        "preparation_digest": runner_preparation._preparation_digest(
            runner_spec.JobSpec.from_dict(public_spec),
            runner_spec.JobSpec.from_dict(worker_spec),
            identity,
        ),
    }
    runner_state._save_status(status)
    seen = {}

    def fake_deploy(**kwargs):
        seen.update(kwargs)
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)

    resp = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"dry_run": True, "checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )

    assert resp.status_code == 200, resp.text
    assert seen["lora_rank"] == 64
    public = api.get(f"/v1/runs/{run_id}", headers=_bearer(key)).json()
    assert "lora_rank" not in public["spec"]["train"]
    assert public["spec"]["train"]["init_from_adapter"] == "source-run/final"
    assert "effective_preparation" not in public


def test_public_spec_does_not_publish_a_storage_ref_whose_phase_was_removed():
    """Redaction must not infer "public" from "this build cannot parse it as internal".

    A persisted worker/effective spec keeps whatever phase was current when it was written. `opsd`
    was removed from the internal-ref grammar (#784), so its locators stopped parsing as internal
    and the redactor left them alone as though they were user-facing refs -- publishing the private
    repo verbatim.
    """

    for ref in ("private-owner/private-source:opsd/source-run", "private-owner/private-source:!"):
        data = runner_state._public_status_spec({"train": {"init_from_adapter": ref}})
        assert "private-owner" not in json.dumps(data), ref
        assert "init_from_adapter" not in data["train"], ref

    # the two grammars this function must still recognize are unaffected: a user-facing ref is
    # preserved, and a known internal phase is rewritten rather than dropped.
    assert (
        runner_state._public_status_spec({"train": {"init_from_adapter": "source-run/step-20"}})[
            "train"
        ]["init_from_adapter"]
        == "source-run/step-20"
    )
    assert (
        runner_state._public_status_spec(
            {"train": {"init_from_adapter": "private-owner/private-source:sft/source-run"}}
        )["train"]["init_from_adapter"]
        == "source-run/final"
    )


def test_malformed_legacy_warmstart_spec_drops_both_topology_keys():
    """The malformed-record fallback must strip alpha as well as rank for a warm start.

    Both keys are rejected alongside `init_from_adapter`, so surfacing either in a public status
    spec yields a payload that cannot be re-submitted. Alpha only reaches this branch now that it
    is user-authorable: while it was managed, `to_dict()` stripped it unconditionally.
    """

    train = runner_state._public_status_spec(
        {
            "train": {
                "init_from_adapter": "source-run/final",
                "lora_rank": 16,
                "lora_alpha": 32,
            },
            "unparseable": object(),
        }
    )["train"]
    assert "lora_rank" not in train
    assert "lora_alpha" not in train

    # a non-warm-start malformed record keeps an authored alpha: it is submittable there.
    kept = runner_state._public_status_spec(
        {"train": {"lora_rank": 16, "lora_alpha": 48}, "unparseable": object()}
    )["train"]
    assert kept["lora_alpha"] == 48


def test_deploy_serving_error_is_recorded_as_failed_deployment(api, monkeypatch):
    """A serving-backend failure during deploy is recorded on the deployment status."""
    import flash.server.asgi.app as app_mod
    from flash.serve.contract.errors import ServingError

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    # Make the run real-deployable: flip its persisted state to "done" (a finished run with
    # trained adapter artifacts). Ownership lives in the DB, so this only changes the gate.
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)

    # The serving backend rejects the registration (e.g. upstream 5xx). deploy_adapter is
    # imported into the app namespace, so patch it there.
    def boom(**kwargs):
        raise ServingError("serving backend unreachable: no engine for base model")

    monkeypatch.setattr(app_mod, "deploy_adapter", boom)

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "failed"
    assert "serving backend unreachable" in resp.json()["error"]
    deployments = api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"]
    assert deployments[0]["deployment"]["state"] == "failed"
    assert "serving backend unreachable" in deployments[0]["deployment"]["error"]


def test_deployment_failure_persisted_matches_default_error():
    from flash.server.routes import serving

    previous = {"state": "ready", "endpoint_name": "https://serve.example"}
    failed = {"state": "failed", "error": "", "previous_deployment": previous}
    status = SimpleNamespace(
        deployment={
            **previous,
            "last_deploy_error": "deployment failed",
            "last_deploy_failed_at": time.time(),
        }
    )

    assert serving._deployment_failure_persisted(status, failed)


def test_deployment_transitions_report_persisted_states_and_skip_dry_run(api, monkeypatch):
    import flash.server.asgi.app as app_mod
    import flash.server.domain.registry.runs as registry

    key = _login()
    run_id = _make_run(api, key, "done")
    revision = f"{run_id}/final"
    calls = []
    monkeypatch.setattr(
        registry,
        "record_training_run",
        lambda **kwargs: calls.append(kwargs["status"]) or True,
    )

    def fake_deploy(**kwargs):
        if not kwargs["dry_run"]:
            kwargs["before_ready"](revision, revision)
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: _smoke_chat_result(revision, revision),
    )

    dry_run = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"dry_run": True, "checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )
    assert dry_run.status_code == 200, dry_run.text
    assert calls == []

    response = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert response.status_code == 200, response.text
    assert [status.deployment["state"] for status in calls] == [
        "queued",
        "smoke_testing",
        "ready",
    ]
    # this fixture's SPEC is a text-only grpo run, so no multimodal targeting is recorded and the
    # smoke falls back to the text challenge. verify_kind follows the adapter's recorded modality,
    # not the base model's image capability, and lora-request attestation is collected only on the
    # image path -- so the text smoke records no attested revision.
    assert calls[-1].deployment["verify_kind"] == "fixed_prompt"
    assert "verify_lora_request_adapter" not in calls[-1].deployment


def test_deployment_reporting_skips_failed_cas_and_reports_failure(api, monkeypatch):
    import flash.server.asgi.app as app_mod
    import flash.server.domain.registry.runs as registry
    from flash.serve.contract.errors import ServingError
    from flash.server.routes import serving

    key = _login()
    run_id = _make_run(api, key, "done")
    calls = []
    monkeypatch.setattr(
        registry,
        "record_training_run",
        lambda **kwargs: calls.append(kwargs["status"]) or True,
    )

    real_mark_pending = serving.mark_deployment_pending
    monkeypatch.setattr(
        serving, "mark_deployment_pending", lambda *_a, **_k: runner_status.get_status(run_id)
    )
    rejected = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert rejected.status_code == 409, rejected.text
    assert calls == []

    monkeypatch.setattr(serving, "mark_deployment_pending", real_mark_pending)
    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **_kwargs: (_ for _ in ()).throw(ServingError("serving unavailable")),
    )
    failed = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert failed.status_code == 200, failed.text
    assert [status.deployment["state"] for status in calls] == ["queued", "failed"]


def test_deployment_job_shutdown_closes_producers(monkeypatch):
    from threading import Event

    import flash.server.asgi.app as app_mod

    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    app_mod._open_deployment_jobs()
    started = Event()
    release = Event()

    def job():
        started.set()
        assert release.wait(2)

    assert app_mod.start_deployment_job(job) is False
    assert started.wait(1)
    assert app_mod._wait_for_deployment_jobs(0.01) is False
    with pytest.raises(RuntimeError, match="shutting down"):
        app_mod.start_deployment_job(lambda: None)
    release.set()
    assert app_mod._wait_for_deployment_jobs(1) is True
    app_mod._open_deployment_jobs()


def test_recover_deployments_recovers_busy_states_on_startup_regardless_of_age(
    monkeypatch,
):
    """Startup recovery fails every busy record whose lock it can take, fresh ones included.

    The per-run flock is the ownership proof: a live lifecycle holds it from the moment the
    request persists `queued` until the deployment ends, so acquiring it means no owner survived.
    Recovery used to also require the record to be 30 minutes old, which left a control plane that
    restarted inside that window with a busy record nothing was working on -- and nothing to
    revisit it, since recovery only runs at startup -- so every retry got 409 until the clock aged
    it out.
    """
    from flash.server.routes import serving

    now = time.time()
    statuses = {
        "run-smoke_testing": SimpleNamespace(
            run_id="run-smoke_testing",
            state="done",
            deployment={
                "state": "smoke_testing",
                "updated_at": now - serving._DEPLOYMENT_STALE_SECONDS - 1,
            },
        ),
        "run-queued": SimpleNamespace(
            run_id="run-queued",
            state="done",
            deployment={"state": "queued", "updated_at": now},
        ),
    }
    marked = []
    reported = []
    monkeypatch.setattr(
        serving.db,
        "all_runs",
        lambda: [{"run_id": status.run_id} for status in statuses.values()],
    )
    monkeypatch.setattr(serving._app, "get_status", lambda run_id: statuses[run_id])

    def mark_failed(run_id, deployment):
        marked.append((run_id, deployment))
        return SimpleNamespace(run_id=run_id, state="done", deployment=deployment)

    monkeypatch.setattr(runner_transitions, "mark_deployment_failed", mark_failed)
    monkeypatch.setattr(
        serving,
        "_report_persisted_transition",
        lambda previous, current, *, persisted: reported.append(
            (previous.run_id, current.deployment, persisted)
        ),
    )

    # both current busy states are recovered, including the just-written queued record.
    assert serving.recover_deployments() == 2
    assert sorted(run_id for run_id, _deployment in marked) == [
        "run-queued",
        "run-smoke_testing",
    ]
    assert all(deployment["state"] == "failed" for _run_id, deployment in marked)
    assert all("control-plane restart" in deployment["error"] for _run_id, deployment in marked)
    assert sorted(run_id for run_id, _deployment, persisted in reported if persisted) == [
        "run-queued",
        "run-smoke_testing",
    ]


def test_deployment_job_is_started_before_waiters_can_observe_it(monkeypatch):
    import flash.server.asgi.app as app_mod

    started_while_locked = []

    class ThreadStub:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            started_while_locked.append(app_mod._DEPLOYMENT_JOBS_LOCK.locked())

    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    monkeypatch.setattr(app_mod.threading, "Thread", ThreadStub)
    app_mod._open_deployment_jobs()

    assert app_mod.start_deployment_job(lambda: None) is False
    assert started_while_locked == [True]
    with app_mod._DEPLOYMENT_JOBS_LOCK:
        app_mod._DEPLOYMENT_JOBS.clear()


def test_deploy_returns_deploying_before_background_job_finishes(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)

    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    started: list[tuple[object, tuple, dict]] = []
    reported = []

    def fake_start(target, *args, **kwargs):
        started.append((target, args, kwargs))
        kwargs["deploy_lock"].release()
        return False

    monkeypatch.setattr(app_mod, "start_deployment_job", fake_start)
    monkeypatch.setattr(runner_reporting, "_report_status_async", reported.append)
    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **_k: pytest.fail("deploy_adapter must run in the background job"),
    )

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "queued"
    assert resp.json()["verify"] is True
    assert len(started) == 1
    deployment_target, deployment_args, deployment_kwargs = started[0]
    assert deployment_target.__name__ == "_finish_deployment"
    assert deployment_args == ()
    assert deployment_kwargs["run_id"] == run_id
    assert deployment_kwargs["deploy_kwargs"]["adapter_prefix"].endswith(run_id)
    assert [status.deployment["state"] for status in reported] == ["queued"]

    deployment = runner_status.get_status(run_id).deployment
    assert deployment["state"] == "queued"
    deployments = api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"]
    assert deployments[0]["deployment"]["state"] == "queued"


def test_concurrent_deploy_returns_409_without_queueing_duplicate(api, monkeypatch):
    import threading

    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)

    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    starts: list[dict] = []

    def fake_start(_target, *_args, **kwargs):
        starts.append(kwargs)
        if len(starts) > 1:
            kwargs["deploy_lock"].release()
        return False

    monkeypatch.setattr(app_mod, "start_deployment_job", fake_start)

    first = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert first.status_code == 200, first.text
    assert first.json()["state"] == "queued"
    assert len(starts) == 1

    responses = []
    errors: list[BaseException] = []

    def deploy_again():
        try:
            responses.append(
                api.post(
                    f"/v1/runs/{run_id}/deploy",
                    json={"checkpoint_id": f"{run_id}/final"},
                    headers=_bearer(key),
                )
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=deploy_again)
    worker.start()
    worker.join(timeout=1)
    blocked = worker.is_alive()

    settled = runner_status.get_status(run_id)
    settled.deployment = {**settled.deployment, "state": "ready", "updated_at": time.time()}
    runner_state._save_status(settled)
    starts[0]["deploy_lock"].release()
    worker.join(timeout=5)

    assert worker.is_alive() is False
    assert errors == []
    assert blocked is False
    assert len(responses) == 1
    assert responses[0].status_code == 409, responses[0].text
    assert responses[0].json()["detail"] == (
        f"run {run_id} already has a deployment in queued state; "
        "run `flash models deployments` to check progress"
    )
    assert len(starts) == 1


@pytest.mark.parametrize("deployment_state", ["ready", None])
def test_concurrent_non_deploy_operation_returns_generic_409(api, monkeypatch, deployment_state):
    import flash.server.asgi.app as app_mod

    starts: list[dict] = []

    def fake_start(_target, *_args, **kwargs):
        starts.append(kwargs)
        return False

    monkeypatch.setattr(app_mod, "start_deployment_job", fake_start)

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    status.deployment = {"state": deployment_state} if deployment_state else None
    runner_state._save_status(status)

    deploy_lock = app_mod._deploy_lock(run_id)
    assert deploy_lock.acquire(blocking=False) is True
    try:
        response = api.post(
            f"/v1/runs/{run_id}/deploy",
            json={"checkpoint_id": f"{run_id}/final"},
            headers=_bearer(key),
        )
    finally:
        deploy_lock.release()

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail == f"another operation is in progress for run {run_id}; retry shortly"
    assert "deployment in" not in detail
    assert "flash models deployments" not in detail
    assert starts == []


def test_deploy_holds_lock_through_background_job_handoff(api, monkeypatch):
    import queue
    import threading

    import flash.server.asgi.app as app_mod
    from flash.server.routes import serving

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)

    observations: queue.Queue[str] = queue.Queue()
    allow_lifecycle = threading.Event()
    lifecycle_started = threading.Event()
    phases: list[str] = []
    recovered: list[int] = []
    observed_states: list[str] = []
    worker_errors: list[BaseException] = []

    def hold_lifecycle(**_kwargs):
        lifecycle_started.set()
        observations.put("lifecycle-started")
        assert allow_lifecycle.wait(timeout=5)

    monkeypatch.setattr(serving, "_finish_deployment_unlocked", hold_lifecycle)

    class ObservedDeployLock:
        def __init__(self, lock):
            self._lock = lock

        def release(self):
            self._lock.release()
            if not lifecycle_started.is_set():
                observations.put("released-before-lifecycle")
                assert allow_lifecycle.wait(timeout=5)

    def fake_start(target, *args, **kwargs):
        kwargs["deploy_lock"] = ObservedDeployLock(kwargs["deploy_lock"])

        def run_target():
            try:
                target(*args, **kwargs)
            except BaseException as exc:
                worker_errors.append(exc)

        worker = threading.Thread(target=run_target)
        worker.start()
        phases.append(observations.get(timeout=5))
        recovered.append(serving.recover_deployments())
        observed_states.append(runner_status.get_status(run_id).deployment["state"])
        allow_lifecycle.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert worker_errors == []
        return False

    monkeypatch.setattr(app_mod, "start_deployment_job", fake_start)

    response = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )

    assert response.status_code == 200, response.text
    assert recovered == [0]
    assert observed_states == ["queued"]
    assert phases == ["lifecycle-started"]
    assert runner_status.get_status(run_id).deployment["state"] == "queued"


def test_deploy_start_failure_persists_terminal_failure(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    reported = []

    def reject_job(*_args, **_kwargs):
        raise app_mod.DeploymentJobStartError("deployment jobs are shutting down")

    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    monkeypatch.setattr(app_mod, "start_deployment_job", reject_job)
    monkeypatch.setattr(runner_reporting, "_report_status_async", reported.append)

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )

    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"]["code"] == "deployment_job_unavailable"
    deployment = runner_status.get_status(run_id).deployment
    assert deployment["state"] == "failed"
    assert deployment["retryable"] is True
    assert "shutting down" in deployment["error"]
    assert [item.deployment["state"] for item in reported] == ["queued", "failed"]


def test_sync_deploy_execution_error_keeps_specific_persisted_outcome(api, monkeypatch):
    from flash.server.routes import serving

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)

    def fail_after_start(**_kwargs):
        current = runner_status.get_status(run_id)
        failed = {
            **current.deployment,
            "state": "failed",
            "error": "specific inline deployment failure",
        }
        runner_transitions.mark_deployment_failed(run_id, failed)
        raise RuntimeError("late status mirror failure")

    monkeypatch.setattr(serving, "_finish_deployment_unlocked", fail_after_start)

    with pytest.raises(RuntimeError, match="late status mirror failure"):
        api.post(
            f"/v1/runs/{run_id}/deploy",
            json={"checkpoint_id": f"{run_id}/final"},
            headers=_bearer(key),
        )

    deployment = runner_status.get_status(run_id).deployment
    assert deployment["state"] == "failed"
    assert deployment["error"] == "specific inline deployment failure"


def test_replay_status_reports_mirrors_all_persisted_outcomes_sequentially(monkeypatch):
    from flash.server.routes import serving

    ready = SimpleNamespace(run_id="run-ready", deployment={"state": "ready"})
    complete = SimpleNamespace(run_id="run-complete", deployment=None)
    reported = []
    monkeypatch.setattr(
        serving.db, "all_runs", lambda: [{"run_id": "run-ready"}, {"run_id": "run-complete"}]
    )
    monkeypatch.setattr(
        serving._app,
        "get_status",
        lambda run_id: ready if run_id == "run-ready" else complete,
    )

    monkeypatch.setattr(runner_reporting, "_report_status", reported.append)

    assert serving.replay_status_reports() == 2
    assert reported == [ready, complete]


def test_replay_status_reports_skips_unreadable_records_and_continues(monkeypatch):
    from flash.server.routes import serving

    first = SimpleNamespace(run_id="run-first")
    last = SimpleNamespace(run_id="run-last")
    outcomes = {
        "run-first": first,
        "run-missing": FileNotFoundError("missing"),
        "run-os-error": OSError("unreadable"),
        "run-type-error": TypeError("invalid type"),
        "run-value-error": ValueError("invalid value"),
        "run-last": last,
    }
    reported = []
    monkeypatch.setattr(serving.db, "all_runs", lambda: [{"run_id": run_id} for run_id in outcomes])

    def get_status(run_id):
        outcome = outcomes[run_id]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(serving._app, "get_status", get_status)
    monkeypatch.setattr(runner_reporting, "_report_status", reported.append)

    assert serving.replay_status_reports() == 2
    assert reported == [first, last]


def test_replay_status_reports_repairs_malformed_report_sequence_and_continues(monkeypatch):
    from flash.server.routes import serving

    malformed = runner_status._runstatus_from_json(
        {
            "run_id": "run-malformed-sequence",
            "state": "done",
            "spec": SPEC,
            "report_sequence": "not-an-integer",
        }
    )
    valid = runner_status._runstatus_from_json(
        {
            "run_id": "run-valid-sequence",
            "state": "done",
            "spec": SPEC,
            "report_sequence": 1,
        }
    )
    statuses = {status.run_id: status for status in (malformed, valid)}
    delivered = []
    monkeypatch.setattr(serving.db, "all_runs", lambda: [{"run_id": key} for key in statuses])
    monkeypatch.setattr(serving._app, "get_status", statuses.__getitem__)
    monkeypatch.setattr(runner_reporting, "_send_status_report", delivered.append)

    runner_reporting._shutdown_status_reporter()
    runner_reporting._open_status_reporter()
    try:
        assert serving.replay_status_reports() == 2
        assert delivered == [malformed, valid]
    finally:
        runner_reporting._shutdown_status_reporter()


def test_replay_status_reports_stops_between_items(monkeypatch):
    from threading import Event

    from flash.server.routes import serving

    stop = Event()
    statuses = {
        "run-a": SimpleNamespace(run_id="run-a"),
        "run-b": SimpleNamespace(run_id="run-b"),
    }
    reported = []
    monkeypatch.setattr(serving.db, "all_runs", lambda: [{"run_id": key} for key in statuses])
    monkeypatch.setattr(serving._app, "get_status", statuses.__getitem__)

    def report(status):
        reported.append(status)
        stop.set()

    monkeypatch.setattr(runner_reporting, "_report_status", report)

    assert serving.replay_status_reports(stop) == 1
    assert reported == [statuses["run-a"]]


def test_replay_status_reports_continues_after_report_failure(monkeypatch):
    from flash.server.routes import serving

    failing = SimpleNamespace(run_id="run-failing", report_sequence=1)
    valid = SimpleNamespace(run_id="run-valid", report_sequence=2)
    statuses = {status.run_id: status for status in (failing, valid)}
    reported = []
    monkeypatch.setattr(
        serving.db,
        "all_runs",
        lambda: [{"run_id": run_id} for run_id in statuses],
    )
    monkeypatch.setattr(serving._app, "get_status", statuses.__getitem__)

    def report(status):
        if status.run_id == "run-failing":
            raise ValueError("invalid stored report")
        runner_reporting._status_report_sequence_unlocked(status)
        reported.append(status)

    monkeypatch.setattr(runner_reporting, "_report_status", report)

    assert serving.replay_status_reports() == 1
    assert reported == [valid]


def test_shutdown_flushes_after_status_replay_failure(monkeypatch):
    from threading import Event

    import flash.server.asgi.app as app_mod
    import flash.server.billing.retry as billing_retry
    import flash.server.domain.ops.reconcile as reconcile
    import flash.server.domain.ops.repo_cleanup as repo_cleanup
    from flash.providers.core import preflight
    from flash.server.routes import serving

    replay_started = Event()
    shutdown_steps = []

    def fail_replay(_stop):
        replay_started.set()
        raise ValueError("invalid stored run status")

    def wait_for_deployments(_timeout):
        shutdown_steps.append("deployment_jobs")
        return True

    def flush_status_reports(_timeout, *, close=False):
        shutdown_steps.append(("status_reports", close))
        return True

    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(preflight, "check_run_preflight", lambda: None)
    monkeypatch.setattr(app_mod, "_open_deployment_jobs", lambda: None)
    monkeypatch.setattr(runner_reporting, "_open_status_reporter", lambda: None)
    monkeypatch.setattr(app_mod, "recover_runs", lambda: None)
    monkeypatch.setattr(serving, "recover_deployments", lambda: 0)
    monkeypatch.setattr(serving, "replay_status_reports", fail_replay)
    monkeypatch.setattr(billing_retry, "charge_retry_enabled", lambda: False)
    monkeypatch.setattr(reconcile, "reconcile_enabled", lambda: False)
    monkeypatch.setattr(repo_cleanup, "repo_cleanup_enabled", lambda: False)
    monkeypatch.setattr(app_mod, "_instance_providers_configured", lambda: False)
    monkeypatch.setattr(app_mod, "_wait_for_deployment_jobs", wait_for_deployments)
    monkeypatch.setattr(runner_reporting, "_shutdown_status_reporter", flush_status_reports)

    with TestClient(app_mod.create_app()):
        assert replay_started.wait(1)

    assert shutdown_steps == ["deployment_jobs", ("status_reports", True)]


def test_status_report_sequence_is_persisted_and_private(api):

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    initial_sequence = status.report_sequence

    status.updated_at = time.time()
    runner_state._save_status(status)
    persisted = runner_status.get_status(run_id)

    assert persisted.report_sequence == initial_sequence + 1
    public = api.get(f"/v1/runs/{run_id}", headers=_bearer(key)).json()
    assert "report_sequence" not in public


def test_save_status_repairs_wrong_typed_report_sequence(api):

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    path = runner_state.runs_file_path(run_id, ".json")
    with open(path) as f:
        stored = json.load(f)
    stored["report_sequence"] = {"corrupt": True}
    with open(path, "w") as f:
        json.dump(stored, f)

    status = runner_status.get_status(run_id)
    runner_state._save_status(status)

    assert runner_status.get_status(run_id).report_sequence == 1


def test_legacy_replay_sequence_advances_first_post_restart_save(api, monkeypatch):
    from flash.server.routes import serving

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    path = runner_state.runs_file_path(run_id, ".json")
    with open(path) as f:
        stored = json.load(f)
    stored["report_sequence"] = 0
    with open(path, "w") as f:
        json.dump(stored, f)

    runner_reporting._shutdown_status_reporter()
    runner_reporting._open_status_reporter()
    for values in (
        runner_reporting._STATUS_REPORT_LAST_SENT,
        runner_reporting._STATUS_REPORT_LAST_ATTEMPTED,
        runner_reporting._STATUS_REPORT_LAST_QUEUED,
    ):
        values.pop(run_id, None)
    reported = []
    monkeypatch.setattr(serving.db, "all_runs", lambda: [{"run_id": run_id}])
    monkeypatch.setattr(
        runner_reporting, "_send_status_report", lambda status: reported.append(status) or True
    )

    try:
        assert serving.replay_status_reports() == 1
        assert runner_reporting._STATUS_REPORT_LAST_SENT[run_id] == 1

        updated = runner_status.get_status(run_id)
        runner_state._save_status(updated)
        assert updated.report_sequence == 2
        runner_reporting._report_status(updated)

        assert runner_reporting._STATUS_REPORT_LAST_SENT[run_id] == 2
        assert len(reported) == 2
    finally:
        runner_reporting._shutdown_status_reporter()


def test_completed_status_report_worker_restarts_orphaned_queue(monkeypatch):
    from collections import deque
    from concurrent.futures import Future
    from threading import Event

    runner_reporting._shutdown_status_reporter()
    run_id = "run-orphaned-report"
    completed = Future()
    completed.set_result(None)
    queued = SimpleNamespace(run_id=run_id)
    with runner_reporting._STATUS_REPORT_CONDITION:
        runner_reporting._STATUS_REPORT_QUEUES[run_id] = deque([(queued, 1, Event(), 1)])
        runner_reporting._STATUS_REPORT_WORKERS[run_id] = completed
        runner_reporting._STATUS_REPORT_ACTIVE.add(run_id)

    started = []

    def start_worker(current_run_id):
        started.append(current_run_id)
        runner_reporting._STATUS_REPORT_ACTIVE.add(current_run_id)
        return True

    monkeypatch.setattr(runner_reporting, "_start_status_report_worker_unlocked", start_worker)
    runner_reporting._discard_status_report_worker(run_id, completed)

    assert started == [run_id]
    assert run_id in runner_reporting._STATUS_REPORT_ACTIVE
    with runner_reporting._STATUS_REPORT_CONDITION:
        runner_reporting._STATUS_REPORT_QUEUES.pop(run_id, None)
        runner_reporting._STATUS_REPORT_WORKERS.pop(run_id, None)
        runner_reporting._STATUS_REPORT_ACTIVE.discard(run_id)


def test_drained_status_worker_does_not_clear_replacement_marker(monkeypatch):

    runner_reporting._shutdown_status_reporter()
    run_id = "run-replacement-report"
    injected = False
    original_notify_all = runner_reporting._STATUS_REPORT_CONDITION.notify_all

    def inject_replacement():
        nonlocal injected
        original_notify_all()
        if not injected and run_id not in runner_reporting._STATUS_REPORT_ACTIVE:
            runner_reporting._STATUS_REPORT_ACTIVE.add(run_id)
            injected = True

    monkeypatch.setattr(runner_reporting._STATUS_REPORT_CONDITION, "notify_all", inject_replacement)
    with runner_reporting._STATUS_REPORT_CONDITION:
        runner_reporting._STATUS_REPORT_ACTIVE.add(run_id)

    runner_reporting._drain_status_report_run(run_id)

    assert injected is True
    assert run_id in runner_reporting._STATUS_REPORT_ACTIVE
    with runner_reporting._STATUS_REPORT_CONDITION:
        runner_reporting._STATUS_REPORT_ACTIVE.discard(run_id)


def test_status_reports_preserve_run_order_without_cross_run_blocking(monkeypatch):
    from threading import Event

    runner_reporting._shutdown_status_reporter()
    run_a = "run-ordered-report-a"
    run_b = "run-ordered-report-b"
    for run_id in (run_a, run_b):
        runner_reporting._STATUS_REPORT_LAST_SENT.pop(run_id, None)
        runner_reporting._STATUS_REPORT_LAST_ATTEMPTED.pop(run_id, None)
        runner_reporting._STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
    started = Event()
    release = Event()
    run_b_reported = Event()
    reported = []

    def send(status):
        if status.run_id == run_a and status.report_sequence == 1:
            started.set()
            assert release.wait(2)
        reported.append((status.run_id, status.deployment["state"]))
        if status.run_id == run_b:
            run_b_reported.set()

    monkeypatch.setattr(runner_reporting, "_send_status_report", send)

    def status(run_id, state, sequence):
        return SimpleNamespace(
            run_id=run_id,
            state="done",
            updated_at=1.0,
            report_sequence=sequence,
            deployment={"state": state, "updated_at": 1.0},
        )

    runner_reporting._report_status_async(status(run_a, "queued", 1))
    assert started.wait(1)
    for sequence in range(2, 7):
        runner_reporting._report_status_async(status(run_a, f"state-{sequence}", sequence))
    runner_reporting._report_status_async(status(run_b, "running", 1))
    try:
        assert run_b_reported.wait(1)
        assert reported == [(run_b, "running")]
    finally:
        release.set()

    assert runner_reporting._wait_for_status_reports(2)
    assert [state for run_id, state in reported if run_id == run_a] == [
        "queued",
        "state-2",
        "state-3",
        "state-4",
        "state-5",
        "state-6",
    ]
    runner_reporting._shutdown_status_reporter()


def test_status_report_workers_recover_and_use_persisted_sequence(monkeypatch):

    runner_reporting._shutdown_status_reporter()
    run_id = "run-stale-report"
    runner_reporting._STATUS_REPORT_LAST_SENT.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_ATTEMPTED.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
    reported = []

    def send(status):
        if status.deployment["state"] == "broken":
            raise RuntimeError("report failed")
        reported.append(status)

    monkeypatch.setattr(runner_reporting, "_send_status_report", send)

    def status(state, sequence, updated_at):
        return SimpleNamespace(
            run_id=run_id,
            state="done",
            updated_at=updated_at,
            report_sequence=sequence,
            deployment={"state": state, "updated_at": updated_at},
        )

    runner_reporting._report_status_async(status("broken", 1, 10.0))
    runner_reporting._report_status(status("ready", 3, 1.0))
    runner_reporting._report_status_async(status("queued", 2, 20.0))
    runner_reporting._report_status(status("undeployed", 4, -1.0))
    assert runner_reporting._wait_for_status_reports(2)

    assert [item.deployment["state"] for item in reported] == ["ready", "undeployed"]
    runner_reporting._shutdown_status_reporter()


def test_sync_older_status_report_skips_newer_in_flight_sequence(monkeypatch):
    from threading import Event

    runner_reporting._shutdown_status_reporter()
    run_id = "run-report-race"
    runner_reporting._STATUS_REPORT_LAST_SENT.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_ATTEMPTED.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
    newer_started = Event()
    release_newer = Event()
    attempted = []

    def send(status):
        attempted.append(status.report_sequence)
        if len(attempted) == 1:
            newer_started.set()
            assert release_newer.wait(2)
        return False

    def status(sequence):
        return SimpleNamespace(
            run_id=run_id,
            state="done",
            updated_at=float(sequence),
            report_sequence=sequence,
            deployment={"state": f"state-{sequence}", "updated_at": float(sequence)},
        )

    monkeypatch.setattr(runner_reporting, "_send_status_report", send)
    runner_reporting._report_status_async(status(2))
    assert newer_started.wait(1)
    try:
        runner_reporting._report_status(status(1))
        assert attempted == [2]
    finally:
        release_newer.set()

    assert runner_reporting._wait_for_status_reports(2)
    assert attempted == [2, 2]
    runner_reporting._shutdown_status_reporter()


def test_sync_equal_status_report_waits_for_async_success_without_duplicate(monkeypatch):
    from threading import Event, Thread

    runner_reporting._shutdown_status_reporter()
    run_id = "run-report-equal-success"
    runner_reporting._STATUS_REPORT_LAST_SENT.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_ATTEMPTED.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
    async_started = Event()
    release_async = Event()
    sync_done = Event()
    attempts = []

    def send(status):
        attempts.append(status.report_sequence)
        async_started.set()
        assert release_async.wait(2)
        return True

    status = SimpleNamespace(
        run_id=run_id,
        state="done",
        updated_at=1.0,
        report_sequence=1,
        deployment={"state": "ready", "updated_at": 1.0},
    )
    monkeypatch.setattr(runner_reporting, "_send_status_report", send)
    runner_reporting._report_status_async(status)
    assert async_started.wait(1)

    def report_sync():
        runner_reporting._report_status(status)
        sync_done.set()

    sync = Thread(target=report_sync)
    sync.start()
    try:
        assert not sync_done.wait(0.05)
    finally:
        release_async.set()
    sync.join(2)

    assert not sync.is_alive()
    assert sync_done.is_set()
    assert attempts == [1]
    assert runner_reporting._STATUS_REPORT_LAST_SENT[run_id] == 1
    runner_reporting._shutdown_status_reporter()


def test_sync_equal_status_report_retries_once_after_async_failure(monkeypatch):
    from threading import Event, Thread

    runner_reporting._shutdown_status_reporter()
    run_id = "run-report-equal-failure"
    runner_reporting._STATUS_REPORT_LAST_SENT.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_ATTEMPTED.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
    async_started = Event()
    release_async = Event()
    sync_done = Event()
    attempts = []

    def send(status):
        attempts.append(status.report_sequence)
        if len(attempts) == 1:
            async_started.set()
            assert release_async.wait(2)
        return len(attempts) == 3

    status = SimpleNamespace(
        run_id=run_id,
        state="done",
        updated_at=1.0,
        report_sequence=1,
        deployment={"state": "ready", "updated_at": 1.0},
    )
    monkeypatch.setattr(runner_reporting, "_send_status_report", send)
    runner_reporting._report_status_async(status)
    assert async_started.wait(1)

    def report_sync():
        runner_reporting._report_status(status)
        sync_done.set()

    sync = Thread(target=report_sync)
    sync.start()
    try:
        assert not sync_done.wait(0.05)
    finally:
        release_async.set()
    sync.join(2)

    assert not sync.is_alive()
    assert sync_done.is_set()
    assert attempts == [1, 1, 1]
    assert runner_reporting._STATUS_REPORT_LAST_SENT[run_id] == 1
    runner_reporting._shutdown_status_reporter()


def test_failed_sync_status_report_sequence_can_be_retried(monkeypatch):

    runner_reporting._shutdown_status_reporter()
    run_id = "run-report-retry"
    runner_reporting._STATUS_REPORT_LAST_SENT.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_ATTEMPTED.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
    attempts = []

    def send(status):
        attempts.append(status.report_sequence)
        return len(attempts) > 1

    monkeypatch.setattr(runner_reporting, "_send_status_report", send)
    status = SimpleNamespace(
        run_id=run_id,
        state="done",
        updated_at=1.0,
        report_sequence=1,
        deployment={"state": "ready", "updated_at": 1.0},
    )

    runner_reporting._report_status(status)
    assert attempts == [1]
    assert run_id not in runner_reporting._STATUS_REPORT_LAST_SENT

    runner_reporting._report_status(status)
    assert attempts == [1, 1]
    assert runner_reporting._STATUS_REPORT_LAST_SENT[run_id] == 1
    runner_reporting._shutdown_status_reporter()


def test_wait_for_status_reports_includes_work_queued_while_waiting(monkeypatch):
    from threading import Event, Thread

    runner_reporting._shutdown_status_reporter()
    run_id = "run-report-wait"
    runner_reporting._STATUS_REPORT_LAST_SENT.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_ATTEMPTED.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
    first_started = Event()
    first_release = Event()
    second_started = Event()
    second_release = Event()
    wait_done = Event()
    wait_result = []

    def send(status):
        if status.report_sequence == 1:
            first_started.set()
            assert first_release.wait(2)
        else:
            second_started.set()
            assert second_release.wait(2)

    monkeypatch.setattr(runner_reporting, "_send_status_report", send)

    def status(sequence):
        return SimpleNamespace(
            run_id=run_id,
            state="done",
            updated_at=float(sequence),
            report_sequence=sequence,
            deployment={"state": f"state-{sequence}", "updated_at": float(sequence)},
        )

    runner_reporting._report_status_async(status(1))
    assert first_started.wait(1)

    def wait_for_reports():
        wait_result.append(runner_reporting._wait_for_status_reports(2))
        wait_done.set()

    waiter = Thread(target=wait_for_reports)
    waiter.start()
    runner_reporting._report_status_async(status(2))
    first_release.set()
    assert second_started.wait(1)
    assert not wait_done.is_set()
    second_release.set()
    waiter.join(2)

    assert wait_result == [True]
    runner_reporting._shutdown_status_reporter()


def test_duplicate_status_drains_preserve_per_run_serialization(monkeypatch):
    from collections import deque
    from threading import Event, Thread

    runner_reporting._shutdown_status_reporter()
    run_id = "run-duplicate-drain"
    for state in (
        runner_reporting._STATUS_REPORT_LAST_SENT,
        runner_reporting._STATUS_REPORT_LAST_ATTEMPTED,
        runner_reporting._STATUS_REPORT_LAST_QUEUED,
    ):
        state.pop(run_id, None)
    first_started = Event()
    release_first = Event()
    second_started = Event()
    attempts = []

    def send(status):
        attempts.append(status.report_sequence)
        if status.report_sequence == 1:
            first_started.set()
            assert release_first.wait(2)
        else:
            second_started.set()
        return True

    def status(sequence):
        return SimpleNamespace(
            run_id=run_id,
            state="done",
            updated_at=float(sequence),
            report_sequence=sequence,
            deployment={"state": f"state-{sequence}", "updated_at": float(sequence)},
        )

    monkeypatch.setattr(runner_reporting, "_send_status_report", send)
    first_done = Event()
    with runner_reporting._STATUS_REPORT_CONDITION:
        runner_reporting._STATUS_REPORT_QUEUES[run_id] = deque([(status(1), 1, first_done, 2)])
        runner_reporting._STATUS_REPORT_LAST_QUEUED[run_id] = 1
        runner_reporting._STATUS_REPORT_PENDING += 1
        runner_reporting._STATUS_REPORT_ACTIVE.add(run_id)

    first_drain = Thread(target=runner_reporting._drain_status_report_run, args=(run_id,))
    first_drain.start()
    assert first_started.wait(1)
    duplicate_drain = Thread(target=runner_reporting._drain_status_report_run, args=(run_id,))
    duplicate_drain.start()
    duplicate_drain.join(1)
    assert not duplicate_drain.is_alive()

    runner_reporting._report_status_async(status(2))
    try:
        assert not second_started.wait(0.05)
    finally:
        release_first.set()
    first_drain.join(2)

    assert not first_drain.is_alive()
    assert runner_reporting._wait_for_status_reports(2)
    assert attempts == [1, 2]
    runner_reporting._shutdown_status_reporter()


def test_async_status_report_drops_when_executor_cannot_start(monkeypatch):

    class BrokenExecutor:
        def __init__(self):
            self.shutdown_calls = []

        def submit(self, *_args, **_kwargs):
            raise RuntimeError("worker start failed")

        def shutdown(self, *, wait, cancel_futures):
            self.shutdown_calls.append((wait, cancel_futures))

    runner_reporting._shutdown_status_reporter()
    run_id = "run-report-fallback"
    runner_reporting._STATUS_REPORT_LAST_SENT.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_ATTEMPTED.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
    reported = []
    executor = BrokenExecutor()
    monkeypatch.setattr(runner_reporting, "_STATUS_REPORT_EXECUTOR", executor)
    monkeypatch.setattr(runner_reporting, "_send_status_report", reported.append)
    status = SimpleNamespace(
        run_id=run_id,
        state="done",
        updated_at=1.0,
        report_sequence=1,
        deployment={"state": "ready", "updated_at": 1.0},
    )

    runner_reporting._report_status_async(status)

    assert reported == []
    assert runner_reporting._STATUS_REPORT_PENDING == 0
    assert run_id not in runner_reporting._STATUS_REPORT_ACTIVE
    assert runner_reporting._STATUS_REPORT_EXECUTOR is None
    assert executor.shutdown_calls == []

    runner_reporting._report_status(status)
    assert reported == [status]
    runner_reporting._shutdown_status_reporter()


def test_failed_executor_submission_preserves_queued_peer_work(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    runner_reporting._shutdown_status_reporter()
    runner_reporting._open_status_reporter()
    peer_run_id = "run-report-peer"
    failed_run_id = "run-report-submit-failed"
    for run_id in (peer_run_id, failed_run_id):
        for values in (
            runner_reporting._STATUS_REPORT_LAST_SENT,
            runner_reporting._STATUS_REPORT_LAST_ATTEMPTED,
            runner_reporting._STATUS_REPORT_LAST_QUEUED,
        ):
            values.pop(run_id, None)
    occupied = Event()
    release = Event()
    peer_ran = Event()
    report_ran = Event()
    executor = ThreadPoolExecutor(max_workers=1)

    def block_worker():
        occupied.set()
        assert release.wait(2)

    busy = executor.submit(block_worker)
    assert occupied.wait(1)
    peer = executor.submit(peer_ran.set)
    monkeypatch.setattr(
        executor,
        "_adjust_thread_count",
        lambda: (_ for _ in ()).throw(RuntimeError("worker start failed")),
    )
    monkeypatch.setattr(runner_reporting, "_STATUS_REPORT_EXECUTOR", executor)
    monkeypatch.setattr(
        runner_reporting, "_send_status_report", lambda _status: report_ran.set() or True
    )
    status = SimpleNamespace(
        run_id=run_id,
        state="done",
        updated_at=1.0,
        report_sequence=1,
        deployment={"state": "ready", "updated_at": 1.0},
    )

    runner_reporting._report_status_async(status)
    assert runner_reporting._STATUS_REPORT_EXECUTOR is executor
    assert runner_reporting._STATUS_REPORT_PENDING == 1

    release.set()
    busy.result(2)
    peer.result(2)
    assert peer_ran.is_set()
    assert report_ran.wait(1)
    assert runner_reporting._wait_for_status_reports(2)
    assert runner_reporting._STATUS_REPORT_EXECUTOR is executor
    runner_reporting._shutdown_status_reporter()


def test_sync_status_report_falls_back_when_executor_cannot_start(monkeypatch):

    class BrokenExecutor:
        def __init__(self):
            self.shutdown_calls = []

        def submit(self, *_args, **_kwargs):
            raise RuntimeError("worker start failed")

        def shutdown(self, *, wait, cancel_futures):
            self.shutdown_calls.append((wait, cancel_futures))

    runner_reporting._shutdown_status_reporter()
    run_id = "run-sync-report-fallback"
    runner_reporting._STATUS_REPORT_LAST_SENT.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_ATTEMPTED.pop(run_id, None)
    runner_reporting._STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
    reported = []
    executor = BrokenExecutor()
    monkeypatch.setattr(runner_reporting, "_STATUS_REPORT_EXECUTOR", executor)
    monkeypatch.setattr(runner_reporting, "_send_status_report", reported.append)
    status = SimpleNamespace(
        run_id=run_id,
        state="done",
        updated_at=1.0,
        report_sequence=1,
        deployment={"state": "ready", "updated_at": 1.0},
    )

    runner_reporting._report_status(status)

    assert reported == [status]
    assert runner_reporting._STATUS_REPORT_PENDING == 0
    assert runner_reporting._STATUS_REPORT_EXECUTOR is None
    assert executor.shutdown_calls == []


def test_closed_status_reporter_drops_late_work_until_reopened(monkeypatch):

    runner_reporting._shutdown_status_reporter(close=True)
    reported = []
    monkeypatch.setattr(runner_reporting, "_send_status_report", reported.append)
    status = SimpleNamespace(
        run_id="run-closed-reporter",
        state="done",
        updated_at=1.0,
        report_sequence=1,
        deployment={"state": "ready", "updated_at": 1.0},
    )

    runner_reporting._report_status_async(status)
    assert reported == []
    assert runner_reporting._STATUS_REPORT_PENDING == 0

    runner_reporting._open_status_reporter()
    runner_reporting._report_status(status)
    assert reported == [status]
    runner_reporting._shutdown_status_reporter()


def test_status_report_shutdown_is_bounded_and_cancels_queued_work(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    runner_reporting._shutdown_status_reporter()
    run_a = "run-report-shutdown-a"
    run_b = "run-report-shutdown-b"
    for run_id in (run_a, run_b):
        runner_reporting._STATUS_REPORT_LAST_SENT.pop(run_id, None)
        runner_reporting._STATUS_REPORT_LAST_ATTEMPTED.pop(run_id, None)
        runner_reporting._STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
    started = Event()
    release = Event()
    reported = []

    def send(status):
        reported.append(status.run_id)
        if status.run_id == run_a:
            started.set()
            assert release.wait(2)

    def status(run_id):
        return SimpleNamespace(
            run_id=run_id,
            state="done",
            updated_at=1.0,
            report_sequence=1,
            deployment={"state": "ready", "updated_at": 1.0},
        )

    monkeypatch.setattr(runner_reporting, "_send_status_report", send)
    monkeypatch.setattr(
        runner_reporting, "_STATUS_REPORT_EXECUTOR", ThreadPoolExecutor(max_workers=1)
    )
    runner_reporting._report_status_async(status(run_a))
    assert started.wait(1)
    runner_reporting._report_status_async(status(run_b))

    assert runner_reporting._shutdown_status_reporter(0.01, close=True) is False
    assert run_a in runner_reporting._STATUS_REPORT_ACTIVE
    release.set()
    assert runner_reporting._wait_for_status_reports(2)
    assert reported == [run_a]

    runner_reporting._open_status_reporter()
    runner_reporting._report_status(status(run_b))
    assert reported == [run_a, run_b]
    runner_reporting._shutdown_status_reporter()


def test_status_report_shutdown_stops_retry_after_active_attempt(monkeypatch):
    from threading import Event

    runner_reporting._shutdown_status_reporter()
    runner_reporting._open_status_reporter()
    run_id = "run-report-stop-retry"
    for values in (
        runner_reporting._STATUS_REPORT_LAST_SENT,
        runner_reporting._STATUS_REPORT_LAST_ATTEMPTED,
        runner_reporting._STATUS_REPORT_LAST_QUEUED,
    ):
        values.pop(run_id, None)
    started = Event()
    release = Event()
    attempts = []

    def send(status):
        attempts.append(status.report_sequence)
        started.set()
        assert release.wait(2)
        return False

    status = SimpleNamespace(
        run_id=run_id,
        state="done",
        updated_at=1.0,
        report_sequence=1,
        deployment={"state": "ready", "updated_at": 1.0},
    )
    monkeypatch.setattr(runner_reporting, "_send_status_report", send)
    runner_reporting._report_status_async(status)
    assert started.wait(1)

    assert runner_reporting._shutdown_status_reporter(0.01, close=True) is False
    release.set()
    assert runner_reporting._wait_for_status_reports(2)
    assert attempts == [1]
    runner_reporting._open_status_reporter()
    runner_reporting._shutdown_status_reporter()


def test_deploy_rejects_verify_false_before_anything_registers(api, monkeypatch):
    # smoke verification is mandatory: an explicit opt-out is a 400 before queuing, and neither
    # serving registration nor alias activation is ever attempted.
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)

    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **kwargs: pytest.fail("verify=false must never reach deploy_adapter"),
    )
    monkeypatch.setattr(
        app_mod,
        "start_deployment_job",
        lambda *args, **kwargs: pytest.fail("verify=false must never queue a deployment"),
    )

    resp = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"verify": False, "checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )

    assert resp.status_code == 400, resp.text
    assert "verify=false is not supported" in resp.json()["detail"]
    assert not runner_status.get_status(run_id).deployment


def test_deploy_rechecks_run_state_before_alias_activation(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)

    def fake_deploy(**kwargs):
        latest = runner_status.get_status(run_id)
        latest.state = "cancelled"
        runner_state._save_status(latest)
        kwargs["before_ready"](f"{run_id}/final", f"{run_id}/final")
        pytest.fail("state recheck must block alias activation")

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "failed"
    assert "run state changed from 'done' to 'cancelled'" in resp.json()["error"]


def test_create_rejects_retired_gpu_class(api):
    key = _login()
    retired_spec = {**SPEC, "gpu": {"type": "RTX A6000"}}

    response = api.post(
        "/v1/runs", json={"spec": retired_spec, "dry_run": True}, headers=_bearer(key)
    )

    assert response.status_code == 400
    assert "unsupported gpu" in response.json()["detail"]


def test_deploy_forwards_structured_outputs_to_serving(api, monkeypatch):
    """The deploy route hands the run's [train].structured_outputs to deploy_adapter so serving can
    register it as the adapter's guided-decoding default (guided-decoding train/serve parity)."""
    import flash.server.asgi.app as app_mod

    key = _login()
    schema = {"type": "object", "properties": {"industries": {"type": "array"}}}
    so_spec = {**SPEC, "train": {**SPEC["train"], "structured_outputs": {"json": schema}}}
    run_id = api.post(
        "/v1/runs", json={"spec": so_spec, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)

    seen: dict = {}

    def fake_start(target, *args, **kwargs):
        seen.update({"target": target, **kwargs})
        kwargs["deploy_lock"].release()
        return False

    monkeypatch.setattr(app_mod, "start_deployment_job", fake_start)

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert resp.status_code == 200, resp.text
    forwarded = seen["deploy_kwargs"]["structured_outputs"]
    assert json.loads(forwarded) == {"json": schema}


def test_thinking_structured_deploy_rejects_verify_false_before_mutation(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    schema = {"type": "object", "required": ["answer"]}
    spec = {
        **SPEC,
        "thinking": True,
        "train": {**SPEC["train"], "structured_outputs": {"json": schema}},
    }
    run_id = api.post(
        "/v1/runs", json={"spec": spec, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)

    monkeypatch.setattr(
        app_mod,
        "start_deployment_job",
        lambda *_a, **_k: pytest.fail("deployment must not be queued"),
    )
    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **_k: pytest.fail("serving mutation must not be attempted"),
    )

    resp = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"verify": False, "checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )

    assert resp.status_code == 400, resp.text
    assert "verify=false is not supported" in resp.json()["detail"]
    assert runner_status.get_status(run_id).deployment is None


def test_deploy_retry_takes_over_stale_busy_record(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    status.deployment = {"state": "deploying", "updated_at": 0.0, "requested_at": 0.0}
    runner_state._save_status(status)
    revision = f"{run_id}/final"

    def fake_deploy(**kwargs):
        kwargs["before_ready"](revision, revision)
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: _smoke_chat_result(revision, revision),
    )

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "ready"


def test_chat_forwards_trained_stop_sequences(api, monkeypatch):
    """A run trained with stop_sequences terminates on its delimiter, not EOS. The deployment smoke
    forwards them, so the adapter verifies and activates -- but if user inference does not, the same
    model runs on to max_tokens or emits trailing text past its answer on every real request."""
    import flash.server.asgi.app as app_mod

    key = _login()
    spec = json.loads(json.dumps(SPEC))
    spec["train"] = {**spec["train"], "stop_sequences": ["</answer>"]}
    run_id = api.post(
        "/v1/runs", json={"spec": spec, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    revision = f"{run_id}/final"
    runner_transitions.mark_deployed(
        run_id,
        {"state": "ready", "endpoint_name": "https://serve.example", "checkpoint_id": revision},
        verification_generation=runner_verified_revisions.verified_checkpoint_generation(run_id),
    )

    seen: dict = {}

    def serve_chat(**kwargs):
        seen.update(kwargs)
        return _managed_chat_result(kwargs["run_id"], content="4")

    monkeypatch.setattr(app_mod, "serve_chat", serve_chat)

    resp = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "checkpoint_id": f"{run_id}/final",
        },
        headers=_bearer(key),
    )

    assert resp.status_code == 200, resp.text
    assert seen["stop"] == ["</answer>"]


def test_chat_sends_no_stop_when_run_configured_none(api, monkeypatch):
    """A run that never configured a delimiter must not receive one."""
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    revision = f"{run_id}/final"
    runner_transitions.mark_deployed(
        run_id,
        {"state": "ready", "endpoint_name": "https://serve.example", "checkpoint_id": revision},
        verification_generation=runner_verified_revisions.verified_checkpoint_generation(run_id),
    )

    seen: dict = {}

    def serve_chat(**kwargs):
        seen.update(kwargs)
        return _managed_chat_result(kwargs["run_id"], content="4")

    monkeypatch.setattr(app_mod, "serve_chat", serve_chat)

    resp = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "checkpoint_id": f"{run_id}/final",
        },
        headers=_bearer(key),
    )

    assert resp.status_code == 200, resp.text
    assert seen["stop"] is None


def test_failed_smoke_revision_cannot_be_exact_chatted(api, monkeypatch):
    import flash.server.asgi.app as app_mod
    from flash.serve.contract.errors import ServingError

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    revision = f"{run_id}/final"

    def fake_deploy(**kwargs):
        kwargs["before_ready"](revision, revision)
        pytest.fail("failed smoke must block alias activation")

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: (_ for _ in ()).throw(ServingError("smoke generation failed")),
    )

    deployment = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )

    assert deployment.status_code == 200, deployment.text
    assert deployment.json()["state"] == "failed"
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset()

    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("unverified checkpoint must not reach serving"),
    )
    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "checkpoint_id": revision,
        },
        headers=_bearer(key),
    )

    assert response.status_code == 409
    assert "has not passed a successful deployment smoke" in response.json()["detail"]


@pytest.mark.parametrize("deployment_state", ["undeployed", "revocation_failed"])
def test_redeploy_after_inactive_deployment_state_is_allowed(api, monkeypatch, deployment_state):
    import flash.server.asgi.app as app_mod
    from flash.serve.contract.errors import ServingError

    key = _login()
    run_id = _make_run(api, key, "done")
    status = runner_status.get_status(run_id)
    status.deployment = {"state": deployment_state, "requested_at": 1.0}
    runner_state._save_status(status)
    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **_kwargs: (_ for _ in ()).throw(ServingError("new adapter failed smoke")),
    )

    response = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )

    assert response.status_code == 200, response.text
    deployment = runner_status.get_status(run_id).deployment
    assert deployment["state"] == "failed"
    assert deployment["requested_at"] != 1.0


def test_deploy_ignores_stored_training_gpu(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    status.spec["gpu"]["type"] = "H200"
    # keep the internal worker-spec carrier: hf_repo + run_id (adapter identity) are platform-managed
    # and stripped from the public spec, so deploy resolves them from effective_preparation.
    runner_state._save_status(status)
    seen: dict = {}

    def fake_start(target, *args, **kwargs):
        seen.update({"target": target, **kwargs})
        kwargs["deploy_lock"].release()
        return False

    monkeypatch.setattr(app_mod, "start_deployment_job", fake_start)

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "queued"
    assert "gpu" not in resp.json()
    assert "gpu_name" not in seen["deploy_kwargs"]


def test_deploy_works_the_same_whether_or_not_gpu_count_was_authored(api, monkeypatch):
    """Deploy must not depend on whether the author wrote gpu.count.

    `gpu_count_auto` was briefly added to the digest-verified set in `effective_spec_from_status`.
    The digest covers the whole public spec including `gpu.type`, which the allocator legitimately
    rewrites onto the stored status when a run is provisioned -- so gating on the marker made an
    ordinary provisioned run fail integrity validation at deploy. An omitted count is the DEFAULT,
    so that was nearly every run. These two specs differ only in that one authors `gpu.count = 1`.
    """
    import flash.server.asgi.app as app_mod

    def fake_start(target, *args, **kwargs):
        kwargs["deploy_lock"].release()
        return False

    monkeypatch.setattr(app_mod, "start_deployment_job", fake_start)

    for gpu_section in ({}, {"count": 1}):
        key = _login()
        spec = {**SPEC, "gpu": gpu_section}
        run_id = api.post(
            "/v1/runs", json={"spec": spec, "dry_run": True}, headers=_bearer(key)
        ).json()["run_id"]
        status = runner_status.get_status(run_id)
        status.state = "done"
        # the allocator writes the class it actually rented onto the public status.
        status.spec["gpu"]["type"] = "H200"
        runner_state._save_status(status)

        resp = api.post(
            f"/v1/runs/{run_id}/deploy",
            json={"checkpoint_id": f"{run_id}/final"},
            headers=_bearer(key),
        )
        assert resp.status_code == 200, f"gpu={gpu_section!r} deploy failed: {resp.text}"


def test_deploy_missing_run_level_adapter_points_at_checkpoint_steps(api, monkeypatch):
    """A run whose finalize never published the run-level <prefix>/adapter (but which streamed
    per-step deployable checkpoints) must not fail run-level deploy with an opaque 502 rank
    error: it returns a 409 telling the caller to `flash models deploy <run>/step-N`."""
    import flash.server.asgi.app as app_mod
    from flash.serve.contract.errors import AdapterConfigMissing

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)

    def boom(**kwargs):
        raise AdapterConfigMissing(
            "could not verify adapter rank: failed to read org/repo:rl/x/adapter/adapter_config.json"
        )

    monkeypatch.setattr(app_mod, "deploy_adapter", boom)
    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: [{"step": 10}, {"step": 40}])

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "failed"
    detail = resp.json()["error"]
    assert "no final adapter" in detail
    assert f"deploy an explicit saved checkpoint such as {run_id}/step-40" in detail


def test_deploy_missing_adapter_without_checkpoints_stays_502(api, monkeypatch):
    """No checkpoints to point at -> keep the 502 with the upstream reason."""
    import flash.server.asgi.app as app_mod
    from flash.serve.contract.errors import AdapterConfigMissing

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)

    def boom(**kwargs):
        raise AdapterConfigMissing("could not verify adapter rank: failed to read org/repo:x")

    monkeypatch.setattr(app_mod, "deploy_adapter", boom)
    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: [])

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "failed"
    assert "failed to read" in resp.json()["error"]


def test_deploy_attributes_adapter_to_run_owning_org(api, monkeypatch):
    """The adapter is registered under the RUN's owning org (its persisted billing_context) so
    serving can authorize external chat by org — not merely whatever key initiated the deploy."""
    import flash.server.asgi.app as app_mod
    from flash.serve.deployment.deploy import Deployment

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    status.billing_context = {"org_id": "run-owner-org"}
    runner_state._save_status(status)

    seen: dict = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return Deployment(
            run_id=run_id,
            model=kwargs["model"],
            adapter_hf_prefix="x/adapter",
            openai_model=f"{run_id}/final",
            endpoint_name="https://serve.example",
            openai_base_url="https://serve.example/v1",
            checkpoint_id=f"{run_id}/final",
            state="ready",
        )

    monkeypatch.setattr(app_mod, "deploy_adapter", capture)

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert resp.status_code == 200, resp.text
    # The run's owning org (billing_context) is what's attributed, not the bare caller key.
    assert seen["org_id"] == "run-owner-org"


def test_deploy_falls_back_to_platform_context_org(api, monkeypatch):
    """An internal/operator deploy has no billing_context but persists the org in
    platform_context; the adapter must still be attributed to that run-owning org."""
    import flash.server.asgi.app as app_mod
    from flash.serve.deployment.deploy import Deployment

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    status.billing_context = None
    status.platform_context = {"org_id": "platform-org"}
    runner_state._save_status(status)

    seen: dict = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return Deployment(
            run_id=run_id,
            model=kwargs["model"],
            adapter_hf_prefix="x/adapter",
            openai_model=f"{run_id}/final",
            endpoint_name="https://serve.example",
            openai_base_url="https://serve.example/v1",
            checkpoint_id=f"{run_id}/final",
            state="ready",
        )

    monkeypatch.setattr(app_mod, "deploy_adapter", capture)

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert resp.status_code == 200, resp.text
    assert seen["org_id"] == "platform-org"


def test_deploy_without_any_org_context_is_rejected(api, monkeypatch):
    """A managed-plane deploy must fail closed when neither the run nor the key names an org.

    Serving authorizes external chat requests against the org that owns the adapter, so silently
    registering a revision with no org would leave a user's weights' reachability up to whatever
    the serving backend does with an unowned adapter. auth gates external keys on org_slug only
    (org_id is a best-effort passthrough), so the orgless-key case is reachable in production.
    """
    import flash.server.asgi.app as app_mod
    import flash.server.platform.auth as auth_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    status.billing_context = None
    status.platform_context = None
    runner_state._save_status(status)
    # a verified identity without org_id (but with the org_slug that auth requires)
    monkeypatch.setattr(
        auth_mod,
        "_cached_identity",
        lambda token: {k: v for k, v in _identity_for_token(token).items() if k != "org_id"},
    )
    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **_k: pytest.fail("an orgless deploy must be rejected before registration"),
    )

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert resp.status_code == 409, resp.text
    assert "owning organization" in resp.json()["detail"]


def test_deployments_listing_requires_internal_scope_and_filters_to_it(api):
    """`/v1/deployments` must not hand the internal key a cross-org listing.

    On a managed plane the internal key is the platform proxy and owns the runs it submitted for
    every org, so the listing follows `deps.manageable_run`: the internal key must name the org
    AND project it lists for, and only that scope's rows come back.
    """

    internal = _bearer("fslo-internal-test")
    project_beta = "33333333-3333-4333-8333-333333333333"
    run_ids: dict[str, str] = {}
    for org, project in (("org-alpha", SPEC["project"]), ("org-beta", project_beta)):
        run_id = api.post(
            "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=internal
        ).json()["run_id"]
        status = runner_status.get_status(run_id)
        status.state = "done"
        status.billing_context = {"org_id": org}
        status.platform_context = None
        status.spec["project"] = project
        status.deployment = {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "checkpoint_id": f"{run_id}/final",
        }
        runner_state._save_status(status)
        run_ids[org] = run_id

    # an unscoped (or half-scoped, or malformed) internal-key call gets no listing at all
    for headers in (
        {"Authorization": internal["Authorization"]},
        internal,  # _bearer adds the org header but no project
        {**internal, "X-Freesolo-Org-Id": "", "X-Freesolo-Project-Id": SPEC["project"]},
        {**internal, "X-Freesolo-Org-Id": "org-alpha", "X-Freesolo-Project-Id": "not-a-uuid"},
    ):
        resp = api.get("/v1/deployments", headers=headers)
        assert resp.status_code == 400, resp.text
        assert "must be scoped" in resp.json()["detail"]

    scoped = api.get(
        "/v1/deployments",
        headers={
            **internal,
            "X-Freesolo-Org-Id": "org-alpha",
            "X-Freesolo-Project-Id": SPEC["project"],
        },
    )
    assert scoped.status_code == 200, scoped.text
    assert [d["run_id"] for d in scoped.json()["deployments"]] == [run_ids["org-alpha"]]

    other = api.get(
        "/v1/deployments",
        headers={
            **internal,
            "X-Freesolo-Org-Id": "org-beta",
            "X-Freesolo-Project-Id": project_beta,
        },
    )
    assert [d["run_id"] for d in other.json()["deployments"]] == [run_ids["org-beta"]]

    # a matching org with the wrong project matches nothing: project is part of the scope,
    # exactly as it is for single-run deployment management
    crossed = api.get(
        "/v1/deployments",
        headers={
            **internal,
            "X-Freesolo-Org-Id": "org-alpha",
            "X-Freesolo-Project-Id": project_beta,
        },
    )
    assert crossed.json()["deployments"] == []

    # the headers are honored only for the internal key: a user key naming someone else's org
    # still sees only its own (here: zero) runs
    snoop = api.get(
        "/v1/deployments",
        headers={
            **_bearer(_login()),
            "X-Freesolo-Org-Id": "org-alpha",
            "X-Freesolo-Project-Id": SPEC["project"],
        },
    )
    assert snoop.status_code == 200, snoop.text
    assert snoop.json()["deployments"] == []


def test_undeploy_serving_error_is_clean_502(api, monkeypatch):
    """An undeploy that hits a serving-backend failure surfaces as a clean 502 (same as deploy),
    not an unhandled 500: ServingError from undeploy_adapter is translated to HTTPException(502)."""
    import flash.server.asgi.app as app_mod
    from flash.serve.contract.errors import ServingError

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    revision = f"{run_id}/final"
    sibling = f"{run_id}/step-20"
    status = runner_status.get_status(run_id)
    status.state = "deployed"
    status.deployment = {"state": "ready", "checkpoint_id": revision}
    runner_state._save_status(status)
    for checkpoint_id in (revision, sibling):
        runner_verified_revisions.add_verified_checkpoint(
            run_id,
            checkpoint_id,
            expected_generation=runner_verified_revisions.verified_checkpoint_generation(run_id),
        )

    def boom(_run_id, **_):
        raise ServingError("serving backend unreachable: could not delete endpoint")

    monkeypatch.setattr(app_mod, "undeploy_adapter", boom)

    resp = api.delete(f"/v1/runs/{run_id}/deploy?checkpoint_id={sibling}", headers=_bearer(key))
    assert resp.status_code == 502, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "deployment_revocation_failed"
    assert detail["retryable"] is True
    assert "serving backend unreachable" in detail["message"]
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset({revision})
    deployment = runner_status.get_status(run_id).deployment
    assert deployment == {"state": "ready", "checkpoint_id": revision}


def test_undeploy_without_status_projection_invalidates_orphaned_ledger(api, monkeypatch):
    import flash.server.asgi.app as app_mod
    import flash.server.domain.registry.runs as registry

    key = _login()
    run_id = _make_run(api, key, "done")
    revision = f"{run_id}/final"
    generation = runner_verified_revisions.verified_checkpoint_generation(run_id)
    assert runner_verified_revisions.add_verified_checkpoint(
        run_id,
        revision,
        expected_generation=generation,
    )
    assert runner_status.get_status(run_id).deployment is None
    monkeypatch.setattr(
        app_mod,
        "undeploy_adapter",
        lambda target, **_: {"run_id": target, "serving_deregistered": False},
    )
    reports = []
    monkeypatch.setattr(
        registry,
        "record_training_run",
        lambda **kwargs: reports.append(kwargs["status"]) or True,
    )

    response = api.delete(
        f"/v1/runs/{run_id}/deploy?checkpoint_id={run_id}/final", headers=_bearer(key)
    )

    assert response.status_code == 200, response.text
    assert runner_verified_revisions.verified_checkpoint_generation(run_id) == generation + 1
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset()
    assert runner_status.get_status(run_id).deployment is None
    assert reports == []


def test_mark_deployed_allows_done_but_not_cancelled(monkeypatch, tmp_path):
    # A finished run (state="done") MUST be deployable: mark_deployed has to record the
    # deployment and flip to "deployed". But a cancelled/failed run must never be flipped
    # to "deployed" (a /cancel racing deployment persisted the terminal state).

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))

    spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "run_id": "dep-1",
    }
    runner_state._save_status(
        runner_state.RunStatus(run_id="dep-1", state="done", spec=spec, remote=None)
    )
    deployment = {
        "state": "ready",
        "endpoint_name": "e",
        "checkpoint_id": "dep-1/final",
    }
    out = runner_transitions.mark_deployed(
        "dep-1",
        deployment,
        verification_generation=runner_verified_revisions.verified_checkpoint_generation("dep-1"),
    )
    assert out.state == "deployed"
    assert out.deployment == deployment

    # cancelled is sticky: the deploy must be refused, state preserved.
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="dep-2", state="cancelled", spec={**spec, "run_id": "dep-2"}, remote=None
        )
    )
    out2 = runner_transitions.mark_deployed("dep-2", {"endpoint_name": "e2"})
    assert out2.state == "cancelled"
    assert out2.deployment is None


def test_mark_deployed_expect_state_cas_blocks_undeploy_race(monkeypatch, tmp_path):
    # Redeploy finalization must NOT clobber an undeploy that raced in mid-warmup: the
    # undeploy wrote `done`/undeployed and deleted the endpoint, so a final mark_deployed
    # that still expects "deployed" must refuse to re-advertise the deleted endpoint.

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))

    spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "run_id": "dep-3",
    }
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="dep-3",
            state="deployed",
            spec=spec,
            remote=None,
            deployment={
                "state": "ready",
                "endpoint_name": "e",
                "checkpoint_id": "dep-3/final",
            },
        )
    )
    # undeploy races in: endpoint torn down, run back to done/undeployed.
    undone = runner_transitions.mark_undeployed("dep-3", "dep-3/final")
    assert undone.state == "done"
    assert undone.deployment["state"] == "undeployed"
    # the deploy that was warming finalizes expecting "deployed" -> refused.
    out = runner_transitions.mark_deployed(
        "dep-3",
        {"state": "ready", "endpoint_name": "e2", "checkpoint_id": "dep-3/final"},
        expect_state="deployed",
    )
    assert out.state == "done"
    assert out.deployment["state"] == "undeployed"  # not re-advertised


def test_mark_checkpoint_deployed_refuses_dry_run(monkeypatch, tmp_path):

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))

    spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "run_id": "dep-dry",
    }
    runner_state._save_status(
        runner_state.RunStatus(run_id="dep-dry", state="dry_run", spec=spec, remote=None)
    )
    out = runner_transitions.mark_checkpoint_deployed("dep-dry", {"endpoint_name": "e"})
    assert out.state == "dry_run"
    assert out.deployment is None


def test_deploy_lock_is_usable_and_weakly_cleaned():
    # threading.Lock() isn't weak-referenceable, so the per-run lock must be a wrapper that
    # both works as a context manager AND can live in the WeakValueDictionary (the raw lock
    # would TypeError on the first deploy). It must also re-enter and serialize.
    import gc

    from flash.server.asgi import app as app_mod

    lk = app_mod._deploy_lock("run-xyz")
    assert app_mod._deploy_lock("run-xyz") is lk  # same lock for the same run while alive
    with lk:
        pass
    with app_mod._deploy_lock("run-xyz"):  # re-acquirable after release
        pass
    # once nothing references it, the weak entry is dropped (no unbounded growth).
    del lk
    gc.collect()
    assert "run-xyz" not in dict(app_mod._DEPLOY_LOCKS)


def test_recover_runs_fails_descriptorless_no_handle_run(monkeypatch, tmp_path):
    # a pre-feature run with neither a handle nor persisted source identity cannot be replaced safely.
    # recovery still reaps possible provider remnants, then fails it without starting another worker.

    import flash.server.platform.db as db_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {},
        "run_id": "nohandle-1",
    }
    runner_state._save_status(
        runner_state.RunStatus(run_id="nohandle-1", state="provisioning", spec=spec, remote=None)
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "nohandle-1"}])
    gced = []
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda s: gced.append(s.run_id))
    resubmitted = []
    monkeypatch.setattr(runner_lifecycle, "_run_job", lambda s, **_kw: resubmitted.append(s.run_id))

    # a handle-less run may have left a phantom instance from a non-idempotent create (Vast PUT
    # /asks) that surfaces via eventual consistency. Recovery must force-reap the run's label across
    # instance providers RIGHT BEFORE resubmitting, so a phantom isn't left writing the same
    # seed-scoped artifacts as the fresh worker. Capture the gc-by-run.
    reaped = []

    class _FakeVast:
        name = "vast"

        def gc(self, s):
            reaped.append(s.run_id)

        def sweep_orphans(self, **k):
            return []

    from flash.providers.core import registry as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [_FakeVast()])

    app_mod.recover_runs()

    assert gced == ["nohandle-1"]
    assert reaped == ["nohandle-1"]
    assert resubmitted == []
    recovered = runner_status.get_status("nohandle-1")
    assert recovered.state == "failed"
    assert recovered.error == (
        "managed source identity is unavailable; descriptor-less attempts cannot be replaced"
    )
    assert recovered.source_verified_attempt is None
    assert "source_provenance" not in recovered.to_dict()


def test_recover_runs_drains_private_cleanup_for_terminal_run(monkeypatch, tmp_path):
    import flash.server.platform.db as db_mod
    from flash.providers.core import registry as providers_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)
    run_id = "terminal-cleanup-recovery"
    remote = {
        "provider": "runpod",
        "endpoint_id": "endpoint-cleanup",
        "job_id": "job-cleanup",
        "attempt": 1,
    }
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=run_id,
            state="cancelled",
            spec={"run_id": run_id, "project": "11111111-1111-4111-8111-111111111111"},
        ),
        _cleanup_remotes=[remote],
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": run_id}])
    drained = []
    monkeypatch.setattr(
        runner_reconciliation, "_drain_cleanup_remotes", lambda rid: drained.append(rid) or set()
    )
    monkeypatch.setattr(providers_mod, "configured_providers", list)

    app_mod.recover_runs()

    # Drained in a background thread (see recover_runs) so an outage-slow teardown can't block
    # the startup path; poll instead of asserting immediately after recover_runs() returns.
    deadline = time.time() + 5
    while not drained and time.time() < deadline:
        time.sleep(0.01)
    assert drained == [run_id]


def test_recover_runs_blocks_expired_handleless_resubmit(monkeypatch, tmp_path):
    import flash.server.platform.db as db_mod
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core import registry as providers_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)
    spec = JobSpec(
        run_id="blocked-expired",
        model="Qwen/Qwen3.5-9B",
        project="11111111-1111-4111-8111-111111111111",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=120),
    )
    created_at = 100.0
    deadline = created_at + float(spec.gpu.max_wall_seconds)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="provisioning",
            spec=spec.to_dict(),
            created_at=created_at,
            source_snapshot=_SOURCE_SNAPSHOT,
            # run_id is platform-managed and stripped from the public spec; a provisioned run always
            # carries the internal worker-spec carrier, which is where recovery resolves its identity.
            effective_preparation={
                "worker_spec": spec.to_internal_dict(),
                "adapter_identity": None,
                "preparation_digest": None,
            },
        ),
        _run_deadline_at=deadline,
        _next_attempt=0,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: deadline + 1.0)
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": spec.run_id}])
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    submitted = []
    monkeypatch.setattr(
        runner_lifecycle,
        "_run_job_background",
        lambda recovered: submitted.append(recovered.run_id),
    )
    monkeypatch.setattr(providers_mod, "configured_providers", list)

    app_mod.recover_runs()

    recovered = runner_status.get_status(spec.run_id)
    assert submitted == []
    assert recovered.state == "failed"
    assert "deadline exhausted" in recovered.error


def test_recover_runs_defers_resubmit_when_instance_not_confirmed_reaped(monkeypatch, tmp_path):
    # an unconfirmed Vast delete may leave a live phantom. recovery must defer while
    # ``run_instances_remaining`` reports it, or a second worker can write the same HF artifacts.
    import flash.server.platform.db as db_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {},
        "run_id": "phantom-1",
    }
    runner_state._save_status(
        runner_state.RunStatus(run_id="phantom-1", state="provisioning", spec=spec, remote=None)
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "phantom-1"}])
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda s: None)
    resubmitted = []
    monkeypatch.setattr(runner_lifecycle, "_run_job", lambda s, **_kw: resubmitted.append(s.run_id))

    reaped = []

    class _FakeVast:
        name = "vast"

        def gc(self, s):  # unconfirmed DELETE -> destroys nothing, returns no error
            reaped.append(s.run_id)

        def run_instances_remaining(self, run_id):  # the phantom is STILL there after gc
            return [4242]

        def sweep_orphans(self, **k):
            return []

    import flash.server.platform.runtime as rt
    from flash.providers.core import registry as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [_FakeVast()])
    # Disable the background retry budget so the defer is a clean no-op for this assertion (no lingering
    # daemon thread polling a torn-down tmp db); the reschedule behavior has its own test below.
    monkeypatch.setattr(rt, "_deferred_resubmit_loop", lambda _spec: None)

    app_mod.recover_runs()

    assert reaped == ["phantom-1"], "must still attempt the force-reap"
    assert resubmitted == [], "must NOT resubmit while an instance for the run may still be live"
    assert runner_status.get_status("phantom-1").state != "failed", (
        "deferred, not failed (later recovery retries)"
    )


def test_recover_runs_defers_when_recorded_provider_unconfigurable(monkeypatch, tmp_path):
    # dropped Vast credentials can hide a live phantom from ``configured_providers``. fail closed for
    # providers recorded at submit, or recovery can launch a second billed worker on the same HF prefix.
    import flash.server.platform.db as db_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {},
        "run_id": "unconf-1",
    }
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="unconf-1",
            state="provisioning",
            spec=spec,
            remote=None,
            submitted_instance_providers=[
                "vast"
            ],  # Vast was configured when this run was submitted
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "unconf-1"}])
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda s: None)
    resubmitted = []
    monkeypatch.setattr(runner_lifecycle, "_run_job", lambda s, **_kw: resubmitted.append(s.run_id))

    import flash.server.platform.runtime as rt
    from flash.providers.core import registry as providers_mod

    # Vast is no longer configured -> omitted from configured_providers(); the real get_provider("vast")
    # still exposes run_instances_remaining, so the recorded-but-unconfigurable provider can't be
    # enumerated -> the guard must fail closed rather than declare clear.
    monkeypatch.setattr(providers_mod, "configured_providers", list)
    monkeypatch.setattr(rt, "_deferred_resubmit_loop", lambda _spec: None)

    app_mod.recover_runs()

    assert resubmitted == [], "must NOT resubmit while an uncheckable Vast phantom may still bill"
    assert runner_status.get_status("unconf-1").state != "failed", (
        "deferred, not failed (later restart retries)"
    )


def test_recover_runs_resubmits_queued_run_despite_unconfigurable_vast(monkeypatch, tmp_path):
    # queued runs never attempted Vast's non-idempotent create, so skip the phantom guard. otherwise
    # removed credentials can defer a run forever even though no provider resource can exist.
    import threading

    import flash.server.platform.db as db_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {},
        "run_id": "queued-1",
    }
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="queued-1",
            state="queued",  # never provisioned -> no create attempted -> no phantom possible
            spec=spec,
            remote=None,
            source_snapshot=_SOURCE_SNAPSHOT,
            submitted_instance_providers=["vast"],  # Vast configured at submit, creds now gone
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "queued-1"}])
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda s: None)

    resubmitted = []
    done = threading.Event()

    def fake_run_job(s, **_kwargs):
        resubmitted.append(s.run_id)
        done.set()

    monkeypatch.setattr(runner_lifecycle, "_run_job", fake_run_job)

    import flash.server.platform.runtime as rt
    from flash.providers.core import registry as providers_mod

    # Vast unconfigurable now: the OLD unconditional guard would fail closed here and defer forever. A
    # queued run must resubmit anyway, because it provably never created the phantom the guard protects
    # against. _confirm_run_clear must not even be consulted (Vast enumeration would raise/defer).
    monkeypatch.setattr(providers_mod, "configured_providers", list)
    monkeypatch.setattr(rt, "_deferred_resubmit_loop", lambda _spec: None)

    app_mod.recover_runs()

    assert done.wait(timeout=5), (
        "queued run must launch a resubmit thread, not defer on a phantom check"
    )
    assert resubmitted == ["queued-1"], (
        "a never-provisioned queued run resubmits despite unconfigurable Vast"
    )
    assert runner_status.get_status("queued-1").state != "failed"


def test_recover_runs_resubmits_when_no_capability_provider_recorded(monkeypatch, tmp_path):
    # The fail-closed must stay SCOPED: a handle-less run on a plane that never configured Vast records
    # no Vast in submitted_instance_providers, so it can't have left a Vast phantom and must still
    # recover (resubmit) even though get_provider("vast") exposes the capability. Guards against the
    # over-broad "any unconfigured capability provider blocks" regression that would strand RunPod/Lambda
    # -only deployments.
    import threading

    import flash.server.platform.db as db_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {},
        "run_id": "novast-1",
    }
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="novast-1",
            state="provisioning",
            spec=spec,
            remote=None,
            source_snapshot=_SOURCE_SNAPSHOT,
            submitted_instance_providers=[],  # no instance provider was available at submit
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "novast-1"}])
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda s: None)
    resubmitted = []
    done = threading.Event()
    monkeypatch.setattr(
        runner_lifecycle, "_run_job", lambda s, **_kw: (resubmitted.append(s.run_id), done.set())
    )

    from flash.providers.core import registry as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", list)

    app_mod.recover_runs()

    assert done.wait(timeout=5), (
        "a run that never recorded Vast must still recover on a Vast-less plane"
    )
    assert resubmitted == ["novast-1"]


def test_recover_runs_ignores_newly_configured_unrecorded_provider(monkeypatch, tmp_path):
    # A provider enabled after submit cannot have owned that run's pre-handle create. Its listing outage
    # must not strand recovery when submitted_instance_providers explicitly says it was not available.
    import threading

    import flash.server.platform.db as db_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {},
        "run_id": "newvast-1",
    }
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="newvast-1",
            state="provisioning",
            spec=spec,
            remote=None,
            source_snapshot=_SOURCE_SNAPSHOT,
            submitted_instance_providers=[],
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "newvast-1"}])
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda s: None)
    resubmitted = []
    done = threading.Event()
    monkeypatch.setattr(
        runner_lifecycle, "_run_job", lambda s, **_kw: (resubmitted.append(s.run_id), done.set())
    )

    class _NewVast:
        name = "vast"

        def gc(self, s):
            raise AssertionError("newly configured unrecorded provider must not be reaped")

        def run_instances_remaining(self, run_id):
            raise AssertionError("newly configured unrecorded provider must not block recovery")

        def sweep_orphans(self, **k):
            return []

    from flash.providers.core import registry as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [_NewVast()])

    app_mod.recover_runs()

    assert done.wait(timeout=5), "newly configured unrecorded Vast must not block recovery"
    assert resubmitted == ["newvast-1"]


def test_recover_runs_deferred_resubmit_retries_until_clear(monkeypatch, tmp_path):
    # bounded background retries must recheck deferred handle-less runs, so a cleared phantom can
    # resubmit without waiting for the next control-plane restart.
    import threading

    import flash.server.platform.db as db_mod
    import flash.server.platform.runtime as rt

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {},
        "run_id": "retry-1",
    }
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="retry-1",
            state="provisioning",
            spec=spec,
            remote=None,
            source_snapshot=_SOURCE_SNAPSHOT,
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "retry-1"}])
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda s: None)
    resubmitted = []
    done = threading.Event()
    monkeypatch.setattr(
        runner_lifecycle, "_run_job", lambda s, **_kw: (resubmitted.append(s.run_id), done.set())
    )
    monkeypatch.setattr(rt, "_DEFERRED_RECOVERY_RETRY_S", 0.01)  # fast background retry

    calls = {"n": 0}

    class _FakeVast:
        name = "vast"

        def gc(self, s):
            pass

        def run_instances_remaining(self, run_id):
            calls["n"] += 1
            return [4242] if calls["n"] == 1 else []  # present once, then cleared

        def sweep_orphans(self, **k):
            return []

    from flash.providers.core import registry as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [_FakeVast()])

    app_mod.recover_runs()  # first check sees the box -> defers + schedules the background retry

    assert done.wait(timeout=5), (
        "the background retry must resubmit once the run is confirmed clear"
    )
    assert resubmitted == ["retry-1"]


def test_recover_runs_resubmits_when_instance_confirmed_clear(monkeypatch, tmp_path):
    # The confirmation gate must not block the normal case: when run_instances_remaining returns []
    # (confirmed no instance for this run remains), the handle-less run resubmits as before.
    import threading

    import flash.server.platform.db as db_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {},
        "run_id": "clear-1",
    }
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="clear-1",
            state="provisioning",
            spec=spec,
            remote=None,
            source_snapshot=_SOURCE_SNAPSHOT,
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "clear-1"}])
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda s: None)
    resubmitted = []
    done = threading.Event()

    claims = []

    def fake_run_job(s, **kwargs):
        # recovery must hand the background launch the attempt it durably reserved, so the
        # replacement runs under that exact claim rather than reserving a second one.
        claims.append(kwargs.get("reserved_claim"))
        resubmitted.append(s.run_id)
        done.set()

    monkeypatch.setattr(runner_lifecycle, "_run_job", fake_run_job)

    class _FakeVast:
        name = "vast"

        def gc(self, s):
            pass

        def run_instances_remaining(self, run_id):  # confirmed clear
            return []

        def sweep_orphans(self, **k):
            return []

    from flash.providers.core import registry as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [_FakeVast()])

    app_mod.recover_runs()

    assert done.wait(timeout=5), "a confirmed-clear run must still resubmit"
    assert resubmitted == ["clear-1"]
    assert len(claims) == 1
    # recovery must hand the launch the attempt it reserved, not let it reserve a second one.
    assert claims[0] is not None
    assert claims[0].attempt == 0


def test_recover_runs_reuses_verified_effective_snapshot_for_no_handle_resubmit(
    monkeypatch, tmp_path
):
    import threading

    import flash.adapters.lora_rank as rank_mod
    import flash.server.platform.db as db_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)

    public_spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "train": {
            "epochs": 1,
            "max_examples": 1,
            "init_from_adapter": "source-run/final",
            "lora_rank": 8,
        },
        "gpu": {},
        "run_id": "nohandle-warm",
    }
    worker_spec = {
        **public_spec,
        "train": {
            **public_spec["train"],
            "init_from_adapter": "org/source-runs:rl/source-run",
            "init_from_adapter_revision": "a" * 40,
            "lora_rank": 32,
        },
    }
    identity = rank_mod.AdapterArtifactIdentity(
        "digest-v1", "config-v1", "adapter_model.safetensors", "weights-v1:123"
    )
    from flash.core.spec import JobSpec

    public_job = JobSpec.from_dict(public_spec)
    worker_job = JobSpec.from_dict(worker_spec)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="nohandle-warm",
            state="provisioning",
            spec=public_spec,
            remote=None,
            source_snapshot=_SOURCE_SNAPSHOT,
            effective_preparation={
                "worker_spec": worker_spec,
                "adapter_identity": identity.to_dict(),
                "version": 1,
                "preparation_digest": runner_preparation._preparation_digest(
                    public_job, worker_job, identity.to_dict()
                ),
            },
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "nohandle-warm"}])
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda s: None)
    monkeypatch.setattr(
        rank_mod,
        "load_hf_adapter_config",
        lambda *a, **k: {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3.5-9B",
            "r": 32,
            "lora_alpha": 64,
        },
    )
    monkeypatch.setattr(rank_mod, "adapter_artifact_identity", lambda *a, **k: identity)
    resubmitted: list[tuple[str, int]] = []
    done = threading.Event()

    def fake_run_job(s, **_kwargs):
        resubmitted.append((s.train.init_from_adapter, s.train.lora_rank))
        done.set()

    monkeypatch.setattr(runner_lifecycle, "_run_job", fake_run_job)

    app_mod.recover_runs()

    assert done.wait(timeout=5), "no-handle recovery must launch a resubmit thread"
    assert resubmitted == [("org/source-runs:rl/source-run", 32)]


def test_recover_runs_rejects_warmstart_artifact_drift(monkeypatch, tmp_path):
    import flash.adapters.lora_rank as rank_mod
    import flash.server.platform.db as db_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)
    public_spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "train": {"init_from_adapter": "source-run/final", "lora_rank": 8},
        "run_id": "drifted-warm",
    }
    worker_spec = {
        **public_spec,
        "train": {
            **public_spec["train"],
            "init_from_adapter": "private-owner/private-repo:rl/source-run",
            "init_from_adapter_revision": "a" * 40,
            "lora_rank": 32,
        },
    }
    from flash.core.spec import JobSpec

    public_job = JobSpec.from_dict(public_spec)
    worker_job = JobSpec.from_dict(worker_spec)
    original_identity = {
        "digest": "original",
        "config_sha256": "config-v1",
        "weight_filename": "adapter_model.safetensors",
        "weight_identity": "weights-v1:123",
    }
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="drifted-warm",
            state="provisioning",
            spec=public_spec,
            effective_preparation={
                "worker_spec": worker_spec,
                "adapter_identity": original_identity,
                "version": 1,
                "preparation_digest": runner_preparation._preparation_digest(
                    public_job, worker_job, original_identity
                ),
            },
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "drifted-warm"}])
    monkeypatch.setattr(
        rank_mod,
        "load_hf_adapter_config",
        lambda *a, **k: {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3.5-9B",
            "r": 32,
            "lora_alpha": 64,
        },
    )
    monkeypatch.setattr(
        rank_mod,
        "adapter_artifact_identity",
        lambda *a, **k: rank_mod.AdapterArtifactIdentity(
            "changed", "config-v2", "adapter_model.safetensors", "weights-v2:123"
        ),
    )
    cleaned = []
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda spec: cleaned.append(spec))
    monkeypatch.setattr(
        runner_lifecycle,
        "_run_job",
        lambda s, **_kw: pytest.fail("drifted warm-start source must not be resubmitted"),
    )

    app_mod.recover_runs()

    status = runner_status.get_status("drifted-warm")
    assert status.state == "failed"
    assert len(cleaned) == 1
    assert cleaned[0].train.init_from_adapter == "source-run/final"
    assert "source-run" in (status.error or "")
    assert "private-owner" not in (status.error or "")
    assert "private-repo" not in (status.error or "")


def test_recover_runs_bad_spec_is_isolated_not_fatal(monkeypatch, tmp_path):
    # Here run #1 has a bad spec and run #2 has a valid no-handle spec: assert run #2 is still
    # resubmitted AND the orphan sweep still runs (the bad spec didn't take down the whole
    # recovery pass).
    import threading

    import flash.providers.runpod.serverless.endpoints as runpod_train
    import flash.server.platform.db as db_mod
    from flash.providers.core import registry as providers_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)

    # Run #1: a malformed spec — local `environment.path` makes from_dict raise.
    bad_spec = {
        "project": "11111111-1111-4111-8111-111111111111",
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "environment": {"path": "/legacy/local/env"},
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {"type": "RTX 5090"},
        "run_id": "bad-1",
    }
    # Run #2: a valid no-handle spec — must still be recovered (resubmitted) despite run #1.
    good_spec = {
        "model": "Qwen/Qwen3.5-9B",
        "project": "11111111-1111-4111-8111-111111111111",
        "algorithm": "grpo",
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {},
        "run_id": "good-2",
    }
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="bad-1",
            state="provisioning",
            spec={**good_spec, "run_id": "bad-1"},
            remote=None,
        )
    )
    bad_raw = runner_status._load_status_json("bad-1")
    bad_raw["spec"] = bad_spec
    with open(runner_state.runs_file_path("bad-1", ".json"), "w") as file:
        json.dump(bad_raw, file)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="good-2",
            state="provisioning",
            spec=good_spec,
            remote=None,
            source_snapshot=_SOURCE_SNAPSHOT,
        )
    )
    # Order matters: the bad run is iterated FIRST, so an unguarded parse would abort here.
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "bad-1"}, {"run_id": "good-2"}])
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda s: None)

    # A malformed spec can't be parsed into a JobSpec, so the good-spec branch's
    # `_gc_run_endpoints(spec)` is unavailable -- yet the aborted attempt may still have
    # registered its uniquely-named RunPod endpoint before crashing, which the no-op RunPod
    # `sweep_orphans` won't reap. recover_runs must instead derive the endpoint name from the
    # RAW persisted status (gpu.type + run_id, no spec parse) and `terminate_endpoint` it.
    terminated = []
    monkeypatch.setattr(
        runpod_train,
        "terminate_endpoint",
        lambda gpu_type, run_id=None: terminated.append((gpu_type, run_id)) or [],
    )

    # The orphan sweep must still run after the loop. recover_runs resolves it via a
    # function-local `from flash.providers.core.registry import configured_providers`, so patch the
    # package attr; record that sweep_orphans fired.
    swept = threading.Event()

    class _FakeProvider:
        # known_labels is part of the real sweep_orphans signature (multi-plane guard); accept it so the
        # actual call prov.sweep_orphans(active_labels=..., known_labels=...) doesn't TypeError (which
        # the recovery suppress would swallow, silently skipping the sweep this test asserts fired).
        def sweep_orphans(self, active_labels=None, known_labels=None):
            swept.set()
            return []

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [_FakeProvider()])

    resubmitted = []
    done = threading.Event()

    def fake_run_job(s, **_kwargs):
        resubmitted.append(s.run_id)
        done.set()

    monkeypatch.setattr(runner_lifecycle, "_run_job", fake_run_job)

    app_mod.recover_runs()

    assert done.wait(timeout=5), "the valid run must still be resubmitted despite a prior bad spec"
    assert resubmitted == ["good-2"], "only the valid run resubmits; the malformed one is skipped"
    assert swept.is_set(), "a malformed spec must not abort the orphan sweep that follows the loop"

    # The malformed run must NOT be silently skipped and left recoverable (it would be
    # retried-then-skipped on every restart forever, invisible to the user). It must be
    # persisted as terminal `failed` with an operator-visible error note, so it surfaces to
    # the user AND drops out of the recoverable set (never re-attempted).
    bad_status = runner_status.get_status("bad-1")
    assert bad_status.state == "failed", "an unparseable persisted spec must be marked failed"
    assert bad_status.state in runner_state.TERMINAL_STATES, (
        "failed is terminal, so it won't recover again"
    )
    assert bad_status.state not in app_mod._RECOVERABLE, "the failed run leaves the recoverable set"
    assert bad_status.error, "the failed run must carry an operator-visible error note"
    assert "unrecoverable" in bad_status.error, (
        "the failure note must explain the malformed spec to the operator"
    )

    # Resource-leak guard: even though the spec couldn't be parsed, the malformed run's RunPod
    # endpoint must still be torn down -- derived from the RAW persisted gpu.type + run_id,
    # not from a JobSpec -- so a crash that registered an endpoint before persisting a handle
    # can't leak it (RunPod's `sweep_orphans` is a no-op and would never catch it).
    assert ("RTX 5090", "bad-1") in terminated, (
        "a malformed-spec run's endpoint must be GC'd by reconstructed name (raw gpu.type + "
        "run_id), since its spec can't be parsed and the RunPod orphan sweep is a no-op"
    )


def test_publish_env_endpoint_publishes_under_managed_account(api, monkeypatch):
    """POST /v1/envs publishes an uploaded package to the managed environment hub."""
    monkeypatch.setenv("GITHUB_TOKEN", "token-for-publish-path")
    import base64
    import io
    import tarfile

    import flash.server.domain.registry.envs as envs_mod

    published_roots: list[str] = []
    monkeypatch.setattr(
        envs_mod,
        "_github_publish_once",
        lambda *, publish_root, **_kwargs: published_roots.append(publish_root),
    )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in (
            ("pyproject.toml", b"[project]\nname='e'\n"),
            ("environment.py", b"def load_environment(**kwargs): return None\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    pkg = base64.b64encode(buf.getvalue()).decode()

    key = _login()
    expected_root = f"org-{key.removeprefix(_USER_PREFIX)}/checkout-bot/myenv"
    resp = api.post(
        "/v1/envs",
        headers=_bearer(key),
        json={
            "name": "MyEnv",
            "package_b64": pkg,
            "project_id": "11111111-1111-4111-8111-111111111111",
        },
    )
    assert resp.status_code == 200
    ref = resp.json()["id"]
    assert ref == expected_root
    assert expected_root in published_roots

    # Unauthenticated requests are rejected.
    assert api.post("/v1/envs", json={"name": "e", "package_b64": pkg}).status_code in (401, 403)


def test_publish_env_ignores_legacy_is_new(api, monkeypatch):
    """Publish mode is determined by the explicit name and server publish id."""
    import base64
    import io
    import tarfile

    import flash.server.domain.registry.envs as envs_mod

    captured: list = []
    monkeypatch.setattr(
        envs_mod, "publish_package", stub_publish_package("key-1/checkout-bot/e", record=captured)
    )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for nm, content in (
            ("pyproject.toml", b"[project]\nname='e'\n"),
            ("environment.py", b"def load_environment(**kwargs): return None\n"),
        ):
            info = tarfile.TarInfo(nm)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    pkg = base64.b64encode(buf.getvalue()).decode()

    resp = api.post(
        "/v1/envs",
        headers=_bearer(_login()),
        json={
            "name": "e",
            "package_b64": pkg,
            "project_id": "11111111-1111-4111-8111-111111111111",
            "is_new": False,
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(captured) == 1
    seen = captured[0]
    assert seen["name"] == "e"
    assert seen["package_b64"] == pkg
    assert seen["key"]["org_slug"].startswith("org-")
    assert "is_new" not in seen


def test_publish_env_forwards_project_id_to_registry(api, monkeypatch):
    """A publish naming a project forwards it to the platform metadata mirror."""
    import base64
    import io
    import tarfile

    import flash.server.domain.registry.environment_registry as registry_mod
    import flash.server.domain.registry.envs as envs_mod

    monkeypatch.setattr(envs_mod, "publish_package", stub_publish_package("key-1/checkout-bot/e"))

    recorded: list[dict] = []
    monkeypatch.setattr(
        registry_mod,
        "record_published_environment",
        lambda **kwargs: recorded.append(kwargs) or True,
    )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for nm, content in (
            ("pyproject.toml", b"[project]\nname='e'\n"),
            ("environment.py", b"def load_environment(**kwargs): return None\n"),
        ):
            info = tarfile.TarInfo(nm)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    pkg = base64.b64encode(buf.getvalue()).decode()

    resp = api.post(
        "/v1/envs",
        headers=_bearer(_login()),
        json={
            "name": "e",
            "package_b64": pkg,
            "project_id": "11111111-1111-4111-8111-111111111111",
        },
    )
    assert resp.status_code == 200, resp.text
    assert recorded[0]["project_id"] == "11111111-1111-4111-8111-111111111111"

    # direct callers cannot bypass explicit project validation.
    recorded.clear()
    resp = api.post(
        "/v1/envs",
        headers=_bearer(_login()),
        json={"name": "e", "package_b64": pkg},
    )
    assert resp.status_code == 400, resp.text
    assert "project_id is required" in resp.text
    assert recorded == []


def test_publish_env_returns_502_when_association_record_returns_false(api, monkeypatch):
    import flash.server.domain.registry.environment_registry as registry
    import flash.server.domain.registry.envs as envs_mod

    events: list[str] = []
    monkeypatch.setattr(
        envs_mod,
        "publish_package",
        lambda **_kwargs: events.append("uploaded") or "acme/checkout-bot/env",
    )
    monkeypatch.setattr(
        registry,
        "record_published_environment",
        lambda **_kwargs: events.append("association-false") or False,
    )
    monkeypatch.setattr(
        envs_mod,
        "delete_package",
        lambda **_kwargs: pytest.fail("uploaded package must not be rolled back"),
    )

    response = api.post(
        "/v1/envs",
        headers=_bearer(_login()),
        json={"name": "env", "package_b64": ENV_PACKAGE_B64, "project_id": SPEC["project"]},
    )

    assert response.status_code == 502
    assert "package may already be uploaded" in response.json()["detail"]
    assert "retry the same publish" in response.json()["detail"]
    assert events == ["uploaded", "association-false"]


def test_publish_env_returns_502_when_association_record_raises(api, monkeypatch):
    import flash.server.domain.registry.environment_registry as registry
    import flash.server.domain.registry.envs as envs_mod

    events: list[str] = []
    monkeypatch.setattr(
        envs_mod,
        "publish_package",
        lambda **_kwargs: events.append("uploaded") or "acme/checkout-bot/env",
    )

    def fail_association(**_kwargs):
        events.append("association-error")
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(registry, "record_published_environment", fail_association)
    monkeypatch.setattr(
        envs_mod,
        "delete_package",
        lambda **_kwargs: pytest.fail("uploaded package must not be rolled back"),
    )

    response = api.post(
        "/v1/envs",
        headers=_bearer(_login()),
        json={"name": "env", "package_b64": ENV_PACKAGE_B64, "project_id": SPEC["project"]},
    )

    assert response.status_code == 502
    assert "package may already be uploaded" in response.json()["detail"]
    assert "retry the same publish" in response.json()["detail"]
    assert events == ["uploaded", "association-error"]


def test_publish_env_retry_repairs_association_after_false_ack(api, monkeypatch):
    import flash.server.domain.registry.environment_registry as registry
    import flash.server.domain.registry.envs as envs_mod

    uploads: list[str] = []
    acknowledgements = iter((False, True))
    monkeypatch.setattr(
        envs_mod,
        "publish_package",
        lambda **_kwargs: uploads.append("acme/checkout-bot/env") or "acme/checkout-bot/env",
    )
    monkeypatch.setattr(
        registry,
        "record_published_environment",
        lambda **_kwargs: next(acknowledgements),
    )
    monkeypatch.setattr(
        envs_mod,
        "delete_package",
        lambda **_kwargs: pytest.fail("retry must not roll back the uploaded package"),
    )
    request = {
        "headers": _bearer(_login()),
        "json": {"name": "env", "package_b64": ENV_PACKAGE_B64, "project_id": SPEC["project"]},
    }

    first = api.post("/v1/envs", **request)
    second = api.post("/v1/envs", **request)

    assert first.status_code == 502
    assert second.status_code == 200, second.text
    assert second.json() == {"id": "acme/checkout-bot/env"}
    assert uploads == ["acme/checkout-bot/env", "acme/checkout-bot/env"]


def _b64_targz(members: dict[str, bytes]) -> str:
    """Deterministic base64 `.tar.gz`.

    `mtime=0` on both the gzip stream and its members matters: gzip embeds a timestamp, so the
    default would make this string differ between xdist workers. Since it is a parametrize
    argument, that lands in the test id and pytest aborts the whole run with "Different tests
    were collected between gw0 and gw1".
    """
    import base64
    import gzip
    import io
    import tarfile

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for nm, content in members.items():
            info = tarfile.TarInfo(nm)
            info.size = len(content)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(content))
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize(
    ("package", "message"),
    [
        # decodes as base64 but is not a gzip stream at all
        pytest.param("bm90IGEgdGFyYmFsbCBhdCBhbGw=", "could not be extracted", id="not-a-gzip"),
        # a valid archive that is missing the required entrypoint
        pytest.param(
            _b64_targz({"readme.txt": b"nope"}),
            "must contain environment.py",
            id="no-entrypoint",
        ),
    ],
)
def test_publish_env_validates_the_archive_before_publication(api, monkeypatch, package, message):
    """Base64-valid but structurally invalid packages keep their own 400."""
    monkeypatch.setenv("GITHUB_TOKEN", "token-for-publish-path")

    response = api.post(
        "/v1/envs",
        headers=_bearer(_login()),
        json={"name": "env", "package_b64": package, "project_id": SPEC["project"]},
    )

    assert response.status_code == 400, response.text
    assert message in response.json()["detail"]


@pytest.mark.parametrize(
    ("field", "value", "status", "message"),
    [
        ("name", "", 400, "missing env name"),
        ("package_b64", False, 400, "must be a base64 string"),
        ("package_b64", "!!!!", 400, "not valid base64"),
        ("package_b64", "", 400, "empty env package"),
    ],
)
def test_publish_env_validates_inputs_before_publication(
    api, monkeypatch, field, value, status, message
):
    """An unpublishable request keeps its own deterministic error."""
    body = {"name": "env", "package_b64": ENV_PACKAGE_B64, "project_id": SPEC["project"]}
    body[field] = value
    response = api.post("/v1/envs", headers=_bearer(_login()), json=body)

    assert response.status_code == status, response.text
    assert message in response.json()["detail"].lower()


def test_publish_env_reports_an_invalid_name_as_a_type_error(api, monkeypatch):
    """A malformed name receives the deterministic type error."""
    response = api.post(
        "/v1/envs",
        headers=_bearer(_login()),
        json={"name": 0, "package_b64": ENV_PACKAGE_B64, "project_id": SPEC["project"]},
    )

    assert response.status_code == 400, response.text
    assert "name must be a string" in response.json()["detail"].lower()


def test_publish_env_falsy_non_string_fields_are_not_coerced(api):
    """Regression: a present-but-falsy non-string `name`/`package_b64` (e.g. 0, False, []) must
    reach publish_package's type check and yield the *type* 400 — not be `or ""`-coerced to an
    empty string first (which would surface a different/misleading 400)."""
    # name = 0 -> hits the name type check, not "missing env name".
    r = api.post(
        "/v1/envs",
        headers=_bearer(_login()),
        json={"name": 0, "package_b64": "x", "project_id": "11111111-1111-4111-8111-111111111111"},
    )
    assert r.status_code == 400, r.text
    assert "name must be a string" in r.text.lower()
    # package_b64 = False (valid string name) -> hits the package type check.
    r2 = api.post(
        "/v1/envs",
        headers=_bearer(_login()),
        json={
            "name": "e",
            "package_b64": False,
            "project_id": "11111111-1111-4111-8111-111111111111",
        },
    )
    assert r2.status_code == 400, r2.text
    assert "must be a base64 string" in r2.text.lower()


def test_delete_env_endpoint_removes_package(api, monkeypatch):
    """DELETE /v1/envs/{id} removes the package and reports it deleted."""
    import flash.server.domain.registry.envs as envs_mod
    from flash.server.domain.registry import environment_registry

    seen: dict = {}

    def fake_delete_package(*, slug, key):
        seen.update(slug=slug, key=key)
        return True

    monkeypatch.setattr(envs_mod, "delete_package", fake_delete_package)
    recorded: dict = {}
    monkeypatch.setattr(
        environment_registry,
        "record_deleted_environment",
        lambda *, slug, project_id, key, org_id=None: (
            recorded.update(slug=slug, project_id=project_id, org_id=org_id) or True
        ),
    )

    resp = api.delete(
        "/v1/envs/acme/checkout-bot/my-env",
        headers={
            **_bearer(_login()),
            "X-Freesolo-Org-Id": "org-acme",
            "X-Freesolo-Project-Id": "11111111-1111-4111-8111-111111111111",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"id": "acme/checkout-bot/my-env", "deleted": True}
    assert seen["slug"] == "acme/checkout-bot/my-env"
    assert recorded["slug"] == "acme/checkout-bot/my-env"
    assert recorded["project_id"] == "11111111-1111-4111-8111-111111111111"
    # the caller-supplied org (web ui delete) reaches the metadata-mirror drop.
    assert recorded["org_id"] == "org-acme"

    # unauthenticated requests are rejected.
    assert api.delete("/v1/envs/acme/checkout-bot/my-env").status_code in (401, 403)


def test_delete_env_missing_mirror_and_package_is_idempotent(api, monkeypatch):
    from fastapi import HTTPException

    import flash.server.domain.registry.envs as envs_mod
    from flash.server.domain.registry import environment_registry

    key = _login()
    namespace = _identity_for_token(key)["org_slug"]
    mirror_error = HTTPException(status_code=404, detail="flash environment not found")
    monkeypatch.setattr(
        environment_registry,
        "require_environment_project",
        lambda **_kwargs: (_ for _ in ()).throw(mirror_error),
    )
    monkeypatch.setattr(
        envs_mod,
        "download_package",
        lambda **_kwargs: (_ for _ in ()).throw(
            envs_mod.EnvPublishError("environment package not found", status=404)
        ),
    )
    monkeypatch.setattr(
        envs_mod,
        "delete_package",
        lambda **_kwargs: pytest.fail("an absent package must remain a no-op"),
    )
    monkeypatch.setattr(
        environment_registry,
        "record_deleted_environment",
        lambda **_kwargs: True,
    )

    response = api.delete(
        f"/v1/envs/{namespace}/checkout-bot/my-env",
        headers={
            **_bearer(key),
            "X-Freesolo-Project-Id": "11111111-1111-4111-8111-111111111111",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"id": f"{namespace}/checkout-bot/my-env", "deleted": False}


def test_delete_env_internal_key_missing_mirror_and_package_is_idempotent(api, monkeypatch):
    from fastapi import HTTPException

    import flash.server.domain.registry.envs as envs_mod
    from flash.server.domain.registry import environment_registry

    monkeypatch.setattr(
        environment_registry,
        "require_environment_project",
        lambda **_kwargs: (_ for _ in ()).throw(
            HTTPException(status_code=404, detail="flash environment not found")
        ),
    )
    monkeypatch.setattr(
        envs_mod,
        "download_package",
        lambda **_kwargs: (_ for _ in ()).throw(
            envs_mod.EnvPublishError("environment package not found", status=404)
        ),
    )
    monkeypatch.setattr(
        envs_mod,
        "delete_package",
        lambda **_kwargs: pytest.fail("an absent package must remain a no-op"),
    )
    monkeypatch.setattr(
        environment_registry,
        "record_deleted_environment",
        lambda **_kwargs: True,
    )

    response = api.delete(
        "/v1/envs/acme/checkout-bot/my-env",
        headers={
            **_bearer("fslo-internal-test"),
            "X-Freesolo-Project-Id": "11111111-1111-4111-8111-111111111111",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"id": "acme/checkout-bot/my-env", "deleted": False}


def test_delete_env_internal_key_missing_mirror_does_not_delete_existing_package(api, monkeypatch):
    from fastapi import HTTPException

    import flash.server.domain.registry.envs as envs_mod
    from flash.server.domain.registry import environment_registry

    mirror_error = HTTPException(status_code=404, detail="flash environment not found")
    monkeypatch.setattr(
        environment_registry,
        "require_environment_project",
        lambda **_kwargs: (_ for _ in ()).throw(mirror_error),
    )
    monkeypatch.setattr(envs_mod, "download_package", lambda **_kwargs: b"package")
    monkeypatch.setattr(
        envs_mod,
        "delete_package",
        lambda **_kwargs: pytest.fail("an unassociated package must not be deleted"),
    )
    monkeypatch.setattr(
        environment_registry,
        "record_deleted_environment",
        lambda **_kwargs: pytest.fail("a failed delete must not touch the mirror"),
    )

    response = api.delete(
        "/v1/envs/acme/checkout-bot/my-env",
        headers={
            **_bearer("fslo-internal-test"),
            "X-Freesolo-Project-Id": "11111111-1111-4111-8111-111111111111",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "flash environment not found"


def test_delete_env_user_key_missing_mirror_requires_matching_namespace(api, monkeypatch):
    from fastapi import HTTPException

    import flash.server.domain.registry.envs as envs_mod
    from flash.server.domain.registry import environment_registry

    key = _login()
    monkeypatch.setattr(
        environment_registry,
        "require_environment_project",
        lambda **_kwargs: (_ for _ in ()).throw(
            HTTPException(status_code=404, detail="flash environment not found")
        ),
    )
    monkeypatch.setattr(
        envs_mod,
        "download_package",
        lambda **_kwargs: pytest.fail("a foreign package must not be probed"),
    )
    monkeypatch.setattr(
        envs_mod,
        "delete_package",
        lambda **_kwargs: pytest.fail("a foreign package must not be deleted"),
    )
    monkeypatch.setattr(
        environment_registry,
        "record_deleted_environment",
        lambda **_kwargs: pytest.fail("a failed delete must not touch the mirror"),
    )

    response = api.delete(
        "/v1/envs/someone-else/checkout-bot/my-env",
        headers={
            **_bearer(key),
            "X-Freesolo-Project-Id": "11111111-1111-4111-8111-111111111111",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "flash environment not found"


def test_delete_env_missing_mirror_surfaces_hub_outage_status(api, monkeypatch):
    from fastapi import HTTPException

    import flash.server.domain.registry.envs as envs_mod
    from flash.server.domain.registry import environment_registry

    key = _login()
    namespace = _identity_for_token(key)["org_slug"]
    monkeypatch.setattr(
        environment_registry,
        "require_environment_project",
        lambda **_kwargs: (_ for _ in ()).throw(
            HTTPException(status_code=404, detail="flash environment not found")
        ),
    )
    monkeypatch.setattr(
        envs_mod,
        "download_package",
        lambda **_kwargs: (_ for _ in ()).throw(
            envs_mod.EnvPublishError(
                "Flash control plane is missing its GitHub environment-hub credential", status=503
            )
        ),
    )
    monkeypatch.setattr(
        envs_mod,
        "delete_package",
        lambda **_kwargs: pytest.fail("a hub outage must not reach the package store"),
    )
    monkeypatch.setattr(
        environment_registry,
        "record_deleted_environment",
        lambda **_kwargs: pytest.fail("a failed delete must not touch the mirror"),
    )

    response = api.delete(
        f"/v1/envs/{namespace}/checkout-bot/my-env",
        headers={
            **_bearer(key),
            "X-Freesolo-Project-Id": "11111111-1111-4111-8111-111111111111",
        },
    )

    # a masked 404 would tell the client "already deleted" and stop it retrying a transient outage.
    assert response.status_code == 503, response.text
    assert "environment-hub credential" in response.json()["detail"]


def test_delete_env_missing_mirror_does_not_delete_existing_package(api, monkeypatch):
    from fastapi import HTTPException

    import flash.server.domain.registry.envs as envs_mod
    from flash.server.domain.registry import environment_registry

    key = _login()
    namespace = _identity_for_token(key)["org_slug"]
    monkeypatch.setattr(
        environment_registry,
        "require_environment_project",
        lambda **_kwargs: (_ for _ in ()).throw(
            HTTPException(status_code=404, detail="flash environment not found")
        ),
    )
    monkeypatch.setattr(envs_mod, "download_package", lambda **_kwargs: b"package")
    monkeypatch.setattr(
        envs_mod,
        "delete_package",
        lambda **_kwargs: pytest.fail("an unassociated package must not be deleted"),
    )
    monkeypatch.setattr(
        environment_registry,
        "record_deleted_environment",
        lambda **_kwargs: pytest.fail("a failed delete must not touch the mirror"),
    )

    response = api.delete(
        f"/v1/envs/{namespace}/checkout-bot/my-env",
        headers={
            **_bearer(key),
            "X-Freesolo-Project-Id": "11111111-1111-4111-8111-111111111111",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "flash environment not found"


def test_delete_env_endpoint_requires_project_header_before_storage(api, monkeypatch):
    import flash.server.domain.registry.envs as envs_mod

    monkeypatch.setattr(
        envs_mod, "delete_package", lambda **_k: pytest.fail("storage must not be touched")
    )
    response = api.delete("/v1/envs/acme/checkout-bot/my-env", headers=_bearer(_login()))
    assert response.status_code == 400
    assert "X-Freesolo-Project-Id is required" in response.text


def test_delete_env_validates_project_and_environment_before_storage(api, monkeypatch):
    import flash.server.domain.registry.envs as envs_mod
    import flash.server.domain.registry.projects as projects_mod
    from flash.server.domain.registry import environment_registry

    events: list[tuple[str, dict]] = []

    def require_project(**kwargs):
        events.append(("project", kwargs))
        return kwargs["project_id"]

    def require_environment(**kwargs):
        events.append(("environment", kwargs))

    def delete_package(**kwargs):
        events.append(("storage", kwargs))
        return True

    monkeypatch.setattr(projects_mod, "require_project_access", require_project)
    monkeypatch.setattr(environment_registry, "require_environment_project", require_environment)
    monkeypatch.setattr(envs_mod, "delete_package", delete_package)
    monkeypatch.setattr(environment_registry, "record_deleted_environment", lambda **_kwargs: True)

    response = api.delete(
        "/v1/envs/acme/checkout-bot/my-env",
        headers={
            **_bearer(_login()),
            "X-Freesolo-Project-Id": "11111111-1111-4111-8111-111111111111",
        },
    )

    assert response.status_code == 200, response.text
    assert [name for name, _kwargs in events] == ["project", "environment", "storage"]
    assert events[0][1]["authorization"].startswith("Bearer ")
    environment_call = events[1][1]
    assert environment_call["slug"] == "acme/checkout-bot/my-env"
    assert environment_call["project_id"] == "11111111-1111-4111-8111-111111111111"
    assert environment_call["key"]["org_id"].startswith("org-")
    assert environment_call["org_id"] is None


def test_delete_env_project_mismatch_blocks_storage(api, monkeypatch):
    from fastapi import HTTPException

    import flash.server.domain.registry.envs as envs_mod
    from flash.server.domain.registry import environment_registry

    monkeypatch.setattr(
        environment_registry,
        "require_environment_project",
        lambda **_kwargs: (_ for _ in ()).throw(
            HTTPException(status_code=409, detail="flash environment belongs to another project")
        ),
    )
    monkeypatch.setattr(
        envs_mod, "delete_package", lambda **_kwargs: pytest.fail("storage must not be touched")
    )

    response = api.delete(
        "/v1/envs/acme/checkout-bot/my-env",
        headers={
            **_bearer(_login()),
            "X-Freesolo-Project-Id": "22222222-2222-4222-8222-222222222222",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "flash environment belongs to another project"


def test_delete_env_endpoint_maps_publish_error_status(api, monkeypatch):
    """A namespace-authorization EnvPublishError surfaces as its HTTP status (403)."""
    import flash.server.domain.registry.envs as envs_mod

    def fake_delete_package(*, slug, key):
        raise envs_mod.EnvPublishError("not your namespace", status=403)

    monkeypatch.setattr(envs_mod, "delete_package", fake_delete_package)
    resp = api.delete(
        "/v1/envs/someone-else/checkout-bot/env",
        headers={
            **_bearer(_login()),
            "X-Freesolo-Project-Id": "11111111-1111-4111-8111-111111111111",
        },
    )
    assert resp.status_code == 403, resp.text


def test_delete_env_endpoint_mirror_failure_is_non_fatal(api, monkeypatch):
    """A failing metadata-mirror delete must not turn a successful delete into a 500."""
    import flash.server.domain.registry.envs as envs_mod
    from flash.server.domain.registry import environment_registry

    monkeypatch.setattr(envs_mod, "delete_package", lambda *, slug, key: True)

    def boom(*, slug, project_id, key, org_id=None):
        raise RuntimeError("backend down")

    monkeypatch.setattr(environment_registry, "record_deleted_environment", boom)
    resp = api.delete(
        "/v1/envs/acme/checkout-bot/my-env",
        headers={
            **_bearer(_login()),
            "X-Freesolo-Project-Id": "11111111-1111-4111-8111-111111111111",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["deleted"] is True


def test_delete_env_endpoint_rejects_non_canonical_id(api, monkeypatch):
    """A non-canonical id (uppercase / trailing slash) is rejected 400 before any storage call."""
    import flash.server.domain.registry.envs as envs_mod
    from flash.server.domain.registry import environment_registry

    monkeypatch.setattr(
        envs_mod, "delete_package", lambda **_k: pytest.fail("storage must not be touched")
    )
    monkeypatch.setattr(
        environment_registry,
        "record_deleted_environment",
        lambda **_k: pytest.fail("mirror must not be touched"),
    )
    for bad in ("Acme/Checkout-Bot/My-Env", "acme/checkout-bot/my-env/"):
        resp = api.delete(
            f"/v1/envs/{bad}",
            headers={
                **_bearer(_login()),
                "X-Freesolo-Project-Id": "11111111-1111-4111-8111-111111111111",
            },
        )
        assert resp.status_code == 400, resp.text


# --------------------------------------------------------------------------------------------
# Deployable RL checkpoints: list + deploy-by-step (incl. a run cancelled mid-RL).
# --------------------------------------------------------------------------------------------
_FAKE_CKPTS = [
    {
        "step": 40,
        "adapter_prefix": "rl/X/checkpoints/step-40",
        "subfolder": "rl/X/checkpoints/step-40/adapter",
        "repo_id": "org/test-runs",
        "repo_type": "dataset",
    },
    {
        "step": 80,
        "adapter_prefix": "rl/X/checkpoints/step-80",
        "subfolder": "rl/X/checkpoints/step-80/adapter",
        "repo_id": "org/test-runs",
        "repo_type": "dataset",
    },
]


_MISSING_SMOKE_REASONING = object()


def _expected_smoke_colour(run_id: str) -> str:
    """The colour this run_id's deployment smoke must answer.

    derived from production rather than hardcoded: the smoke picks one of several trusted colours
    per run, so a fixed literal here would break whenever the hash landed elsewhere.
    """
    from flash.server.routes import serving_smoke

    expected, _messages = serving_smoke._smoke_image_challenge(run_id)
    return expected


def _smoke_chat_result(
    checkpoint_id: str,
    checkpoint: str,
    content: str | None = None,
    *,
    reasoning_content: object = _MISSING_SMOKE_REASONING,
) -> dict:
    # a serve_chat response that passes exact checkpoint provenance validation
    if content is None:
        content = _expected_smoke_colour(checkpoint_id.rsplit("/", 1)[0])
    message = {"content": content}
    if reasoning_content is not _MISSING_SMOKE_REASONING:
        message["reasoning_content"] = reasoning_content
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "freesolo": {"checkpoint_id": checkpoint_id},
        "_freesolo_headers": {"checkpoint_id": checkpoint_id},
        "_freesolo_lora_request_adapter": checkpoint_id,
    }


class _FakeDeployment:
    def __init__(self, adapter_prefix):
        self.adapter_prefix = adapter_prefix

    def to_dict(self):
        return {
            "state": "ready",
            "run_id": "X",
            "adapter_hf_prefix": f"{self.adapter_prefix}/adapter",
        }


def _make_run(api, key, state):
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]

    status = runner_status.get_status(run_id)
    status.state = state
    runner_state._save_status(status)
    return run_id


def test_list_checkpoints_endpoint(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    key = _login()
    run_id = _make_run(api, key, "done")
    r = api.get(f"/v1/runs/{run_id}/checkpoints", headers=_bearer(key))
    assert r.status_code == 200, r.text
    assert [c["step"] for c in r.json()["checkpoints"]] == [40, 80]


def test_deploy_specific_checkpoint_of_finished_run(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    captured = {}
    key = _login()
    run_id = _make_run(api, key, "done")
    revision = f"{run_id}/step-40"

    def fake_deploy(**kwargs):
        captured.update(kwargs)
        kwargs["before_ready"](revision, f"{run_id}/step-40")
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: _smoke_chat_result(revision, f"{run_id}/step-40"),
    )

    r = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"checkpoint_id": f"{run_id}/step-40"},
        headers=_bearer(key),
    )
    assert r.status_code == 200, r.text
    # Served the step-40 checkpoint's adapter, not the run's final adapter.
    assert captured["adapter_prefix"].endswith("/checkpoints/step-40")
    assert r.json()["checkpoint_step"] == 40
    # A finished run flips to `deployed` as usual.

    assert runner_status.get_status(run_id).state == "deployed"


def test_deploy_checkpoint_of_cancelled_run_keeps_terminal_state(api, monkeypatch):
    """The headline fix: a run cancelled mid-RL can deploy a checkpoint, and stays `cancelled`."""
    import flash.server.asgi.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)

    key = _login()
    run_id = _make_run(api, key, "cancelled")
    revision = f"{run_id}/step-80"

    def fake_deploy(**kwargs):
        kwargs["before_ready"](revision, f"{run_id}/step-80")
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: _smoke_chat_result(revision, f"{run_id}/step-80"),
    )
    r = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"checkpoint_id": f"{run_id}/step-80"},
        headers=_bearer(key),
    )
    assert r.status_code == 200, r.text
    assert r.json()["checkpoint_step"] == 80
    # Training outcome preserved (NOT flipped to `deployed`)...
    assert runner_status.get_status(run_id).state == "cancelled"
    # ...but the serving deployment is recorded and listed as active.
    deployments = api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"]
    assert any(d["run_id"] == run_id for d in deployments)


@pytest.mark.parametrize("state", ["queued", "provisioning", "running", "failed"])
def test_deploy_checkpoint_ignores_run_state_once_step_exists(api, monkeypatch, state):
    """A resolved checkpoint step proves the adapter exists, so run state does not gate serving it."""
    import flash.server.asgi.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)

    key = _login()
    run_id = _make_run(api, key, state)
    revision = f"{run_id}/step-40"

    def fake_deploy(**kwargs):
        kwargs["before_ready"](revision, f"{run_id}/step-40")
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: _smoke_chat_result(revision, f"{run_id}/step-40"),
    )
    r = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"checkpoint_id": f"{run_id}/step-40"},
        headers=_bearer(key),
    )
    assert r.status_code == 200, r.text
    assert r.json()["checkpoint_step"] == 40
    status = runner_status.get_status(run_id)
    assert status.state == state
    assert status.deployment["checkpoint_step"] == 40
    deployments = api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"]
    assert any(d["run_id"] == run_id and d["state"] == state for d in deployments)


def test_deploy_checkpoint_promotes_if_run_finishes_during_registration(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    key = _login()
    run_id = _make_run(api, key, "running")
    revision = f"{run_id}/step-40"

    def fake_deploy(**kwargs):
        status = runner_status.get_status(run_id)
        status.state = "done"
        runner_state._save_status(status)
        kwargs["before_ready"](revision, f"{run_id}/step-40")
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: _smoke_chat_result(revision, f"{run_id}/step-40"),
    )

    r = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"checkpoint_id": f"{run_id}/step-40"},
        headers=_bearer(key),
    )
    assert r.status_code == 200, r.text
    status = runner_status.get_status(run_id)
    assert status.state == "deployed"
    assert status.deployment["checkpoint_step"] == 40


def test_deploy_checkpoint_preserves_final_deploy_that_wins_cas(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    undeploys = []
    monkeypatch.setattr(app_mod, "undeploy_adapter", lambda run_id, **_: undeploys.append(run_id))

    key = _login()
    run_id = _make_run(api, key, "done")

    final_deployment = {
        "state": "ready",
        "endpoint_name": "final",
        "checkpoint_id": f"{run_id}/final",
    }

    def fake_deploy(**kwargs):
        runner_transitions.mark_deployed(
            run_id,
            final_deployment,
            verification_generation=runner_verified_revisions.verified_checkpoint_generation(
                run_id
            ),
        )
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)

    r = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"checkpoint_id": f"{run_id}/step-40"},
        headers=_bearer(key),
    )
    assert r.status_code == 200, r.text
    assert undeploys == []
    status = runner_status.get_status(run_id)
    assert status.state == "deployed"
    assert status.deployment == final_deployment


def test_deploy_checkpoint_of_dry_run_run_is_409(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **_k: pytest.fail("dry-run run must not touch serving"),
    )

    key = _login()
    run_id = _make_run(api, key, "dry_run")
    r = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"checkpoint_id": f"{run_id}/step-40"},
        headers=_bearer(key),
    )
    assert r.status_code == 409, r.text
    assert "dry-run runs cannot be deployed" in r.json()["detail"]


def test_undeploy_checkpoint_of_running_run_keeps_training_state(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    monkeypatch.setattr(app_mod, "deploy_adapter", lambda **k: _FakeDeployment(k["adapter_prefix"]))
    monkeypatch.setattr(app_mod, "undeploy_adapter", lambda run_id, **_: [run_id])

    key = _login()
    run_id = _make_run(api, key, "running")
    r = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"checkpoint_id": f"{run_id}/step-40"},
        headers=_bearer(key),
    )
    assert r.status_code == 200, r.text
    revision = f"{run_id}/step-40"
    runner_verified_revisions.add_verified_checkpoint(
        run_id,
        revision,
        expected_generation=runner_verified_revisions.verified_checkpoint_generation(run_id),
    )

    r = api.delete(f"/v1/runs/{run_id}/deploy?checkpoint_id={revision}", headers=_bearer(key))
    assert r.status_code == 200, r.text
    status = runner_status.get_status(run_id)
    assert status.state == "running"
    assert status.deployment["state"] == "undeployed"
    assert runner_verified_revisions.read_verified_checkpoints(run_id) == frozenset()


def test_deploy_cancelled_run_without_step_is_409(api):
    """Without a step, a cancelled run is still undeployable (no final adapter)."""
    key = _login()
    run_id = _make_run(api, key, "cancelled")
    r = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert r.status_code == 409, r.text


def test_deploy_unknown_step_is_404_with_available(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    key = _login()
    run_id = _make_run(api, key, "done")
    r = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"checkpoint_id": f"{run_id}/step-999"},
        headers=_bearer(key),
    )
    assert r.status_code == 404, r.text
    assert "available: 40, 80" in r.json()["detail"]


def test_deploy_rejects_non_integer_step(api, monkeypatch):
    """A bool (True->1) or non-integer step must be rejected, not silently coerced. An all-digit string
    over Python's 4300-digit int()-conversion limit must also be a clean 400, not int()->uncaught 500."""
    import flash.server.asgi.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    key = _login()
    run_id = _make_run(api, key, "done")
    for bad in (True, 40.9, "40.9", "1" * 5000):
        r = api.post(
            f"/v1/runs/{run_id}/deploy",
            json={"checkpoint_id": f"{run_id}/step-{bad}"},
            headers=_bearer(key),
        )
        assert r.status_code == 400, f"{bad!r} -> {r.status_code} {r.text}"


def test_create_run_records_managed_environment_use(api, monkeypatch):
    import flash.server.domain.registry.environment_registry as registry

    calls: list[dict] = []
    monkeypatch.setattr(
        registry,
        "record_environment_use",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    spec = {**SPEC, "environment": {"id": "acme/checkout-bot/my-env"}}
    key = _login()

    resp = api.post(
        "/v1/runs",
        headers=_bearer(key),
        json={"spec": spec, "dry_run": True},
    )

    assert resp.status_code == 200, resp.text
    assert calls
    assert calls[0]["slug"] == "acme/checkout-bot/my-env"
    assert calls[0]["project_id"] == "11111111-1111-4111-8111-111111111111"
    assert calls[0]["run_id"] == resp.json()["run_id"]
    assert calls[0]["key"]["org_id"] == f"org-{key.removeprefix(_USER_PREFIX)}"


def test_internal_run_environment_use_merges_header_org(api, monkeypatch):
    import flash.server.domain.registry.environment_registry as registry

    calls: list[dict] = []
    monkeypatch.setattr(
        registry,
        "record_environment_use",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    spec = {**SPEC, "environment": {"id": "acme/checkout-bot/my-env"}}

    response = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": spec, "dry_run": True},
    )

    assert response.status_code == 200, response.text
    assert calls[0]["key"]["org_id"] == "org-test"


def test_create_run_records_flash_training_run(api, monkeypatch):
    import flash.server.domain.registry.runs as registry

    calls: list[dict] = []
    monkeypatch.setattr(
        registry,
        "record_training_run",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    key = _login()

    resp = api.post(
        "/v1/runs",
        headers=_bearer(key),
        json={"spec": SPEC, "dry_run": True},
    )

    assert resp.status_code == 200, resp.text
    assert calls
    last = calls[-1]
    assert last["status"].run_id == resp.json()["run_id"]
    # Org attribution rides on the persisted platform_context (org_id/user_id/api_key_id),
    # which submit_job reports for us — create_run no longer double-POSTs with an explicit key.
    assert last["status"].platform_context["org_id"] == f"org-{key.removeprefix(_USER_PREFIX)}"
    assert last["status"].platform_context["project_id"] == "11111111-1111-4111-8111-111111111111"


# --- export: copy a trained adapter to a user-owned HuggingFace repo ----------------------


def _finished_run(api, key) -> str:
    """Submit a run and flip it to `done` (a finished run with a trained final adapter)."""

    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner_status.get_status(run_id)
    status.state = "done"
    runner_state._save_status(status)
    return run_id


def test_export_copies_final_adapter_to_user_repo(api, monkeypatch):
    """A finished run's final adapter is read privately and exported with a public source ref."""
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _finished_run(api, key)
    # The platform auto-assigns each run a per-run HF dataset repo under the OPERATOR's org, so
    # only the control plane (operator token) can read the source. hf_repo is platform-managed and
    # stripped from the public spec, so read it back from the internal worker-spec carrier.
    src_repo = runner_status.get_status(run_id).effective_preparation["worker_spec"]["train"][
        "hf_repo"
    ]

    seen: dict = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return "https://huggingface.co/me/adapters"

    monkeypatch.setattr(app_mod, "export_adapter", capture)

    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={
            "repository": "me/adapters",
            "hf_token": "hf_user",
            "checkpoint_id": f"{run_id}/final",
        },
        headers=_bearer(key),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["repository"] == "me/adapters"
    assert body["url"] == "https://huggingface.co/me/adapters"
    assert body["source"] == f"{run_id}/final"
    assert src_repo not in resp.text
    assert "step" not in body
    # the operator still reads the private source internally; only the response uses the public ref.
    assert seen["source_repo"] == src_repo
    assert seen["source_subfolder"] == f"rl/{run_id}/adapter"
    assert seen["dest_repo"] == "me/adapters"
    assert seen["dest_token"] == "hf_user"
    assert seen["private"] is True  # private by default
    assert seen["base_model"] == SPEC["model"]


def test_export_sends_the_runner_assigned_revision_not_the_public_blank(api, monkeypatch):
    """Export must read the pin from the EFFECTIVE spec, not the public one.

    A runner-assigned pin is stripped from the public spec (it cannot carry the marker that labels
    it), so `spec.model_revision` reads "" for every auto-pinned SFT run. The worker stamps the real
    sha into adapter_config.json from its INTERNAL spec, and `export_adapter` refuses a stamped
    revision that disagrees with the one it is handed:

        if existing_revision and existing_revision != base_model_revision: raise ValueError(...)

    which the route turns into a 404. So passing the public half breaks export for exactly the runs
    the auto-pin exists to serve -- and for warm starts that inherited the pin.

    Asserting the value rather than just capturing kwargs is the point: the pre-existing export
    tests captured `base_model_revision` and never checked it, which is why this reached review.
    """
    import flash.server.asgi.app as app_mod
    from flash.core.spec import JobSpec

    key = _login()
    run_id = _finished_run(api, key)
    status = runner_status.get_status(run_id)
    # the shape a real auto-pinned submit persists: worker half carries pin + marker, public half
    # carries neither. re-digest the way submit does so the snapshot stays internally consistent.
    snapshot = status.effective_preparation
    assert not status.spec.get("model_revision"), status.spec
    snapshot["worker_spec"]["model_revision"] = "a" * 40
    snapshot["worker_spec"]["model_revision_auto"] = True
    snapshot["preparation_digest"] = runner_preparation._preparation_digest(
        JobSpec.from_dict(status.spec),
        JobSpec.from_dict(snapshot["worker_spec"]),
        snapshot.get("adapter_identity"),
    )
    runner_state._save_status(status)

    seen: dict = {}
    monkeypatch.setattr(
        app_mod,
        "export_adapter",
        lambda **kwargs: (seen.update(kwargs), "https://huggingface.co/me/adapters")[1],
    )

    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={
            "repository": "me/adapters",
            "hf_token": "hf_user",
            "checkpoint_id": f"{run_id}/final",
        },
        headers=_bearer(key),
    )

    assert resp.status_code == 200, resp.text
    assert seen["base_model_revision"] == "a" * 40


@pytest.mark.parametrize("operation", ["export", "undeploy"])
def test_unauthorized_deployment_operation_returns_while_lock_is_held(api, monkeypatch, operation):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    import flash.server.asgi.app as app_mod

    owner = _login()
    other = _login()
    run_id = _finished_run(api, owner)
    calls = []
    monkeypatch.setattr(app_mod, "export_adapter", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(app_mod, "undeploy_adapter", lambda target: calls.append(target))

    deploy_lock = app_mod._deploy_lock(run_id)
    assert deploy_lock.acquire(blocking=False) is True
    executor = ThreadPoolExecutor(max_workers=1)
    if operation == "export":
        future = executor.submit(
            api.post,
            f"/v1/runs/{run_id}/export",
            json={},
            headers=_bearer(other),
        )
    else:
        future = executor.submit(
            api.delete,
            f"/v1/runs/{run_id}/deploy?checkpoint_id={run_id}/final",
            headers=_bearer(other),
        )
    try:
        try:
            response = future.result(timeout=2)
        except TimeoutError:
            pytest.fail(f"unauthorized {operation} waited for the deployment lock")
        assert deploy_lock.acquire(blocking=False) is False
    finally:
        deploy_lock.release()
        executor.shutdown(wait=True)

    assert response.status_code == 404, response.text
    assert calls == []


def test_invalid_export_returns_while_lock_is_held(api, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    import flash.server.asgi.app as app_mod

    owner = _login()
    run_id = _finished_run(api, owner)
    calls = []
    monkeypatch.setattr(app_mod, "export_adapter", lambda **kwargs: calls.append(kwargs))

    deploy_lock = app_mod._deploy_lock(run_id)
    assert deploy_lock.acquire(blocking=False) is True
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        api.post,
        f"/v1/runs/{run_id}/export",
        json={},
        headers=_bearer(owner),
    )
    try:
        try:
            response = future.result(timeout=2)
        except TimeoutError:
            pytest.fail("invalid export waited for the deployment lock")
        assert deploy_lock.acquire(blocking=False) is False
    finally:
        deploy_lock.release()
        executor.shutdown(wait=True)

    assert response.status_code == 400, response.text
    assert "repository" in response.json()["detail"]
    assert calls == []


@pytest.mark.parametrize("operation", ["export", "undeploy"])
def test_deployment_operation_rechecks_authorization_after_waiting(api, monkeypatch, operation):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import flash.server.asgi.app as app_mod
    from flash.server.routes import serving as serving_routes

    owner = _login()
    run_id = _finished_run(api, owner)
    authorization_checked = threading.Event()
    authorization_attempts = []
    authorized_statuses = []
    calls = []
    authorization_name = "owned_run" if operation == "export" else "manageable_run"
    real_authorize = getattr(serving_routes, authorization_name)

    def observed_authorize(*args, **kwargs):
        authorization_attempts.append(None)
        status = real_authorize(*args, **kwargs)
        authorized_statuses.append(status)
        authorization_checked.set()
        return status

    monkeypatch.setattr(serving_routes, authorization_name, observed_authorize)
    monkeypatch.setattr(app_mod, "export_adapter", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(app_mod, "undeploy_adapter", lambda target: calls.append(target))

    deploy_lock = app_mod._deploy_lock(run_id)
    assert deploy_lock.acquire(blocking=False) is True
    executor = ThreadPoolExecutor(max_workers=1)
    if operation == "export":
        future = executor.submit(
            api.post,
            f"/v1/runs/{run_id}/export",
            json={"repository": "me/a", "hf_token": "hf"},
            headers=_bearer(owner),
        )
    else:
        future = executor.submit(
            api.delete,
            f"/v1/runs/{run_id}/deploy?checkpoint_id={run_id}/final",
            headers=_bearer(owner),
        )
    try:
        assert authorization_checked.wait(timeout=2)
        assert len(authorization_attempts) == 1
        assert len(authorized_statuses) == 1
        _db_mod.delete_run(run_id)
    finally:
        deploy_lock.release()
    response = future.result(timeout=5)
    executor.shutdown(wait=True)

    assert response.status_code == 404, response.text
    assert len(authorization_attempts) == 2
    assert len(authorized_statuses) == 1
    assert calls == []


def test_export_public_flag_sets_private_false(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _finished_run(api, key)
    seen: dict = {}
    monkeypatch.setattr(
        app_mod,
        "export_adapter",
        lambda **kw: (seen.update(kw), "https://huggingface.co/me/a")[1],
    )
    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={
            "repository": "me/a",
            "hf_token": "hf",
            "private": False,
            "checkpoint_id": f"{run_id}/final",
        },
        headers=_bearer(key),
    )
    assert resp.status_code == 200, resp.text
    assert seen["private"] is False


def test_export_validates_repository_and_token(api):
    key = _login()
    run_id = _finished_run(api, key)
    missing_repo = api.post(
        f"/v1/runs/{run_id}/export",
        json={"hf_token": "hf", "checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )
    assert missing_repo.status_code == 400
    assert "repository" in missing_repo.json()["detail"]
    missing_token = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )
    assert missing_token.status_code == 400
    assert "hf_token" in missing_token.json()["detail"]
    # SHAPE check: a HF repo id is EXACTLY two non-empty segments — reject no-slash AND the over-/
    # under-segmented forms that the old "at least one '/'" check let through (which would 404/400 deep
    # in hf_hub). These produce the "owner/name" shape error.
    for bad_repo in ("noslash", "owner/name/extra", "owner//name", "/name", "name/"):
        malformed = api.post(
            f"/v1/runs/{run_id}/export",
            json={"repository": bad_repo, "hf_token": "hf", "checkpoint_id": f"{run_id}/final"},
            headers=_bearer(key),
        )
        assert malformed.status_code == 400, bad_repo
        assert "owner/name" in malformed.json()["detail"]
    # GRAMMAR check: two segments but NOT a valid HF repo id — embedded whitespace, a segment that
    # starts/ends with '-' or '.', a '--'/'..' run, or a >96-char name. The full Hub grammar
    # (huggingface_hub.validate_repo_id) must reject these FAST with a 400, not let export_adapter
    # download the private source adapter first and hit a wrapped 502 from create_repo.
    for bad_repo in (
        "owner/ name",
        "own er/name",
        "owner/na\tme",
        "owner/na me",
        "owner/-bad",
        "owner/bad-",
        "owner/.bad",
        "owner/bad--name",
        "owner/ba..d",
        "owner/" + "x" * 97,
    ):
        malformed = api.post(
            f"/v1/runs/{run_id}/export",
            json={"repository": bad_repo, "hf_token": "hf", "checkpoint_id": f"{run_id}/final"},
            headers=_bearer(key),
        )
        assert malformed.status_code == 400, bad_repo
        assert "valid HuggingFace repo id" in malformed.json()["detail"], bad_repo


def test_export_unfinished_run_is_409(api, monkeypatch):
    """A run with no trained final adapter (never finished) can't be exported — and the HF copy
    is never attempted."""
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]

    def must_not_run(**kw):
        raise AssertionError("export_adapter must not run for an unfinished run")

    monkeypatch.setattr(app_mod, "export_adapter", must_not_run)
    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf", "checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )
    assert resp.status_code == 409, resp.text


def test_export_missing_artifacts_is_404(api, monkeypatch):
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _finished_run(api, key)

    def boom(**kw):
        raise ValueError("no adapter artifacts found at org/test-runs:... (nothing to export)")

    monkeypatch.setattr(app_mod, "export_adapter", boom)
    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf", "checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )
    assert resp.status_code == 404, resp.text
    assert "no adapter artifacts" in resp.json()["detail"]


def test_export_hf_failure_is_clean_502(api, monkeypatch):
    """An HF transport/permission failure (download or upload) surfaces as a clean 502 carrying the
    real reason — not an unhandled 500 (mirrors the deploy/undeploy ServingError handling)."""
    import flash.server.asgi.app as app_mod
    from flash.serve.contract.errors import ServingError

    key = _login()
    run_id = _finished_run(api, key)

    def boom(**kw):
        raise ServingError("could not upload adapter to me/a: 403 Forbidden")

    monkeypatch.setattr(app_mod, "export_adapter", boom)
    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf", "checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )
    assert resp.status_code == 502, resp.text
    assert "could not upload" in resp.json()["detail"]


def test_export_step_targets_the_checkpoint_adapter(api, monkeypatch):
    """Step export targets the exact per-step checkpoint `flash deploy RUN_ID/step-N` would serve; an
    unknown step 404s with the available list (resolved against published checkpoints)."""
    import flash.server.asgi.app as app_mod

    key = _login()
    run_id = _finished_run(api, key)
    monkeypatch.setattr(
        app_mod,
        "list_checkpoints",
        lambda spec: [
            {
                "step": 40,
                "subfolder": f"rl/{run_id}/checkpoints/step-40/adapter",
                "repo_id": "org/test-runs",
                "repo_type": "dataset",
            }
        ],
    )
    seen: dict = {}
    monkeypatch.setattr(
        app_mod,
        "export_adapter",
        lambda **kw: (seen.update(kw), "https://huggingface.co/me/a")[1],
    )
    ok = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf", "checkpoint_id": f"{run_id}/step-40"},
        headers=_bearer(key),
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["step"] == 40
    assert ok.json()["source"] == f"{run_id}/step-40"
    assert "org/test-runs" not in ok.text
    assert seen["source_subfolder"] == f"rl/{run_id}/checkpoints/step-40/adapter"
    assert seen["base_model"] == SPEC["model"]

    bad = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf", "checkpoint_id": f"{run_id}/step-99"},
        headers=_bearer(key),
    )
    assert bad.status_code == 404, bad.text
    assert "step 99" in bad.json()["detail"]


# --- export: product-analytics report ------------------------------------------------------


def test_export_reports_product_analytics_event(api, monkeypatch):
    """A successful export fires the platform product-event reporter (best-effort) with the
    destination repo, url, and step; the report failing must never fail the export itself."""
    import flash.server.asgi.app as app_mod
    import flash.server.domain.registry.runs as runs

    key = _login()
    run_id = _finished_run(api, key)
    monkeypatch.setattr(
        app_mod, "export_adapter", lambda **kw: "https://huggingface.co/me/adapters"
    )

    seen: dict = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(runs, "record_model_exported", capture)

    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={
            "repository": "me/adapters",
            "hf_token": "hf_user",
            "checkpoint_id": f"{run_id}/final",
        },
        headers=_bearer(key),
    )
    assert resp.status_code == 200, resp.text
    assert seen["repository"] == "me/adapters"
    assert seen["url"] == "https://huggingface.co/me/adapters"
    assert seen["step"] is None
    assert seen["status"].run_id == run_id


def test_export_succeeds_even_when_analytics_report_raises(api, monkeypatch):
    import flash.server.asgi.app as app_mod
    import flash.server.domain.registry.runs as runs

    key = _login()
    run_id = _finished_run(api, key)
    monkeypatch.setattr(
        app_mod, "export_adapter", lambda **kw: "https://huggingface.co/me/adapters"
    )

    def boom(**_kwargs):
        raise RuntimeError("backend unreachable")

    monkeypatch.setattr(runs, "record_model_exported", boom)

    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={
            "repository": "me/adapters",
            "hf_token": "hf_user",
            "checkpoint_id": f"{run_id}/final",
        },
        headers=_bearer(key),
    )
    assert resp.status_code == 200, resp.text


def test_record_model_exported_posts_allowlisted_event(monkeypatch):
    """The reporter posts the flash_model_exported event with org/user attribution and the
    export detail; no org in context disables the report entirely."""
    import flash.server.domain.registry.runs as runs

    posted: dict = {}
    monkeypatch.setattr(
        runs,
        "_post",
        lambda path, body: posted.update({"path": path, "body": body}) or True,
    )

    class _Status:
        def __init__(self):
            self.run_id = "run-9"
            self.platform_context = {"org_id": "org-A", "user_id": "user-1"}
            self.billing_context = None
            self.spec = {
                "project": "11111111-1111-4111-8111-111111111111",
                "model": "Qwen/Qwen3.5-9B",
            }

    ok = runs.record_model_exported(
        status=_Status(),
        repository="me/adapters",
        url="https://huggingface.co/me/adapters",
        step=120,
    )
    assert ok is True
    assert posted["path"] == "/api/flash/events/internal"
    assert posted["body"]["orgId"] == "org-A"
    assert posted["body"]["userId"] == "user-1"
    assert posted["body"]["event"] == "flash_model_exported"
    props = posted["body"]["properties"]
    assert props == {
        "project": "11111111-1111-4111-8111-111111111111",
        "run_id": "run-9",
        "repository": "me/adapters",
        "url": "https://huggingface.co/me/adapters",
        "step": 120,
        "model": "Qwen/Qwen3.5-9B",
    }

    class _NoOrg:
        def __init__(self):
            self.run_id = "run-9"
            self.platform_context = None
            self.billing_context = None
            self.spec = {}

    posted.clear()
    assert (
        runs.record_model_exported(status=_NoOrg(), repository="x/y", url="https://x", step=None)
        is False
    )
    assert posted == {}


@pytest.mark.parametrize(
    ("plane_misconfigured", "status_code"),
    [
        # the plane's own credential is unset: nothing the submitter can author fixes it
        (True, 503),
        # the submitted shape exceeds the broker's limits: their spec is the fix
        (False, 400),
    ],
)
def test_managed_teacher_rejection_reports_which_side_is_at_fault(
    api, monkeypatch, tmp_path, plane_misconfigured, status_code
) -> None:
    """A plane outage must not be reported as a bad request.

    The submit path funnels every exception into one 400, which is right for a spec the user must
    change and wrong for a credential only an operator can set. Reporting both as a client error
    would re-create, at the HTTP layer, the same "misconfiguration looks like a spec error"
    conflation that hoisting this gate to submit time exists to end.
    """
    import flash.envs.loading.loader as envs_loader
    from tests._helpers.teacher import configure_managed_teacher

    configure_managed_teacher(monkeypatch)
    # offline: the image-opd preflight ahead of this gate resolves the environment reference to
    # inspect its dataset. point it at an empty local dir so no github request is made.
    offline_env = tmp_path / "offline-env"
    offline_env.mkdir()
    (offline_env / "environment.py").write_text("")
    monkeypatch.setattr(envs_loader, "_resolve_ref_sha", lambda *_a, **_k: "0" * 40)
    monkeypatch.setattr(
        envs_loader,
        "_resolve_environment_reference",
        lambda *_a, **_k: str(offline_env / "environment.py"),
    )
    train = {**SPEC["train"], "teacher_model": "glm-5.2"}
    if plane_misconfigured:
        monkeypatch.delenv("PARASAIL_API_KEY", raising=False)
    else:
        # a shape whose planned teacher score items exceed the broker ceiling. the multiplier is
        # 192 (64 turns x 3 no-signal attempts), so this asks for ~9.8M against a 1M limit.
        train |= {"max_steps": 100, "batch_size": 64, "group_size": 8, "max_examples": 6400}
    spec = {**SPEC, "algorithm": "opd", "train": train}

    response = api.post("/v1/runs", headers=_bearer(_login()), json={"spec": spec, "dry_run": True})

    assert response.status_code == status_code, response.text
    detail = str(response.json()["detail"])
    # the reason has to survive the mapping too: a bare status code still leaves an operator
    # guessing which of the two plane-side names is unset.
    assert ("PARASAIL_API_KEY" in detail) is plane_misconfigured, detail

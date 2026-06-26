"""Cross-component integration: the *real* Flash ``ApiClient`` driving the
*real* control-plane FastAPI app in-process.

The freesolo SDK / training worker talks to the managed Flash service through
``flash.client.http.ApiClient`` (stdlib ``urllib``). Its unit tests
(``test_client.py``) assert request shaping against a hand-rolled fake server,
and ``test_server_api.py`` drives the app with a ``TestClient`` and raw JSON --
but nothing exercises the *pair* together, so the client's spec serialisation
and the server's ``spec_from_dict`` validation can drift apart silently.

Here the real ``ApiClient`` runs against the real ``create_app()`` ASGI app: the
client's ``urllib`` transport is shimmed onto a Starlette ``TestClient`` so no
socket is opened, but the client's request building, JSON (de)serialisation and
``ApiError``/``ClientError`` mapping are all the production code paths. A drift
on either side -- a renamed RunStatus field, a tightened spec validator, a
changed status code -- fails here.

Runs offline (``uv run pytest``); ``_run_job`` is stubbed so
``create_run`` never provisions a GPU.
"""

from __future__ import annotations

import importlib
import io
import itertools
import os
import urllib.error
import urllib.parse
import urllib.request

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from flash.client.http import ApiClient, ApiError

# A representative managed-run spec -- the shape the freesolo bridge / SDK
# submits: catalog model, Freesolo environment id, and an HF repo for artifacts.
SPEC = {
    "model": "Qwen/Qwen3.5-4B",
    "algorithm": "grpo",
    "environment": {"id": "freesolo/gsm8k", "params": {"max_examples": 8}},
    "train": {"steps": 1, "seeds": [0], "hf_repo": "org/test-runs"},
    "gpu": {"type": "RTX 5090"},
}

_USER_PREFIX = "fslo-user-"
_counter = itertools.count()


def _token() -> str:
    """A fresh freesolo user token accepted by the fixture's stub verify."""
    return f"{_USER_PREFIX}{next(_counter)}"


def _identity_for_token(token: str) -> dict[str, str]:
    if not token.startswith(_USER_PREFIX):
        return {}
    suffix = token.removeprefix(_USER_PREFIX)
    return {
        "email": f"user-{suffix}@example.com",
        "key_prefix": "fslo_test",
        "org_id": f"org-{suffix}",
    }


class _FakeUrlResponse:
    """Mimics the ``urllib`` response object ``ApiClient._request`` consumes."""

    def __init__(self, body: bytes, status: int) -> None:
        self._body = body
        self.status = status
        self.code = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeUrlResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    """Yields a factory building real ``ApiClient``s wired into one in-process
    control-plane app via a ``urllib`` -> ``TestClient`` shim."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test,rp-test-2")
    monkeypatch.setenv("LAMBDA_API_KEY", "lam-test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    # runpod.keys caches the parsed pool on first read; reset so the startup preflight reads THIS
    # RUNPOD_API_KEY (the autouse _offline fixture also resets, but make the fixture self-contained).
    import flash.providers.runpod.keys as runpod_keys

    runpod_keys.reset()

    import flash.runner as runner
    import flash.server.auth as auth_mod
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))
    # Keep ``create_run`` offline: the spec is really validated and a run row is
    # really recorded, but the GPU job body is a no-op (no provisioning thread
    # work). The client request/response contract is unaffected.
    monkeypatch.setattr(runner, "_run_job", lambda *a, **k: None)

    import flash.server.app as app_mod

    importlib.reload(app_mod)
    # The Lambda key above (required by the startup preflight) also makes
    # configured_providers() treat it as live, so create_app()'s lifespan recover_runs() would
    # dispatch real sweep_orphans() list calls — and the urllib->TestClient shim below isn't installed
    # until AFTER create_app() runs. Stub the provider set to empty so startup stays hermetic.
    import flash.providers as providers_mod
    import flash.providers.runpod.train.endpoints as rp_endpoints
    import flash.server.run_registry as run_registry

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [], raising=False)
    # FREESOLO_INTERNAL_KEY also makes startup run the RunPod slot-store reconcile
    # (reconcile_endpoint_slots() -> runpod.slots.reconcile() urllib POST) BEFORE the urllib->
    # TestClient shim below is installed, so it would hit real network. No-op it at the entry.
    monkeypatch.setattr(rp_endpoints, "reconcile_endpoint_slots", lambda *a, **k: None, raising=False)
    # And the same key makes each run-status update best-effort report via run_registry._post(); with
    # the urllib shim that POST would otherwise route into THIS app (no such route) and log a 404 warn
    # on every status change. Stub it (as the other server fixtures do) to keep output clean.
    monkeypatch.setattr(run_registry, "_post", lambda *a, **k: False, raising=False)
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token.startswith(_USER_PREFIX))
    monkeypatch.setattr(auth_mod, "_cached_identity", _identity_for_token)
    with TestClient(app_mod.create_app()) as test_client:

        def fake_urlopen(req: urllib.request.Request, timeout: float | None = None):
            parsed = urllib.parse.urlsplit(req.full_url)
            path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            headers = dict(req.header_items())
            resp = test_client.request(req.get_method(), path, content=req.data, headers=headers)
            body = resp.content
            if resp.status_code >= 400:
                raise urllib.error.HTTPError(
                    req.full_url,
                    resp.status_code,
                    resp.reason_phrase,
                    resp.headers,  # type: ignore[arg-type]
                    io.BytesIO(body),
                )
            return _FakeUrlResponse(body, resp.status_code)

        # Route the client's urllib through the ASGI app (no real socket).
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        def factory(api_key: str | None = None, *, login: bool = True) -> ApiClient:
            if api_key is None and login:
                api_key = _token()
            return ApiClient(api_url="http://testserver", api_key=api_key)

        yield factory


def test_health_and_identity_roundtrip(make_client) -> None:
    client = make_client()

    health = client.health()
    assert health["ok"] is True
    assert health["service"] == "flash"

    me = client.me()
    # The server resolves the bearer to a per-key identity and the client parses
    # the JSON body back into a dict the CLI prints.
    assert me["key_prefix"]
    assert "email" in me


def test_create_status_list_cancel_lifecycle(make_client) -> None:
    client = make_client()

    created = client.create_run(SPEC)
    run_id = created["run_id"]
    assert run_id
    assert created["state"] == "queued"
    # The server echoes the normalised spec; the client must receive it intact.
    assert created["spec"]["model"] == SPEC["model"]
    assert created["spec"]["algorithm"] == "grpo"

    fetched = client.get_run(run_id)
    assert fetched["run_id"] == run_id

    listed = client.list_runs()
    assert run_id in [r["run_id"] for r in listed]

    cancelled = client.cancel_run(run_id)
    assert cancelled["run_id"] == run_id
    # Cancelling a non-terminal run flips it to "cancelled".
    assert cancelled["state"] == "cancelled"


def test_logs_offset_paging_roundtrip(make_client) -> None:
    import flash.runner as runner

    client = make_client()
    run_id = client.create_run(SPEC)["run_id"]

    log_path = os.path.join(runner.RUNS_DIR, f"{run_id}.log")
    with open(log_path, "w") as fh:
        fh.write("first line\n")

    page = client.get_logs(run_id)
    assert page["logs"] == "first line\n"
    assert page["state"] == "queued"

    with open(log_path, "a") as fh:
        fh.write("second line\n")
    page2 = client.get_logs(run_id, offset=page["offset"])
    assert page2["logs"] == "second line\n"


def test_unknown_run_maps_to_api_error_404(make_client) -> None:
    client = make_client()
    with pytest.raises(ApiError) as exc:
        client.get_run("run-does-not-exist")
    assert exc.value.status == 404


def test_bad_spec_maps_to_api_error_400(make_client) -> None:
    client = make_client()
    # A spec missing ``model`` must be rejected by the server's validator and the
    # client must surface the server's ``detail`` message, not a bare 500.
    with pytest.raises(ApiError) as exc:
        client.create_run({"algorithm": "grpo"})
    assert exc.value.status == 400
    assert "model" in str(exc.value).lower()


def test_local_env_path_rejected_through_client(make_client) -> None:
    client = make_client()
    bad = {**SPEC, "environment": {"id": "custom", "path": "/home/user/env.py"}}
    with pytest.raises(ApiError) as exc:
        client.create_run(bad)
    assert exc.value.status == 400


def test_tenant_isolation_across_clients(make_client) -> None:
    client_a = make_client()
    client_b = make_client()

    run_id = client_a.create_run(SPEC)["run_id"]

    # Owner sees it; the other tenant cannot read, list, or cancel it.
    assert client_a.get_run(run_id)["run_id"] == run_id
    assert run_id not in [r["run_id"] for r in client_b.list_runs()]
    with pytest.raises(ApiError) as read_exc:
        client_b.get_run(run_id)
    assert read_exc.value.status == 404
    # Cross-tenant cancel must also 404 (ownership check on the cancel path),
    # and the owner's run must be untouched afterwards.
    with pytest.raises(ApiError) as cancel_exc:
        client_b.cancel_run(run_id)
    assert cancel_exc.value.status == 404
    assert client_a.get_run(run_id)["state"] == "queued"


def test_missing_key_is_unauthorized(make_client) -> None:
    anon = make_client(api_key=None, login=False)
    with pytest.raises(ApiError) as exc:
        anon.me()
    assert exc.value.status == 401


def test_deploy_dry_run_and_deployments_roundtrip(make_client) -> None:
    client = make_client()
    run_id = client.create_run(SPEC)["run_id"]

    # A dry-run deploy validates the serving path without provisioning a GPU.
    deployment = client.deploy(run_id, dry_run=True)
    assert deployment["state"] == "dry_run"
    assert "mode" not in deployment

    # Dry-run deploys never count as active deployments.
    assert client.deployments() == []


def test_chat_without_deployment_maps_to_api_error(make_client) -> None:
    client = make_client()
    run_id = client.create_run(SPEC)["run_id"]

    # Chatting a run that was never deployed must be refused with a clear 409,
    # which the client surfaces as an ApiError (not a hang or opaque 500).
    with pytest.raises(ApiError) as exc:
        client.chat(run_id, messages=[{"role": "user", "content": "hi"}])
    assert exc.value.status == 409

"""Completion-time run billing: the billing client POST shape and server submit contract."""

from __future__ import annotations

import importlib
import io
import json
import time
import urllib.error
import urllib.request

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

SPEC = {
    "model": "Qwen/Qwen3.5-4B",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {"steps": 1, "seeds": [0], "hf_repo": "org/test-runs"},
    "gpu": {"type": "RTX 5090"},
}

_USER_PREFIX = "fslo-user-"


def _identity_for_token(token: str) -> dict[str, str]:
    if not token.startswith(_USER_PREFIX):
        return {}
    suffix = token.removeprefix(_USER_PREFIX)
    identity = {"email": f"user-{suffix}@example.com", "key_prefix": "fslo_test"}
    if suffix != "noorg":
        identity["org_id"] = f"org-{suffix}"
    return identity


def _bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


# --------------------------------------------------------------------------- client unit


def _spec(monkeypatch):
    """Parse SPEC into a JobSpec offline.

    SPEC is a catalog model on a concrete GPU pin, so the parse never probes HF anyway; the
    autouse ``_offline`` conftest fixture also stubs ``fetch_hf_params_b`` -- no env switch.
    """
    from flash.schema import spec_from_dict

    return spec_from_dict(SPEC, run_id="run-1")


def test_cents_rounds_and_floors_at_zero():
    from flash.server.billing import _cents

    assert _cents(12.34) == 1234
    assert _cents(0.0) == 0
    assert _cents(-5.0) == 0  # never negative


def test_cents_rounds_half_up_not_bankers():
    """Money rounding is round-HALF-UP, not Python's ties-to-even (which would undercharge a
    half-cent tie, e.g. $0.005 -> 0)."""
    from flash.server.billing import _cents

    assert _cents(0.005) == 1  # ties-to-even would give 0
    assert _cents(0.015) == 2  # ties-to-even would give 2 as well, but via half-up here
    assert _cents(0.025) == 3  # ties-to-even would give 2 -> half-up gives 3
    assert _cents(1.005) == 101  # classic banker's-rounding trap


def _completed_status(monkeypatch, *, cost_usd: float = 12.345, org_id: str = "org-A"):
    from flash.runner import RunStatus

    spec = _spec(monkeypatch)
    return RunStatus(
        run_id=spec.run_id,
        state="done",
        spec=spec.to_dict(),
        cost_usd=cost_usd,
        remote={"provider": "runpod", "allocated_gpu": "RTX 5090"},
        billing_context={"org_id": org_id},
        billing_state="pending",
    )


def test_charge_posts_completed_run_cost_and_parses_response(monkeypatch):
    from flash.server import billing

    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"amountCents": 1234, "balanceCents": 8766}).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data)
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    out = billing.charge_completed_run(
        internal_key="fslo-internal", status=_completed_status(monkeypatch)
    )
    assert out == {"amountCents": 1234, "balanceCents": 8766}
    assert captured["url"].endswith("/api/billing/training-usage/internal")
    assert captured["method"] == "POST"
    assert captured["headers"]["authorization"] == "Bearer fslo-internal"
    body = captured["body"]
    assert body["orgId"] == "org-A"
    assert body["runId"] == "run-1"
    assert body["costCents"] == 1235
    assert body["gpu"] == "RTX 5090"
    assert body["provider"] == "runpod"
    assert body["method"] == "grpo"
    assert body["model"] == "Qwen/Qwen3.5-4B"
    assert body["estimate"] == {
        "totalUsd": 12.345,
        "costBasis": "final",
        "costSource": "run_status.cost_usd",
    }


def test_charge_completed_run_raises_billing_error_on_402(monkeypatch):
    from flash.server import billing

    def fake_urlopen(req, timeout=None):
        body = json.dumps(
            {
                "detail": {
                    "error": "Insufficient balance. Top up in Billing settings.",
                    "code": "insufficient_balance",
                }
            }
        ).encode()
        raise urllib.error.HTTPError(req.full_url, 402, "Payment Required", {}, io.BytesIO(body))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(billing.BillingError) as exc:
        billing.charge_completed_run(internal_key="tok", status=_completed_status(monkeypatch))
    assert exc.value.status_code == 402
    assert "Insufficient balance" in exc.value.detail


def test_charge_completed_run_unreachable_raises_503(monkeypatch):
    from flash.server import billing

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(billing.BillingError) as exc:
        billing.charge_completed_run(internal_key="tok", status=_completed_status(monkeypatch))
    assert exc.value.status_code == 503


def test_charge_completed_run_requires_billing_org(monkeypatch):
    from flash.server import billing

    status = _completed_status(monkeypatch, org_id="")

    with pytest.raises(billing.BillingError) as exc:
        billing.charge_completed_run(internal_key="tok", status=status)
    assert exc.value.status_code == 400
    assert "org id" in exc.value.detail


def test_http_error_detail_falls_back_to_reason():
    """When the error body is missing/unparseable, the detail keeps the backend REASON
    (not a bare ``billing failed (<code>)``) so a 4xx/5xx stays diagnosable."""
    from flash.server.billing import _http_error_detail

    # Unparseable body -> reason phrase is preserved in the detail.
    bad_body = urllib.error.HTTPError(
        "http://x/api/billing/training-usage", 502, "Bad Gateway", {}, io.BytesIO(b"<html>nope")
    )
    detail = _http_error_detail(bad_body)
    assert "502" in detail
    assert "Bad Gateway" in detail

    # Empty JSON object (no `detail`) -> still falls back to code + reason, not just the code.
    empty = urllib.error.HTTPError(
        "http://x/api/billing/training-usage", 402, "Payment Required", {}, io.BytesIO(b"{}")
    )
    assert "Payment Required" in _http_error_detail(empty)


# ------------------------------------------------------------------------- create_run gate


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test,rp-test-2")
    monkeypatch.setenv("LAMBDA_API_KEY", "lam-test")
    monkeypatch.setenv("HYPERSTACK_API_KEY", "hyp-test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    import flash.runner as runner
    import flash.server.auth as auth_mod
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))
    # Keep submit offline: validate + record, but the GPU job body is a no-op.
    monkeypatch.setattr(runner, "_run_job", lambda *a, **k: None)

    import flash.server.app as app_mod

    importlib.reload(app_mod)
    # The Lambda/Hyperstack keys above (required by the startup preflight) make configured_providers()
    # treat both as live, so startup recover_runs() would dispatch real sweep_orphans() list calls.
    # And the dummy FREESOLO_INTERNAL_KEY enables the best-effort backend reporting path: every
    # /v1/runs submit -> _report_status() -> run_registry._post() would urllib-POST the real backend
    # (or wait out its 10s timeout). These billing tests assert on the API response, not on reporting,
    # so stub both to keep startup + submit hermetic (CPU-only, no network).
    import flash.providers as providers_mod
    import flash.server.run_registry as run_registry

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [], raising=False)
    monkeypatch.setattr(run_registry, "_post", lambda *a, **k: False, raising=False)
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token.startswith(_USER_PREFIX))
    monkeypatch.setattr(auth_mod, "_cached_identity", _identity_for_token)
    with TestClient(app_mod.create_app()) as client:
        yield client


def test_submit_records_pending_completion_billing(api):
    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-1"))
    assert res.status_code == 200, res.text
    body = res.json()
    run_id = body["run_id"]
    assert body["billing_state"] == "pending"
    assert body["billing_context"] == {"org_id": "org-1"}
    # The run was accepted + recorded.
    listed = api.get("/v1/runs", headers=_bearer("fslo-user-1")).json()["runs"]
    assert [r["run_id"] for r in listed] == [run_id]


def test_external_submit_requires_org_for_completion_billing(api):
    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-noorg"))
    assert res.status_code == 400, res.text
    assert "org id" in res.json()["detail"]
    assert api.get("/v1/runs", headers=_bearer("fslo-user-noorg")).json()["runs"] == []


def test_dry_run_skips_billing(api):
    res = api.post("/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer("fslo-user-3"))
    assert res.status_code == 200, res.text
    assert res.json()["state"] == "dry_run"
    assert res.json()["billing_state"] is None
    assert res.json()["billing_context"] is None


def test_internal_identity_skips_billing(api, monkeypatch):
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-secret")

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-internal-secret"))
    assert res.status_code == 200, res.text
    assert res.json()["billing_state"] is None
    assert res.json()["billing_context"] is None


def test_external_identity_with_internal_prefix_is_still_billed(api, monkeypatch):
    import flash.server.auth as auth_mod

    token = "fslo-user-spoof"
    auth_mod._identity_cache[token] = (
        {"key_prefix": "internal", "email": "user@example.com", "org_id": "org-spoof"},
        time.time() + auth_mod._VERIFY_CACHE_TTL_S,
    )

    me = api.get("/v1/me", headers=_bearer(token))
    assert me.status_code == 200
    assert me.json()["kind"] == "freesolo_api_key"
    assert me.json()["key_prefix"].startswith("fslo_")
    assert me.json()["key_prefix"] != "internal"

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer(token))
    assert res.status_code == 200, res.text
    assert res.json()["billing_state"] == "pending"
    assert res.json()["billing_context"] == {"org_id": "org-spoof"}


def test_submit_failure_records_nothing(api, monkeypatch):
    """If submit_job fails, no run row is left behind and no billing reversal is needed."""
    import flash.server.app as app_mod

    def failing_submit(spec, dry_run=False, background=True, **kwargs):
        raise RuntimeError("provider out of capacity")

    monkeypatch.setattr(app_mod, "submit_job", failing_submit)

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-r"))
    assert res.status_code == 400, res.text
    assert "out of capacity" in res.json()["detail"]
    assert api.get("/v1/runs", headers=_bearer("fslo-user-r")).json()["runs"] == []


def test_record_run_failure_does_not_submit(api, monkeypatch):
    """If ``db.record_run`` fails (e.g. SQLite locked/full), submit_job is not reached."""
    import flash.server.app as app_mod
    import flash.server.db as db_mod

    def failing_record(run_id, key_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db_mod, "record_run", failing_record)
    # submit_job must NOT be reached if record_run already failed.
    monkeypatch.setattr(
        app_mod,
        "submit_job",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("submit must not run")),
    )

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-rec"))
    assert res.status_code == 400, res.text
    assert "database is locked" in res.json()["detail"]


def test_completion_hook_charges_final_cost(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.runner import RunStatus, lifecycle

    spec = _spec(monkeypatch)
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    runner._save_status(
        RunStatus(
            run_id=spec.run_id,
            state="done",
            spec=spec.to_dict(),
            cost_usd=1.23,
            billing_context={"org_id": "org-A"},
            billing_state="pending",
        )
    )

    calls = []

    def fake_charge(*, internal_key, status):
        calls.append((internal_key, status.run_id, status.cost_usd))
        return {"amountCents": 123, "balanceCents": 877, "replay": False}

    monkeypatch.setattr("flash.server.billing.charge_completed_run", fake_charge)
    log = io.StringIO()

    lifecycle._charge_completed_run_best_effort(spec, log)

    assert calls == [("fslo-internal", "run-1", 1.23)]
    status = runner.get_status("run-1")
    assert status.billing_state == "charged"
    assert status.billing_error is None
    assert status.billing_charge == {"amountCents": 123, "balanceCents": 877, "replay": False}
    assert "billing charged" in log.getvalue()


def test_completion_hook_records_missing_internal_key(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.runner import RunStatus, lifecycle

    spec = _spec(monkeypatch)
    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    runner._save_status(
        RunStatus(
            run_id=spec.run_id,
            state="done",
            spec=spec.to_dict(),
            cost_usd=1.23,
            billing_context={"org_id": "org-A"},
            billing_state="pending",
        )
    )
    monkeypatch.setattr(
        "flash.server.billing.charge_completed_run",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not charge without internal key")),
    )

    lifecycle._charge_completed_run_best_effort(spec, io.StringIO())

    status = runner.get_status("run-1")
    assert status.billing_state == "failed"
    assert "FREESOLO_INTERNAL_KEY" in (status.billing_error or "")

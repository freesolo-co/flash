"""Estimate-based run billing: the control plane charges the pre-flight estimate at submit.

CPU-only / offline. Two layers are covered:
  * the billing client ``flash.server.billing.charge_run_estimate`` (FLASH_SKIP_NET no-op,
    the POST shape, and error translation), and
  * the ``POST /v1/runs`` gate (charge fires for a normal user submit, a 402 blocks the run
    and records nothing, and --dry-run / the internal service identity skip billing).
"""

from __future__ import annotations

import importlib
import io
import json
import urllib.error
import urllib.request

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

SPEC = {
    "model": "Qwen/Qwen3.5-4B",
    "algorithm": "grpo",
    "environment": {"id": "primeintellect/gsm8k"},
    "train": {"steps": 1, "seeds": [0], "hf_repo": "org/test-runs"},
    "gpu": {"type": "RTX 5090"},
}

_USER_PREFIX = "fslo-user-"


def _bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


# --------------------------------------------------------------------------- client unit


def _spec(monkeypatch):
    """Parse SPEC into a JobSpec offline (FLASH_SKIP_NET avoids any HF probe)."""
    monkeypatch.setenv("FLASH_SKIP_NET", "1")
    from flash.schema import spec_from_dict

    return spec_from_dict(SPEC, run_id="run-1")


def test_cents_rounds_and_floors_at_zero():
    from flash.server.billing import _cents

    assert _cents(12.34) == 1234
    assert _cents(0.0) == 0
    assert _cents(-5.0) == 0  # never negative


def test_charge_is_noop_when_offline(monkeypatch):
    from flash.server import billing

    spec = _spec(monkeypatch)  # leaves FLASH_SKIP_NET set

    def explode(*a, **k):
        raise AssertionError("no network call may happen under FLASH_SKIP_NET")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    assert billing.charge_run_estimate(token="tok", spec=spec) == {}


def test_charge_posts_estimate_and_parses_response(monkeypatch):
    from flash.server import billing

    spec = _spec(monkeypatch)
    monkeypatch.delenv("FLASH_SKIP_NET", raising=False)

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

    out = billing.charge_run_estimate(token="fslo-user-7", spec=spec)
    assert out == {"amountCents": 1234, "balanceCents": 8766}
    assert captured["url"].endswith("/api/billing/training-usage")
    assert captured["method"] == "POST"
    assert captured["headers"]["authorization"] == "Bearer fslo-user-7"
    body = captured["body"]
    assert body["runId"] == "run-1"
    assert isinstance(body["costCents"], int)
    assert body["costCents"] >= 0
    assert body["method"] == "grpo"
    assert body["model"] == "Qwen/Qwen3.5-4B"


def test_charge_raises_billing_error_on_402(monkeypatch):
    from flash.server import billing

    spec = _spec(monkeypatch)
    monkeypatch.delenv("FLASH_SKIP_NET", raising=False)

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
        billing.charge_run_estimate(token="tok", spec=spec)
    assert exc.value.status_code == 402
    assert "Insufficient balance" in exc.value.detail


def test_charge_unreachable_raises_503(monkeypatch):
    from flash.server import billing

    spec = _spec(monkeypatch)
    monkeypatch.delenv("FLASH_SKIP_NET", raising=False)

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(billing.BillingError) as exc:
        billing.charge_run_estimate(token="tok", spec=spec)
    assert exc.value.status_code == 503


# ------------------------------------------------------------------------- create_run gate


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test")
    monkeypatch.setenv("PRIME_API_KEY", "pit-test")
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
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token.startswith(_USER_PREFIX))
    with TestClient(app_mod.create_app()) as client:
        yield client


def test_submit_charges_the_estimate(api, monkeypatch):
    calls = []

    def fake_charge(*, token, spec):
        calls.append((token, spec.run_id))
        return {"amountCents": 999}

    monkeypatch.setattr("flash.server.billing.charge_run_estimate", fake_charge)

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-1"))
    assert res.status_code == 200, res.text
    run_id = res.json()["run_id"]
    assert len(calls) == 1
    assert calls[0] == ("fslo-user-1", run_id)
    # The run was accepted + recorded.
    listed = api.get("/v1/runs", headers=_bearer("fslo-user-1")).json()["runs"]
    assert [r["run_id"] for r in listed] == [run_id]


def test_insufficient_balance_blocks_and_records_nothing(api, monkeypatch):
    from flash.server.billing import BillingError

    def boom(*, token, spec):
        raise BillingError(402, "Insufficient balance. Top up in Billing settings.")

    monkeypatch.setattr("flash.server.billing.charge_run_estimate", boom)

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-2"))
    assert res.status_code == 402, res.text
    assert "Insufficient balance" in res.json()["detail"]
    # A blocked submit must not leave a run row behind.
    assert api.get("/v1/runs", headers=_bearer("fslo-user-2")).json()["runs"] == []


def test_dry_run_skips_billing(api, monkeypatch):
    def boom(*, token, spec):
        raise AssertionError("dry-run must not be billed")

    monkeypatch.setattr("flash.server.billing.charge_run_estimate", boom)

    res = api.post("/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer("fslo-user-3"))
    assert res.status_code == 200, res.text
    assert res.json()["state"] == "dry_run"


def test_internal_identity_skips_billing(api, monkeypatch):
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-secret")

    def boom(*, token, spec):
        raise AssertionError("the internal service identity has no org to bill")

    monkeypatch.setattr("flash.server.billing.charge_run_estimate", boom)

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-internal-secret"))
    assert res.status_code == 200, res.text

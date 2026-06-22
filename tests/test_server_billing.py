"""Estimate-based run billing: the billing client (POST shape + error translation, network
stubbed) and the ``POST /v1/runs`` charge gate (charge fires, 402 blocks, dry-run/internal skip)."""

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
    "environment": {"id": "primeintellect/gsm8k"},
    "train": {"steps": 1, "seeds": [0], "hf_repo": "org/test-runs"},
    "gpu": {"type": "RTX 5090"},
}

_USER_PREFIX = "fslo-user-"


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


def test_charge_posts_estimate_and_parses_response(monkeypatch):
    from flash.server import billing

    spec = _spec(monkeypatch)

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

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(billing.BillingError) as exc:
        billing.charge_run_estimate(token="tok", spec=spec)
    assert exc.value.status_code == 503


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


def test_reverse_run_charge_posts_reversal(monkeypatch):
    """The refund POSTs a reversal for the run, forwarding the original charge's amount for the
    backend to match the exact debit."""
    from flash.server import billing

    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"reversed": True, "amountCents": 50}).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = billing.reverse_run_charge(token="tok", run_id="r1", charge={"amountCents": 50})
    assert out == {"reversed": True, "amountCents": 50}
    assert captured["url"].endswith("/api/billing/training-usage/reverse")
    assert captured["body"] == {"runId": "r1", "reverse": True, "costCents": 50}


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


def test_external_identity_with_internal_prefix_is_still_billed(api, monkeypatch):
    import flash.server.auth as auth_mod

    token = "fslo-user-spoof"
    auth_mod._identity_cache[token] = (
        {"key_prefix": "internal", "email": "user@example.com"},
        time.time() + auth_mod._VERIFY_CACHE_TTL_S,
    )
    calls = []

    def fake_charge(*, token, spec):
        calls.append((token, spec.run_id))
        return {"amountCents": 123}

    monkeypatch.setattr("flash.server.billing.charge_run_estimate", fake_charge)

    me = api.get("/v1/me", headers=_bearer(token))
    assert me.status_code == 200
    assert me.json()["kind"] == "freesolo_api_key"
    assert me.json()["key_prefix"].startswith("fslo_")
    assert me.json()["key_prefix"] != "internal"

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer(token))
    assert res.status_code == 200, res.text
    assert len(calls) == 1
    assert calls[0] == (token, res.json()["run_id"])


def test_submit_failure_after_charge_reverses_the_debit(api, monkeypatch):
    """If the charge succeeds but ``submit_job`` then fails, the debit is reversed (with run id +
    original charge) and no run row is left behind."""
    import flash.server.app as app_mod

    charged = {"amountCents": 777}
    monkeypatch.setattr(
        "flash.server.billing.charge_run_estimate", lambda *, token, spec: charged
    )

    def failing_submit(spec, dry_run=False, background=True):
        raise RuntimeError("provider out of capacity")

    monkeypatch.setattr(app_mod, "submit_job", failing_submit)

    reversed_calls = []
    monkeypatch.setattr(
        "flash.server.billing.reverse_run_charge",
        lambda *, token, run_id, charge: reversed_calls.append((token, run_id, charge)) or {},
    )

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-r"))
    assert res.status_code == 400, res.text
    assert "out of capacity" in res.json()["detail"]
    # The charge was reversed for exactly this run, forwarding the original charge.
    assert len(reversed_calls) == 1
    token, run_id, charge = reversed_calls[0]
    assert token == "fslo-user-r"
    assert charge == charged
    assert isinstance(run_id, str)
    assert run_id
    # A failed submit leaves no run row behind (so a retry re-charges cleanly).
    assert api.get("/v1/runs", headers=_bearer("fslo-user-r")).json()["runs"] == []


def test_record_run_failure_after_charge_reverses_the_debit(api, monkeypatch):
    """If the charge succeeds but ``db.record_run`` then fails (e.g. SQLite locked/full), the
    org is debited for a run that never started — the reversal must still fire (record_run is
    inside the same try as submit).."""
    import flash.server.app as app_mod
    import flash.server.db as db_mod

    monkeypatch.setattr(
        "flash.server.billing.charge_run_estimate", lambda *, token, spec: {"amountCents": 42}
    )

    def failing_record(run_id, key_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db_mod, "record_run", failing_record)
    # submit_job must NOT be reached if record_run already failed.
    monkeypatch.setattr(
        app_mod,
        "submit_job",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("submit must not run")),
    )

    reversed_calls = []
    monkeypatch.setattr(
        "flash.server.billing.reverse_run_charge",
        lambda *, token, run_id, charge: reversed_calls.append((run_id, charge)) or {},
    )

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-rec"))
    assert res.status_code == 400, res.text
    assert "database is locked" in res.json()["detail"]
    assert len(reversed_calls) == 1
    assert reversed_calls[0][1] == {"amountCents": 42}


def test_refund_failure_does_not_mask_submit_error(api, monkeypatch):
    """A best-effort refund: if the reversal ITSELF fails, the original submit error is still
    surfaced (the refund failure is swallowed + logged, never raised to the client)."""
    import flash.server.app as app_mod

    monkeypatch.setattr(
        "flash.server.billing.charge_run_estimate", lambda *, token, spec: {"amountCents": 5}
    )

    def failing_submit(spec, dry_run=False, background=True):
        raise RuntimeError("provider out of capacity")

    monkeypatch.setattr(app_mod, "submit_job", failing_submit)

    def failing_reverse(*, token, run_id, charge):
        raise RuntimeError("billing down during refund")

    monkeypatch.setattr("flash.server.billing.reverse_run_charge", failing_reverse)

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-rr"))
    # The submit error wins; the refund failure does not turn into a 500.
    assert res.status_code == 400, res.text
    assert "out of capacity" in res.json()["detail"]
    assert api.get("/v1/runs", headers=_bearer("fslo-user-rr")).json()["runs"] == []


def test_submit_failure_for_internal_identity_does_not_refund(api, monkeypatch):
    """The internal service identity is never charged, so a submit failure must NOT attempt a
    (meaningless) reversal."""
    import flash.server.app as app_mod

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-secret")

    def failing_submit(spec, dry_run=False, background=True):
        raise RuntimeError("provider out of capacity")

    monkeypatch.setattr(app_mod, "submit_job", failing_submit)
    monkeypatch.setattr(
        "flash.server.billing.reverse_run_charge",
        lambda *, token, run_id, charge: (_ for _ in ()).throw(
            AssertionError("internal identity was never charged; must not refund")
        ),
    )

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-internal-secret"))
    assert res.status_code == 400, res.text

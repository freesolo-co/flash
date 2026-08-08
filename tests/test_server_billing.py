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
    "project": "11111111-1111-4111-8111-111111111111",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {"epochs": 1, "max_examples": 1},
    "gpu": {},
}

_USER_PREFIX = "fslo-user-"


def _identity_for_token(token: str) -> dict[str, str]:
    if not token.startswith(_USER_PREFIX):
        return {}
    suffix = token.removeprefix(_USER_PREFIX)
    identity = {
        "email": f"user-{suffix}@example.com",
        "key_prefix": "fslo_test",
        "org_slug": f"org-{suffix}",
    }
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
        "costBasis": "estimate",
        "costSource": "run_status.cost_usd",
    }


def test_charge_bills_cost_usd_for_a_cancelled_run(monkeypatch):
    """A cancelled run is charged its cost_usd exactly like a completed one -- the cancel path already
    set cost_usd to the estimate at the steps actually run, so charge_completed_run just bills it."""
    from flash.runner import RunStatus
    from flash.server import billing

    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"amountCents": 250, "balanceCents": 9750, "replay": False}).encode()

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data)
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    spec = _spec(monkeypatch)
    status = RunStatus(
        run_id=spec.run_id,
        state="cancelled",
        spec=spec.to_dict(),
        cost_usd=2.50,  # cancel path priced this at the actual steps run
        remote={"provider": "runpod", "allocated_gpu": "RTX 5090"},
        billing_context={"org_id": "org-A"},
        billing_state="pending",
    )

    out = billing.charge_completed_run(internal_key="fslo-internal", status=status)
    assert out == {"amountCents": 250, "balanceCents": 9750, "replay": False}
    body = captured["body"]
    assert body["runId"] == "run-1"
    assert body["costCents"] == 250  # the cancel estimate on cost_usd
    assert body["estimate"]["costBasis"] == "estimate"


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
    # Keep submit offline: validate + record, but the GPU job body is a no-op.
    monkeypatch.setattr(runner, "_run_job", lambda *a, **k: None)

    import flash.server.app as app_mod

    importlib.reload(app_mod)
    # configured provider keys would trigger orphan sweeps and status reporting. stub both because
    # these billing tests assert only on the API response and must remain network-free.
    import flash.providers as providers_mod
    import flash.server.projects as projects_mod
    import flash.server.run_registry as run_registry

    monkeypatch.setattr(providers_mod, "configured_providers", list, raising=False)
    monkeypatch.setattr(
        projects_mod,
        "require_project_access",
        lambda *, project_id, **_kwargs: project_id,
    )
    monkeypatch.setattr(run_registry, "_post", lambda *a, **k: False, raising=False)
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token.startswith(_USER_PREFIX))
    monkeypatch.setattr(auth_mod, "_cached_identity", _identity_for_token)
    # The new submit-time budget precheck would urllib-POST the real backend; stub it to a pass so
    # the default submit path stays hermetic. Gate-specific tests below override this per-test.
    import flash.server.billing as billing_mod

    monkeypatch.setattr(billing_mod, "precheck_training_run", lambda **k: {"ok": True})
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


def test_budget_precheck_uses_prepared_estimate_before_record_run(api, monkeypatch):
    import flash.server.billing as billing_mod
    import flash.server.db as db_mod

    events = []
    original_record_run = db_mod.record_run

    def capture_precheck(**kwargs):
        events.append(("precheck", kwargs["estimate_usd"]))
        return {"ok": True}

    def capture_record_run(*args, **kwargs):
        events.append(("record", args[0]))
        return original_record_run(*args, **kwargs)

    monkeypatch.setattr(billing_mod, "precheck_training_run", capture_precheck)
    monkeypatch.setattr(db_mod, "record_run", capture_record_run)

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-1"))

    assert res.status_code == 200, res.text
    assert events[0] == ("precheck", res.json()["estimated_cost_usd"])
    assert events[1] == ("record", res.json()["run_id"])


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


def test_submit_blocked_when_precheck_402(api, monkeypatch):
    # a hard 402 from the budget precheck rejects the run up front, before any GPU is allocated,
    # and the run is never recorded.
    import flash.server.billing as billing_mod

    def _block(**k):
        raise billing_mod.BillingError(402, "insufficient balance")

    monkeypatch.setattr(billing_mod, "precheck_training_run", _block)
    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-1"))
    assert res.status_code == 402, res.text
    assert "insufficient" in res.text
    # rejected before record_run: nothing persisted for this user.
    assert api.get("/v1/runs", headers=_bearer("fslo-user-1")).json()["runs"] == []


def test_submit_fails_open_when_precheck_unreachable(api, monkeypatch):
    # a non-402 billing error (backend unreachable / 5xx) must NOT block training; the completion
    # charge is the backstop. The run is still accepted and recorded.
    import flash.server.billing as billing_mod

    def _unreachable(**k):
        raise billing_mod.BillingError(503, "billing service unavailable")

    monkeypatch.setattr(billing_mod, "precheck_training_run", _unreachable)
    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-1"))
    assert res.status_code == 200, res.text
    assert [
        r["run_id"] for r in api.get("/v1/runs", headers=_bearer("fslo-user-1")).json()["runs"]
    ] == [res.json()["run_id"]]


def test_dry_run_verifies_affordability_and_still_persists(api, monkeypatch):
    # a dry run is the pre-submit validation gate, so it must also answer "can this org afford
    # this run". the precheck is verify-only (moves no money), so running it here costs nothing
    # and stops a config from passing --dry-run only to be rejected 402 on real submission.
    import flash.server.billing as billing_mod
    import flash.server.db as db_mod

    events = []
    original_record_run = db_mod.record_run

    def capture_precheck(**kwargs):
        events.append(("precheck", kwargs["org_id"]))
        return {"ok": True}

    def capture_record_run(*args, **kwargs):
        events.append(("record", args[0]))
        return original_record_run(*args, **kwargs)

    monkeypatch.setattr(billing_mod, "precheck_training_run", capture_precheck)
    monkeypatch.setattr(db_mod, "record_run", capture_record_run)
    res = api.post(
        "/v1/runs",
        json={"spec": SPEC, "dry_run": True},
        headers=_bearer("fslo-user-1"),
    )

    assert res.status_code == 200, res.text
    assert res.json()["state"] == "dry_run"
    # verified before the run is recorded, and it stays a dry run: no billing context is attached
    assert events == [("precheck", "org-1"), ("record", res.json()["run_id"])]
    assert res.json()["billing_state"] is None
    assert res.json()["billing_context"] is None


def test_dry_run_blocked_when_org_cannot_afford_the_estimate(api, monkeypatch):
    # the whole point of validating billing on --dry-run: an unaffordable config fails here
    # instead of passing validation and being rejected only when the user really submits.
    import flash.server.billing as billing_mod

    def _block(**k):
        raise billing_mod.BillingError(402, "insufficient balance")

    monkeypatch.setattr(billing_mod, "precheck_training_run", _block)
    res = api.post(
        "/v1/runs",
        json={"spec": SPEC, "dry_run": True},
        headers=_bearer("fslo-user-1"),
    )

    assert res.status_code == 402, res.text
    assert "insufficient" in res.text


def test_dry_run_fails_open_when_billing_backend_is_unreachable(api, monkeypatch):
    # a billing-infra blip must not block local validation, matching real submission's behavior.
    import flash.server.billing as billing_mod

    def _unreachable(**k):
        raise billing_mod.BillingError(503, "billing service unavailable")

    monkeypatch.setattr(billing_mod, "precheck_training_run", _unreachable)
    res = api.post(
        "/v1/runs",
        json={"spec": SPEC, "dry_run": True},
        headers=_bearer("fslo-user-1"),
    )

    assert res.status_code == 200, res.text
    assert res.json()["state"] == "dry_run"
    # failing open is intentional, but the response must not imply cost was validated: the same
    # spec can still be rejected 402 once the backend recovers.
    assert res.json()["affordability_verified"] is False


def test_dry_run_reports_affordability_verified_when_the_check_ran(api, monkeypatch):
    """A real pass and a failed-open skip must be distinguishable, since both answer 200."""
    import flash.server.billing as billing_mod

    monkeypatch.setattr(billing_mod, "precheck_training_run", lambda **k: {"ok": True})
    res = api.post(
        "/v1/runs",
        json={"spec": SPEC, "dry_run": True},
        headers=_bearer("fslo-user-1"),
    )

    assert res.status_code == 200, res.text
    assert res.json()["affordability_verified"] is True


def test_dry_run_reports_unverified_when_internal_reporting_is_off(api, monkeypatch):
    """The other fail-open path: no internal key, so the precheck returns before calling billing."""
    import flash.server._internal_client as internal_mod

    monkeypatch.setattr(internal_mod, "internal_key", lambda: None)
    res = api.post(
        "/v1/runs",
        json={"spec": SPEC, "dry_run": True},
        headers=_bearer("fslo-user-1"),
    )

    assert res.status_code == 200, res.text
    assert res.json()["affordability_verified"] is False


def test_billable_dry_run_without_an_org_is_rejected_like_a_real_submit(api, monkeypatch):
    # the org requirement belongs to the key, not the mode. when it was checked only for real
    # submits, an org-less billable key skipped the affordability conjunct and got 200 from
    # --dry-run, then 400 for the identical spec on submit: the preview contradicting the launch.
    import flash.server.billing as billing_mod

    def _unexpected(**kwargs):
        raise AssertionError("affordability cannot be checked without an org")

    monkeypatch.setattr(billing_mod, "precheck_training_run", _unexpected)
    res = api.post(
        "/v1/runs",
        json={"spec": SPEC, "dry_run": True},
        headers=_bearer("fslo-user-noorg"),
    )

    assert res.status_code == 400, res.text
    assert "org id" in res.text


def test_unsupported_spec_reports_itself_rather_than_insufficient_balance(api, monkeypatch):
    # static launch validation must precede billing. otherwise an unsupported spec reports 402 and
    # sends the user to top up for a failure money cannot fix.
    import flash.server.billing as billing_mod

    def _block(**k):
        raise billing_mod.BillingError(402, "insufficient balance")

    monkeypatch.setattr(billing_mod, "precheck_training_run", _block)
    unsupported_spec = {
        **SPEC,
        # image-bearing OPD is single-turn only, so a multi-turn environment carrying an image
        # record can never launch regardless of the org's balance.
        "algorithm": "opd",
        "environment": {
            **SPEC["environment"],
            "params": {
                # multi_turn rides in params, not as an [environment] key: the spec schema rejects
                # unknown top-level environment keys, and the validator reads either.
                "multi_turn": True,
                "records": [
                    {
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
                                ],
                            }
                        ]
                    }
                ],
            },
        },
    }
    res = api.post(
        "/v1/runs",
        json={"spec": unsupported_spec, "dry_run": True},
        headers=_bearer("fslo-user-1"),
    )

    assert res.status_code == 400, res.text
    assert "image-bearing opd is not supported" in res.text
    assert "insufficient" not in res.text


def test_internal_submit_skips_affordability_precheck_before_persistence(api, monkeypatch):
    import flash.server.billing as billing_mod
    import flash.server.db as db_mod

    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-secret")
    events = []
    original_record_run = db_mod.record_run

    def unexpected_precheck(**kwargs):
        events.append(("precheck", kwargs["org_id"]))
        raise AssertionError("internal submit must not authorize budget")

    def capture_record_run(*args, **kwargs):
        events.append(("record", args[0]))
        return original_record_run(*args, **kwargs)

    monkeypatch.setattr(billing_mod, "precheck_training_run", unexpected_precheck)
    monkeypatch.setattr(db_mod, "record_run", capture_record_run)
    res = api.post(
        "/v1/runs",
        json={"spec": SPEC},
        headers=_bearer("fslo-internal-secret"),
    )

    assert res.status_code == 200, res.text
    assert res.json()["billing_context"] is None
    assert events == [("record", res.json()["run_id"])]


def test_external_identity_with_internal_prefix_is_still_billed(api, monkeypatch):
    import flash.server.auth as auth_mod

    token = "fslo-user-spoof"
    auth_mod._verify_cache[token] = (
        True,
        {
            "key_prefix": "internal",
            "email": "user@example.com",
            "org_id": "org-spoof",
            "org_slug": "org-spoof",
        },
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


def _pending_profile(monkeypatch, *, profile_run_id: str, estimate_usd: float = 0.25):
    """Make every submission miss its profile, so the route takes the pending branch.

    Returns the list every ``submit_job`` call is appended to, so a test can prove what was
    launched and what was not.
    """
    from dataclasses import replace as _replace

    import flash.runner as runner
    import flash.server.app as app_mod

    submitted: list = []

    def prepare(spec, **_kwargs):
        profile_spec = _replace(
            spec,
            run_id=profile_run_id,
            workload_profile_kind="sft",
            workload_profile_input_digest="c" * 64,
            workload_profile_producer_version="1.2.3",
        )
        raise runner.WorkloadProfilePending(
            profile_run_id,
            "required",
            prepared_job=runner.PreparedJob(
                public_spec=profile_spec,
                worker_spec=profile_spec,
                estimated_cost_usd=estimate_usd,
            ),
        )

    monkeypatch.setattr(app_mod, "prepare_job", prepare)
    monkeypatch.setattr(app_mod, "submit_job", lambda spec, **kw: submitted.append((spec, kw)))
    return submitted


_SFT_SPEC = {**SPEC, "algorithm": "sft", "train": {"epochs": 1, "max_examples": 8}}


def test_a_profile_is_prechecked_and_persisted_as_its_own_charge(api, monkeypatch):
    """The profile is a separate job, so it gets its own affordability check at its own estimate.

    Charging it against the training estimate would gate a $0.25 cpu job on whether the org can
    afford the gpu run it has not been quoted for yet -- and would bill the two as one.
    """
    import flash.server.billing as billing_mod
    from flash.server import db

    profile_run_id = "profile-sft-" + "c" * 64
    submitted = _pending_profile(monkeypatch, profile_run_id=profile_run_id)
    prechecked = []
    monkeypatch.setattr(
        billing_mod,
        "precheck_training_run",
        lambda **kw: prechecked.append(kw["estimate_usd"]) or {"ok": True},
    )

    res = api.post("/v1/runs", json={"spec": _SFT_SPEC}, headers=_bearer("fslo-user-1"))

    assert res.status_code == 409, res.text
    # exactly one precheck, at the profile's own quote -- not the training run's.
    assert prechecked == [0.25]
    assert [r["run_id"] for r in db.all_runs()] == [profile_run_id]
    assert [r["kind"] for r in db.all_runs()] == ["profile"]
    assert len(submitted) == 1


def test_an_unaffordable_profile_leaves_no_run_and_no_launch(api, monkeypatch):
    """402 on the profile must not leave the deterministic id claimed by a run that never started.

    The id is global to the workload, so a stranded row is not one user's problem: every later
    submitter of this config would be told to wait for a profile nobody is running.
    """
    import flash.server.billing as billing_mod
    from flash.server import db

    profile_run_id = "profile-sft-" + "c" * 64
    submitted = _pending_profile(monkeypatch, profile_run_id=profile_run_id)

    def _block(**_kw):
        raise billing_mod.BillingError(402, "insufficient balance")

    monkeypatch.setattr(billing_mod, "precheck_training_run", _block)

    res = api.post("/v1/runs", json={"spec": _SFT_SPEC}, headers=_bearer("fslo-user-1"))

    assert res.status_code == 402, res.text
    assert db.all_runs() == []
    assert submitted == []


def test_a_profile_that_cannot_be_launched_releases_its_claim(api, monkeypatch):
    """Same invariant on the other failure: the claim is released when the launch does not happen."""
    import flash.server.app as app_mod
    from flash.server import db

    profile_run_id = "profile-sft-" + "c" * 64
    _pending_profile(monkeypatch, profile_run_id=profile_run_id)
    monkeypatch.setattr(
        app_mod,
        "submit_job",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("provider out of capacity")),
    )

    res = api.post("/v1/runs", json={"spec": _SFT_SPEC}, headers=_bearer("fslo-user-1"))

    assert res.status_code == 400, res.text
    assert db.all_runs() == []


def test_a_profile_that_failed_after_persisting_its_run_keeps_its_claim(api, monkeypatch, tmp_path):
    """Keep the claim when launch already persisted its queued run.

    Deleting it makes the owner read 404 while later submitters wait on the live deterministic id,
    which has no terminal-state takeover path.
    """
    import os

    import flash.server.app as app_mod
    from flash.runner import runs_file_path
    from flash.server import db

    profile_run_id = "profile-sft-" + "c" * 64
    _pending_profile(monkeypatch, profile_run_id=profile_run_id)

    def submit_then_fail(*_a, **_k):
        # stand in for submit_job's real ordering: status on disk, then a raise from _report_status
        # or the thread start below it.
        path = runs_file_path(profile_run_id, ".json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"run_id": profile_run_id, "state": "queued"}, f)
        raise RuntimeError("status reporter unreachable")

    monkeypatch.setattr(app_mod, "submit_job", submit_then_fail)

    res = api.post("/v1/runs", json={"spec": _SFT_SPEC}, headers=_bearer("fslo-user-1"))

    assert res.status_code == 400, res.text
    assert [r["run_id"] for r in db.all_runs()] == [profile_run_id], (
        "the claim was deleted for a run that exists, so its owner reads 404 while later "
        "submitters wait on it forever"
    )


def test_a_profile_miss_creates_no_training_run_and_no_training_charge(api, monkeypatch):
    """The user asked to train and was not charged for training: no run, no billing context.

    This is the invariant that makes a separate profile charge defensible. If a miss also recorded
    a training run, the profile would be a surcharge on top of the run rather than its own job.
    """
    from flash.server import db

    profile_run_id = "profile-sft-" + "c" * 64
    _pending_profile(monkeypatch, profile_run_id=profile_run_id)

    res = api.post("/v1/runs", json={"spec": _SFT_SPEC}, headers=_bearer("fslo-user-1"))

    assert res.status_code == 409, res.text
    assert res.json()["detail"]["code"] == "workload_profile_pending"
    # every persisted row, not just this user's: the one record is the profile, under the kind that
    # separates it. a training row here would mean the profile became a surcharge on a run the
    # user was never quoted. the run LISTING is not the instrument -- it reads the status store,
    # which the stubbed submit never writes, so it would read empty either way.
    assert [(r["run_id"], r["kind"]) for r in db.all_runs()] == [(profile_run_id, "profile")]


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

    lifecycle._charge_completed_run_by_id(spec.run_id, log)

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

    lifecycle._charge_completed_run_by_id(spec.run_id, io.StringIO())

    status = runner.get_status("run-1")
    assert status.billing_state == "failed"
    assert "FREESOLO_INTERNAL_KEY" in (status.billing_error or "")


# --------------------------------------------------------- warm-start error attribution


def test_route_blames_the_adapter_only_for_tagged_failures(api, monkeypatch):
    # a genuine adapter-resolution failure keeps the actionable 400 that names the source.
    import flash.runner as runner_mod
    import flash.server.app as app_mod

    # raise the class off the reloaded module: the api fixture reloads flash.runner, so a symbol
    # imported at module scope here would be a different object than the route's.
    def _prepare(*args, **kwargs):
        raise runner_mod.WarmStartPreparationError("private-owner/private-repo:sft/step-20")

    monkeypatch.setattr(app_mod, "prepare_job", _prepare)
    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-1"))

    assert res.status_code == 400, res.text
    assert "train.init_from_adapter source" in res.text
    # the reason stays server-side: it can name internal storage paths. see
    # test_server_api.py::test_create_run_redacts_internal_warmstart_preparation_error.
    assert "private-owner" not in res.text


def test_route_does_not_blame_the_adapter_for_unrelated_failures(api, monkeypatch):
    # prepare_job also performs gpu, budget, and environment checks. with init_from_adapter set,
    # unrelated failures must retain their own attribution.
    import flash.server.app as app_mod

    def _prepare(*args, **kwargs):
        raise ValueError("no gpu class satisfies the requested memory")

    monkeypatch.setattr(app_mod, "prepare_job", _prepare)
    warm_start_spec = {**SPEC, "train": {**SPEC["train"], "init_from_adapter": "source-run"}}
    res = api.post("/v1/runs", json={"spec": warm_start_spec}, headers=_bearer("fslo-user-1"))

    assert "no gpu class satisfies" in res.text
    assert "init_from_adapter" not in res.text

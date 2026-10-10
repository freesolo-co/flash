"""Completion-time run billing: the billing client POST shape and server submit contract."""

from __future__ import annotations

import importlib
import io
import json
import time
import urllib.error
import urllib.request

import pytest

import flash.providers._lifecycle.net.worker as provider_worker
import flash.runner.lifecycle.preparation as runner_preparation
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.supervise.lifecycle as runner_lifecycle

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tests._helpers.source_snapshot import valid_source_snapshot

SPEC = {
    "model": "Qwen/Qwen3.5-9B",
    "project": "11111111-1111-4111-8111-111111111111",
    "algorithm": "grpo",
    # A hub slug, because this fixture drives the HOSTED api: the managed plane accepts
    # `namespace/project/name` only, and a `github:` ref is refused at submit (see test_server_standalone).
    "environment": {"id": "acme/example-project/gsm8k"},
    "train": {"epochs": 1, "max_examples": 1},
    "gpu": {},
}

_USER_PREFIX = "fslo-user-"
_SOURCE_SNAPSHOT = valid_source_snapshot()


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

    SPEC is a catalog model on a concrete GPU pin, so the parse never probes HF.
    """
    from flash.schema import spec_from_dict

    return spec_from_dict(SPEC, run_id="run-1")


def test_cents_rounds_and_floors_at_zero():
    from flash.server.billing.charges import _cents

    assert _cents(12.34) == 1234
    assert _cents(0.0) == 0
    assert _cents(-5.0) == 0  # never negative


def test_cents_rounds_half_up_not_bankers():
    """Money rounding is round-HALF-UP, not Python's ties-to-even (which would undercharge a
    half-cent tie, e.g. $0.005 -> 0)."""
    from flash.server.billing.charges import _cents

    assert _cents(0.005) == 1  # ties-to-even would give 0
    assert _cents(0.015) == 2  # ties-to-even would give 2 as well, but via half-up here
    assert _cents(0.025) == 3  # ties-to-even would give 2 -> half-up gives 3
    assert _cents(1.005) == 101  # classic banker's-rounding trap


def _completed_status(monkeypatch, *, cost_usd: float = 12.345, org_id: str = "org-A"):
    from flash.runner.lifecycle.state import RunStatus

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
    from flash.server.billing import charges as billing

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
    assert body["model"] == "Qwen/Qwen3.5-9B"
    assert body["estimate"] == {
        "totalUsd": 12.345,
        "costBasis": "estimate",
        "costSource": "run_status.cost_usd",
    }


def test_charge_bills_cost_usd_for_a_cancelled_run(monkeypatch):
    """A cancelled run is charged its cost_usd exactly like a completed one -- the cancel path already
    set cost_usd to the estimate at the steps actually run, so charge_completed_run just bills it."""
    from flash.runner.lifecycle.state import RunStatus
    from flash.server.billing import charges as billing

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
    from flash.server.billing import charges as billing

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
    from flash.server.billing import charges as billing

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(billing.BillingError) as exc:
        billing.charge_completed_run(internal_key="tok", status=_completed_status(monkeypatch))
    assert exc.value.status_code == 503


def test_charge_completed_run_requires_billing_org(monkeypatch):
    from flash.server.billing import charges as billing

    status = _completed_status(monkeypatch, org_id="")

    with pytest.raises(billing.BillingError) as exc:
        billing.charge_completed_run(internal_key="tok", status=status)
    assert exc.value.status_code == 400
    assert "org id" in exc.value.detail


def test_http_error_detail_falls_back_to_reason():
    """When the error body is missing/unparseable, the detail keeps the backend REASON
    (not a bare ``billing failed (<code>)``) so a 4xx/5xx stays diagnosable."""
    from flash.server.billing.charges import _http_error_detail

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
    # runpod.auth caches the parsed pool on first read; reset so the startup preflight reads THIS
    # RUNPOD_API_KEY (the autouse _offline fixture also resets, but make the fixture self-contained).
    import flash.providers.runpod.client.auth as runpod_keys

    runpod_keys.reset()
    import flash.server.platform.auth as auth_mod
    import flash.server.platform.db as db_mod

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(provider_worker, "publish_source_snapshot", lambda _repo: _SOURCE_SNAPSHOT)
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))
    # Keep submit offline: validate + record, but the GPU job body is a no-op.
    monkeypatch.setattr(runner_lifecycle, "_run_job", lambda *a, **k: None)

    import flash.server.asgi.app as app_mod

    importlib.reload(app_mod)
    # configured provider keys would trigger orphan sweeps and status reporting. stub both because
    # these billing tests assert only on the API response and must remain network-free.
    import flash.server.domain.registry.projects as projects_mod
    import flash.server.domain.registry.runs as runs
    from flash.providers.core import registry as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", list, raising=False)
    monkeypatch.setattr(
        projects_mod,
        "require_project_access",
        lambda *, project_id, **_kwargs: project_id,
    )
    # A hub environment is validated against the Freesolo backend at submit. These are BILLING
    # tests, so stub it to a pass the same way `require_project_access` is stubbed above; without
    # it every submit here 4xxs on environment authorization before reaching any billing code.
    import flash.server.domain.registry.environment_registry as environment_registry_mod

    monkeypatch.setattr(
        environment_registry_mod,
        "require_environment_project",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(runs, "_post", lambda *a, **k: False, raising=False)
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token.startswith(_USER_PREFIX))
    monkeypatch.setattr(auth_mod, "_cached_identity", _identity_for_token)
    # The new submit-time budget precheck would urllib-POST the real backend; stub it to a pass so
    # the default submit path stays hermetic. Gate-specific tests below override this per-test.
    import flash.server.billing.charges as billing_mod

    monkeypatch.setattr(billing_mod, "precheck_training_run", lambda **k: {"ok": True})
    with TestClient(app_mod.create_app()) as client:
        # startup preflight needs the fake token above; billing requests do not. keeping `ghp-test`
        # here turns an offline billing test into a real environment-pin request to GitHub.
        monkeypatch.delenv("GITHUB_TOKEN")
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
    import flash.server.billing.charges as billing_mod
    import flash.server.platform.db as db_mod

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
    import flash.server.billing.charges as billing_mod

    def _block(**k):
        raise billing_mod.BillingError(402, "insufficient balance")

    monkeypatch.setattr(billing_mod, "precheck_training_run", _block)
    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-1"))
    assert res.status_code == 402, res.text
    assert "insufficient" in res.text
    # rejected before record_run: nothing persisted for this user.
    assert api.get("/v1/runs", headers=_bearer("fslo-user-1")).json()["runs"] == []


def test_a_nonexistent_environment_is_refused_before_the_402(api, monkeypatch):
    """A run that can never launch must not be reported as an affordability problem.

    The budget precheck runs on the request path, so a gate placed after it answers "insufficient
    balance" for a typo'd environment -- sending the user to top up, for a run no balance can buy.
    Both faults are live here at once, and the spec's own defect has to win.
    """
    import flash.envs.loading.loader as env_loader
    import flash.server.billing.charges as billing_mod
    from flash.envs.meta.identity import GitHubPermanentError

    def _block(**k):
        raise billing_mod.BillingError(402, "insufficient balance")

    def _permanent(_parsed, *_a, **_k):
        raise GitHubPermanentError("GitHub environment request failed (404): Not Found")

    monkeypatch.setattr(billing_mod, "precheck_training_run", _block)
    monkeypatch.setattr(env_loader, "_github_token", lambda: "ghp_test")
    monkeypatch.setattr(env_loader, "_resolve_ref_sha", _permanent)

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-1"))

    assert res.status_code == 400, res.text
    assert "could not be resolved on GitHub" in res.text
    assert "insufficient" not in res.text


def test_transient_environment_resolve_is_attempted_once_per_request(api, monkeypatch):
    """packaged opd defers after one bounded request-side attempt on a transient failure."""
    import flash.envs.loading.loader as env_loader
    from flash.envs.meta.identity import GitHubUnavailableError

    calls = []

    def _transient(_parsed, *_args, **_kwargs):
        calls.append(1)
        raise GitHubUnavailableError("GitHub server error (503, transient)")

    monkeypatch.setattr(env_loader, "_github_token", lambda: "ghp_test")
    monkeypatch.setattr(env_loader, "_resolve_ref_sha", _transient)
    monkeypatch.setattr(
        "flash.server.domain.teacher.broker.preflight_validate_managed_teacher",
        lambda _spec: None,
    )
    spec = {
        **SPEC,
        "algorithm": "opd",
        "train": {**SPEC["train"], "teacher_model": "deepseek-v4-pro"},
    }

    res = api.post(
        "/v1/runs",
        json={"spec": spec, "dry_run": True},
        headers=_bearer("fslo-user-1"),
    )

    assert res.status_code == 200, res.text
    assert res.json()["state"] == "dry_run"
    assert calls == [1], f"environment ref was resolved {len(calls)} times"


def test_tokenless_packaged_opd_defers_without_anonymous_github_lookup(api, monkeypatch):
    """a tokenless plane must not turn a private packaged opd environment into a false 404."""
    import flash.envs.loading.loader as env_loader
    from flash.envs.meta.identity import GitHubPermanentError

    calls = []

    def _anonymous_404(_parsed, *_args, **_kwargs):
        calls.append(1)
        raise GitHubPermanentError("GitHub environment request failed (404): Not Found")

    monkeypatch.setattr(env_loader, "_github_token", lambda: None)
    monkeypatch.setattr(env_loader, "_resolve_ref_sha", _anonymous_404)
    monkeypatch.setattr(
        "flash.server.domain.teacher.broker.preflight_validate_managed_teacher",
        lambda _spec: None,
    )
    spec = {
        **SPEC,
        "algorithm": "opd",
        "train": {**SPEC["train"], "teacher_model": "deepseek-v4-pro"},
    }

    res = api.post(
        "/v1/runs",
        json={"spec": spec, "dry_run": True},
        headers=_bearer("fslo-user-1"),
    )

    assert res.status_code == 200, res.text
    assert res.json()["state"] == "dry_run"
    assert calls == []


def test_submit_fails_open_when_precheck_unreachable(api, monkeypatch):
    # a non-402 billing error (backend unreachable / 5xx) must not block training; the completion
    # charge is the backstop. the run is still accepted and recorded.
    import flash.server.billing.charges as billing_mod

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
    import flash.server.billing.charges as billing_mod
    import flash.server.platform.db as db_mod

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
    import flash.server.billing.charges as billing_mod

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
    import flash.server.billing.charges as billing_mod

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
    import flash.server.billing.charges as billing_mod

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
    import flash.server.platform.internal_client as internal_mod

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
    import flash.server.billing.charges as billing_mod

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
    import flash.server.billing.charges as billing_mod

    def _block(**k):
        raise billing_mod.BillingError(402, "insufficient balance")

    monkeypatch.setattr(billing_mod, "precheck_training_run", _block)
    unsupported_spec = {
        **SPEC,
        # image records distilled from a text-only teacher. the subject of this test is the
        # ORDERING, so the case only has to be something static validation refuses; it deliberately
        # is not a warm-start or a multi-turn-image case, because both of those were once
        # unsupported and have since been implemented, and each silently turned this into a test
        # that asserted nothing the moment its premise became supported. a teacher that cannot see
        # images is a property of the teacher, not a gap waiting to be closed.
        "algorithm": "opd",
        "train": {**SPEC["train"], "teacher_model": "deepseek-v4-pro"},
        "environment": {
            **SPEC["environment"],
            "params": {
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
    assert "cannot see images" in res.text
    assert "insufficient" not in res.text


def test_internal_submit_skips_affordability_precheck_before_persistence(api, monkeypatch):
    import flash.server.billing.charges as billing_mod
    import flash.server.platform.db as db_mod

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
    import flash.server.platform.auth as auth_mod

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
    import flash.server.asgi.app as app_mod

    def failing_submit(spec, dry_run=False, background=True, **kwargs):
        raise RuntimeError("provider out of capacity")

    monkeypatch.setattr(app_mod, "submit_job", failing_submit)

    res = api.post("/v1/runs", json={"spec": SPEC}, headers=_bearer("fslo-user-r"))
    assert res.status_code == 400, res.text
    assert "out of capacity" in res.json()["detail"]
    assert api.get("/v1/runs", headers=_bearer("fslo-user-r")).json()["runs"] == []


def test_record_run_failure_does_not_submit(api, monkeypatch):
    """If ``db.record_run`` fails (e.g. SQLite locked/full), submit_job is not reached."""
    import flash.server.asgi.app as app_mod
    import flash.server.platform.db as db_mod

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


_SFT_SPEC = {**SPEC, "algorithm": "sft", "train": {"epochs": 1, "max_examples": 8}}


def test_completion_hook_charges_final_cost(monkeypatch, tmp_path):
    from flash.runner.lifecycle.state import RunStatus
    from flash.runner.supervise import lifecycle

    spec = _spec(monkeypatch)
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal")
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    runner_state._save_status(
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

    monkeypatch.setattr("flash.server.billing.charges.charge_completed_run", fake_charge)
    log = io.StringIO()

    lifecycle._charge_completed_run_by_id(spec.run_id, log)

    assert calls == [("fslo-internal", "run-1", 1.23)]
    status = runner_status.get_status("run-1")
    assert status.billing_state == "charged"
    assert status.billing_error is None
    assert status.billing_charge == {"amountCents": 123, "balanceCents": 877, "replay": False}
    assert "billing charged" in log.getvalue()


def test_completion_hook_records_missing_internal_key(monkeypatch, tmp_path):
    from flash.runner.lifecycle.state import RunStatus
    from flash.runner.supervise import lifecycle

    spec = _spec(monkeypatch)
    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    runner_state._save_status(
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
        "flash.server.billing.charges.charge_completed_run",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not charge without internal key")),
    )

    lifecycle._charge_completed_run_by_id(spec.run_id, io.StringIO())

    status = runner_status.get_status("run-1")
    assert status.billing_state == "failed"
    assert "FREESOLO_INTERNAL_KEY" in (status.billing_error or "")


# --------------------------------------------------------- warm-start error attribution


def test_route_blames_the_adapter_only_for_tagged_failures(api, monkeypatch):
    # a genuine adapter-resolution failure keeps the actionable 400 that names the source.

    import flash.server.asgi.app as app_mod

    # raise the class off the reloaded module: the api fixture reloads flash.runner, so a symbol
    # imported at module scope here would be a different object than the route's.
    def _prepare(*args, **kwargs):
        raise runner_preparation.WarmStartPreparationError("private-owner/private-repo:sft/step-20")

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
    import flash.server.asgi.app as app_mod

    def _prepare(*args, **kwargs):
        raise ValueError("no gpu class satisfies the requested memory")

    monkeypatch.setattr(app_mod, "prepare_job", _prepare)
    warm_start_spec = {
        **SPEC,
        "train": {**SPEC["train"], "init_from_adapter": "source-run/final"},
    }
    res = api.post("/v1/runs", json={"spec": warm_start_spec}, headers=_bearer("fslo-user-1"))

    assert "no gpu class satisfies" in res.text
    assert "init_from_adapter" not in res.text

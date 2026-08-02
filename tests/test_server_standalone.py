"""Standalone mode is a deployment shape, not a permission downgrade.

``FLASH_STANDALONE`` exists so a self-hosted plane -- one with no Freesolo backend behind it --
can run at all: the normal paths validate every bearer token, project, and environment against a
SaaS backend the self-hoster does not have. The risk in that seam is obvious, so these tests pin
the direction it fails in: standalone must accept FEWER credentials than managed mode, never more.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from flash.server import auth


@pytest.fixture(autouse=True)
def _clear_verify_state():
    with auth._verify_cache_lock:
        auth._verify_cache.clear()
        auth._verify_inflight.clear()
    yield
    with auth._verify_cache_lock:
        auth._verify_cache.clear()
        auth._verify_inflight.clear()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_standalone_is_on_for_the_documented_spellings(monkeypatch, value: str) -> None:
    monkeypatch.setenv(auth.STANDALONE_ENV, value)
    assert auth.standalone() is True


@pytest.mark.parametrize("value", ["", "  ", "0", "false", "no", "off", "maybe"])
def test_standalone_is_off_by_default_and_for_anything_else(monkeypatch, value: str) -> None:
    """Managed mode is the default. A typo'd value must not silently relax project ownership."""
    monkeypatch.setenv(auth.STANDALONE_ENV, value)
    assert auth.standalone() is False

    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert auth.standalone() is False


def test_standalone_rejects_an_external_token_instead_of_trusting_it(monkeypatch) -> None:
    """The failure direction that matters.

    Managed mode verifies an external bearer token against the backend. Standalone has no backend
    to ask -- and the tempting shortcut, treating "cannot verify" as "accept", would turn every
    self-hosted plane into an open one. Unverifiable means rejected; the operator key is the only
    credential standalone honours.
    """
    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")

    def _boom(*_args, **_kwargs):
        raise AssertionError("standalone must not call the Freesolo backend to verify a token")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    assert auth.authenticate("Bearer some-external-user-key") is None
    assert auth.authenticate("Bearer ") is None
    assert auth.authenticate(None) is None


def test_standalone_accepts_the_operator_key_as_internal(monkeypatch, tmp_path) -> None:
    """The operator key is the whole trust boundary of a standalone plane, so it must still work."""
    from flash.server import db

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    # Resolving the key registers it, so point the db at tmp rather than the operator's real one.
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))

    def _boom(*_args, **_kwargs):
        raise AssertionError("the operator key is resolved locally, with no backend call")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    row = auth.authenticate("Bearer operator-key")
    assert row is not None
    assert row["auth_kind"] == "internal"


def test_standalone_disables_backend_reporting_at_the_shared_gate(monkeypatch) -> None:
    """A standalone plane SETS the internal key -- it is how its own clients authenticate.

    Every best-effort reporter (billing, checkpoint registration, environment recording) gates on
    that key, so without this they would all POST the operator's key to api.freesolo.co and log a
    warning per run. One gate, so the reporters cannot drift apart.
    """
    from flash.server._internal_client import internal_key

    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert internal_key() == "operator-key"

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    assert internal_key() is None


def test_standalone_disables_the_backend_polling_loops(monkeypatch) -> None:
    """Cost reconciliation and charge retry poll on a timer against a backend that isn't there."""
    from flash.server.billing_retry import charge_retry_enabled
    from flash.server.reconcile import reconcile_enabled

    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert reconcile_enabled() is True
    assert charge_retry_enabled() is True

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    assert reconcile_enabled() is False
    assert charge_retry_enabled() is False


def test_standalone_disables_the_artifact_gc_sweep(monkeypatch) -> None:
    """The GC is the one background loop gated on HF_TOKEN rather than the internal key, so it
    survived the first pass and shipped the operator's key to serve.freesolo.co on every startup --
    it confirms the live set against the hosted serving registry before deleting. It can only ever
    delete inside the hardcoded Freesolo-Co/flashrun-* allowlist, which a self-hoster's token does
    not own, so standalone loses nothing by skipping it."""
    from flash.server.repo_cleanup import repo_cleanup_enabled

    monkeypatch.setenv("HF_TOKEN", "hf-operator-token")
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert repo_cleanup_enabled() is True

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    assert repo_cleanup_enabled() is False


def test_standalone_falls_back_to_the_in_process_slot_semaphore(monkeypatch) -> None:
    """The shared RunPod slot store is a backend table; standalone caps concurrency in-process."""
    from flash.providers.runpod import slots

    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert slots.internal_key() == "operator-key"

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    assert slots.internal_key() is None


def test_standalone_refuses_to_default_the_serving_url(monkeypatch) -> None:
    """Every serving request carries FREESOLO_INTERNAL_KEY, which on a self-hosted plane is the
    credential controlling that plane. Falling back to the hosted default would ship it to a
    service the operator does not run, on an ordinary `flash deploy`/`chat`. Raising covers every
    caller (serving_openai_base_url included) rather than stripping the header at one call site."""
    from flash.serve import deploy

    monkeypatch.delenv("FREESOLO_SERVING_URL", raising=False)
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    # managed: the hosted default is correct and stays
    assert deploy.DEFAULT_FREESOLO_SERVING_URL in deploy.serving_base_url()

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    with pytest.raises(deploy.ServingError) as excinfo:
        deploy.serving_base_url()
    assert "FREESOLO_SERVING_URL" in str(excinfo.value)
    # the OpenAI base url derives from the same resolver, so it is covered too
    with pytest.raises(deploy.ServingError):
        deploy.serving_openai_base_url()

    # an explicitly configured backend is honoured in standalone
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serving.example.internal")
    assert "serving.example.internal" in deploy.serving_base_url()


def test_standalone_operator_key_survives_surrounding_whitespace(monkeypatch, tmp_path) -> None:
    """The preflight tests the STRIPPED value, so an env file with a trailing newline starts a plane
    that then rejects the only credential it accepts -- every request 401 behind a preflight that
    reported success. Both sides strip, so startup and authentication agree."""
    from flash.server import db

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key\n")
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))

    row = auth.authenticate("Bearer operator-key")
    assert row is not None
    assert row["auth_kind"] == "internal"


def test_a_whitespace_only_operator_key_authenticates_nothing(monkeypatch) -> None:
    """A blank key must never become a usable credential once both sides strip."""
    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "   ")
    assert auth.authenticate("Bearer    ") is None
    assert auth.authenticate("Bearer ") is None


def test_artifact_namespace_is_operator_configurable(monkeypatch) -> None:
    """Flash CREATES the HF dataset repos it streams artifacts through, so the namespace has to be
    one the operator's HF_TOKEN can write to. Hardcoding Freesolo's made self-hosting impossible:
    the assignment runs on every submit, and a self-hoster cannot create Freesolo-Co/flashrun-*, so
    the run died at upload before training started."""
    from flash import runner
    from flash.server import repo_cleanup

    monkeypatch.delenv("FLASH_HF_NAMESPACE", raising=False)
    assert runner.artifact_namespace() == runner._DEFAULT_ARTIFACT_NAMESPACE
    assert runner.managed_hf_repo_for_environment("env-1").startswith(
        f"{runner._DEFAULT_ARTIFACT_NAMESPACE}/"
    )

    monkeypatch.setenv("FLASH_HF_NAMESPACE", "self-hoster")
    assert runner.artifact_namespace() == "self-hoster"
    assert runner.managed_hf_repo_for_environment("env-1").startswith("self-hoster/")
    # the GC allowlist must follow the same namespace, or it silently stops matching
    assert repo_cleanup._is_managed_env_repo("self-hoster/flashrun-env-1-abc") is True
    assert repo_cleanup._is_managed_env_repo("someone-else/flashrun-env-1-abc") is False
    # a blank override falls back rather than producing a "/flashrun-*" repo id
    monkeypatch.setenv("FLASH_HF_NAMESPACE", "   ")
    assert runner.artifact_namespace() == runner._DEFAULT_ARTIFACT_NAMESPACE


def test_a_whitespace_only_provider_key_reads_as_unconfigured(monkeypatch) -> None:
    """`is_configured()` decides whether a provider is advertised to the allocator and whether the
    startup preflight passes. A whitespace-only key (a stray newline in an env file) must not make
    a plane advertise a substrate every allocation then fails on."""
    from flash.providers._auth import load_provider_key

    monkeypatch.setenv("LAMBDA_API_KEY", "   \n ")
    assert load_provider_key("LAMBDA_API_KEY") is None
    monkeypatch.setenv("LAMBDA_API_KEY", "  real-key  ")
    assert load_provider_key("LAMBDA_API_KEY") == "real-key"
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    assert load_provider_key("LAMBDA_API_KEY") is None


def test_standalone_does_not_charge_a_recovered_managed_run(monkeypatch, tmp_path) -> None:
    """A standalone plane started against an existing state directory can hold a run that still
    carries a managed-mode billing_context. Charging it would send the operator's key to
    FREESOLO_BASE_URL and bill an organization this plane has no relationship with, so the inline
    charge has to read the SHARED gate rather than the env var directly."""
    from flash.runner import lifecycle

    charged: list[str] = []

    class _Status:
        billing_context: ClassVar[dict[str, str]] = {"org_id": "org-from-the-managed-plane"}
        billing_state = "pending"
        state = "done"
        cost_usd = 12.0

    recorded: list[dict] = []
    monkeypatch.setattr("flash.runner.get_status", lambda _run_id: _Status(), raising=False)
    monkeypatch.setattr(
        "flash.runner.record_billing_state",
        lambda run_id, **kw: recorded.append(kw),
        raising=False,
    )

    def _charge(_key, _status):
        charged.append("called")
        return {}

    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    log = (tmp_path / "run.log").open("w")

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    lifecycle._apply_charge_with_state("run-1", log, charge_call=_charge, noun="terminal")
    assert charged == [], "standalone must not reach the Freesolo billing backend"
    assert recorded
    assert recorded[-1]["billing_state"] == "failed"
    assert "standalone" in recorded[-1]["billing_error"]

    # managed mode with the same key still charges -- the gate narrows standalone only
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    lifecycle._apply_charge_with_state("run-1", log, charge_call=_charge, noun="terminal")
    assert charged == ["called"]
    log.close()

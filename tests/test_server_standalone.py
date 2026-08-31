"""Standalone mode is a deployment shape, not a permission downgrade.

``FLASH_STANDALONE`` exists so a self-hosted plane -- one with no Freesolo backend behind it --
can run at all: the normal paths validate every bearer token, project, and environment against a
SaaS backend the self-hoster does not have. The risk in that seam is obvious, so these tests pin
the direction it fails in: standalone must accept FEWER credentials than managed mode, never more.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

import flash.runner.accounting.artifacts as runner_artifacts
import flash.serve.contract.errors as serving_errors
import flash.serve.contract.urls as serving_urls
import flash.serve.request.transport as serving_transport
from flash.server.platform import auth


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
    from flash.server.platform import db

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
    from flash.server.platform.internal_client import internal_key

    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert internal_key() == "operator-key"

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    assert internal_key() is None


def test_standalone_disables_the_backend_polling_loops(monkeypatch) -> None:
    """Cost reconciliation and charge retry poll on a timer against a backend that isn't there."""
    from flash.server.billing.retry import charge_retry_enabled
    from flash.server.domain.ops.reconcile import reconcile_enabled

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
    from flash.server.domain.ops.repo_cleanup import repo_cleanup_enabled

    monkeypatch.setenv("HF_TOKEN", "hf-operator-token")
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert repo_cleanup_enabled() is True

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    assert repo_cleanup_enabled() is False


def test_standalone_refuses_to_default_the_serving_url(monkeypatch) -> None:
    """Every serving request carries FREESOLO_INTERNAL_KEY, which on a self-hosted plane is the
    credential controlling that plane. Falling back to the hosted default would ship it to a
    service the operator does not run, on an ordinary `flash deploy`/`chat`. Raising covers every
    caller (serving_openai_base_url included) rather than stripping the header at one call site."""
    monkeypatch.delenv("FREESOLO_SERVING_URL", raising=False)
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    # managed: the hosted default is correct and stays
    assert serving_urls.default_serving_url() in serving_urls.serving_base_url()

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    with pytest.raises(serving_errors.ServingError) as excinfo:
        serving_urls.serving_base_url()
    assert "FREESOLO_SERVING_URL" in str(excinfo.value)
    # the OpenAI base url derives from the same resolver, so it is covered too
    with pytest.raises(serving_errors.ServingError):
        serving_transport.serving_openai_base_url()

    # an explicitly configured backend is honoured in standalone
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serving.example.internal")
    assert "serving.example.internal" in serving_urls.serving_base_url()


# Every spelling below resolves to Freesolo-operated infrastructure. An operator carrying any of
# them in an old `.env` must not have the plane's root credential shipped to a third party.
HOSTED_SERVING_URL_SPELLINGS = [
    "https://serve.freesolo.co",
    "https://serve.freesolo.co/",
    "https://serve.freesolo.co/v1",
    "  https://serve.freesolo.co  ",
    "https://SERVE.FREESOLO.CO",
    "https://Serve.Freesolo.Co/v1",
    "http://serve.freesolo.co",
    "https://serve.freesolo.co:443",
    "https://serve.freesolo.co./v1",
    "https://user:pw@serve.freesolo.co",
    "serve.freesolo.co",
    "https://api.freesolo.co",
    "https://freesolo.co",
    # The dev-channel serving plane is Freesolo-operated too, so a standalone plane must be
    # refused it on exactly the same grounds. Listed explicitly because the dev default became
    # reachable when `default_serving_url` started deriving from CHANNEL: the guard matches on the
    # parsed host and needed no change, and this pins that it did not need one.
    "https://serve-dev.freesolo.co",
    "https://serve-dev.freesolo.co/v1",
]


@pytest.mark.parametrize("configured", HOSTED_SERVING_URL_SPELLINGS)
def test_standalone_refuses_an_explicitly_configured_hosted_serving_url(monkeypatch, configured):
    """SET-to-the-hosted-URL is the case an unset-only guard misses, and it is the LIKELY one.

    A self-hoster who started from a copied `.env` -- or who ran managed first -- has
    FREESOLO_SERVING_URL already assigned to the hosted default. Guarding only the unset case
    means `flash deploy`/`chat` still POST to serve.freesolo.co with FREESOLO_INTERNAL_KEY in
    X-Freesolo-Internal-Key, which on a standalone plane is the credential controlling the plane.

    Matched on the parsed HOST, so the guard is not a string compare against one canonical
    spelling: scheme, case, port, trailing dot, credentials, and /v1 suffix all vary in real
    config files and every one of them reaches the same third party.
    """
    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    monkeypatch.setenv("FREESOLO_SERVING_URL", configured)

    with pytest.raises(serving_errors.ServingError) as excinfo:
        serving_urls.serving_base_url()
    assert "FREESOLO_SERVING_URL" in str(excinfo.value)
    # same resolver underneath, so the OpenAI base url cannot be used to route around it
    with pytest.raises(serving_errors.ServingError):
        serving_transport.serving_openai_base_url()

    # managed mode is unaffected: the hosted backend is exactly what it should be talking to.
    monkeypatch.delenv(auth.STANDALONE_ENV)
    assert serving_urls.serving_base_url()


@pytest.mark.parametrize(
    "configured",
    [
        "https://serving.example.internal",
        "http://localhost:8000",
        "https://my-serve.example.com/v1",
        # a host that merely CONTAINS the domain is not ours -- the guard must not over-block
        "https://notfreesolo.co",
        "https://freesolo.co.evil.example.com",
        "https://serve.freesolo.co.attacker.test",
    ],
)
def test_standalone_still_allows_a_serving_backend_the_operator_runs(monkeypatch, configured):
    """The counterpart: over-blocking would make self-hosted serving impossible, which is the
    feature. Suffix matching is on a dotted boundary, so `freesolo.co.evil.example.com` and
    `notfreesolo.co` are other people's hosts and stay allowed."""
    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    monkeypatch.setenv("FREESOLO_SERVING_URL", configured)
    assert serving_urls.serving_base_url()
    assert serving_transport.serving_openai_base_url().endswith("/v1")


def test_standalone_operator_key_survives_surrounding_whitespace(monkeypatch, tmp_path) -> None:
    """The preflight tests the STRIPPED value, so an env file with a trailing newline starts a plane
    that then rejects the only credential it accepts -- every request 401 behind a preflight that
    reported success. Both sides strip, so startup and authentication agree."""
    from flash.server.platform import db

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
    from flash.server.domain.ops import repo_cleanup

    monkeypatch.delenv("FLASH_HF_NAMESPACE", raising=False)
    assert runner_artifacts.artifact_namespace() == runner_artifacts._DEFAULT_ARTIFACT_NAMESPACE
    assert runner_artifacts.managed_hf_repo_for_environment("env-1").startswith(
        f"{runner_artifacts._DEFAULT_ARTIFACT_NAMESPACE}/"
    )

    monkeypatch.setenv("FLASH_HF_NAMESPACE", "self-hoster")
    assert runner_artifacts.artifact_namespace() == "self-hoster"
    assert runner_artifacts.managed_hf_repo_for_environment("env-1").startswith("self-hoster/")
    # the GC allowlist must follow the same namespace, or it silently stops matching
    assert repo_cleanup._is_managed_env_repo("self-hoster/flashrun-env-1-abc") is True
    assert repo_cleanup._is_managed_env_repo("someone-else/flashrun-env-1-abc") is False
    # a blank override falls back rather than producing a "/flashrun-*" repo id
    monkeypatch.setenv("FLASH_HF_NAMESPACE", "   ")
    assert runner_artifacts.artifact_namespace() == runner_artifacts._DEFAULT_ARTIFACT_NAMESPACE


def test_a_whitespace_only_provider_key_reads_as_unconfigured(monkeypatch) -> None:
    """`is_configured()` decides whether a provider is advertised to the allocator and whether the
    startup preflight passes. A whitespace-only key (a stray newline in an env file) must not make
    a plane advertise a substrate every allocation then fails on."""
    from flash.providers._lifecycle.net.auth import load_provider_key

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
    from flash.runner.supervise import lifecycle

    charged: list[str] = []

    class _Status:
        billing_context: ClassVar[dict[str, str]] = {"org_id": "org-from-the-managed-plane"}
        billing_state = "pending"
        state = "done"
        cost_usd = 12.0

    recorded: list[dict] = []
    monkeypatch.setattr(
        "flash.runner.lifecycle.status.get_status", lambda _run_id: _Status(), raising=False
    )
    monkeypatch.setattr(
        "flash.runner.accounting.costs.record_billing_state",
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


def test_standalone_run_ownership_survives_operator_key_rotation(monkeypatch, tmp_path) -> None:
    """Rotating FREESOLO_INTERNAL_KEY must not orphan the runs the old key started.

    Standalone is single-tenant, so the operator key owns every run. Deriving the owner row from
    the key's HASH meant a rotation (or a re-run of the quickstart's `openssl rand`) minted a new
    row with a new id, and every run -- matched by `runs.key_id` -- vanished: absent from the
    listing, 404 on status/logs/cancel. An in-flight job would keep spending with no supported way
    for the new credential to stop it, so rotating a COMPROMISED key was the thing that cost you
    control of the plane.
    """
    from flash.server.platform import db

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))

    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key-v1")
    before = auth.authenticate("Bearer operator-key-v1")
    assert before is not None
    db.record_run("run-started-before-rotation", before["id"])

    # the operator rotates the secret (leak, restart, policy) -- a different key VALUE entirely.
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key-v2")
    after = auth.authenticate("Bearer operator-key-v2")
    assert after is not None
    assert after["auth_kind"] == "internal"

    assert after["id"] == before["id"], "the rotated key must own the runs the old key started"
    assert db.run_owner("run-started-before-rotation") == after["id"]
    assert [r["run_id"] for r in db.runs_for_key(after["id"])] == ["run-started-before-rotation"]

    # and the old secret stops working -- rotation still revokes.
    assert auth.authenticate("Bearer operator-key-v1") is None


def test_the_standalone_owner_row_is_not_reachable_by_presenting_a_token(monkeypatch, tmp_path):
    """The owner row is keyed on a sentinel, not a hash, so no token can resolve to it directly."""
    from flash.server.platform import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    owner = db.ensure_standalone_owner()

    assert db.lookup_key(db._STANDALONE_OWNER_HASH) is None
    assert db.lookup_key("standalone-operator") is None
    # idempotent: a second call returns the SAME row, never a second owner.
    assert db.ensure_standalone_owner()["id"] == owner["id"]


def test_adoption_does_not_write_once_every_run_is_already_owned(monkeypatch, tmp_path):
    """`ensure_standalone_owner` runs on EVERY authenticated request, so in the steady state it
    must issue NO write statements at all.

    SQLite has one write slot and takes it for any write regardless of rows touched, so a no-op
    write still serializes concurrent status/logs/submit behind whichever request is recording a
    run. Both writes here are affected: the adoption `UPDATE` (which additionally cannot use
    `runs_key_idx` -- `WHERE key_id != ?` plans as `SCAN runs`) and the `INSERT OR IGNORE` that
    provisions the owner row.

    Asserted on ALL writes rather than on `UPDATE runs` specifically. An earlier version of this
    test named the UPDATE, and so kept passing while the unguarded INSERT held the write slot on
    every single request -- the narrower assertion could not see the second half of its own claim.

    Counted at the sqlite layer rather than timed: a timing assertion on a small fixture is noise,
    and asserting "no write was issued" is what actually distinguishes the implementations.
    """
    from flash.server.platform import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))

    # a run belonging to some OTHER key row -- the managed-history case adoption exists for.
    stale = db.ensure_internal_key("a-previous-managed-key")
    db.record_run("run-from-before-standalone", stale["id"])

    # recorded through sqlite's own trace callback rather than by wrapping `execute`: it sees every
    # statement the driver actually runs, and `sqlite3.Connection` is immutable so it cannot be
    # patched anyway.
    writes: list[str] = []
    real_connect = db._connect

    def tracing_connect():
        conn = real_connect()
        conn.set_trace_callback(
            lambda sql: (
                writes.append(" ".join(sql.split()))
                if sql.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE"))
                else None
            )
        )
        return conn

    monkeypatch.setattr(db, "_connect", tracing_connect)

    # first call: there IS a foreign row, so the adoption must run and claim it.
    owner = db.ensure_standalone_owner()
    assert db.run_owner("run-from-before-standalone") == owner["id"]
    assert any(w.upper().startswith("UPDATE RUNS") for w in writes), (
        "the backlog was not adopted on the first call"
    )

    # steady state: the owner row exists and every run is owned, so no further call may issue ANY
    # write -- not the adoption UPDATE, and not the owner-row INSERT either.
    writes.clear()
    for _ in range(3):
        assert db.ensure_standalone_owner()["id"] == owner["id"]
    assert writes == [], f"took the write slot with nothing to do: {writes}"

    # a foreign row appearing LATER is still adopted -- the guard is a check, not a one-shot latch.
    later = db.ensure_internal_key("another-key")
    db.record_run("run-appearing-later", later["id"])
    assert db.ensure_standalone_owner()["id"] == owner["id"]
    assert db.run_owner("run-appearing-later") == owner["id"]


def test_the_owner_row_is_still_provisioned_on_a_cold_database(monkeypatch, tmp_path):
    """The counterpart to the steady-state check above: the owner INSERT sits behind a read now,
    so a database that has never seen it must still get one -- and the first call is the only
    chance, since nothing else provisions that row."""
    from flash.server.platform import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))

    writes: list[str] = []
    real_connect = db._connect

    def tracing_connect():
        conn = real_connect()
        conn.set_trace_callback(
            lambda sql: (
                writes.append(" ".join(sql.split()))
                if sql.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE"))
                else None
            )
        )
        return conn

    monkeypatch.setattr(db, "_connect", tracing_connect)

    assert db.lookup_key(db._STANDALONE_OWNER_HASH) is None
    owner = db.ensure_standalone_owner()
    assert owner["id"]
    assert owner["key_hash"] == db._STANDALONE_OWNER_HASH
    assert any(w.upper().startswith("INSERT OR IGNORE INTO API_KEYS") for w in writes), (
        f"the owner row was never provisioned on a cold db: {writes}"
    )

    # and a run recorded against it resolves back, so the row is usable and not just present.
    db.record_run("run-on-a-fresh-plane", owner["id"])
    assert db.run_owner("run-on-a-fresh-plane") == owner["id"]

    # second call on the now-warm db: same row, and no write at all.
    writes.clear()
    assert db.ensure_standalone_owner()["id"] == owner["id"]
    assert writes == [], f"re-provisioned an owner row that already existed: {writes}"


def test_standalone_startup_requires_a_writable_artifact_namespace(monkeypatch) -> None:
    """Without FLASH_HF_NAMESPACE a self-hoster's artifacts default to a namespace their HF_TOKEN
    cannot write to, so every run dies at artifact upload -- AFTER preflight called the plane
    healthy. Fail at startup, where the operator can act on it."""
    from flash.providers.core.preflight import PreflightError, check_run_preflight

    monkeypatch.setenv("HF_TOKEN", "hf-operator-token")
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    monkeypatch.setenv("VAST_API_KEY", "vast-key")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    monkeypatch.delenv("FLASH_HF_NAMESPACE", raising=False)

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    with pytest.raises(PreflightError, match="FLASH_HF_NAMESPACE"):
        check_run_preflight()

    # set: boots.
    monkeypatch.setenv("FLASH_HF_NAMESPACE", "self-hoster")
    check_run_preflight()

    # and it is standalone-ONLY: the managed plane's token owns the default namespace.
    monkeypatch.delenv("FLASH_HF_NAMESPACE")
    monkeypatch.delenv(auth.STANDALONE_ENV)
    check_run_preflight()


def test_startup_rejects_an_artifact_namespace_that_cannot_form_a_repo_id(monkeypatch) -> None:
    """Presence was the whole check, so a malformed namespace boots and then fails every submit.

    `owner/repo` is the natural spelling for anyone used to HuggingFace ids, and
    `managed_hf_repo_for_environment` appends the repo name to it -- producing a three-segment id
    HuggingFace rejects while creating the artifact repo, long after preflight called the plane
    healthy.
    """
    from flash.providers.core.preflight import PreflightError, check_run_preflight

    monkeypatch.setenv("HF_TOKEN", "hf-operator-token")
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    monkeypatch.setenv("VAST_API_KEY", "vast-key")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    monkeypatch.setenv(auth.STANDALONE_ENV, "1")

    monkeypatch.setenv("FLASH_HF_NAMESPACE", "owner/repo")
    with pytest.raises(PreflightError, match="FLASH_HF_NAMESPACE"):
        check_run_preflight()

    # not just the slash: anything HuggingFace will not accept as an id segment.
    monkeypatch.setenv("FLASH_HF_NAMESPACE", "bad name")
    with pytest.raises(PreflightError, match="FLASH_HF_NAMESPACE"):
        check_run_preflight()

    # a real namespace still boots -- the guard must reject malformed values, not all of them.
    monkeypatch.setenv("FLASH_HF_NAMESPACE", "self-hoster")
    check_run_preflight()

    # and the value it accepts must actually build a valid id, which is the property that failed.
    from huggingface_hub.utils import validate_repo_id

    from flash.runner.accounting.artifacts import managed_hf_repo_for_environment

    validate_repo_id(managed_hf_repo_for_environment("github:owner/project/envs@main:gsm8k"))


def test_the_env_template_does_not_preset_the_hosted_serving_url() -> None:
    """`.env.example` is the documented starting point, and the standalone serving guard only
    rejects an UNSET value. An active assignment in the template would satisfy that guard and ship
    the plane's root credential to serve.freesolo.co on the first deploy -- the precise leak the
    guard exists to prevent."""
    import pathlib

    template = pathlib.Path(__file__).resolve().parent.parent / ".env.example"
    for lineno, line in enumerate(template.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        name, _, value = stripped.partition("=")
        if name.strip() == "FREESOLO_SERVING_URL":
            raise AssertionError(
                f".env.example:{lineno} assigns FREESOLO_SERVING_URL={value!r}; it must stay "
                "commented so a standalone plane's serving guard actually fires"
            )


def test_the_serving_header_carries_the_same_key_the_plane_authenticates(monkeypatch) -> None:
    """`authenticate` strips the operator key, so the client header must strip it too.

    A trailing newline is routine in a `.env` file. Unstripped it authenticates against the plane
    but is an ILLEGAL header value, so httpx rejects the request before it leaves; a stray space
    authenticates and then presents a different credential to the serving backend. Either way
    deploy/undeploy/chat break for a configuration the plane itself accepts."""
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key\n")
    assert serving_transport._internal_key_header() == {"X-Freesolo-Internal-Key": "operator-key"}

    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "  operator-key  ")
    header = serving_transport._internal_key_header()
    assert header == {"X-Freesolo-Internal-Key": "operator-key"}
    # the exact value the plane would accept, byte for byte.
    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    row = auth.authenticate(f"Bearer {header['X-Freesolo-Internal-Key']}")
    assert row is not None
    assert row["auth_kind"] == "internal"

    # blank collapses to NO header, matching what an unset key already does.
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "   ")
    assert serving_transport._internal_key_header() == {}


def test_a_blank_github_token_is_not_forwarded_as_a_credential(monkeypatch) -> None:
    """GitHub REJECTS a malformed bearer token rather than falling back to anonymous, so a
    whitespace-only GITHUB_TOKEN makes PUBLIC environment repos fail -- repos that load fine with
    no token at all. Every consumer must read blank as absent."""
    from flash.envs.loading import loader
    from flash.server.domain.registry import envs as server_envs

    monkeypatch.setenv("GITHUB_TOKEN", "   \n  ")
    assert loader._github_token() is None
    assert server_envs._github_token() is None
    assert "Authorization" not in loader._github_headers("application/vnd.github+json")

    # a real token still authenticates, stripped.
    monkeypatch.setenv("GITHUB_TOKEN", " ghp_real \n")
    assert loader._github_token() == "ghp_real"
    assert (
        loader._github_headers("application/vnd.github+json")["Authorization"] == "Bearer ghp_real"
    )


def test_a_blank_github_token_is_not_shipped_to_the_worker(monkeypatch) -> None:
    """The worker's git askpass branches on presence, so forwarding a blank token turns an
    anonymous public clone into an authenticated one with an invalid credential."""
    from flash.core.spec import JobSpec, TrainSpec
    from flash.providers._lifecycle.net.worker import build_worker_env

    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )

    monkeypatch.setenv("GITHUB_TOKEN", "  \t ")
    monkeypatch.setenv("HF_TOKEN", " hf_real ")
    env = build_worker_env(spec)
    assert "GITHUB_TOKEN" not in env
    # the real token still travels, stripped: this is a blank-only rejection, not a blanket one.
    assert env["HF_TOKEN"] == "hf_real"


def test_self_hosting_docs_do_not_promise_an_endpoint_concurrency_cap() -> None:
    """Flash caps nothing here, so the docs must not imply it does.

    The slot store that was meant to hold the account to 58 endpoints never enforced anything and
    has been removed; RunPod's own account limit is the only ceiling. Promising a cap invites a
    self-hoster to run bursts that RunPod, not Flash, ends up rejecting.
    """
    import pathlib

    doc = (pathlib.Path(__file__).resolve().parent.parent / "SELF_HOSTING.md").read_text()
    assert "in-process semaphore, which is the correct" not in doc
    assert "RunPod endpoint concurrency is not capped by Flash" in doc


def test_the_serving_repos_are_not_on_the_training_path() -> None:
    """The catalog names a serving checkpoint per model, separate from the trained base model.

    The FP8 serving checkpoints are public as of 2026-08-20, so this is no longer about a 401.
    It stays because the field is *serving* metadata: training loads the base model, and a training
    path that resolved `serve_model_id` would silently train against a quantized checkpoint instead
    of the model the user asked for. Public repos make that failure quieter, not less wrong.

    Asserted against the CODE, not just the doc: a doc-only check keeps passing the moment a
    future caller starts reading the field, which is exactly when the claim stops being true.
    """
    import pathlib

    from flash.core.catalog import MODELS, SERVING_MODEL_REPOS

    root = pathlib.Path(__file__).resolve().parent.parent / "flash"
    # every module that runs while a job trains -- allocation, launch, the worker itself.
    trainers = [p for d in ("engine", "runner", "providers") for p in (root / d).rglob("*.py")]
    assert trainers, "expected training-path modules to exist"
    readers = [
        str(p.relative_to(root.parent))
        for p in trainers
        if "serve_model_id" in p.read_text() or "SERVING_MODEL_REPOS" in p.read_text()
    ]
    assert readers == [], f"training path now reads the serving checkpoints: {readers}"

    # and the field is genuinely populated, so the check above is not vacuous.
    assert any(m.serving and m.serving.serve_model_id for m in MODELS.values())
    assert SERVING_MODEL_REPOS

    doc = (root.parent / "SELF_HOSTING.md").read_text()
    assert "informational\nonly" in doc or "informational only" in doc


def test_enabling_standalone_adopts_runs_already_in_the_store(monkeypatch, tmp_path) -> None:
    """Switching FLASH_STANDALONE on must not orphan the runs already recorded.

    A plane that ran managed first -- or ran on an earlier build -- has runs pointing at whichever
    key row created them. Provisioning the sentinel owner without adopting them reproduces exactly
    the bug the sentinel exists to fix: empty listing, 404 on status/logs/cancel, and an in-flight
    job still burning GPU hours that the operator's only credential cannot stop.

    Adoption is sound because standalone is SINGLE-TENANT: one principal, so every run in this
    store is already the operator's and there is no second identity to take a run away from.
    """
    from flash.server.platform import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")

    # managed first: an internal-key run and an external user-key run.
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    managed = auth.authenticate("Bearer operator-key")
    assert managed is not None
    db.record_run("run-from-managed-internal", managed["id"])
    external = db.ensure_external_key("user-key-abc", email="user@example.com")
    db.record_run("run-from-managed-external", external["id"])

    # flip to standalone against the SAME state directory.
    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    owner = auth.authenticate("Bearer operator-key")
    assert owner is not None
    assert owner["id"] != managed["id"], "the sentinel row is deliberately not the key-hash row"

    adopted = sorted(r["run_id"] for r in db.runs_for_key(owner["id"]))
    assert adopted == ["run-from-managed-external", "run-from-managed-internal"]
    assert db.run_owner("run-from-managed-internal") == owner["id"]
    assert db.run_owner("run-from-managed-external") == owner["id"]

    # idempotent: re-authenticating does not churn ownership or lose a later run.
    db.record_run("run-from-standalone", owner["id"])
    for _ in range(3):
        auth.authenticate("Bearer operator-key")
    assert len(db.runs_for_key(owner["id"])) == 3


def test_managed_mode_never_reassigns_a_users_run(monkeypatch, tmp_path) -> None:
    """The adoption above is standalone-ONLY. In managed mode two identities coexist, so pulling
    runs onto one row would hand another user's run to the internal key."""
    from flash.server.platform import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)

    internal = auth.authenticate("Bearer operator-key")
    assert internal is not None
    external = db.ensure_external_key("user-key-abc", email="user@example.com")
    db.record_run("run-internal", internal["id"])
    db.record_run("run-external", external["id"])

    for _ in range(3):
        auth.authenticate("Bearer operator-key")

    assert db.run_owner("run-external") == external["id"]
    assert [r["run_id"] for r in db.runs_for_key(internal["id"])] == ["run-internal"]
    assert [r["run_id"] for r in db.runs_for_key(external["id"])] == ["run-external"]


# ---------------------------------------------------------------------------
# Model admission does not vary by deployment
# ---------------------------------------------------------------------------
def _deps_module():
    """``_deps`` imports fastapi (the `server` extra). CI syncs it (`uv sync --extra server --dev`)
    so these tests really run there; a client-only checkout skips instead of failing on an optional
    dependency it was never expected to have."""
    pytest.importorskip("fastapi")
    from flash.server.platform import deps as _deps

    return _deps


def _environment_for(standalone_value: str | None) -> dict:
    """The ``[environment]`` block the given plane accepts.

    The two planes take disjoint environment sources -- the managed one runs hub slugs only, a
    self-hosted one runs explicit GitHub refs only -- so a test about anything ELSE has to vary this
    with the deployment or it trips the environment gate before reaching what it means to assert.
    """
    if standalone_value is None:
        return {"id": "owner/project/env"}
    return {"id": "github:owner/repo@main:env/environment.py"}


def test_the_catalog_binds_on_both_deployments(monkeypatch) -> None:
    """Self-hosting relaxes billing boundaries, not the catalog -- an uncataloged model is refused
    on either plane.

    FLASH_STANDALONE used to unlock `model_policy = "allow"`, which synthesized a ModelInfo from an
    HF size lookup: 4 of ~20 fields set, KV/attention geometry left zeroed, and the result fed into
    the VRAM equations wearing the same type as a curated entry. An operator on their own GPUs still
    gets that badly-sized run, so the flag is gone and adding a model means adding a real catalog
    entry. That makes admission a property of the model, not of the deployment.
    """
    _deps = _deps_module()
    from fastapi import HTTPException

    from tests._helpers.specs import raw_spec

    for standalone_value in ("1", None):
        if standalone_value is None:
            monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
        else:
            monkeypatch.setenv(auth.STANDALONE_ENV, standalone_value)
        # each plane's own accepted environment form: the two are disjoint, and this test is about
        # the MODEL, so it must not trip the environment gate on either side.
        payload = {
            "spec": raw_spec(
                model="meta-llama/Llama-3.1-8B",
                environment=_environment_for(standalone_value),
            )
        }
        with pytest.raises(HTTPException) as ei:
            _deps._parse_spec(payload, run_id="r")
        assert ei.value.status_code == 400
        assert "fork Flash" in str(ei.value.detail)


def test_a_catalog_run_parses_on_both_deployments(monkeypatch) -> None:
    """The counterpart: a curated run is identical on either plane, so the check above is really
    about the model and not about the parser refusing everything."""
    _deps = _deps_module()
    from tests._helpers.specs import raw_spec

    for standalone_value in ("1", None):
        if standalone_value is None:
            monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
        else:
            monkeypatch.setenv(auth.STANDALONE_ENV, standalone_value)
        spec = raw_spec(environment=_environment_for(standalone_value))
        assert _deps._parse_spec({"spec": spec}, run_id="r").model == spec["model"]


# ---------------------------------------------------------------------------
# Environment admission DOES vary by deployment
# ---------------------------------------------------------------------------
_GITHUB_ENV_FORMS: tuple[str, ...] = (
    "github:owner/repo@main:env/environment.py",
    "https://github.com/owner/repo/tree/main/env",
)


@pytest.mark.parametrize("env_id", _GITHUB_ENV_FORMS)
def test_the_managed_service_refuses_a_direct_github_environment(monkeypatch, env_id: str) -> None:
    """The hub is the only repo the managed plane can vouch for, so it is the only one it runs.

    A ``github:`` ref names a repo Freesolo has no relationship with: nothing reviewed it, nothing
    associated it with a project, and ``flash env push`` never wrote it. Accepting one would run
    unreviewed code under a Freesolo run, so the hosted plane refuses both spellings of it.
    """
    _deps = _deps_module()
    from fastapi import HTTPException

    from tests._helpers.specs import raw_spec

    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    payload = {"spec": raw_spec(environment={"id": env_id})}
    with pytest.raises(HTTPException) as ei:
        _deps._parse_spec(payload, run_id="r")
    assert ei.value.status_code == 400
    assert "hub" in str(ei.value.detail)


@pytest.mark.parametrize("env_id", _GITHUB_ENV_FORMS)
def test_a_self_hosted_plane_still_accepts_a_direct_github_environment(
    monkeypatch, env_id: str
) -> None:
    """The gate must not brick self-hosting, which is the whole reason it is a gate and not a delete.

    A standalone plane cannot publish to Freesolo's hub and a local ``path`` is rejected outright,
    so the explicit GitHub forms are its ONLY way to name an environment. If this fails, a
    self-hosted operator has no environment source at all.
    """
    _deps = _deps_module()

    from tests._helpers.specs import raw_spec

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    spec = _deps._parse_spec({"spec": raw_spec(environment={"id": env_id})}, run_id="r")
    assert spec.environment.id == env_id


def test_the_hub_slug_is_accepted_on_the_managed_plane(monkeypatch) -> None:
    """The counterpart to the refusal: the managed form works on the plane that owns the hub, so
    the check above is really about the GitHub forms and not about the hosted parser refusing
    environments at large."""
    _deps = _deps_module()
    from tests._helpers.specs import raw_spec

    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    spec = _deps._parse_spec(
        {"spec": raw_spec(environment={"id": "owner/project/env"})}, run_id="r"
    )
    assert spec.environment.id == "owner/project/env"


# every spelling that `canonical_managed_environment_slug` maps onto the private hub. a self-hosted
# plane can read none of them, so admitting any one is the same defect wearing a different syntax.
_MANAGED_HUB_FORMS: tuple[str, ...] = (
    "owner/project/env",
    "github:freesolo-co/environment-hub@main:owner/project/env/environment.py",
    "github:FREESOLO-CO/Environment-Hub@main:owner/project/env/environment.py",
    "https://github.com/freesolo-co/environment-hub/tree/main/owner/project/env",
)


@pytest.mark.parametrize("env_id", _MANAGED_HUB_FORMS)
def test_a_self_hosted_plane_refuses_an_id_naming_the_private_hub(monkeypatch, env_id: str) -> None:
    """A slug names the one repo a self-hosted plane provably cannot read.

    ``managed_slug_to_github_ref`` maps every slug onto ``freesolo-co/environment-hub``, which is
    private and hardcoded with no override. Accepted, the id passes submit, survives the
    best-effort sha pin, and fails on a rented GPU as a bare GitHub 404 naming a repo the operator
    has no relationship with -- after the run has cost money. Refusing at submit makes that failure
    free and self-explanatory. Parametrized over all four spellings because they canonicalize to
    the same unreadable repo.
    """
    _deps = _deps_module()
    from fastapi import HTTPException

    from tests._helpers.specs import raw_spec

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    with pytest.raises(HTTPException) as ei:
        _deps._parse_spec({"spec": raw_spec(environment={"id": env_id})}, run_id="r")
    assert ei.value.status_code == 400
    detail = str(ei.value.detail)
    # names the cause and the way out, not just "rejected"
    assert "freesolo-co/environment-hub" in detail
    assert "github:OWNER/REPO@REF:" in detail


def test_a_self_hosted_plane_refuses_a_malformed_ref_into_the_hub(monkeypatch) -> None:
    """A ref that targets the hub but is shaped wrong is still the unreachable repo.

    ``canonical_managed_environment_slug`` raises rather than returns for this, so the guard has to
    treat the raise as "names the hub". Reporting a shape complaint instead would send the operator
    fixing the path of a repo they were never going to be able to read.
    """
    _deps = _deps_module()
    from fastapi import HTTPException

    from tests._helpers.specs import raw_spec

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    env_id = "github:freesolo-co/environment-hub@main:too-few-segments/environment.py"
    with pytest.raises(HTTPException) as ei:
        _deps._parse_spec({"spec": raw_spec(environment={"id": env_id})}, run_id="r")
    assert ei.value.status_code == 400
    assert "freesolo-co/environment-hub" in str(ei.value.detail)


def test_a_missing_environment_id_keeps_the_schema_error_on_a_self_hosted_plane(monkeypatch):
    """The new refusal must not intercept an absent id, exactly as the hosted one does not.

    An unset ``[environment] id`` is a different mistake from naming the hub, and the schema already
    explains it. Answering "names Freesolo's managed environment hub" for an id the caller never set
    would be actively misleading.
    """
    _deps = _deps_module()
    from fastapi import HTTPException

    from tests._helpers.specs import raw_spec

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    with pytest.raises(HTTPException) as ei:
        _deps._parse_spec({"spec": raw_spec(environment={})}, run_id="r")
    assert ei.value.status_code == 400
    assert "must set [environment] id" in str(ei.value.detail)


def test_a_missing_environment_id_keeps_the_schema_error_on_the_hosted_plane(monkeypatch) -> None:
    """The gate must not intercept an absent id and answer with the wrong message.

    ``[environment] id`` missing is a different mistake from naming an unsupported source, and the
    schema already explains it (with the `flash env push` hint). Reporting "not a Freesolo
    environment id" for an id the caller never set would send them looking for a typo.
    """
    _deps = _deps_module()
    from fastapi import HTTPException

    from tests._helpers.specs import raw_spec

    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    with pytest.raises(HTTPException) as ei:
        _deps._parse_spec({"spec": raw_spec(environment={})}, run_id="r")
    assert ei.value.status_code == 400
    assert "must set [environment] id" in str(ei.value.detail)


def test_standalone_owner_manages_its_own_deployments_without_org_headers(
    monkeypatch, tmp_path
) -> None:
    """`flash models deploy` must work on the plane the operator owns outright.

    Standalone resolves its one operator credential to ``auth_kind == "internal"`` against the
    key-independent standalone-owner row, so `manageable_run` sent the plane's own run owner down
    the internal branch, which demands ``X-Freesolo-Org-Id``. The CLI's deploy, poll, and undeploy
    calls send no such header and a standalone plane has no organization directory to take one
    from, so deploying a run the caller demonstrably owned answered 404 with no way to succeed.
    """
    _deps = _deps_module()
    from flash.server.platform import db

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))

    key = auth.authenticate("Bearer operator-key")
    assert key is not None
    assert key["auth_kind"] == "internal"
    db.record_run("run-standalone", key["id"])

    sentinel = object()
    monkeypatch.setattr(_deps, "_load_status", lambda _run_id: sentinel)

    # no org/project headers, exactly as the CLI sends them
    assert _deps.manageable_run("run-standalone", key) is sentinel


def test_standalone_deployment_management_still_refuses_a_run_it_does_not_own(
    monkeypatch, tmp_path
) -> None:
    """The exact-owner path must not become a blanket bypass for the internal key."""
    _deps = _deps_module()
    from fastapi import HTTPException

    from flash.server.platform import db

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))

    key = auth.authenticate("Bearer operator-key")
    assert key is not None
    # a real second key row, so the run is genuinely owned by someone else
    other = db.ensure_internal_key("some-other-key")
    assert other["id"] != key["id"]
    db.record_run("run-someone-else", other["id"])

    monkeypatch.setattr(
        _deps,
        "_load_status",
        lambda _run_id: (_ for _ in ()).throw(
            AssertionError("must not load a run this key does not own")
        ),
    )

    with pytest.raises(HTTPException) as ei:
        _deps.manageable_run("run-someone-else", key)
    assert ei.value.status_code == 404


def test_standalone_deploy_needs_no_org_while_managed_fails_closed(monkeypatch) -> None:
    """The org fail-closed gate on deploy is a managed-plane rule.

    A standalone plane has no organization directory, so requiring an org there would make every
    deploy impossible; managed mode is where an org-unscoped adapter registration would hand the
    serving backend an unowned revision, so THAT is where the deploy must be refused.
    """
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from flash.server.routes import serving

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    serving._require_deploy_org("run-1", None)  # single-tenant: nothing to name, no rejection

    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    with pytest.raises(HTTPException) as ei:
        serving._require_deploy_org("run-1", None)
    assert ei.value.status_code == 409
    assert "owning organization" in str(ei.value.detail)
    serving._require_deploy_org("run-1", "org-1")  # managed with an org still deploys


def test_standalone_serving_scope_is_stable_across_deploy_chat_and_undeploy(monkeypatch) -> None:
    from flash.serve.deployment import deploy as serving_deploy
    from flash.server.platform.internal_client import run_serving_org_id
    from flash.server.routes import serving, serving_chat

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    monkeypatch.setenv("FREESOLO_SERVING_URL", "http://serving.test")
    status = SimpleNamespace(
        run_id="run-standalone",
        state="done",
        spec={},
        billing_context=None,
        platform_context=None,
        deployment={"state": "ready", "checkpoint_id": "run-standalone/final"},
    )
    assert auth.serving_org_id(None) == auth.STANDALONE_SERVING_ORG_ID
    assert run_serving_org_id(status) == auth.STANDALONE_SERVING_ORG_ID

    registrations: list[dict] = []
    monkeypatch.setattr(serving_deploy, "_registered_adapter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(serving_deploy, "resolve_artifact_revision", lambda _repo: "a" * 40)
    monkeypatch.setattr(
        serving_deploy.adapter_check,
        "adapter_artifact_metadata",
        lambda *_args, **_kwargs: SimpleNamespace(
            lora_rank=16,
            artifact_digest="b" * 64,
            targets_images=False,
        ),
    )
    monkeypatch.setattr(serving_deploy, "_require_serving_capabilities", lambda **_kwargs: set())
    monkeypatch.setattr(serving_deploy, "_wait_checkpoint_ready", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        serving_deploy.transport,
        "serving_request",
        lambda _method, _url, **kwargs: (
            registrations.append(kwargs) or SimpleNamespace(status_code=200)
        ),
    )
    deployment = serving_deploy.deploy_adapter(
        "run-standalone",
        "Qwen/Qwen3.5-9B",
        "org/repo",
        "sft/run-standalone",
        org_id=None,
        lora_rank=16,
    )
    assert deployment.checkpoint_id == "run-standalone/final"
    assert registrations[0]["org_id"] == auth.STANDALONE_SERVING_ORG_ID
    assert registrations[0]["json"]["org_id"] == auth.STANDALONE_SERVING_ORG_ID

    monkeypatch.setattr(serving, "manageable_run", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(
        serving._app,
        "undeploy_adapter",
        lambda checkpoint_id, *, org_id: {
            "checkpoint_id": checkpoint_id,
            "disabled_checkpoints": [checkpoint_id],
            "serving_deregistered": True,
            "org_id": org_id,
        },
    )
    monkeypatch.setattr(serving, "mark_undeployed", lambda *_args: status)
    monkeypatch.setattr(serving, "_report_persisted_transition", lambda *_args, **_kwargs: None)
    undeployed = serving.undeploy(
        "run-standalone",
        "run-standalone/final",
        {"id": 1, "auth_kind": "internal"},
    )
    assert undeployed["disabled_checkpoints"] == ["run-standalone/final"]

    monkeypatch.setattr(serving_chat, "manageable_run", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(
        serving_chat, "_verified_checkpoints", lambda _status: {"run-standalone/final"}
    )
    monkeypatch.setattr(
        serving_chat,
        "effective_spec_from_status",
        lambda _status: SimpleNamespace(
            model="Qwen/Qwen3.5-9B",
            train=SimpleNamespace(hf_repo="org/repo", stop_sequences=()),
            thinking=False,
        ),
    )
    _request, _messages, _spec, _checkpoint, org_id = serving_chat._resolve_chat_request(
        "run-standalone",
        {
            "checkpoint_id": "run-standalone/final",
            "messages": [{"role": "user", "content": "hi"}],
        },
        {"id": 1, "auth_kind": "internal"},
        None,
        None,
    )
    assert org_id == auth.STANDALONE_SERVING_ORG_ID


def test_managed_serving_scope_never_synthesizes_an_org(monkeypatch) -> None:
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert auth.serving_org_id(None) == ""
    assert auth.serving_org_id("org-1") == "org-1"


def test_standalone_deployment_listing_stays_exact_key_scoped(monkeypatch) -> None:
    """Standalone keeps the unscoped exact-key listing its operator CLI sends (no org headers)."""
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from flash.server.routes import serving

    internal_key = {"id": 1, "auth_kind": "internal"}

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    assert serving._deployment_listing_scope(internal_key, None, None) is None

    # managed mode: the internal key must name its scope; a user key stays key-scoped
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    with pytest.raises(HTTPException) as ei:
        serving._deployment_listing_scope(internal_key, None, None)
    assert ei.value.status_code == 400
    project = "11111111-1111-4111-8111-111111111111"
    assert serving._deployment_listing_scope(internal_key, "org-1", project) == ("org-1", project)
    assert (
        serving._deployment_listing_scope({"id": 2, "auth_kind": "freesolo_api_key"}, None, None)
        is None
    )

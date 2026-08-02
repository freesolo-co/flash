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


def test_standalone_does_not_reach_for_the_shared_slot_store(monkeypatch) -> None:
    """The shared RunPod slot store is a backend table, so standalone must not try to lease from it.

    That is all this pins. It does NOT mean concurrency is capped instead: the in-process semaphore
    behind the store is claimed from `get_train_endpoint`, which the live deploy path replaced, so
    neither enforces the ceiling (see the note at RUNPOD_ENDPOINT_SLOT_CAP)."""
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


def test_standalone_run_ownership_survives_operator_key_rotation(monkeypatch, tmp_path) -> None:
    """Rotating FREESOLO_INTERNAL_KEY must not orphan the runs the old key started.

    Standalone is single-tenant, so the operator key owns every run. Deriving the owner row from
    the key's HASH meant a rotation (or a re-run of the quickstart's `openssl rand`) minted a new
    row with a new id, and every run -- matched by `runs.key_id` -- vanished: absent from the
    listing, 404 on status/logs/cancel. An in-flight job would keep spending with no supported way
    for the new credential to stop it, so rotating a COMPROMISED key was the thing that cost you
    control of the plane.
    """
    from flash.server import db

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
    from flash.server import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    owner = db.ensure_standalone_owner()

    assert db.lookup_key(db._STANDALONE_OWNER_HASH) is None
    assert db.lookup_key("standalone-operator") is None
    # idempotent: a second call returns the SAME row, never a second owner.
    assert db.ensure_standalone_owner()["id"] == owner["id"]


def test_standalone_startup_requires_a_writable_artifact_namespace(monkeypatch) -> None:
    """Without FLASH_HF_NAMESPACE a self-hoster's artifacts default to a namespace their HF_TOKEN
    cannot write to, so every run dies at artifact upload -- AFTER preflight called the plane
    healthy. Fail at startup, where the operator can act on it."""
    from flash.providers.preflight import PreflightError, check_run_preflight

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
    from flash.serve import deploy

    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key\n")
    assert deploy._internal_key_header() == {"X-Freesolo-Internal-Key": "operator-key"}

    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "  operator-key  ")
    header = deploy._internal_key_header()
    assert header == {"X-Freesolo-Internal-Key": "operator-key"}
    # the exact value the plane would accept, byte for byte.
    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    row = auth.authenticate(f"Bearer {header['X-Freesolo-Internal-Key']}")
    assert row is not None
    assert row["auth_kind"] == "internal"

    # blank collapses to NO header, matching what an unset key already does.
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "   ")
    assert deploy._internal_key_header() == {}


def test_a_blank_github_token_is_not_forwarded_as_a_credential(monkeypatch) -> None:
    """GitHub REJECTS a malformed bearer token rather than falling back to anonymous, so a
    whitespace-only GITHUB_TOKEN makes PUBLIC environment repos fail -- repos that load fine with
    no token at all. Every consumer must read blank as absent."""
    from flash.envs import loader
    from flash.server import envs as server_envs

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
    from flash.providers._worker import build_worker_env
    from flash.spec import JobSpec, TrainSpec

    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=10, hf_repo="owner/runs"),
        seed=0,
    )

    monkeypatch.setenv("GITHUB_TOKEN", "  \t ")
    monkeypatch.setenv("HF_TOKEN", " hf_real ")
    env = build_worker_env(spec, 0)
    assert "GITHUB_TOKEN" not in env
    # the real token still travels, stripped: this is a blank-only rejection, not a blanket one.
    assert env["HF_TOKEN"] == "hf_real"


def test_self_hosting_docs_do_not_promise_an_endpoint_concurrency_cap() -> None:
    """The slot store and its in-process fallback are both claimed from `get_train_endpoint`, which
    the live `jobs.py::_deploy_once` path replaced. Telling a self-hoster the semaphore is "the
    correct cap" invites them to run bursts that RunPod, not Flash, ends up rejecting."""
    import pathlib

    doc = (pathlib.Path(__file__).resolve().parent.parent / "SELF_HOSTING.md").read_text()
    assert "in-process semaphore, which is the correct" not in doc
    assert "RunPod endpoint concurrency is not capped by Flash" in doc


def test_the_private_fp8_serving_repos_are_not_on_the_training_path() -> None:
    """The catalog names a per-model FP8 repo under the managed namespace, and most are PRIVATE.

    A self-hoster's HF_TOKEN cannot read them, so if anything on the training path resolved
    `serve_model_id` the run would die on a 401 against a repo they can neither see nor fix --
    and the catalog would be advertising a model Flash cannot actually train. The field is
    serving metadata; this pins that it stays that way.

    Asserted against the CODE, not just the doc: a doc-only check keeps passing the moment a
    future caller starts reading the field, which is exactly when the claim stops being true.
    """
    import pathlib

    from flash.catalog import MODELS, SERVING_FP8_MODEL_REPOS

    root = pathlib.Path(__file__).resolve().parent.parent / "flash"
    # every module that runs while a job trains -- allocation, launch, the worker itself.
    trainers = [p for d in ("engine", "runner", "providers") for p in (root / d).rglob("*.py")]
    assert trainers, "expected training-path modules to exist"
    readers = [
        str(p.relative_to(root.parent))
        for p in trainers
        if "serve_model_id" in p.read_text() or "SERVING_FP8_MODEL_REPOS" in p.read_text()
    ]
    assert readers == [], f"training path now reads the private FP8 serving repos: {readers}"

    # and the field is genuinely populated, so the check above is not vacuous.
    assert any(m.serving and m.serving.serve_model_id for m in MODELS.values())
    assert SERVING_FP8_MODEL_REPOS

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
    from flash.server import db

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
    from flash.server import db

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

"""Env resolution: registry user-param preservation + the control-plane resolve-once package hook.

  - registry.load_environment forwards a user [environment.params] entry verbatim to the SDK loader
    and threads the control-plane Azure hints (resolved_package_url + resolved_sha) out-of-band as
    POSITIONAL-ONLY args, so a user param of the same name never collides with them.
  - runner._assign_resolved_env_pkg resolves a managed slug ONCE to an Azure Blob SAS URL + package
    SHA and pins them on the spec; any failure leaves them empty (the worker then fails fast — there
    is no GitHub fallback).
"""

# ----------------------------- registry: user-param preservation ---------------------------------


def _fake_loader(captured):
    # Mirrors the real signature: the two hints are POSITIONAL-ONLY (the `/`), so a user param of
    # the same name lands in **kwargs instead of binding to the internal hints.
    def loader(env_id, resolved_package_url=None, resolved_sha=None, /, **kwargs):
        captured["env_id"] = env_id
        captured["resolved_package_url"] = resolved_package_url
        captured["resolved_sha"] = resolved_sha
        captured["kwargs"] = kwargs
        return object()

    return loader


def test_registry_preserves_user_params_and_threads_hints_out_of_band(monkeypatch):
    import flash.envs.adapter as adapter
    from flash.envs.registry import load_environment

    captured = {}
    monkeypatch.setattr(adapter, "load_freesolo_environment", _fake_loader(captured))

    load_environment(
        "owner/env",
        params={"resolved_sha": "user-value", "difficulty": "hard"},
        resolved_package_url="https://blob/sas",
        resolved_sha="a" * 64,
    )

    # The user's freeform params are forwarded verbatim (NOT stripped)...
    assert captured["kwargs"]["resolved_sha"] == "user-value"
    assert captured["kwargs"]["difficulty"] == "hard"
    # ...and the control-plane hints ride out-of-band as the positional-only args, no collision.
    assert captured["resolved_package_url"] == "https://blob/sas"
    assert captured["resolved_sha"] == "a" * 64


def test_registry_user_param_named_like_hint_is_forwarded_not_consumed(monkeypatch):
    import flash.envs.adapter as adapter
    from flash.envs.registry import load_environment

    captured = {}
    monkeypatch.setattr(adapter, "load_freesolo_environment", _fake_loader(captured))

    load_environment(
        "owner/env",
        params={"resolved_package_url": "user-data"},
        resolved_package_url="https://blob/sas",
    )

    assert captured["resolved_package_url"] == "https://blob/sas"  # control-plane hint (positional)
    assert captured["kwargs"]["resolved_package_url"] == "user-data"  # user's param -> SDK


def test_registry_omits_hints_when_unset(monkeypatch):
    import flash.envs.adapter as adapter
    from flash.envs.registry import load_environment

    captured = {}
    monkeypatch.setattr(adapter, "load_freesolo_environment", _fake_loader(captured))
    load_environment("owner/env", params={"difficulty": "hard"})
    assert captured["resolved_package_url"] is None
    assert captured["resolved_sha"] is None
    assert captured["kwargs"] == {"difficulty": "hard"}


# ----------------------------- runner: resolve-once Azure package hook ----------------------------

_ENV = "owner/env"


def _record(sha="b" * 64):
    from flash.server.environment_store import EnvironmentRecord

    return EnvironmentRecord(
        slug=_ENV,
        namespace="owner",
        name="env",
        blob_container="flash-environments",
        blob_key="flash-envs/owner/env/package.tar.gz",
        package_sha256=sha,
        size_bytes=10,
        version=1,
    )


def test_assign_resolves_when_indexed(monkeypatch):
    from flash import runner
    from flash.server import azure_blob, environment_store
    from flash.spec import EnvironmentSpec, JobSpec

    monkeypatch.setattr(environment_store, "lookup", lambda slug: _record())
    monkeypatch.setattr(azure_blob, "read_sas_url", lambda blob_key, **k: "https://blob/sas")

    out = runner._assign_resolved_env_pkg(JobSpec(environment=EnvironmentSpec(id=_ENV)))
    assert out.environment.resolved_package_url == "https://blob/sas"
    assert out.environment.resolved_sha == "b" * 64
    assert out.environment.id == _ENV  # untouched fields survive the rebuild


def test_assign_best_effort_on_failure(monkeypatch):
    from flash import runner
    from flash.server import environment_store
    from flash.spec import EnvironmentSpec, JobSpec

    def boom(_slug):
        raise RuntimeError("azure down")

    monkeypatch.setattr(environment_store, "lookup", boom)
    out = runner._assign_resolved_env_pkg(JobSpec(environment=EnvironmentSpec(id=_ENV)))
    assert out.environment.resolved_package_url == ""  # submission never blocks on Azure


def test_assign_noop_when_not_indexed(monkeypatch):
    from flash import runner
    from flash.server import environment_store
    from flash.spec import EnvironmentSpec, JobSpec

    monkeypatch.setattr(environment_store, "lookup", lambda _slug: None)
    out = runner._assign_resolved_env_pkg(JobSpec(environment=EnvironmentSpec(id=_ENV)))
    assert out.environment.resolved_package_url == ""


def test_assign_noop_without_env_or_already_resolved():
    from flash import runner
    from flash.spec import EnvironmentSpec, JobSpec

    # No env id.
    assert runner._assign_resolved_env_pkg(JobSpec()).environment.resolved_package_url == ""
    # Already resolved.
    pinned = JobSpec(environment=EnvironmentSpec(id=_ENV, resolved_package_url="https://x"))
    assert (
        runner._assign_resolved_env_pkg(pinned).environment.resolved_package_url == "https://x"
    )


def test_assign_noop_for_non_managed_id(monkeypatch):
    from flash import runner
    from flash.server import environment_store
    from flash.spec import EnvironmentSpec, JobSpec

    def boom(_slug):
        raise AssertionError("lookup must not be called for a non-managed id")

    monkeypatch.setattr(environment_store, "lookup", boom)
    # A value with a ':' is not a managed slug -> nothing to resolve, no lookup.
    out = runner._assign_resolved_env_pkg(JobSpec(environment=EnvironmentSpec(id="not:a/slug")))
    assert out.environment.resolved_package_url == ""

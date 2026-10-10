import flash.providers._lifecycle.net.worker as provider_worker
import flash.runner.accounting.artifacts as runner_artifacts
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
import flash.runner.supervise.lifecycle as runner_lifecycle

"""Tests for the combined env ref->SHA rate-limit handling (#209) + resolve-once pin (#214).

Covers the review feedback on both PRs:
  - #209: _urlopen raises the typed GitHubRateLimitError on 429 / rate-limit 403, and a plain
    RuntimeError on a non-rate-limit 403 (so only rate limits are reclassified as retriable).
  - #214 (comment 1): runner_artifacts._assign_resolved_env_sha resolves the env ref->sha once and pins it
    on the spec; failures leave it empty for authoritative controller staging.
  - #214 (comment 2): registry.load_environment forwards a user [environment.params] entry named
    "resolved_sha" verbatim to the SDK loader and threads the control-plane pin under the reserved
    `pinned_sha` kwarg instead of stripping the user value.
"""

import io
import urllib.error
import urllib.request

import pytest

from tests._helpers.source_snapshot import valid_source_snapshot


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.github.com/x", code, "err", {}, io.BytesIO(body.encode())
    )


def _patch_no_sleep(monkeypatch):
    # _urlopen does `import time; time.sleep(...)`; patch the module attr so the retry loop is fast.
    import time

    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)


def test_urlopen_raises_typed_error_on_429(monkeypatch):
    from flash.envs.loading.loader import GitHubRateLimitError, _urlopen

    _patch_no_sleep(monkeypatch)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(429, "slow down")),
    )
    with pytest.raises(GitHubRateLimitError):
        _urlopen(urllib.request.Request("https://api.github.com/x"))


def test_urlopen_raises_typed_error_on_rate_limit_403(monkeypatch):
    from flash.envs.loading.loader import GitHubRateLimitError, _urlopen

    _patch_no_sleep(monkeypatch)
    body = '{"message": "API rate limit exceeded for user ID 1"}'
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(_http_error(403, body))
    )
    with pytest.raises(GitHubRateLimitError):
        _urlopen(urllib.request.Request("https://api.github.com/x"))


@pytest.mark.parametrize("code", [401, 403])
def test_urlopen_treats_credential_failure_as_permanent(monkeypatch, code):
    # a 401, or a 403 that is not a rate limit, is a token this plane cannot fix by waiting. it must
    # be non-retriable (so the run does not loop on a fresh worker) AND typed permanent, so the
    # submit-time preflight fails closed instead of deferring the same error past gpu allocation.
    from flash.envs.loading.loader import GitHubPermanentError, GitHubRateLimitError, _urlopen

    _patch_no_sleep(monkeypatch)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(code, '{"message": "Bad credentials"}')),
    )
    with pytest.raises(GitHubPermanentError) as exc:
        _urlopen(urllib.request.Request("https://api.github.com/x"))
    assert not isinstance(exc.value, GitHubRateLimitError)


def test_urlopen_rate_limit_403_stays_transient_not_permanent(monkeypatch):
    # the rate-limit 403 is claimed before the credential branch: it is a quota to wait out, not a
    # bad token, so widening the permanent set must not swallow it.
    from flash.envs.loading.loader import GitHubPermanentError, GitHubRateLimitError, _urlopen

    _patch_no_sleep(monkeypatch)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(
            _http_error(403, '{"message": "API rate limit exceeded"}')
        ),
    )
    with pytest.raises(GitHubRateLimitError) as exc:
        _urlopen(urllib.request.Request("https://api.github.com/x"))
    assert not isinstance(exc.value, GitHubPermanentError)


@pytest.mark.parametrize("code", [404, 422])
def test_urlopen_raises_permanent_error_on_settled_github_answer(monkeypatch, code):
    from flash.envs.loading.loader import GitHubPermanentError, _urlopen

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(code, '{"message": "Not Found"}')),
    )

    with pytest.raises(GitHubPermanentError, match=rf"\({code}\)"):
        _urlopen(urllib.request.Request("https://api.github.com/x"), max_rate_limit_retries=0)


def test_urlopen_raises_unavailable_error_on_server_failure(monkeypatch):
    from flash.envs.loading.loader import GitHubUnavailableError, _urlopen

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(503, "service unavailable")),
    )

    with pytest.raises(GitHubUnavailableError, match="503"):
        _urlopen(urllib.request.Request("https://api.github.com/x"), max_rate_limit_retries=0)


def test_urlopen_success_returns_bytes(monkeypatch):
    from flash.envs.loading.loader import _urlopen

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"payload"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert _urlopen(urllib.request.Request("https://api.github.com/x")) == b"payload"


def test_rate_limit_error_is_retriable_runtime_error():
    # The worker's top-level handler treats GitHubRateLimitError as retriable precisely because it
    # is a RuntimeError subclass caught by `isinstance(e, (RetriableInfraError, GitHubRateLimitError))`.
    from flash.envs.loading.adapter import GitHubRateLimitError

    assert issubclass(GitHubRateLimitError, RuntimeError)
    assert isinstance(GitHubRateLimitError("x"), GitHubRateLimitError)


# ----------------------------- registry: user-param preservation (#214 comment 2) -----------------


def _fake_loader(captured):
    # Mirrors the real signature: pinned_sha is POSITIONAL-ONLY (the `/`), so a user param named
    # "pinned_sha" lands in **kwargs instead of binding to the internal pin.
    def loader(env_id, pinned_sha=None, /, **kwargs):
        captured["env_id"] = env_id
        captured["pinned_sha"] = pinned_sha
        captured["kwargs"] = kwargs
        return object()

    return loader


def test_registry_preserves_user_params_and_pins_out_of_band(monkeypatch):
    import flash.envs.loading.adapter as adapter
    from flash.envs.loading.base import load_environment

    captured = {}
    monkeypatch.setattr(adapter, "load_freesolo_environment", _fake_loader(captured))

    load_environment(
        "owner/env",
        params={"resolved_sha": "user-value", "difficulty": "hard"},
        resolved_sha="a" * 40,
    )

    # The user's freeform params are forwarded verbatim (NOT stripped)...
    assert captured["kwargs"]["resolved_sha"] == "user-value"
    assert captured["kwargs"]["difficulty"] == "hard"
    # ...and the control-plane pin rides out-of-band as the positional-only arg, no collision.
    assert captured["pinned_sha"] == "a" * 40


def test_registry_user_pinned_sha_param_is_forwarded_not_consumed(monkeypatch):
    # Even a user param literally named "pinned_sha" must reach the SDK loader untouched while the
    # control-plane pin stays separate (the whole point of making it positional-only).
    import flash.envs.loading.adapter as adapter
    from flash.envs.loading.base import load_environment

    captured = {}
    monkeypatch.setattr(adapter, "load_freesolo_environment", _fake_loader(captured))

    load_environment("owner/env", params={"pinned_sha": "user-data"}, resolved_sha="a" * 40)

    assert captured["pinned_sha"] == "a" * 40  # control-plane pin (positional)
    assert captured["kwargs"]["pinned_sha"] == "user-data"  # user's param, forwarded to the SDK


def test_registry_omits_pin_when_unset(monkeypatch):
    import flash.envs.loading.adapter as adapter
    from flash.envs.loading.base import load_environment

    captured = {}
    monkeypatch.setattr(adapter, "load_freesolo_environment", _fake_loader(captured))
    load_environment("owner/env", params={"difficulty": "hard"})
    assert captured["pinned_sha"] is None
    assert captured["kwargs"] == {"difficulty": "hard"}


# ----------------------------- runner: resolve-once pin (#214 comment 1) --------------------------

_GH_ENV = "github:owner/repo@main:env/environment.py"


def test_assign_resolved_env_sha_pins_when_resolver_succeeds(monkeypatch):
    import flash.envs.loading.loader as adapter
    from flash.core.spec import EnvironmentSpec, JobSpec

    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, *a, **k: "b" * 40)
    spec = JobSpec(environment=EnvironmentSpec(id=_GH_ENV))
    out = runner_artifacts._assign_resolved_env_sha(spec)
    assert out.environment.resolved_sha == "b" * 40
    # Untouched fields survive the rebuild.
    assert out.environment.id == _GH_ENV


def test_assign_resolved_env_sha_uses_fast_no_retry_resolver(monkeypatch):
    # the control-plane preflight must never block run creation on github retries: it resolves with a
    # short timeout and zero rate-limit retries before authoritative controller staging.
    import flash.envs.loading.loader as adapter
    from flash.core.spec import EnvironmentSpec, JobSpec

    seen = {}

    def fake_resolve(parsed, pinned_sha=None, *, timeout=60.0, max_rate_limit_retries=5):
        seen["timeout"] = timeout
        seen["max_rate_limit_retries"] = max_rate_limit_retries
        return "d" * 40

    monkeypatch.setattr(adapter, "_resolve_ref_sha", fake_resolve)
    runner_artifacts._assign_resolved_env_sha(JobSpec(environment=EnvironmentSpec(id=_GH_ENV)))
    assert seen["max_rate_limit_retries"] == 0
    assert seen["timeout"] <= 15.0


def test_assign_resolved_env_sha_best_effort_on_failure(monkeypatch):
    import flash.envs.loading.loader as adapter
    from flash.core.spec import EnvironmentSpec, JobSpec

    def boom(*a, **k):
        raise RuntimeError("github down")

    monkeypatch.setattr(adapter, "_resolve_ref_sha", boom)
    spec = JobSpec(environment=EnvironmentSpec(id=_GH_ENV))
    out = runner_artifacts._assign_resolved_env_sha(spec)
    assert out.environment.resolved_sha == ""  # controller staging resolves before allocation


def test_assign_resolved_env_sha_noop_without_env_or_already_pinned(monkeypatch):
    import flash.envs.loading.loader as adapter
    from flash.core.spec import EnvironmentSpec, JobSpec

    # Must never touch the network when there is nothing to resolve.
    def boom(*a, **k):
        raise AssertionError("resolver must not be called")

    monkeypatch.setattr(adapter, "_resolve_ref_sha", boom)
    # No env id.
    assert runner_artifacts._assign_resolved_env_sha(JobSpec()).environment.resolved_sha == ""
    # Already pinned.
    pinned = JobSpec(environment=EnvironmentSpec(id=_GH_ENV, resolved_sha="c" * 40))
    assert runner_artifacts._assign_resolved_env_sha(pinned).environment.resolved_sha == "c" * 40


def test_background_submit_keeps_environment_staging_off_creation_path(monkeypatch, tmp_path):
    import threading

    from flash.core.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))

    main_thread = threading.current_thread().ident
    staging_threads: list[int | None] = []
    ran = threading.Event()

    def fake_stage(spec, **_kwargs):
        staging_threads.append(threading.current_thread().ident)
        return spec

    def fake_run_job(spec, **_kwargs):
        runner_artifacts.stage_environment_package(spec)
        ran.set()

    monkeypatch.setattr(runner_artifacts, "stage_environment_package", fake_stage)
    monkeypatch.setattr(runner_lifecycle, "_run_job", fake_run_job)
    monkeypatch.setattr(
        provider_worker,
        "publish_source_snapshot",
        lambda _repo=None: valid_source_snapshot(),
    )
    # the submit-time 404 gate resolves the ref too, and this spec names a placeholder repo that
    # really does 404. its subject is WHICH THREAD pins, so stub the gate out rather than let a live
    # GitHub lookup decide whether the test runs. pass the spec through unpinned: the gate returns
    # what it resolved, and returning a pinned spec here would answer the question under test.
    monkeypatch.setattr(
        runner_artifacts,
        "preflight_validate_environment_ref",
        lambda spec: (spec, False),
    )

    spec = JobSpec(
        run_id="flash-bg-resolve",
        model="Qwen/Qwen3.5-9B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=1),
        gpu=GpuSpec(type=""),
        environment=EnvironmentSpec(id=_GH_ENV),
    )
    status = runner_submit.submit_job(spec, background=True)

    assert status.run_id == "flash-bg-resolve"
    assert runner_status.get_status("flash-bg-resolve").state == "queued"
    assert main_thread not in staging_threads
    assert ran.wait(timeout=5.0)
    assert staging_threads
    assert all(thread_id != main_thread for thread_id in staging_threads)

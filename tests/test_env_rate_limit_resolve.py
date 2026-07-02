"""Tests for the combined env ref->SHA rate-limit handling (#209) + resolve-once pin (#214).

Covers the review feedback on both PRs:
  - #209: _urlopen raises the typed GitHubRateLimitError on 429 / rate-limit 403, and a plain
    RuntimeError on a non-rate-limit 403 (so only rate limits are reclassified as retriable).
  - #214 (comment 1): runner._assign_resolved_env_sha resolves the env ref->sha ONCE and pins it
    on the spec; failures leave it empty (worker falls back).
  - #214 (comment 2): registry.load_environment forwards a user [environment.params] entry named
    "resolved_sha" verbatim to the SDK loader and threads the control-plane pin under the reserved
    `pinned_sha` kwarg instead of stripping the user value.
"""

import io
import urllib.error
import urllib.request

import pytest


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.github.com/x", code, "err", {}, io.BytesIO(body.encode())
    )


def _patch_no_sleep(monkeypatch):
    # _urlopen does `import time; time.sleep(...)`; patch the module attr so the retry loop is fast.
    import time

    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)


def test_urlopen_raises_typed_error_on_429(monkeypatch):
    from flash.envs.loader import GitHubRateLimitError, _urlopen

    _patch_no_sleep(monkeypatch)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(_http_error(429, "slow down"))
    )
    with pytest.raises(GitHubRateLimitError):
        _urlopen(urllib.request.Request("https://api.github.com/x"))


def test_urlopen_raises_typed_error_on_rate_limit_403(monkeypatch):
    from flash.envs.loader import GitHubRateLimitError, _urlopen

    _patch_no_sleep(monkeypatch)
    body = '{"message": "API rate limit exceeded for user ID 1"}'
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(_http_error(403, body))
    )
    with pytest.raises(GitHubRateLimitError):
        _urlopen(urllib.request.Request("https://api.github.com/x"))


def test_urlopen_non_rate_limit_403_stays_plain_runtime_error(monkeypatch):
    # A 403 that is NOT a rate limit (auth failure, repo not found) must stay a plain RuntimeError
    # (non-retriable) so the run fails fast instead of looping on a fresh worker forever.
    from flash.envs.loader import GitHubRateLimitError, _urlopen

    _patch_no_sleep(monkeypatch)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(403, '{"message": "Bad credentials"}')),
    )
    with pytest.raises(RuntimeError) as exc:
        _urlopen(urllib.request.Request("https://api.github.com/x"))
    assert not isinstance(exc.value, GitHubRateLimitError)


def test_urlopen_success_returns_bytes(monkeypatch):
    from flash.envs.loader import _urlopen

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
    from flash.envs.adapter import GitHubRateLimitError

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
    import flash.envs.adapter as adapter
    from flash.envs.registry import load_environment

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
    import flash.envs.adapter as adapter
    from flash.envs.registry import load_environment

    captured = {}
    monkeypatch.setattr(adapter, "load_freesolo_environment", _fake_loader(captured))

    load_environment("owner/env", params={"pinned_sha": "user-data"}, resolved_sha="a" * 40)

    assert captured["pinned_sha"] == "a" * 40  # control-plane pin (positional)
    assert captured["kwargs"]["pinned_sha"] == "user-data"  # user's param, forwarded to the SDK


def test_registry_omits_pin_when_unset(monkeypatch):
    import flash.envs.adapter as adapter
    from flash.envs.registry import load_environment

    captured = {}
    monkeypatch.setattr(adapter, "load_freesolo_environment", _fake_loader(captured))
    load_environment("owner/env", params={"difficulty": "hard"})
    assert captured["pinned_sha"] is None
    assert captured["kwargs"] == {"difficulty": "hard"}


# ----------------------------- runner: resolve-once pin (#214 comment 1) --------------------------

_GH_ENV = "github:owner/repo@main:env/environment.py"


def test_assign_resolved_env_sha_pins_when_resolver_succeeds(monkeypatch):
    import flash.envs.loader as adapter
    from flash import runner
    from flash.spec import EnvironmentSpec, JobSpec

    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, *a, **k: "b" * 40)
    spec = JobSpec(environment=EnvironmentSpec(id=_GH_ENV))
    out = runner._assign_resolved_env_sha(spec)
    assert out.environment.resolved_sha == "b" * 40
    # Untouched fields survive the rebuild.
    assert out.environment.id == _GH_ENV


def test_assign_resolved_env_sha_uses_fast_no_retry_resolver(monkeypatch):
    # The control-plane pin must never block run creation on GitHub retries: it resolves with a
    # short timeout and zero rate-limit retries (the worker keeps the full retry budget).
    import flash.envs.loader as adapter
    from flash import runner
    from flash.spec import EnvironmentSpec, JobSpec

    seen = {}

    def fake_resolve(parsed, pinned_sha=None, *, timeout=60.0, max_rate_limit_retries=5):
        seen["timeout"] = timeout
        seen["max_rate_limit_retries"] = max_rate_limit_retries
        return "d" * 40

    monkeypatch.setattr(adapter, "_resolve_ref_sha", fake_resolve)
    runner._assign_resolved_env_sha(JobSpec(environment=EnvironmentSpec(id=_GH_ENV)))
    assert seen["max_rate_limit_retries"] == 0
    assert seen["timeout"] <= 15.0


def test_assign_resolved_env_sha_best_effort_on_failure(monkeypatch):
    import flash.envs.loader as adapter
    from flash import runner
    from flash.spec import EnvironmentSpec, JobSpec

    def boom(*a, **k):
        raise RuntimeError("github down")

    monkeypatch.setattr(adapter, "_resolve_ref_sha", boom)
    spec = JobSpec(environment=EnvironmentSpec(id=_GH_ENV))
    out = runner._assign_resolved_env_sha(spec)
    assert out.environment.resolved_sha == ""  # submission never blocks; worker resolves itself


def test_assign_resolved_env_sha_noop_without_env_or_already_pinned(monkeypatch):
    import flash.envs.loader as adapter
    from flash import runner
    from flash.spec import EnvironmentSpec, JobSpec

    # Must never touch the network when there is nothing to resolve.
    def boom(*a, **k):
        raise AssertionError("resolver must not be called")

    monkeypatch.setattr(adapter, "_resolve_ref_sha", boom)
    # No env id.
    assert runner._assign_resolved_env_sha(JobSpec()).environment.resolved_sha == ""
    # Already pinned.
    pinned = JobSpec(environment=EnvironmentSpec(id=_GH_ENV, resolved_sha="c" * 40))
    assert runner._assign_resolved_env_sha(pinned).environment.resolved_sha == "c" * 40


def test_background_submit_defers_env_sha_off_creation_path(monkeypatch, tmp_path):
    """#217: submit_job(background=True) (the managed API path) must NOT resolve the env ref->sha
    synchronously — the GitHub commits API is on a thread, so a slow/rate-limited resolve cannot
    block or delay run creation. The status is saved + reported FIRST, then the pin is deferred into
    the background run thread."""
    import threading

    from flash import runner
    from flash.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))

    main_thread = threading.current_thread().ident
    resolve_threads: list[int | None] = []
    ran = threading.Event()

    def fake_resolve(spec):
        # Record WHICH thread asked for the env-sha pin (creation path vs background thread).
        resolve_threads.append(threading.current_thread().ident)
        return spec

    def fake_run_job(spec, **kwargs):
        ran.set()

    monkeypatch.setattr(runner, "_assign_resolved_env_sha", fake_resolve)
    monkeypatch.setattr(runner, "_run_job", fake_run_job)

    spec = JobSpec(
        run_id="flash-bg-resolve",
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(steps=1),
        gpu=GpuSpec(type="RTX 4090"),
        environment=EnvironmentSpec(id=_GH_ENV),
    )
    status = runner.submit_job(spec, background=True)

    # Run creation returned immediately with a persisted queued record...
    assert status.run_id == "flash-bg-resolve"
    assert runner.get_status("flash-bg-resolve").state == "queued"
    # ...and the env-sha resolve had NOT run on the creating (request) thread by the time we returned.
    assert main_thread not in resolve_threads

    # The background thread resolves the pin (off the critical path) and then runs the job.
    assert ran.wait(timeout=5.0)
    assert resolve_threads, "background thread must resolve the env ref->sha"
    assert all(tid != main_thread for tid in resolve_threads)

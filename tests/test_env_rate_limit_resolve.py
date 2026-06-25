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
    from flash.envs.adapter import GitHubRateLimitError, _urlopen

    _patch_no_sleep(monkeypatch)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(_http_error(429, "slow down"))
    )
    with pytest.raises(GitHubRateLimitError):
        _urlopen(urllib.request.Request("https://api.github.com/x"))


def test_urlopen_raises_typed_error_on_rate_limit_403(monkeypatch):
    from flash.envs.adapter import GitHubRateLimitError, _urlopen

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
    from flash.envs.adapter import GitHubRateLimitError, _urlopen

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
    from flash.envs.adapter import _urlopen

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


def test_registry_preserves_user_resolved_sha_param_and_pins_under_reserved_name(monkeypatch):
    import flash.envs.adapter as adapter
    from flash.envs.registry import load_environment

    captured = {}

    def fake_loader(env_id, **kwargs):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(adapter, "load_freesolo_environment", fake_loader)

    load_environment(
        "owner/env",
        params={"resolved_sha": "user-value", "difficulty": "hard"},
        resolved_sha="a" * 40,
    )

    # The user's freeform param is forwarded verbatim (NOT stripped)...
    assert captured["kwargs"]["resolved_sha"] == "user-value"
    assert captured["kwargs"]["difficulty"] == "hard"
    # ...and the control-plane pin rides under the reserved internal name, no collision.
    assert captured["kwargs"]["pinned_sha"] == "a" * 40


def test_registry_omits_pin_when_unset(monkeypatch):
    import flash.envs.adapter as adapter
    from flash.envs.registry import load_environment

    captured = {}
    monkeypatch.setattr(
        adapter,
        "load_freesolo_environment",
        lambda env_id, **kwargs: captured.setdefault("kwargs", kwargs) or object(),
    )
    load_environment("owner/env", params={"difficulty": "hard"})
    assert "pinned_sha" not in captured["kwargs"]
    assert captured["kwargs"] == {"difficulty": "hard"}


# ----------------------------- runner: resolve-once pin (#214 comment 1) --------------------------

_GH_ENV = "github:owner/repo@main:env/environment.py"


def test_assign_resolved_env_sha_pins_when_resolver_succeeds(monkeypatch):
    import flash.envs.adapter as adapter
    from flash import runner
    from flash.spec import EnvironmentSpec, JobSpec

    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, *a, **k: "b" * 40)
    spec = JobSpec(environment=EnvironmentSpec(id=_GH_ENV))
    out = runner._assign_resolved_env_sha(spec)
    assert out.environment.resolved_sha == "b" * 40
    # Untouched fields survive the rebuild.
    assert out.environment.id == _GH_ENV


def test_assign_resolved_env_sha_best_effort_on_failure(monkeypatch):
    import flash.envs.adapter as adapter
    from flash import runner
    from flash.spec import EnvironmentSpec, JobSpec

    def boom(*a, **k):
        raise RuntimeError("github down")

    monkeypatch.setattr(adapter, "_resolve_ref_sha", boom)
    spec = JobSpec(environment=EnvironmentSpec(id=_GH_ENV))
    out = runner._assign_resolved_env_sha(spec)
    assert out.environment.resolved_sha == ""  # submission never blocks; worker resolves itself


def test_assign_resolved_env_sha_noop_without_env_or_already_pinned(monkeypatch):
    import flash.envs.adapter as adapter
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

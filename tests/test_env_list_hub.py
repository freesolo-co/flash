"""Focused coverage for listing the github-backed environment hub."""

from __future__ import annotations

import pytest

import flash.envs.loading.loader as loader
import flash.server.domain.registry.envs as domain


def _tree(*entries: dict, truncated: bool = False) -> dict:
    return {"tree": list(entries), "truncated": truncated}


def _directory(path: str, sha: str) -> dict:
    return {"path": path, "type": "tree", "sha": sha}


def _blob(path: str) -> dict:
    return {"path": path, "type": "blob", "sha": f"sha-{path}"}


def _fake_hub(monkeypatch, trees: dict[str, dict], calls: list | None = None):
    def fake_download_json(_ref, url, _context, **kwargs):
        if calls is not None:
            calls.append((url, kwargs))
        treeish = url.split("/git/trees/")[1].split("?")[0]
        return trees[treeish]

    monkeypatch.setattr(loader, "_download_github_json", fake_download_json)


def test_lists_only_environment_directories_in_two_tree_reads(monkeypatch):
    calls: list = []
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_directory("acme", "sha-acme"), _directory("rival", "sha-rival")),
            "sha-acme": _tree(
                _directory("project", "sha-project"),
                _blob("project/my-env/environment.py"),
                _directory("notes", "sha-notes"),
                _blob("notes/readme.md"),
                _blob("nested/deeper/too-deep/environment.py"),
            ),
        },
        calls,
    )

    assert loader.list_managed_namespace_slugs("acme") == ["acme/project/my-env"]
    assert len(calls) == 2
    assert calls[1][0].endswith("?recursive=1")
    assert all(call[1] == {"timeout": 20.0, "max_rate_limit_retries": 1} for call in calls)
    assert not any("sha-rival" in call[0] for call in calls)


def test_absent_namespace_is_empty(monkeypatch):
    _fake_hub(monkeypatch, {"main": _tree(_directory("rival", "sha-rival"))})

    assert loader.list_managed_namespace_slugs("acme") == []


def test_malformed_namespace_entry_fails_instead_of_looking_empty(monkeypatch):
    _fake_hub(monkeypatch, {"main": _tree({"path": "acme", "type": "tree"})})

    with pytest.raises(RuntimeError, match="no usable sha"):
        loader.list_managed_namespace_slugs("acme")


def test_truncated_namespace_tree_fails_instead_of_under_reporting(monkeypatch):
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_directory("acme", "sha-acme")),
            "sha-acme": _tree(_blob("project/my-env/environment.py"), truncated=True),
        },
    )

    with pytest.raises(RuntimeError, match="truncated"):
        loader.list_managed_namespace_slugs("acme")


def test_listing_uses_tree_api_and_never_clones(monkeypatch):
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_directory("acme", "sha-acme")),
            "sha-acme": _tree(_blob("project/my-env/environment.py")),
        },
    )
    monkeypatch.setattr(
        domain, "_run_git", lambda *_args, **_kwargs: pytest.fail("listing must not clone")
    )

    assert loader.list_managed_namespace_slugs("acme") == ["acme/project/my-env"]


def _user_key() -> dict:
    return {"org_slug": "acme", "org_id": "org-1"}


def test_domain_derives_namespace_from_authenticated_key(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    seen: list[str] = []
    monkeypatch.setattr(
        loader,
        "list_managed_namespace_slugs",
        lambda namespace: seen.append(namespace) or [f"{namespace}/project/my-env"],
    )

    assert domain.list_namespace_slugs(key=_user_key()) == ["acme/project/my-env"]
    assert seen == ["acme"]


def test_domain_refuses_org_agnostic_internal_key(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        loader,
        "list_managed_namespace_slugs",
        lambda *_args: pytest.fail("internal key must not reach the hub"),
    )

    with pytest.raises(domain.EnvPublishError) as excinfo:
        domain.list_namespace_slugs(key={"auth_kind": "internal"})
    assert excinfo.value.status == 403


def test_malformed_recursive_path_fails_instead_of_hiding_an_env(monkeypatch):
    """The path is the only field that decides whether an entry is a marker.

    Skipping an entry with a missing or non-string path silently drops an environment that may be
    published, which is the silent-empty answer this endpoint exists to remove. The namespace read
    already raises on an unusable type, so the recursive read has to be consistent with it.
    """
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_directory("acme", "sha-acme")),
            "sha-acme": _tree({"type": "blob", "sha": "sha-x"}),
        },
    )

    with pytest.raises(RuntimeError, match="no usable path"):
        loader.list_managed_namespace_slugs("acme")


def test_domain_requires_server_github_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with pytest.raises(domain.EnvPublishError) as excinfo:
        domain.list_namespace_slugs(key=_user_key())
    assert excinfo.value.status == 503


def test_domain_translates_hub_failures_without_returning_empty(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    def fail(_namespace):
        raise RuntimeError("hub tree was truncated")

    monkeypatch.setattr(loader, "list_managed_namespace_slugs", fail)

    with pytest.raises(domain.EnvPublishError) as excinfo:
        domain.list_namespace_slugs(key=_user_key())
    assert excinfo.value.status == 502


def test_domain_reports_a_quota_refusal_as_rate_limited(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    def fail(_namespace):
        raise loader.GitHubRateLimitError("GitHub API rate limit exceeded (429)")

    monkeypatch.setattr(loader, "list_managed_namespace_slugs", fail)

    with pytest.raises(domain.EnvPublishError) as excinfo:
        domain.list_namespace_slugs(key=_user_key())
    assert excinfo.value.status == 429


def test_domain_reports_an_outage_as_unavailable_not_rate_limited(monkeypatch):
    """A 5xx or a dead connection is GitHub being unreachable, and 429 misdescribes it.

    Both are retriable, so the loader raised one type for both and every outage reached the caller
    as "you are rate limited" -- telling them to back off against a quota they never hit, while the
    real cause went unreported.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    def fail(_namespace):
        raise loader.GitHubUnavailableError("GitHub server error (502, transient)")

    monkeypatch.setattr(loader, "list_managed_namespace_slugs", fail)

    with pytest.raises(domain.EnvPublishError) as excinfo:
        domain.list_namespace_slugs(key=_user_key())
    assert excinfo.value.status == 503
    assert "rate limit" not in str(excinfo.value).lower()


def test_the_worker_still_reschedules_on_either_transient_cause():
    """Splitting the type must not change what the worker does: both still retry.

    The worker classifies on the shared base, so an outage that used to arrive as the rate-limit
    type keeps rescheduling the run rather than failing it permanently.
    """
    from flash.engine.worker.entry.worker import _worker_failure_flags

    for exc in (
        loader.GitHubRateLimitError("quota"),
        loader.GitHubUnavailableError("outage"),
    ):
        assert _worker_failure_flags(exc)["retriable"]

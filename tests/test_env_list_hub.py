"""Focused coverage for listing the github-backed environment hub."""

from __future__ import annotations

import pytest

import flash.envs.loader as loader
import flash.server.domain.envs as domain


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
                _directory("my-env", "sha-env"),
                _blob("my-env/environment.py"),
                _directory("notes", "sha-notes"),
                _blob("notes/readme.md"),
                _blob("nested/deeper/environment.py"),
            ),
        },
        calls,
    )

    assert loader.list_managed_namespace_slugs("acme") == ["acme/my-env"]
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
            "sha-acme": _tree(_blob("my-env/environment.py"), truncated=True),
        },
    )

    with pytest.raises(RuntimeError, match="truncated"):
        loader.list_managed_namespace_slugs("acme")


def test_listing_uses_tree_api_and_never_clones(monkeypatch):
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_directory("acme", "sha-acme")),
            "sha-acme": _tree(_blob("my-env/environment.py")),
        },
    )
    monkeypatch.setattr(
        domain, "_run_git", lambda *_args, **_kwargs: pytest.fail("listing must not clone")
    )

    assert loader.list_managed_namespace_slugs("acme") == ["acme/my-env"]


def _user_key() -> dict:
    return {"org_slug": "acme", "org_id": "org-1"}


def test_domain_derives_namespace_from_authenticated_key(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    seen: list[str] = []
    monkeypatch.setattr(
        loader,
        "list_managed_namespace_slugs",
        lambda namespace: seen.append(namespace) or [f"{namespace}/my-env"],
    )

    assert domain.list_namespace_slugs(key=_user_key()) == ["acme/my-env"]
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

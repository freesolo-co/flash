"""Listing published environments off the GitHub-backed hub.

The list reads the hub with the tree API rather than the non-shallow clone publish/delete use: the
hub is hundreds of MB, and a read-only list must not pay for a checkout it throws away.
"""

from __future__ import annotations

import pytest

import flash.envs.loader as loader
import flash.server.domain.envs as envs_mod


def _tree(*entries: dict) -> dict:
    return {"tree": list(entries)}


def _dir(path: str, sha: str) -> dict:
    return {"path": path, "type": "tree", "sha": sha}


def _blob(path: str) -> dict:
    return {"path": path, "type": "blob", "sha": f"sha-{path}"}


def _fake_hub(monkeypatch, trees: dict[str, dict], *, record: list | None = None):
    """Serve canned tree payloads keyed by the treeish in the requested URL."""

    def fake_download_json(ref, url, context):
        if record is not None:
            record.append(url)
        treeish = url.split("/git/trees/")[1].split("?")[0]
        if treeish not in trees:
            raise AssertionError(f"unexpected tree request for {treeish!r}")
        return trees[treeish]

    monkeypatch.setattr(loader, "_download_github_json", fake_download_json)


def test_lists_only_directories_that_contain_an_environment_file(monkeypatch):
    """An env is a directory holding environment.py — what publish writes.

    Paths in the recursive namespace response are relative to the namespace, so ``<name>/
    environment.py`` marks an environment. A directory without that file is not one, and a blob
    deeper than one level belongs to an environment already identified by its own entry.
    """
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_dir("acme", "sha-acme"), _dir("other-org", "sha-other")),
            "sha-acme": _tree(
                _dir("my-env", "sha-my-env"),
                _blob("my-env/environment.py"),
                _blob("my-env/README.md"),
                _dir("beta", "sha-beta"),
                _blob("beta/environment.py"),
                _dir("not-an-env", "sha-not-an-env"),
                _blob("not-an-env/notes.txt"),
                _blob("README.md"),
                _blob("nested/deeper/environment.py"),
            ),
        },
    )

    assert loader.list_managed_namespace_slugs("acme") == ["acme/beta", "acme/my-env"]


def test_lists_the_namespace_in_one_recursive_request(monkeypatch):
    """Two requests total, regardless of how many environments the org has published.

    A per-environment tree read would make an org with 100 envs spend 102 calls of the shared
    GITHUB_TOKEN's rate budget on a single list, so the namespace subtree is fetched recursively.
    """
    seen: list[str] = []
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_dir("acme", "sha-acme")),
            "sha-acme": _tree(
                *[
                    entry
                    for i in range(25)
                    for entry in (_dir(f"env-{i}", f"sha-{i}"), _blob(f"env-{i}/environment.py"))
                ]
            ),
        },
        record=seen,
    )

    assert len(loader.list_managed_namespace_slugs("acme")) == 25
    assert len(seen) == 2, seen
    assert seen[1].endswith("?recursive=1"), seen[1]


def test_absent_namespace_directory_is_an_empty_list(monkeypatch):
    """An org that has published nothing has no hub directory yet; that is empty, not an error."""
    _fake_hub(monkeypatch, {"main": _tree(_dir("other-org", "sha-other"))})

    assert loader.list_managed_namespace_slugs("acme") == []


def test_never_reads_another_namespace(monkeypatch):
    """Only the caller's namespace subtree is walked, so a list can't enumerate another org."""
    seen: list[str] = []
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_dir("acme", "sha-acme"), _dir("rival", "sha-rival")),
            "sha-acme": _tree(_dir("my-env", "sha-my-env"), _blob("my-env/environment.py")),
        },
        record=seen,
    )

    assert loader.list_managed_namespace_slugs("acme") == ["acme/my-env"]
    assert not any("sha-rival" in url for url in seen)


def test_listing_never_clones_the_hub(monkeypatch):
    """A read-only list must not shell out to git — the hub clone is hundreds of MB."""
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_dir("acme", "sha-acme")),
            "sha-acme": _tree(_dir("my-env", "sha-my-env"), _blob("my-env/environment.py")),
        },
    )
    monkeypatch.setattr(
        envs_mod, "_run_git", lambda *a, **k: pytest.fail("listing must not clone the hub")
    )

    assert loader.list_managed_namespace_slugs("acme") == ["acme/my-env"]


def test_truncated_tree_response_raises_instead_of_under_reporting(monkeypatch):
    """A truncated listing must fail loudly; a short list would read as "those are all of them".

    This matters more with one recursive request than it did with a walk: the recursive response
    carries every environment, so truncation silently drops real ones.
    """
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_dir("acme", "sha-acme")),
            "sha-acme": {
                "tree": [_dir("my-env", "sha-my-env"), _blob("my-env/environment.py")],
                "truncated": True,
            },
        },
    )

    with pytest.raises(RuntimeError, match="truncated"):
        loader.list_managed_namespace_slugs("acme")


def test_rejects_an_unsafe_namespace(monkeypatch):
    monkeypatch.setattr(
        loader,
        "_download_github_json",
        lambda *a, **k: pytest.fail("must be refused before any request"),
    )

    for bad in ("../etc", "acme/extra", ""):
        with pytest.raises(RuntimeError, match="unsafe managed environment namespace"):
            loader.list_managed_namespace_slugs(bad)


def _user_key(**extra) -> dict:
    return {"org_slug": "acme", "org_id": "org-1", **extra}


def test_domain_list_uses_the_key_namespace_not_the_caller(monkeypatch):
    """The namespace comes from the authenticated key, so a caller cannot ask for another org."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    seen: dict = {}

    def fake_list(namespace):
        seen["namespace"] = namespace
        return [f"{namespace}/my-env"]

    monkeypatch.setattr(loader, "list_managed_namespace_slugs", fake_list)

    assert envs_mod.list_namespace_slugs(key=_user_key()) == ["acme/my-env"]
    assert seen["namespace"] == "acme"


def test_domain_list_refuses_the_org_agnostic_internal_key(monkeypatch):
    """The internal key has no namespace of its own, so it must be refused, not answered empty."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setattr(
        loader,
        "list_managed_namespace_slugs",
        lambda *a, **k: pytest.fail("internal key must not reach the hub"),
    )

    with pytest.raises(envs_mod.EnvPublishError) as excinfo:
        envs_mod.list_namespace_slugs(key={"auth_kind": "internal"})
    assert excinfo.value.status == 403


def test_domain_list_reports_an_unconfigured_server_as_503(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(
        loader,
        "list_managed_namespace_slugs",
        lambda *a, **k: pytest.fail("must refuse before any request"),
    )

    with pytest.raises(envs_mod.EnvPublishError) as excinfo:
        envs_mod.list_namespace_slugs(key=_user_key())
    assert excinfo.value.status == 503
    assert "GITHUB_TOKEN" in str(excinfo.value)


def test_domain_list_translates_a_hub_failure_into_502_not_empty(monkeypatch):
    """The core of the bug: a failed read must never be indistinguishable from "nothing published"."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")

    def boom(namespace):
        raise RuntimeError("GitHub tree response for environment directory 'acme' was truncated")

    monkeypatch.setattr(loader, "list_managed_namespace_slugs", boom)

    with pytest.raises(envs_mod.EnvPublishError) as excinfo:
        envs_mod.list_namespace_slugs(key=_user_key())
    assert excinfo.value.status == 502


def test_domain_list_translates_rate_limiting_into_429(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")

    def boom(namespace):
        raise loader.GitHubRateLimitError("rate limit exceeded", throttled=True)

    monkeypatch.setattr(loader, "list_managed_namespace_slugs", boom)

    with pytest.raises(envs_mod.EnvPublishError) as excinfo:
        envs_mod.list_namespace_slugs(key=_user_key())
    assert excinfo.value.status == 429


def test_a_github_outage_is_502_not_429(monkeypatch):
    """GitHubRateLimitError also covers persistent 5xx and network failure, which are not throttling.

    Answering 429 for an upstream outage tells the caller to back off, when backing off cannot help
    and nothing they did caused it. Only a genuine rate limit sets ``throttled``.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")

    for message in ("GitHub returned 502", "network unreachable"):

        def boom(namespace, message=message):
            raise loader.GitHubRateLimitError(message)

        monkeypatch.setattr(loader, "list_managed_namespace_slugs", boom)

        with pytest.raises(envs_mod.EnvPublishError) as excinfo:
            envs_mod.list_namespace_slugs(key=_user_key())
        assert excinfo.value.status == 502, message


def test_only_a_real_rate_limit_is_marked_throttled():
    """The flag defaults to False, so a new raise site cannot accidentally claim throttling."""
    assert loader.GitHubRateLimitError("x").throttled is False
    assert loader.GitHubRateLimitError("x", throttled=True).throttled is True


def test_rate_limit_error_still_reschedules_worker_retries():
    """The retry semantics existing handlers rely on must survive adding the flag.

    ``flash/engine/worker/__init__.py`` isinstance-checks this class to stamp retriable=True, so it
    has to stay a RuntimeError subclass and keep matching that check.
    """
    assert issubclass(loader.GitHubRateLimitError, RuntimeError)
    assert isinstance(loader.GitHubRateLimitError("x", throttled=True), loader.GitHubRateLimitError)

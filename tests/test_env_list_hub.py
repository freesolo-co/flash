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


def _fake_hub(
    monkeypatch, trees: dict[str, dict], *, record: list | None = None, budgets: list | None = None
):
    """Serve canned tree payloads keyed by the treeish in the requested URL."""

    def fake_download_json(ref, url, context, **kwargs):
        if record is not None:
            record.append(url)
        if budgets is not None:
            budgets.append(kwargs)
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


def test_both_list_reads_are_bounded_for_an_interactive_caller(monkeypatch):
    """Neither read may inherit the publish path's batch retry budget.

    On the default (6 attempts x 120s socket + up to 180s backoff, per read) a hard-down GitHub
    holds the request ~30 minutes before the domain layer can return its controlled 502 -- so every
    caller times out first and the useful status is never delivered.
    """
    budgets: list[dict] = []
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_dir("acme", "sha-acme")),
            "sha-acme": _tree(_dir("my-env", "sha-my-env"), _blob("my-env/environment.py")),
        },
        budgets=budgets,
    )

    assert loader.list_managed_namespace_slugs("acme") == ["acme/my-env"]
    assert len(budgets) == 2, budgets
    for passed in budgets:
        assert passed["timeout"] == 20.0, passed
        assert passed["max_rate_limit_retries"] == 1, passed


def test_publish_path_reads_keep_the_batch_budget(monkeypatch):
    """The bound is scoped to the interactive list; a background download still gets full retries.

    `_download_github_json`'s defaults are what the publish and package paths rely on, so lowering
    them globally would make a large managed download give up early.
    """
    import inspect

    sig = inspect.signature(loader._download_github_json)
    assert sig.parameters["timeout"].default == 120.0
    assert sig.parameters["max_rate_limit_retries"].default == 5


@pytest.mark.parametrize(
    "bad_sha",
    [None, 123, "", {"oid": "abc"}],
    ids=["missing", "not-a-string", "empty", "object"],
)
def test_a_namespace_entry_with_an_unusable_sha_raises_instead_of_reporting_empty(
    monkeypatch, bad_sha
):
    """A malformed entry must NOT look like an absent namespace.

    This is the PR's own bug in miniature: the namespace directory IS there, so answering "nothing
    published" tells the user their publish did nothing. A hub response we cannot read has to fail
    loudly and become the controlled 502.
    """
    entry = {"path": "acme", "type": "tree"}
    if bad_sha is not None:
        entry["sha"] = bad_sha
    _fake_hub(monkeypatch, {"main": _tree(entry, _dir("other-org", "sha-other"))})

    with pytest.raises(RuntimeError, match="no usable sha"):
        loader.list_managed_namespace_slugs("acme")


def test_an_unusable_sha_reaches_the_caller_as_502(monkeypatch):
    """The loud failure must arrive as the controlled gateway error, not an uncaught 500."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")

    def boom(namespace):
        raise RuntimeError(
            f"GitHub tree entry for environment namespace {namespace!r} has no usable sha; "
            "the hub listing could not be read"
        )

    monkeypatch.setattr(loader, "list_managed_namespace_slugs", boom)

    with pytest.raises(envs_mod.EnvPublishError) as excinfo:
        envs_mod.list_namespace_slugs(key=_user_key())
    assert excinfo.value.status == 502


@pytest.mark.parametrize(
    "bad_type",
    [None, 123, "", {"kind": "blob"}],
    ids=["missing", "not-a-string", "empty", "object"],
)
def test_a_marker_entry_with_an_unusable_type_raises_instead_of_being_skipped(
    monkeypatch, bad_type
):
    """A `<name>/environment.py` entry we cannot classify must not be silently dropped.

    Same failure class as the namespace-sha case one level down: the marker path IS present, so
    skipping it reports a shorter list -- or "nothing published" for a single-env org -- which reads
    as a publish that did nothing. Only a well-formed non-blob (a directory genuinely named
    environment.py) is a legitimate skip; an unreadable one has to become the controlled 502.
    """
    entry = {"path": "my-env/environment.py"}
    if bad_type is not None:
        entry["type"] = bad_type
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_dir("acme", "sha-acme")),
            "sha-acme": _tree(_dir("my-env", "sha-my-env"), entry),
        },
    )

    with pytest.raises(RuntimeError, match="unusable type"):
        loader.list_managed_namespace_slugs("acme")


def test_a_directory_named_environment_py_is_still_skipped(monkeypatch):
    """The guard above must not turn a legitimate non-blob into an error.

    A directory named environment.py is well-formed and simply is not the file publish writes, so
    it stays a quiet skip. Without this the new validation would reject a readable hub response.
    """
    _fake_hub(
        monkeypatch,
        {
            "main": _tree(_dir("acme", "sha-acme")),
            "sha-acme": _tree(
                _dir("my-env", "sha-my-env"),
                _dir("my-env/environment.py", "sha-dir"),
                _dir("real", "sha-real"),
                _blob("real/environment.py"),
            ),
        },
    )

    assert loader.list_managed_namespace_slugs("acme") == ["acme/real"]


def test_a_non_utf8_tree_body_becomes_a_controlled_failure(monkeypatch):
    """A non-UTF-8 body must not escape as an uncaught 500.

    `json.loads` raises UnicodeDecodeError on undecodable bytes, and that is NOT a JSONDecodeError
    (it is a sibling under ValueError), so naming only JSONDecodeError let it past both this helper
    and the domain layer's RuntimeError mapping.
    """
    monkeypatch.setattr(
        loader,
        "_urlopen",
        lambda *_a, **_k: b'{"tree": [{"path": "\xff\xfe", "type": "tree"}]}',
    )
    ref = loader._parse_github_environment_ref(
        f"github:{loader._DEFAULT_MANAGED_ENV_REPO}@main:acme/environment.py"
    )

    with pytest.raises(RuntimeError, match="invalid response"):
        loader._download_github_json(ref, "https://api.github.com/x/git/trees/main", "tree")


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


def test_a_truncated_hub_response_is_502_not_an_uncaught_500(monkeypatch):
    """A truncated GitHub body must reach the caller as a controlled 502.

    `http.client.IncompleteRead` is an HTTPException -- not a URLError, ConnectionError, OSError, or
    RuntimeError -- so before it was named in `_urlopen`'s retry clause it escaped both the retry
    loop and this layer's `except RuntimeError`, surfacing as an uncaught HTTP 500.
    """
    import http.client
    import urllib.request

    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")

    class _TruncatedBody:
        def read(self, *_a):
            raise http.client.IncompleteRead(b"partial")

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: _TruncatedBody())
    monkeypatch.setattr("time.sleep", lambda _s: None)

    with pytest.raises(envs_mod.EnvPublishError) as excinfo:
        envs_mod.list_namespace_slugs(key=_user_key())
    assert excinfo.value.status == 502
    assert excinfo.value.status != 429, "an upstream fault is not the caller being throttled"


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

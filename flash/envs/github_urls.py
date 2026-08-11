"""URL, path, and payload-shape helpers for the GitHub requests behind environment loading.

Split out of ``flash.envs.loader`` to keep that module under the file-size gate. Everything here is
pure: it builds or validates strings and never performs I/O, holds state, or reads the environment.

``_github_headers`` deliberately stayed in ``loader`` rather than joining this group, even though it
looks like a sibling. It reads the token through ``loader._github_token``, and tests patch that name
ON ``loader`` (``monkeypatch.setattr(adapter, "_github_token", ...)``); moving the caller here would
make it resolve the token in this module's namespace instead and silently ignore that patch.

``loader`` re-exports these names, since that is the import site callers and tests already use.
"""

from __future__ import annotations

import urllib.parse

from flash.envs.refs import (
    GitHubEnvironmentRef,
    _is_safe_github_path_parts,
    _normalize_env_path,
)

_DEFAULT_MANAGED_ENV_REPO = "freesolo-co/environment-hub"


def _managed_hub_package_root(ref: GitHubEnvironmentRef) -> str:
    if ref.repo_full_name.lower() != _DEFAULT_MANAGED_ENV_REPO.lower():
        return ""
    parts = [part for part in ref.path.split("/") if part]
    if len(parts) < 2 or not _is_safe_github_path_parts(tuple(parts[:2])):
        return ""
    return "/".join(parts[:2])


def _github_contents_url(ref: GitHubEnvironmentRef, path: str) -> str:
    quoted_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/") if part)
    return (
        f"https://api.github.com/repos/{ref.repo_full_name}/contents/{quoted_path}"
        f"?ref={urllib.parse.quote(ref.ref, safe='')}"
    )


def _github_tree_url(ref: GitHubEnvironmentRef, treeish: str, *, recursive: bool = False) -> str:
    url = (
        f"https://api.github.com/repos/{ref.repo_full_name}/git/trees/"
        f"{urllib.parse.quote(treeish, safe='')}"
    )
    if recursive:
        url = f"{url}?recursive=1"
    return url


def _safe_contents_path(path: object, root_parts: list[str]) -> str:
    if not isinstance(path, str):
        raise RuntimeError("GitHub contents response did not include a path")
    try:
        normalized = _normalize_env_path(path)
    except ValueError as exc:
        raise RuntimeError(f"unsafe path in environment contents: {path!r}") from exc
    parts = normalized.split("/")
    if parts[: len(root_parts)] != root_parts:
        raise RuntimeError(f"unexpected path in environment contents: {path!r}")
    return normalized


def _github_response_message(payload: object) -> str:
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str) and message:
            return f" ({message})"
    return ""

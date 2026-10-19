"""The grammar of an environment identifier.

An environment is addressed either by its managed hub slug ``<org-slug>/<project-slug>/<name>``
or by an explicit ``github:`` reference. This module owns parsing, validation, and
canonicalization of both; fetching and caching live in :mod:`flash.envs.loading.loader`, which
re-exports these names so existing importers are unaffected.

Environment names are unique per PROJECT, so the owning project is part of the identity and
of the hub directory path.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass

_DEFAULT_GITHUB_REF = "main"
_DEFAULT_ENVIRONMENT_PATH = "environment.py"
_DEFAULT_MANAGED_ENV_REPO = "freesolo-co/environment-hub"

_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_GITHUB_SAFE_PART_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class GitHubTransientError(RuntimeError):
    """A GitHub failure worth retrying later; worker handler stamps retriable=True.

    The base of the two causes below, which the worker treats identically -- reschedule -- and the
    control plane does not: a caller told "rate limited" backs off against a quota, and there is no
    quota to back off against when GitHub is simply down.
    """


class GitHubRateLimitError(GitHubTransientError):
    """The request was refused against a quota: a 429, or a 403 that says rate limit."""


class GitHubUnavailableError(GitHubTransientError):
    """GitHub could not answer at all: a 5xx, a timeout, or a connection failure.

    Distinct from the quota case because the honest status differs. Reporting this as 429 tells the
    caller to slow down and, if they honour a Retry-After they were never given, to wait out a limit
    that was never reached -- while the actual cause, GitHub being unreachable, goes unreported.
    """


class GitHubPermanentError(RuntimeError):
    """GitHub answered, and the answer will not change on a retry: 401, 403, 404, or 422.

    Deliberately NOT a GitHubTransientError. Waiting is the right response to a blip and the wrong
    response to a typo, and the two are indistinguishable once both are a bare RuntimeError -- which
    is what let a misspelled environment repo defer past submit and rent a GPU to discover its own
    404. The env ref->sha pin stays best-effort for the transient base, and fails closed on this.

    A 404 is not proof the repo does not exist -- GitHub also 404s a private repo the token cannot
    read, rather than leaking its existence with a 403. Both are permanent for this caller: no
    amount of retrying makes an unreadable ref resolvable, and the fix (check the name, or grant the
    token access) is the user's either way. The message carries GitHub's own text so it names which.

    The credential codes reach here only after the rate-limit shapes (429, and the 403 that says
    rate limit) have been claimed as transient, so a 401 or a surviving 403 is a token this plane
    cannot fix by waiting: invalid, expired, or missing the scope. Same user-side fix as a typo.
    """


@dataclass(frozen=True)
class GitHubEnvironmentRef:
    owner: str
    repo: str
    ref: str
    path: str

    @property
    def repo_full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    def canonical(self) -> str:
        return f"github:{self.repo_full_name}@{self.ref}:{self.path}"


def is_github_environment_ref(value: str) -> bool:
    return _parse_github_environment_ref(value) is not None


def is_managed_environment_slug(value: str) -> bool:
    return _parse_managed_environment_slug(value) is not None


def is_freesolo_environment_id(value: str) -> bool:
    return is_managed_environment_slug(value) or is_github_environment_ref(value)


def canonical_environment_id(value: str) -> str:
    """Return the stable identity spelling for a managed slug or GitHub reference."""
    text = (value or "").strip()
    managed = canonical_managed_environment_slug(text)
    if managed is not None:
        return managed
    parsed = _parse_github_environment_ref(text)
    if parsed is None:
        raise ValueError(f"not a Freesolo environment id: {value!r}")
    return parsed.canonical()


def managed_slug_to_github_ref(value: str) -> str:
    parsed = _parse_managed_environment_slug(value)
    if parsed is None:
        raise ValueError(f"not a Freesolo environment slug: {value!r}")
    namespace, project, name = parsed
    return (
        f"github:{_DEFAULT_MANAGED_ENV_REPO}@{_DEFAULT_GITHUB_REF}:"
        f"{namespace}/{project}/{name}/{_DEFAULT_ENVIRONMENT_PATH}"
    )


def canonical_managed_environment_slug(value: str) -> str | None:
    if _parse_managed_environment_slug(value) is not None:
        return value

    parsed = _parse_github_environment_ref(value)
    if parsed is None:
        if _targets_managed_environment_repo(value):
            raise ValueError(_managed_environment_ref_error())
        return None

    if parsed.repo_full_name.lower() != _DEFAULT_MANAGED_ENV_REPO.lower():
        return None

    parts = parsed.path.split("/")
    if (
        parsed.ref != _DEFAULT_GITHUB_REF
        or len(parts) != 4
        or parts[3] != _DEFAULT_ENVIRONMENT_PATH
        or not _is_safe_github_path_parts(tuple(parts[:3]))
    ):
        raise ValueError(_managed_environment_ref_error())
    return "/".join(parts[:3])


def _targets_managed_environment_repo(value: str) -> bool:
    text = (value or "").strip()
    if not text.startswith("github:"):
        return False
    repo_ref = text[len("github:") :].partition(":")[0]
    repo = repo_ref.partition("@")[0]
    return repo.lower() == _DEFAULT_MANAGED_ENV_REPO.lower()


def _managed_environment_ref_error() -> str:
    return (
        "managed environment GitHub reference must be "
        f"github:{_DEFAULT_MANAGED_ENV_REPO}@{_DEFAULT_GITHUB_REF}:"
        f"<namespace>/<project>/<name>/{_DEFAULT_ENVIRONMENT_PATH}"
    )


def _parse_managed_environment_slug(value: str) -> tuple[str, str, str] | None:
    """Split a managed env id into ``(namespace, project, name)``.

    Environment names are unique per PROJECT, so the owning project is part of the identity and
    of the hub directory path: ``<org-slug>/<project-slug>/<name>``.
    """
    text = (value or "").strip()
    if not text or ":" in text:
        return None
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme or parsed.netloc:
        return None
    parts = text.split("/")
    if len(parts) != 3 or not _is_safe_github_path_parts(tuple(parts)):
        return None
    return parts[0], parts[1], parts[2]


def _parse_github_environment_ref(value: str) -> GitHubEnvironmentRef | None:
    text = (value or "").strip()
    if not text:
        return None
    if text.startswith("github:"):
        body = text[len("github:") :]
        repo_ref, sep, path = body.partition(":")
        try:
            path = _normalize_env_path(path)
        except ValueError:
            return None
        if not sep:
            path = _DEFAULT_ENVIRONMENT_PATH
        repo_part, at, ref = repo_ref.partition("@")
        if not at:
            ref = _DEFAULT_GITHUB_REF
        if not ref:
            return None
        if not _is_safe_github_path_parts((ref,)):
            return None
        owner_repo = repo_part.split("/")
        if len(owner_repo) == 2 and _is_safe_github_path_parts(owner_repo):
            return GitHubEnvironmentRef(owner_repo[0], owner_repo[1], ref, path)
        return None

    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != "github.com":
        return None
    parts = [urllib.parse.unquote(p) for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    repo = repo[:-4] if repo.endswith(".git") else repo
    if not _is_safe_github_path_parts((owner, repo)):
        return None
    if len(parts) >= 5 and parts[2] in {"blob", "tree"}:
        ref = parts[3]
        if not _is_safe_github_path_parts((ref,)):
            return None
        raw_path = "/".join(parts[4:])
        try:
            path = _normalize_env_path(raw_path)
        except ValueError:
            return None
        if parts[2] == "tree" and raw_path and not path.endswith(".py"):
            path = f"{path.rstrip('/')}/{_DEFAULT_ENVIRONMENT_PATH}"
    elif len(parts) == 2:
        ref = _DEFAULT_GITHUB_REF
        path = _DEFAULT_ENVIRONMENT_PATH
    else:
        return None
    return GitHubEnvironmentRef(owner, repo, ref, path)


def _normalize_env_path(path: str | None) -> str:
    if not path:
        return _DEFAULT_ENVIRONMENT_PATH
    raw = path.strip()
    if not raw:
        return _DEFAULT_ENVIRONMENT_PATH
    raw = raw.replace("\\", "/")
    if raw.startswith("/"):
        raise ValueError(f"unsafe environment path: {path!r}")
    parts = [part for part in raw.split("/") if part]
    if not parts:
        return _DEFAULT_ENVIRONMENT_PATH
    if any(part == ".." or part == "." for part in parts):
        raise ValueError(f"unsafe environment path: {path!r}")
    return "/".join(parts)


def _is_safe_github_path_parts(parts: list[str] | tuple[str, ...]) -> bool:
    if not parts:
        return False
    if any(part in {".", "..", ""} for part in parts):
        return False
    return all(_GITHUB_SAFE_PART_RE.fullmatch(part) for part in parts)


def is_commit_sha(value: str) -> bool:
    """True when value is a full 40-hex-char git commit id (an immutable ref)."""
    return _COMMIT_SHA_RE.fullmatch(value) is not None


def github_environment_ref_is_pinned(value: str) -> bool:
    """True when a ``github:`` environment id names an immutable commit.

    Only the SHA case is decidable from the id. Branches and tags share one string shape, so callers
    must give symbolic-ref advice that is valid for either rather than assuming the ref is movable.
    """
    parsed = _parse_github_environment_ref(value)
    return parsed is not None and is_commit_sha(parsed.ref)

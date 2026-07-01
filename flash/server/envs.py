"""Managed Freesolo environment publishing."""

from __future__ import annotations

import base64
import gzip
import io
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
from pathlib import Path

_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_MEMBERS = 5000
_MAX_SCAN_MEMBERS = 200_000
_DEFAULT_GITHUB_REPO = "freesolo-co/environment-hub"
_GITHUB_BRANCH = "main"
_DEFAULT_ENVIRONMENT_FILE = "environment.py"
# Genuine repo-control directories that live at the ROOT of the environment-hub checkout. A DELETE
# runs `git rm -r -- <org-slug>/<name>` directly against that checkout, so a slug whose namespace is
# one of these would target tracked repo infrastructure (e.g. `DELETE /v1/envs/.github/workflows`).
# Org slugs can never be dot-prefixed, so these are never publishable namespaces and are always safe
# to reject on delete.
_REPO_CONTROL_TOP_LEVEL_PATHS = {
    ".github",
    ".git",
}
# Top-level segments barred from env-package CONTENTS by `_safe_extract`. A strict SUPERSET of the
# repo-control set: it also bars a top-level `source/` dir inside a package. `source` is intentionally
# NOT a repo-control namespace, so the delete validator must reject only
# `_REPO_CONTROL_TOP_LEVEL_PATHS`, not this set, or `source/<name>` envs become undeletable.
_BLOCKED_TOP_LEVEL_PATHS = _REPO_CONTROL_TOP_LEVEL_PATHS | {"source"}
_TAR_METADATA_TYPES = {
    tarfile.XHDTYPE,
    tarfile.XGLTYPE,
    tarfile.GNUTYPE_LONGNAME,
    tarfile.GNUTYPE_LONGLINK,
}
_GIT_TIMEOUT_S = 180
_GIT_PUSH_RETRY_DELAYS_SECONDS = (2.0, 5.0)
_NAMESPACE_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")


def _human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.0f} MB"


class EnvPublishError(Exception):
    def __init__(self, message: str, *, status: int = 400):
        super().__init__(message)
        self.status = status


def namespace_for(key: dict) -> str:
    org = key.get("org") if isinstance(key.get("org"), dict) else {}
    raw = key.get("org_slug") or org.get("slug")
    slug = str(raw or "").strip()
    if not slug:
        raise EnvPublishError(
            "authenticated Freesolo key must include an org slug (used to derive the hub namespace) — "
            "publish with a key created at https://freesolo.co/sign-in (`flash login`)"
        )
    if not _NAMESPACE_RE.fullmatch(slug):
        raise EnvPublishError("authenticated Freesolo key has an invalid org slug")
    return slug


def _sanitize_name(name: str) -> str:
    from flash.schema import normalize_env_name_segment

    return normalize_env_name_segment(name) or "env"


def _publish_slug_for_name(name: str, key: dict) -> tuple[str, str]:
    caller_namespace = namespace_for(key)
    raw = str(name or "").strip()
    if "/" not in raw:
        return caller_namespace, _sanitize_name(raw)
    parts = [part.strip() for part in raw.split("/")]
    if len(parts) != 2 or not all(parts):
        raise EnvPublishError("env name with namespace must be 'namespace/name'")
    namespace = parts[0]
    if not _NAMESPACE_RE.fullmatch(namespace):
        raise EnvPublishError("env namespace must match [a-z0-9][a-z0-9._-]*")
    clean = _sanitize_name(parts[1])
    if namespace != caller_namespace:
        raise EnvPublishError(
            "env namespace must match your Freesolo org namespace "
            f"({caller_namespace}/...); got {namespace}/{clean}",
            status=403,
        )
    return namespace, clean


class _LimitedReader:
    """Bounds total bytes read from a *decompressed* tar stream. A GNU LONGNAME/LONGLINK or PAX header
    payload is consumed inside ``tarfile.next()`` and never yielded as a member, so per-member size
    accounting can't see it — a tiny gzip can declare a multi-GB header and OOM the process. Reads are
    clamped to the remaining budget so a single oversized header read allocates at most the limit, then
    raises ``EnvPublishError``."""

    def __init__(self, raw, limit: int):
        self._raw = raw
        self._remaining = limit

    def read(self, size: int = -1) -> bytes:
        want = self._remaining + 1 if size is None or size < 0 else min(size, self._remaining + 1)
        chunk = self._raw.read(want)
        self._remaining -= len(chunk)
        if self._remaining < 0:
            raise EnvPublishError(
                f"env package is too large uncompressed (limit {_human_mb(_MAX_UNCOMPRESSED_BYTES)})"
            )
        return chunk


def _safe_extract(tar_bytes: bytes, dest: Path) -> None:
    root = dest.resolve()
    # Stream backstop > the per-member content cap by the max header+padding overhead (<=1KB/member),
    # so it never false-rejects a legitimate package but still bounds an oversized header payload.
    stream_cap = _MAX_UNCOMPRESSED_BYTES + _MAX_MEMBERS * 1024 + (1 << 20)
    try:
        reader = _LimitedReader(gzip.GzipFile(fileobj=io.BytesIO(tar_bytes)), stream_cap)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            total = 0
            scanned = 0
            extracted = 0
            for member in tar:
                scanned += 1
                if scanned > _MAX_SCAN_MEMBERS:
                    raise EnvPublishError(
                        f"env package has too many entries to scan (limit {_MAX_SCAN_MEMBERS})"
                    )
                if member.type in _TAR_METADATA_TYPES:
                    continue
                segments: list[str] = []
                for segment in member.name.replace("\\", "/").split("/"):
                    if not segment or segment == ".":
                        continue
                    if segment == "..":
                        raise EnvPublishError(f"unsafe path in env package: {member.name!r}")
                    segments.append(segment)
                if not segments:
                    continue
                normalized_name = "/".join(segments)
                target = (dest / normalized_name).resolve()
                if target != root and root not in target.parents:
                    raise EnvPublishError(f"unsafe path in env package: {member.name!r}")
                if segments[0] in _BLOCKED_TOP_LEVEL_PATHS:
                    raise EnvPublishError(
                        "env packages must not contain repo-control or source top-level paths"
                    )
                if member.islnk() or member.issym():
                    raise EnvPublishError(f"links are not allowed in env packages: {member.name!r}")
                if not (member.isreg() or member.isdir()):
                    raise EnvPublishError(
                        f"only regular files and directories are allowed in env packages, "
                        f"but {member.name!r} is a special file"
                    )
                extracted += 1
                if extracted > _MAX_MEMBERS:
                    raise EnvPublishError(
                        f"env package has too many members (limit {_MAX_MEMBERS})"
                    )
                total += max(0, member.size)
                if total > _MAX_UNCOMPRESSED_BYTES:
                    raise EnvPublishError(
                        "env package is too large uncompressed "
                        f"(limit {_human_mb(_MAX_UNCOMPRESSED_BYTES)})"
                    )
                member.name = normalized_name
                tar.extract(member, dest)
    except tarfile.TarError as exc:
        raise EnvPublishError(f"env package is not a valid .tar.gz archive: {exc}") from exc
    except OSError as exc:
        raise EnvPublishError(f"env package could not be extracted: {exc}") from exc


def _github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN")


def _redact(value: str, token: str) -> str:
    if not token:
        return value
    return value.replace(token, "<redacted>").replace(
        urllib.parse.quote(token, safe=""), "<redacted>"
    )


def _credentialed_repo_url(repo: str, token: str) -> str:
    quoted = urllib.parse.quote(token, safe="")
    return f"https://x-access-token:{quoted}@github.com/{repo}.git"


def _run_git(
    cwd: Path, args: list[str], *, token: str, operation: str = "upload"
) -> subprocess.CompletedProcess[str]:
    # ``operation`` is the user-facing verb for the action that ran this git command ("upload" for
    # publish, "delete" for delete) so a git failure reports what the caller actually attempted
    # instead of a misleading "upload" on the delete path. The trailing preposition mirrors the
    # wording used elsewhere in this module ("environments to Freesolo" for upload, "from Freesolo"
    # for delete — see `_staged_has_changes` / `delete_package`).
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        direction = "from" if operation == "delete" else "to"
        raise EnvPublishError(
            f"git is required to {operation} environments {direction} Freesolo", status=503
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise EnvPublishError(
            f"Freesolo environment {operation} git command timed out after {_GIT_TIMEOUT_S}s",
            status=504,
        ) from exc
    if proc.returncode != 0:
        output = f"{proc.stdout or ''}\n{proc.stderr or ''}".strip()
        cmd = "git " + " ".join(args)
        raise EnvPublishError(
            _redact(
                f"Freesolo environment {operation} failed during `{cmd}`: {output[:1000]}", token
            ),
            status=502,
        )
    return proc


def _is_retryable_git_publish_error(message: str) -> bool:
    lowered = message.lower()
    permanent = (
        "authentication failed",
        "could not read username",
        "repository not found",
        "permission denied",
        "403",
        "401",
    )
    if any(marker in lowered for marker in permanent):
        return False
    retryable = (
        "failed to push some refs",
        "fetch first",
        "non-fast-forward",
        "stale info",
        "cannot lock ref",
        "connection reset",
        "operation timed out",
        "the remote end hung up",
        "early eof",
        "index.lock",
        "rebase",
    )
    return any(marker in lowered for marker in retryable)


def _copy_package_to_checkout(*, source: Path, checkout: Path, publish_root: str) -> None:
    target = checkout / publish_root
    checkout_root = checkout.resolve()
    target_root = target.resolve()
    if target_root != checkout_root and checkout_root not in target_root.parents:
        raise EnvPublishError("unsafe environment publish path")
    shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _commit_environment_update(
    *, checkout: Path, publish_root: str, message: str, token: str
) -> bool:
    _run_git(checkout, ["config", "user.name", "freesolo-bot"], token=token)
    _run_git(checkout, ["config", "user.email", "bot@freesolo.co"], token=token)
    _run_git(checkout, ["add", "-A", "--", publish_root], token=token)
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", publish_root],
            cwd=checkout,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise EnvPublishError(
            f"Freesolo environment upload git command timed out after {_GIT_TIMEOUT_S}s",
            status=504,
        ) from exc
    if proc.returncode == 0:
        return False
    if proc.returncode != 1:
        output = f"{proc.stdout or ''}\n{proc.stderr or ''}".strip()
        raise EnvPublishError(
            _redact(
                f"Freesolo environment upload failed during staged diff check: {output}", token
            ),
            status=502,
        )
    _run_git(checkout, ["commit", "-m", message], token=token)
    return True


def _push_environment_commit(*, checkout: Path, token: str) -> None:
    _run_git(checkout, ["pull", "--rebase", "origin", _GITHUB_BRANCH], token=token)
    _run_git(checkout, ["push", "origin", f"HEAD:{_GITHUB_BRANCH}"], token=token)


def _push_environment_delete(*, checkout: Path, publish_root: str, token: str) -> None:
    """Rebase onto the remote tip, re-apply the slug removal, then push.

    A concurrent publish can add files under the same ``publish_root`` between our clone and this
    push. ``git pull --rebase`` only replays the removals our delete commit recorded, so a
    non-conflicting concurrent addition (e.g. a new sidecar while ``environment.py`` is unchanged)
    survives the rebase: the push would succeed and we'd report ``deleted: true`` while the slug
    directory still exists partially. Re-run ``git rm -r`` against the freshly rebased tree and fold
    any newly-tracked paths into the delete commit so the pushed state has the slug fully removed.
    """
    _run_git(
        checkout, ["pull", "--rebase", "origin", _GITHUB_BRANCH], token=token, operation="delete"
    )
    _run_git(
        checkout,
        ["rm", "-r", "--quiet", "--ignore-unmatch", "--", publish_root],
        token=token,
        operation="delete",
    )
    if _staged_has_changes(checkout):
        _run_git(checkout, ["commit", "--amend", "--no-edit"], token=token, operation="delete")
    _run_git(
        checkout, ["push", "origin", f"HEAD:{_GITHUB_BRANCH}"], token=token, operation="delete"
    )


def _github_publish_once(
    *,
    dest: Path,
    repo: str,
    token: str,
    publish_root: str,
    message: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="flash-env-hub-") as tmp:
        tmp_path = Path(tmp)
        checkout = tmp_path / "environment-hub"
        _run_git(
            tmp_path,
            [
                "clone",
                "--branch",
                _GITHUB_BRANCH,
                "--single-branch",
                _credentialed_repo_url(repo, token),
                str(checkout),
            ],
            token=token,
        )
        _copy_package_to_checkout(source=dest, checkout=checkout, publish_root=publish_root)
        if _commit_environment_update(
            checkout=checkout,
            publish_root=publish_root,
            message=message,
            token=token,
        ):
            _push_environment_commit(checkout=checkout, token=token)


def _github_publish(dest: Path, *, name: str, key: dict) -> str:
    token = _github_token()
    if not token:
        raise EnvPublishError(
            "GITHUB_TOKEN is required to upload environments to Freesolo",
            status=503,
        )
    repo = _DEFAULT_GITHUB_REPO
    ns, clean = _publish_slug_for_name(name, key)
    publish_root = f"{ns}/{clean}"
    if not (dest / _DEFAULT_ENVIRONMENT_FILE).is_file():
        raise EnvPublishError("env package must contain environment.py")
    if not any(path.is_file() for path in dest.rglob("*")):
        raise EnvPublishError("env package contains no files")
    message = f"Upload Flash environment {ns}/{clean}"

    last_error: EnvPublishError | None = None
    max_attempts = len(_GIT_PUSH_RETRY_DELAYS_SECONDS) + 1
    for attempt in range(max_attempts):
        if attempt:
            time.sleep(_GIT_PUSH_RETRY_DELAYS_SECONDS[attempt - 1])
        try:
            _github_publish_once(
                dest=dest,
                repo=repo,
                token=token,
                publish_root=publish_root,
                message=message,
            )
            return f"{ns}/{clean}"
        except EnvPublishError as exc:
            last_error = exc
            if attempt == max_attempts - 1 or not _is_retryable_git_publish_error(str(exc)):
                raise
    assert last_error is not None
    raise last_error


def publish_package(*, package_b64: str, name: str, key: dict) -> str:
    if not isinstance(name, str):
        raise EnvPublishError("env name must be a string")
    if not isinstance(package_b64, str):
        raise EnvPublishError("env package must be a base64 string")
    if not name:
        raise EnvPublishError("missing env name")
    max_encoded = ((_MAX_UPLOAD_BYTES + 2) // 3) * 4 + 3
    if len(package_b64) > max_encoded:
        raise EnvPublishError(
            f"env package upload is too large (limit {_human_mb(_MAX_UPLOAD_BYTES)} compressed)",
            status=413,
        )
    try:
        tar_bytes = base64.b64decode(package_b64, validate=True)
    except Exception as exc:
        raise EnvPublishError("env package is not valid base64") from exc
    if not tar_bytes:
        raise EnvPublishError("empty env package")
    if len(tar_bytes) > _MAX_UPLOAD_BYTES:
        raise EnvPublishError(
            f"env package upload is too large (limit {_human_mb(_MAX_UPLOAD_BYTES)} compressed)",
            status=413,
        )
    with tempfile.TemporaryDirectory(prefix="flash-env-publish-") as tmp:
        dest = Path(tmp)
        _safe_extract(tar_bytes, dest)
        return _github_publish(dest, name=name, key=key)


_SLUG_SEGMENT_RE = re.compile(r"^[a-z0-9._-]+$")


def _validate_slug(slug: str) -> tuple[str, str]:
    """Validate a ``namespace/name`` environment id and return its two segments.

    Rejects anything that isn't exactly two non-empty path-safe segments — in particular
    ``..`` and stray separators — so the slug can be used directly as a git pathspec / on-disk
    publish root without traversal risk (mirrors the guarantees `_sanitize_name` gives publish).
    """
    if not isinstance(slug, str):
        raise EnvPublishError("env id must be a string")
    # Reject — never silently normalize — anything that isn't ALREADY the canonical two-segment id.
    # Do NOT strip leading/trailing '/' OR surrounding whitespace: requests like
    # `DELETE /v1/envs/ns/env/` (trailing slash captured by the :path param) or an encoded-padded
    # `ns/env%20` (FastAPI decodes to `ns/env `) must FAIL, not be trimmed to `ns/env`. Otherwise the
    # package is deleted under the canonical slug while the response / downstream mirroring carry the
    # non-canonical id, leaving a stale UI row. Split on the raw value so any empty (stray separator)
    # or whitespace-padded segment fails the per-segment check below.
    parts = slug.split("/")
    if len(parts) != 2 or not all(parts):
        raise EnvPublishError("env id must be 'namespace/name'")
    for segment in parts:
        if segment in {".", ".."} or not _SLUG_SEGMENT_RE.match(segment):
            raise EnvPublishError(f"invalid env id segment: {segment!r}")
    namespace, name = parts
    # The delete path runs `git rm -r -- <namespace>/<name>` directly against the hub checkout, so
    # the top-level path component (the namespace) must never be a genuine repo-control directory.
    # An internal-key delete bypasses the namespace-ownership check in delete_package, so this
    # validator is the only barrier, and a request like `DELETE /v1/envs/.github/workflows` would
    # otherwise remove tracked repo infrastructure. Reject ONLY `_REPO_CONTROL_TOP_LEVEL_PATHS`, NOT
    # the wider publish-content blocklist: `source` is a legitimately publishable org slug, so
    # blocking it here would leave those envs publishable-but-undeletable.
    if namespace in _REPO_CONTROL_TOP_LEVEL_PATHS:
        raise EnvPublishError(f"invalid env id segment: {namespace!r}")
    return namespace, name


def canonical_env_id(slug: str) -> str:
    """Validate ``slug`` and return the canonical ``namespace/name`` id.

    Public wrapper over :func:`_validate_slug` so the route can normalize the id ONCE up front and
    use the same canonical value for deletion, the metadata-mirror drop, and the response — never a
    non-canonical variant. Raises :class:`EnvPublishError` (400) for anything that isn't already
    canonical, so a padded / stray-separator id is rejected rather than silently trimmed.
    """
    namespace, name = _validate_slug(slug)
    return f"{namespace}/{name}"


def _staged_has_changes(checkout: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=checkout,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise EnvPublishError(
            "git is required to delete environments from Freesolo", status=503
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise EnvPublishError(
            f"Freesolo environment delete git command timed out after {_GIT_TIMEOUT_S}s",
            status=504,
        ) from exc
    # `git diff --cached --quiet` exits 0 (nothing staged) or 1 (staged changes). Treat ONLY 1 as
    # "has changes"; any other code (e.g. 128 for a broken/unexpected repo state) is an error, not a
    # signal to commit-and-push, so surface it as a controlled EnvPublishError instead of a bare 500.
    if proc.returncode not in (0, 1):
        output = f"{proc.stdout or ''}\n{proc.stderr or ''}".strip()
        raise EnvPublishError(
            f"Freesolo environment delete failed during staged diff check: {output}",
            status=502,
        )
    return proc.returncode == 1


def _github_delete_once(*, repo: str, token: str, publish_root: str, message: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="flash-env-hub-") as tmp:
        tmp_path = Path(tmp)
        checkout = tmp_path / "environment-hub"
        _run_git(
            tmp_path,
            [
                "clone",
                "--branch",
                _GITHUB_BRANCH,
                "--single-branch",
                _credentialed_repo_url(repo, token),
                str(checkout),
            ],
            token=token,
            operation="delete",
        )
        target = checkout / publish_root
        checkout_root = checkout.resolve()
        target_root = target.resolve()
        if target_root != checkout_root and checkout_root not in target_root.parents:
            raise EnvPublishError("unsafe environment delete path")
        if not target.exists():
            # Idempotent: nothing published under this slug, so there is nothing to remove.
            #
            # The `git clone --single-branch` above already fetched the branch tip, so `target`
            # reflects the latest published state as of that fetch and this check runs microseconds
            # later with no intervening network round-trip. A `git pull` here would not meaningfully
            # close the publish/delete race: it can only shrink the (sub-second, clone-unpack) window
            # between a fetch and this check, never eliminate it — a publish landing one instant after
            # *any* fetch is still unseen. Reporting `deleted: false` for a slug absent in the freshly
            # cloned tip is the correct answer for the state we observed; a publish racing in
            # afterwards is a genuinely concurrent, unordered event. We therefore accept this race
            # rather than pay an extra round-trip on every delete. (The inverse race — a concurrent
            # publish landing while the slug *does* exist — is handled in `_push_environment_delete`.)
            return False
        _run_git(checkout, ["config", "user.name", "freesolo-bot"], token=token, operation="delete")
        _run_git(
            checkout, ["config", "user.email", "bot@freesolo.co"], token=token, operation="delete"
        )
        _run_git(
            checkout,
            ["rm", "-r", "--quiet", "--ignore-unmatch", "--", publish_root],
            token=token,
            operation="delete",
        )
        if not _staged_has_changes(checkout):
            # The directory was present on disk but untracked (never committed) — nothing to push.
            return False
        _run_git(checkout, ["commit", "-m", message], token=token, operation="delete")
        _push_environment_delete(checkout=checkout, publish_root=publish_root, token=token)
        return True


def _github_delete(slug: str, *, token: str) -> bool:
    repo = _DEFAULT_GITHUB_REPO
    message = f"Delete Flash environment {slug}"
    last_error: EnvPublishError | None = None
    max_attempts = len(_GIT_PUSH_RETRY_DELAYS_SECONDS) + 1
    for attempt in range(max_attempts):
        if attempt:
            time.sleep(_GIT_PUSH_RETRY_DELAYS_SECONDS[attempt - 1])
        try:
            return _github_delete_once(
                repo=repo, token=token, publish_root=slug, message=message
            )
        except EnvPublishError as exc:
            last_error = exc
            if attempt == max_attempts - 1 or not _is_retryable_git_publish_error(str(exc)):
                raise
    assert last_error is not None
    raise last_error


def delete_package(*, slug: str, key: dict) -> bool:
    """Remove a published environment from the hub.

    Returns ``True`` when a package was removed and ``False`` when it was already absent
    (idempotent). Authorization mirrors publish's namespace isolation: a user key may delete
    only environments in its own org-slug namespace, while the internal service key
    (``auth_kind == "internal"``) may delete any environment.
    """
    namespace, name = _validate_slug(slug)
    canonical = f"{namespace}/{name}"
    caller_namespace = None if key.get("auth_kind") == "internal" else namespace_for(key)
    if caller_namespace is not None and namespace != caller_namespace:
        raise EnvPublishError(
            "you can only delete environments in your own namespace "
            f"({caller_namespace}/…); got {canonical!r}",
            status=403,
        )
    token = _github_token()
    if not token:
        raise EnvPublishError(
            "GITHUB_TOKEN is required to delete environments from Freesolo",
            status=503,
        )
    return _github_delete(canonical, token=token)

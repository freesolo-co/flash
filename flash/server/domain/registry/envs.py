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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from flash.envs.loading.loader import _github_token
from flash.envs.package.direct_tokens import DirectTokenScanError, package_contains_direct_token
from flash.envs.package.limits import (
    ARCHIVE_MEMBER_LIMIT,
    ARCHIVE_SCAN_MEMBER_LIMIT,
    TAR_METADATA_TYPES,
    LimitedArchiveReader,
    archive_stream_limit,
    tar_member_segments,
)

_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_DOWNLOAD_BYTES = _MAX_UNCOMPRESSED_BYTES
_MAX_MEMBERS = ARCHIVE_MEMBER_LIMIT
_MAX_SCAN_MEMBERS = ARCHIVE_SCAN_MEMBER_LIMIT
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


def publish_slug_for_name(name: str, key: dict, project_slug: str) -> tuple[str, str, str]:
    """The ``(namespace, project, name)`` a publish of ``name`` by ``key`` would write.

    Environment names are unique per PROJECT, so the owning project is part of the published
    identity: the destination is ``<org-slug>/<project-slug>/<name>``, which is also the
    directory the package occupies in the hub repository.

    Pure and side-effect free, so the publish route can resolve the destination slug before
    uploading anything. Accepts a bare name, or a fully qualified id whose namespace and project
    segments must match the caller's org and the project being published to -- a qualified id
    that disagrees is refused rather than silently redirected.
    """
    caller_namespace = namespace_for(key)
    project = str(project_slug or "").strip()
    if not project:
        # The route resolves the slug through `require_project_access_slug`, which already refuses
        # an empty one and says WHY (standalone has no project directory; a validation response
        # missing it is an upstream fault). Neither cause is the caller's key, so this no longer
        # blames one -- it stays as the last guard for a direct domain-level caller, which is why
        # it is a 500: reaching here means a caller skipped the resolution that would have
        # explained it.
        raise EnvPublishError(
            "the project's slug was not resolved before publishing, so the environment's "
            "destination is unknown",
            status=500,
        )
    if not _NAMESPACE_RE.fullmatch(project):
        raise EnvPublishError("project slug must match [a-z0-9][a-z0-9._-]*")
    raw = str(name or "").strip()
    if "/" not in raw:
        return caller_namespace, project, _sanitize_name(raw)
    parts = [part.strip() for part in raw.split("/")]
    if len(parts) != 3 or not all(parts):
        raise EnvPublishError("env name with namespace must be 'namespace/project/name'")
    namespace, given_project = parts[0], parts[1]
    if not _NAMESPACE_RE.fullmatch(namespace):
        raise EnvPublishError("env namespace must match [a-z0-9][a-z0-9._-]*")
    clean_name = _sanitize_name(parts[2])
    if namespace != caller_namespace:
        raise EnvPublishError(
            "env namespace must match your Freesolo org namespace "
            f"({caller_namespace}/...); got {namespace}/{given_project}/{clean_name}",
            status=403,
        )
    if given_project != project:
        raise EnvPublishError(
            "env project segment must match the project you are publishing to "
            f"({caller_namespace}/{project}/...); got {namespace}/{given_project}/{clean_name}",
            status=403,
        )
    return namespace, project, clean_name


def _safe_extract(tar_bytes: bytes, dest: Path) -> None:
    root = dest.resolve()
    try:
        reader = LimitedArchiveReader(
            gzip.GzipFile(fileobj=io.BytesIO(tar_bytes)),
            archive_stream_limit(_MAX_UNCOMPRESSED_BYTES, _MAX_MEMBERS),
            lambda: EnvPublishError(
                f"env package is too large uncompressed (limit {_human_mb(_MAX_UNCOMPRESSED_BYTES)})"
            ),
        )
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
                if member.type in TAR_METADATA_TYPES:
                    continue
                segments = tar_member_segments(
                    member.name,
                    unsafe_error=lambda name: EnvPublishError(
                        f"unsafe path in env package: {name!r}"
                    ),
                )
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


def _token_representation_pattern(token: str) -> re.Pattern[str]:
    encoded = urllib.parse.quote(token, safe="")
    parts = []
    cursor = 0
    for match in re.finditer(r"%[0-9A-F]{2}", encoded):
        parts.append(re.escape(encoded[cursor : match.start()]))
        octet = match.group()[1:]
        parts.append(
            "%"
            + "".join(
                f"[{digit.lower()}{digit.upper()}]" if digit in "ABCDEF" else digit
                for digit in octet
            )
        )
        cursor = match.end()
    parts.append(re.escape(encoded[cursor:]))
    return re.compile(f"(?:{re.escape(token)}|{''.join(parts)})")


def _redact(value: str, token: str) -> str:
    if not token:
        return value
    return _token_representation_pattern(token).sub("<redacted>", value)


def _repo_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


@contextmanager
def _git_credential_env(token: str) -> Iterator[dict[str, str]]:
    """Yield a git environment backed by short-lived, mode-restricted askpass files."""
    with tempfile.TemporaryDirectory(prefix="flash-git-auth-") as tmp:
        auth_dir = Path(tmp)
        token_file = auth_dir / "token"
        askpass = auth_dir / "askpass.sh"
        token_fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fchmod(token_fd, 0o600)
        with os.fdopen(token_fd, "w", encoding="utf-8") as stream:
            stream.write(token)
        askpass_fd = os.open(askpass, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
        os.fchmod(askpass_fd, 0o700)
        with os.fdopen(askpass_fd, "w", encoding="utf-8") as stream:
            stream.write(
                "#!/bin/sh\n"
                'case "$1" in\n'
                "  *Username*) printf '%s\\n' x-access-token ;;\n"
                '  *Password*) exec cat "$FLASH_GIT_TOKEN_FILE" ;;\n'
                "  *) exit 1 ;;\n"
                "esac\n"
            )

        token_pattern = _token_representation_pattern(token) if token else None
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "GH_TOKEN",
                "GITHUB_PAT",
                "GITHUB_TOKEN",
                "GIT_CONFIG_COUNT",
                "GIT_CONFIG_PARAMETERS",
            }
            and not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
            and not (
                token_pattern
                and (
                    token_pattern.search(key) is not None or token_pattern.search(value) is not None
                )
            )
        }
        env.update(
            {
                "GIT_ASKPASS": str(askpass),
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "credential.helper",
                "GIT_CONFIG_VALUE_0": "",
                "GIT_TERMINAL_PROMPT": "0",
                "FLASH_GIT_TOKEN_FILE": str(token_file),
            }
        )
        yield env


def _checkout_child(checkout: Path, publish_root: str, *, operation: str) -> Path:
    target = checkout / publish_root
    checkout_root = checkout.resolve()
    target_root = target.resolve()
    if target_root != checkout_root and checkout_root not in target_root.parents:
        raise EnvPublishError(f"unsafe environment {operation} path")
    return target


def _run_git(
    cwd: Path, args: list[str], *, token: str, operation: str = "upload"
) -> subprocess.CompletedProcess[str]:
    # ``operation`` is the user-facing verb for the action that ran this git command ("upload" for
    # publish, "delete" for delete) so a git failure reports what the caller actually attempted
    # instead of a misleading "upload" on the delete path. The trailing preposition mirrors the
    # wording used elsewhere in this module ("environments to Freesolo" for upload, "from Freesolo"
    # for delete — see `_staged_has_changes` / `delete_package`).
    try:
        with _git_credential_env(token) as env:
            proc = subprocess.run(
                ["git", *args],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_S,
            )
    except FileNotFoundError as exc:
        direction = "from" if operation in {"delete", "download"} else "to"
        raise EnvPublishError(
            f"git is required to {operation} environments {direction} Freesolo", status=503
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise EnvPublishError(
            f"Freesolo environment {operation} git command timed out after {_GIT_TIMEOUT_S}s",
            status=504,
        ) from exc
    if proc.returncode != 0:
        output = _redact(f"{proc.stdout or ''}\n{proc.stderr or ''}".strip(), token)
        cmd = "git " + " ".join(args)
        raise EnvPublishError(
            _redact(
                f"Freesolo environment {operation} failed during `{cmd}`: {output[:1000]}", token
            ),
            status=502,
        )
    return proc


def _clone_hub_checkout(
    *,
    parent: Path,
    checkout: Path,
    repo: str,
    token: str,
    operation: str,
    shallow: bool = False,
) -> None:
    args = ["clone"]
    if shallow:
        args.extend(["--depth", "1"])
    args.extend(
        [
            "--branch",
            _GITHUB_BRANCH,
            "--single-branch",
            _repo_url(repo),
            str(checkout),
        ]
    )
    _run_git(parent, args, token=token, operation=operation)


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
    target = _checkout_child(checkout, publish_root, operation="publish")
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
        with _git_credential_env(token) as env:
            proc = subprocess.run(
                ["git", "diff", "--cached", "--quiet", "--", publish_root],
                cwd=checkout,
                env=env,
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

    Re-run ``git rm -r`` after rebasing so concurrent files under the slug are deleted too.
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
    if _staged_has_changes(checkout, token=token):
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
        _clone_hub_checkout(
            parent=tmp_path,
            checkout=checkout,
            repo=repo,
            token=token,
            operation="upload",
        )
        _copy_package_to_checkout(source=dest, checkout=checkout, publish_root=publish_root)
        if _commit_environment_update(
            checkout=checkout,
            publish_root=publish_root,
            message=message,
            token=token,
        ):
            _push_environment_commit(checkout=checkout, token=token)


def _github_publish(
    dest: Path,
    *,
    name: str,
    key: dict,
    project_slug: str,
) -> str:
    token = _github_token()
    if not token:
        raise EnvPublishError(
            "GITHUB_TOKEN is required for the Flash control plane to publish environments",
            status=503,
        )
    repo = _DEFAULT_GITHUB_REPO
    ns, project, clean = publish_slug_for_name(name, key, project_slug)
    publish_root = f"{ns}/{project}/{clean}"
    if not (dest / _DEFAULT_ENVIRONMENT_FILE).is_file():
        raise EnvPublishError("env package must contain environment.py")
    if not any(path.is_file() for path in dest.rglob("*")):
        raise EnvPublishError("env package contains no files")
    message = f"Upload Flash environment {publish_root}"

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
            # the id the caller pastes into `[environment] id` must be the SAME three segments
            # the package was written under, or it resolves to nothing.
            return publish_root
        except EnvPublishError as exc:
            last_error = exc
            if attempt == max_attempts - 1 or not _is_retryable_git_publish_error(str(exc)):
                raise
    assert last_error is not None
    raise last_error


def validate_publish_inputs(*, package_b64: object, name: object) -> bytes:
    """Check a publish request's own inputs and return its decoded package.

    Pure and side-effect free, so the route can reject an unpublishable request before doing
    anything observable — no ownership lookup, no backend call, no upload. Raises the same
    :class:`EnvPublishError` (400/413) the publish itself would raise, so the failure a caller
    sees does not depend on when the check ran.
    """
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
    return tar_bytes


def publish_package(
    *,
    package_b64: str,
    name: str,
    key: dict,
    project_slug: str,
) -> str:
    """Publish a package to the hub under ``<namespace>/<project>/<name>``."""
    tar_bytes = validate_publish_inputs(package_b64=package_b64, name=name)
    with tempfile.TemporaryDirectory(prefix="flash-env-publish-") as tmp:
        dest = Path(tmp)
        _safe_extract(tar_bytes, dest)
        try:
            contains_direct_token = package_contains_direct_token(dest)
        except DirectTokenScanError:
            raise EnvPublishError("env package could not be scanned safely") from None
        if contains_direct_token:
            raise EnvPublishError(
                "env package contains a direct access token; remove it before publishing"
            )
        return _github_publish(dest, name=name, key=key, project_slug=project_slug)


_SLUG_SEGMENT_RE = re.compile(r"^[a-z0-9._-]+$")


def _validate_slug(slug: str) -> tuple[str, str, str]:
    """Validate a ``namespace/project/name`` environment id and return its three segments.

    Rejects anything that isn't exactly three non-empty path-safe segments — in particular
    ``..`` and stray separators — so the slug can be used directly as a git pathspec / on-disk
    publish root without traversal risk (mirrors the guarantees `_sanitize_name` gives publish).
    """
    if not isinstance(slug, str):
        raise EnvPublishError("env id must be a string")
    # reject rather than normalize: whitespace or stray separators must not delete the canonical
    # slug
    # while returning a non-canonical id. validate the raw segments.
    parts = slug.split("/")
    if len(parts) != 3 or not all(parts):
        raise EnvPublishError("env id must be 'namespace/project/name'")
    for segment in parts:
        if segment in {".", ".."} or not _SLUG_SEGMENT_RE.match(segment):
            raise EnvPublishError(f"invalid env id segment: {segment!r}")
    namespace, project, name = parts
    # delete runs git rm against the hub checkout, so block repo-control top-level paths here.
    # use only _REPO_CONTROL_TOP_LEVEL_PATHS because valid namespaces such as source must remain
    # deletable.
    if namespace in _REPO_CONTROL_TOP_LEVEL_PATHS:
        raise EnvPublishError(f"invalid env id segment: {namespace!r}")
    return namespace, project, name


def canonical_env_id(slug: str) -> str:
    """Validate ``slug`` and return the canonical ``namespace/project/name`` id.

    Public wrapper over :func:`_validate_slug` so the route can normalize the id ONCE up front and
    use the same canonical value for deletion, the metadata-mirror drop, and the response — never a
    non-canonical variant. Raises :class:`EnvPublishError` (400) for anything that isn't already
    canonical, so a padded / stray-separator id is rejected rather than silently trimmed.
    """
    namespace, project, name = _validate_slug(slug)
    return f"{namespace}/{project}/{name}"


def _require_namespace_access(canonical: str, key: dict, *, action: str) -> None:
    namespace = canonical.split("/", 1)[0]
    caller_namespace = None if key.get("auth_kind") == "internal" else namespace_for(key)
    if caller_namespace is not None and namespace != caller_namespace:
        raise EnvPublishError(
            f"you can only {action} environments in your own namespace "
            f"({caller_namespace}/...); got {canonical!r}",
            status=403,
        )


def _package_checkout_directory(source: Path) -> bytes:
    if not (source / _DEFAULT_ENVIRONMENT_FILE).is_file():
        raise EnvPublishError("environment package not found", status=404)

    total = 0
    members = 0
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(source.rglob("*")):
            rel = path.relative_to(source).as_posix()
            if rel.split("/", 1)[0] in _BLOCKED_TOP_LEVEL_PATHS:
                raise EnvPublishError(
                    "env packages must not contain repo-control or source top-level paths",
                    status=502,
                )
            if path.is_symlink():
                raise EnvPublishError(f"links are not allowed in env packages: {rel!r}", status=502)
            if path.is_file():
                total += path.stat().st_size
                if total > _MAX_UNCOMPRESSED_BYTES:
                    raise EnvPublishError(
                        "env package is too large uncompressed "
                        f"(limit {_human_mb(_MAX_UNCOMPRESSED_BYTES)})",
                        status=413,
                    )
            elif not path.is_dir():
                raise EnvPublishError(
                    f"only regular files and directories are allowed in env packages, "
                    f"but {rel!r} is a special file",
                    status=502,
                )

            members += 1
            if members > _MAX_MEMBERS:
                raise EnvPublishError(f"env package has too many members (limit {_MAX_MEMBERS})")
            tar.add(path, arcname=rel, recursive=False)
            if buf.tell() > _MAX_DOWNLOAD_BYTES:
                raise EnvPublishError(
                    f"env package download is too large (limit {_human_mb(_MAX_DOWNLOAD_BYTES)})",
                    status=413,
                )

    data = buf.getvalue()
    if len(data) > _MAX_DOWNLOAD_BYTES:
        raise EnvPublishError(
            f"env package download is too large (limit {_human_mb(_MAX_DOWNLOAD_BYTES)})",
            status=413,
        )
    return data


def _github_download_once(*, repo: str, token: str, publish_root: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="flash-env-hub-") as tmp:
        tmp_path = Path(tmp)
        checkout = tmp_path / "environment-hub"
        _clone_hub_checkout(
            parent=tmp_path,
            checkout=checkout,
            repo=repo,
            token=token,
            operation="download",
            shallow=True,
        )
        target = _checkout_child(checkout, publish_root, operation="download")
        if not target.is_dir():
            raise EnvPublishError("environment package not found", status=404)
        return _package_checkout_directory(target)


def _github_download(slug: str, *, token: str) -> bytes:
    return _github_download_once(repo=_DEFAULT_GITHUB_REPO, token=token, publish_root=slug)


def _authorized_hub_request(slug: str, key: dict, *, action: str) -> tuple[str, str]:
    """Resolve one hub operation's target and credential, or refuse it.

    Every hub operation asks the same two questions in the same order -- is this key allowed to touch
    this namespace, and does the control plane have a hub credential at all -- and they must stay in
    that order: a caller with no access should be told so whether or not the server happens to be
    configured. Returns the canonical ``namespace/project/name`` and the token.
    """
    namespace, project, name = _validate_slug(slug)
    canonical = f"{namespace}/{project}/{name}"
    _require_namespace_access(canonical, key, action=action)
    token = _github_token()
    if not token:
        # both halves are load-bearing and each has a test: GITHUB_TOKEN names the variable to set,
        # and "control plane" says whose it is -- this is a 503 because the SERVER is unconfigured,
        # not a 4xx telling the caller to supply a credential of their own.
        raise EnvPublishError(
            f"GITHUB_TOKEN is required for the Flash control plane to {action} environments",
            status=503,
        )
    return canonical, token


def download_package(*, slug: str, key: dict) -> bytes:
    """Return a tar.gz package for a published environment from the GitHub hub."""
    canonical, token = _authorized_hub_request(slug, key, action="download")
    return _github_download(canonical, token=token)


def list_namespace_slugs(*, key: dict) -> list[str]:
    """Return the published ``namespace/project/name`` slugs the caller's org owns, sorted.

    Reads the hub through the GitHub tree API rather than the clone the publish/delete paths use:
    the hub is hundreds of MB and clones non-shallow, so a read-only list must not pay for a
    checkout it would immediately throw away.

    The namespace comes from the authenticated key, never from the caller, so this cannot enumerate
    another org. The internal key is org-agnostic and has no namespace of its own to list, so it is
    refused here instead of being silently answered with an empty list.
    """
    if key.get("auth_kind") == "internal":
        raise EnvPublishError(
            "listing environments requires a Freesolo user key, which carries the org namespace "
            "to list; the internal service key is org-agnostic",
            status=403,
        )
    namespace = namespace_for(key)
    if not _github_token():
        raise EnvPublishError(
            "GITHUB_TOKEN is required for the Flash control plane to list environments",
            status=503,
        )
    from flash.envs.loading import loader

    try:
        return loader.list_managed_namespace_slugs(namespace)
    except loader.GitHubRateLimitError as exc:
        raise EnvPublishError(
            f"Freesolo environment list is rate limited: {exc}", status=429
        ) from exc
    except loader.GitHubUnavailableError as exc:
        # 503, not 429: a 5xx or a connection failure is GitHub being unreachable, and telling the
        # caller they exceeded a quota is both wrong and unactionable -- there is no quota to wait
        # out, and the real cause never reaches them.
        raise EnvPublishError(
            f"Freesolo environment list is temporarily unavailable: {exc}", status=503
        ) from exc
    except RuntimeError as exc:
        raise EnvPublishError(f"Freesolo environment list failed: {exc}", status=502) from exc


def _staged_has_changes(checkout: Path, *, token: str = "") -> bool:
    try:
        with _git_credential_env(token) as env:
            proc = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=checkout,
                env=env,
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
            _redact(
                f"Freesolo environment delete failed during staged diff check: {output}", token
            ),
            status=502,
        )
    return proc.returncode == 1


def _github_delete_once(*, repo: str, token: str, publish_root: str, message: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="flash-env-hub-") as tmp:
        tmp_path = Path(tmp)
        checkout = tmp_path / "environment-hub"
        _clone_hub_checkout(
            parent=tmp_path,
            checkout=checkout,
            repo=repo,
            token=token,
            operation="delete",
        )
        target = _checkout_child(checkout, publish_root, operation="delete")
        if not target.exists():
            # absence in the freshly cloned tip is idempotent. another pull cannot eliminate the
            # publish/delete race; concurrent additions to an existing slug are handled on push.
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
        if not _staged_has_changes(checkout, token=token):
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
            return _github_delete_once(repo=repo, token=token, publish_root=slug, message=message)
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
    canonical, token = _authorized_hub_request(slug, key, action="delete")
    return _github_delete(canonical, token=token)

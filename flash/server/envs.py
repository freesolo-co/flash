"""Managed Freesolo environment publishing.

``POST /v1/envs`` accepts a packaged Freesolo environment and uploads it to the
managed environment hub. The returned id is a Freesolo environment slug
(``namespace/name``) that Flash resolves internally.
"""

from __future__ import annotations

import base64
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
_DEFAULT_GITHUB_REPO = "freesolo-co/environment-hub"
_GITHUB_BRANCH = "main"
_DEFAULT_ENVIRONMENT_FILE = "environment.py"
_BLOCKED_TOP_LEVEL_PATHS = {
    ".github",
    ".git",
    "source",
}
_GIT_TIMEOUT_S = 180
_GIT_PUSH_RETRY_DELAYS_SECONDS = (2.0, 5.0)


def _human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.0f} MB"


class EnvPublishError(Exception):
    def __init__(self, message: str, *, status: int = 400):
        super().__init__(message)
        self.status = status


# Reserved hub namespace for the operator/internal service key. Its /v1/me identity is synthetic
# and shared (no per-user email), so it can't derive a namespace from an email like a user key does —
# but it IS a trusted identity that can submit runs and read everything, so it must be able to
# publish environments too. Matches the slug `internal@freesolo.co` would yield (see
# flash.server.db.ensure_internal_key) so the namespace is stable regardless of the row's email.
_INTERNAL_NAMESPACE = "internal-freesolo-co"


def namespace_for(key: dict) -> str:
    # Special case RESERVED for the internal service key ONLY (auth_kind == "internal"): give it a
    # fixed namespace instead of requiring an email. Every other key is a freesolo USER key and still
    # must carry a real email — we never loosen that, so two different users can't collide on one
    # namespace.
    if key.get("auth_kind") == "internal":
        return _INTERNAL_NAMESPACE
    email = str(key.get("email") or "")
    if "@" not in email:
        raise EnvPublishError(
            "authenticated Freesolo key must include an email (used to derive the hub namespace) — "
            "publish with a key created at https://freesolo.co/sign-in (`flash login`)"
        )
    slug = re.sub(r"[^a-z0-9]+", "-", email.lower()).strip("-")
    return slug or "user"


def _sanitize_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")
    if slug in {".", ".."} or not re.search(r"[a-z0-9]", slug):
        return "env"
    return slug or "env"


def _safe_extract(tar_bytes: bytes, dest: Path) -> None:
    root = dest.resolve()
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            total = 0
            for count, member in enumerate(tar, start=1):
                if count > _MAX_MEMBERS:
                    raise EnvPublishError(
                        f"env package has too many members (limit {_MAX_MEMBERS})"
                    )
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


def _github_repo() -> str:
    return _DEFAULT_GITHUB_REPO


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


def _run_git(cwd: Path, args: list[str], *, token: str) -> subprocess.CompletedProcess[str]:
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
        raise EnvPublishError(
            "git is required to upload environments to Freesolo", status=503
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise EnvPublishError(
            f"Freesolo environment upload git command timed out after {_GIT_TIMEOUT_S}s",
            status=504,
        ) from exc
    if proc.returncode != 0:
        output = f"{proc.stdout or ''}\n{proc.stderr or ''}".strip()
        cmd = "git " + " ".join(args)
        raise EnvPublishError(
            _redact(f"Freesolo environment upload failed during `{cmd}`: {output[:1000]}", token),
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


def _environment_file_relative_path(root: Path) -> str:
    canonical = root / _DEFAULT_ENVIRONMENT_FILE
    if canonical.is_file():
        return _DEFAULT_ENVIRONMENT_FILE
    raise EnvPublishError("env package must contain environment.py")


def _github_publish(dest: Path, *, name: str, key: dict) -> str:
    token = _github_token()
    if not token:
        raise EnvPublishError(
            "GITHUB_TOKEN is required to upload environments to Freesolo",
            status=503,
        )
    repo = _github_repo()
    ns = namespace_for(key)
    clean = _sanitize_name(name)
    publish_root = f"{ns}/{clean}"
    _environment_file_relative_path(dest)
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

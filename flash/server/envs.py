"""Managed Freesolo environment publishing.

``POST /v1/envs`` accepts a packaged Freesolo environment and uploads it to a
managed GitHub repository. The returned id is a GitHub-backed environment ref
that the worker resolves through ``freesolo.environments``.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_MEMBERS = 5000
_MAX_GITHUB_FILE_BYTES = 1024 * 1024
_DEFAULT_GITHUB_REPO = "freesolo-co/environment-hub"
_DEFAULT_GITHUB_BRANCH = "main"
_DEFAULT_ENVIRONMENT_FILE = "freesolo/environment.py"


def _human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.0f} MB"


class EnvPublishError(Exception):
    def __init__(self, message: str, *, status: int = 400):
        super().__init__(message)
        self.status = status


class _GitHubApiError(EnvPublishError):
    def __init__(self, message: str, status: int):
        super().__init__(message, status=status)


def namespace_for(key: dict) -> str:
    email = str(key.get("email") or "")
    if "@" in email:
        raw = email
    elif key.get("id") is not None:
        raw = f"key-{key['id']}"
    else:
        raw = str(key.get("key_prefix") or "user")
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return slug or "user"


def _sanitize_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")
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
                target = (dest / member.name).resolve()
                if target != root and root not in target.parents:
                    raise EnvPublishError(f"unsafe path in env package: {member.name!r}")
                member_path = member.name.replace("\\", "/")
                segments = [segment for segment in member_path.split("/") if segment]
                if not segments:
                    continue
                if segments[0] in {".github", ".git"}:
                    raise EnvPublishError(
                        "env packages must not contain .github or .git top-level paths"
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
                tar.extract(member, dest)
    except tarfile.TarError as exc:
        raise EnvPublishError(f"env package is not a valid .tar.gz archive: {exc}") from exc
    except OSError as exc:
        raise EnvPublishError(f"env package could not be extracted: {exc}") from exc


def _github_repo() -> str:
    return os.environ.get("FLASH_ENV_GITHUB_REPO") or _DEFAULT_GITHUB_REPO


def _github_branch() -> str:
    return os.environ.get("FLASH_ENV_GITHUB_BRANCH") or _DEFAULT_GITHUB_BRANCH


def _github_token() -> str | None:
    return os.environ.get("FLASH_ENV_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")


def _github_json(
    method: str,
    url: str,
    *,
    token: str,
    body: dict | None = None,
) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "freesolo-flash",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise _GitHubApiError(
            f"GitHub environment upload failed ({exc.code}): {detail[:500]}",
            status=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise EnvPublishError(
            f"GitHub environment upload failed: {exc.reason}",
            status=502,
        ) from exc


def _existing_file_sha(repo: str, branch: str, path: str, token: str) -> str | None:
    quoted = urllib.parse.quote(path, safe="/")
    url = f"https://api.github.com/repos/{repo}/contents/{quoted}?ref={urllib.parse.quote(branch)}"
    try:
        data = _github_json("GET", url, token=token)
    except _GitHubApiError as exc:
        if exc.status == 404:
            return None
        raise
    sha = data.get("sha")
    return str(sha) if sha else None


def _put_github_file(
    *,
    repo: str,
    branch: str,
    path: str,
    data: bytes,
    token: str,
    message: str,
) -> None:
    quoted = urllib.parse.quote(path, safe="/")
    if len(data) > _MAX_GITHUB_FILE_BYTES:
        raise EnvPublishError(
            f"environment upload file {path!r} exceeds GitHub Contents API limit "
            f"({_MAX_GITHUB_FILE_BYTES} bytes)",
            status=413,
        )
    url = f"https://api.github.com/repos/{repo}/contents/{quoted}"
    body = {
        "message": message,
        "content": base64.b64encode(data).decode("ascii"),
        "branch": branch,
    }
    sha = _existing_file_sha(repo, branch, path, token)
    if sha:
        body["sha"] = sha
    _github_json("PUT", url, token=token, body=body)


def _environment_file_relative_path(root: Path) -> str:
    canonical = root / _DEFAULT_ENVIRONMENT_FILE
    if canonical.is_file():
        return _DEFAULT_ENVIRONMENT_FILE
    matches = sorted(path for path in root.rglob("environment.py") if path.is_file())
    if matches:
        return matches[0].relative_to(root).as_posix()
    raise EnvPublishError(
        "env package must contain freesolo/environment.py or another environment.py entrypoint"
    )


def _github_publish(dest: Path, *, name: str, key: dict) -> str:
    token = _github_token()
    if not token:
        raise EnvPublishError(
            "FLASH_ENV_GITHUB_TOKEN or GITHUB_TOKEN is required to upload environments to GitHub",
            status=503,
        )
    repo = _github_repo()
    branch = _github_branch()
    ns = namespace_for(key)
    clean = _sanitize_name(name)
    publish_root = f"environments/{ns}/{clean}"
    env_rel = _environment_file_relative_path(dest)
    files = sorted(path for path in dest.rglob("*") if path.is_file())
    if not files:
        raise EnvPublishError("env package contains no files")
    message = f"Upload Flash environment {ns}/{clean}"
    for file_path in files:
        rel = file_path.relative_to(dest).as_posix()
        _put_github_file(
            repo=repo,
            branch=branch,
            path=f"{publish_root}/{rel}",
            data=file_path.read_bytes(),
            token=token,
            message=message,
        )
    return f"github:{repo}@{branch}:{publish_root}/{env_rel}"


def publish_package(*, package_b64: str, name: str, is_new: bool, key: dict) -> str:
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

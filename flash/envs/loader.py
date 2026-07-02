"""Loading and reference resolution for Freesolo SDK environments.

Resolves managed slugs / github: refs / URLs to local files (with a bounded on-disk cache),
probes packaged datasets, and constructs the :class:`~flash.envs.adapter.FreesoloEnvironment`.
Split out of ``flash.envs.adapter`` (which keeps the runtime environment class); the public
names remain importable from ``flash.envs.adapter``.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

_DEFAULT_GITHUB_REF = "main"
_DEFAULT_ENVIRONMENT_PATH = "environment.py"
_DEFAULT_MANAGED_ENV_REPO = "freesolo-co/environment-hub"
_CACHE_ROOT = Path(os.environ.get("FLASH_ENV_CACHE_DIR", "/tmp/flash-env-cache"))
# bound the on-disk env cache so it cannot grow without limit (one subdir per env
# content-sha, ~30-80 MB each). evicted LRU by dir mtime, which we bump on cache hit.
_CACHE_MAX_ENTRIES = int(os.environ.get("FLASH_ENV_CACHE_MAX_ENTRIES", "32"))
_CACHE_MAX_BYTES = int(os.environ.get("FLASH_ENV_CACHE_MAX_BYTES", str(4 * 1024 * 1024 * 1024)))
# never evict an entry used within this window; a concurrent run may be loading it.
_CACHE_MIN_AGE_SECONDS = 600
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_TARBALL_BYTES = 1024 * 1024 * 1024
_MAX_CONTENTS_JSON_BYTES = 16 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 5000
_MAX_ARCHIVE_SCAN_MEMBERS = 200_000
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_GITHUB_SAFE_PART_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_DATASET_SPLIT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TAR_METADATA_TYPES = {
    tarfile.XHDTYPE,
    tarfile.XGLTYPE,
    tarfile.GNUTYPE_LONGNAME,
    tarfile.GNUTYPE_LONGLINK,
}
_CANONICAL_INPUT_KEY = "input"
_CANONICAL_OUTPUT_KEY = "output"
_ENV_DEP_INSTALL_TIMEOUT_S = int(os.environ.get("FLASH_ENV_DEP_INSTALL_TIMEOUT_S", "900"))
_INSTALLED_ENV_DEPS: set[tuple[str, ...]] = set()


class GitHubRateLimitError(RuntimeError):
    """Persistent GitHub rate-limit; worker handler stamps retriable=True for rescheduling."""


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


def managed_slug_to_github_ref(value: str) -> str:
    parsed = _parse_managed_environment_slug(value)
    if parsed is None:
        raise ValueError(f"not a Freesolo environment slug: {value!r}")
    namespace, name = parsed
    return (
        f"github:{_DEFAULT_MANAGED_ENV_REPO}@{_DEFAULT_GITHUB_REF}:"
        f"{namespace}/{name}/{_DEFAULT_ENVIRONMENT_PATH}"
    )


def _parse_managed_environment_slug(value: str) -> tuple[str, str] | None:
    text = (value or "").strip()
    if not text or ":" in text:
        return None
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme or parsed.netloc:
        return None
    parts = text.split("/")
    if len(parts) != 2 or not _is_safe_github_path_parts(tuple(parts)):
        return None
    return parts[0], parts[1]


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
    if not raw:
        return _DEFAULT_ENVIRONMENT_PATH
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


def _github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN")


def _is_commit_sha(value: str) -> bool:
    return _COMMIT_SHA_RE.fullmatch(value) is not None


def _resolve_ref_sha(
    parsed: GitHubEnvironmentRef,
    pinned_sha: str | None = None,
    *,
    timeout: float = 60.0,
    max_rate_limit_retries: int = 5,
) -> str:
    # Control plane pins sha once; workers skip GitHub entirely on a fan-out.
    if pinned_sha and _is_commit_sha(pinned_sha):
        return pinned_sha
    if _is_commit_sha(parsed.ref):
        return parsed.ref
    # Symbolic refs are NOT cached in-process: managed slugs point at environment-hub@main which moves.
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "freesolo-flash"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    commit_url = f"https://api.github.com/repos/{parsed.repo_full_name}/commits/{urllib.parse.quote(parsed.ref, safe='')}"
    req = urllib.request.Request(commit_url, headers=headers)
    data = _urlopen(req, timeout=timeout, max_rate_limit_retries=max_rate_limit_retries)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to resolve GitHub environment ref {parsed.canonical()}: invalid response"
        ) from exc
    sha = payload.get("sha")
    if not isinstance(sha, str) or not _is_commit_sha(sha):
        raise RuntimeError(f"Failed to resolve GitHub environment ref {parsed.canonical()}")
    return sha


def _iter_capped_chunks(resp: object, max_bytes: int) -> Iterator[bytes]:
    total = 0
    while True:
        chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
        if not chunk:
            break
        if total + len(chunk) > max_bytes:
            raise RuntimeError(
                f"GitHub response body exceeded the maximum allowed size ({max_bytes} bytes); "
                "download aborted"
            )
        total += len(chunk)
        yield chunk


def _read_capped(resp: object, max_bytes: int) -> bytes:
    return b"".join(_iter_capped_chunks(resp, max_bytes))


def _copy_capped(resp: object, max_bytes: int, out: BinaryIO) -> None:
    for chunk in _iter_capped_chunks(resp, max_bytes):
        out.write(chunk)


def _urlopen(
    req: urllib.request.Request,
    *,
    timeout: float = 60.0,
    max_rate_limit_retries: int = 5,
    max_bytes: int | None = None,
    out: BinaryIO | None = None,
) -> bytes:
    """Fetch bytes for a GitHub request with jittered retry on rate limits."""
    import random
    import time

    _RATE_LIMIT_BASE_DELAY = 10.0
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if max_bytes is not None:
                    if out is not None:
                        _copy_capped(resp, max_bytes, out)
                        return b""
                    return _read_capped(resp, max_bytes)
                if out is not None:
                    shutil.copyfileobj(resp, out, length=_DOWNLOAD_CHUNK_BYTES)
                    return b""
                return resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            remaining = (exc.headers.get("X-RateLimit-Remaining") if exc.headers else None) or ""
            is_rate_limit = exc.code == 429 or (
                exc.code == 403 and (remaining.strip() == "0" or "rate limit" in body.lower())
            )
            is_transient = is_rate_limit or exc.code >= 500
            if is_transient and attempt < max_rate_limit_retries:
                delay = max(
                    _RATE_LIMIT_BASE_DELAY,
                    min(45.0, _RATE_LIMIT_BASE_DELAY * (attempt + 1) * random.uniform(0.5, 1.5)),
                )
                time.sleep(delay)
                attempt += 1
                continue
            if is_rate_limit:
                raise GitHubRateLimitError(
                    f"GitHub API rate limit exceeded ({exc.code}): {body[:300]}"
                ) from exc
            if exc.code >= 500:
                raise GitHubRateLimitError(
                    f"GitHub server error ({exc.code}, transient) after {attempt} retries: {body[:300]}"
                ) from exc
            raise RuntimeError(
                f"GitHub environment request failed ({exc.code}): {body[:500]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < max_rate_limit_retries:
                delay = max(
                    _RATE_LIMIT_BASE_DELAY,
                    min(45.0, _RATE_LIMIT_BASE_DELAY * (attempt + 1) * random.uniform(0.5, 1.5)),
                )
                time.sleep(delay)
                attempt += 1
                continue
            reason = getattr(exc, "reason", exc)
            raise GitHubRateLimitError(
                f"GitHub environment request failed after {attempt} retries (transient network): {reason}"
            ) from exc


def _download_github_tarball(ref: GitHubEnvironmentRef) -> Path:
    url = f"https://api.github.com/repos/{ref.repo_full_name}/tarball/{urllib.parse.quote(ref.ref, safe='')}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "freesolo-flash",
    }
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    tar_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="flash-env-tar-", suffix=".tar.gz", delete=False
        ) as spill:
            tar_path = Path(spill.name)
            _urlopen(
                urllib.request.Request(url, headers=headers),
                timeout=120.0,
                max_bytes=_MAX_TARBALL_BYTES,
                out=spill,
            )
    except BaseException:
        if tar_path is not None:
            with contextlib.suppress(OSError):
                tar_path.unlink()
        raise
    return tar_path


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


def _github_headers(accept: str) -> dict[str, str]:
    headers = {"Accept": accept, "User-Agent": "freesolo-flash"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


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


def _download_github_json(ref: GitHubEnvironmentRef, url: str, context: str) -> Any:
    data = _urlopen(
        urllib.request.Request(url, headers=_github_headers("application/vnd.github+json")),
        timeout=120.0,
        max_bytes=_MAX_CONTENTS_JSON_BYTES,
    )
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "GitHub environment request failed for "
            f"{ref.repo_full_name}@{ref.ref}:{context}: invalid response"
        ) from exc


def _github_response_message(payload: object) -> str:
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str) and message:
            return f" ({message})"
    return ""


def _github_tree_entries(ref: GitHubEnvironmentRef, treeish: str, context: str) -> list[dict]:
    payload = _download_github_json(ref, _github_tree_url(ref, treeish), context)
    if not isinstance(payload, dict) or not isinstance(payload.get("tree"), list):
        raise RuntimeError(
            f"GitHub path {context!r} is not an environment directory"
            f"{_github_response_message(payload)}"
        )
    if payload.get("truncated"):
        raise RuntimeError(
            f"GitHub tree response for environment directory {context!r} was truncated"
        )
    entries = payload["tree"]
    if not all(isinstance(entry, dict) for entry in entries):
        raise RuntimeError("GitHub tree response included an invalid entry")
    return entries


def _resolve_github_directory_tree_sha(ref: GitHubEnvironmentRef, repo_dir: str) -> str:
    treeish = ref.ref
    current = ""
    for part in [part for part in repo_dir.split("/") if part]:
        entries = _github_tree_entries(ref, treeish, current or ref.ref)
        match = next(
            (
                entry
                for entry in entries
                if entry.get("path") == part
                and entry.get("type") == "tree"
                and isinstance(entry.get("sha"), str)
            ),
            None,
        )
        current = f"{current}/{part}" if current else part
        if match is None:
            raise RuntimeError(f"GitHub path {repo_dir!r} is not an environment directory")
        treeish = match["sha"]
    return treeish


def _download_github_directory(ref: GitHubEnvironmentRef, repo_dir: str, dest: Path) -> Path:
    """Download one GitHub directory into a repo-shaped tree under ``dest``."""
    repo_root = dest / "repo"
    root_parts = [part for part in repo_dir.split("/") if part]
    state = {"members": 0, "bytes": 0}

    def record_member(path: str) -> None:
        state["members"] += 1
        if state["members"] > _MAX_ARCHIVE_MEMBERS:
            raise RuntimeError(f"env package has too many members (limit {_MAX_ARCHIVE_MEMBERS})")
        target = (repo_root / path).resolve()
        root = repo_root.resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"unsafe path in environment contents: {path!r}")

    def download_file(path: str, declared_size: object, mode: object = None) -> None:
        record_member(path)
        if (
            isinstance(declared_size, int)
            and declared_size >= 0
            and state["bytes"] + declared_size > _MAX_ARCHIVE_BYTES
        ):
            raise RuntimeError(
                "environment archive is too large uncompressed "
                f"({state['bytes'] + declared_size} bytes; limit {_MAX_ARCHIVE_BYTES} bytes)"
            )
        remaining = _MAX_ARCHIVE_BYTES - state["bytes"]
        target = repo_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            _urlopen(
                urllib.request.Request(
                    _github_contents_url(ref, path),
                    headers=_github_headers("application/vnd.github.raw"),
                ),
                timeout=120.0,
                max_bytes=remaining,
                out=out,
            )
        state["bytes"] += target.stat().st_size
        if state["bytes"] > _MAX_ARCHIVE_BYTES:
            raise RuntimeError(
                "environment archive is too large uncompressed "
                f"({state['bytes']} bytes; limit {_MAX_ARCHIVE_BYTES} bytes)"
            )
        if isinstance(mode, str):
            with contextlib.suppress(ValueError):
                target.chmod(int(mode, 8) & 0o777)

    def create_dir(path: str) -> None:
        record_member(path)
        (repo_root / path).mkdir(parents=True, exist_ok=True)

    create_dir(repo_dir)
    package_tree_sha = _resolve_github_directory_tree_sha(ref, repo_dir)
    payload = _download_github_json(
        ref,
        _github_tree_url(ref, package_tree_sha, recursive=True),
        repo_dir,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("tree"), list):
        raise RuntimeError(
            f"GitHub path {repo_dir!r} is not an environment directory"
            f"{_github_response_message(payload)}"
        )
    if payload.get("truncated"):
        raise RuntimeError(
            f"GitHub tree response for environment directory {repo_dir!r} was truncated"
        )
    for entry in payload["tree"]:
        if not isinstance(entry, dict):
            raise RuntimeError("GitHub tree response included an invalid entry")
        rel_path = entry.get("path")
        if not isinstance(rel_path, str):
            raise RuntimeError("GitHub tree response included an entry without a path")
        child_path = _safe_contents_path(f"{repo_dir}/{rel_path}", root_parts)
        kind = entry.get("type")
        if kind == "tree":
            create_dir(child_path)
        elif kind == "blob" and entry.get("mode") != "120000":
            download_file(child_path, entry.get("size"), entry.get("mode"))
        else:
            raise RuntimeError(f"unsupported entry in environment contents: {child_path!r}")
    return repo_root


class _LimitedReader:
    """Reader wrapper that caps decompressed tar bytes, including header payloads."""

    def __init__(self, raw, limit: int):
        self._raw = raw
        self._remaining = limit

    def read(self, size: int = -1) -> bytes:
        want = self._remaining + 1 if size is None or size < 0 else min(size, self._remaining + 1)
        chunk = self._raw.read(want)
        self._remaining -= len(chunk)
        if self._remaining < 0:
            raise RuntimeError(
                f"environment archive is too large uncompressed (limit {_MAX_ARCHIVE_BYTES} bytes)"
            )
        return chunk


def _extract_github_tarball(ref: GitHubEnvironmentRef, dest: Path, subdir: str = "") -> Path:
    tarball = _download_github_tarball(ref)
    try:
        return _safe_extract_archive(tarball, dest, subdir=subdir)
    finally:
        if isinstance(tarball, Path):
            with contextlib.suppress(OSError):
                tarball.unlink()


def _safe_extract_archive(
    tar_source: bytes | bytearray | Path, dest: Path, subdir: str = ""
) -> Path:
    """Extract a GitHub repo tarball and optionally keep only one repo subdirectory."""
    if isinstance(tar_source, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(prefix="flash-env-tar-", suffix=".tar.gz") as spill:
            spill.write(tar_source)
            spill.seek(0)
            return _safe_extract_archive_file(spill, dest, subdir)
    with tar_source.open("rb") as spill:
        return _safe_extract_archive_file(spill, dest, subdir)


def _safe_extract_archive_file(tar_file: BinaryIO, dest: Path, subdir: str = "") -> Path:
    """Extract a GitHub repo tarball and optionally keep only one repo subdirectory."""
    root = dest.resolve()
    want = [p for p in subdir.split("/") if p] if subdir else []
    top_dirs: set[str] = set()
    total = 0
    extracted = 0
    scanned = 0
    stream_cap = _MAX_ARCHIVE_BYTES + _MAX_ARCHIVE_MEMBERS * 1024 + (1 << 20)
    reader = _LimitedReader(gzip.GzipFile(fileobj=tar_file), stream_cap)
    with tarfile.open(fileobj=reader, mode="r|") as tar:
        for member in tar:
            scanned += 1
            if scanned > _MAX_ARCHIVE_SCAN_MEMBERS:
                raise RuntimeError(
                    f"env package has too many entries to scan (limit {_MAX_ARCHIVE_SCAN_MEMBERS})"
                )
            if member.type in _TAR_METADATA_TYPES:
                continue
            raw = [p for p in member.name.replace("\\", "/").split("/") if p and p != "."]
            if not raw:
                continue
            top_dirs.add(raw[0])
            if want and raw[1 : 1 + len(want)] != want:
                continue
            if ".." in raw:
                raise RuntimeError(f"unsafe path in environment archive: {member.name!r}")
            normalized_name = "/".join(raw)
            target = (dest / normalized_name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe path in environment archive: {member.name!r}")
            if member.islnk() or member.issym() or not (member.isreg() or member.isdir()):
                continue
            extracted += 1
            if extracted > _MAX_ARCHIVE_MEMBERS:
                raise RuntimeError(
                    f"env package has too many members (limit {_MAX_ARCHIVE_MEMBERS})"
                )
            total += max(0, member.size)
            if total > _MAX_ARCHIVE_BYTES:
                raise RuntimeError(
                    f"environment archive is too large uncompressed ({total} bytes; limit {_MAX_ARCHIVE_BYTES} bytes)"
                )
            member.name = normalized_name
            tar.extract(member, dest)
    if len(top_dirs) != 1:
        raise RuntimeError("environment archive had an unexpected layout")
    extracted_dir = dest / next(iter(top_dirs))
    if extracted_dir.exists() and not extracted_dir.is_dir():
        raise RuntimeError("environment archive had an unexpected layout")
    if not extracted_dir.is_dir():
        if want:
            extracted_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise RuntimeError("environment archive did not extract to a directory")
    return extracted_dir


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                total += os.stat(os.path.join(root, name)).st_size
    return total


def _evict_env_cache(keep: Path) -> None:
    # evict least-recently-used cache dirs until both the entry count and total
    # size are under their caps. never remove `keep` (just written) or anything
    # used within the last _CACHE_MIN_AGE_SECONDS (a concurrent run may be
    # loading it); overshooting the caps briefly is safer than deleting live env.
    try:
        entries = [p for p in _CACHE_ROOT.iterdir() if p.is_dir()]
    except OSError:
        return
    now = time.time()
    entries.sort(key=_safe_mtime)  # oldest first
    sizes = {p: _dir_size_bytes(p) for p in entries}
    total = sum(sizes.values())
    count = len(entries)
    for p in entries:
        if count <= _CACHE_MAX_ENTRIES and total <= _CACHE_MAX_BYTES:
            break
        if p == keep or now - _safe_mtime(p) < _CACHE_MIN_AGE_SECONDS:
            continue
        shutil.rmtree(p, ignore_errors=True)
        total -= sizes.get(p, 0)
        count -= 1


def _resolve_github_environment_file(env_ref: str, pinned_sha: str | None = None) -> Path:
    parsed = _parse_github_environment_ref(env_ref)
    if parsed is None:
        raise ValueError(f"not a GitHub environment ref: {env_ref!r}")
    resolved_ref = _resolve_ref_sha(parsed, pinned_sha=pinned_sha)
    package_root = _managed_hub_package_root(parsed)
    if parsed.repo_full_name.lower() == _DEFAULT_MANAGED_ENV_REPO.lower() and not package_root:
        raise ValueError(
            "managed environment hub refs must include a namespace/name environment path"
        )
    cache_scope = "managed-hub" if package_root else "github"
    cache_key = hashlib.sha256(
        f"{cache_scope}:github:{parsed.repo_full_name}@{resolved_ref}:{parsed.path}".encode()
    ).hexdigest()[:24]
    cache_dir = _CACHE_ROOT / cache_key
    env_file = cache_dir / parsed.path
    if env_file.is_dir():
        env_file = env_file / _DEFAULT_ENVIRONMENT_PATH
    if env_file.is_file():
        # mark as recently used so LRU eviction keeps hot envs.
        with contextlib.suppress(OSError):
            os.utime(cache_dir)
        return env_file
    tmp_parent = Path(tempfile.mkdtemp(prefix="flash-env-github-"))
    resolved = GitHubEnvironmentRef(
        parsed.owner,
        parsed.repo,
        resolved_ref,
        parsed.path,
    )
    try:
        if package_root:
            # The shared managed hub can be much larger than one environment. Download only the
            # requested package so worker cache/extraction limits apply to that env, not the hub.
            extracted = _download_github_directory(resolved, package_root, tmp_parent)
        else:
            # Generic GitHub refs keep repo-level sidecars available to relative paths/imports.
            # User-facing pulls filter to the requested env subtree in flash.envs.pull.
            extracted = _extract_github_tarball(resolved, tmp_parent)
        candidate = extracted / parsed.path
        if candidate.is_dir():
            candidate = candidate / _DEFAULT_ENVIRONMENT_PATH
        required_entrypoint = candidate.relative_to(extracted).as_posix()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"environment archive did not contain required entrypoint {required_entrypoint!r}"
            )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.copytree(extracted, cache_dir)
        _evict_env_cache(keep=cache_dir)
        return cache_dir / candidate.relative_to(extracted)
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)


def _resolve_environment_reference(env_ref: str, pinned_sha: str | None = None) -> str:
    if is_managed_environment_slug(env_ref):
        return str(
            _resolve_github_environment_file(managed_slug_to_github_ref(env_ref), pinned_sha)
        )
    parsed = _parse_github_environment_ref(env_ref)
    if parsed is None:
        path = Path(env_ref)
        if path.exists():
            return str(path)
        return env_ref
    return str(_resolve_github_environment_file(env_ref, pinned_sha))


def _resolve_path_arg(value: object, base_dir: Path) -> object:
    if not isinstance(value, str) or not value:
        return value
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme or Path(value).is_absolute():
        return value
    candidate = base_dir / value
    return str(candidate) if candidate.exists() else value


def _load_contract_text(path: str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    if not candidate.is_file():
        return ""
    try:
        return candidate.read_text(encoding="utf-8")
    except UnicodeError:
        return candidate.read_text(errors="replace")


def _import_freesolo_environment_tools():
    try:
        from freesolo.datasets.records import load_task_examples, task_example_from_record
        from freesolo.environments import (
            EnvironmentEpisode,
            EnvironmentMultiTurn,
            EnvironmentSingleTurn,
            EnvironmentTurn,
            load_environment,
        )

        return {
            "EnvironmentEpisode": EnvironmentEpisode,
            "EnvironmentMultiTurn": EnvironmentMultiTurn,
            "EnvironmentSingleTurn": EnvironmentSingleTurn,
            "EnvironmentTurn": EnvironmentTurn,
            "load_environment": load_environment,
            "load_task_examples": load_task_examples,
            "task_example_from_record": task_example_from_record,
        }
    except ImportError as exc:
        raise ImportError(
            "the 'freesolo' package is required to run Freesolo environments; "
            "install it (for example `uv pip install freesolo`) or use a worker image "
            "that includes the Freesolo SDK"
        ) from exc


def _pyproject_dependencies(pyproject_path: Path) -> list[str]:
    if not pyproject_path.is_file():
        return []
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"could not read environment pyproject.toml: {exc}") from exc
    project = pyproject.get("project")
    if not isinstance(project, dict):
        return []
    deps = project.get("dependencies")
    if deps is None:
        return []
    if not isinstance(deps, list) or not all(isinstance(dep, str) for dep in deps):
        raise RuntimeError(
            "environment pyproject.toml [project].dependencies must be a list of strings"
        )
    return [dep.strip() for dep in deps if dep.strip()]


def _dependency_source_key(
    base_dir: Path, pyproject_path: Path, requirements_path: Path
) -> tuple[str, ...]:
    parts = [str(base_dir.resolve())]
    for path in (pyproject_path, requirements_path):
        try:
            st = path.stat()
        except OSError:
            parts.append(f"{path.name}:missing")
        else:
            parts.append(f"{path.name}:{st.st_mtime_ns}:{st.st_size}")
    return tuple(parts)


def _run_dependency_install(base_dir: Path, args: list[str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        *args,
    ]
    print(f"[env-deps] installing environment dependencies: {' '.join(args)}", flush=True)
    try:
        proc = subprocess.run(cmd, cwd=base_dir, timeout=_ENV_DEP_INSTALL_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "environment dependency install timed out "
            f"after {_ENV_DEP_INSTALL_TIMEOUT_S}s: {' '.join(args)}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            "environment dependency install failed "
            f"(exit {proc.returncode}): {' '.join(args)}"
        )


def _install_environment_dependencies(base_dir: Path) -> None:
    pyproject_path = base_dir / "pyproject.toml"
    requirements_path = base_dir / "requirements.txt"
    pyproject_deps = _pyproject_dependencies(pyproject_path)
    has_requirements = requirements_path.is_file()
    if not pyproject_deps and not has_requirements:
        return

    key = _dependency_source_key(base_dir, pyproject_path, requirements_path)
    if key in _INSTALLED_ENV_DEPS:
        return
    if pyproject_deps:
        _run_dependency_install(base_dir, pyproject_deps)
    if has_requirements:
        _run_dependency_install(base_dir, ["-r", str(requirements_path)])
    _INSTALLED_ENV_DEPS.add(key)


def _packaged_dataset_file(base_dir: Path, name: str) -> Path | None:
    """First existing packaged dataset file for split `name`, in the canonical shapes:
    datasets/<name>.jsonl, datasets/<name>.json, <name>.jsonl, <name>.json."""
    for rel in (f"datasets/{name}.jsonl", f"datasets/{name}.json", f"{name}.jsonl", f"{name}.json"):
        candidate = base_dir / rel
        if candidate.is_file():
            return candidate
    return None


def _validate_packaged_dataset_split(split: str) -> str:
    if not _DATASET_SPLIT_RE.fullmatch(split):
        raise ValueError(
            "[environment.params] split must be a simple dataset name "
            "(letters, numbers, '.', '_', '-' only; no slashes or traversal)"
        )
    return split


def load_freesolo_environment(env_id: str, pinned_sha: str | None = None, /, **kwargs):
    # pinned_sha is positional-only so user [environment.params] named "pinned_sha" goes to **kwargs, not here.
    from flash.envs.adapter import FreesoloEnvironment

    reference = _resolve_environment_reference(env_id, pinned_sha)
    reference_path = Path(reference)
    base_dir = reference_path.parent if reference_path.exists() else Path.cwd()
    _install_environment_dependencies(base_dir)
    tools = _import_freesolo_environment_tools()

    params = dict(kwargs)
    source = params.pop("records", None)
    dataset_path = params.get("dataset_path")
    if source is None and dataset_path:
        resolved_dataset_path = _resolve_path_arg(dataset_path, base_dir)
        params["dataset_path"] = resolved_dataset_path
        source = resolved_dataset_path
    # [environment.params] split selects which packaged dataset file Flash trains on. It used to
    # be forwarded to the SDK only, so SFT (and GRPO problem selection driven off dataset())
    # SILENTLY trained on the default datasets/train.jsonl even when a side split was requested.
    split = params.get("split")
    split = split.strip() if isinstance(split, str) else None
    if split:
        split = _validate_packaged_dataset_split(split)
    if source is None:
        wanted = split if split and split != "train" else "train"
        found = _packaged_dataset_file(base_dir, wanted)
        if found is None and wanted != "train" and _packaged_dataset_file(base_dir, "train"):
            # A default train.jsonl exists but the requested split file does not: refuse to fall
            # back silently (that trains on the wrong targets); envs with no packaged dataset at
            # all keep the SDK path, which may implement split itself.
            raise ValueError(
                f"[environment.params] split={split!r} was requested but no "
                f"datasets/{split}.jsonl (or {split}.json) exists in the environment; "
                "refusing to fall back to the default train split. Package the split file "
                "or drop the split param."
            )
        if found is not None:
            params.setdefault("dataset_path", str(found))
            source = str(found)

    contract_path = _resolve_path_arg(params.get("contract_path"), base_dir)
    if isinstance(contract_path, str):
        params["contract_path"] = contract_path
    else:
        params.setdefault("contract_path", str(base_dir / "TRAINING_CONTRACT.md"))
    contract_text = str(
        params.pop("contract_text", "") or _load_contract_text(params["contract_path"])
    )

    sdk_env = tools["load_environment"](reference, **params)
    return FreesoloEnvironment(
        sdk_env,
        env_id,
        source=source,
        contract_text=contract_text,
    )

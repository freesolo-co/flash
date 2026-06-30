"""Adapter that runs Freesolo SDK environments on Flash."""

from __future__ import annotations

import ast
import contextlib
import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import tokenize
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from flash.envs.base import BaseEnvironment
from flash.envs.registry import _FLASH_TRAIN_MAX_EXAMPLES

_DEFAULT_GITHUB_REF = "main"
_DEFAULT_ENVIRONMENT_PATH = "environment.py"
_DEFAULT_MANAGED_ENV_REPO = "freesolo-co/environment-hub"
_CACHE_ROOT = Path(os.environ.get("FLASH_ENV_CACHE_DIR", "/tmp/flash-env-cache"))
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_TARBALL_BYTES = 1024 * 1024 * 1024
_MAX_CONTENTS_JSON_BYTES = 16 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 5000
_MAX_ARCHIVE_SCAN_MEMBERS = 200_000
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_GITHUB_SAFE_PART_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_TAR_METADATA_TYPES = {
    tarfile.XHDTYPE,
    tarfile.XGLTYPE,
    tarfile.GNUTYPE_LONGNAME,
    tarfile.GNUTYPE_LONGLINK,
}
_CANONICAL_INPUT_KEY = "input"
_CANONICAL_OUTPUT_KEY = "output"


def _resolved_import_path(base: Path, module: str | None, level: int) -> Path | None:
    if module is None:
        return None
    parts = module.split(".")
    if any(not part.isidentifier() for part in parts):
        return None

    root = base.parent
    for _ in range(max(level - 1, 0)):
        root = root.parent

    candidate = root.joinpath(*parts)
    file_candidate = candidate.with_suffix(".py")
    if file_candidate.is_file():
        return file_candidate
    init_candidate = candidate / "__init__.py"
    if init_candidate.is_file():
        return init_candidate
    return None


def _load_environment_params(
    path: Path, target_name: str = "load_environment", seen: set[Path] | None = None
) -> tuple[set[str], bool]:
    seen = seen or set()
    resolved_path = path.resolve()
    if resolved_path in seen or len(seen) >= 8:
        return set(), False
    seen.add(resolved_path)
    try:
        with tokenize.open(path) as source:
            tree = ast.parse(source.read(), filename=os.fspath(path))
    except Exception:
        return set(), False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == target_name:
            args = node.args
            names = {
                arg.arg
                for arg in (
                    *args.posonlyargs,
                    *args.args,
                    *args.kwonlyargs,
                )
            }
            return names, args.kwarg is not None
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) != target_name:
                    continue
                import_path = _resolved_import_path(path, node.module, node.level)
                if import_path is None:
                    continue
                return _load_environment_params(import_path, alias.name, seen)
    return set(), False


def _apply_training_max_examples(reference: str, params: dict[str, Any]) -> dict[str, Any]:
    if _FLASH_TRAIN_MAX_EXAMPLES not in params:
        return params

    params.pop(_FLASH_TRAIN_MAX_EXAMPLES)
    if "max_examples" in params or "limit" in params:
        return params

    ref_path = Path(reference)
    names, accepts_kwargs = _load_environment_params(ref_path) if ref_path.is_file() else (set(), False)
    # Flash applies train.max_examples after a fixed-seed shuffle; passing that value into
    # loaders would pre-truncate the dataset and change which rows are sampled.
    if "max_examples" in names:
        params["max_examples"] = None
    if "limit" in names:
        params["limit"] = None
    if not names.intersection({"max_examples", "limit"}) and accepts_kwargs:
        params["max_examples"] = None
    return params


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
    payload = _download_github_json(
        ref,
        _github_tree_url(ref, f"{ref.ref}:{repo_dir}", recursive=True),
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


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


class _ScoredResponseText(str):
    """String-compatible response passed to SDK scorers.

    The string value is the answer-only completion so existing graders keep their old behavior.
    Thinking-aware scorers can opt into the structured views.
    """

    completion: str
    thinking: str | None
    raw: str

    def __new__(cls, completion: str, *, raw: str, thinking: str | None):
        obj = str.__new__(cls, completion)
        obj.completion = completion
        obj.thinking = thinking
        obj.raw = raw
        return obj


def _completion_for_scoring(completion: str, state: dict | None) -> str:
    if state:
        raw = state.get("raw")
        if not isinstance(raw, str):
            return completion
        answer = state.get("completion")
        thinking = state.get("thinking")
        return _ScoredResponseText(
            answer if isinstance(answer, str) else completion,
            raw=raw,
            thinking=thinking if isinstance(thinking, str) else None,
        )
    return completion


class FreesoloEnvironment(BaseEnvironment):
    """Flash environment backed by ``freesolo.environments``."""

    def __init__(
        self,
        sdk_env: object,
        env_id: str,
        *,
        source: object | None,
        contract_text: str = "",
    ):
        super().__init__(id=env_id)
        self._env = sdk_env
        self._source = source
        self._contract_text = contract_text
        tools = _import_freesolo_environment_tools()
        self._task_example_from_record = tools["task_example_from_record"]
        self._load_task_examples = tools["load_task_examples"]
        self._EnvironmentEpisode = tools["EnvironmentEpisode"]
        self._EnvironmentMultiTurn = tools["EnvironmentMultiTurn"]
        self._EnvironmentTurn = tools["EnvironmentTurn"]
        self.multi_turn = isinstance(sdk_env, tools["EnvironmentMultiTurn"])
        self.is_tool_env = False
        self._max_turns_cache: int | None = None
        self._dataset_cache: list[dict] | None = None

    @property
    def max_turns(self) -> int:
        """Batch-level turn ceiling: dataset-wide max of per-example budgets, clamped to [8, 64]."""
        if self._max_turns_cache is not None:
            return self._max_turns_cache
        cap = 8
        if self.multi_turn:
            cap = 24
            best: int | None = None
            for ex in self.dataset():
                try:
                    turns = int(self._env.max_episode_turns(self._task_example(ex)))
                except Exception:
                    continue
                if best is None or turns > best:
                    best = turns
            if best is not None:
                cap = max(8, min(64, best))
        self._max_turns_cache = cap
        return cap

    def _task_example(self, example: dict):
        return self._task_example_from_record(self._canonical_record(example))

    @staticmethod
    def _canonical_record(record: dict) -> dict:
        raw = dict(record)
        canonical = {}
        if _CANONICAL_INPUT_KEY not in raw:
            raise ValueError("Freesolo dataset records must contain an input field")
        canonical[_CANONICAL_INPUT_KEY] = raw[_CANONICAL_INPUT_KEY]
        if _CANONICAL_OUTPUT_KEY in raw:
            canonical[_CANONICAL_OUTPUT_KEY] = raw[_CANONICAL_OUTPUT_KEY]
        if raw.get("id") is not None:
            canonical["id"] = raw["id"]
        metadata = raw.get("metadata")
        if isinstance(metadata, dict) and metadata:
            canonical["metadata"] = metadata
        return canonical

    def _reward_to_breakdown(self, reward) -> dict[str, float]:
        out: dict[str, float] = {}
        for metric in getattr(reward, "metrics", ()) or ():
            score = getattr(metric, "score", None)
            if score is not None:
                name = str(getattr(metric, "name", "") or "metric")
                key = name
                idx = 1
                while key in out:
                    idx += 1
                    key = f"{name}_{idx}"
                out[key] = float(score)
        out["total"] = float(getattr(reward, "score", 0.0))
        return out

    def dataset(self) -> list[dict]:
        if self._dataset_cache is not None:
            return self._dataset_cache
        if self._source is None:
            rows = getattr(self._env, "dataset", None) or getattr(self._env, "examples", None)
            if rows is None:
                raise ValueError(
                    "Freesolo environment has no dataset source. Set "
                    "[environment.params] dataset_path or records so Flash can train."
                )
            examples = self._load_task_examples(rows)
        else:
            examples = self._load_task_examples(self._source)
        records = []
        for example in examples:
            raw = dict(getattr(example, "record", {}) or {})
            if _CANONICAL_INPUT_KEY not in raw and getattr(example, "input", None) is not None:
                raw[_CANONICAL_INPUT_KEY] = example.input
            if getattr(example, "id", None) is not None:
                raw.setdefault("id", example.id)
            if getattr(example, "output", None) is not None:
                raw.setdefault(_CANONICAL_OUTPUT_KEY, _json_safe(example.output))
            metadata = getattr(example, "metadata", None)
            if isinstance(metadata, dict) and metadata:
                raw.setdefault("metadata", metadata)
            record = self._canonical_record(raw)
            records.append(record)
        self._dataset_cache = records
        return records

    def prompt_messages(self, example: dict) -> list[dict]:
        messages = self._env.start_episode(self._task_example(example), self._contract_text)
        return [dict(message) for message in messages]

    def sft_completion(self, example: dict) -> list[dict]:
        """Target completion messages for one SFT example; falls back to raw record output."""
        fn = getattr(self._env, "sft_completion", None)
        if callable(fn):
            msgs = fn(self._task_example(example))
            if msgs:
                return [dict(m) for m in msgs]
        value = example.get(_CANONICAL_OUTPUT_KEY)
        if isinstance(value, list) and value and all(isinstance(m, dict) for m in value):
            return [dict(m) for m in value]
        if (
            isinstance(value, dict)
            and list(value) == ["messages"]
            and isinstance(value["messages"], list)
        ):
            return [dict(m) for m in value["messages"]]
        return [{"role": "assistant", "content": "" if value is None else str(value)}]

    def _single(self, results, method: str):
        if len(results) != 1:
            raise RuntimeError(f"Freesolo environment {method} returned the wrong length")
        return results[0]

    def _score_one(self, completion: str, example: dict, state: dict | None):
        if state and self.multi_turn:
            return self._score_episode(example, state)
        rewards = self._env.score_responses(
            self._task_example(example), [_completion_for_scoring(completion, state)]
        )
        return self._single(rewards, "score_responses")

    def scores_breakdown(
        self, completion: str, example: dict, state: dict | None = None
    ) -> dict[str, float]:
        return self._reward_to_breakdown(self._score_one(completion, example, state))

    def reward(self, completion: str, example: dict, state: dict | None = None) -> float:
        return float(getattr(self._score_one(completion, example, state), "score", 0.0))

    def _grouped_score(self, items, *, task_of, payload_of, scorer, method: str) -> list[float]:
        """Group rollouts that share an example (input order preserved), score each group with ONE
        concurrent ``scorer(task, payloads)`` call, then scatter rewards back to per-item order.
        ``task_of(ex, st)`` builds a group's task; ``payload_of(st)`` its per-rollout payload."""
        groups: dict[str, dict] = {}
        order: list[str] = []
        for i, (ex, st) in enumerate(items):
            key = json.dumps(ex, sort_keys=True, default=str)
            grp = groups.get(key)
            if grp is None:
                grp = groups[key] = {"task": task_of(ex, st), "idxs": [], "payloads": []}
                order.append(key)
            grp["idxs"].append(i)
            grp["payloads"].append(payload_of(st))
        out: list[float] = [0.0] * len(items)
        for key in order:
            grp = groups[key]
            rewards = scorer(grp["task"], grp["payloads"])
            if len(rewards) != len(grp["payloads"]):
                raise RuntimeError(f"Freesolo environment {method} returned the wrong length")
            for idx, rw in zip(grp["idxs"], rewards, strict=True):
                out[idx] = float(rw.score)
        return out

    def reward_many(self, items: list[tuple[dict, dict]]) -> list[float]:
        """Reward for many ``(example, state)`` rollouts at once, in input order.

        Rollouts that share a task go through ONE batched scoring call, which the env scores
        concurrently (``Environment.max_score_concurrency``) — replacing one blocking scoring call
        per rollout. For a judge / network-reward env (where scoring dominates) this is the analogue
        of batched generation: a GRPO group's whole completion set overlaps its judge round-trips
        instead of N serial GPU-idle calls. Multi-turn groups go through ``score_episodes``,
        single-turn through ``score_responses`` (an episode's reward is just ``score_response`` on
        its final text, so the two are equivalent for the one-prompt-one-response case). Equals one
        :meth:`reward` per item: each path scores every rollout independently — ``score_responses``
        runs ``score_response`` per completion and ``_reward_to_breakdown(...)['total']`` is exactly
        ``reward.score`` — so batching changes only concurrency, not values.

        Honors ``reward_thread_safe``: an env whose scorer keeps mutable or thread-bound state opts out
        with ``reward_thread_safe = False`` and MUST NOT be raced. Batching a group's whole completion
        set into one ``score_responses`` / ``score_episodes`` call hands them to the env's concurrent
        scorer (``max_score_concurrency``), so for an opted-out env we fall back to the proven serial
        path — one single-item :meth:`reward` per rollout, in input order — exactly as the pre-batching
        code did. Same values; only the concurrency is dropped."""
        if not self.reward_thread_safe:
            # Single-item scoring per rollout (each reward() makes a ONE-element score_responses /
            # score_episodes call, so the env's concurrent scorer never sees a batch to parallelize).
            # reward() reads the rollout's own response_text/episode from its state, like the batched
            # paths below — passing it as the completion is a no-op for the multi-turn (state) branch.
            return [self.reward(str(st.get("response_text") or ""), ex, st) for ex, st in items]
        if not self.multi_turn:
            return self._grouped_score(
                items,
                task_of=lambda ex, st: self._task_example(ex),
                payload_of=lambda st: _completion_for_scoring(
                    str(st.get("response_text") or ""), st
                ),
                scorer=self._env.score_responses,
                method="score_responses",
            )
        return self._grouped_score(
            items,
            task_of=lambda ex, st: st.get("task") or self._task_example(ex),
            payload_of=self._episode_from_state,
            scorer=self._env.score_episodes,
            method="score_episodes",
        )

    @property
    def reward_thread_safe(self) -> bool:
        """Whether reward() may be called concurrently; delegates to the underlying env."""
        return bool(getattr(self._env, "reward_thread_safe", True))

    def grade(self, completion: str, example: dict, state: dict | None = None) -> bool:
        return bool(self._score_one(completion, example, state).resolved_success())

    def tools(self) -> list:
        return []

    def new_rollout_state(self, example: dict) -> dict:
        task = self._task_example(example)
        prompt = [dict(message) for message in self._env.start_episode(task, self._contract_text)]
        try:
            episode_turns: int | None = int(self._env.max_episode_turns(task))
        except Exception:
            episode_turns = None
        return {
            "task": task,
            "prompt": [dict(message) for message in prompt],
            "messages": [dict(message) for message in prompt],
            "turns": [],
            "done": False,
            "response_text": "",
            "turn": 0,
            "max_episode_turns": episode_turns,
        }

    def record_model_turn(self, state: dict, content: str) -> dict:
        msg = {"role": "assistant", "content": content}
        state.setdefault("messages", []).append(msg)
        state.setdefault("turns", []).append(
            self._EnvironmentTurn(role="assistant", content=content)
        )
        state["response_text"] = content
        return msg

    def env_reply(self, messages: list[dict], state: dict) -> list[dict]:
        if not self.multi_turn:
            return []
        task = state.get("task")
        if task is None:
            raise RuntimeError("missing Freesolo rollout task state")
        assistant_response = str(state.get("response_text") or "")
        step = self._env.step_episode(task, list(messages), assistant_response)
        state["done"] = bool(step.done)
        if step.final_response_text is not None:
            state["response_text"] = step.final_response_text
        state["turn"] = int(state.get("turn", 0)) + 1
        if step.metadata:
            state.setdefault("step_metadata", []).append(step.metadata)
        replies = [dict(message) for message in step.messages]
        state.setdefault("messages", []).extend(replies)
        for message in replies:
            state.setdefault("turns", []).append(
                self._EnvironmentTurn(
                    role=str(message.get("role", "")),
                    content=str(message.get("content", "")),
                )
            )
        return replies

    def rollout_done(self, state: dict, max_turns: int | None = None) -> bool:
        if not self.multi_turn:
            return True
        if bool(state.get("done")):
            return True
        # Per-example budget takes precedence over batch-wide cap.
        cap = state.get("max_episode_turns")
        if cap is None:
            cap = max_turns
        return cap is not None and int(state.get("turn", 0)) >= int(cap)

    def _episode_from_state(self, state: dict):
        return self._EnvironmentEpisode(
            messages=tuple(state.get("messages") or ()),
            response_text=str(state.get("response_text") or ""),
            turns=tuple(state.get("turns") or ()),
            metadata={"steps": state.get("step_metadata", [])}
            if state.get("step_metadata")
            else {},
        )

    def _score_episode(self, example: dict, state: dict):
        task = state.get("task") or self._task_example(example)
        rewards = self._env.score_episodes(task, [self._episode_from_state(state)])
        return self._single(rewards, "score_episodes")

    def reward_from_messages(
        self, completion_msgs: list[dict], example: dict, prompt_msgs: list[dict] | None = None
    ) -> float:
        messages = [*(prompt_msgs or []), *completion_msgs]
        response_text = ""
        turns = []
        for message in completion_msgs:
            content = str(message.get("content", ""))
            role = str(message.get("role", ""))
            turns.append(self._EnvironmentTurn(role=role, content=content))
            if role == "assistant":
                response_text = content
        episode = self._EnvironmentEpisode(
            messages=tuple(dict(m) for m in messages),
            response_text=response_text,
            turns=tuple(turns),
        )
        rewards = self._env.score_episodes(self._task_example(example), [episode])
        return float(self._single(rewards, "score_episodes").score)


def load_freesolo_environment(
    env_id: str, pinned_sha: str | None = None, /, **kwargs
) -> FreesoloEnvironment:
    # pinned_sha is positional-only so user [environment.params] named "pinned_sha" goes to **kwargs, not here.
    tools = _import_freesolo_environment_tools()
    reference = _resolve_environment_reference(env_id, pinned_sha)
    reference_path = Path(reference)
    base_dir = reference_path.parent if reference_path.exists() else Path.cwd()

    params = _apply_training_max_examples(reference, dict(kwargs))
    source = params.pop("records", None)
    dataset_path = params.get("dataset_path")
    if source is None and dataset_path:
        resolved_dataset_path = _resolve_path_arg(dataset_path, base_dir)
        params["dataset_path"] = resolved_dataset_path
        source = resolved_dataset_path
    if source is None:
        for rel in (
            "datasets/train.jsonl",
            "datasets/train.json",
            "train.jsonl",
            "train.json",
        ):
            candidate = base_dir / rel
            if candidate.is_file():
                params.setdefault("dataset_path", str(candidate))
                source = str(candidate)
                break

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


__all__ = [
    "FreesoloEnvironment",
    "GitHubEnvironmentRef",
    "is_freesolo_environment_id",
    "is_github_environment_ref",
    "is_managed_environment_slug",
    "load_freesolo_environment",
    "managed_slug_to_github_ref",
]

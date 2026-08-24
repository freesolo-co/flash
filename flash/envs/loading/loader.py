"""Loading and reference resolution for Freesolo SDK environments.

Resolves managed slugs / github: refs / URLs to local files (with a bounded on-disk cache),
probes packaged datasets, and constructs the :class:`~flash.envs.loading.adapter.FreesoloEnvironment`.
Split out of ``flash.envs.adapter`` (which keeps the runtime environment class); the public
names remain importable from ``flash.envs.adapter``.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from flash.envs.meta import cache_security

# the underscored names are re-exports: tests and content.multimodal reach them via loader.
from flash.envs.meta.dataset_selection import (
    _packaged_dataset_file as _packaged_dataset_file,
)
from flash.envs.meta.dataset_selection import (
    _plural_dataset_file as _plural_dataset_file,
)
from flash.envs.meta.dataset_selection import (
    _validate_packaged_dataset_split as _validate_packaged_dataset_split,
)
from flash.envs.meta.dataset_selection import (
    env_dataset_rows,
    select_dataset_source,
)

# the env-id grammar lives in its own module; the underscored names are re-exports because
# tests and sibling modules have always reached them through loader.
from flash.envs.meta.identity import (
    _DEFAULT_ENVIRONMENT_PATH as _DEFAULT_ENVIRONMENT_PATH,
)
from flash.envs.meta.identity import (
    _DEFAULT_GITHUB_REF as _DEFAULT_GITHUB_REF,
)
from flash.envs.meta.identity import (
    _DEFAULT_MANAGED_ENV_REPO as _DEFAULT_MANAGED_ENV_REPO,
)
from flash.envs.meta.identity import (
    _GITHUB_SAFE_PART_RE as _GITHUB_SAFE_PART_RE,
)
from flash.envs.meta.identity import (
    GitHubEnvironmentRef as GitHubEnvironmentRef,
)
from flash.envs.meta.identity import (
    GitHubPermanentError as GitHubPermanentError,
)
from flash.envs.meta.identity import (
    GitHubRateLimitError as GitHubRateLimitError,
)
from flash.envs.meta.identity import (
    GitHubTransientError as GitHubTransientError,
)
from flash.envs.meta.identity import (
    GitHubUnavailableError as GitHubUnavailableError,
)
from flash.envs.meta.identity import (
    _is_safe_github_path_parts as _is_safe_github_path_parts,
)
from flash.envs.meta.identity import (
    _managed_environment_ref_error as _managed_environment_ref_error,
)
from flash.envs.meta.identity import (
    _normalize_env_path as _normalize_env_path,
)
from flash.envs.meta.identity import (
    _parse_github_environment_ref as _parse_github_environment_ref,
)
from flash.envs.meta.identity import (
    _parse_managed_environment_slug as _parse_managed_environment_slug,
)
from flash.envs.meta.identity import (
    _targets_managed_environment_repo as _targets_managed_environment_repo,
)
from flash.envs.meta.identity import (
    canonical_managed_environment_slug as canonical_managed_environment_slug,
)
from flash.envs.meta.identity import (
    is_commit_sha as is_commit_sha,
)
from flash.envs.meta.identity import (
    is_freesolo_environment_id as is_freesolo_environment_id,
)
from flash.envs.meta.identity import (
    is_github_environment_ref as is_github_environment_ref,
)
from flash.envs.meta.identity import (
    is_managed_environment_slug as is_managed_environment_slug,
)
from flash.envs.meta.identity import (
    managed_slug_to_github_ref as managed_slug_to_github_ref,
)
from flash.envs.package.limits import (
    ARCHIVE_MEMBER_LIMIT,
    ARCHIVE_SCAN_MEMBER_LIMIT,
    LimitedArchiveReader,
    archive_stream_limit,
)
from flash.envs.package.unpack import extract_validated_archive_members

_CACHE_ROOT_DIR_NAME = "env-cache"


def _default_cache_root() -> Path:
    """where the on-disk env cache lives: a directory private to the current user.

    the cache holds code that ``load_environment`` imports and executes, and its keys are
    fully predictable (a sha of a public repo/ref/path), so a shared world-writable root
    would let any other local account pre-create the tree and plant an ``environment.py``
    that the cache-hit path hands back with no network call and no integrity check. prefer a
    home-owned location; worker containers can be homeless, so fall back to a uid-suffixed
    dir under the temp root rather than a shared name.

    deliberately not env-tunable, see ``_ensure_cache_root``.
    """
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg and Path(xdg).is_absolute():
        # vetted exactly like the home branch below: XDG_CACHE_HOME is commonly inherited from
        # a container image and points at another account's home, which an arbitrary uid cannot
        # create under. selecting it unconditionally means _ensure_cache_root dies with
        # PermissionError instead of falling through to the uid-scoped temp root.
        xdg_root = Path(xdg) / "flash" / _CACHE_ROOT_DIR_NAME
        if cache_security.cache_root_is_creatable(xdg_root):
            return xdg_root
    home = Path(os.path.expanduser("~"))
    if home.is_absolute() and home.is_dir():
        # the whole path is vetted, not just `home`: an existing root-owned `~/.cache` makes
        # the home-based root uncreatable even when home itself is fine.
        home_root = home / ".cache" / "flash" / _CACHE_ROOT_DIR_NAME
        if cache_security.cache_root_is_creatable(home_root):
            return home_root
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(tempfile.gettempdir()) / f"flash-env-cache-{uid}"


_CACHE_ROOT = _default_cache_root()
# bound the on-disk env cache so it cannot grow without limit (one subdir per env
# content-sha, ~30-80 MB each). evicted LRU by dir mtime, which we bump on cache hit.
_CACHE_MAX_ENTRIES = 32
_CACHE_MAX_BYTES = 4 * 1024 * 1024 * 1024
# never evict an entry used within this window; a concurrent run may be loading it.
_CACHE_MIN_AGE_SECONDS = 600
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_TARBALL_BYTES = 1024 * 1024 * 1024
_MAX_CONTENTS_JSON_BYTES = 16 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_ARCHIVE_MEMBERS = ARCHIVE_MEMBER_LIMIT
_MAX_ARCHIVE_SCAN_MEMBERS = ARCHIVE_SCAN_MEMBER_LIMIT


def _resolve_ref_sha(
    parsed: GitHubEnvironmentRef,
    pinned_sha: str | None = None,
    *,
    timeout: float = 60.0,
    max_rate_limit_retries: int = 5,
    deadline_at: float | None = None,
) -> str:
    # Control plane pins sha once; workers skip GitHub entirely on a fan-out.
    if pinned_sha and is_commit_sha(pinned_sha):
        return pinned_sha
    if is_commit_sha(parsed.ref):
        return parsed.ref
    # Symbolic refs are NOT cached in-process: managed slugs point at environment-hub@main which moves.
    headers = _github_headers("application/vnd.github+json")
    commit_url = f"https://api.github.com/repos/{parsed.repo_full_name}/commits/{urllib.parse.quote(parsed.ref, safe='')}"
    req = urllib.request.Request(commit_url, headers=headers)
    request_options = {
        "timeout": timeout,
        "max_rate_limit_retries": max_rate_limit_retries,
    }
    if deadline_at is not None:
        request_options["deadline_at"] = deadline_at
    data = _urlopen(req, **request_options)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to resolve GitHub environment ref {parsed.canonical()}: invalid response"
        ) from exc
    sha = payload.get("sha")
    if not isinstance(sha, str) or not is_commit_sha(sha):
        raise RuntimeError(f"Failed to resolve GitHub environment ref {parsed.canonical()}")
    return sha


def _iter_capped_chunks(resp: object, max_bytes: int) -> Iterator[bytes]:
    # per-attempt accounting is also per-download accounting: _urlopen rewinds and truncates the
    # sink before every attempt, so the bytes this generator caps are exactly the bytes that
    # survive on disk. a retried download can never leave more than max_bytes behind.
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


def _urlopen(
    req: urllib.request.Request,
    *,
    timeout: float = 60.0,
    max_rate_limit_retries: int = 5,
    max_bytes: int | None = None,
    out: BinaryIO | None = None,
    deadline_at: float | None = None,
) -> bytes:
    """Fetch bytes for a GitHub request with jittered retry on rate limits.

    ``out`` must be seekable: a failure part-way through the body is retried from scratch, and the
    sink is rewound and truncated first so a partial prefix can never be concatenated with the
    retry's full body.
    """
    import random

    _RATE_LIMIT_BASE_DELAY = 10.0

    def backoff_delay(attempt: int) -> float:
        return max(
            _RATE_LIMIT_BASE_DELAY,
            min(45.0, _RATE_LIMIT_BASE_DELAY * (attempt + 1) * random.uniform(0.5, 1.5)),
        )

    def drain(resp) -> bytes:
        """Consume the response, honouring the byte cap and the caller's sink.

        Returns ``b""`` whenever ``out`` is given: the bytes went to the file, and also returning
        them would hold the whole download in memory, which is the thing streaming to ``out`` avoids.
        """
        chunks = _iter_capped_chunks(resp, max_bytes) if max_bytes is not None else None
        if out is None:
            return b"".join(chunks) if chunks is not None else resp.read()
        if chunks is not None:
            for chunk in chunks:
                out.write(chunk)
        else:
            shutil.copyfileobj(resp, out, length=_DOWNLOAD_CHUNK_BYTES)
        return b""

    # every attempt writes the whole body from this offset, so the sink holds one attempt's bytes.
    sink_start = out.tell() if out is not None else 0

    def remaining_timeout() -> float:
        if deadline_at is None:
            return timeout
        remaining = float(deadline_at) - time.time()
        if remaining <= 0:
            raise GitHubUnavailableError(
                "GitHub environment request exceeded the authoritative run deadline"
            )
        return min(timeout, remaining)

    def sleep_before_retry(delay: float) -> None:
        if deadline_at is None:
            time.sleep(delay)
            return
        remaining = float(deadline_at) - time.time()
        if remaining <= 0:
            raise GitHubUnavailableError(
                "GitHub environment request exceeded the authoritative run deadline"
            )
        time.sleep(min(delay, remaining))

    attempt = 0
    while True:
        try:
            request_timeout = remaining_timeout()
            if out is not None:
                out.seek(sink_start)
                out.truncate()
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                return drain(resp)
        except urllib.error.HTTPError as exc:
            # urllib can raise an HTTPError with fp=None; exc.read() is an AttributeError there.
            body = exc.read().decode("utf-8", "replace") if exc.fp is not None else ""
            remaining = (exc.headers.get("X-RateLimit-Remaining") if exc.headers else None) or ""
            is_rate_limit = exc.code == 429 or (
                exc.code == 403 and (remaining.strip() == "0" or "rate limit" in body.lower())
            )
            is_transient = is_rate_limit or exc.code >= 500
            if is_transient and attempt < max_rate_limit_retries:
                sleep_before_retry(backoff_delay(attempt))
                attempt += 1
                continue
            if is_rate_limit:
                raise GitHubRateLimitError(
                    f"GitHub API rate limit exceeded ({exc.code}): {body[:300]}"
                ) from exc
            if exc.code >= 500:
                raise GitHubUnavailableError(
                    f"GitHub server error ({exc.code}, transient) after {attempt} retries: {body[:300]}"
                ) from exc
            # 404 (no such repo, or one the token cannot read) and 422 (a ref that does not exist,
            # e.g. "No commit found for SHA: main" on a master-default repo) are settled answers.
            # 401 and a non-rate-limit 403 join them: the rate-limit shapes are already claimed
            # above, so what is left is a token this plane cannot fix by waiting -- invalid,
            # expired, or lacking the scope. deferring those rents a GPU to rediscover the same
            # credential error on the worker.
            # Typed so the submit-time pin can fail closed on them while still deferring a blip;
            # every other code keeps the untyped RuntimeError it has always raised.
            if exc.code in (401, 403, 404, 422):
                raise GitHubPermanentError(
                    f"GitHub environment request failed ({exc.code}): {body[:500]}"
                ) from exc
            raise RuntimeError(
                f"GitHub environment request failed ({exc.code}): {body[:500]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < max_rate_limit_retries:
                sleep_before_retry(backoff_delay(attempt))
                attempt += 1
                continue
            reason = getattr(exc, "reason", exc)
            raise GitHubUnavailableError(
                f"GitHub environment request failed after {attempt} retries (transient network): {reason}"
            ) from exc


def _download_github_tarball(
    ref: GitHubEnvironmentRef,
    *,
    deadline_at: float | None = None,
) -> Path:
    url = f"https://api.github.com/repos/{ref.repo_full_name}/tarball/{urllib.parse.quote(ref.ref, safe='')}"
    headers = _github_headers("application/vnd.github+json")
    tar_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="flash-env-tar-", suffix=".tar.gz", delete=False
        ) as spill:
            tar_path = Path(spill.name)
            request_options = {
                "timeout": 120.0,
                "max_bytes": _MAX_TARBALL_BYTES,
                "out": spill,
            }
            if deadline_at is not None:
                request_options["deadline_at"] = deadline_at
            _urlopen(
                urllib.request.Request(url, headers=headers),
                **request_options,
            )
    except BaseException:
        if tar_path is not None:
            with contextlib.suppress(OSError):
                tar_path.unlink()
        raise
    return tar_path


def _managed_hub_package_root(ref: GitHubEnvironmentRef) -> str:
    """The directory holding ONE environment's package: ``<org>/<project>/<name>``.

    All three segments, not two: the package root is what gets downloaded and copied into the
    cache entry, and ``<org>/<project>`` is the project directory holding every environment
    the project has published. Stopping at two would fetch all of them to import one.
    """
    if ref.repo_full_name.lower() != _DEFAULT_MANAGED_ENV_REPO.lower():
        return ""
    parts = [part for part in ref.path.split("/") if part]
    if len(parts) < 3 or not _is_safe_github_path_parts(tuple(parts[:3])):
        return ""
    return "/".join(parts[:3])


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


def _github_token() -> str | None:
    """The GitHub token, or ``None`` when unset OR blank.

    Blank must collapse to ``None``, not fall through as a truthy string: a whitespace-only
    GITHUB_TOKEN would otherwise build ``Authorization: Bearer <whitespace>``, and GitHub REJECTS a
    malformed credential rather than treating the request as anonymous - so a public repo that
    loads fine with no token at all would fail with one that is merely blank.
    """
    return (os.environ.get("GITHUB_TOKEN") or "").strip() or None


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


def _download_github_json(
    ref: GitHubEnvironmentRef,
    url: str,
    context: str,
    *,
    timeout: float = 120.0,
    max_rate_limit_retries: int = 5,
    deadline_at: float | None = None,
) -> object:
    request_options = {"timeout": timeout, "max_bytes": _MAX_CONTENTS_JSON_BYTES}
    if deadline_at is not None:
        request_options["deadline_at"] = deadline_at
    if max_rate_limit_retries != 5:
        request_options["max_rate_limit_retries"] = max_rate_limit_retries
    data = _urlopen(
        urllib.request.Request(url, headers=_github_headers("application/vnd.github+json")),
        **request_options,
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


def _github_tree_entries(
    ref: GitHubEnvironmentRef,
    treeish: str,
    context: str,
    *,
    recursive: bool = False,
    timeout: float = 120.0,
    max_rate_limit_retries: int = 5,
    deadline_at: float | None = None,
) -> list[dict]:
    request_options = {
        "timeout": timeout,
        "max_rate_limit_retries": max_rate_limit_retries,
    }
    if deadline_at is not None:
        request_options["deadline_at"] = deadline_at
    payload = _download_github_json(
        ref,
        _github_tree_url(ref, treeish, recursive=recursive),
        context,
        **request_options,
    )
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


def _resolve_github_directory_tree_sha(
    ref: GitHubEnvironmentRef,
    repo_dir: str,
    *,
    deadline_at: float | None = None,
) -> str:
    treeish = ref.ref
    current = ""
    for part in [part for part in repo_dir.split("/") if part]:
        request_options = {"deadline_at": deadline_at} if deadline_at is not None else {}
        entries = _github_tree_entries(
            ref,
            treeish,
            current or ref.ref,
            **request_options,
        )
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


def _download_github_directory(
    ref: GitHubEnvironmentRef,
    repo_dir: str,
    dest: Path,
    *,
    deadline_at: float | None = None,
) -> Path:
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
            request_options = {
                "timeout": 120.0,
                "max_bytes": remaining,
                "out": out,
            }
            if deadline_at is not None:
                request_options["deadline_at"] = deadline_at
            _urlopen(
                urllib.request.Request(
                    _github_contents_url(ref, path),
                    headers=_github_headers("application/vnd.github.raw"),
                ),
                **request_options,
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
    request_options = {"deadline_at": deadline_at} if deadline_at is not None else {}
    package_tree_sha = _resolve_github_directory_tree_sha(
        ref,
        repo_dir,
        **request_options,
    )
    payload = _download_github_json(
        ref,
        _github_tree_url(ref, package_tree_sha, recursive=True),
        repo_dir,
        **request_options,
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


def _extract_github_tarball(
    ref: GitHubEnvironmentRef,
    dest: Path,
    *,
    deadline_at: float | None = None,
) -> Path:
    request_options = {"deadline_at": deadline_at} if deadline_at is not None else {}
    tarball = _download_github_tarball(ref, **request_options)
    try:
        return _safe_extract_archive(tarball, dest)
    finally:
        if isinstance(tarball, Path):
            with contextlib.suppress(OSError):
                tarball.unlink()


def _safe_extract_archive(tar_source: bytes | bytearray | Path, dest: Path) -> Path:
    """Extract a GitHub repo tarball; expects a single top-level repo directory."""
    if isinstance(tar_source, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(prefix="flash-env-tar-", suffix=".tar.gz") as spill:
            spill.write(tar_source)
            spill.seek(0)
            return _safe_extract_archive_file(spill, dest)
    with tar_source.open("rb") as spill:
        return _safe_extract_archive_file(spill, dest)


def _safe_extract_archive_file(tar_file: BinaryIO, dest: Path) -> Path:
    """Extract a GitHub repo tarball; expects a single top-level repo directory."""
    top_dirs: set[str] = set()
    reader = LimitedArchiveReader(
        gzip.GzipFile(fileobj=tar_file),
        archive_stream_limit(_MAX_ARCHIVE_BYTES, _MAX_ARCHIVE_MEMBERS),
        lambda: RuntimeError(
            f"environment archive is too large uncompressed (limit {_MAX_ARCHIVE_BYTES} bytes)"
        ),
    )
    extract_validated_archive_members(
        reader,
        extract_base=dest,
        content_byte_limit=_MAX_ARCHIVE_BYTES,
        extracted_member_limit=_MAX_ARCHIVE_MEMBERS,
        scanned_member_limit=_MAX_ARCHIVE_SCAN_MEMBERS,
        segment_observer=lambda segments: top_dirs.add(segments[0]),
    )
    if len(top_dirs) != 1:
        raise RuntimeError("environment archive had an unexpected layout")
    extracted_dir = dest / next(iter(top_dirs))
    if extracted_dir.exists() and not extracted_dir.is_dir():
        raise RuntimeError("environment archive had an unexpected layout")
    if not extracted_dir.is_dir():
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


def _ensure_cache_root() -> Path:
    """create the env cache root 0700 and refuse it if it is not private to this user.

    creation is per-component and EEXIST-tolerant (``make_private_dir``): the ancestors have to
    be created 0700 too, or the ancestor walk below rejects the path this call just made, and
    tolerating EEXIST is the race-safe create -- whoever wins, the checks below decide whether
    the winner's directory is trustworthy. ``lstat`` rather than ``stat`` so a pre-created
    symlink pointing the cache somewhere attacker-controlled is rejected instead of followed.
    the root stays hardcoded (no ``FLASH_ENV_CACHE_DIR``) on purpose: an ambient var that
    redirects where executable environment code is read from is the same hazard.
    """
    root = _CACHE_ROOT
    cache_security.make_private_dir(root)
    cache_security.validate_cache_root_ancestors(root)
    info = os.lstat(root)
    uid = os.getuid() if hasattr(os, "getuid") else info.st_uid
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"env cache root {root} is not a directory; refusing to use it")
    if info.st_uid != uid:
        raise RuntimeError(
            f"env cache root {root} is owned by uid {info.st_uid}, not {uid}; "
            "refusing to load environment code from it -- remove or reassign it"
        )
    # any group/other access on the root is refused, not just the write bits: a 0755/0710
    # root lets a same-group account traverse into cached entries, and entry CONTENTS can
    # legitimately carry group-writable modes (the contents-API path mkdirs parents under the
    # ambient umask, ancient git trees carry 100664 blobs, copytree preserves both), where
    # in-place tampering keeps the victim's uid and so still passes the entry ownership
    # vetting. nobody but this user ever needs to look inside the cache. mode bits mean
    # nothing on windows (mkdir(mode=0o700) does not establish them there, and a freshly
    # created, perfectly private root commonly reports group/other bits), so this check --
    # like the ancestor walk above -- is posix-only.
    if hasattr(os, "getuid") and info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise RuntimeError(
            f"env cache root {root} is accessible to group/other "
            f"(mode {info.st_mode & 0o777:04o}); "
            "refusing to load environment code from it -- chmod 700 it"
        )
    return root


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
            "managed environment hub refs must include a namespace/project/name environment path"
        )
    cache_scope = "managed-hub" if package_root else "github"
    cache_key = hashlib.sha256(
        f"{cache_scope}:github:{parsed.repo_full_name}@{resolved_ref}:{parsed.path}".encode()
    ).hexdigest()[:24]
    cache_dir = _ensure_cache_root() / cache_key
    # vet ANY existing entry at this key before looking inside it, not just one that already
    # holds the expected entrypoint. a foreign-owned directory missing the entrypoint used to
    # skip these checks entirely and fall straight to the download, which then writes the
    # environment into a directory another account owns. untrusted here means planted before
    # the root's permissions were last repaired, or swapped in since: never import it -- clear
    # it and fall through to a fresh download. raises if the entry cannot be removed, which has
    # to happen HERE: continuing would download the environment only for copytree to fail on
    # the entry still sitting there, and the alternative -- using it -- is what is refused.
    # a cache entry is a DIRECTORY, whoever owns it: a regular file at the key (manual cache
    # corruption, an interrupted write) passes the ownership check when we own it, and the
    # download path's rmtree(ignore_errors=True) then swallows NotADirectoryError and leaves
    # copytree to fail with FileExistsError on this key forever.
    if os.path.lexists(cache_dir) and not (
        cache_security.trust_cache_entry(cache_dir) and cache_dir.is_dir()
    ):
        cache_security.discard_untrusted_entry(cache_dir)
    env_file = cache_dir / parsed.path
    if env_file.is_dir():
        env_file = env_file / _DEFAULT_ENVIRONMENT_PATH
    if env_file.is_file():
        if cache_security.trust_cache_entry(env_file):
            # mark as recently used so LRU eviction keeps hot envs.
            with contextlib.suppress(OSError):
                os.utime(cache_dir)
            return env_file
        # a foreign file inside our own directory condemns the whole entry, same as above.
        cache_security.discard_untrusted_entry(cache_dir)
    tmp_parent = Path(tempfile.mkdtemp(prefix="flash-env-github-"))
    resolved = GitHubEnvironmentRef(
        parsed.owner,
        parsed.repo,
        resolved_ref,
        parsed.path,
    )
    try:
        if package_root:
            # the shared managed hub can be much larger than one environment. download only the
            # requested package so worker cache/extraction limits apply to that env, not the hub.
            extracted = _download_github_directory(resolved, package_root, tmp_parent)
        else:
            # generic github refs keep repo-level sidecars available to relative paths/imports.
            # user-facing pulls filter to the requested env subtree in flash.envs.loading.pull.
            extracted = _extract_github_tarball(resolved, tmp_parent)
        candidate = extracted / parsed.path
        if candidate.is_dir():
            candidate = candidate / _DEFAULT_ENVIRONMENT_PATH
        required_entrypoint = candidate.relative_to(extracted).as_posix()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"environment archive did not contain required entrypoint {required_entrypoint!r}"
            )
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
            EnvironmentTurn,
            load_environment,
        )

        return {
            "EnvironmentEpisode": EnvironmentEpisode,
            "EnvironmentMultiTurn": EnvironmentMultiTurn,
            "EnvironmentTurn": EnvironmentTurn,
            "load_environment": load_environment,
            "load_task_examples": load_task_examples,
            "task_example_from_record": task_example_from_record,
        }
    except ImportError as exc:
        # report what actually failed and where. a fixed "install freesolo" message is wrong
        # whenever the SDK IS installed and something underneath it raised -- a missing
        # transitive dep, a version conflict, a partially removed package after a tool upgrade.
        # the interpreter matters too: `flash` runs from its own uv-tool environment, so the
        # freesolo the user sees on their PATH is not necessarily the one that failed here.
        raise ImportError(
            f"could not import the Freesolo environment tools: {exc.__class__.__name__}: {exc}. "
            f"The interpreter that failed is {sys.executable}. "
            "If the 'freesolo' package is genuinely missing, install it (for example "
            "`uv pip install freesolo`) or use a worker image that includes the Freesolo SDK; "
            "if it is already installed, the error above is the real failure."
        ) from exc


def _load_resolved_freesolo_environment(
    env_id: str,
    reference: str,
    params: dict,
):
    """Load one already-resolved local entrypoint while retaining its canonical id."""
    from flash.envs.loading.adapter import FreesoloEnvironment

    tools = _import_freesolo_environment_tools()
    reference_path = Path(reference)
    base_dir = reference_path.parent if reference_path.exists() else Path.cwd()

    params = dict(params)
    source = params.pop("records", None)
    selection = select_dataset_source(params, base_dir, source, _resolve_path_arg)
    source = selection.source

    contract_path = _resolve_path_arg(params.get("contract_path"), base_dir)
    if isinstance(contract_path, str):
        params["contract_path"] = contract_path
    else:
        params.setdefault("contract_path", str(base_dir / "TRAINING_CONTRACT.md"))
    contract_text = str(
        params.pop("contract_text", "") or _load_contract_text(params["contract_path"])
    )

    sdk_env = tools["load_environment"](reference, **params)
    # an env that generates or owns every row in load_environment needs no packaged file, and a
    # datasets/ directory is then just raw or eval assets, so it must be able to load. an env
    # whose in-code dataset is empty still needs the file, so it still lands here, where the
    # message names the layout problem instead of the adapter's generic empty-dataset one. an
    # explicitly requested side split lands here too, env rows or not: the layout hid any
    # packaged split file, and rows the env supplies in code cannot be verified against the
    # requested split, so training on them would silently undo the split guarantee.
    if selection.datasets_dir_unread and (selection.side_split or not env_dataset_rows(sdk_env)):
        raise ValueError(
            "environment package has a top-level 'datasets/' directory, which Flash never "
            "reads (it probes dataset/<split>.jsonl or dataset/<split>.json). Rename the "
            "directory to 'dataset/', or set [environment.params] dataset_path to the exact "
            "file to train on." + selection.unread_split_hint
        )
    return FreesoloEnvironment(
        sdk_env,
        env_id,
        source=source,
        prefer_env_dataset=selection.source_is_dataset_file,
        contract_text=contract_text,
        package_root=base_dir,
    )


def load_freesolo_environment(env_id: str, pinned_sha: str | None = None, /, **kwargs):
    # pinned_sha is positional-only so a user param named pinned_sha cannot shadow it.
    reference = _resolve_environment_reference(env_id, pinned_sha)
    return _load_resolved_freesolo_environment(env_id, reference, kwargs)


from flash.envs.meta.namespace_listing import (  # noqa: E402
    list_managed_namespace_slugs as list_managed_namespace_slugs,
)

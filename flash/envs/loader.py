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
import http.client
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from flash.envs.package.limits import (
    ARCHIVE_MEMBER_LIMIT,
    ARCHIVE_SCAN_MEMBER_LIMIT,
    LimitedArchiveReader,
    archive_stream_limit,
)
from flash.envs.package.unpack import extract_validated_archive_members

_DEFAULT_GITHUB_REF = "main"
_DEFAULT_ENVIRONMENT_PATH = "environment.py"
_DEFAULT_MANAGED_ENV_REPO = "freesolo-co/environment-hub"
_CACHE_ROOT = Path("/tmp/flash-env-cache")
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
_DATASET_SPLIT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class GitHubRateLimitError(RuntimeError):
    """Persistent transient GitHub failure; worker stamps retriable=True for rescheduling.

    Raised for three distinct causes that are all worth retrying: a real rate limit, a persistent
    5xx, and a network failure. ``throttled`` says which -- True only for an actual rate limit --
    so a caller mapping this onto an HTTP status can answer 429 for throttling and 502 for an
    upstream outage instead of blaming the caller for GitHub being down. The retry semantics every
    existing handler relies on are unchanged.
    """

    def __init__(self, *args, throttled: bool = False):
        super().__init__(*args)
        self.throttled = throttled


# Reference parsing and safety checks live in `flash.envs.refs` (pure syntax, no I/O); they are
# re-exported here because `flash.envs.loader` -- and `flash.envs.adapter` through it -- is the
# import site every caller already uses.
from flash.envs.refs import (  # noqa: E402,F401
    GitHubEnvironmentRef,
    _is_safe_github_path_parts,
    _normalize_env_path,
    _parse_github_environment_ref,
    _parse_managed_environment_slug,
    canonical_managed_environment_slug,
    is_commit_sha,
    is_freesolo_environment_id,
    is_github_environment_ref,
    is_managed_environment_slug,
    managed_slug_to_github_ref,
)


def _github_token() -> str | None:
    """The GitHub token, or ``None`` when unset OR blank.

    Blank must collapse to ``None``, not fall through as a truthy string: a whitespace-only
    GITHUB_TOKEN would otherwise build ``Authorization: Bearer <whitespace>``, and GitHub REJECTS a
    malformed credential rather than treating the request as anonymous - so a public repo that
    loads fine with no token at all would fail with one that is merely blank.
    """
    return (os.environ.get("GITHUB_TOKEN") or "").strip() or None


def _resolve_ref_sha(
    parsed: GitHubEnvironmentRef,
    pinned_sha: str | None = None,
    *,
    timeout: float = 60.0,
    max_rate_limit_retries: int = 5,
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
    data = _urlopen(req, timeout=timeout, max_rate_limit_retries=max_rate_limit_retries)
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

    attempt = 0
    while True:
        try:
            if out is not None:
                out.seek(sink_start)
                out.truncate()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return drain(resp)
        except urllib.error.HTTPError as exc:
            # urllib can raise an HTTPError with fp=None; exc.read() is an AttributeError there.
            # the read can also truncate: GitHub can return a 429/5xx and then drop the connection.
            # that must not propagate -- we are inside this block's own handler, and Python does not
            # offer an exception raised here to the sibling clause below, so a truncated error body
            # would escape unretried as an uncaught 500 instead of the classified 429/502. the
            # status and headers still classify the failure, so an unreadable body just goes empty.
            try:
                body = exc.read().decode("utf-8", "replace") if exc.fp is not None else ""
            except (ConnectionError, http.client.IncompleteRead, OSError):
                body = ""
            remaining = (exc.headers.get("X-RateLimit-Remaining") if exc.headers else None) or ""
            is_rate_limit = exc.code == 429 or (
                exc.code == 403 and (remaining.strip() == "0" or "rate limit" in body.lower())
            )
            is_transient = is_rate_limit or exc.code >= 500
            if is_transient and attempt < max_rate_limit_retries:
                time.sleep(backoff_delay(attempt))
                attempt += 1
                continue
            if is_rate_limit:
                raise GitHubRateLimitError(
                    f"GitHub API rate limit exceeded ({exc.code}): {body[:300]}", throttled=True
                ) from exc
            if exc.code >= 500:
                raise GitHubRateLimitError(
                    f"GitHub server error ({exc.code}, transient) after {attempt} retries: {body[:300]}"
                ) from exc
            raise RuntimeError(
                f"GitHub environment request failed ({exc.code}): {body[:500]}"
            ) from exc
        # IncompleteRead is an HTTPException: NOT a URLError, TimeoutError, ConnectionError, or even
        # an OSError, so without naming it a truncated body escaped this loop unretried -- and,
        # being no RuntimeError either, escaped every caller's translation as an uncaught 500.
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
        ) as exc:
            if attempt < max_rate_limit_retries:
                time.sleep(backoff_delay(attempt))
                attempt += 1
                continue
            reason = getattr(exc, "reason", exc)
            raise GitHubRateLimitError(
                f"GitHub environment request failed after {attempt} retries (transient network): {reason}"
            ) from exc


def _download_github_tarball(ref: GitHubEnvironmentRef) -> Path:
    url = f"https://api.github.com/repos/{ref.repo_full_name}/tarball/{urllib.parse.quote(ref.ref, safe='')}"
    headers = _github_headers("application/vnd.github+json")
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


def _download_github_json(
    ref: GitHubEnvironmentRef,
    url: str,
    context: str,
    *,
    timeout: float = 120.0,
    max_rate_limit_retries: int = 5,
) -> object:
    data = _urlopen(
        urllib.request.Request(url, headers=_github_headers("application/vnd.github+json")),
        timeout=timeout,
        max_bytes=_MAX_CONTENTS_JSON_BYTES,
        max_rate_limit_retries=max_rate_limit_retries,
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
) -> list[dict]:
    payload = _download_github_json(
        ref,
        _github_tree_url(ref, treeish, recursive=recursive),
        context,
        timeout=timeout,
        max_rate_limit_retries=max_rate_limit_retries,
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


# An interactive list must fail fast enough that its 429/502 reaches the caller. Two reads at
# (1 + 1 attempts) x 20s + one <=45s backoff each is ~2 minutes worst case, versus ~30 minutes on the
# batch default. `flash.client.http.ENV_LIST_CLIENT_TIMEOUT_SECONDS` is sized to outlast this.
_LIST_READ_BUDGET = {"timeout": 20.0, "max_rate_limit_retries": 1}


def list_managed_namespace_slugs(namespace: str) -> list[str]:
    """List ``namespace/name`` slugs published under one managed-hub namespace, sorted.

    Tree reads and no clone: find the namespace directory in the hub root, then read it one level
    deep and keep the subdirectories that actually contain ``environment.py`` -- which is what
    ``_github_publish`` writes, so a stray file at the namespace root is not an environment.
    Authentication is the ambient ``GITHUB_TOKEN`` that every other GitHub read here uses.

    An absent namespace directory means the org has published nothing yet and returns an empty
    list. Every other failure propagates, because "no environments" must never be how a broken hub
    read looks: that is indistinguishable from a publish that silently did nothing.

    Both reads are deliberately bounded well below the publish path's budget. This serves a person
    waiting at a terminal, and inheriting the batch default (6 attempts x 120s socket + up to 180s
    backoff, twice over) would let a hard-down GitHub hold the request ~30 minutes before returning
    the controlled 502 -- long enough that every caller times out first and sees a generic failure
    instead of the upstream status. One retry at a 20s socket timeout answers within ~1 minute
    worst case, which is what makes the 429/502 actually reachable by a client.
    """
    if not _is_safe_github_path_parts((namespace,)):
        raise RuntimeError(f"unsafe managed environment namespace: {namespace!r}")
    ref = _parse_github_environment_ref(
        f"github:{_DEFAULT_MANAGED_ENV_REPO}@{_DEFAULT_GITHUB_REF}:"
        f"{namespace}/{_DEFAULT_ENVIRONMENT_PATH}"
    )
    if ref is None:  # pragma: no cover - the literal above always parses
        raise RuntimeError("could not build a managed environment reference")

    root = next(
        (
            entry
            for entry in _github_tree_entries(ref, ref.ref, ref.ref, **_LIST_READ_BUDGET)
            if entry.get("path") == namespace
            and entry.get("type") == "tree"
            and isinstance(entry.get("sha"), str)
        ),
        None,
    )
    if root is None:
        return []
    # ONE recursive fetch of the namespace subtree, not a tree request per environment: an org with
    # 100 envs would otherwise spend 102 calls of the shared GITHUB_TOKEN's budget on a list. Paths
    # in a recursive response are relative to the namespace, so `<name>/environment.py` identifies an
    # environment directly and no per-directory read is needed.
    slugs = []
    for entry in _github_tree_entries(
        ref, root["sha"], namespace, recursive=True, **_LIST_READ_BUDGET
    ):
        path = entry.get("path")
        if entry.get("type") != "blob" or not isinstance(path, str):
            continue
        parts = path.split("/")
        if len(parts) != 2 or parts[1] != _DEFAULT_ENVIRONMENT_PATH:
            continue
        if not _is_safe_github_path_parts((parts[0],)):
            continue
        slugs.append(f"{namespace}/{parts[0]}")
    return sorted(slugs)


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


def _extract_github_tarball(ref: GitHubEnvironmentRef, dest: Path) -> Path:
    tarball = _download_github_tarball(ref)
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


def _packaged_dataset_file(base_dir: Path, name: str) -> Path | None:
    """First existing packaged dataset file for split `name`.

    ``dataset/`` is canonical for new environments because a top-level ``datasets/``
    directory shadows the Hugging Face ``datasets`` package in local scripts.
    """
    for rel in (
        f"dataset/{name}.jsonl",
        f"dataset/{name}.json",
        f"{name}.jsonl",
        f"{name}.json",
    ):
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

    tools = _import_freesolo_environment_tools()
    reference = _resolve_environment_reference(env_id, pinned_sha)
    reference_path = Path(reference)
    base_dir = reference_path.parent if reference_path.exists() else Path.cwd()

    params = dict(kwargs)
    source = params.pop("records", None)
    dataset_path = params.get("dataset_path")
    if source is None and dataset_path:
        resolved_dataset_path = _resolve_path_arg(dataset_path, base_dir)
        params["dataset_path"] = resolved_dataset_path
        source = resolved_dataset_path
    # [environment.params] split selects which packaged dataset file Flash trains on. It used to
    # be forwarded to the SDK only, so SFT (and GRPO problem selection driven off dataset())
    # SILENTLY trained on the default dataset/train.jsonl even when a side split was requested.
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
                f"dataset/{split}.jsonl or {split}.json exists in the environment; "
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
        package_root=base_dir,
    )

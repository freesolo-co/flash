"""Adapter that runs Freesolo SDK environments on Flash.

Flash environment ids are Freesolo Hub slugs (``namespace/name``). Explicit
low-level refs remain parseable for compatibility. The canonical generated environment file is
``environment.py`` and its
``load_environment`` function must return a Freesolo SDK environment:
``EnvironmentSingleTurn`` or ``EnvironmentMultiTurn``.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flash.envs.base import BaseEnvironment

_DEFAULT_GITHUB_REF = "main"
_DEFAULT_ENVIRONMENT_PATH = "environment.py"
_DEFAULT_MANAGED_ENV_REPO = "freesolo-co/environment-hub"
_CACHE_ROOT = Path(os.environ.get("FLASH_ENV_CACHE_DIR", "/tmp/flash-env-cache"))
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
# Ceiling on the COMPRESSED whole-repo tarball download. GitHub serves only repo-wide tarballs (no
# per-subdirectory tarball), so pulling one small env from the shared environment-hub still fetches
# the whole repo; this ceiling must therefore bound the entire hub, NOT a single env. The per-env
# uncompressed extract stays capped at _MAX_ARCHIVE_BYTES after filtering to the requested subtree.
_MAX_TARBALL_BYTES = 1024 * 1024 * 1024
# Bytes pulled per streaming read. The download helpers read the response in chunks (rather than one
# resp.read()) so an oversize body aborts after at most one chunk past the limit, never buffering the
# full (up to _MAX_TARBALL_BYTES) body in a single allocation.
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 5000
# Upper bound on the TOTAL number of tar headers iterated (not just extracted). The whole-repo
# tarball can hold far more entries than the requested subtree extracts, so the per-subtree
# _MAX_ARCHIVE_MEMBERS cap alone leaves the scan loop unbounded: a tarball of many tiny/empty files
# compresses to well under _MAX_TARBALL_BYTES yet would force iterating millions of headers. This
# caps that CPU/time DoS while staying generous enough for a large shared hub of many envs.
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


class GitHubRateLimitError(RuntimeError):
    """Raised when the GitHub API rate-limits us (HTTP 429, or a 403 whose body says "rate limit").

    Raised by ``_urlopen`` only after the in-process jittered retry is exhausted, so it signals a
    *persistent* limit. The worker's top-level handler catches it and stamps ``retriable=True`` so
    the control plane reschedules the job on a fresh worker once the limit window resets, instead of
    permanently failing the run. The real spawn-wave mitigation is the control-plane resolve-once
    pin (EnvironmentSpec.resolved_sha) that lets workers skip the GitHub resolve entirely.
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
        if not _is_safe_environment_path(raw_path):
            return None
        try:
            path = _normalize_env_path(raw_path)
        except ValueError:
            return None
        if not raw_path:
            path = _DEFAULT_ENVIRONMENT_PATH
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


def _is_safe_environment_path(path: str) -> bool:
    if not path:
        return True
    raw = path.strip().replace("\\", "/")
    if raw.startswith("/"):
        return False
    parts = [part for part in raw.split("/") if part]
    if not parts:
        return True
    return not any(part in {".", ".."} for part in parts)


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
    # Resolve-once hook: the control plane resolves ref->sha ONCE (runner._assign_resolved_env_sha)
    # and threads the pinned commit sha through, so every worker in a fan-out short-circuits here
    # without hitting GitHub at all — this is what actually defuses a cold spawn wave (the prior
    # in-process cache could not, since each worker is a separate process). Same effect as the
    # immutable-ref fast path below, but applied to a symbolic ref (e.g. "main"). Only a real
    # 40-char sha is trusted; anything else falls through to a live resolve. The control plane
    # passes a short timeout + max_rate_limit_retries=0 so its best-effort pin can't block run
    # creation; the worker keeps the full retry budget.
    if pinned_sha and _is_commit_sha(pinned_sha):
        return pinned_sha
    if _is_commit_sha(parsed.ref):
        return parsed.ref
    # No pin (legacy spec / non-managed ref / control-plane resolve failed): resolve the symbolic
    # ref live every time. We deliberately do NOT cache symbolic refs in-process — a long-lived
    # process must see a moved branch (managed slugs point at environment-hub@main, which moves on
    # `flash env push`), and the immutable-sha cases above already skip the network.
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


def _read_capped(resp: object, max_bytes: int) -> bytearray:
    """Read ``resp`` in ``_DOWNLOAD_CHUNK_BYTES`` chunks into one growing buffer, aborting once the
    accumulated size exceeds ``max_bytes``. Returning the ``bytearray`` directly — rather than
    retaining a list of chunks and ``b"".join``-ing them — keeps peak memory at ~``max_bytes`` + one
    chunk, NOT 2x: a near-``_MAX_TARBALL_BYTES`` body never needs a second contiguous copy. It also
    bails out early on an oversize download instead of fetching the whole thing only to reject it.
    Callers treat the result as a read-only bytes buffer (``len`` / ``io.BytesIO`` / file write all
    accept a ``bytearray``)."""
    buf = bytearray()
    while True:
        chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
        if not chunk:
            break
        if len(buf) + len(chunk) > max_bytes:
            raise RuntimeError(
                f"GitHub response body exceeded the maximum allowed size ({max_bytes} bytes); "
                "download aborted"
            )
        buf.extend(chunk)
    return buf


def _urlopen(
    req: urllib.request.Request,
    *,
    timeout: float = 60.0,
    max_rate_limit_retries: int = 5,
    max_bytes: int | None = None,
) -> bytes | bytearray:
    """Fetch bytes for a GitHub request, surviving GitHub's secondary rate limit.

    On a 429 or a 403 whose body says "rate limit" (a cold spawn wave of workers all hitting the
    commits/tarball endpoint trips GitHub's abuse detection), retry up to ``max_rate_limit_retries``
    times with jitter so concurrent workers don't all retry in lockstep. If the limit persists past
    the retries, raise ``GitHubRateLimitError`` so the worker's top-level handler stamps
    ``retriable=True`` and the run reschedules on a fresh worker, instead of hard-failing on a
    transient limit. Any other HTTP / URL error raises a plain ``RuntimeError`` (non-retriable).

    ``max_rate_limit_retries=0`` makes it fail fast (one request, no sleeps): the control plane uses
    that for its best-effort resolve-once pin so a persistent limit can never block run creation —
    the long retry belongs on the worker, which can afford to wait.

    When ``max_bytes`` is set the body is streamed in chunks and aborts early once it exceeds that
    ceiling (see ``_read_capped``), so a large/hostile response can't be buffered whole in one
    allocation. Callers that download whole bodies (tarball, raw file) pass their size cap here.
    """
    import random
    import time

    _RATE_LIMIT_BASE_DELAY = 10.0
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return _read_capped(resp, max_bytes) if max_bytes is not None else resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            # GitHub signals an exhausted primary limit with a 403 + `X-RateLimit-Remaining: 0`
            # header; the body may be empty or omit the phrase "rate limit", so check the header too
            # rather than relying on substring matching alone. Secondary (abuse) limits come back 429.
            remaining = (exc.headers.get("X-RateLimit-Remaining") if exc.headers else None) or ""
            is_rate_limit = exc.code == 429 or (
                exc.code == 403 and (remaining.strip() == "0" or "rate limit" in body.lower())
            )
            if is_rate_limit and attempt < max_rate_limit_retries:
                delay = max(
                    _RATE_LIMIT_BASE_DELAY,
                    min(45.0, _RATE_LIMIT_BASE_DELAY * (attempt + 1) * random.uniform(0.5, 1.5)),
                )
                time.sleep(delay)
                attempt += 1
                continue
            if is_rate_limit:
                # Persistent limit: signal retriable so the control plane reschedules (#209).
                raise GitHubRateLimitError(
                    f"GitHub API rate limit exceeded ({exc.code}): {body[:300]}"
                ) from exc
            raise RuntimeError(
                f"GitHub environment request failed ({exc.code}): {body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GitHub environment request failed: {exc.reason}") from exc


def _download_github_tarball(ref: GitHubEnvironmentRef) -> bytes | bytearray:
    # Callers download the tarball for the ALREADY-resolved commit sha (see
    # _resolve_github_environment_file), which extracts it into the content-addressed disk cache
    # under _CACHE_ROOT/<hash(repo@sha:path)> and never re-downloads on a hit. So reuse is handled
    # on disk; we deliberately do NOT also retain the (up to _MAX_ARCHIVE_BYTES) archive bytes in a
    # module-level cache for the worker's lifetime — that wasted hundreds of MiB of RAM per process.
    url = f"https://api.github.com/repos/{ref.repo_full_name}/tarball/{urllib.parse.quote(ref.ref, safe='')}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "freesolo-flash",
    }
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # Stream-cap the whole-repo tarball at the larger whole-repo ceiling (the per-env subtree is
    # filtered + capped later, in _safe_extract_archive) so a small env pull from a big shared hub
    # isn't rejected just because unrelated siblings make the repo tarball large — while still
    # bailing out early (in _urlopen) instead of buffering a near-1 GiB body whole. The post-read
    # length check stays as defense-in-depth for any path that bypasses the streaming cap.
    data = _urlopen(
        urllib.request.Request(url, headers=headers), timeout=120.0, max_bytes=_MAX_TARBALL_BYTES
    )
    if len(data) > _MAX_TARBALL_BYTES:
        raise RuntimeError(
            f"repository tarball is too large ({len(data)} bytes; limit {_MAX_TARBALL_BYTES} bytes)"
        )
    return data


def _safe_extract_archive(tar_bytes: bytes | bytearray, dest: Path, subdir: str = "") -> Path:
    """Extract the GitHub repo tarball under ``dest`` and return its single top-level directory.

    When ``subdir`` is given (the environment's path within the repo, e.g. ``ns/name``), ONLY that
    subtree is extracted, and only its members count toward ``_MAX_ARCHIVE_MEMBERS`` /
    ``_MAX_ARCHIVE_BYTES``. This matters for the shared environment-hub: a small env pull must not
    fail (or pay the cost) because unrelated sibling environments push the whole repo over the
    limits. Unrelated members are skipped entirely, so a stray unsafe path in some OTHER env can't
    abort this pull either."""
    root = dest.resolve()
    want = [p for p in subdir.split("/") if p] if subdir else []
    top_dirs: set[str] = set()
    total = 0
    extracted = 0
    scanned = 0
    # Spill the compressed body to a temp FILE and read the tar from disk instead of wrapping it in
    # io.BytesIO(...), which copies the buffer: a near-_MAX_TARBALL_BYTES body would otherwise peak at
    # ~2x in RAM (the downloaded buffer plus its BytesIO copy) and OOM even for a tiny env pulled from
    # a big shared hub. ``del tar_bytes`` releases the in-RAM copy once it is on disk.
    with tempfile.NamedTemporaryFile(prefix="flash-env-tar-", suffix=".tar.gz") as spill:
        spill.write(tar_bytes)
        spill.seek(0)
        del tar_bytes
        with tarfile.open(fileobj=spill, mode="r:gz") as tar:
            for member in tar:
                # Bound the TOTAL headers iterated, not just the extracted subset: filtering to a
                # small subtree still walks every member of a (possibly huge) whole-repo tarball, so
                # without this a tarball packed with many tiny entries is a CPU/time DoS even when
                # extraction is tiny. Count every member yielded, before the metadata/subtree skips.
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
                # ``raw[0]`` is the archive's single top dir; ``raw[1:]`` is the path within the repo.
                # Track the top dir from EVERY member (so the layout is known even when the requested
                # subtree is absent), but only extract/count members inside that subtree.
                top_dirs.add(raw[0])
                if want and raw[1 : 1 + len(want)] != want:
                    continue
                if ".." in raw:
                    raise RuntimeError(f"unsafe path in environment archive: {member.name!r}")
                extracted += 1
                if extracted > _MAX_ARCHIVE_MEMBERS:
                    raise RuntimeError(
                        f"env package has too many members (limit {_MAX_ARCHIVE_MEMBERS})"
                    )
                normalized_name = "/".join(raw)
                target = (dest / normalized_name).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"unsafe path in environment archive: {member.name!r}")
                if member.islnk() or member.issym() or not (member.isreg() or member.isdir()):
                    continue
                total += max(0, member.size)
                if total > _MAX_ARCHIVE_BYTES:
                    raise RuntimeError(
                        f"environment archive is too large uncompressed ({total} bytes; limit {_MAX_ARCHIVE_BYTES} bytes)"
                    )
                member.name = normalized_name
                tar.extract(member, dest)
    if len(top_dirs) != 1:
        raise RuntimeError("environment archive had an unexpected layout")
    # The top dir always exists; create it even when the requested subtree had no members, so the
    # caller can report the missing environment directory itself (rather than a layout error here).
    extracted_dir = dest / next(iter(top_dirs))
    extracted_dir.mkdir(parents=True, exist_ok=True)
    return extracted_dir


def _resolve_github_environment_file(env_ref: str, pinned_sha: str | None = None) -> Path:
    parsed = _parse_github_environment_ref(env_ref)
    if parsed is None:
        raise ValueError(f"not a GitHub environment ref: {env_ref!r}")
    # Only forward pinned_sha when set, so the unpatched _resolve_ref_sha(parsed) call shape stays
    # single-arg (a test monkeypatch like ``lambda parsed: ...`` keeps working).
    resolved_ref = (
        _resolve_ref_sha(parsed, pinned_sha=pinned_sha) if pinned_sha else _resolve_ref_sha(parsed)
    )
    cache_key = hashlib.sha256(
        f"github:{parsed.repo_full_name}@{resolved_ref}:{parsed.path}".encode()
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
        # Full-repo extraction here (NO subdir filter): this resolver feeds env EXECUTION/training,
        # where an env can reference repo-level sidecars via relative params/imports (e.g.
        # ``dataset_path = "../datasets/train.jsonl"`` -> a sibling outside its own subtree). Filtering
        # to just the env dir would drop those, so the env loads but fails at run time. The subtree
        # filter is for ``pull_environment_package`` only (the user pulling ONE env to local disk).
        extracted = _safe_extract_archive(_download_github_tarball(resolved), tmp_parent)
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
    # pinned_sha (resolve-once hook, item 3): when the control plane already resolved this env's
    # ref->sha, it threads the commit sha here so the GitHub commits API is skipped entirely. None
    # (the default and today's behavior) means the worker resolves the ref itself.
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


def _coerce_environment_github_ref(env_ref: str) -> GitHubEnvironmentRef:
    """Resolve a Freesolo slug / ``github:`` ref / github.com URL to a ``GitHubEnvironmentRef``.

    A managed slug (``owner/name``) is first expanded to its environment-hub ref, so the same pull
    path serves both managed environments and raw GitHub references.
    """
    candidate = env_ref
    if is_managed_environment_slug(candidate):
        candidate = managed_slug_to_github_ref(candidate)
    parsed = _parse_github_environment_ref(candidate)
    if parsed is None:
        raise ValueError(f"not a Freesolo or GitHub environment reference: {env_ref!r}")
    return parsed


def _safe_repo_relative_path(path: str) -> str:
    """Validate a user-supplied file path inside an environment (reject absolute / ``..`` escapes)."""
    raw = (path or "").strip().replace("\\", "/")
    if not raw or raw.startswith("/"):
        raise ValueError(f"invalid environment file path: {path!r}")
    parts = [part for part in raw.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"invalid environment file path: {path!r}")
    return "/".join(parts)


def _environment_dir(env_path: str) -> str:
    """The environment's directory within the repo, given a ref's path.

    A ref may name the entrypoint FILE (``.../environment.py`` or a custom ``.../custom.py``) or the
    environment DIRECTORY itself (``.../envs/e``). All resolve to the same dir. ``rpartition('/')``
    alone would mis-handle a directory ref (treating its last segment as a filename), so strip a
    trailing ``*.py`` entrypoint file and otherwise treat the whole path as the directory. (Sidecars
    like ``datasets/train.jsonl`` are relative to this dir, so a custom entrypoint name must not be
    mistaken for a directory.)"""
    parts = [p for p in env_path.split("/") if p]
    if parts and parts[-1].endswith(".py"):
        parts = parts[:-1]
    return "/".join(parts)


def _environment_entrypoint(env_path: str) -> str:
    """The entrypoint filename a ref names — a trailing ``*.py`` (custom or ``environment.py``), or
    the default ``environment.py`` when the ref names a directory."""
    parts = [p for p in env_path.split("/") if p]
    if parts and parts[-1].endswith(".py"):
        return parts[-1]
    return _DEFAULT_ENVIRONMENT_PATH


def environment_local_dirname(env_ref: str) -> str:
    """A sensible local directory name for a pulled environment (used as the default output dir)."""
    slug = _parse_managed_environment_slug(env_ref)
    if slug is not None:
        return slug[1]
    ref = _coerce_environment_github_ref(env_ref)
    env_dir = _environment_dir(ref.path)
    return env_dir.rsplit("/", 1)[-1] if env_dir else ref.repo


def download_environment_file(env_ref: str, rel_path: str, *, timeout: float = 120.0) -> bytes | bytearray:
    """Download a single file from a published environment as raw bytes.

    Uses GitHub's ``application/vnd.github.raw`` media type, which streams the file's bytes
    directly. The JSON "contents" API (what ``gh api contents`` hits) instead base64-encodes the
    body and returns an EMPTY ``content`` for blobs over 1 MB — so dataset files like
    ``datasets/train.jsonl`` silently come back empty. The raw media type streams the blob's bytes
    directly (GitHub caps raw blobs around 100 MB), sidestepping that 1 MB limit; this helper
    additionally rejects any response over ``_MAX_ARCHIVE_BYTES`` (256 MiB). ``rel_path`` is
    relative to the environment directory. A ``GITHUB_TOKEN`` is required for private/internal
    environment repos.
    """
    ref = _coerce_environment_github_ref(env_ref)
    safe_rel = _safe_repo_relative_path(rel_path)
    env_dir = _environment_dir(ref.path)
    full_path = f"{env_dir}/{safe_rel}" if env_dir else safe_rel
    quoted_path = "/".join(urllib.parse.quote(part, safe="") for part in full_path.split("/"))
    url = (
        f"https://api.github.com/repos/{ref.repo_full_name}/contents/{quoted_path}"
        f"?ref={urllib.parse.quote(ref.ref, safe='')}"
    )
    headers = {"Accept": "application/vnd.github.raw", "User-Agent": "freesolo-flash"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = _urlopen(
        urllib.request.Request(url, headers=headers), timeout=timeout, max_bytes=_MAX_ARCHIVE_BYTES
    )
    if len(data) > _MAX_ARCHIVE_BYTES:
        raise RuntimeError(
            f"environment file is too large ({len(data)} bytes; limit {_MAX_ARCHIVE_BYTES} bytes)"
        )
    return data


def _atomic_copy2(src: Path, dst: Path) -> None:
    """Copy ``src`` onto ``dst`` atomically: stage to a temp file in ``dst``'s directory, then
    ``os.replace``. A mid-copy failure (disk full / quota / read error) therefore never truncates an
    existing ``dst`` — the swap is all-or-nothing."""
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".flash-env-pull-")
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _rm_path(p: Path) -> None:
    """Remove ``p`` whether it is a symlink, file, or directory (symlink dirs are unlinked, not
    recursed)."""
    if p.is_symlink() or not p.is_dir():
        p.unlink()
    else:
        shutil.rmtree(p)


def _replace_opposite_type(target: Path, build) -> None:
    """Replace an existing ``target`` (a file/dir being swapped for the OPPOSITE type, or a symlink)
    without destroying it until the replacement is ready: move it aside into a fresh UNIQUE scratch
    dir, run ``build()`` to create the new ``target``, drop the backup on success, and RESTORE the
    original if ``build`` raises (a mid-copy disk/quota/read error). The naive
    ``unlink()``/``rmtree()``-then-copy would leave the user with nothing if the copy failed. A unique
    scratch dir (NOT a fixed ``.<name>.flash-old`` sibling) is used so a pre-existing user file/dir of
    that conventional name is never clobbered."""
    scratch = Path(tempfile.mkdtemp(prefix=".flash-env-old-", dir=target.parent))
    backup = scratch / target.name
    try:
        os.replace(target, backup)  # atomic move-aside (file, dir, or symlink)
        try:
            build()
        except BaseException:
            if target.is_symlink() or target.exists():
                _rm_path(target)  # drop any partial replacement
            os.replace(backup, target)  # restore the original intact
            raise
    finally:
        # Drop the scratch dir: on success it still holds the saved original (no longer needed); on
        # failure the original has already been moved back out, leaving it empty.
        shutil.rmtree(scratch, ignore_errors=True)


def _merge_into_dir(source: Path, dest: Path) -> None:
    """Recursively copy ``source``'s contents into existing ``dest``, overwriting same-named
    children. Unlike ``shutil.copytree(dirs_exist_ok=True)``, a child of ``dest`` that is a SYMLINK is
    replaced (never followed) — so a pre-existing ``datasets -> /elsewhere`` link in the destination
    (e.g. the cwd, when merging in place with ``--force``) can't redirect writes outside the requested
    output tree. Each file is replaced atomically (stage + ``os.replace``) so a failed merge never
    leaves an existing file truncated. A child being replaced by one of the OPPOSITE type (existing
    file vs incoming dir, or vice versa) OR a child symlink is swapped via move-aside/restore, so a
    mid-copy failure can neither destroy the user's existing child nor lose their symlink."""
    dest.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = dest / child.name
        build = (
            functools.partial(_merge_into_dir, child, target)
            if child.is_dir()
            else functools.partial(_atomic_copy2, child, target)
        )
        if target.is_symlink():
            # Replace the link ITSELF (never write through it), but keep it until the replacement
            # succeeds: move it aside and restore on a mid-copy failure, exactly like the
            # opposite-type paths below. An eager unlink would lose the link if the copy then failed.
            _replace_opposite_type(target, build)
        elif child.is_dir():
            if target.exists() and not target.is_dir():
                # incoming dir over an existing file: keep the file until the subtree is fully built.
                _replace_opposite_type(target, build)
            else:
                _merge_into_dir(child, target)
        else:
            if target.is_dir():
                # incoming file over an existing dir: keep the dir until the file copy succeeds.
                _replace_opposite_type(target, build)
            else:
                _atomic_copy2(child, target)


def pull_environment_package(env_ref: str, dest: str | Path, *, overwrite: bool = False) -> Path:
    """Download a published environment's whole directory to ``dest`` and return it.

    Fetches the repository tarball (not the per-file contents API), so it is unaffected by the
    1 MB JSON-contents limit and pulls every shipped file — ``environment.py`` plus sidecars such
    as ``datasets/train.jsonl``. Only the environment's own subdirectory is written to ``dest``,
    never the rest of the (shared) environment-hub repo.
    """
    ref = _coerce_environment_github_ref(env_ref)
    env_dir = _environment_dir(ref.path)
    # A managed slug always expands to ``<namespace>/<name>/environment.py`` (a subdir), so env_dir
    # is non-empty for managed ids. An empty env_dir only arises from an explicit github ref to a
    # repo-ROOT environment.py — a dedicated single-env repo, where copying the whole repo is
    # intended. Guard the one case where that would dump the entire SHARED hub: a root-level ref that
    # resolves to the managed environment-hub repo. Pull a specific ``namespace/name`` id instead.
    # GitHub owner/repo names are case-insensitive, so compare case-folded — otherwise a ref like
    # ``github:Freesolo-Co/Environment-Hub@main:environment.py`` would slip past the guard and
    # extract the entire shared hub root.
    if not env_dir and ref.repo_full_name.lower() == _DEFAULT_MANAGED_ENV_REPO.lower():
        raise ValueError(
            f"refusing to pull the whole shared environment hub ({_DEFAULT_MANAGED_ENV_REPO}); "
            "specify a managed environment id (namespace/name) or a github ref to its subdirectory"
        )
    dest_path = Path(dest)
    # "occupied" = a symlink (replacing a link needs explicit consent), a file, or a non-empty
    # directory; an empty real dir is fine to populate in place.
    occupied = dest_path.is_symlink() or (
        dest_path.exists() and (not dest_path.is_dir() or any(dest_path.iterdir()))
    )
    if occupied and not overwrite:
        raise FileExistsError(
            f"destination {dest_path} already exists (pass overwrite=True to replace)"
        )
    tmp_parent = Path(tempfile.mkdtemp(prefix="flash-env-pull-"))
    try:
        extracted = _safe_extract_archive(_download_github_tarball(ref), tmp_parent, subdir=env_dir)
        source = extracted / env_dir if env_dir else extracted
        if not source.is_dir():
            raise FileNotFoundError(
                f"environment directory {env_dir or '.'!r} not found in {ref.repo_full_name}@{ref.ref}"
            )
        # Verify the entrypoint is actually present before reporting a successful pull — a ref can
        # name a directory that exists but lacks its entrypoint (the resolver rejects that same
        # case), which would otherwise leave the user with an invalid package and no error. Honor a
        # custom entrypoint filename from the ref (e.g. ``.../custom.py``), not just environment.py.
        entrypoint = _environment_entrypoint(ref.path)
        if not (source / entrypoint).is_file():
            raise FileNotFoundError(
                f"environment entrypoint {entrypoint!r} not found in "
                f"{env_dir or '.'!r} of {ref.repo_full_name}@{ref.ref}"
            )
        # Create any missing parent dirs (mirrors the single-file download) so a nested output path
        # works.
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # When the destination IS the current working directory (or an ancestor of it) — e.g.
        # ``-o .`` — we must never rmtree/rename it (``shutil.rmtree('.')`` deletes its contents
        # before raising, and you can't rename over a live dir). Merge the env's files in place,
        # overwriting same-named children, instead of swapping the directory itself.
        cwd = Path.cwd().resolve()
        dest_resolved = dest_path.resolve()
        in_place = dest_resolved == cwd or dest_resolved in cwd.parents
        # A symlink destination (even a dangling one) is NOT unlinked up front: doing so before the
        # copy would lose the user's link if the copy then failed. It goes through the staged swap
        # below (which replaces the link itself, never writing through it / touching its target) so the
        # link survives until the replacement is ready and is restored on failure.
        dest_is_symlink = dest_path.is_symlink()
        if not dest_is_symlink and not dest_path.exists():
            shutil.copytree(source, dest_path)
        elif not dest_is_symlink and (in_place or (dest_path.is_dir() and not any(dest_path.iterdir()))):
            # An existing empty dir, or the cwd/an ancestor: populate in place (preserve the dir and
            # its perms; overwrite only same-named children). `occupied` already gated on overwrite.
            # Use the symlink-safe merge so a child symlink in the destination can't redirect writes
            # outside the output tree.
            _merge_into_dir(source, dest_path)
        else:
            # Overwriting an OCCUPIED destination (file / non-empty dir) OR replacing a SYMLINK: build
            # the new tree in a sibling staging dir, move the live path ASIDE (``os.replace`` handles a
            # file, dir, OR symlink), swap the new one in, and only then drop the old — so the previous
            # package or link is never removed until the replacement is ready, and is restored intact if
            # the swap fails.
            staging_parent = Path(tempfile.mkdtemp(prefix=".flash-env-pull-", dir=dest_path.parent))
            try:
                staging = staging_parent / dest_path.name
                shutil.copytree(source, staging)
                backup = staging_parent / (dest_path.name + ".old")
                os.replace(dest_path, backup)  # atomic move-aside (works for a file, dir, or symlink)
                try:
                    os.replace(staging, dest_path)
                except BaseException:
                    os.replace(backup, dest_path)  # restore the original on failure
                    raise
            finally:
                shutil.rmtree(staging_parent, ignore_errors=True)
        return dest_path
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)


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
        """Rollout turn ceiling the worker reads for the batch-level loop cap.

        A pure multi-turn freesolo env sets a *per-example* budget via
        ``max_episode_turns(example)`` (e.g. #user turns + a tool-iteration budget per
        turn). The batch cap the rollout loop uses must be at least the largest such
        budget, or it would truncate the deepest scenarios before they finish (e.g.
        support-chat's 4-customer-turn rollouts need ~20 turns, not 8). Take the
        dataset-wide max once, bounded so a pathological env can't make rollouts
        unbounded; single-turn / non-multi-turn envs keep the small default. The exact
        per-example budget is still enforced in :meth:`rollout_done`.
        """
        if self._max_turns_cache is not None:
            return self._max_turns_cache
        cap = 8
        if self.multi_turn:
            cap = 24  # safe default if no per-example budget can be read at all
            best: int | None = None  # running max — no intermediate list for large datasets
            for ex in self.dataset():  # cached; see dataset()
                # Per-example so ONE malformed row (or an env whose max_episode_turns raises on it)
                # is skipped rather than discarding every budget and silently falling back to 24,
                # which would reintroduce the truncation this is meant to prevent.
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
        # Parse once and cache: the worker reads ``env.dataset()`` AND ``env.max_turns`` (which
        # also scans the dataset), so without this a multi-turn run would parse/load the whole
        # dataset twice at startup.
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
            task = getattr(example, "task", None)
            if _CANONICAL_INPUT_KEY not in raw and task is not None:
                raw[_CANONICAL_INPUT_KEY] = task
            task_id = getattr(example, "task_id", None)
            if task_id is not None:
                raw.setdefault("id", task_id)
            expected = getattr(example, "expected_output", None)
            if expected is not None:
                raw.setdefault(_CANONICAL_OUTPUT_KEY, _json_safe(expected))
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
        """Target completion messages to append after the prompt for one SFT example.

        Delegates to the freesolo-sdk env's first-class ``Environment.sft_completion``, which turns
        the record's ``output`` into the target messages: a MULTI-TURN target trajectory —
        assistant turns, tool calls, tool results, replies (authored as ``output = {"messages":
        [...]}`` or a bare message list) — when the row ships one, else a single assistant turn from
        a scalar output. So the SFT example shape is owned by the freesolo-sdk dataset layer
        (``freesolo.datasets.target_messages``), not a flash-only convention; ``len(...) > 1`` is
        multi-turn. Falls back to reading the raw record only for an older installed SDK that
        predates the method."""
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

    def scores_breakdown(
        self, completion: str, example: dict, state: dict | None = None
    ) -> dict[str, float]:
        if state and self.multi_turn:
            reward = self._score_episode(example, state)
        else:
            rewards = self._env.score_responses(self._task_example(example), [completion])
            if len(rewards) != 1:
                raise RuntimeError("Freesolo environment score_responses returned the wrong length")
            reward = rewards[0]
        return self._reward_to_breakdown(reward)

    def reward(self, completion: str, example: dict, state: dict | None = None) -> float:
        return float(self.scores_breakdown(completion, example, state)["total"])

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
            # Single-turn: group rollouts of the same example so their completions go through ONE
            # score_responses() call (env-concurrent), mirroring the multi-turn score_episodes()
            # batching below. Without this, a GRPO step's whole completion batch was scored by
            # serial per-rollout reward() calls — one blocking judge/API round-trip at a time, with
            # the GPU idle throughout. Scoring grades the rollout's actual response (stored on the
            # state), not "" (which would score every item empty).
            groups: dict[str, dict] = {}
            order: list[str] = []
            for i, (ex, st) in enumerate(items):
                key = json.dumps(ex, sort_keys=True, default=str)
                grp = groups.get(key)
                if grp is None:
                    grp = groups[key] = {
                        "task": self._task_example(ex),
                        "idxs": [],
                        "responses": [],
                    }
                    order.append(key)
                grp["idxs"].append(i)
                grp["responses"].append(str(st.get("response_text") or ""))
            out: list[float] = [0.0] * len(items)
            for key in order:
                grp = groups[key]
                rewards = self._env.score_responses(grp["task"], grp["responses"])
                if len(rewards) != len(grp["responses"]):
                    raise RuntimeError(
                        "Freesolo environment score_responses returned the wrong length"
                    )
                for idx, rw in zip(grp["idxs"], rewards, strict=True):
                    out[idx] = float(rw.score)
            return out
        groups: dict[str, dict] = {}
        order: list[str] = []
        for i, (ex, st) in enumerate(items):
            # Group rollouts of the same example (a GRPO group shares one prompt) so their episodes
            # are scored together; the example dict is the stable grouping key.
            key = json.dumps(ex, sort_keys=True, default=str)
            grp = groups.get(key)
            if grp is None:
                grp = groups[key] = {
                    "task": st.get("task") or self._task_example(ex),
                    "idxs": [],
                    "episodes": [],
                }
                order.append(key)
            grp["idxs"].append(i)
            grp["episodes"].append(self._episode_from_state(st))
        out: list[float] = [0.0] * len(items)
        for key in order:
            grp = groups[key]
            rewards = self._env.score_episodes(grp["task"], grp["episodes"])
            if len(rewards) != len(grp["episodes"]):
                raise RuntimeError("Freesolo environment score_episodes returned the wrong length")
            for idx, rw in zip(grp["idxs"], rewards, strict=True):
                out[idx] = float(rw.score)
        return out

    @property
    def reward_thread_safe(self) -> bool:
        """Whether ``reward`` may be called concurrently across rollouts (multiturn_rollout's
        ``_score_rollouts`` thread-pool fallback, used when the env has no ``reward_many``). The
        verifiers reward contract is a pure scorer — ``score_responses`` reads the per-call inputs +
        immutable env config — so the default is True. An underlying env whose scorer keeps mutable
        state or a thread-bound client opts out with ``reward_thread_safe = False`` (scored serially)."""
        return bool(getattr(self._env, "reward_thread_safe", True))

    def grade(self, completion: str, example: dict, state: dict | None = None) -> bool:
        if state and self.multi_turn:
            reward = self._score_episode(example, state)
        else:
            rewards = self._env.score_responses(self._task_example(example), [completion])
            if len(rewards) != 1:
                raise RuntimeError("Freesolo environment score_responses returned the wrong length")
            reward = rewards[0]
        return bool(reward.resolved_success())

    def tools(self) -> list:
        return []

    def new_rollout_state(self, example: dict) -> dict:
        task = self._task_example(example)
        prompt = [dict(message) for message in self._env.start_episode(task, self._contract_text)]
        # Per-example turn budget (env's max_episode_turns) so rollout_done caps THIS
        # rollout at its own budget rather than the batch-wide ceiling -- a single-turn
        # scenario stops after a few turns while a deep one gets its full budget.
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
        # Prefer THIS rollout's own per-example budget (set in new_rollout_state); fall
        # back to the batch-wide cap the worker passes. The env normally terminates via
        # step.done well before either, so this is a non-termination guard.
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
        if len(rewards) != 1:
            raise RuntimeError("Freesolo environment score_episodes returned the wrong length")
        return rewards[0]

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
        if len(rewards) != 1:
            raise RuntimeError("Freesolo environment score_episodes returned the wrong length")
        return float(rewards[0].score)


def load_freesolo_environment(
    env_id: str, pinned_sha: str | None = None, /, **kwargs
) -> FreesoloEnvironment:
    # pinned_sha is a POSITIONAL-ONLY resolve-once hook (the control-plane-pinned commit sha). It is
    # positional-only (the `/`) precisely so a user [environment.params] entry of ANY name — even
    # one literally named "pinned_sha" — lands in **kwargs and is forwarded verbatim to the Freesolo
    # SDK loader, never binding to or shadowing this internal pin. None (default) preserves today's
    # behavior — the worker resolves the env ref->sha itself.
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

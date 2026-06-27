"""Adapter that runs Freesolo SDK environments on Flash.

Flash environment ids are Freesolo Hub slugs (``namespace/name``). Explicit
low-level refs remain parseable for compatibility. The canonical generated environment file is
``environment.py`` and its
``load_environment`` function must return a Freesolo SDK environment:
``EnvironmentSingleTurn`` or ``EnvironmentMultiTurn``.
"""

from __future__ import annotations

import hashlib
import io
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
_MAX_ARCHIVE_MEMBERS = 5000
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


def _urlopen(
    req: urllib.request.Request, *, timeout: float = 60.0, max_rate_limit_retries: int = 5
) -> bytes:
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
    """
    import random
    import time

    _RATE_LIMIT_BASE_DELAY = 10.0
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
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
                delay = max(_RATE_LIMIT_BASE_DELAY, min(45.0, _RATE_LIMIT_BASE_DELAY * (attempt + 1) * random.uniform(0.5, 1.5)))
                time.sleep(delay)
                attempt += 1
                continue
            if is_rate_limit:
                # Persistent limit: signal retriable so the control plane reschedules (#209).
                raise GitHubRateLimitError(
                    f"GitHub API rate limit exceeded ({exc.code}): {body[:300]}"
                ) from exc
            raise RuntimeError(f"GitHub environment request failed ({exc.code}): {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GitHub environment request failed: {exc.reason}") from exc


def _download_github_tarball(ref: GitHubEnvironmentRef) -> bytes:
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
    data = _urlopen(urllib.request.Request(url, headers=headers), timeout=120.0)
    if len(data) > _MAX_ARCHIVE_BYTES:
        raise RuntimeError(
            f"environment archive is too large ({len(data)} bytes; "
            f"limit {_MAX_ARCHIVE_BYTES} bytes)"
        )
    return data


def _safe_extract_archive(tar_bytes: bytes, dest: Path) -> Path:
    root = dest.resolve()
    top_dirs: set[str] = set()
    total = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for count, member in enumerate(tar, start=1):
            if count > _MAX_ARCHIVE_MEMBERS:
                raise RuntimeError(
                    f"env package has too many members (limit {_MAX_ARCHIVE_MEMBERS})"
                )
            if member.type in _TAR_METADATA_TYPES:
                continue
            parts: list[str] = []
            for part in member.name.replace("\\", "/").split("/"):
                if not part or part == ".":
                    continue
                if part == "..":
                    raise RuntimeError(f"unsafe path in environment archive: {member.name!r}")
                parts.append(part)
            if not parts:
                continue
            normalized_name = "/".join(parts)
            target = (dest / normalized_name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe path in environment archive: {member.name!r}")
            if member.islnk() or member.issym() or not (member.isreg() or member.isdir()):
                continue
            top_dirs.add(parts[0])
            total += max(0, member.size)
            if total > _MAX_ARCHIVE_BYTES:
                raise RuntimeError(
                    f"environment archive is too large uncompressed ({total} bytes; limit {_MAX_ARCHIVE_BYTES} bytes)"
                )
            member.name = normalized_name
            tar.extract(member, dest)
    if len(top_dirs) != 1:
        raise RuntimeError("environment archive had an unexpected layout")
    extracted = dest / next(iter(top_dirs))
    if not extracted.is_dir():
        raise RuntimeError("environment archive did not extract to a directory")
    return extracted


def _resolve_github_environment_file(env_ref: str, pinned_sha: str | None = None) -> Path:
    parsed = _parse_github_environment_ref(env_ref)
    if parsed is None:
        raise ValueError(f"not a GitHub environment ref: {env_ref!r}")
    resolved_ref = _resolve_ref_sha(parsed, pinned_sha=pinned_sha)
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
            # freesolo>=0.2.49 TaskExample exposes input/id/output; read those first and fall back
            # to the pre-0.2.49 task/task_id/expected_output names. Reading only the legacy names
            # (as before) silently returned None on the pinned SDK, so an inline-dataset env whose
            # TaskExample carried an empty .record built an empty canonical record and crashed at
            # startup with "records must contain an input field".
            task = getattr(example, "input", None)
            if task is None:
                task = getattr(example, "task", None)
            if _CANONICAL_INPUT_KEY not in raw and task is not None:
                raw[_CANONICAL_INPUT_KEY] = task
            task_id = getattr(example, "id", None)
            if task_id is None:
                task_id = getattr(example, "task_id", None)
            if task_id is not None:
                raw.setdefault("id", task_id)
            expected = getattr(example, "output", None)
            if expected is None:
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
        if isinstance(value, dict) and list(value) == ["messages"] and isinstance(value["messages"], list):
            return [dict(m) for m in value["messages"]]
        return [{"role": "assistant", "content": "" if value is None else str(value)}]

    def _single(self, results, method: str):
        if len(results) != 1:
            raise RuntimeError(f"Freesolo environment {method} returned the wrong length")
        return results[0]

    def _score_one(self, completion: str, example: dict, state: dict | None):
        if state and self.multi_turn:
            return self._score_episode(example, state)
        rewards = self._env.score_responses(self._task_example(example), [completion])
        return self._single(rewards, "score_responses")

    def scores_breakdown(
        self, completion: str, example: dict, state: dict | None = None
    ) -> dict[str, float]:
        return self._reward_to_breakdown(self._score_one(completion, example, state))

    def reward(self, completion: str, example: dict, state: dict | None = None) -> float:
        return float(getattr(self._score_one(completion, example, state), "score", 0.0))

    def reward_many(self, items: list[tuple[dict, dict]]) -> list[float]:
        """Reward for many ``(example, state)`` rollouts at once, in input order.

        For multi-turn, episodes that share a task go through ONE ``score_episodes`` call, which the
        env scores concurrently (``Environment.max_score_concurrency``) — replacing one blocking
        scoring call per rollout. For a judge / network-reward env (where scoring dominates) this is
        the multi-turn analogue of batched generation. Equals one :meth:`reward` per item:
        ``score_episodes`` scores each episode independently, so batching changes only concurrency,
        not values. Single-turn falls back to per-item :meth:`reward`."""
        if not self.multi_turn:
            # Single-turn scoring ignores state and grades the completion, so pass the rollout's
            # actual response (stored on the state) — not "" (which would score every item empty).
            return [self.reward(str(st.get("response_text") or ""), ex, st) for ex, st in items]
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
        return bool(self._score_one(completion, example, state).resolved_success())

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

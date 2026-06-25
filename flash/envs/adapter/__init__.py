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
from pathlib import Path
from typing import Any

from flash.envs.base import BaseEnvironment

_DEFAULT_ENVIRONMENT_PATH = "environment.py"
_CACHE_ROOT = Path(os.environ.get("FLASH_ENV_CACHE_DIR", "/tmp/flash-env-cache"))
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 5000
_DOWNLOAD_TIMEOUT_S = 120.0
_SLUG_SAFE_PART_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_TAR_METADATA_TYPES = {
    tarfile.XHDTYPE,
    tarfile.XGLTYPE,
    tarfile.GNUTYPE_LONGNAME,
    tarfile.GNUTYPE_LONGLINK,
}
_CANONICAL_INPUT_KEY = "input"
_CANONICAL_OUTPUT_KEY = "output"


def is_managed_environment_slug(value: str) -> bool:
    return _parse_managed_environment_slug(value) is not None


def is_freesolo_environment_id(value: str) -> bool:
    # Managed Hub slugs (``namespace/name``) are the only environment id form Flash accepts now that
    # package storage is Azure Blob (the old ``github:`` / github.com URL refs are gone).
    return is_managed_environment_slug(value)


def _parse_managed_environment_slug(value: str) -> tuple[str, str] | None:
    text = (value or "").strip()
    if not text or ":" in text:
        return None
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme or parsed.netloc:
        return None
    parts = text.split("/")
    if len(parts) != 2 or not _is_safe_slug_parts(tuple(parts)):
        return None
    return parts[0], parts[1]


def _is_safe_slug_parts(parts: list[str] | tuple[str, ...]) -> bool:
    if not parts:
        return False
    if any(part in {".", "..", ""} for part in parts):
        return False
    return all(_SLUG_SAFE_PART_RE.fullmatch(part) for part in parts)


def _download_bytes(url: str) -> bytes:
    """GET the bytes at ``url`` (an Azure Blob read SAS URL). No auth header — the SAS carries it."""
    req = urllib.request.Request(url, headers={"User-Agent": "freesolo-flash"})
    try:
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(
            f"environment package download failed (HTTP {exc.code}): {body[:300]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"environment package download failed: {exc.reason}") from exc


def _extract_env_package(tar_bytes: bytes, dest: Path) -> None:
    """Extract a published env ``.tar.gz`` (environment.py at the root) safely into ``dest``."""
    root = dest.resolve()
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
            total += max(0, member.size)
            if total > _MAX_ARCHIVE_BYTES:
                raise RuntimeError(
                    f"environment archive is too large uncompressed ({total} bytes; "
                    f"limit {_MAX_ARCHIVE_BYTES} bytes)"
                )
            member.name = normalized_name
            tar.extract(member, dest)


def _find_env_entrypoint(base: Path) -> Path | None:
    """Locate ``environment.py`` in an extracted package (at the root, or one wrapper dir down)."""
    if not base.is_dir():
        return None
    direct = base / _DEFAULT_ENVIRONMENT_PATH
    if direct.is_file():
        return direct
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        nested = subdirs[0] / _DEFAULT_ENVIRONMENT_PATH
        if nested.is_file():
            return nested
    return None


def _download_env_package(url: str, sha256: str | None) -> Path:
    """Download + verify + extract an env package from a SAS URL; return the cached environment.py.

    Content-addressed by sha (or the URL when no sha is pinned): a second run with the same package
    reuses the extracted disk cache under ``_CACHE_ROOT`` and never re-downloads.
    """
    cache_key = hashlib.sha256((sha256 or url).encode()).hexdigest()[:24]
    cache_dir = _CACHE_ROOT / cache_key
    cached = _find_env_entrypoint(cache_dir)
    if cached is not None:
        return cached
    data = _download_bytes(url)
    if len(data) > _MAX_ARCHIVE_BYTES:
        raise RuntimeError(
            f"environment archive is too large ({len(data)} bytes; limit {_MAX_ARCHIVE_BYTES} bytes)"
        )
    if sha256:
        got = hashlib.sha256(data).hexdigest()
        if got != sha256:
            raise RuntimeError(
                f"environment package checksum mismatch (expected {sha256}, got {got}); refusing "
                f"to load a tampered or truncated package"
            )
    tmp_parent = Path(tempfile.mkdtemp(prefix="flash-env-pkg-"))
    try:
        _extract_env_package(data, tmp_parent)
        entry = _find_env_entrypoint(tmp_parent)
        if entry is None:
            raise FileNotFoundError("environment package did not contain environment.py")
        base = entry.parent
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.copytree(base, cache_dir)
        return cache_dir / entry.name
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)


def _resolve_environment_reference(
    env_id: str, resolved_package_url: str | None = None, resolved_sha: str | None = None
) -> str:
    # Managed slug (namespace/name): the control plane resolves it to an Azure Blob read SAS URL at
    # submit (runner._assign_resolved_env_pkg) and threads it here. The worker downloads + verifies
    # + extracts the package and returns the path to environment.py. There is NO GitHub fallback.
    if is_managed_environment_slug(env_id):
        if resolved_package_url:
            return str(_download_env_package(resolved_package_url, resolved_sha or None))
        # Offline/local checkout at this path (tests), else a clear unresolved error.
        local = Path(env_id)
        if local.exists():
            return str(local)
        raise RuntimeError(
            f"environment {env_id!r} could not be resolved to a package: no Azure Blob URL was "
            f"provided (it may not be published — run `flash env push`)"
        )
    # A direct local path (offline tests / programmatic use).
    local = Path(env_id)
    if local.exists():
        return str(local)
    raise ValueError(
        f"unrecognized environment id {env_id!r}: expected a published slug 'namespace/name' "
        f"or an existing local path"
    )


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
        if isinstance(value, dict) and list(value) == ["messages"] and isinstance(value["messages"], list):
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

    @property
    def reward_thread_safe(self) -> bool:
        """Whether ``reward`` may be called concurrently across rollouts (multiturn_rollout scores a
        batch in a thread pool). The verifiers reward contract is a pure scorer — ``score_responses``
        reads the per-call inputs + immutable env config — so the default is True. An underlying env
        whose scorer keeps mutable state or a thread-bound client opts out with
        ``reward_thread_safe = False`` and is scored serially."""
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
    env_id: str,
    resolved_package_url: str | None = None,
    resolved_sha: str | None = None,
    /,
    **kwargs,
) -> FreesoloEnvironment:
    # resolved_package_url + resolved_sha are POSITIONAL-ONLY resolve-once hooks (the control-plane
    # Azure Blob read SAS URL + the package SHA-256). Positional-only (the `/`) precisely so a user
    # [environment.params] entry of ANY name lands in **kwargs and is forwarded verbatim to the
    # Freesolo SDK loader, never binding to or shadowing these internal hints. None (the default)
    # leaves the env unresolved unless ``env_id`` is an existing local path.
    tools = _import_freesolo_environment_tools()
    reference = _resolve_environment_reference(env_id, resolved_package_url, resolved_sha)
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
    "is_freesolo_environment_id",
    "is_managed_environment_slug",
    "load_freesolo_environment",
]

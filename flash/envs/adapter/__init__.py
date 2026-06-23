"""Adapter that runs Freesolo SDK environments on Flash.

Flash environment ids now reference source in GitHub instead of package wheels.
The canonical generated environment file is ``freesolo/environment.py`` and its
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
_DEFAULT_ENVIRONMENT_PATH = "freesolo/environment.py"
_CACHE_ROOT = Path(os.environ.get("FLASH_ENV_CACHE_DIR", "/tmp/flash-env-cache"))
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


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
        owner_repo = repo_part.split("/", 1)
        if len(owner_repo) == 2 and all(owner_repo):
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
    if len(parts) >= 5 and parts[2] in {"blob", "tree"}:
        ref = parts[3]
        raw_path = "/".join(parts[4:])
        try:
            path = _normalize_env_path(raw_path)
        except ValueError:
            return None
        if not raw_path:
            path = _DEFAULT_ENVIRONMENT_PATH
        if parts[2] == "tree" and raw_path and not path.endswith(".py"):
            path = f"{path.rstrip('/')}/{_DEFAULT_ENVIRONMENT_PATH}"
    else:
        ref = _DEFAULT_GITHUB_REF
        path = _DEFAULT_ENVIRONMENT_PATH
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


def _github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN")


def _is_commit_sha(value: str) -> bool:
    return _COMMIT_SHA_RE.fullmatch(value) is not None


def _resolve_ref_sha(parsed: GitHubEnvironmentRef) -> str:
    if _is_commit_sha(parsed.ref):
        return parsed.ref
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "freesolo-flash"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    commit_url = f"https://api.github.com/repos/{parsed.repo_full_name}/commits/{urllib.parse.quote(parsed.ref, safe='')}"
    req = urllib.request.Request(commit_url, headers=headers)
    data = _urlopen(req, timeout=60.0)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to resolve commit for {parsed.canonical()}: invalid GitHub response"
        ) from exc
    sha = payload.get("sha")
    if not isinstance(sha, str) or not _is_commit_sha(sha):
        raise RuntimeError(f"Failed to resolve commit for {parsed.canonical()}")
    return sha


def _urlopen(req: urllib.request.Request, *, timeout: float = 60.0) -> bytes:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"GitHub request failed ({exc.code}): {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub request failed: {exc.reason}") from exc


def _download_github_tarball(ref: GitHubEnvironmentRef) -> bytes:
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
            f"GitHub environment archive is too large ({len(data)} bytes; "
            f"limit {_MAX_ARCHIVE_BYTES} bytes)"
        )
    return data


def _safe_extract_archive(tar_bytes: bytes, dest: Path) -> Path:
    root = dest.resolve()
    top_dirs: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if not parts:
                continue
            top_dirs.add(parts[0])
            target = (dest / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe path in GitHub environment archive: {member.name!r}")
            if member.islnk() or member.issym() or not (member.isreg() or member.isdir()):
                continue
            tar.extract(member, dest)
    if len(top_dirs) != 1:
        raise RuntimeError("GitHub environment archive had an unexpected layout")
    extracted = dest / next(iter(top_dirs))
    if not extracted.is_dir():
        raise RuntimeError("GitHub environment archive did not extract to a directory")
    return extracted


def _resolve_github_environment_file(env_ref: str) -> Path:
    parsed = _parse_github_environment_ref(env_ref)
    if parsed is None:
        raise ValueError(f"not a GitHub environment ref: {env_ref!r}")
    resolved_ref = _resolve_ref_sha(parsed)
    cache_key = hashlib.sha256(
        f"github:{parsed.repo_full_name}@{resolved_ref}:{parsed.path}".encode()
    ).hexdigest()[:24]
    cache_dir = _CACHE_ROOT / cache_key
    env_file = cache_dir / parsed.path
    if env_file.is_file():
        return env_file
    tmp_parent = Path(tempfile.mkdtemp(prefix="flash-env-github-"))
    try:
        extracted = _safe_extract_archive(_download_github_tarball(parsed), tmp_parent)
        candidate = extracted / parsed.path
        if candidate.is_dir():
            nested = candidate / _DEFAULT_ENVIRONMENT_PATH
            flat = candidate / "environment.py"
            candidate = nested if nested.is_file() else flat
        if not candidate.is_file():
            raise FileNotFoundError(
                f"GitHub environment {parsed.canonical()} did not contain {parsed.path!r}"
            )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.copytree(extracted, cache_dir)
        return cache_dir / candidate.relative_to(extracted)
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)


def _resolve_environment_reference(env_ref: str) -> str:
    parsed = _parse_github_environment_ref(env_ref)
    if parsed is None:
        path = Path(env_ref)
        if path.exists():
            return str(path)
        return env_ref
    return str(_resolve_github_environment_file(env_ref))


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
        self.max_turns = 8

    def _task_example(self, example: dict):
        return self._task_example_from_record(example)

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
            record = dict(getattr(example, "record", {}) or {})
            record.setdefault("task", getattr(example, "task", ""))
            task_id = getattr(example, "task_id", None)
            if task_id is not None:
                record.setdefault("id", task_id)
            expected = getattr(example, "expected_output", None)
            if expected is not None:
                record.setdefault("expected_output", _json_safe(expected))
            metadata = getattr(example, "metadata", None)
            if isinstance(metadata, dict) and metadata:
                record.setdefault("metadata", metadata)
            records.append(record)
        return records

    def prompt_messages(self, example: dict) -> list[dict]:
        messages = self._env.start_episode(self._task_example(example), self._contract_text)
        return [dict(message) for message in messages]

    def sft_target(self, example: dict) -> str:
        for key in ("completion", "expected_output", "output", "target", "answer"):
            if example.get(key) is not None:
                value = example[key]
                if isinstance(value, list) and value and isinstance(value[-1], dict):
                    return str(value[-1].get("content", ""))
                return str(value)
        return ""

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
        messages = [dict(message) for message in self._env.start_episode(task, self._contract_text)]
        return {
            "task": task,
            "messages": messages,
            "turns": [],
            "done": False,
            "response_text": "",
            "turn": 0,
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
        return max_turns is not None and int(state.get("turn", 0)) >= int(max_turns)

    def _episode_from_state(self, state: dict):
        return self._EnvironmentEpisode(
            messages=tuple(state.get("messages") or ()),
            response_text=str(state.get("response_text") or ""),
            turns=tuple(state.get("turns") or ()),
            metadata={"steps": state.get("step_metadata", [])} if state.get("step_metadata") else {},
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


def load_freesolo_environment(env_id: str, **kwargs) -> FreesoloEnvironment:
    tools = _import_freesolo_environment_tools()
    reference = _resolve_environment_reference(env_id)
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
    contract_text = str(params.pop("contract_text", "") or _load_contract_text(params["contract_path"]))

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
    "is_github_environment_ref",
    "load_freesolo_environment",
]

"""Adapter that runs Freesolo SDK environments on Flash.

Loading, reference resolution, and dataset probing live in :mod:`flash.envs.loading.loader`; this
module keeps the environment adapter focused on episode execution and scoring.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from flash.envs.loading.base import (
    REWARD_GROUP_CONCURRENCY,
    BaseEnvironment,
    RolloutReward,
    map_bounded,
    with_system_prompt,
)
from flash.envs.loading.loader import (
    GitHubEnvironmentRef,
    GitHubRateLimitError,
    GitHubTransientError,
    GitHubUnavailableError,
    _import_freesolo_environment_tools,
    canonical_managed_environment_slug,
    env_dataset_rows,
    is_freesolo_environment_id,
    is_github_environment_ref,
    is_managed_environment_slug,
    load_freesolo_environment,
    managed_slug_to_github_ref,
)
from flash.teacher.limits import (
    OPD_DEFAULT_EPISODE_TURNS,
    OPD_MAX_EPISODE_TURNS,
    OPD_MIN_EPISODE_TURNS,
)

_CANONICAL_INPUT_KEY = "input"
_CANONICAL_OUTPUT_KEY = "output"
# ceiling on a .json dataset file the override diagnostic is willing to parse just to count rows.
_MAX_ROW_COUNT_JSON_BYTES = 16 * 1024 * 1024


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _copied_message_list(value: Any, *, row_id: Any, source: str) -> list[dict]:
    """Copy `value` as a chat message list, naming the offending row and entries on a bad shape."""
    if not isinstance(value, list):
        raise ValueError(
            f"sft output for row id {row_id!r} has a {source} value of type "
            f"{type(value).__name__}; expected a list of message objects"
        )
    invalid_indexes = [
        index for index, message in enumerate(value) if not isinstance(message, dict)
    ]
    if invalid_indexes:
        raise ValueError(
            f"sft output for row id {row_id!r} contains non-object {source} entries at "
            f"indexes {invalid_indexes}; expected a list of message objects"
        )
    return [dict(message) for message in value]


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
        prefer_env_dataset: bool = False,
        contract_text: str = "",
        package_root: str | Path | None = None,
    ):
        super().__init__(id=env_id)
        self._env = sdk_env
        self._source = source
        # set by the loader when source is a dataset_path file the sdk env itself received as a
        # param: the env instance's own dataset then wins over re-reading the raw file, so an env
        # that filters/subsamples/augments in load_environment trains on what it built.
        self._prefer_env_dataset = bool(prefer_env_dataset)
        self._contract_text = contract_text
        self.package_root = Path(package_root).resolve() if package_root is not None else None
        tools = _import_freesolo_environment_tools()
        self._task_example_from_record = tools["task_example_from_record"]
        self._load_task_examples = tools["load_task_examples"]
        self._EnvironmentEpisode = tools["EnvironmentEpisode"]
        self._EnvironmentTurn = tools["EnvironmentTurn"]
        self.multi_turn = isinstance(sdk_env, tools["EnvironmentMultiTurn"])
        # only an explicit class capability opts in. instance data and environment params cannot
        # turn dynamic image observations on accidentally.
        self.image_observations = getattr(type(sdk_env), "image_observations", False) is True
        self.is_tool_env = False
        self._max_turns_cache: int | None = None
        self._dataset_cache: list[dict] | None = None
        # cached dataset rows own the task that prompt preparation mutates. single-turn scoring
        # reuses it directly; multi-turn rollouts clone its prepared episode state per sibling.
        self._row_tasks: dict[int, Any] = {}
        # whether this run samples <think> blocks. the worker sets it from the JobSpec once the
        # env is loaded; off by default so a CLI-side load (flash env test) grades raw text, which
        # is what an echo/replay harness feeds it.
        self.thinking = False
        # whether the chat template pre-opens an unclosed <think> in every assistant generation
        # prompt (Qwen with enable_thinking does). the worker derives this from a REAL rendered
        # prompt, never from the thinking flag, and sets it alongside .thinking -- a template that
        # ignores enable_thinking must not have its tagless answers read as unterminated reasoning.
        # both parsers need it or a turn truncated before </think> grades as the whole ramble here
        # while the single-turn path correctly grades it empty.
        self.prompt_opens_thinking = False

    @property
    def max_turns(self) -> int:
        """Batch-level turn ceiling: dataset-wide max of per-example budgets, clamped to [8, 64]."""
        if self._max_turns_cache is not None:
            return self._max_turns_cache
        cap = OPD_MIN_EPISODE_TURNS
        if self.multi_turn:
            cap = OPD_DEFAULT_EPISODE_TURNS
            best: int | None = None
            for ex in self.dataset():
                try:
                    turns = int(self._env.max_episode_turns(self._task_example(ex)))
                except Exception:
                    continue
                if best is None or turns > best:
                    best = turns
            if best is not None:
                cap = max(OPD_MIN_EPISODE_TURNS, min(OPD_MAX_EPISODE_TURNS, best))
        self._max_turns_cache = cap
        return cap

    def _task_example(self, example: dict):
        """Return a cached dataset row's stable task, or a fresh task for any other row."""
        task = self._row_tasks.get(id(example))
        if task is not None:
            return task
        return self._task_example_from_record(self._canonical_record(example))

    def _with_system_prompt(self, messages: list[dict]) -> list[dict]:
        return with_system_prompt(messages, self._contract_text)

    def _canonical_record(self, record: dict) -> dict:
        raw = dict(record)
        if _CANONICAL_INPUT_KEY not in raw:
            raise ValueError("Freesolo dataset records must contain an input field")
        # Record ids are always auto-generated (see dataset()); they are never
        # required from, or read out of, the user's source data.
        return raw

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

    def _source_row_count(self) -> int | None:
        """row count of the packaged dataset file, for the override log line only."""
        if not isinstance(self._source, str):
            return None
        path = Path(self._source)
        try:
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as f:
                    return sum(1 for line in f if line.strip())
            # a .json dataset can only be counted by parsing all of it, and the env has already
            # replaced it. give the diagnostic up rather than let it be the allocation that runs
            # the worker out of memory on a file nothing else in this path reads.
            if path.stat().st_size > _MAX_ROW_COUNT_JSON_BYTES:
                return None
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if isinstance(loaded, list):
            return len(loaded)
        if isinstance(loaded, dict) and isinstance(loaded.get("records"), list):
            return len(loaded["records"])
        return None

    def dataset(self) -> list[dict]:
        if self._dataset_cache is not None:
            return self._dataset_cache
        # the environment owns its dataset: when the instance exposes one (the documented
        # `self.dataset = load_task_examples(...)` pattern, including filtering/subsampling),
        # it wins over re-reading the dataset_path file it was handed; the file is only the
        # fallback for envs with no in-code dataset. explicit [environment.params] records
        # never reach the env, so they keep precedence over a hardcoded env dataset.
        env_rows = env_dataset_rows(self._env)
        env_precedence = self._source is None or self._prefer_env_dataset
        # only an ABSENT attribute means "no in-code dataset". a dataset the env built and left
        # empty (a filter that matched nothing) is a deliberate answer, so it must not fall
        # through to the unfiltered packaged file: training on the rows the env rejected is a
        # silent wrong-data run, worse than refusing to start. an explicit non-default
        # dataset_path/records keeps the file authoritative, so there is nothing to refuse there.
        if env_rows is not None and not env_rows and env_precedence:
            raise ValueError(
                f"environment produced 0 rows ({self.id}): its in-code dataset is empty, and "
                "Flash will not fall back to the packaged dataset file (that would train on the "
                "rows the environment filtered out). Fix the filter that emptied it, or remove "
                "the dataset attribute to train on the packaged file."
            )
        use_env_rows = bool(env_rows) and env_precedence
        if use_env_rows:
            examples = self._load_task_examples(env_rows)
        elif self._source is not None:
            examples = self._load_task_examples(self._source)
        else:
            raise ValueError(
                "Freesolo environment has no dataset source. Set "
                "[environment.params] dataset_path or records so Flash can train."
            )
        if use_env_rows and self._source is not None:
            file_rows = self._source_row_count()
            if file_rows is not None and file_rows != len(examples):
                print(
                    f"[envs] training on the environment's own dataset ({len(examples)} rows), "
                    f"not the packaged file {self._source} ({file_rows} rows)"
                )
        records = []
        row_tasks: dict[int, Any] = {}
        for example in examples:
            raw = dict(getattr(example, "record", {}) or {})
            if _CANONICAL_INPUT_KEY not in raw and getattr(example, "input", None) is not None:
                raw[_CANONICAL_INPUT_KEY] = example.input
            if getattr(example, "output", None) is not None:
                raw.setdefault(_CANONICAL_OUTPUT_KEY, _json_safe(example.output))
            # Stamp the SDK-assigned positional id onto the record, overriding
            # any id the source data carried. Ids are always auto-generated so
            # they stay unique and deterministic (local SDK == remote worker).
            example_id = getattr(example, "id", None)
            if example_id:
                raw["id"] = example_id
            record = self._canonical_record(raw)
            records.append(record)
            # single-turn prompt and scoring hooks share the row-owned task.
            row_tasks[id(record)] = self._task_example_from_record(record)
        self._dataset_cache = records
        self._row_tasks = row_tasks
        return records

    def prompt_messages(self, example: dict) -> list[dict]:
        messages = self._env.start_episode(self._task_example(example), self._contract_text)
        return self._with_system_prompt(messages)

    def sft_completion(self, example: dict) -> list[dict]:
        """Target completion messages for one SFT example; falls back to raw record output."""
        messages, _coerced_scalar_output = self.sft_completion_with_provenance(example)
        return messages

    def sft_completion_with_provenance(self, example: dict) -> tuple[list[dict], bool]:
        """Return completion messages and whether a scalar output was coerced into one turn.

        the flag marks ONLY the final ``str(value)`` coercion. an explicitly structured target --
        the env's own hook, a message list, or a ``{"messages": [...]}`` container -- is not
        coerced, so a dataset that already encodes real trajectories never trips the collapse
        warning.
        """
        fn = getattr(self._env, "sft_completion", None)
        if callable(fn):
            msgs = fn(self._task_example(example))
            if msgs:
                return [dict(m) for m in msgs], False
        row_id = example.get("id")
        value = example.get(_CANONICAL_OUTPUT_KEY)
        if isinstance(value, list) and any(isinstance(message, dict) for message in value):
            return _copied_message_list(value, row_id=row_id, source="output"), False
        if isinstance(value, dict) and "messages" in value:
            sibling_keys = sorted(str(key) for key in value if key != "messages")
            if sibling_keys:
                raise ValueError(
                    f"sft output for row id {row_id!r} has 'messages' alongside sibling keys "
                    f"{sibling_keys}; expected exactly {{'messages': [...]}}"
                )
            return _copied_message_list(
                value["messages"], row_id=row_id, source="'messages'"
            ), False
        return [{"role": "assistant", "content": "" if value is None else str(value)}], True

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

    def scores_breakdown_many(self, items: list[tuple[dict, dict]]) -> list[dict[str, float]]:
        """Named reward components for many single-turn rollouts, in input order."""
        if self.multi_turn:
            raise RuntimeError(
                "scores_breakdown_many is only available for single-turn environments"
            )
        if not self.reward_thread_safe:
            return [
                self.scores_breakdown(str(state.get("response_text") or ""), example, state)
                for example, state in items
            ]
        results = self._grouped_results(
            items,
            task_of=lambda example, state: self._task_example(example),
            payload_of=lambda state: _completion_for_scoring(
                str(state.get("response_text") or ""), state
            ),
            scorer=self._env.score_responses,
            method="score_responses",
        )
        return [self._reward_to_breakdown(result) for result in results]

    def reward(self, completion: str, example: dict, state: dict | None = None) -> float:
        return float(getattr(self._score_one(completion, example, state), "score", 0.0))

    def reward_with_error(
        self, completion: str, example: dict, state: dict | None = None
    ) -> tuple[float, str, str]:
        """This completion's reward, the scorer's ``RewardResult.error``, and the text scored.

        ``reward`` above keeps only ``score``, so a scorer that crashed behind the SDK's guard and
        one that judged the answer wrong both reach training as ``0.0``. All three values come from a
        single scoring call because scoring is not guaranteed to be pure: a rate-limited judge or a
        flaky dependency can answer differently the second time, so re-scoring to read the error
        can report one that did not produce this reward -- and bills a paid judge twice.

        The third value is what the grader received, which the caller cannot reconstruct: for a
        multi-turn episode it is ``state["response_text"]``, which ``env_reply`` replaces outright
        when ``step_episode`` returns a ``final_response_text`` override.
        """
        result = self._score_one(completion, example, state)
        return (
            float(getattr(result, "score", 0.0)),
            str(getattr(result, "error", "") or ""),
            self._scored_completion_text(completion, state),
        )

    def _scored_completion_text(self, completion: str, state: dict | None) -> str:
        """The exact text ``_score_one`` hands the grader for this call.

        Mirrors the branch in ``_score_one``: a multi-turn episode is graded from the rollout
        state, whose ``response_text`` ``env_reply`` overrides when the env supplies a
        ``final_response_text``; every other path grades the completion passed in.
        """
        if state and self.multi_turn:
            return str(self._episode_from_state(state).response_text or "")
        return str(_completion_for_scoring(completion, state) or "")

    @staticmethod
    def _turn_rewards_from_result(result) -> tuple[float, ...] | None:
        metadata = getattr(result, "metadata", None)
        if metadata is None:
            return None
        if not isinstance(metadata, Mapping):
            print("[grpo][warn] malformed per_turn_rewards metadata; using episode reward")
            return None
        values = metadata.get("per_turn_rewards")
        if values is None:
            return None
        if not isinstance(values, Iterable) or isinstance(
            values, (str, bytes, bytearray, Mapping, set, frozenset)
        ):
            print("[grpo][warn] malformed per_turn_rewards metadata; using episode reward")
            return None
        try:
            return tuple(float(value) for value in values)
        except (TypeError, ValueError):
            print("[grpo][warn] malformed per_turn_rewards metadata; using episode reward")
            return None

    def _grouped_results(self, items, *, task_of, payload_of, scorer, method: str) -> list:
        """Group rollouts that share an example and scatter full reward results in input order."""
        groups: dict[tuple[str, int | None], dict] = {}
        order: list[tuple[str, int | None]] = []
        for i, (ex, st) in enumerate(items):
            # only the FIRST member's task is passed to the scorer, so members must agree on it.
            # sibling rollouts of one row carry their own session task that each episode mutated
            # (`new_rollout_state`), and those are not interchangeable: grouping them on the row
            # value alone would score every sibling against whichever session happened to be first.
            session_task = st.get("task")
            key = (
                json.dumps(ex, sort_keys=True, default=str),
                None if session_task is None else id(session_task),
            )
            grp = groups.get(key)
            if grp is None:
                grp = groups[key] = {"task": task_of(ex, st), "idxs": [], "payloads": []}
                order.append(key)
            grp["idxs"].append(i)
            grp["payloads"].append(payload_of(st))
        out: list = [None] * len(items)

        def score_group(key: tuple[str, int | None]) -> tuple[tuple[str, int | None], list]:
            grp = groups[key]
            rewards = scorer(grp["task"], grp["payloads"])
            if len(rewards) != len(grp["payloads"]):
                raise RuntimeError(f"Freesolo environment {method} returned the wrong length")
            return key, rewards

        # groups are scored CONCURRENTLY, not one after another. each call already scores its own
        # completions concurrently, but a serial loop over groups still serializes the round trips
        # ACROSS prompts, so a step with N prompts pays N judge latencies end to end while the gpu
        # idles. the env's own pool is per-instance and sized once (freesolo Environment
        # `_score_concurrently` / `max_score_concurrency`), so overlapping calls SHARE those workers
        # and the provider-facing rate stays capped where it already was -- this removes a barrier
        # rather than raising concurrency.
        #
        # on a raise, map_bounded pays for the groups already in flight where the serial loop
        # stopped at the failing one, and rl_train.py:1001 re-scores the batch serially, so those
        # calls are billed twice. bounded at one pool width; REWARD_GROUP_CONCURRENCY sets it.
        scored = map_bounded(
            order,
            score_group,
            cap=REWARD_GROUP_CONCURRENCY,
            serial=not self.reward_thread_safe,
        )

        for key, rewards in scored:
            for idx, reward in zip(groups[key]["idxs"], rewards, strict=True):
                out[idx] = reward
        return out

    def _grouped_score(self, items, *, task_of, payload_of, scorer, method: str) -> list[float]:
        """Group rollouts that share an example and scatter scalar scores in input order."""
        return [
            float(result.score)
            for result in self._grouped_results(
                items,
                task_of=task_of,
                payload_of=payload_of,
                scorer=scorer,
                method=method,
            )
        ]

    def reward_many(self, items: list[tuple[dict, dict]]) -> list[float]:
        """Reward many ``(example, state)`` rollouts in input order.

        Rollouts sharing a task use one ``score_responses`` or ``score_episodes`` batch, changing
        concurrency but not values. ``reward_thread_safe = False`` must use serial single-item
        :meth:`reward` calls because the scorer may hold mutable or thread-bound state.
        """
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

    def rollout_rewards_many(self, items: list[tuple[dict, dict]]) -> list[RolloutReward]:
        """Score terminal episodes once and return typed rewards in input order."""
        if not self.multi_turn:
            raise RuntimeError("rollout_rewards_many is only available for multi-turn environments")

        score_episodes = self._env.score_episodes
        if not self.reward_thread_safe:

            def _serial_score_episodes(task, episodes):
                return [
                    self._single(self._env.score_episodes(task, [episode]), "score_episodes")
                    for episode in episodes
                ]

            score_episodes = _serial_score_episodes

        results = self._grouped_results(
            items,
            task_of=lambda ex, st: st.get("task") or self._task_example(ex),
            payload_of=self._episode_from_state,
            scorer=score_episodes,
            method="score_episodes",
        )
        return [
            RolloutReward(
                episode=float(result.score),
                turns=self._turn_rewards_from_result(result),
            )
            for result in results
        ]

    @property
    def reward_thread_safe(self) -> bool:
        """Whether reward() may be called concurrently; delegates to the underlying env."""
        return bool(getattr(self._env, "reward_thread_safe", True))

    def grade(self, completion: str, example: dict, state: dict | None = None) -> bool:
        return bool(self._score_one(completion, example, state).resolved_success())

    def tools(self) -> list:
        return []

    def new_rollout_state(self, example: dict, prepared_prompt: list[dict] | None = None) -> dict:
        if prepared_prompt is None:
            record = deepcopy(self._canonical_record(example))
            task = self._task_example_from_record(record)
            prompt = self._with_system_prompt(self._env.start_episode(task, self._contract_text))
        else:
            prepared_task = self._row_tasks.get(id(example))
            if prepared_task is None:
                raise RuntimeError("prepared Freesolo rollout requires its dataset row task")
            task = deepcopy(prepared_task)
            prompt = deepcopy(prepared_prompt)
        try:
            episode_turns: int | None = int(self._env.max_episode_turns(task))
        except Exception:
            episode_turns = None
        messages = (
            [dict(message) for message in prompt] if prepared_prompt is None else deepcopy(prompt)
        )
        return {
            "task": task,
            "prompt": prompt,
            "messages": messages,
            "turns": [],
            "done": False,
            "response_text": "",
            "raw_response_text": "",
            "turn": 0,
            "max_episode_turns": episode_turns,
        }

    def record_model_turn(self, state: dict, content: str) -> dict:
        # the transcript keeps the raw turn -- it is what the model actually emitted, and what the
        # next turn must be conditioned on. the text handed to the scorer is answer-only, matching
        # what the single-turn path grades. without this the two modes give a grader different shapes
        # of the same completion, so an env can score correctly in one mode and silently mis-grade in
        # the other.
        msg = {"role": "assistant", "content": content}
        state.setdefault("messages", []).append(msg)
        state.setdefault("turns", []).append(
            self._EnvironmentTurn(role="assistant", content=content)
        )
        state["response_text"] = self._scored_turn_text(content)
        # step_episode drives the episode, it does not grade it: it parses the action, and often
        # requires assistant_response to equal messages[-1]["content"]. that message is raw, so
        # handing it the stripped text would step the env on something the model never emitted.
        state["raw_response_text"] = content
        return msg

    def _scored_turn_text(self, content: str):
        """The assistant turn as the scorer should see it: answer-only, reasoning available.

        Both parsers get ``prompt_opens_thinking``, exactly as the single-turn path forwards its
        own ``_prompt_opens_thinking`` (flash/engine/worker/entry/rl.py). Without it a turn truncated
        before ``</think>`` is tagless reasoning that ``strip_think`` returns whole as the answer,
        so the rollout can be rewarded for unfinished thinking that single-turn grading scores 0.
        """
        if not self.thinking:
            return content
        from flash.content.thinking import strip_think, thinking_text

        opened = self.prompt_opens_thinking
        answer = strip_think(content, prompt_opened_thinking=opened)
        return _ScoredResponseText(
            answer if isinstance(answer, str) else content,
            raw=content,
            thinking=thinking_text(content, prompt_opened_thinking=opened),
        )

    def env_reply(self, messages: list[dict], state: dict) -> list[dict]:
        if not self.multi_turn:
            return []
        task = state.get("task")
        if task is None:
            raise RuntimeError("missing Freesolo rollout task state")
        # the raw turn, not the scored one: see record_model_turn. falls back to response_text for a
        # state built by something other than record_model_turn (where the two are the same text).
        raw = state.get("raw_response_text")
        if not isinstance(raw, str):
            raw = str(state.get("response_text") or "")
        step = self._env.step_episode(task, list(messages), raw)
        state["done"] = bool(step.done)
        if step.final_response_text is not None:
            # the env overrode the episode's answer, so it is already the text to grade -- do not
            # strip it. keep the raw view in step with it for any later turn.
            final = str(step.final_response_text)
            # wrap the override so `.completion` is env-authored while `.raw` and `.thinking` remain
            # the model's original turn (flash/cli/scaffold/__init__.py). a bare str would discard those
            # scorer views. `raw_response_text` becomes the override because later turns step on it.
            previous = state.get("response_text")
            state["response_text"] = (
                _ScoredResponseText(
                    final,
                    raw=raw,
                    thinking=getattr(previous, "thinking", None),
                )
                if self.thinking
                else final
            )
            state["raw_response_text"] = final
        state["turn"] = int(state.get("turn", 0)) + 1
        if step.metadata:
            state.setdefault("step_metadata", []).append(step.metadata)
        replies = [deepcopy(dict(message)) for message in step.messages]
        scored_replies = deepcopy(replies)
        state.setdefault("messages", []).extend(scored_replies)
        for message in scored_replies:
            # score_episode receives a structural copy of the environment's raw chat content. the
            # parent bridge flattens only the separate child-bound reply returned below.
            state.setdefault("turns", []).append(
                self._EnvironmentTurn(
                    role=str(message.get("role", "")),
                    content=deepcopy(message.get("content", "")),
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
        # not str(): response_text may be a _ScoredResponseText carrying .raw/.thinking, and str()
        # on a str subclass drops back to a plain str, losing the structured views.
        response_text = state.get("response_text")
        return self._EnvironmentEpisode(
            messages=tuple(state.get("messages") or ()),
            response_text="" if response_text is None else response_text,
            turns=tuple(state.get("turns") or ()),
            metadata={"steps": state.get("step_metadata", [])}
            if state.get("step_metadata")
            else {},
        )

    def _score_episode(self, example: dict, state: dict):
        task = state.get("task") or self._task_example(example)
        rewards = self._env.score_episodes(task, [self._episode_from_state(state)])
        return self._single(rewards, "score_episodes")


__all__ = [
    "FreesoloEnvironment",
    "GitHubEnvironmentRef",
    "GitHubRateLimitError",
    "GitHubTransientError",
    "GitHubUnavailableError",
    "canonical_managed_environment_slug",
    "is_freesolo_environment_id",
    "is_github_environment_ref",
    "is_managed_environment_slug",
    "load_freesolo_environment",
    "managed_slug_to_github_ref",
]

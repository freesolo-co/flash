"""Adapter that runs Freesolo SDK environments on Flash.

Loading / reference resolution / dataset probing live in :mod:`flash.envs.loader`; the loader-side
public names are re-exported here so existing ``flash.envs.adapter`` import paths keep working.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from flash.envs.base import BaseEnvironment, RolloutReward
from flash.envs.loader import (
    GitHubEnvironmentRef,
    GitHubRateLimitError,
    _import_freesolo_environment_tools,
    canonical_managed_environment_slug,
    is_freesolo_environment_id,
    is_github_environment_ref,
    is_managed_environment_slug,
    load_freesolo_environment,
    managed_slug_to_github_ref,
)
from flash.opd_limits import (
    OPD_DEFAULT_EPISODE_TURNS,
    OPD_MAX_EPISODE_TURNS,
    OPD_MIN_EPISODE_TURNS,
)

_CANONICAL_INPUT_KEY = "input"
_CANONICAL_OUTPUT_KEY = "output"

# how many TASK GROUPS may be scored at once. this bounds in-flight scorer CALLS, not the requests
# they make: each call fans out inside the env's own per-instance pool, whose max_score_concurrency
# is enforced globally across overlapping calls, so this cannot raise the provider-facing rate. it
# exists only to keep a large rollout batch from creating one thread per prompt.
_REWARD_GROUP_CONCURRENCY = 8


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
        package_root: str | Path | None = None,
    ):
        super().__init__(id=env_id)
        self._env = sdk_env
        self._source = source
        self._contract_text = contract_text
        self.package_root = Path(package_root).resolve() if package_root is not None else None
        tools = _import_freesolo_environment_tools()
        self._task_example_from_record = tools["task_example_from_record"]
        self._load_task_examples = tools["load_task_examples"]
        self._EnvironmentEpisode = tools["EnvironmentEpisode"]
        self._EnvironmentTurn = tools["EnvironmentTurn"]
        self.multi_turn = isinstance(sdk_env, tools["EnvironmentMultiTurn"])
        self.is_tool_env = False
        self._max_turns_cache: int | None = None
        self._dataset_cache: list[dict] | None = None
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
        return self._task_example_from_record(self._canonical_record(example))

    def _with_system_prompt(self, messages: list[dict]) -> list[dict]:
        """Ensure the training contract rides as the system prompt (fill blank / prepend)."""
        system_text = str(self._contract_text or "").strip()
        out = [dict(message) for message in messages]
        if not system_text:
            return out
        first_blank_system_index: int | None = None
        for index, message in enumerate(out):
            if str(message.get("role") or "").strip().lower() != "system":
                continue
            content = message.get("content")
            has_content = bool(content.strip()) if isinstance(content, str) else bool(content)
            if has_content:
                return out
            if first_blank_system_index is None:
                first_blank_system_index = index
        if first_blank_system_index is not None:
            out[first_blank_system_index]["content"] = system_text
            return out
        return [{"role": "system", "content": system_text}, *out]

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
        self._dataset_cache = records
        return records

    def prompt_messages(self, example: dict) -> list[dict]:
        messages = self._env.start_episode(self._task_example(example), self._contract_text)
        return self._with_system_prompt(messages)

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
        out: list = [None] * len(items)

        def score_group(key: str) -> tuple[str, list]:
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
        # a non-thread-safe env never reaches here: reward_many/rollout_rewards_many route it to a
        # serial scorer above. one group still runs inline, since a pool for a single call is pure
        # overhead and keeps the common single-prompt path exactly as it was.
        if not self.reward_thread_safe or len(order) <= 1:
            scored = [score_group(key) for key in order]
        else:
            pool = ThreadPoolExecutor(max_workers=min(_REWARD_GROUP_CONCURRENCY, len(order)))
            try:
                # `map` propagates the first exception and preserves input order, so a scorer that
                # raises fails the step exactly as the serial loop did.
                #
                # a FAILING scorer costs more work here than it did serially: the serial loop
                # stopped AT the failing group, this also pays for whatever is already running.
                # rl_train.py:1001 catches the raise and re-scores the batch serially, so those
                # calls are billed twice. the excess is bounded at one pool width -- measured on a
                # 40-group batch at width 8, 10 groups execute when group 2 raises and 28 when
                # group 20 does. `map` and `cancel_futures=True` hold that bound INDEPENDENTLY:
                # map's result generator cancels the pending futures when the exception abandons
                # it, and shutdown cancels whatever is still queued. either alone gives 10; only
                # submit-then-gather with a plain shutdown runs all 40. both are kept because the
                # cost is a keyword argument and neither is obviously the one a later edit keeps.
                #
                # two ways to shrink the excess further were measured and rejected. an abort flag
                # checked before each call saves exactly ONE group (10 -> 9, 28 -> 27), because the
                # waste is work already running when the failure surfaces, not work queued behind
                # it -- not worth the branch. returning partial results is worse than it looks:
                # `out` would hold `None` for unscored rows, and both `_grouped_score`
                # (`float(result.score)`) and multiturn_reward_scoring's `_validated_reward`
                # (`float(reward.episode)`) dereference every element, so a None row converts a
                # partial success into an AttributeError that the multi-turn path does not catch.
                # the honest bound is the pool width, and _REWARD_GROUP_CONCURRENCY sets it.
                scored = list(pool.map(score_group, order))
            finally:
                pool.shutdown(wait=True, cancel_futures=True)

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

    def new_rollout_state(self, example: dict) -> dict:
        task = self._task_example(example)
        prompt = self._with_system_prompt(self._env.start_episode(task, self._contract_text))
        try:
            episode_turns: int | None = int(self._env.max_episode_turns(task))
        except Exception:
            episode_turns = None
        messages = [dict(message) for message in prompt]
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
        own ``_prompt_opens_thinking`` (flash/engine/worker/rl.py). Without it a turn truncated
        before ``</think>`` is tagless reasoning that ``strip_think`` returns whole as the answer,
        so the rollout can be rewarded for unfinished thinking that single-turn grading scores 0.
        """
        if not self.thinking:
            return content
        from flash.thinking import strip_think, thinking_text

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
            # the model's original turn (flash/cli/training_doc.py). a bare str would discard those
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

    def reward_from_messages(
        self, completion_msgs: list[dict], example: dict, prompt_msgs: list[dict] | None = None
    ) -> float:
        messages = [*(prompt_msgs or []), *completion_msgs]
        response_text: object = ""
        turns = []
        for message in completion_msgs:
            content = str(message.get("content", ""))
            role = str(message.get("role", ""))
            turns.append(self._EnvironmentTurn(role=role, content=content))
            if role == "assistant":
                # same shape the single-turn path grades; see record_model_turn.
                response_text = self._scored_turn_text(content)
        episode = self._EnvironmentEpisode(
            messages=tuple(dict(m) for m in messages),
            response_text=response_text,
            turns=tuple(turns),
        )
        rewards = self._env.score_episodes(self._task_example(example), [episode])
        return float(self._single(rewards, "score_episodes").score)


__all__ = [
    "FreesoloEnvironment",
    "GitHubEnvironmentRef",
    "GitHubRateLimitError",
    "canonical_managed_environment_slug",
    "is_freesolo_environment_id",
    "is_github_environment_ref",
    "is_managed_environment_slug",
    "load_freesolo_environment",
    "managed_slug_to_github_ref",
]

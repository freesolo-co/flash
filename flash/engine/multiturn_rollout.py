"""Multi-turn / tool GRPO rollout for TRL's experimental ``rollout_func`` (colocate vLLM).

Drives a Freesolo ``EnvironmentMultiTurn`` turn loop, returning the full interleaved token
sequence with an ``env_mask`` (1=model, 0=env/tool) for multi-turn credit assignment.
:func:`rollout_one` is unit-testable on CPU; :func:`build_rollout_func` wires the real engine.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import json
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TypedDict


class RolloutResult(TypedDict):
    """Token-aligned fields returned per rollout."""

    prompt_ids: list[int]
    completion_ids: list[int]
    logprobs: list[float]
    env_mask: list[int]
    reward: float


RolloutCompletion = tuple[str, list[int], list[float], str]


class RolloutRequestExhaustedError(RuntimeError):
    """Raised when one logical assistant turn exhausts its physical request attempts."""

    def __init__(self, *, attempts: int, reason: str):
        self.attempts = attempts
        self.reason = reason
        super().__init__(
            f"multi-turn rollout request exhausted {attempts} physical attempt(s): {reason}"
        )


class TurnRecord(TypedDict):
    """One assistant turn of a multi-turn episode, as :func:`rollout_one_records` emits it.

    ``prefix_ids`` are the student token ids of the whole transcript BEFORE this turn (initial prompt
    + every prior assistant turn + every inter-turn env "glue"), i.e. the exact on-policy context the
    student sampled this turn's completion after. ``context_messages`` is the parallel message list at
    the same point (prompt + prior turns + prior env replies), for callers that render a separate
    teacher/scoring prompt from it. ``gen`` is whatever the injected ``generate`` callable returned for
    this turn (opaque to the driver beyond the four attributes it reads: ``completion_ids``,
    ``completion_text``, ``truncated``, ``skip``) — e.g. OPD hands back its ``_GenResult`` so the
    per-turn record feeds straight into the existing single-turn scoring/loss path. Distilling each
    turn against ``prefix_ids`` is the multi-turn on-policy-distillation objective: the episode's total
    reverse-KL over student-generated tokens is the sum of per-turn reverse-KLs, each conditioned on the
    transcript so far."""

    prefix_ids: list[int]
    gen: object
    context_messages: list[dict]


_ROLLOUT_FIELDS: tuple[str, ...] = (
    "prompt_ids",
    "completion_ids",
    "logprobs",
    "env_mask",
    "reward",
)


def _strip_none(obj):
    """Drop None-valued dict entries recursively. Dataset.from_list unifies a list<struct> schema and
    materializes ``key: null`` on rows that lacked a key another row added (e.g. a `name`/`tool_calls`
    message field); stripping those nulls keys an Arrow-materialized prompt identically to the raw row."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def _prompt_key(prompt) -> str:
    """Stable string key for a dataset ``prompt`` value (insensitive to Arrow null-injection)."""
    try:
        return json.dumps(_strip_none(prompt), sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(prompt)


class _LRUCache:
    """Tiny bounded LRU cache (string key -> ``list[int]``).

    Evicts the LRU entry when full rather than freezing, so recently-seen keys stay cached over a
    long diverse run. Not thread-safe; each cache is owned by a single closure.
    """

    __slots__ = ("_data", "maxsize")

    def __init__(self, maxsize: int):
        if maxsize <= 0:
            raise ValueError("LRU cache maxsize must be positive")
        self.maxsize = maxsize
        self._data: OrderedDict[str, list[int]] = OrderedDict()

    def get(self, key: str) -> list[int] | None:
        """Return cached value (MRU-bumped) or None."""
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def put(self, key: str, value: list[int]) -> None:
        """Insert/refresh ``key``, evicting LRU entry if at capacity."""
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


def build_examples_index(rows: list[dict], prompt_of: Callable[[dict], object]) -> dict:
    """Map each row's prompt key to the example row. Collisions keep the last row."""
    return {_prompt_key(prompt_of(r)): r for r in rows}


def index_collisions(rows: list[dict], prompt_of: Callable[[dict], object]) -> int:
    """Rows dropped by prompt-key collisions in :func:`build_examples_index`."""
    return len(rows) - len({_prompt_key(prompt_of(r)) for r in rows})


def _dedup_seam_terminator(prev_completion_ids: list[int], glue: list[int]) -> list[int]:
    """Collapse a duplicate turn terminator at the turn/env seam.

    ``env_glue`` leads with the assistant turn's terminator (e.g. ``<|im_end|>``), which a
    naturally-stopped assistant turn also keeps as its final token — so the raw stream would carry two.
    When the previous turn's last token equals ``glue[0]`` drop the glue copy, keeping the assistant's
    own token (real logprob / model-generated) so the stream has exactly one terminator per turn,
    matching the chat template + SFT transcripts. Shared by :func:`rollout_one`,
    :func:`_advance_after_turn`, and :func:`rollout_one_records` so the three paths can't drift."""
    if glue and prev_completion_ids and prev_completion_ids[-1] == glue[0]:
        return glue[1:]
    return glue


def rollout_one(
    *,
    example: dict,
    active_env,
    render: Callable[[list, bool], list[int]],
    generate: Callable[[list, int], tuple[list[int], list[float], str]],
    env_glue: Callable[[list], list[int]],
    max_turns: int,
    per_turn_max_tokens: int,
    engine_max_len: int | None = None,
) -> RolloutResult:
    """Run one multi-turn/tool rollout and return TRL ``rollout_func`` fields for it."""
    state = active_env.new_rollout_state(example)
    initial_messages = state.get("prompt") or state.get("messages")
    if not isinstance(initial_messages, list):
        raise KeyError("multi-turn rollout state must include prompt or messages")
    messages = [dict(m) for m in initial_messages]
    prompt_ids = render(messages, True)
    cur_ids = list(prompt_ids)  # invariant: cur_ids == prompt_ids + completion_ids so far
    token_budget = (engine_max_len - len(prompt_ids) - 8) if engine_max_len else None
    completion_ids: list[int] = []
    logprobs: list[float] = []
    env_mask: list[int] = []

    turns = 0
    while True:
        max_new = per_turn_max_tokens
        if token_budget is not None:
            remaining = token_budget - len(completion_ids)
            if remaining <= 0:
                break
            max_new = min(max_new, remaining)
        asst_ids, asst_lp, text = generate(cur_ids, max_new)
        completion_ids.extend(asst_ids)
        logprobs.extend(asst_lp)
        env_mask.extend([1] * len(asst_ids))
        cur_ids.extend(asst_ids)
        active_env.record_model_turn(state, text)
        messages.append({"role": "assistant", "content": text})
        turns += 1

        if token_budget is not None and len(completion_ids) >= token_budget:
            break
        if turns >= max_turns or active_env.rollout_done(state, max_turns):
            break
        env_msgs = active_env.env_reply(messages, state)
        if not env_msgs:
            break
        messages.extend(env_msgs)
        # Don't append glue if the env step finished — no next model turn.
        if active_env.rollout_done(state, max_turns):
            break

        glue = _dedup_seam_terminator(completion_ids, env_glue(env_msgs))
        if token_budget is not None and len(completion_ids) + len(glue) > token_budget:
            break
        completion_ids.extend(glue)
        logprobs.extend([0.0] * len(glue))
        env_mask.extend([0] * len(glue))
        cur_ids.extend(glue)

    reward = active_env.reward("", example, state)
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask,
        "reward": float(reward),
    }


def rollout_one_records(
    *,
    example: dict,
    active_env,
    render: Callable[[list, bool], list[int]],
    generate: Callable[[list, int], object],
    env_glue: Callable[[list], list[int]],
    max_turns: int,
    per_turn_max_tokens: int,
    engine_max_len: int | None = None,
    on_turn_generated: Callable[[], None] | None = None,
) -> list[TurnRecord]:
    """Drive one multi-turn episode and return a per-turn :class:`TurnRecord` list (NOT a flat masked
    sequence like :func:`rollout_one`). Used by OPD, which distils each assistant turn as an independent
    single-turn sample conditioned on the transcript so far.

    The env turn loop is identical to :func:`rollout_one` (``new_rollout_state`` → generate →
    ``record_model_turn`` → ``rollout_done`` → ``env_reply`` → glue), and the tokenizer-sensitive seam
    dedup / engine-budget accounting reuse the same helpers, so the two paths can't drift. The
    differences are all OPD-shaped:

    - ``generate(prefix_ids, max_new)`` returns an OBJECT (opaque here) exposing ``completion_ids``
      (``list[int] | None``), ``completion_text`` (``str``), ``truncated`` (``bool``) and ``skip``
      (``bool``) — OPD passes its ``_GenResult`` after termination/trim/U+FFFD gates have already been
      applied, so the record drops straight into the existing teacher-scoring + batched-loss path.
    - Every turn (including a truncated/empty one) is recorded so the caller counts it; a
      truncated/skip turn ENDS the episode (a student that didn't terminate its turn, or emitted an
      empty/invalid completion, can't meaningfully continue — and its bad turn is skipped, not
      distilled, by the caller's no-loss sample handling).
    - ``on_turn_generated`` (optional) fires AFTER each turn's generation to refresh the worker stall
      clock and advance the sample counter — a many-turn episode is a long serial stretch of GPU
      generates + teacher-less env steps that would otherwise emit no progress ping.

    ``prefix_ids`` and ``context_messages`` are snapshotted BEFORE generation, so each record carries
    the exact student-token and message context the teacher must condition on for that turn.
    """
    state = active_env.new_rollout_state(example)
    initial_messages = state.get("prompt") or state.get("messages")
    if not isinstance(initial_messages, list):
        raise KeyError("multi-turn rollout state must include prompt or messages")
    messages = [dict(m) for m in initial_messages]
    prompt_ids = render(messages, True)
    cur_ids = list(prompt_ids)  # invariant: cur_ids == prompt_ids + completion tokens so far
    token_budget = (engine_max_len - len(prompt_ids) - 8) if engine_max_len else None
    records: list[TurnRecord] = []

    turns = 0
    while True:
        completion_so_far = len(cur_ids) - len(prompt_ids)
        max_new = per_turn_max_tokens
        if token_budget is not None:
            remaining = token_budget - completion_so_far
            if remaining <= 0:
                break
            max_new = min(max_new, remaining)
        # Snapshot the student-token prefix and message context BEFORE generation: this is what the
        # teacher must condition on to score THIS turn (the transcript up to, but excluding, the turn).
        prefix_ids = list(cur_ids)
        context_messages = [dict(m) for m in messages]
        gen = generate(prefix_ids, max(1, max_new))
        if on_turn_generated is not None:
            on_turn_generated()
        records.append({"prefix_ids": prefix_ids, "gen": gen, "context_messages": context_messages})
        text = getattr(gen, "completion_text", "") or ""
        active_env.record_model_turn(state, text)
        messages.append({"role": "assistant", "content": text})
        turns += 1
        # A turn that didn't terminate naturally (truncated) or produced no usable text (skip) ends the
        # episode: it's recorded (counted) but not distilled, and continuing from a broken turn is
        # pointless (the student can't end its turn / said nothing).
        if getattr(gen, "truncated", False) or getattr(gen, "skip", False):
            break
        asst_ids = getattr(gen, "completion_ids", None) or []
        cur_ids.extend(asst_ids)
        completion_so_far = len(cur_ids) - len(prompt_ids)
        if token_budget is not None and completion_so_far >= token_budget:
            break
        if turns >= max_turns or active_env.rollout_done(state, max_turns):
            break
        env_msgs = active_env.env_reply(messages, state)
        if not env_msgs:
            break
        messages.extend(env_msgs)
        # Don't append glue if the env step finished — no next model turn.
        if active_env.rollout_done(state, max_turns):
            break
        glue = _dedup_seam_terminator(asst_ids, env_glue(env_msgs))
        if token_budget is not None and completion_so_far + len(glue) > token_budget:
            break
        cur_ids.extend(glue)

    return records


class _RolloutState:
    """Mutable per-rollout accumulator for :func:`rollout_async` (mirrors :func:`rollout_one` locals)."""

    __slots__ = (
        "budget",
        "completion_ids",
        "cur_ids",
        "done",
        "env_mask",
        "example",
        "logprobs",
        "messages",
        "prompt_ids",
        "state",
        "turns",
    )

    def __init__(self, example, messages, prompt_ids, state, budget):
        self.example = example
        self.messages = messages
        self.prompt_ids = prompt_ids
        self.cur_ids = list(prompt_ids)  # invariant: cur_ids == prompt_ids + completion_ids so far
        self.completion_ids: list[int] = []
        self.logprobs: list[float] = []
        self.env_mask: list[int] = []
        self.state = state
        self.turns = 0
        self.budget = budget
        self.done = False

    def result(self, reward: float) -> RolloutResult:
        return {
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "logprobs": self.logprobs,
            "env_mask": self.env_mask,
            "reward": float(reward),
        }


def _advance_after_turn(
    r: _RolloutState,
    asst_ids: list[int],
    asst_lp: list[float],
    text: str,
    *,
    active_env,
    env_glue: Callable[[list], list[int]],
    max_turns: int,
) -> None:
    """Fold one assistant turn into ``r`` and run its env step. Sets ``r.done`` when finished."""
    r.completion_ids.extend(asst_ids)
    r.logprobs.extend(asst_lp)
    r.env_mask.extend([1] * len(asst_ids))
    r.cur_ids.extend(asst_ids)
    active_env.record_model_turn(r.state, text)
    r.messages.append({"role": "assistant", "content": text})
    r.turns += 1
    if r.budget is not None and len(r.completion_ids) >= r.budget:
        r.done = True
        return
    if r.turns >= max_turns or active_env.rollout_done(r.state, max_turns):
        r.done = True
        return
    env_msgs = active_env.env_reply(r.messages, r.state)
    if not env_msgs:
        r.done = True
        return
    r.messages.extend(env_msgs)
    if active_env.rollout_done(r.state, max_turns):
        r.done = True
        return
    glue = _dedup_seam_terminator(r.completion_ids, env_glue(env_msgs))
    if r.budget is not None and len(r.completion_ids) + len(glue) > r.budget:
        r.done = True
        return
    r.completion_ids.extend(glue)
    r.logprobs.extend([0.0] * len(glue))
    r.env_mask.extend([0] * len(glue))
    r.cur_ids.extend(glue)


def _build_rollout_states(
    examples: list[dict],
    active_env,
    render: Callable[[list, bool], list[int]],
    engine_max_len: int | None,
) -> list[_RolloutState]:
    """Initialise one :class:`_RolloutState` per example for :func:`rollout_async`."""
    rollouts: list[_RolloutState] = []
    for example in examples:
        state = active_env.new_rollout_state(example)
        initial_messages = state.get("prompt") or state.get("messages")
        if not isinstance(initial_messages, list):
            raise KeyError("multi-turn rollout state must include prompt or messages")
        messages = [dict(m) for m in initial_messages]
        prompt_ids = render(messages, True)
        budget = (engine_max_len - len(prompt_ids) - 8) if engine_max_len else None
        rollouts.append(_RolloutState(example, messages, prompt_ids, state, budget))
    return rollouts


def _turn_budget(r: _RolloutState, per_turn_max_tokens: int) -> int | None:
    """Max new tokens for ``r``'s next turn (capped to engine headroom). Returns None and sets r.done when exhausted."""
    max_new = per_turn_max_tokens
    if r.budget is not None:
        remaining = r.budget - len(r.completion_ids)
        if remaining <= 0:
            r.done = True
            return None
        max_new = min(max_new, remaining)
    return max(1, max_new)


def _score_rollouts(active_env, rollouts: list[_RolloutState]) -> list[float]:
    """Reward each rollout in input order, using reward_many, concurrent, or serial scoring."""
    reward_many = getattr(active_env, "reward_many", None)
    if callable(reward_many):
        rewards = reward_many([(r.example, r.state) for r in rollouts])
        if len(rewards) != len(rollouts):
            raise RuntimeError("env.reward_many returned the wrong number of rewards")
        return [float(x) for x in rewards]

    def _score(r: _RolloutState) -> float:
        return float(active_env.reward("", r.example, r.state))

    if len(rollouts) <= 1 or not getattr(active_env, "reward_thread_safe", True):
        return [_score(r) for r in rollouts]
    pool = ThreadPoolExecutor(max_workers=min(16, len(rollouts)))
    try:
        futures = {pool.submit(_score, r): i for i, r in enumerate(rollouts)}
        scores: list[float] = [0.0] * len(rollouts)
        for fut in as_completed(futures):
            scores[futures[fut]] = fut.result()  # re-raises the first failed scorer
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return scores


_PHYSICAL_REQUEST_COUNTER = itertools.count()
_PHYSICAL_REQUEST_COUNTER_LOCK = threading.Lock()
_ROLLOUT_REQUEST_TIMEOUT_MIN_SECONDS = 600.0
_ROLLOUT_REQUEST_TIMEOUT_MAX_SECONDS = 3600.0


def _next_physical_request_id() -> str:
    """Return a process-unique physical request id across rollout invocations."""
    with _PHYSICAL_REQUEST_COUNTER_LOCK:
        return f"flash-mt-{next(_PHYSICAL_REQUEST_COUNTER)}"


def resolve_rollout_request_timeout_seconds(engine_max_len: int) -> float:
    """Resolve the platform-managed absolute timeout for one physical rollout request."""
    return min(
        _ROLLOUT_REQUEST_TIMEOUT_MAX_SECONDS,
        max(_ROLLOUT_REQUEST_TIMEOUT_MIN_SECONDS, 0.5 * float(engine_max_len)),
    )


@dataclass
class _LogicalRequest:
    rollout: _RolloutState
    prefix_ids: tuple[int, ...]
    max_tokens: int
    initial: bool
    attempts: int = 0
    started_at: float = 0.0


def rollout_async(
    *,
    examples: list[dict],
    active_env,
    render: Callable[[list, bool], list[int]],
    submit: Callable[[str, list[int], int, bool], None],
    poll: Callable[[], list[RolloutCompletion]],
    busy: Callable[[], bool],
    abort: Callable[[list[str]], None],
    env_glue: Callable[[list], list[int]],
    max_turns: int,
    per_turn_max_tokens: int,
    engine_max_len: int | None = None,
    request_timeout_seconds: float | None = None,
    request_max_attempts: int = 2,
    monotonic: Callable[[], float] = time.monotonic,
    request_id_factory: Callable[[], str] = _next_physical_request_id,
) -> list[RolloutResult]:
    """Run continuously batched multi-turn rollouts with per-physical-request deadlines.

    A logical assistant turn snapshots its accepted prefix and sampling limits. A timed-out physical
    attempt is synchronously aborted before an identical retry gets a fresh process-unique id. Only a
    successful completed result reaches the environment worker, so retries never replay opaque env
    calls or mutate transcript state. There is deliberately no episode wall-clock deadline. Timeout
    enforcement is cooperative between engine polls and cannot interrupt one blocking engine step.
    """
    if request_max_attempts < 1:
        raise ValueError("request_max_attempts must be at least 1")
    if request_timeout_seconds is not None and request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive when set")

    rollouts = _build_rollout_states(examples, active_env, render, engine_max_len)
    by_id: dict[str, _LogicalRequest] = {}
    to_env: queue.Queue = queue.Queue()
    to_submit: queue.Queue = queue.Queue()

    def start_attempt(logical: _LogicalRequest) -> None:
        req_id = request_id_factory()
        logical.attempts += 1
        logical.started_at = monotonic()
        by_id[req_id] = logical
        try:
            submit(req_id, list(logical.prefix_ids), logical.max_tokens, logical.initial)
        except Exception:
            by_id.pop(req_id, None)
            with contextlib.suppress(Exception):
                abort([req_id])
            raise

    def start_logical(r: _RolloutState, prefix: list[int], max_new: int, initial: bool) -> None:
        start_attempt(
            _LogicalRequest(
                rollout=r,
                prefix_ids=tuple(prefix),
                max_tokens=int(max_new),
                initial=bool(initial),
            )
        )

    def expire_requests(now: float) -> None:
        if request_timeout_seconds is None:
            return
        for req_id, logical in list(by_id.items()):
            if now - logical.started_at < request_timeout_seconds:
                continue
            by_id.pop(req_id)
            abort([req_id])
            if logical.attempts >= request_max_attempts:
                raise RolloutRequestExhaustedError(
                    attempts=logical.attempts,
                    reason="absolute request timeout",
                )
            start_attempt(logical)

    def env_worker() -> None:
        while True:
            item = to_env.get()
            if item is None:
                return
            r, asst_ids, asst_lp, text = item
            try:
                _advance_after_turn(
                    r,
                    asst_ids,
                    asst_lp,
                    text,
                    active_env=active_env,
                    env_glue=env_glue,
                    max_turns=max_turns,
                )
                max_new = None if r.done else _turn_budget(r, per_turn_max_tokens)
            except Exception as exc:  # propagate to the main thread (engine owner)
                to_submit.put(("error", exc))
                return
            to_submit.put(("done", r) if max_new is None else ("next", r, list(r.cur_ids), max_new))

    worker = threading.Thread(target=env_worker, daemon=True)
    worker.start()
    n = len(rollouts)
    completed = 0

    def take(msg) -> None:
        nonlocal completed
        if msg[0] == "error":
            err = msg[1]
            raise err.with_traceback(err.__traceback__)
        if msg[0] == "done":
            completed += 1
        else:
            _, r, prefix, max_new = msg
            start_logical(r, prefix, max_new, False)

    try:
        for r in rollouts:
            max_new = _turn_budget(r, per_turn_max_tokens)
            if max_new is None:
                completed += 1
            else:
                start_logical(r, list(r.cur_ids), max_new, r.turns == 0)
        while completed < n:
            progressed = False
            while True:
                try:
                    take(to_submit.get_nowait())
                    progressed = True
                except queue.Empty:
                    break
            if completed >= n:
                break
            if busy():
                finished = poll()
                expire_requests(monotonic())
                for req_id, asst_ids, asst_lp, text in finished:
                    logical = by_id.pop(req_id, None)
                    if logical is None:
                        continue
                    to_env.put((logical.rollout, asst_ids, asst_lp, text))
                    progressed = True
            else:
                expire_requests(monotonic())
                if not progressed:
                    # the environment worker is mid-advance; block briefly instead of spinning.
                    with contextlib.suppress(queue.Empty):
                        take(to_submit.get(timeout=0.1))
    finally:
        active = list(by_id)
        by_id.clear()
        if active:
            with contextlib.suppress(Exception):
                abort(active)
        to_env.put(None)
        worker.join()

    rewards = _score_rollouts(active_env, rollouts)
    return [r.result(rw) for r, rw in zip(rollouts, rewards, strict=True)]


def render_message_ids(tok, messages, add_generation_prompt: bool, *, thinking: bool) -> list[int]:
    """Render ``messages`` with the chat template and tokenize to a flat ``list[int]``."""
    text = tok.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        enable_thinking=thinking,
    )
    return [int(t) for t in tok(text, add_special_tokens=False).input_ids]


def make_env_glue(tok, *, thinking: bool, cache_size: int = 8192) -> Callable[[list], list[int]]:
    """Build the inter-turn "glue" tokenizer used by both the GRPO rollout (:func:`build_rollout_func`)
    and OPD's per-turn rollout (:func:`rollout_one_records`).

    Given the env's reply messages for a turn, returns the token ids that sit BETWEEN the assistant
    turn and the next assistant generation prompt (the turn terminator + the env/observation messages +
    the next assistant header). Computed with the probe trick — render ``[{assistant: PROBE}, *env]``
    with ``add_generation_prompt=True`` and take everything after the probe — so the history is never
    re-rendered (Qwen3's template doesn't round-trip a re-rendered transcript). Results are LRU-cached
    by the env messages. Raises ``ValueError`` if the model's chat template doesn't insert assistant
    content verbatim (token-aligned multi-turn is then unsupported for that model)."""
    cache = _LRUCache(cache_size)
    probe = "flash-env-glue-probe"

    def env_glue(env_messages: list) -> list[int]:
        cache_key = json.dumps(env_messages, sort_keys=True, default=str)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        text = tok.apply_chat_template(
            [{"role": "assistant", "content": probe}, *env_messages],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=thinking,
        )
        first = text.find(probe)
        if first == -1 or text.find(probe, first + len(probe)) != -1:
            raise ValueError(
                "multi-turn env_glue could not uniquely locate its probe in the rendered chat "
                "template; this model's template does not insert assistant content verbatim, so "
                "token-aligned multi-turn rollout is unsupported for it (use a single-turn/tool "
                "env or a different model)."
            )
        glue_text = text[first + len(probe) :]
        glue = [int(t) for t in tok(glue_text, add_special_tokens=False).input_ids]
        cache.put(cache_key, glue)
        return glue

    return env_glue


def _engine_vocab_size(engine) -> int | None:
    """Best-effort vocab size from the colocate vLLM engine, or None. Never raises."""
    try:
        mc = engine.llm_engine.model_config
    except Exception:
        return None
    for attr in ("get_vocab_size", "get_hf_config_vocab_size"):
        getter = getattr(mc, attr, None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                pass
    try:
        return int(mc.hf_text_config.vocab_size)
    except Exception:
        return None


def build_rollout_func(
    *,
    active_env,
    tok,
    examples_by_key: dict,
    max_completion: int,
    max_turns: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
    thinking: bool,
    engine_max_len: int | None = None,
    structured_outputs: dict | None = None,
    request_timeout_seconds: float | None = None,
    request_max_attempts: int = 2,
    monotonic: Callable[[], float] = time.monotonic,
    request_id_factory: Callable[[], str] = _next_physical_request_id,
):
    """Return a TRL ``rollout_func`` closure that drives ``active_env`` on the colocate engine."""
    from vllm import SamplingParams  # gpu-only; lazy import so the module loads on CPU

    try:
        from vllm.sampling_params import RequestOutputKind

        _output_kind = RequestOutputKind.FINAL_ONLY
    except Exception:
        _output_kind = None

    # [train] structured_outputs: resolve the params class once; constructed per request below
    # (vLLM's processor stamps a per-request backend on the instance, so sharing one is unsafe).
    # No silent fallback — a configured constraint that can't be applied must fail the run, not
    # train on unconstrained text the reward believes is schema-bound.
    _SOParams = None
    if structured_outputs:
        from vllm.sampling_params import StructuredOutputsParams

        _SOParams = StructuredOutputsParams

    _render_cache = _LRUCache(8192)

    def render(messages: list, add_generation_prompt: bool) -> list[int]:
        cache_key = (
            f"{add_generation_prompt}\x00{json.dumps(messages, sort_keys=True, default=str)}"
        )
        cached = _render_cache.get(cache_key)
        if cached is not None:
            return cached
        ids = render_message_ids(tok, messages, add_generation_prompt, thinking=thinking)
        _render_cache.put(cache_key, ids)
        return ids

    env_glue = make_env_glue(tok, thinking=thinking)

    def rollout_func(prompts, trainer):
        engine = trainer.vllm_generation.llm
        llm_engine = engine.llm_engine
        # TRL's rollout_func path only wakes tags=["weights"]; wake kv_cache too or step 0 faults (flash #162).
        sleep_mode = bool(getattr(getattr(trainer, "args", None), "vllm_enable_sleep_mode", False))
        vocab_size = _engine_vocab_size(engine)
        active_ids: set[str] = set()

        def submit(req_id: str, prefix_ids: list[int], max_tokens: int, initial: bool) -> None:
            """Enqueue one assistant-turn request."""
            if not prefix_ids:
                raise ValueError(
                    "multi-turn rollout produced an empty prompt for engine.add_request()"
                )
            if initial:
                lo, hi = min(prefix_ids), max(prefix_ids)
                if lo < 0 or (vocab_size is not None and hi >= vocab_size):
                    raise ValueError(
                        f"multi-turn rollout prompt has out-of-range token id(s) [{lo}, {hi}] for "
                        f"vocab size {vocab_size} (tokenizer/model mismatch)"
                    )
            sp_kwargs = {
                "max_tokens": max(1, int(max_tokens)),
                "temperature": temperature,
                "top_p": top_p,
                "logprobs": 1,  # include the sampled token's logprob at each position
                "stop": list(stop) if stop else None,
            }
            if _output_kind is not None:
                sp_kwargs["output_kind"] = _output_kind
            if _SOParams is not None:
                # Every assistant turn is constrained — mid-rollout turns included — since the
                # env contract has no per-turn schema channel; unconstrained env/tool turns are
                # not generated here.
                sp_kwargs["structured_outputs"] = _SOParams(**copy.deepcopy(structured_outputs))
            llm_engine.add_request(
                req_id, {"prompt_token_ids": list(prefix_ids)}, SamplingParams(**sp_kwargs)
            )
            active_ids.add(req_id)

        def poll() -> list[RolloutCompletion]:
            """Step the engine and return explicit completed-result tuples."""
            finished: list[RolloutCompletion] = []
            for out in llm_engine.step():
                if not getattr(out, "finished", False):
                    continue
                comp = out.outputs[0]
                token_ids = list(comp.token_ids)
                lps: list[float] = []
                for pos, tid in enumerate(token_ids):
                    entry = comp.logprobs[pos] if comp.logprobs else None
                    lp = entry.get(tid) if entry else None
                    lps.append(float(getattr(lp, "logprob", 0.0)) if lp is not None else 0.0)
                req_id = str(out.request_id)
                active_ids.discard(req_id)
                finished.append((req_id, token_ids, lps, str(comp.text)))
            return finished

        def abort(ids: list[str]) -> None:
            if not ids:
                return
            llm_engine.abort_request(list(ids))
            active_ids.difference_update(ids)

        def busy() -> bool:
            return bool(llm_engine.has_unfinished_requests())

        woke = False
        try:
            if sleep_mode:
                engine.wake_up(tags=["kv_cache"])
                woke = True
            examples = [examples_by_key.get(_prompt_key(p), {"prompt": p}) for p in prompts]
            rollouts = rollout_async(
                examples=examples,
                active_env=active_env,
                render=render,
                submit=submit,
                poll=poll,
                busy=busy,
                env_glue=env_glue,
                max_turns=max_turns,
                per_turn_max_tokens=max_completion,
                engine_max_len=engine_max_len,
                abort=abort,
                request_timeout_seconds=request_timeout_seconds,
                request_max_attempts=request_max_attempts,
                monotonic=monotonic,
                request_id_factory=request_id_factory,
            )
            out: dict[str, list] = {k: [] for k in _ROLLOUT_FIELDS}
            for r in rollouts:
                for k in out:
                    out[k].append(r[k])
            return out
        finally:
            # Abort in-flight requests on error so they don't corrupt the next GRPO step.
            if active_ids:
                with contextlib.suppress(Exception):
                    llm_engine.abort_request(list(active_ids))
            if woke:
                engine.sleep(level=2)

    return rollout_func

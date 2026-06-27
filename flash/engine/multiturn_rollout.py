"""Multi-turn / tool GRPO rollout for TRL's experimental ``rollout_func`` (colocate vLLM).

Drives a Freesolo ``EnvironmentMultiTurn`` turn loop, returning the full interleaved token
sequence with an ``env_mask`` (1=model, 0=env/tool) for multi-turn credit assignment.
:func:`rollout_one` is unit-testable on CPU; :func:`build_rollout_func` wires the real engine.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict


class RolloutResult(TypedDict):
    """Token-aligned fields returned per rollout."""

    prompt_ids: list[int]
    completion_ids: list[int]
    logprobs: list[float]
    env_mask: list[int]
    reward: float


_ROLLOUT_FIELDS: tuple[str, ...] = (
    "prompt_ids",
    "completion_ids",
    "logprobs",
    "env_mask",
    "reward",
)


def _prompt_key(prompt) -> str:
    """Stable string key for a dataset ``prompt`` value."""
    try:
        return json.dumps(prompt, sort_keys=True, default=str)
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

        glue = env_glue(env_msgs)
        # Collapse a duplicate turn terminator at the seam: env_glue leads with the assistant turn's
        # terminator (e.g. <|im_end|>), which vLLM also keeps as the last token of a naturally-stopped
        # turn — keep the assistant's own (env_mask=1, real logprob) over the env copy. See
        # _advance_after_turn (the rollout_async twin) for the same fix.
        if glue and completion_ids and completion_ids[-1] == glue[0]:
            glue = glue[1:]
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
    glue = env_glue(env_msgs)
    # env_glue's text starts at the assistant turn's terminator (e.g. <|im_end|>); vLLM also keeps that
    # terminator as the final token of a naturally-stopped turn (it's in asst_ids). Collapse the duplicate
    # at the seam so the stream has exactly one terminator per turn (matches the chat template + SFT
    # transcripts), keeping the assistant's own token (env_mask=1, real logprob) over the env copy.
    if glue and r.completion_ids and r.completion_ids[-1] == glue[0]:
        glue = glue[1:]
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


def rollout_async(
    *,
    examples: list[dict],
    active_env,
    render: Callable[[list, bool], list[int]],
    submit: Callable[[str, list[int], int, bool], None],
    poll: Callable[[], list[tuple[str, list[int], list[float], str]]],
    busy: Callable[[], bool],
    env_glue: Callable[[list], list[int]],
    max_turns: int,
    per_turn_max_tokens: int,
    engine_max_len: int | None = None,
) -> list[RolloutResult]:
    """Run ``len(examples)`` multi-turn rollouts with continuous-batched generation.

    Rollouts are not turn-synchronized: each turn is an independent engine request submitted as
    soon as the previous one finishes. Main thread owns the engine; a worker thread owns the env.
    Results are byte-identical to one :func:`rollout_one` per example, in input order.
    """
    rollouts = _build_rollout_states(examples, active_env, render, engine_max_len)
    by_id: dict[str, _RolloutState] = {}
    counter = 0
    to_env: queue.Queue = queue.Queue()
    to_submit: queue.Queue = queue.Queue()

    def do_submit(r: _RolloutState, prefix: list[int], max_new: int, initial: bool) -> None:
        nonlocal counter
        req_id = f"r{counter}"
        counter += 1
        by_id[req_id] = r
        submit(req_id, prefix, max_new, initial)

    def env_worker() -> None:
        while True:
            item = to_env.get()
            if item is None:
                return
            r, asst_ids, asst_lp, text = item
            try:
                _advance_after_turn(
                    r, asst_ids, asst_lp, text,
                    active_env=active_env, env_glue=env_glue, max_turns=max_turns,
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
            do_submit(r, prefix, max_new, False)

    try:
        for r in rollouts:
            max_new = _turn_budget(r, per_turn_max_tokens)
            if max_new is None:
                completed += 1
            else:
                do_submit(r, list(r.cur_ids), max_new, r.turns == 0)
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
                for req_id, asst_ids, asst_lp, text in poll():
                    to_env.put((by_id.pop(req_id), asst_ids, asst_lp, text))
            elif not progressed:
                # Worker is mid-advance; block briefly instead of spinning.
                with contextlib.suppress(queue.Empty):
                    take(to_submit.get(timeout=0.1))
    finally:
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
):
    """Return a TRL ``rollout_func`` closure that drives ``active_env`` on the colocate engine."""
    from vllm import SamplingParams  # gpu-only; lazy import so the module loads on CPU

    try:
        from vllm.sampling_params import RequestOutputKind

        _final_only_kind = RequestOutputKind.FINAL_ONLY
    except Exception:
        _final_only_kind = None

    _render_cache = _LRUCache(8192)

    def render(messages: list, add_generation_prompt: bool) -> list[int]:
        cache_key = f"{add_generation_prompt}\x00{json.dumps(messages, sort_keys=True, default=str)}"
        cached = _render_cache.get(cache_key)
        if cached is not None:
            return cached
        ids = render_message_ids(tok, messages, add_generation_prompt, thinking=thinking)
        _render_cache.put(cache_key, ids)
        return ids

    _glue_cache = _LRUCache(8192)

    def env_glue(env_messages: list) -> list[int]:
        cache_key = json.dumps(env_messages, sort_keys=True, default=str)
        cached = _glue_cache.get(cache_key)
        if cached is not None:
            return cached
        # Render a probe assistant turn + env messages, then take everything after the probe
        # to get the inter-turn glue without re-rendering history (Qwen3 doesn't round-trip).
        probe = "flash-env-glue-probe"
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
        _glue_cache.put(cache_key, glue)
        return glue

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
                raise ValueError("multi-turn rollout produced an empty prompt for engine.add_request()")
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
            if _final_only_kind is not None:
                sp_kwargs["output_kind"] = _final_only_kind
            llm_engine.add_request(
                req_id, {"prompt_token_ids": list(prefix_ids)}, SamplingParams(**sp_kwargs)
            )
            active_ids.add(req_id)

        def poll() -> list[tuple[str, list[int], list[float], str]]:
            """Step the engine; return finished (req_id, token_ids, logprobs, text) tuples."""
            finished: list[tuple[str, list[int], list[float], str]] = []
            for out in llm_engine.step():
                if not getattr(out, "finished", False):
                    continue
                comp = out.outputs[0]
                token_ids = list(comp.token_ids)
                lps: list[float] = []
                for pos, tid in enumerate(token_ids):
                    entry = (comp.logprobs or [])[pos] if comp.logprobs else None
                    lp = entry.get(tid) if entry else None
                    lps.append(float(getattr(lp, "logprob", 0.0)) if lp is not None else 0.0)
                active_ids.discard(out.request_id)
                finished.append((out.request_id, token_ids, lps, comp.text))
            return finished

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

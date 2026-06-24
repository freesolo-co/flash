"""Multi-turn / tool GRPO rollout for TRL's experimental ``rollout_func`` (colocate vLLM).

TRL's ``GRPOTrainer`` generates a single assistant turn per prompt, which cannot drive a
Freesolo ``EnvironmentMultiTurn`` turn loop (model turn -> env reply -> ...). This
module supplies a ``rollout_func`` that:

  * drives the env's turn loop via the adapter helpers (``new_rollout_state`` /
    ``record_model_turn`` / ``env_reply`` / ``rollout_done``), so the *env* owns tool
    execution, ``StatefulToolEnv`` state threading, and any simulated-user turns;
  * returns the FULL interleaved token sequence as ``completion_ids`` together with an
    ``env_mask`` that marks model-generated tokens (``1``, trained) vs tool/env tokens
    (``0``, masked out of the loss). ``env_mask`` is TRL's documented mechanism for
    multi-turn credit assignment (it is treated internally as the tool mask), so only the
    policy's own tokens get advantage while the env tokens still provide context for the
    forward pass;
  * scores each rollout with the environment reward (``reward_from_messages``) and returns
    it as an extra field consumed by a pass-through ``reward_func``.

Token alignment assumes a **prefix-preserving** chat template: appending a message must not
retokenize earlier turns (the same assumption TRL's native tool loop documents; auto-patched
for Qwen3 / DeepSeek-V3). The env segment between two model turns is taken as the suffix of a
full re-render; if the prefix invariant is violated the rollout raises (fails loudly) rather
than mis-masking model vs env tokens and silently mistraining.

The core (:func:`rollout_one`) is pure Python and takes injected ``render``/``generate``
callables so it can be unit-tested without a GPU/tokenizer; :func:`build_rollout_func` wires
the real tokenizer + the colocate vLLM engine into it at runtime.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
from collections.abc import Callable
from typing import TypedDict


class RolloutResult(TypedDict):
    """Token-aligned fields returned per rollout for TRL's ``rollout_func``."""

    prompt_ids: list[int]
    completion_ids: list[int]
    logprobs: list[float]
    env_mask: list[int]
    reward: float


# Field names shared between a single RolloutResult and the batched dict-of-lists that
# build_rollout_func returns. Kept as a plain tuple (not RolloutResult.__annotations__) so
# the batch accumulator's key source isn't a single-rollout type whose value types (float,
# list[int], ...) deliberately differ from the accumulator's list-of-those.
_ROLLOUT_FIELDS: tuple[str, ...] = (
    "prompt_ids",
    "completion_ids",
    "logprobs",
    "env_mask",
    "reward",
)


def _prompt_key(prompt) -> str:
    """Stable key for mapping a dataset ``prompt`` value back to its example row."""
    try:
        return json.dumps(prompt, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(prompt)


def build_examples_index(rows: list[dict], prompt_of: Callable[[dict], object]) -> dict:
    """Map each row's rendered ``prompt`` value to the example row (for reward/answer lookup).

    Collisions (two rows producing the same prompt) keep the last row and are reported by the
    caller via :func:`index_collisions`; duplicates are rare in training data and only affect
    which ``answer``/``info`` a shared prompt scores against.
    """
    return {_prompt_key(prompt_of(r)): r for r in rows}


def index_collisions(rows: list[dict], prompt_of: Callable[[dict], object]) -> int:
    """Number of rows dropped by prompt-key collisions in :func:`build_examples_index`."""
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
    """Run one multi-turn/tool rollout and return TRL ``rollout_func`` fields for it.

    Args:
        example: the dataset row carried into environment scoring.
        active_env: the Freesolo environment adapter (drives the turn loop + scoring).
        render: ``render(messages, add_generation_prompt) -> token_ids`` (chat template) — used
            only for the INITIAL prompt.
        generate: ``generate(prefix_token_ids, max_tokens) -> (token_ids, token_logprobs,
            text)`` for one sampled assistant turn (model tokens + sampling logprobs + text);
            ``max_tokens`` bounds that turn so it can't overflow the engine context.
        env_glue: ``env_glue(env_messages) -> token_ids`` — the tokens that CLOSE the
            just-finished assistant turn, render the env reply message(s), and OPEN the next
            generation prompt. The running token sequence is built incrementally from these
            (the model's generated ids + env glue), never by re-rendering the whole
            conversation — so a chat template that does not round-trip prior turns (e.g. Qwen3's
            empty ``<think>`` block, which is injected into the generation prompt but stripped
            from history) stays token-aligned instead of failing the old prefix check.
        max_turns: hard cap on model turns (defense against a non-terminating env).

    Returns a dict with ``prompt_ids``, ``completion_ids``, ``logprobs``, ``env_mask`` (all
    token-aligned) and the scalar ``reward`` for this rollout.
    """
    state = active_env.new_rollout_state(example)
    initial_messages = state.get("prompt") or state.get("messages")
    if not isinstance(initial_messages, list):
        raise KeyError("multi-turn rollout state must include prompt or messages")
    messages = [dict(m) for m in initial_messages]
    prompt_ids = render(messages, True)
    cur_ids = list(prompt_ids)  # invariant: cur_ids == prompt_ids + completion_ids so far
    # Per-rollout completion cap so prompt + accumulated completion never exceeds the colocate
    # engine's context (which would overflow the next generate()); leave a small margin.
    token_budget = (engine_max_len - len(prompt_ids) - 8) if engine_max_len else None
    completion_ids: list[int] = []
    logprobs: list[float] = []
    env_mask: list[int] = []

    turns = 0
    while True:
        # Bound THIS turn's generation by the remaining engine headroom so even a single
        # generate() can't push prompt+completion past the context (the cap below stops the
        # loop AFTER a turn; this stops the turn itself from overflowing).
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
        # If the env step finished the episode (it can set done / hit its budget while replying),
        # stop here: do NOT append the next-generation glue — there is no next model turn, and the
        # glue would leave a trailing assistant prompt in completion_ids (and could trigger one
        # more generate()).
        if active_env.rollout_done(state, max_turns):
            break

        # Env-segment tokens = close the just-finished assistant turn + render the env reply +
        # open the next generation prompt, computed INCREMENTALLY (env_glue) rather than by
        # re-rendering the whole conversation. Masked (0) — they are not the policy's tokens —
        # but kept in completion_ids so the next turn conditions on them. Building the sequence
        # by id-concatenation (model ids + glue) keeps it token-aligned even for templates that
        # don't round-trip history (Qwen3's empty <think> block), which the old re-render +
        # prefix-check could not handle.
        glue = env_glue(env_msgs)
        # Don't append glue that would push prompt+completion past the engine budget (the next
        # generate() would be skipped anyway); end the rollout cleanly instead of returning an
        # over-length sequence that could break the trainer's forward/loss pass.
        if token_budget is not None and len(completion_ids) + len(glue) > token_budget:
            break
        completion_ids.extend(glue)
        logprobs.extend([0.0] * len(glue))
        env_mask.extend([0] * len(glue))
        cur_ids.extend(glue)

    # Score with the ACTUAL rollout state (not a fresh one) so reward funcs see the tool/env
    # state the rollout accumulated. state["completion"] holds the full transcript.
    reward = active_env.reward("", example, state)
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask,
        "reward": float(reward),
    }


class _RolloutState:
    """Mutable per-rollout accumulator for the turn-synchronized batched loop (:func:`rollout_batch`).

    Holds exactly the running fields :func:`rollout_one` keeps in locals, so the two paths produce
    byte-identical token alignment / env_mask / reward — the only difference is that batched
    generation advances every still-active rollout's assistant turn in ONE vLLM call.
    """

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
        self.budget = budget  # max completion tokens (engine headroom), or None
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
    """Fold one freshly-sampled assistant turn into rollout ``r`` and run its env step, mirroring the
    body of :func:`rollout_one`'s loop EXACTLY. Sets ``r.done`` when the rollout should stop. Used by
    :func:`rollout_batch` so the batched and single-rollout paths can never drift."""
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
    """Initialise one :class:`_RolloutState` per example (initial prompt rendered, engine budget
    computed). Shared by the synchronized (:func:`rollout_batch`) and async
    (:func:`rollout_async`) paths so both start from byte-identical state."""
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
    """Max new tokens for ``r``'s next assistant turn, bounded by the remaining engine headroom so
    prompt+completion can't overflow the context. Returns ``None`` and marks ``r.done`` when the
    headroom is already exhausted. Identical cap for both rollout paths (no drift)."""
    max_new = per_turn_max_tokens
    if r.budget is not None:
        remaining = r.budget - len(r.completion_ids)
        if remaining <= 0:  # prompt already fills the context -> this rollout is done
            r.done = True
            return None
        max_new = min(max_new, remaining)
    return max(1, max_new)


def _score_rollouts(active_env, rollouts: list[_RolloutState]) -> list[float]:
    """Reward for each rollout, in order. Uses ``active_env.reward_many`` when the env provides it
    (one batched, env-concurrent scoring call per task instead of a blocking call per rollout — the
    win for judge/expensive-reward envs), else one sequential ``active_env.reward()`` per rollout.
    Both yield identical values; reward_many only changes scoring concurrency."""
    reward_many = getattr(active_env, "reward_many", None)
    if callable(reward_many):
        rewards = reward_many([(r.example, r.state) for r in rollouts])
        if len(rewards) != len(rollouts):
            raise RuntimeError("env.reward_many returned the wrong number of rewards")
        return [float(x) for x in rewards]
    return [float(active_env.reward("", r.example, r.state)) for r in rollouts]


def rollout_batch(
    *,
    examples: list[dict],
    active_env,
    render: Callable[[list, bool], list[int]],
    batched_generate: Callable[
        [list[list[int]], list[int]], list[tuple[list[int], list[float], str]]
    ],
    env_glue: Callable[[list], list[int]],
    max_turns: int,
    per_turn_max_tokens: int,
    engine_max_len: int | None = None,
) -> list[RolloutResult]:
    """Run ``len(examples)`` multi-turn rollouts with TURN-SYNCHRONIZED batched generation.

    Semantically identical to calling :func:`rollout_one` once per example (same token alignment,
    env_mask, per-rollout reward, and input ordering), but every still-active rollout's assistant
    turn is sampled in a SINGLE ``batched_generate`` call instead of one ``engine.generate`` per
    prompt per turn. vLLM is built to decode many sequences at once, so this is the dominant
    multi-turn speedup; combined with vLLM prefix caching (on for the colocate engine) each
    rollout's growing prefix and the shared in-group prompt are computed once and reused across
    turns. Rollouts that finish (budget / max_turns / env done) drop out of the batch; the rest
    keep generating until all are done.

    ``batched_generate(prefixes, max_tokens_list)`` returns ``(token_ids, logprobs, text)`` per
    prefix, in input order. It is injected so the loop is unit-testable on CPU.
    """
    rollouts = _build_rollout_states(examples, active_env, render, engine_max_len)

    # Advance all rollouts in lockstep: each iteration samples one assistant turn for every
    # still-active rollout in a single batched call, then runs each rollout's env step.
    while True:
        batch: list[_RolloutState] = []
        max_news: list[int] = []
        for r in rollouts:
            if r.done:
                continue
            max_new = _turn_budget(r, per_turn_max_tokens)
            if max_new is None:  # engine headroom exhausted -> _turn_budget marked r.done
                continue
            batch.append(r)
            max_news.append(max_new)
        if not batch:
            break
        gen = batched_generate([r.cur_ids for r in batch], max_news)
        for r, (asst_ids, asst_lp, text) in zip(batch, gen, strict=True):
            _advance_after_turn(
                r, asst_ids, asst_lp, text,
                active_env=active_env, env_glue=env_glue, max_turns=max_turns,
            )

    # Score with the ACTUAL accumulated rollout state (matches rollout_one), batched per task.
    rewards = _score_rollouts(active_env, rollouts)
    return [r.result(rw) for r, rw in zip(rollouts, rewards, strict=True)]


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
    """Run ``len(examples)`` multi-turn rollouts with CONTINUOUS-BATCHED generation (no turn barrier).

    Same result as :func:`rollout_batch` / one :func:`rollout_one` per example — identical token
    alignment, env_mask, per-rollout reward and input order — but rollouts are NOT advanced in
    lockstep. Each rollout's assistant turn is an independent engine request; the moment one
    finishes, its env step runs and its NEXT turn is submitted, so the decode batch stays full
    instead of stalling at a turn boundary while the slowest rollout's turn (and then every
    rollout's env reply) completes. For high-variance multi-turn (rollouts of very different
    depths) this keeps the GPU busy across the many turn boundaries the synchronized path idles at.

    The work is split across two threads so the per-turn ENV work (env reply + glue render — the
    overhead that bounds an otherwise GPU-light rollout) overlaps the GPU decode instead of blocking
    it: the MAIN thread owns the engine (submit / poll / busy) and a single WORKER thread owns the
    env (``_advance_after_turn``). vLLM's ``step()`` runs the model forward in CUDA with the GIL
    released, so the worker advances finished turns DURING the decode of the still-running ones. The
    two threads share no mutable state — only thread-safe queues, and each next-turn prefix is handed
    over as a copy — so the env (not thread-safe) is touched by exactly one thread and the engine by
    exactly one thread. Results are byte-identical to one :func:`rollout_one` per example (a rollout
    keeps at most one request in flight, so its turns stay strictly sequential), in input order.

    The engine is injected as three callables so the loop is unit-testable on CPU:
      * ``submit(req_id, prefix_ids, max_tokens, initial)`` — enqueue one assistant-turn request
        (``initial`` marks a turn-0 prompt, the only externally-rendered ids worth bounds-checking);
      * ``poll()`` — return ``(req_id, token_ids, logprobs, text)`` for every request that FINISHED
        since the last call (``[]`` if none finished this step);
      * ``busy()`` — whether any request is still in flight.
    """
    rollouts = _build_rollout_states(examples, active_env, render, engine_max_len)
    by_id: dict[str, _RolloutState] = {}
    counter = 0
    to_env: queue.Queue = queue.Queue()  # main -> worker: finished turns to fold + run the env step
    to_submit: queue.Queue = queue.Queue()  # worker -> main: ("next", r, prefix, max_new) | ("done", r)

    def do_submit(r: _RolloutState, prefix: list[int], max_new: int, initial: bool) -> None:
        nonlocal counter
        req_id = f"r{counter}"
        counter += 1
        by_id[req_id] = r
        submit(req_id, prefix, max_new, initial)

    def env_worker() -> None:
        # Owns the env: fold each finished turn, run its env step (env reply + glue render), and hand
        # the next-turn prefix (a copy) back to the main thread — or signal the rollout is done. An
        # env/template error here must propagate to the main thread (which owns the engine), not die
        # silently in this thread and hang the main loop waiting for a result that never comes.
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
            except Exception as exc:  # surfaced + re-raised on the main thread (engine owner)
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
            # Re-raise the worker's env/template error on the main thread (the engine owner),
            # preserving the ORIGINAL worker traceback so the stack points at the real failing line.
            err = msg[1]
            raise err.with_traceback(err.__traceback__)
        if msg[0] == "done":
            completed += 1
        else:
            _, r, prefix, max_new = msg
            do_submit(r, prefix, max_new, False)

    try:
        for r in rollouts:  # prime turn 0 on the main thread
            max_new = _turn_budget(r, per_turn_max_tokens)
            if max_new is None:
                completed += 1
            else:
                do_submit(r, list(r.cur_ids), max_new, r.turns == 0)
        while completed < n:
            progressed = False
            while True:  # submit every next-turn the worker has produced (and count finished ones)
                try:
                    take(to_submit.get_nowait())
                    progressed = True
                except queue.Empty:
                    break
            if completed >= n:
                break
            if busy():  # step the engine; hand finished turns to the worker (overlaps its env work)
                for req_id, asst_ids, asst_lp, text in poll():
                    to_env.put((by_id.pop(req_id), asst_ids, asst_lp, text))
            elif not progressed:
                # nothing in flight and nothing newly ready: the worker is mid-advance — block on its
                # next output instead of spinning (every rollout is in exactly one stage, so this
                # can't deadlock: the only state with all queues + in-flight empty is all-done).
                with contextlib.suppress(queue.Empty):
                    take(to_submit.get(timeout=0.1))
    finally:
        to_env.put(None)
        worker.join()

    # Score with the ACTUAL accumulated rollout state (matches rollout_one), batched per task.
    rewards = _score_rollouts(active_env, rollouts)
    return [r.result(rw) for r, rw in zip(rollouts, rewards, strict=True)]


def render_message_ids(tok, messages, add_generation_prompt: bool, *, thinking: bool) -> list[int]:
    """Render ``messages`` with the chat template, then tokenize to a flat ``list[int]``.

    Render to text first, then tokenize — the return shape of apply_chat_template(tokenize=True)
    varies by tokenizer, whereas tok(text).input_ids is reliably a flat list[int] (matches the
    single-turn render_prompt path). add_special_tokens=False because the template already
    emits the special tokens. Shared by the GRPO rollout closure and mid-run eval so both
    produce identical token alignment.
    """
    text = tok.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        enable_thinking=thinking,
    )
    return [int(t) for t in tok(text, add_special_tokens=False).input_ids]


def _engine_vocab_size(engine) -> int | None:
    """Best-effort vocab size of the colocate vLLM engine, or None if it can't be read.

    Used only for a cheap fail-loud bounds check on the pre-tokenized prompt ids before they
    reach ``engine.generate`` (the ``prompt_token_ids`` path does no bounds checking, so an
    out-of-range id would otherwise surface as an opaque CUDA illegal-access). Never raises.
    """
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
    """Return a TRL ``rollout_func`` closure that drives ``active_env`` on the colocate engine.

    The closure reaches the in-process vLLM engine through ``trainer.vllm_generation.llm`` and
    samples each assistant turn with per-token logprobs. It returns exactly ONE rollout per
    prompt in the slice TRL passes: TRL's ``RepeatSampler`` already repeats each unique prompt
    ``num_generations`` times before calling ``rollout_func`` (the consecutive repeats form the
    GRPO group), so the closure must NOT multiply by ``num_generations`` again.
    """
    from vllm import SamplingParams  # gpu-only; imported lazily so the module loads on CPU

    try:
        # FINAL_ONLY makes each manual add_request emit exactly one RequestOutput, at finish, with
        # the complete turn (matching LLM.generate); without it the engine streams a cumulative
        # output every step. Optional so the CPU import (stubbed vllm) still works — poll() filters
        # on `finished` either way.
        from vllm.sampling_params import RequestOutputKind

        _final_only_kind = RequestOutputKind.FINAL_ONLY
    except Exception:
        _final_only_kind = None

    _render_cache: dict[str, list[int]] = {}
    _RENDER_CACHE_MAX = 8192

    def render(messages: list, add_generation_prompt: bool) -> list[int]:
        # The initial-prompt render is identical for every rollout in a GRPO group (they share one
        # prompt), so cache it by content instead of re-rendering num_generations times per step.
        cache_key = f"{add_generation_prompt}\x00{json.dumps(messages, sort_keys=True, default=str)}"
        cached = _render_cache.get(cache_key)
        if cached is not None:
            return cached
        ids = render_message_ids(tok, messages, add_generation_prompt, thinking=thinking)
        if len(_render_cache) < _RENDER_CACHE_MAX:
            _render_cache[cache_key] = ids
        return ids

    _glue_cache: dict[str, list[int]] = {}
    _GLUE_CACHE_MAX = 8192

    def env_glue(env_messages: list) -> list[int]:
        # The inter-turn glue is a pure function of env_messages (+ this closure's tokenizer /
        # thinking). Within a GRPO group every rollout gets the SAME env reply each turn, and many
        # turns repeat env messages across rollouts and steps, so apply_chat_template would
        # otherwise re-render byte-identical glue dozens-to-hundreds of times — the dominant per-turn
        # CPU cost in the (otherwise overhead-bound) multi-turn rollout. Cache by env-message
        # content; bounded so an env whose every reply is unique can't grow it without limit.
        cache_key = json.dumps(env_messages, sort_keys=True, default=str)
        cached = _glue_cache.get(cache_key)
        if cached is not None:
            return cached
        # Tokens between two assistant turns: close the previous assistant turn, render the env
        # reply message(s), and open the next generation prompt. Derived by rendering a probe
        # assistant turn followed by the env messages (+ generation prompt) and taking everything
        # AFTER the probe content — so the glue is exactly the template's inter-turn wrapper,
        # whatever it is (Qwen's <|im_end|> + user turn + <|im_start|>assistant + <think> block).
        # This avoids re-rendering history (which Qwen3 does not round-trip) and matches how the
        # model actually conditioned during generation. The probe is plain text the template
        # inserts verbatim into assistant content; its FIRST occurrence is the probe turn.
        probe = "flash-env-glue-probe"
        text = tok.apply_chat_template(
            [{"role": "assistant", "content": probe}, *env_messages],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=thinking,
        )
        # Locate the probe to slice off the inter-turn glue. Fail LOUD with context if the
        # template did not insert the assistant content verbatim (some templates strip/escape it,
        # or could emit the probe more than once) instead of a bare "substring not found".
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
        if len(_glue_cache) < _GLUE_CACHE_MAX:
            _glue_cache[cache_key] = glue
        return glue

    def rollout_func(prompts, trainer):
        engine = trainer.vllm_generation.llm
        # The colocate engine is a vLLM `LLM`; its V1 `LLMEngine` exposes the public
        # add_request / step / has_unfinished_requests loop that lets us decode many rollouts'
        # turns CONTINUOUSLY (a finished turn's slot refills with another rollout's next turn)
        # instead of one synchronized batched decode per turn.
        llm_engine = engine.llm_engine
        # Colocate vLLM sleep mode (GRPOConfig.vllm_enable_sleep_mode, ON for large / long-context
        # runs) offloads BOTH the rollout engine's weights and its KV cache between steps. TRL's
        # rollout_func path (GRPOTrainer._generate) calls vllm_generation.sync_weights() — which
        # wakes only tags=["weights"] — and then hands control to this closure, but, UNLIKE TRL's
        # own single-turn generate() path, it never wakes tags=["kv_cache"]. So the first decode
        # below would run against a freed/offloaded KV cache and fault with CUDA "illegal memory
        # access" on step 0. Wake the KV cache here and re-sleep after the whole batch, mirroring
        # trl.generation.vllm_generation.generate (and trl.experimental.openenv). No-op when sleep
        # mode is off (small/short-context runs keep the engine resident). See flash issue #162.
        sleep_mode = bool(getattr(getattr(trainer, "args", None), "vllm_enable_sleep_mode", False))
        vocab_size = _engine_vocab_size(engine)
        active_ids: set[str] = set()  # submitted-but-not-finished requests, for abort-on-exit

        def submit(req_id: str, prefix_ids: list[int], max_tokens: int, initial: bool) -> None:
            """Enqueue one assistant-turn request on the colocate engine."""
            if not prefix_ids:
                # Fail loudly on a degenerate prompt instead of letting it reach the embedding gather
                # as an opaque async CUDA illegal-access (the failure mode #162 was first mistaken
                # for): the prompt_token_ids path does no bounds checking.
                raise ValueError("multi-turn rollout produced an empty prompt for engine.add_request()")
            if initial:
                # Turn-0 prefixes are the only externally-rendered initial prompts (later turns are
                # vLLM-generated / tokenizer glue, already in range); validate each, since the
                # prompt_token_ids path does no bounds checking and an out-of-range id would surface
                # as an opaque CUDA illegal-access.
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
            """Advance the engine one step; return (req_id, token_ids, logprobs, text) for every
            request that finished this step (``[]`` if none did / a dummy batch ran)."""
            finished: list[tuple[str, list[int], list[float], str]] = []
            for out in llm_engine.step():
                if not getattr(out, "finished", False):
                    continue
                comp = out.outputs[0]
                token_ids = list(comp.token_ids)
                # comp.logprobs is a list (per position) of {token_id: Logprob}; pull the sampled
                # token's logprob at each position.
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

        # Wake the KV cache for the whole batch (see the note above), then re-sleep so the engine
        # returns to its fully-offloaded state and the optimizer step has the freed memory back.
        # `woke` is set AFTER a successful wake so the finally re-sleeps ONLY when we actually woke
        # the engine — a wake_up() that raises leaves the engine asleep (its resting state), and we
        # must not then call sleep() on it; a failure DURING the rollout still re-sleeps.
        woke = False
        try:
            if sleep_mode:
                engine.wake_up(tags=["kv_cache"])
                woke = True
            # ONE rollout per prompt: TRL's RepeatSampler already repeats each unique prompt
            # num_generations times BEFORE handing the slice to rollout_func (trl 1.6/1.7:
            # `prompts = [x["prompt"] for x in inputs]`, no dedup), and it expects exactly
            # len(prompts) completions back — the GRPO group is the consecutive num_generations rows
            # of the same prompt. rollout_async returns one result per example in input order, so
            # the group stays aligned.
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
            # Abort any still-in-flight requests so a mid-rollout error (e.g. an env_glue/template
            # failure on a later turn) can't leak live requests into the engine and corrupt the
            # next GRPO step. No-op on the success path (every request finished -> active_ids empty).
            if active_ids:
                with contextlib.suppress(Exception):
                    llm_engine.abort_request(list(active_ids))
            if woke:
                engine.sleep(level=2)

    return rollout_func

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

import json
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

    # Advance all rollouts in lockstep: each iteration samples one assistant turn for every
    # still-active rollout in a single batched call, then runs each rollout's env step.
    while True:
        batch: list[_RolloutState] = []
        max_news: list[int] = []
        for r in rollouts:
            if r.done:
                continue
            max_new = per_turn_max_tokens
            if r.budget is not None:
                remaining = r.budget - len(r.completion_ids)
                if remaining <= 0:  # prompt already fills the context -> this rollout is done
                    r.done = True
                    continue
                max_new = min(max_new, remaining)
            batch.append(r)
            max_news.append(max(1, max_new))
        if not batch:
            break
        gen = batched_generate([r.cur_ids for r in batch], max_news)
        for r, (asst_ids, asst_lp, text) in zip(batch, gen, strict=True):
            _advance_after_turn(
                r, asst_ids, asst_lp, text,
                active_env=active_env, env_glue=env_glue, max_turns=max_turns,
            )

    # Score with the ACTUAL accumulated rollout state (matches rollout_one).
    return [r.result(float(active_env.reward("", r.example, r.state))) for r in rollouts]


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

    def render(messages: list, add_generation_prompt: bool) -> list[int]:
        return render_message_ids(tok, messages, add_generation_prompt, thinking=thinking)

    def env_glue(env_messages: list) -> list[int]:
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
        return [int(t) for t in tok(glue_text, add_special_tokens=False).input_ids]

    def rollout_func(prompts, trainer):
        engine = trainer.vllm_generation.llm
        # Colocate vLLM sleep mode (GRPOConfig.vllm_enable_sleep_mode, ON for large / long-context
        # runs) offloads BOTH the rollout engine's weights and its KV cache between steps. TRL's
        # rollout_func path (GRPOTrainer._generate) calls vllm_generation.sync_weights() — which
        # wakes only tags=["weights"] — and then hands control to this closure, but, UNLIKE TRL's
        # own single-turn generate() path, it never wakes tags=["kv_cache"]. So engine.generate()
        # below would run against a freed/offloaded KV cache and fault with CUDA "illegal memory
        # access" on the very first generate of step 0. Wake the KV cache here and re-sleep after
        # the whole batch, mirroring trl.generation.vllm_generation.generate (and
        # trl.experimental.openenv). No-op when sleep mode is off (small/short-context runs keep
        # the engine resident across steps). See flash issue #162.
        sleep_mode = bool(getattr(getattr(trainer, "args", None), "vllm_enable_sleep_mode", False))
        vocab_size = _engine_vocab_size(engine)
        bounds_checked = False  # the initial prompts (the first batch) are validated once; see below

        def batched_generate(prefixes: list[list[int]], max_tokens_list: list[int]):
            """Sample ONE assistant turn for every active rollout in a single vLLM decode.

            Replaces the old one-generate-per-prompt-per-turn loop: vLLM batches the sequences and
            (with prefix caching, on for the colocate engine) reuses each rollout's growing prefix
            and the shared in-group prompt KV. Returns (token_ids, logprobs, text) per prefix, in
            input order.
            """
            nonlocal bounds_checked
            for p in prefixes:
                # Fail loudly on a degenerate prompt instead of letting it reach the embedding gather
                # as an opaque async CUDA illegal-access (the failure mode #162 was first mistaken
                # for): the prompt_token_ids path does no bounds checking.
                if not p:
                    raise ValueError("multi-turn rollout produced an empty prompt for engine.generate()")
            if not bounds_checked:
                # The FIRST batch is turn 0 — every prefix is an externally-rendered initial prompt,
                # the only segment that can carry tokenizer/model-mismatch ids (later turns' prefixes
                # are vLLM-generated / tokenizer glue, in range). Validate each one; the
                # prompt_token_ids path does no bounds checking.
                bounds_checked = True
                for p in prefixes:
                    lo, hi = min(p), max(p)
                    if lo < 0 or (vocab_size is not None and hi >= vocab_size):
                        raise ValueError(
                            f"multi-turn rollout prompt has out-of-range token id(s) [{lo}, {hi}] for "
                            f"vocab size {vocab_size} (tokenizer/model mismatch)"
                        )
            # Per-prompt SamplingParams: each rollout has its own remaining-token budget.
            sps = [
                SamplingParams(
                    max_tokens=max(1, int(mt)),
                    temperature=temperature,
                    top_p=top_p,
                    logprobs=1,  # include the sampled token's logprob at each position
                    stop=list(stop) if stop else None,
                )
                for mt in max_tokens_list
            ]
            # vLLM's LLM.generate takes prompts (TokensPrompt-style dicts), not a `prompt_token_ids`
            # kwarg — pass pre-tokenized ids as {"prompt_token_ids": ...}, ONE per active rollout.
            outs = engine.generate(
                [{"prompt_token_ids": list(p)} for p in prefixes],
                sampling_params=sps,
                use_tqdm=False,
            )
            results: list[tuple[list[int], list[float], str]] = []
            for out in outs:
                comp = out.outputs[0]
                token_ids = list(comp.token_ids)
                # comp.logprobs is a list (per position) of {token_id: Logprob}; pull the sampled
                # token's logprob at each position.
                lps: list[float] = []
                for pos, tid in enumerate(token_ids):
                    entry = (comp.logprobs or [])[pos] if comp.logprobs else None
                    lp = entry.get(tid) if entry else None
                    lps.append(float(getattr(lp, "logprob", 0.0)) if lp is not None else 0.0)
                results.append((token_ids, lps, comp.text))
            return results

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
            # of the same prompt. rollout_batch advances every active rollout's turn in a single
            # batched_generate call and returns results in input order, so the group stays aligned.
            examples = [examples_by_key.get(_prompt_key(p), {"prompt": p}) for p in prompts]
            rollouts = rollout_batch(
                examples=examples,
                active_env=active_env,
                render=render,
                batched_generate=batched_generate,
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
            if woke:
                engine.sleep(level=2)

    return rollout_func

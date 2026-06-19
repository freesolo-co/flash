"""Multi-turn / tool GRPO rollout for TRL's experimental ``rollout_func`` (colocate vLLM).

TRL's ``GRPOTrainer`` generates a single assistant turn per prompt, which cannot drive a
verifiers ``MultiTurnEnv`` / ``ToolEnv`` turn loop (model turn -> env reply -> ...). This
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
  * scores each rollout with the env's weighted rubric (``reward_from_messages``) and returns
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
    max_turns: int,
    per_turn_max_tokens: int,
    engine_max_len: int | None = None,
    on_warn: Callable[[str], None] | None = None,
) -> RolloutResult:
    """Run one multi-turn/tool rollout and return TRL ``rollout_func`` fields for it.

    Args:
        example: the dataset row (carries ``answer``/``info`` for the rubric).
        active_env: the ``VerifiersEnvironment`` adapter (drives the turn loop + scoring).
        render: ``render(messages, add_generation_prompt) -> token_ids`` (chat template).
        generate: ``generate(prefix_token_ids, max_tokens) -> (token_ids, token_logprobs,
            text)`` for one sampled assistant turn (model tokens + sampling logprobs + text);
            ``max_tokens`` bounds that turn so it can't overflow the engine context.
        max_turns: hard cap on model turns (defense against a non-terminating env).

    Returns a dict with ``prompt_ids``, ``completion_ids``, ``logprobs``, ``env_mask`` (all
    token-aligned) and the scalar ``reward`` for this rollout.
    """
    state = active_env.new_rollout_state(example)
    messages = [dict(m) for m in state["prompt"]]
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

        # Env-segment tokens = the suffix added by re-rendering the conversation (with the next
        # generation prompt) beyond what we already have. Masked (0) — they are not the
        # policy's tokens — but kept in completion_ids so the next turn conditions on them. This
        # REQUIRES a prefix-preserving template (appending a message must not retokenize earlier
        # turns); otherwise the model/env token boundary is wrong and the loss mask is garbage —
        # so fail loudly rather than silently mis-train.
        new_ids = render(messages, True)
        if new_ids[: len(cur_ids)] != cur_ids:
            msg = (
                "multi-turn rollout requires a prefix-preserving chat template (appending a "
                "message must not retokenize earlier turns); this model's template is not. Use "
                "a single-turn/tool env, or a model whose template is prefix-preserving."
            )
            if on_warn:
                on_warn(msg)
            raise ValueError(msg)
        env_seg = new_ids[len(cur_ids) :]
        completion_ids.extend(env_seg)
        logprobs.extend([0.0] * len(env_seg))
        env_mask.extend([0] * len(env_seg))
        cur_ids = list(new_ids)

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
    num_generations_attr: str = "num_generations",
):
    """Return a TRL ``rollout_func`` closure that drives ``active_env`` on the colocate engine.

    The closure reaches the in-process vLLM engine through ``trainer.vllm_generation.llm`` and
    samples each assistant turn with per-token logprobs; ``num_generations`` rollouts are
    produced per prompt (TRL requires the flattened per-prompt grouping).
    """
    from vllm import SamplingParams  # gpu-only; imported lazily so the module loads on CPU

    def render(messages: list, add_generation_prompt: bool) -> list[int]:
        return render_message_ids(tok, messages, add_generation_prompt, thinking=thinking)

    def rollout_func(prompts, trainer):
        engine = trainer.vllm_generation.llm
        num_gen = int(getattr(trainer, num_generations_attr, 1) or 1)

        def generate(prefix_ids: list[int], max_tokens: int):
            sp = SamplingParams(
                max_tokens=max(1, int(max_tokens)),
                temperature=temperature,
                top_p=top_p,
                logprobs=1,  # include the sampled token's logprob at each position
                stop=list(stop) if stop else None,
            )
            # vLLM's LLM.generate takes prompts (TokensPrompt-style dicts), not a
            # `prompt_token_ids` kwarg — pass pre-tokenized ids as {"prompt_token_ids": ...}.
            out = engine.generate(
                [{"prompt_token_ids": list(prefix_ids)}],
                sampling_params=sp,
                use_tqdm=False,
            )
            comp = out[0].outputs[0]
            token_ids = list(comp.token_ids)
            # comp.logprobs is a list (per position) of {token_id: Logprob}; pull the sampled
            # token's logprob at each position.
            lps: list[float] = []
            for pos, tid in enumerate(token_ids):
                entry = (comp.logprobs or [])[pos] if comp.logprobs else None
                lp = entry.get(tid) if entry else None
                lps.append(float(getattr(lp, "logprob", 0.0)) if lp is not None else 0.0)
            return token_ids, lps, comp.text

        # One accumulator list per rollout field (batched dict-of-lists across all rollouts).
        out: dict[str, list] = {k: [] for k in _ROLLOUT_FIELDS}
        for prompt in prompts:
            example = examples_by_key.get(_prompt_key(prompt), {"prompt": prompt})
            for _ in range(num_gen):
                r = rollout_one(
                    example=example,
                    active_env=active_env,
                    render=render,
                    generate=generate,
                    max_turns=max_turns,
                    per_turn_max_tokens=max_completion,
                    engine_max_len=engine_max_len,
                    on_warn=print,
                )
                for k in out:
                    out[k].append(r[k])
        return out

    return rollout_func

"""Pure multi-turn text GRPO on OpenRLHF (parity plan #8).

Reuses the exact, already-tested TRL multi-turn rollout core in
:mod:`flash.engine.multiturn_rollout` (token glue, seam de-duplication, per-turn
environment masks, episode scoring) and drives the live Flash environment only through the
authenticated localhost session bridge from :mod:`flash.engine.worker.openrlhf_session_bridge`
(parity plan #7), so no environment implementation or secret is serialized into a Ray rollout
actor.

Two halves:

* :class:`FlashEnvSessionDriver` (parent-only) adapts one Flash multi-turn env into the bridge's
  :class:`~flash.engine.worker.openrlhf_session_bridge.SessionDriver` protocol. ``step`` folds the
  model's assistant turn into the running transcript with the same ``record_model_turn`` /
  ``env_reply`` / ``rollout_done`` semantics and the same probe-trick inter-turn glue + seam
  de-duplication as TRL, and returns the *token ids* the next generation must be conditioned on plus
  the exact per-token environment mask (1 = assistant/trainable, 0 = glue/observation). ``close``
  scores the finished episode with ``reward_from_messages``. No token is ever re-rendered.

* :func:`build_multiturn_action_ranges` converts that per-token env mask into the
  OpenRLHF ``action_ranges`` (contiguous assistant spans) that the multi-turn ``AgentExecutor``
  consumes, so the OpenRLHF experience trains exactly the assistant tokens TRL would, with episode
  reward. Per-turn credit is intentionally *not* introduced here (that is the pinned-OpenRLHF
  data-model change in parity plan #11); this path uses TRL's default episode reward.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flash.engine.multiturn_rollout import (
    _dedup_seam_terminator,
    make_env_glue,
    render_message_ids,
)

# Sentinel wire values kept provider-neutral and JSON-safe. The bridge only moves opaque
# observations; these are the observation payload this driver emits.
_OBS_TOKENS = "token_ids"
_OBS_ENV_MASK = "env_mask"


class FlashEnvSessionDriver:
    """Adapts one Flash multi-turn env onto the bridge ``SessionDriver`` protocol (parent-only).

    The bridge injects this via its ``session_factory`` and never lets a rollout actor touch the
    env directly. ``action`` on :meth:`step` is the assistant turn ``{"text", "token_ids"}``; the
    returned observation carries the inter-turn glue token ids + env mask the child must append
    verbatim before the next generation, so token alignment is decided here in the parent using the
    same tokenizer the child generates against.
    """

    __slots__ = (
        "_budget",
        "_completion_len",
        "_env",
        "_env_glue",
        "_example",
        "_max_turns",
        "_messages",
        "_state",
        "_thinking",
        "_tokenizer",
        "_turns",
    )

    def __init__(
        self,
        env: Any,
        example: Any,
        *,
        tokenizer: Any,
        thinking: bool,
        max_turns: int,
        completion_budget: int | None,
        env_glue: Callable[[list], list[int]] | None = None,
    ) -> None:
        self._env = env
        self._example = example
        self._tokenizer = tokenizer
        self._thinking = bool(thinking)
        self._max_turns = int(max_turns)
        self._budget = completion_budget
        self._completion_len = 0
        self._turns = 0
        # Reuse the exact probe-trick glue builder so env glue is byte-identical to TRL/SFT/OPD.
        self._env_glue = env_glue or make_env_glue(tokenizer, thinking=self._thinking)
        state = env.new_rollout_state(example)
        initial = state.get("prompt") or state.get("messages")
        if not isinstance(initial, list):
            raise KeyError("multi-turn rollout state must include prompt or messages")
        self._state = state
        self._messages = [dict(m) for m in initial]

    # --- bridge SessionDriver protocol -------------------------------------------------
    def initial_observation(self) -> dict[str, Any]:
        """Prompt token ids the first generation is conditioned on (add_generation_prompt=True)."""
        return {_OBS_TOKENS: render_message_ids(
            self._tokenizer, self._messages, True, thinking=self._thinking
        )}

    def step(self, action: Any) -> tuple[Any, bool]:
        """Fold one assistant turn, run the env, and return (glue observation, done)."""
        if not isinstance(action, dict):
            raise TypeError("multi-turn action must be a mapping with text and token_ids")
        asst_text = action["text"]
        asst_ids = [int(t) for t in action["token_ids"]]
        self._completion_len += len(asst_ids)
        self._turns += 1
        self._env.record_model_turn(self._state, asst_text)
        self._messages.append({"role": "assistant", "content": asst_text})

        # Termination truth table matches TRL _advance_after_turn exactly.
        if self._budget is not None and self._completion_len >= self._budget:
            return {_OBS_TOKENS: [], _OBS_ENV_MASK: []}, True
        if self._turns >= self._max_turns or self._env.rollout_done(self._state, self._max_turns):
            return {_OBS_TOKENS: [], _OBS_ENV_MASK: []}, True
        env_msgs = self._env.env_reply(self._messages, self._state)
        if not env_msgs:
            return {_OBS_TOKENS: [], _OBS_ENV_MASK: []}, True
        self._messages.extend(env_msgs)
        if self._env.rollout_done(self._state, self._max_turns):
            return {_OBS_TOKENS: [], _OBS_ENV_MASK: []}, True

        # One assistant terminator, not two: keep the model's own final token, drop the glue copy.
        glue = _dedup_seam_terminator(action["token_ids"], self._env_glue(env_msgs))
        if self._budget is not None and self._completion_len + len(glue) > self._budget:
            return {_OBS_TOKENS: [], _OBS_ENV_MASK: []}, True
        self._completion_len += len(glue)
        return {_OBS_TOKENS: glue, _OBS_ENV_MASK: [0] * len(glue)}, False

    def score(self) -> float:
        """Episode reward over the finished transcript (TRL reward_from_messages)."""
        reward = float(self._env.reward_from_messages(self._messages, self._example))
        if reward != reward or reward in (float("inf"), float("-inf")):
            raise ValueError("multi-turn environment reward must be finite")
        return reward

    def close(self) -> None:
        closer = getattr(self._env, "close", None)
        if callable(closer):
            closer()


def build_multiturn_action_ranges(
    prompt_len: int, per_turn_action_lens: list[int], glue_lens: list[int]
) -> list[tuple[int, int]]:
    """Contiguous assistant-token spans (OpenRLHF ``action_ranges``) over the flat rollout stream.

    The stream is ``prompt || (action_0 || glue_0) || (action_1 || glue_1) || ...``; the trailing
    turn has no glue. Each returned ``(start, end)`` is exactly the tokens the assistant generated in
    that turn, so the policy loss trains those and only those (env glue/observation tokens are the
    complement and stay masked), matching TRL's per-token ``env_mask`` 1-spans.
    """
    n = len(per_turn_action_lens)
    if len(glue_lens) != max(0, n - 1):
        raise ValueError(
            "multi-turn glue segment count must be exactly one fewer than the assistant turns"
        )
    ranges: list[tuple[int, int]] = []
    cursor = int(prompt_len)
    for i, action_len in enumerate(per_turn_action_lens):
        start = cursor
        end = start + int(action_len)
        ranges.append((start, end))
        cursor = end + (int(glue_lens[i]) if i < len(glue_lens) else 0)
    return ranges


# --- child-side executor (parity plan #8b) ---------------------------------------------
# The OpenRLHF ``MultiTurnAgentExecutor`` re-tokenizes each env ``observation_text`` (agent.py),
# which would drift from the token-exact inter-turn glue the parent bridge already decided. To keep
# the stream byte-identical to TRL/SFT/OPD, Flash drives the loop through a token-exact executor that
# appends the bridge's glue *token ids* verbatim and records assistant ``action_ranges``. Live
# generation (colocated ``llm_engine``) + the HTTP bridge client are GPU-path only; the mask
# assembly below is pure and CPU-proved.

_MT_TOKENS = _OBS_TOKENS
_MT_ENV_MASK = _OBS_ENV_MASK


def build_response_mask(total_len: int, action_ranges: list[tuple[int, int]]) -> list[int]:
    """Per-token response mask over the flat rollout stream (1 = assistant/trainable, 0 = prompt or
    inter-turn env glue/observation).

    Mirrors OpenRLHF ``samples_generator`` (``action_mask[start:end] = 1`` for each action range)
    *before* its ``[1:]`` learner shift, so the trained tokens are exactly the assistant spans
    :func:`build_multiturn_action_ranges` emits — i.e. TRL's per-token ``env_mask`` 1-spans.
    """
    n = int(total_len)
    if n < 0:
        raise ValueError("total_len must be non-negative")
    mask = [0] * n
    for start, end in action_ranges:
        s = max(0, int(start))
        e = min(n, int(end))
        for i in range(s, e):
            mask[i] = 1
    return mask


def assemble_multiturn_rollout(
    prompt_ids: list[int],
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble one finished multi-turn episode into the OpenRLHF sample shape (token-exact).

    ``turns`` is the ordered per-turn record the executor collects: each ``{"action_ids": [...],
    "glue_ids": [...]}`` where ``glue_ids`` is the post-turn inter-turn glue (empty on the final
    turn). Returns ``token_ids`` (prompt || action_0 || glue_0 || action_1 || ...), ``action_ranges``
    (assistant spans), and ``response_mask`` (1 on those spans). This is the exact stream and mask
    OpenRLHF would build from this executor's ``action_ranges``; it never re-tokenizes, so it stays
    byte-identical to the parent bridge / TRL glue.
    """
    token_ids: list[int] = list(prompt_ids)
    per_turn_action_lens: list[int] = []
    glue_lens: list[int] = []
    for i, turn in enumerate(turns):
        action_ids = [int(t) for t in turn["action_ids"]]
        per_turn_action_lens.append(len(action_ids))
        token_ids.extend(action_ids)
        glue_ids = [int(t) for t in (turn.get("glue_ids") or [])]
        if i < len(turns) - 1:
            glue_lens.append(len(glue_ids))
            token_ids.extend(glue_ids)
        elif glue_ids:
            raise ValueError("the final multi-turn turn must not carry inter-turn glue")
    action_ranges = build_multiturn_action_ranges(len(prompt_ids), per_turn_action_lens, glue_lens)
    response_mask = build_response_mask(len(token_ids), action_ranges)
    return {
        "token_ids": token_ids,
        "action_ranges": action_ranges,
        "response_mask": response_mask,
    }


def _load_openrlhf_agent_base():
    """Lazily import the OpenRLHF agent primitives (present only in the cu13 child image)."""
    from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor

    return AgentInstanceBase, MultiTurnAgentExecutor


class FlashBridgeAgentInstance:
    """Child-side :class:`~openrlhf.utils.agent.AgentInstanceBase` that drives one Flash multi-turn
    episode through the authenticated localhost session bridge (parity #7).

    It holds only a bridge *client* + session id/lease — never the environment itself — so no env
    implementation or secret is serialized into the Ray rollout actor. ``reset`` opens a bridge
    session and returns the initial prompt observation; ``step`` posts the assistant turn and returns
    the env's token-exact glue observation + done + episode reward. The concrete OpenRLHF base is
    mixed in by :func:`flash_multiturn_agent_instance_cls` so this module imports without OpenRLHF.
    """

    def __init__(self, bridge_client: Any) -> None:
        self._bridge = bridge_client
        self._session: str | None = None
        self._lease: str | None = None

    async def reset(self, states: dict, **kwargs) -> dict:
        opened = self._bridge.reset(states)
        self._session = opened["session_id"]
        self._lease = opened["lease"]
        return {_MT_TOKENS: list(opened[_MT_TOKENS])}

    async def step(self, state_dict: dict, **kwargs) -> dict:
        action = state_dict["action"]
        obs, done, reward = self._bridge.step(self._session, self._lease, action)
        return {
            "environment_feedback": {
                _MT_TOKENS: list(obs.get(_MT_TOKENS, [])),
                _MT_ENV_MASK: list(obs.get(_MT_ENV_MASK, [])),
            },
            "done": bool(done),
            "rewards": reward,
            "scores": reward,
        }


def flash_multiturn_agent_instance_cls():
    """Return ``FlashBridgeAgentInstance`` as a concrete ``AgentInstanceBase`` subclass (child only)."""
    agent_base, _ = _load_openrlhf_agent_base()
    return type("FlashBridgeAgentInstance", (FlashBridgeAgentInstance, agent_base), {})

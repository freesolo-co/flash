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

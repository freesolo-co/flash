"""Offline contract checks for local Freesolo environments."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from . import render
from .envpush import _err, _resolve_local_env_entrypoint

_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_ECHO_RESPONSE = "test"
_PREVIEW_CHARS = 200
_TURN_SAFETY_BUFFER = 2


def _prompt_error(messages: object) -> str | None:
    if not isinstance(messages, list) or not messages:
        return "prompt must be a non-empty list"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return f"prompt message {index} must be a dict"
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            return f"prompt message {index} must have a non-empty string role"
        if role.strip().lower() not in _ALLOWED_ROLES:
            return f"prompt message {index} has unsupported role {role!r}"
        if "content" not in message:
            return f"prompt message {index} must have a content key"
    return None


def _reference_turns(env, example: dict) -> list[str]:
    messages = env.sft_completion(example)
    assistant = [
        message
        for message in messages
        if isinstance(message, dict)
        and str(message.get("role") or "").strip().lower() == "assistant"
    ]
    selected = assistant or [message for message in messages if isinstance(message, dict)]
    return [
        content if isinstance(content := message.get("content"), str) else str(content or "")
        for message in selected
    ]


def _resolve_policy(reference_turns: list[str]) -> str:
    return "replay" if "".join(reference_turns).strip() else "echo"


def _preview(value: object) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(f"{item.get('role', '?')}: {item.get('content', '')}")
            else:
                parts.append(str(item))
        text = " | ".join(parts)
    else:
        text = str(value)
    text = " ".join(text.split())
    if not text:
        return "(empty)"
    if len(text) <= _PREVIEW_CHARS:
        return text
    return f"{text[: _PREVIEW_CHARS - 3]}..."


def _check_prompt(messages: object) -> list[dict]:
    error = _prompt_error(messages)
    if error:
        raise ValueError(f"prompt is not well-formed: {error}")
    return messages


def _drive_single_turn(env, example: dict):
    prompt = _check_prompt(env.prompt_messages(example))
    reference_turns = _reference_turns(env, example)
    policy = _resolve_policy(reference_turns)
    response = "\n".join(reference_turns) if policy == "replay" else _ECHO_RESPONSE
    reward = float(env.reward(response, example))
    return policy, 1, reward, prompt, [response]


def _drive_multi_turn(env, example: dict):
    state = env.new_rollout_state(example)
    prompt = _check_prompt(state.get("prompt") or state.get("messages"))
    reference_turns = _reference_turns(env, example)
    policy = _resolve_policy(reference_turns)
    hard_cap = int(env.max_turns)
    per_example_cap = state.get("max_episode_turns")
    effective_cap = int(per_example_cap) if per_example_cap is not None else hard_cap
    safety_limit = effective_cap + _TURN_SAFETY_BUFFER
    turn_index = 0
    responses: list[str] = []
    safety_tripped = False

    while not env.rollout_done(state, max_turns=hard_cap):
        if policy == "replay" and turn_index < len(reference_turns):
            content = reference_turns[turn_index]
        else:
            content = _ECHO_RESPONSE
        responses.append(content)
        env.record_model_turn(state, content)
        env.env_reply(state["messages"], state)
        turn_index += 1
        if turn_index > safety_limit:
            safety_tripped = True
            break

    if safety_tripped:
        raise RuntimeError(f"episode exceeded the turn safety limit ({safety_limit})")
    if not env.rollout_done(state, max_turns=hard_cap):
        raise RuntimeError(f"episode did not terminate within the turn cap ({effective_cap})")

    reward = float(env.reward("", example, state))
    return policy, turn_index, reward, prompt, responses


def _load_failure(reason: str) -> int:
    _err(f"env test failed: {reason}")
    print("0/0 episodes passed contract checks")
    return _err("overall: FAIL")


def cmd_env_test(args) -> int:
    """Load a local environment and drive deterministic offline contract checks."""
    try:
        _, _, entrypoint, _ = _resolve_local_env_entrypoint(Path(args.path))
        from flash.envs.loader import load_freesolo_environment

        env = load_freesolo_environment(str(entrypoint))
        dataset = env.dataset()
    except Exception as exc:
        reason = str(exc) or exc.__class__.__name__
        return _load_failure(reason.replace("cannot publish", "cannot test"))

    if not dataset:
        return _load_failure("dataset is empty")

    episode_count = min(max(1, int(args.episodes)), len(dataset))
    passed = 0
    for index, example in enumerate(dataset[:episode_count], start=1):
        policy = "replay"
        turns = 0
        reward: float | None = None
        prompt: object = []
        responses: list[str] = []
        failure: str | None = None
        try:
            if env.multi_turn:
                policy, turns, reward, prompt, responses = _drive_multi_turn(env, example)
            else:
                policy, turns, reward, prompt, responses = _drive_single_turn(env, example)
            if not math.isfinite(reward):
                raise ValueError(f"reward is not finite: {reward}")
        except Exception as exc:
            failure = str(exc) or exc.__class__.__name__

        reward_text = "n/a" if reward is None else f"{reward:.6f}"
        print(f"episode {index}: policy={policy} turns={turns} reward={reward_text}")
        print(f"  prompt: {_preview(prompt)}")
        print(f"  response: {_preview(responses)}")
        if failure:
            _err(f"episode {index} failed contract checks: {failure}")
            continue

        passed += 1
        if policy == "replay" and reward is not None and reward <= 0.0:
            print(
                f"WARNING: replay gold answer scored low (reward={reward:.6f}); "
                "check the reward function",
                file=sys.stderr,
            )

    print(f"{passed}/{episode_count} episodes passed contract checks")
    if passed != episode_count:
        return _err("overall: FAIL")
    print(render.ok("overall: PASS") if render.styled() else "overall: PASS")
    return 0

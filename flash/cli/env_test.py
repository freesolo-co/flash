"""Offline contract checks for local Freesolo environments."""

from __future__ import annotations

import math
import sys
import tomllib
from pathlib import Path

from . import render
from .envpush import _err, _resolve_local_env_entrypoint

_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_ECHO_RESPONSE = "test"
_PREVIEW_CHARS = 200
_DEFAULT_EPISODES = 3


def _report_algorithm_shape(env) -> None:
    """Say up front which algorithms this environment's shape rules out.

    opsd phase 1 refuses multi-turn and tool environments, but that guard lives at worker boot:
    past admission, past gpu allocation, past billing. a multi-turn env therefore pays for a gpu
    to learn something knowable offline, since `multi_turn` is derived from the sdk base class at
    load time and is already settled here. report it rather than let the gpu do it.
    """
    multi_turn = bool(getattr(env, "multi_turn", False))
    tool_env = bool(getattr(env, "is_tool_env", False))
    if not (multi_turn or tool_env):
        return
    shape = "tool-using" if tool_env else "multi-turn"
    message = (
        f'this environment is {shape}; algorithm = "opsd" will be rejected at worker '
        "startup (after the gpu is allocated and billed). sft, grpo and opd support this shape."
    )
    print(render.warn(message) if render.styled() else f"warning: {message}", file=sys.stderr)


def _check_messages(messages: object, label: str) -> list[dict]:
    """Validate that `messages` is a well-formed chat message list and return it."""
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{label} is not well-formed: {label} must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{label} is not well-formed: {label} message {index} must be a dict")
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(
                f"{label} is not well-formed: {label} message {index} "
                "must have a non-empty string role"
            )
        if role.strip().lower() not in _ALLOWED_ROLES:
            raise ValueError(
                f"{label} is not well-formed: {label} message {index} has unsupported role {role!r}"
            )
        if "content" not in message:
            raise ValueError(
                f"{label} is not well-formed: {label} message {index} must have a content key"
            )
        # content is a string, a multimodal content-block list, or null for a native
        # assistant tool-call message. reject scalars like ints so a malformed message is
        # not reported as a false pass.
        content = message["content"]
        if content is not None and not isinstance(content, (str, list)):
            raise ValueError(
                f"{label} is not well-formed: {label} message {index} content "
                "must be a string, a content-block list, or null"
            )
    return messages


def _message_text(content: object) -> str:
    # mirror flash.multimodal.assistant_completion_text block handling: a message's replay
    # text is its string content, or the concatenation of its openai-style text blocks; any
    # other shape (null tool-call content, image-only blocks) yields no replay text.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _reference_turns(env, example: dict) -> list[str]:
    # the sft_completion gold answer stands in for the missing policy model. validate its
    # envelope like the prompt so a malformed completion (scalar content, missing role)
    # fails the episode instead of silently falling back to echo. text is extracted the
    # same way the real reward path grades a completion, so a gold answer expressed as
    # openai-style text blocks is replayed instead of echoed; text-free turns (null content
    # or image-only blocks) yield an empty replay string that is kept in place.
    messages = _check_messages(env.sft_completion(example), "sft_completion")
    # only assistant turns stand in for the policy model; a gold completion with no assistant
    # message must NOT replay user/system text as the model response -- yield no replay text so
    # _resolve_policy falls back to echo.
    assistant = [m for m in messages if m["role"].strip().lower() == "assistant"]
    # assistant turns only (a gold with no assistant message must echo, not replay user/system);
    # keep text-free turns positionally (empty string) so multi-turn replay stays aligned.
    return [_message_text(m["content"]) for m in assistant]


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


def _new_record() -> dict:
    """Mutable per-episode record so a failing episode still reports real progress."""
    return {"policy": "n/a", "turns": 0, "reward": None, "prompt": [], "responses": []}


def _drive_single_turn(env, example: dict, record: dict) -> None:
    prompt = _check_messages(env.prompt_messages(example), "prompt")
    record["prompt"] = prompt
    reference_turns = _reference_turns(env, example)
    policy = _resolve_policy(reference_turns)
    record["policy"] = policy
    response = (
        "\n".join(turn for turn in reference_turns if turn)
        if policy == "replay"
        else _ECHO_RESPONSE
    )
    record["responses"] = [response]
    record["turns"] = 1
    record["reward"] = float(env.reward(response, example))


def _drive_multi_turn(env, example: dict, record: dict) -> None:
    state = env.new_rollout_state(example)
    record["prompt"] = _check_messages(state.get("prompt") or state.get("messages"), "prompt")
    reference_turns = _reference_turns(env, example)
    policy = _resolve_policy(reference_turns)
    record["policy"] = policy
    # mirror the worker turn loop (flash/engine/multiturn_rollout.py): drive one model
    # turn, then stop at the hard turn ceiling, on the env's own done signal, or when the
    # env yields no reply. the hard cap is fixed at what the trainer passes (env.max_turns)
    # and the turn counter rises every turn until it reaches the cap, so a cooperatively-
    # stepping env terminates here exactly as it would in training; no separate
    # non-termination guard is needed.
    hard_cap = int(env.max_turns)
    turns = 0
    while True:
        if policy == "replay" and turns < len(reference_turns):
            content = reference_turns[turns]
        else:
            content = _ECHO_RESPONSE
        record["responses"].append(content)
        env.record_model_turn(state, content)
        turns += 1
        record["turns"] = turns
        if turns >= hard_cap or env.rollout_done(state, max_turns=hard_cap):
            break
        env_msgs = env.env_reply(state["messages"], state)
        if not env_msgs:
            break
        # the env's own reply messages feed the chat template for the next turn in the real
        # rollout, so validate their envelope here too: a malformed reply that would break
        # remotely must fail the episode instead of slipping through on a finite reward.
        _check_messages(env_msgs, "env_reply")
        if env.rollout_done(state, max_turns=hard_cap):
            break

    record["reward"] = float(env.reward("", example, state))


def _load_failure(reason: str) -> int:
    _err(f"env test failed: {reason}")
    print("0/0 episodes passed contract checks")
    return _err("overall: FAIL")


def _env_params(args) -> dict:
    """Build the ``load_environment()`` kwargs from ``--split`` / ``--param KEY=VALUE``.

    Mirrors ``[environment.params]`` so the local gate can validate the split a run actually
    trains on. Without this the gate always loaded ``dataset/train.jsonl`` and could pass while
    the configured split was never exercised.
    """
    params: dict = {}
    for item in getattr(args, "param", None) or []:
        key, sep, raw = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--param must be KEY=VALUE (got {item!r})")
        # parse as a toml value so types match [environment.params]; fall back to a bare string
        # for unquoted text, which is what users type most often.
        try:
            params[key] = tomllib.loads(f"v = {raw.strip()}")["v"]
        except tomllib.TOMLDecodeError:
            params[key] = raw.strip()
    split = getattr(args, "split", None)
    if split and str(split).strip():
        params["split"] = str(split).strip()
    return params


def cmd_env_test(args) -> int:
    """Load a local environment and drive deterministic offline contract checks.

    a fully non-returning environment hook (one that never yields control back) cannot be
    interrupted in-process, so run this under a ci job timeout to bound that class of
    defect; the per-episode turn cap (env.max_turns) bounds any cooperatively-stepping
    multi-turn loop exactly as the trainer does.
    """
    try:
        _, _, entrypoint, _ = _resolve_local_env_entrypoint(Path(args.path))
    except (Exception, SystemExit) as exc:
        reason = str(exc) or exc.__class__.__name__
        return _load_failure(reason.replace("cannot publish", "cannot test"))

    try:
        params = _env_params(args)
    except ValueError as exc:
        return _load_failure(str(exc))

    try:
        from flash.envs.loader import load_freesolo_environment

        # resolve to an absolute path so the loader takes its local-file branch; a bare
        # relative dir like `my-env` matches the managed-slug pattern and would otherwise
        # resolve remotely, breaking the offline contract.
        env = load_freesolo_environment(str(entrypoint.resolve()), **params)
        dataset = env.dataset()
    except (Exception, SystemExit) as exc:
        reason = str(exc) or exc.__class__.__name__
        return _load_failure(reason)

    if not dataset:
        return _load_failure("dataset is empty")

    _report_algorithm_shape(env)

    episode_count = min(_DEFAULT_EPISODES, len(dataset))
    passed = 0
    replayed = 0
    scored_zero = 0
    for index, example in enumerate(dataset[:episode_count], start=1):
        record = _new_record()
        failure: str | None = None
        try:
            if env.multi_turn:
                _drive_multi_turn(env, example, record)
            else:
                _drive_single_turn(env, example, record)
            reward = record["reward"]
            if reward is None or not math.isfinite(reward):
                raise ValueError(f"reward is not finite: {reward}")
        except (Exception, SystemExit) as exc:
            failure = str(exc) or exc.__class__.__name__

        reward = record["reward"]
        reward_text = "n/a" if reward is None else f"{reward:.6f}"
        print(
            f"episode {index}: policy={record['policy']} "
            f"turns={record['turns']} reward={reward_text}"
        )
        print(f"  prompt: {_preview(record['prompt'])}")
        print(f"  response: {_preview(record['responses'])}")
        if failure:
            _err(f"episode {index} failed contract checks: {failure}")
            continue

        passed += 1
        if record["policy"] == "replay" and reward is not None:
            replayed += 1
            if reward <= 0.0:
                scored_zero += 1
                message = (
                    f"replay gold answer scored low (reward={reward:.6f}); "
                    "check the reward function"
                )
                print(
                    render.warn(message) if render.styled() else f"warning: {message}",
                    file=sys.stderr,
                )

    print(f"{passed}/{episode_count} episodes passed contract checks")
    if passed != episode_count:
        return _err("overall: FAIL")
    # every replayed gold answer scoring 0 means the grader cannot recognize its own reference
    # answers: a broken reward function or a missing runtime dependency, not a hard dataset. that
    # must fail the gate, since passing here sends a run to a gpu that can only see flat-zero reward.
    if replayed and scored_zero == replayed:
        _err(
            f"all {replayed} replayed gold answer(s) scored 0.0; the reward function cannot score "
            "its own reference answers. check the grader and that its runtime dependencies are "
            "installed in this environment."
        )
        return _err("overall: FAIL")
    print(render.ok("overall: PASS") if render.styled() else "overall: PASS")
    return 0

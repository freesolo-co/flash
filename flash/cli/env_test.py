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
# a completion no reasonable grader accepts, used as the negative control for the flat-reward
# check below. deliberately not _ECHO_RESPONSE: that string already stands for "there is no gold
# answer to replay", and reusing it would conflate the two.
_NEGATIVE_CONTROL = "flash env test negative control: this answer is deliberately wrong."
_PREVIEW_CHARS = 200
_DEFAULT_EPISODES = 3


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


def _turn_is_representable(message: dict) -> bool:
    """Whether replaying this assistant turn as plain text reproduces the reference faithfully.

    The driver replays each assistant turn as a text-only message. A native tool call
    (``content=None`` plus ``tool_calls``) or an image-only turn therefore reaches the grader as an
    empty turn stripped of the payload the reference actually carried. A grader that correctly
    rejects that mutilated transcript scores zero, which is a property of the replay, not of the
    reward function -- so such episodes must not feed the flat-reward grader gate.
    """
    if message.get("tool_calls"):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        # text blocks survive extraction verbatim; anything else (an image block) is dropped.
        return all(isinstance(block, dict) and block.get("type") == "text" for block in content)
    # bare null content with no tool_calls carries no payload at all, so replaying it as the
    # empty string loses nothing. only a null that stands in for tool_calls (above) is lossy.
    return content is None


def _reference_turns(env, example: dict) -> tuple[list[str], bool]:
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
    # keep text-free turns positionally (empty string) so multi-turn replay stays aligned. the second
    # element reports whether any turn lost content in that flattening; see _turn_is_representable.
    partial = any(not _turn_is_representable(m) for m in assistant)
    return [_message_text(m["content"]) for m in assistant], partial


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
    return {
        "policy": "n/a",
        "turns": 0,
        "reward": None,
        "prompt": [],
        "responses": [],
        # True when the gold transcript could not be replayed verbatim (see _turn_is_representable).
        "partial_replay": False,
    }


def _drive_single_turn(env, example: dict, record: dict) -> None:
    prompt = _check_messages(env.prompt_messages(example), "prompt")
    record["prompt"] = prompt
    reference_turns, record["partial_replay"] = _reference_turns(env, example)
    policy = _resolve_policy(reference_turns)
    record["policy"] = policy
    response = (
        "\n".join(turn for turn in reference_turns if turn)
        if policy == "replay"
        else _ECHO_RESPONSE
    )
    # a single-turn gold that spans several assistant messages (a tool-call trajectory, say) is
    # joined above so the preview shows the whole reference, but the real single-turn scorer sees
    # only the last assistant message (flash/multimodal.py assistant_completion_text, called from
    # flash/engine/worker/rl.py). the joined text is therefore not what training would grade, so a
    # grader that correctly accepts only the final answer must not feed the flat-reward gate.
    if policy == "replay" and len([turn for turn in reference_turns if turn]) > 1:
        record["partial_replay"] = True
    record["responses"] = [response]
    record["turns"] = 1
    record["reward"] = float(env.reward(response, example))


def _drive_multi_turn(env, example: dict, record: dict) -> None:
    state = env.new_rollout_state(example)
    record["prompt"] = _check_messages(state.get("prompt") or state.get("messages"), "prompt")
    reference_turns, record["partial_replay"] = _reference_turns(env, example)
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


def _negative_control_reward(env, example: dict) -> float | None:
    """Score a deliberately wrong single-turn answer, or None when that is not meaningful.

    This is the comparison that makes the flat-reward gate scale-independent: a grader is unusable
    for RL when a wrong answer scores the same as the gold one, whatever that number is. Only the
    single-turn path is controlled -- a multi-turn reward reads the accumulated rollout state, and
    swapping one completion string cannot produce a comparable wrong episode.
    """
    if env.multi_turn:
        return None
    try:
        control = float(env.reward(_NEGATIVE_CONTROL, example))
    except (Exception, SystemExit):
        # the grader may reject an off-distribution string outright. that is its prerogative and
        # says nothing about the gold answer, so decline to judge rather than fail the episode.
        return None
    return control if math.isfinite(control) else None


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

    episode_count = min(_DEFAULT_EPISODES, len(dataset))
    passed = 0
    controlled = 0
    scored_flat = 0
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
        if record["policy"] == "replay" and reward is not None and not record["partial_replay"]:
            # the absolute value of a gold reward proves nothing: the contract accepts any finite
            # scalar, so an env may legitimately score its reference 0.0 with worse completions
            # below it. what makes a grader unusable for RL is that it cannot SEPARATE a good
            # completion from a bad one, so score a deliberately wrong answer and compare.
            control = _negative_control_reward(env, example)
            if control is None:
                # no comparable wrong answer to score (multi-turn, or the grader rejected the
                # control outright), so this episode carries no evidence either way. counting it
                # as controlled would let the gate below speak for episodes it never tested.
                continue
            controlled += 1
            if control == reward:
                scored_flat += 1
                message = (
                    f"replay gold answer and a deliberately wrong answer both scored "
                    f"{reward:.6f}; check the reward function"
                )
                print(
                    render.warn(message) if render.styled() else f"warning: {message}",
                    file=sys.stderr,
                )

    print(f"{passed}/{episode_count} episodes passed contract checks")
    if passed != episode_count:
        return _err("overall: FAIL")
    # a grader that hands a deliberately wrong answer the same score as its own gold answer on
    # every sampled episode cannot rank completions at all: a broken reward function or a missing
    # runtime dependency, not a hard dataset. that must fail the gate, since passing here sends a
    # run to a gpu whose advantages are identically zero. episodes whose transcript the driver
    # could not reproduce verbatim, and those whose control could not be scored, are excluded
    # above. this samples the first few rows, so it is deliberately a claim about the sample:
    # separation anywhere in it is enough to pass.
    if controlled and scored_flat == controlled:
        _err(
            f"all {controlled} replayed episode(s) scored a deliberately wrong answer exactly as "
            "high as the gold answer; the reward function cannot rank completions. check the "
            "grader and that its runtime dependencies are installed in this environment."
        )
        return _err("overall: FAIL")
    print(render.ok("overall: PASS") if render.styled() else "overall: PASS")
    return 0

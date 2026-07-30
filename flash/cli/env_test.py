"""Offline contract checks for local Freesolo environments."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from flash.catalog import normalize_algorithm, samples_on_policy

from . import render
from .envpush import _err, _resolve_local_env_entrypoint

_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_ECHO_RESPONSE = "test"
# wrong completions offered to the flat-reward check below as negative controls. deliberately not
# _ECHO_RESPONSE: that string already stands for "there is no gold answer to replay", and reusing
# it would conflate the two. the two single-character fillers are what make the set usable against
# the substring graders this repo ships by default (see _control_is_disjoint): they draw on
# disjoint alphabets, so no single gold answer can occur inside both.
_CONTROL_CANDIDATES = (
    "flash env test negative control: this answer is deliberately wrong.",
    "z" * 64,
    "0" * 64,
)
_PREVIEW_CHARS = 200
_DEFAULT_EPISODES = 3


def _check_messages(messages: object, label: str) -> list[dict]:
    """Validate that `messages` is a well-formed chat message list and return it."""
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{label} is not well-formed: {label} must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"{label} is not well-formed: {label} message {index} must be a dict"
            )
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(
                f"{label} is not well-formed: {label} message {index} "
                "must have a non-empty string role"
            )
        if role.strip().lower() not in _ALLOWED_ROLES:
            raise ValueError(
                f"{label} is not well-formed: {label} message {index} "
                f"has unsupported role {role!r}"
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
        # the gold text of each replayed turn, which a negative control has to be wrong against.
        "reference_turns": [],
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
    # flash/engine/worker/rl.py). compare against exactly that: whenever the joined text differs
    # from what training would grade, the reward below is not the reward the run would compute, so
    # the episode must not feed the flat-reward gate. an equality test rather than a count of
    # non-empty turns, because assistant_completion_text returns a trailing empty content verbatim
    # -- a gold ending in "" would otherwise be graded here as its earlier text and look faithful.
    if policy == "replay" and response != reference_turns[-1]:
        record["partial_replay"] = True
    record["responses"] = [response]
    # what the single-turn scorer actually grades, which is the one text a control must be wrong
    # against. the earlier assistant turns are already excluded from the gate via partial_replay.
    record["reference_turns"] = [response] if policy == "replay" else []
    record["turns"] = 1
    record["reward"] = float(env.reward(response, example))


def _run_rollout(env, example: dict, turn_content) -> tuple[dict, list[str]]:
    """Drive one offline multi-turn rollout, returning the final state and the turns taken.

    `turn_content(index)` supplies the model's text for each turn, so the same loop can replay a
    gold transcript or a deliberately wrong one.
    """
    state = env.new_rollout_state(example)
    # mirror the worker turn loop (flash/engine/multiturn_rollout.py): drive one model
    # turn, then stop at the hard turn ceiling, on the env's own done signal, or when the
    # env yields no reply. the hard cap is fixed at what the trainer passes (env.max_turns)
    # and the turn counter rises every turn until it reaches the cap, so a cooperatively-
    # stepping env terminates here exactly as it would in training; no separate
    # non-termination guard is needed.
    hard_cap = int(env.max_turns)
    responses: list[str] = []
    while True:
        content = turn_content(len(responses))
        responses.append(content)
        env.record_model_turn(state, content)
        if len(responses) >= hard_cap or env.rollout_done(state, max_turns=hard_cap):
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
    return state, responses


def _drive_multi_turn(env, example: dict, record: dict) -> None:
    reference_turns, record["partial_replay"] = _reference_turns(env, example)
    policy = _resolve_policy(reference_turns)
    record["policy"] = policy

    def content(index: int) -> str:
        if policy == "replay" and index < len(reference_turns):
            return reference_turns[index]
        return _ECHO_RESPONSE

    state, responses = _run_rollout(env, example, content)
    record["prompt"] = _check_messages(state.get("prompt") or state.get("messages"), "prompt")
    record["responses"].extend(responses)
    record["turns"] = len(responses)
    record["reference_turns"] = reference_turns
    # the env kept the rollout going past the gold transcript, so the tail of `responses` is echo
    # filler rather than the reference. the reward below then grades a transcript no correct policy
    # would produce, which is not the gold reward and must not feed the flat-reward gate -- the same
    # reason a partially representable transcript is excluded.
    if policy == "replay" and len(responses) != len(reference_turns):
        record["partial_replay"] = True

    record["reward"] = float(env.reward("", example, state))


def _control_is_disjoint(control: str, reference: str) -> bool:
    """Report whether `control` is a usable wrong answer for a gold `reference`.

    The default graders in this repo accept a completion when the gold text occurs anywhere inside
    it (`BaseEnvironment.grade`, and the `exact_match_reward` written by `flash env setup`). A
    control that happens to contain the gold answer is therefore *correct* under those graders, and
    reading equal scores as "this grader cannot rank" would fail a working environment. Require the
    control to contain neither the gold text nor any of its words, in either direction.
    """
    gold = reference.strip().casefold()
    if not gold:
        return False
    lowered = control.casefold()
    if gold in lowered or lowered in gold:
        return False
    return not any(word in lowered for word in gold.split())


def _score_control(env, example: dict, control: str, multi_turn: bool) -> float:
    """Score one deliberately wrong answer the way training would score it.

    A grader that raises here is not inconclusive: GRPO's reward_fn catches exactly that and scores
    the completion 0.0 (flash/engine/worker/rl.py), so training would see a real number. Record that
    same 0.0, otherwise a row that genuinely separates gold from wrong is dropped and a later tied
    row can fail the whole sample alone.

    Raises ValueError when the grader returns a non-finite score, which is the same contract
    violation the gold answer is already failed for: the policy reaches this scorer with completions
    no more expected than these, and NaN or infinity there yields unusable samples.
    """
    try:
        if multi_turn:
            # a multi-turn reward reads the accumulated rollout state, so a comparable wrong
            # episode has to be driven, not assembled -- replay the same loop answering `control`
            # at every turn.
            state, _ = _run_rollout(env, example, lambda _index: control)
            score = float(env.reward("", example, state))
        else:
            score = float(env.reward(control, example))
    except (Exception, SystemExit):
        # mirrors reward_fn's own except branch: the run would score this 0.0 and carry on.
        score = 0.0
    if not math.isfinite(score):
        raise ValueError(f"reward is not finite for a non-reference completion: {score}")
    return score


def _negative_control_rewards(env, example: dict, references: list[str]) -> list[float] | None:
    """Score deliberately wrong answers, or None when that is not meaningful.

    This is the comparison that makes the flat-reward gate scale-independent: a grader is unusable
    for RL when a wrong answer scores the same as the gold one, whatever that number is.

    Every usable control is scored rather than just the first, so an episode counts as separated
    when *any* of them ranks below the gold answer. That keeps a permissive but working grader --
    one that accepts a wrong English sentence for an open-ended task, say -- from being reported as
    unable to rank, since the degenerate controls still fail it.

    Returns None when no control is provably wrong for this example, so the caller can exclude the
    episode instead of drawing a conclusion the evidence does not support.
    """
    usable = [
        control
        for control in _CONTROL_CANDIDATES
        # every gold turn has to be wrong under the control, since a multi-turn reward reads the
        # whole transcript: a control matching any single turn is not a wrong episode.
        if all(_control_is_disjoint(control, reference) for reference in references)
    ]
    scores = [_score_control(env, example, control, env.multi_turn) for control in usable]
    return scores or None


def _load_failure(reason: str) -> int:
    _err(f"env test failed: {reason}")
    print("0/0 episodes passed contract checks")
    return _err("overall: FAIL")


def _grades_completions(args) -> bool:
    """Whether the run this environment is for will actually call ``env.reward()``.

    The SFT worker builds rows from ``dataset()``, ``prompt_messages()`` and ``sft_completion()``
    and never scores anything (``flash/engine/worker/sft.py``), so an SFT-only environment may
    legitimately ship a placeholder scorer that returns one constant. Failing that on reward
    quality would reject a working environment, and this command has no config to infer intent
    from -- hence the explicit flag. Unset means "unknown", which is not grounds to fail; the
    ranking check still reports its finding as a warning.
    """
    algorithm = getattr(args, "algorithm", None)
    if not algorithm or not str(algorithm).strip():
        return False
    return samples_on_policy(normalize_algorithm(str(algorithm).strip()))


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

    grades = _grades_completions(args)

    try:
        from flash.envs.loader import load_freesolo_environment

        # resolve to an absolute path so the loader takes its local-file branch; a bare
        # relative dir like `my-env` matches the managed-slug pattern and would otherwise
        # resolve remotely, breaking the offline contract.
        env = load_freesolo_environment(str(entrypoint.resolve()))
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
    inverted = 0
    for index, example in enumerate(dataset[:episode_count], start=1):
        record = _new_record()
        failure: str | None = None
        controls: list[float] | None = None
        try:
            if env.multi_turn:
                _drive_multi_turn(env, example, record)
            else:
                _drive_single_turn(env, example, record)
            reward = record["reward"]
            if reward is None or not math.isfinite(reward):
                raise ValueError(f"reward is not finite: {reward}")
            if record["policy"] == "replay" and not record["partial_replay"]:
                # the absolute value of a gold reward proves nothing: the contract accepts any
                # finite scalar, so an env may legitimately score its reference 0.0 with worse
                # completions below it. what makes a grader unusable for RL is that it cannot
                # SEPARATE a good completion from a bad one, so score wrong answers and compare.
                # scored inside this guard because a non-finite control breaks the same reward
                # contract as a non-finite gold score and must fail the episode the same way.
                controls = _negative_control_rewards(
                    env, example, record["reference_turns"] or record["responses"][:1]
                )
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
        # controls is None when no wrong answer could be shown to be wrong for this example
        # (multi-turn, or no control disjoint from the gold text), so the episode carries no
        # evidence either way. counting it as controlled would let the gate below speak for
        # episodes it never tested.
        if controls is not None and reward is not None:
            controlled += 1
            if any(control > reward for control in controls):
                # strictly worse than a deliberately wrong answer. GRPO maximizes this number, so
                # the run would train away from the gold answers -- a broken reward direction is
                # worse than a flat one, and no amount of separation elsewhere redeems it.
                inverted += 1
                message = (
                    f"a deliberately wrong answer scored higher ({max(controls):.6f}) than the "
                    f"replayed gold answer ({reward:.6f}); the reward direction looks inverted"
                )
                print(
                    render.warn(message) if render.styled() else f"warning: {message}",
                    file=sys.stderr,
                )
            elif all(control == reward for control in controls):
                scored_flat += 1
                message = (
                    f"replay gold answer and {len(controls)} deliberately wrong answer(s) all "
                    f"scored {reward:.6f}; check the reward function"
                )
                print(
                    render.warn(message) if render.styled() else f"warning: {message}",
                    file=sys.stderr,
                )

    print(f"{passed}/{episode_count} episodes passed contract checks")
    if passed != episode_count:
        return _err("overall: FAIL")
    # a grader that hands every deliberately wrong answer the same score as its own gold answer on
    # every sampled episode cannot rank completions at all: a broken reward function or a missing
    # runtime dependency, not a hard dataset. one that ranks them ABOVE gold is worse still, since
    # GRPO would maximize its way off the references. both send a run to a gpu that cannot learn
    # what the dataset teaches. episodes whose transcript the driver could not reproduce verbatim,
    # and those with no control provably wrong for their gold answer, are excluded above. this
    # samples the first few rows, so the flat finding is deliberately a claim about the sample:
    # separation anywhere in it is enough to pass.
    #
    # only failed for an algorithm that actually consumes reward(). SFT never calls it, so a
    # placeholder scorer there is not a defect, and without --algorithm the intent is unknown --
    # report the finding without failing rather than block a working environment.
    finding = ""
    if inverted:
        finding = (
            f"{inverted} replayed episode(s) scored a deliberately wrong answer higher than the "
            "gold answer; the reward direction is inverted and training would move away from the "
            "references. check the grader's sign."
        )
    elif controlled and scored_flat == controlled:
        finding = (
            f"all {controlled} replayed episode(s) scored every deliberately wrong answer exactly "
            "as high as the gold answer; the reward function cannot rank completions. check the "
            "grader and that its runtime dependencies are installed in this environment."
        )
    if finding and grades:
        _err(finding)
        return _err("overall: FAIL")
    if finding:
        message = f"{finding} pass --algorithm to fail on this instead of warning."
        print(
            render.warn(message) if render.styled() else f"warning: {message}",
            file=sys.stderr,
        )
    print(render.ok("overall: PASS") if render.styled() else "overall: PASS")
    return 0

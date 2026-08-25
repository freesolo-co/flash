"""Offline contract checks for local Freesolo environments."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from flash.cli.commands.env.ops.push import _err, _resolve_local_env_entrypoint
from flash.cli.commands.env.testing.episode import _effective_turn_cap
from flash.cli.commands.env.testing.evaluations import (
    _check_evaluation_suites,
    _normalize_prompt_images,
)
from flash.cli.commands.env.testing.params import _env_params
from flash.cli.commands.env.testing.warnings import (
    _warn_on_low_replay_reward,
    _warn_on_repeated_rendered_roles,
    _warn_on_unfinished_replay,
    _warn_on_uniformly_zero_rewards,
)
from flash.cli.ui import render

_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_ECHO_RESPONSE = "test"
# the one algorithm that trains from the environment reward. sft optimizes a supervised loss and opd
# a teacher token loss; neither reads `env.reward` anywhere (flash/engine/worker/entry/sft.py, opd.py), so
# a scorer they never call cannot be evidence of anything for them.
_REWARD_DRIVEN_ALGORITHM = "grpo"
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
    # the canonical definition, so replay text and the reward path's own extraction cannot drift.
    from flash.content.multimodal import message_content_text

    return message_content_text(content)


def _gold_completion(env, example: dict) -> list[dict]:
    """The env's gold completion messages, envelope-checked.

    Separate from `_reference_turns` so one episode calls `sft_completion` -- user code -- exactly
    once and both the replay text and the rendered-role check below read the same messages. Validate
    the envelope like the prompt so a malformed completion (scalar content, missing role) fails the
    episode instead of silently falling back to echo.
    """
    return _check_messages(env.sft_completion(example), "sft_completion")


def _reference_turns(messages: list[dict]) -> list[str]:
    # the sft_completion gold answer stands in for the missing policy model. text is extracted the
    # same way the real reward path grades a completion, so a gold answer expressed as
    # openai-style text blocks is replayed instead of echoed; text-free turns (null content
    # or image-only blocks) yield an empty replay string that is kept in place.
    #
    # only assistant turns stand in for the policy model; a gold completion with no assistant
    # message must NOT replay user/system text as the model response -- yield no replay text so
    # _resolve_policy falls back to echo.
    assistant = [m for m in messages if m["role"].strip().lower() == "assistant"]
    # assistant turns only (a gold with no assistant message must echo, not replay user/system);
    # keep text-free turns positionally (empty string) so multi-turn replay stays aligned.
    return [_message_text(m["content"]) for m in assistant]


def _resolve_policy(reference_turns: list[str]) -> str:
    return "replay" if "".join(reference_turns).strip() else "echo"


def _gold_lacks_replayable_text(completion: list[dict], reference_turns: list[str]) -> bool:
    """Whether a text-free gold completion is a real target rather than an absent answer.

    Both echo the same way, so the policy alone cannot separate them, but the remedies are
    opposite. What separates them is the PAYLOAD: a native tool call leaves `content` null because
    the arguments live in `tool_calls`, and that row is a correct SFT target this command simply
    cannot send. A row with no answer -- `output: ""` -- renders as an assistant turn with content
    `""`, which is absence, not a payload elsewhere.

    Keyed on a non-`content` payload rather than on null content alone: `content: null` with
    nothing beside it is an empty turn, not a tool call. Image-only content never reaches here --
    `freesolo.datasets.target_messages` rejects a non-string, non-null content before this runs --
    so it is deliberately not claimed as a case.
    """
    if not completion or "".join(reference_turns).strip():
        return False
    return any(
        message["role"].strip().lower() == "assistant"
        and message.get("content") is None
        and any(key not in ("role", "content") and message[key] for key in message)
        for message in completion
    )


def _junk_response(reference_turns: list[str]) -> str:
    """A deliberately wrong answer, guaranteed to differ from this row's gold one.

    The control is only evidence while it is wrong. `_ECHO_RESPONSE` is the fixed string `test`,
    and a reference answer that happens to be `test` collides with it, so the probe below fed the
    grader the correct answer and read back the gold reward -- making a scorer that separates
    perfectly look like one that pays junk as much as gold, and failing a working environment
    Lengthening past every reference turn terminates: the turns are finite and each pass
    grows the string. Compared stripped, because a control differing from gold only in whitespace
    is one a grader may well still score as correct."""
    gold = {turn.strip() for turn in reference_turns}
    junk = _ECHO_RESPONSE
    while junk.strip() in gold:
        junk = f"not {junk}"
    return junk


def _carries_thinking_markup(reference_turns: list[str]) -> bool:
    """Whether a gold answer is written in reasoning markup this command cannot reproduce.

    A thinking run grades what `_scored_turn_text` leaves behind, which strips the `<think>` span
    before the reward ever sees it (flash/envs/loading/adapter.py). This command has no run config to read
    `thinking` from -- it builds the environment locally, where it defaults off (adapter.py) -- so
    it replays the tagged reference verbatim. Against a strict answer-only grader every reference
    then scores zero, and the gate below reported a working environment as unable to recognize its
    own gold answers. The reward is still printed and still warned about; only the
    blocking conclusion is withheld, because the evidence for it cannot be produced here."""
    return any("<think>" in turn or "</think>" in turn for turn in reference_turns)


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


def _report_scorer_error(record: dict) -> None:
    """Print the scorer's own error, for any policy, when it reported one.

    A scorer that crashed and a scorer that judged are both reported as 0.0 by
    ``FreesoloEnvironment.reward``, which keeps only ``RewardResult.score``. Surfacing the
    discarded ``error`` names a missing dependency instantly. This is deliberately not limited to
    replayed episodes or to low rewards: an ``echo`` episode drives no gold answer and so leaves
    the grpo gate with nothing to count, which is exactly when a silent crash would otherwise
    reach ``overall: PASS`` unreported.
    """
    scorer_error = record.get("scorer_error") or ""
    if not scorer_error:
        return
    print(f"  scorer error: {scorer_error}", file=sys.stderr)


def _new_record() -> dict:
    """Mutable per-episode record so a failing episode still reports real progress."""
    return {
        "policy": "n/a",
        "turns": 0,
        "reward": None,
        "prompt": [],
        "responses": [],
        "thinking_markup": False,
        "replay_incomplete": False,
        # the scorer's own `RewardResult.error` from the call that produced `reward`, captured
        # there rather than re-scored later: a non-deterministic scorer (a rate-limited judge, a
        # flaky dependency) can return a different result the second time, so a re-score can miss
        # the error that actually produced this reward -- and would charge a paid judge twice.
        "scorer_error": "",
        # the text the grader actually received, captured at the scoring call. not derivable from
        # `responses`: a multi-turn env whose `step_episode` returns `final_response_text` has the
        # adapter replace `state["response_text"]` with that override, and the scorer is fed the
        # replacement -- so printing the replayed turns would label text that was never scored.
        # None, not "": an empty string is a real graded value (an env that overrode the answer to
        # nothing, a text-free final turn) and the case a reader most needs named, so it must not
        # read as "never captured" and fall back to the replayed turns.
        "scored_text": None,
        # the multi-turn rollout state, kept so the per-turn check below can ask about the episode
        # that was actually scored rather than a fresh one. None for single-turn, which has no state
        # and no per-turn vector either.
        "state": None,
        # whether the episode stopped because it ran out of turns rather than because the env said
        # it was done. a gold answer that cannot finish inside the cap is the signature of an env
        # no trajectory can complete, and the scalar reward alone never shows it.
        "hit_turn_cap": False,
        # whether the gold trajectory is LONGER than the effective turn cap, so only a prefix could
        # be replayed. a cap/dataset mismatch, not an environment defect, and the only unfinished
        # shape here whose cause is fully determined by the inputs.
        "gold_exceeds_cap": False,
        # the effective ceiling and the reference length, kept so the mismatch above can name both
        # numbers -- an author cannot act on "raise the cap" without knowing what to raise it to.
        "turn_cap": 0,
        "reference_turns": 0,
        # the gold completion messages as returned, kept so the rendered-role check can inspect the
        # concatenation SFT would train on rather than the per-turn replay this command scores.
        "completion": [],
        # whether a gold completion existed but carried no text this command could replay -- an
        # assistant turn whose content is null because the payload is in `tool_calls`. such a row
        # falls back to echo exactly like a row with no gold at all, and the two are
        # indistinguishable from the policy alone, but the remedy is opposite: one author must
        # supply an answer, the other already did and must not be told to overwrite it with text.
        "gold_without_replayable_text": False,
    }


def _drive_single_turn(env, example: dict, record: dict, *, force_echo: bool = False) -> None:
    prompt = _check_messages(env.prompt_messages(example), "prompt")
    record["prompt"] = prompt
    completion = _gold_completion(env, example)
    record["completion"] = completion
    reference_turns = _reference_turns(completion)
    policy = "echo" if force_echo else _resolve_policy(reference_turns)
    record["policy"] = policy
    # not `policy == "echo"`: under `force_echo` the junk probe throws a perfectly good reference
    # away on purpose, and recording that as a text-free gold would misreport the row.
    record["gold_without_replayable_text"] = not force_echo and _gold_lacks_replayable_text(
        completion, reference_turns
    )
    record["thinking_markup"] = _carries_thinking_markup(reference_turns)
    response = (
        "\n".join(turn for turn in reference_turns if turn)
        if policy == "replay"
        else _junk_response(reference_turns)
    )
    record["responses"] = [response]
    record["turns"] = 1
    record["reward"], record["scorer_error"], record["scored_text"] = _score_with_error(
        env, response, example
    )


def _new_multi_turn_replay_state(env, example: dict, record: dict) -> dict:
    state = env.new_rollout_state(example)
    # `prompt` only, never `messages`: the two are not spellings of one field. `new_rollout_state`
    # seeds `messages` with a COPY of `prompt` and appends each turn onto it, so falling back to it
    # records the growing transcript where the frozen initial prefix belongs. every producer sets
    # both, so an absent `prompt` is a corrupt state and `_check_messages` rejects it by name.
    record["prompt"] = _check_messages(state.get("prompt"), "prompt")
    _normalize_prompt_images(env, example, record["prompt"])
    return state


def _validate_multi_turn_reply(env, example: dict, state: dict, messages: object) -> None:
    _check_messages(messages, "env_reply")
    _normalize_prompt_images(env, example, [dict(message) for message in state["messages"]])


def _drive_multi_turn(env, example: dict, record: dict, *, force_echo: bool = False) -> None:
    state = _new_multi_turn_replay_state(env, example, record)
    completion = _gold_completion(env, example)
    record["completion"] = completion
    reference_turns = _reference_turns(completion)
    policy = "echo" if force_echo else _resolve_policy(reference_turns)
    record["policy"] = policy
    record["gold_without_replayable_text"] = not force_echo and _gold_lacks_replayable_text(
        completion, reference_turns
    )
    record["thinking_markup"] = _carries_thinking_markup(reference_turns)
    # mirror the worker turn loop (flash/engine/worker/train/rl/child/multiturn.py): drive one model
    # turn, then stop at the hard turn ceiling, on the env's own done signal, or when the
    # env yields no reply. the turn counter rises every turn until it reaches the cap, so a
    # cooperatively-stepping env terminates here exactly as it would in training; no separate
    # non-termination guard is needed.
    #
    # the ceiling is the EFFECTIVE one, not `env.max_turns`: a row that sets `max_episode_turns`
    # below the dataset-wide cap is stopped by its own budget, and `rollout_done` gives that budget
    # precedence (flash/envs/loading/adapter.py). comparing the turn count against `env.max_turns` alone
    # left `stopped_at_ceiling` False for exactly those episodes, so a dead environment held under a
    # short per-example cap reached `overall: PASS` with no warning -- the silent pass this check
    # exists to catch. shared with `flash env eval`, which derives the same limit for the same
    # reason, so the two commands cannot drift.
    hard_cap = _effective_turn_cap(env, state)
    turns = 0
    # mirrors the worker's own flag: True while the newest turn has not been through env_reply.
    env_step_pending = False
    # whether the loop ended at the trainer's ceiling rather than on the env's own signal. this is
    # only HALF the verdict: the last model turn has not been applied yet at that point (see the
    # deferred env_reply below), so an env that finishes exactly on its last allowed turn still
    # looks unfinished here. the conclusion is drawn after that turn is applied.
    stopped_at_ceiling = False
    # whether the env had declared the episode over at the moment the gold answer ran out, sampled
    # mid-loop because that is the only point where the reference can be judged on its own. None
    # while the gold answer is still being replayed.
    gold_finished: bool | None = None
    while True:
        if policy == "replay" and turns < len(reference_turns):
            content = reference_turns[turns]
        else:
            # a reference shorter than the episode is padded so the rollout still reaches its own
            # termination, but the trajectory graded at the end is then part gold and part junk.
            # scoring it is not evidence about whether the grader recognizes its references, so the
            # flag keeps it out of the blocking gate's totals.
            if policy == "replay":
                record["replay_incomplete"] = True
            content = _junk_response(reference_turns)
        record["responses"].append(content)
        env.record_model_turn(state, content)
        env_step_pending = True
        turns += 1
        record["turns"] = turns
        if turns >= hard_cap or env.rollout_done(state, max_turns=hard_cap):
            # only the ceiling counts as being cut off. an env that declared itself done, or one
            # that ended by yielding no reply below, finished on its own terms.
            stopped_at_ceiling = turns >= hard_cap
            break
        env_msgs = env.env_reply(state["messages"], state)
        env_step_pending = False
        if gold_finished is None and turns >= len(reference_turns):
            # the gold answer has just run out and its last turn is applied: this is the only
            # moment the reference can be judged on its own, before junk padding touches the state.
            gold_finished = _episode_completed(env, state, hard_cap)
        if not env_msgs:
            break
        if env.rollout_done(state, max_turns=hard_cap):
            break
        # the env's own reply messages feed the chat template for the next turn in the real
        # rollout, so validate their envelope and accumulated images here too: a malformed reply
        # that would break remotely must fail the episode instead of slipping through on a finite
        # reward. normalize copies so the check cannot replace authoritative rollout messages.
        _validate_multi_turn_reply(env, example, state, env_msgs)

    # the driver-side exits above stop before the inter-turn env_reply, so the last replayed turn
    # is still unapplied. a stateful env would then score a board or transcript missing the last
    # thing the model did, so give it that turn before scoring -- only the inter-turn glue is
    # skipped, since no further model turn is conditioned on the reply. this mirrors the worker
    # loops (opd_train / rl_train), which check termination before requesting an env_reply for the
    # same reason: this command exists to catch a contract break before a paid run does, and it can
    # only do that while it scores the state the run would score.
    #
    # env_step_pending is False once the last generated turn has already been through env_reply (an
    # env that replied with nothing, a natural finish) -- stepping again there would be a spurious
    # extra move. rollout_done covers the env having declared the episode over.
    if env_step_pending and not env.rollout_done(state, hard_cap):
        # apply the final action's side effects, but do not validate its reply as a future prompt:
        # no later model generation consumes it in the online path.
        env.env_reply(state["messages"], state)
    if gold_finished is None:
        # the gold answer never ran out inside the loop -- it covered the whole episode, so the
        # break came first and the mid-loop sample never fired. NOW is its moment: the deferred
        # env_reply above has applied its last turn, which is precisely the state that says whether
        # a full-length reference completed the episode or merely exhausted the budget.
        gold_finished = _episode_completed(env, state, hard_cap)
    # decided HERE, not at the break: the last model turn is applied only by the env_reply above, so
    # asking before it would report an env that solves the task on its final allowed turn as one
    # that never finishes -- and send the author to fix termination logic that works.
    #
    # a short gold answer is judged at the moment it ran out (`gold_finished`) rather than at the
    # end. the driver pads the tail with junk, and junk cannot advance the env, so the final state
    # says nothing about the reference either way. sampling mid-loop keeps BOTH halves reportable: a
    # five-move gold answer against a twelve-turn cap on an environment no move can ever solve --
    # the exact field case this warning exists for -- is still reported, and so is a short gold
    # answer on a healthy env, because at that sampling point the two are INDISTINGUISHABLE: both
    # are not-done, one because it never can be and one because a real rollout would simply have
    # continued. that ambiguity is not resolvable here, so it is not resolved here -- it is stated.
    # `replay_incomplete` selects which of the two claims the warning is allowed to make.
    #
    # `rollout_done` deliberately is NOT consulted here. the real adapter
    # (flash/envs/loading/adapter.py `rollout_done`) returns True from `turn >= cap` ALONE, so at the ceiling
    # it is True for a healthy env and a dead one alike -- ANDing it in made this warning
    # unsatisfiable in production while the fakes, which override it to a bare `False`, kept the
    # tests green. what distinguishes the two is the env's own completion signal, which is what
    # `_episode_completed` reads.
    unfinished = gold_finished is False if gold_finished is not None else True
    record["hit_turn_cap"] = stopped_at_ceiling and unfinished
    # a reference LONGER than the cap is the one case here that is not ambiguous at all. the driver
    # replays a prefix and stops, so `replay_incomplete` never gets set (that flag means the driver
    # ran PAST the reference and padded), and the episode lands in the branch that blames the
    # environment. but the dataset itself states the trajectory needs more turns than the cap
    # allows, so the environment's termination logic was never reached, let alone shown to be
    # broken. the mismatch is checkable from these two numbers alone, and its remedy is exact --
    # raise the cap or shorten the row -- so it is reported as itself rather than as a guess
    # between two other explanations.
    #
    # deliberately NOT gated on `stopped_at_ceiling`: the mismatch is a property of the two numbers
    # alone. an env that happens to declare itself done while the capped PREFIX is being applied
    # leaves `gold_finished` true and `hit_turn_cap` false, but the trailing gold turns were still
    # dropped and never replayed -- silently, since every other check passes. whether the prefix
    # terminated the episode says nothing about whether the row and the cap agree.
    record["gold_exceeds_cap"] = policy == "replay" and len(reference_turns) > hard_cap
    record["turn_cap"] = hard_cap
    record["reference_turns"] = len(reference_turns)
    record["reward"], record["scorer_error"], record["scored_text"] = _score_with_error(
        env, "", example, state
    )
    record["state"] = state


def _episode_completed(env, state: dict, hard_cap: int) -> bool:
    """Whether the ENVIRONMENT declared this episode over, as opposed to running out of turns.

    `rollout_done` cannot answer this. The real adapter (flash/envs/loading/adapter.py) returns True from
    `turn >= cap` alone, so at the ceiling it is True for a healthy environment and for one no
    rollout can ever finish -- which is exactly the distinction the turn-cap warning is built on.

    The env's own completion flag is the signal that separates them, read from the state the adapter
    itself sets (`state["done"]`). An env that exposes no such flag is treated as finished, so a
    non-standard state shape produces silence rather than a warning nobody can act on.
    """
    if "done" in state:
        return bool(state.get("done"))
    # no completion flag to read: fall back to the env's own predicate, but only below the ceiling,
    # where it still reflects a real decision rather than the cap.
    turn = int(state.get("turn", 0) or 0)
    if turn >= hard_cap:
        return True
    try:
        return bool(env.rollout_done(state, max_turns=hard_cap))
    except (Exception, SystemExit):
        return True


def _junk_reward(env, example: dict) -> float | None:
    """What a deliberately wrong answer scores, or None if the probe could not produce a number.

    One stateful-env pass, run only after every real episode has been scored. Callers share this
    single result rather than each driving their own: scoring is not guaranteed to be pure, a
    second pass can bill a paid judge twice, and two probes of the same env could disagree.

    """
    try:
        probe = _new_record()
        if env.multi_turn:
            _drive_multi_turn(env, example, probe, force_echo=True)
        else:
            _drive_single_turn(env, example, probe, force_echo=True)
        junk_reward = probe["reward"]
    except (Exception, SystemExit):
        return None
    if junk_reward is None or not math.isfinite(junk_reward):
        return None
    return junk_reward


def _scores_gold_no_better_than_junk(junk_reward: float | None, gold_reward: float) -> bool:
    """Whether junk scores at least as well as this gold answer.

    A gold score of zero can still beat negative junk, so compare them directly. A probe error is
    not evidence of flat reward because an env may require parseable answers.
    """
    if junk_reward is None:
        return False
    return junk_reward >= gold_reward


def _score_with_error(
    env, completion: str, example: dict, state: dict | None = None
) -> tuple[float, str, str]:
    """Score one completion, returning ``(reward, scorer_error, scored_text)`` from one call.

    ``FreesoloEnvironment.reward`` returns ``float(result.score)``, so a scorer that crashed behind
    the SDK's guard and one that deliberately scored zero are indistinguishable by reward alone.
    The error is read from the same call that produced the reward rather than a second one:
    scoring is not guaranteed to be pure, so a re-score can report an error that did not produce
    this reward -- and would bill a paid judge twice per episode. Envs without the richer hook
    (anything not backed by the Freesolo adapter) fall back to the plain reward, no error, and the
    completion as passed -- which is what such an env grades, having no episode override path.
    """
    reward_with_error = getattr(env, "reward_with_error", None)
    if callable(reward_with_error):
        reward, error, scored = reward_with_error(completion, example, state)
        # `scored` is not coerced through `or ""`: an empty graded answer is a real value, and the
        # caller distinguishes "" (captured, empty) from None (never scored) to decide whether to
        # fall back to the replayed turns.
        return float(reward), str(error or ""), str(scored)
    return float(env.reward(completion, example, state)), "", completion


def _separates_on_turn_rewards(env, example: dict, state: dict | None, turn_count: int) -> bool:
    """Whether per-turn rewards separate an episode whose scalar does not.

    ``credit_assignment = "per_turn"`` trains from metadata returned through
    ``rollout_rewards_many`` (flash/envs/loading/adapter.py), not ``env.reward``. Distinct turn values prove
    separation; missing, invalid, or unavailable vectors leave the scalar authoritative.

    A vector is only evidence if the PAID run would actually use it, so the checks here mirror
    `_validated_reward` (flash/engine/worker/train/rl/rollout/scoring.py) exactly: a non-number, a count
    that disagrees with the assistant turns emitted, or a non-finite value all make the worker
    discard the vector and fall back to the episode reward. Accepting one this command would not --
    two rewards for three turns, a NaN -- claimed separation on a run that trains from the flat
    scalar after all, which then suppressed both the all-zero warning and the blocking gate and let
    a zero-advantage run print `overall: PASS`. Whatever the worker refuses, this refuses.
    """
    rollout_rewards_many = getattr(env, "rollout_rewards_many", None)
    if not callable(rollout_rewards_many):
        return False
    try:
        rewards = rollout_rewards_many([(example, state or {})])
    except (Exception, SystemExit):
        return False
    if not rewards:
        return False
    turns = getattr(rewards[0], "turns", None)
    if not turns:
        return False
    try:
        coerced = tuple(float(value) for value in turns)
    except (TypeError, ValueError):
        return False
    if len(coerced) != turn_count or not all(math.isfinite(value) for value in coerced):
        return False
    return len(set(coerced)) > 1


@dataclass
class _Tally:
    """What the episode loop measured, as the two gates below need to read it.

    One value rather than five loose locals because the gates disagree about which episodes count,
    and the distinction is the whole subtlety: `replayed`/`replayed_zero` are the BLOCKING gate's
    evidence and count only what it can hold responsible, while `interpretable_rewards` and
    `scored_episode` are the advisory warning's and span every algorithm and policy.
    """

    # only replay episodes carry a gold answer to score. an echo episode has none, so its reward
    # says nothing about the grader and is counted in neither total.
    #
    # a probe subject is `(example, state, turns)`. the turn count rides along because the per-turn
    # probe has to check a returned vector's length against the assistant turns THAT episode emitted,
    # exactly as the paid worker does -- see `_separates_on_turn_rewards`.
    replayed: int = 0
    replayed_zero: list[tuple[dict, dict | None, int]] = field(default_factory=list)
    # whether any episode replayed a gold answer at all, including the ones the blocking gate
    # excludes (reasoning markup, an incomplete replay). an all-echo run scored only deliberate
    # junk, for which zero is the CORRECT answer, so the warning must not read that as a broken
    # grader -- but it is still worth saying that nothing was measured.
    replayed_any: bool = False
    # whether any echo episode was chosen because its gold answer carried no replayable TEXT rather
    # than because no gold answer existed. changes only the remedy the all-echo warning gives: a
    # native tool-call target is already correct and must not be told to add assistant text.
    gold_without_replayable_text: bool = False
    # the first episode that produced a finite reward, whatever its policy. used only as the subject
    # of the advisory warning's separation probe.
    scored_episode: tuple[dict, dict | None, int] | None = None
    # every finite reward whose value says something about the ENVIRONMENT, across all policies and
    # algorithms. the blocking gate counts only the replay episodes it can hold responsible and
    # abstains for a non-grpo algorithm or a junk probe that raised, so a run where nothing scored
    # anything reached `overall: PASS` with no line saying so; this list is what makes the
    # uniformly-zero warning independent of those abstentions. it is NOT independent of the two
    # abstentions that are about this command's own fidelity -- see `observe`.
    interpretable_rewards: list[float] = field(default_factory=list)

    def observe(self, example: dict, record: dict) -> None:
        """Record one episode that passed its contract checks."""
        reward = record["reward"]
        if reward is None:
            return
        if self.scored_episode is None:
            # a probe subject for the advisory warning, kept independently of `replayed_zero`. that
            # list is the BLOCKING gate's evidence and deliberately excludes a thinking-markup or
            # incomplete replay -- but a per-turn vector on such an episode still proves the grader
            # separates, so the warning needs a subject the gate skips.
            self.scored_episode = (example, record["state"], record["turns"])
        # an ECHO episode's reward IS interpretable: zero is the correct score for deliberate junk,
        # and the warning has separate, weaker wording for a run that scored only junk. recorded
        # here because the replay-only bookkeeping below does not apply to it.
        if record["policy"] != "replay":
            self.interpretable_rewards.append(reward)
            self.gold_without_replayable_text |= record["gold_without_replayable_text"]
            return
        self.replayed_any = True
        # a reference written in reasoning markup cannot be replayed faithfully from here (see
        # `_carries_thinking_markup`), so its score is not evidence about the grader and is kept out
        # of the blocking gate's totals. neither is a multi-turn episode whose gold answer ran out
        # before the rollout did: the graded trajectory is then part reference and part junk, and a
        # zero says nothing about the reference the gate never ran.
        #
        # excluded from the ADVISORY warning's evidence too, not only the gate's. both zeros are
        # artifacts of replaying from HERE rather than statements about the environment, so counting
        # them made the warning say "this run measured nothing" about a working reasoning env -- the
        # very conclusion the gate abstains from. the per-episode low-reward warning still prints
        # the number, which is the part that is true.
        if record["thinking_markup"] or record["replay_incomplete"]:
            return
        self.interpretable_rewards.append(reward)
        self.replayed += 1
        # an exact zero is what a scorer that recognized nothing returns, so it stays the signature
        # this gate looks for. it is confirmed against a wrong answer once every episode has run,
        # not here: see `_scores_gold_no_better_than_junk`.
        if reward == 0.0:
            self.replayed_zero.append((example, record["state"], record["turns"]))


def _check_grader(env, algorithm: str, tally: _Tally) -> bool:
    """Decide LS-005 and report whatever the run failed to measure.

    Returns whether the grader was shown to recognize its own reference answers. Blocks only when
    every gold answer scores zero and deliberate junk scores at least as well; centered rewards may
    legitimately score gold at zero, and partial failures remain warnings. The gate applies only to
    grpo, which trains from env.reward -- sft and opd use other losses.
    """
    # the blocking gate's own condition, unchanged: every episode the gate can hold responsible
    # scored zero. deliberately NOT widened with "and no other episode scored anything" -- an echo
    # row or a thinking-markup row that happens to pay 0.5 says nothing about whether the grader
    # recognizes its references, and letting one disable the block would turn a broken grader's
    # verdict from FAIL back into PASS.
    all_accountable_gold_scored_zero = bool(tally.replayed) and (
        len(tally.replayed_zero) == tally.replayed
    )
    # read from `interpretable_rewards`, not every score: a run whose only rewards came from replays
    # this command cannot reproduce faithfully has not shown that the environment measures nothing,
    # only that THIS COMMAND could not measure it. saying so anyway reported a working reasoning
    # environment as unmeasured on every run.
    uniformly_zero = bool(tally.interpretable_rewards) and all(
        reward == 0.0 for reward in tally.interpretable_rewards
    )
    # the probe's subject: the blocking gate's own episode when it has one, otherwise any episode
    # that scored. the fallback matters -- a gold answer in reasoning markup keeps `replayed_zero`
    # empty, and without a subject the warning could gather no evidence and fired on graders that
    # separate perfectly.
    probe_subject = (
        tally.replayed_zero[0] if all_accountable_gold_scored_zero else tally.scored_episode
    )
    # never probe a run that replayed no gold answer at all: every episode was already `echo`, which
    # IS the junk answer, so a junk probe would re-score what was just scored and prove nothing.
    blocking_gate_asks = all_accountable_gold_scored_zero and algorithm == _REWARD_DRIVEN_ALGORITHM
    probe_worth_running = (
        probe_subject is not None and tally.replayed_any and (blocking_gate_asks or uniformly_zero)
    )
    # NOT free, and NOT unconditional: `rollout_rewards_many` runs the environment's own
    # `score_episodes` (flash/envs/loading/adapter.py), which may be the same paid judge the episodes just
    # used. `probe_worth_running` alone is too loose a rule to spend it under, because
    # `uniformly_zero` is algorithm-independent: an all-zero `--algorithm sft` run reached both
    # probes where `dev` reached neither, billing a counting grader 3 times against dev's 1.
    #
    # what the extra spend would buy is `separates`, and `separates` only ever suppresses a WARNING.
    # so it is worth a paid call exactly when the answer could change what this command prints AND
    # the run is one whose reward drives training. for sft and opd the reward is not read at all
    # (`_REWARD_DRIVEN_ALGORITHM`), so the warning's weaker wording is the honest output either way
    # and the judge is left alone. a healthy environment reaches neither probe under any algorithm
    # and is billed exactly what it was before this change (measured: 1 -> 1, sft and grpo).
    separates_on_turns = (
        probe_worth_running
        and algorithm == _REWARD_DRIVEN_ALGORITHM
        and _separates_on_turn_rewards(env, *probe_subject)
    )
    # the junk probe drives a whole extra episode through user code and can bill a paid judge, so it
    # is spent only where its answer changes an outcome: the blocking gate asking (grpo, accountable
    # gold all zero), or the advisory warning about to fire on a run whose gold answers all scored
    # zero. the second condition is `all_accountable_gold_scored_zero`, NOT merely `uniformly_zero`:
    # a run whose accountable gold was never zero has nothing for the probe to compare against, and
    # driving it there bought an unusable number at the cost of an extra billed episode.
    #
    # bounded by algorithm for the same reason as the per-turn probe above: for sft and opd its
    # answer can only soften a warning about a reward those algorithms never read, which is not
    # worth an extra billed episode through user code.
    junk_reward = (
        _junk_reward(env, probe_subject[0])
        if probe_worth_running
        and not separates_on_turns
        and all_accountable_gold_scored_zero
        and algorithm == _REWARD_DRIVEN_ALGORITHM
        else None
    )
    if (
        algorithm == _REWARD_DRIVEN_ALGORITHM
        and all_accountable_gold_scored_zero
        and not separates_on_turns
        and _scores_gold_no_better_than_junk(junk_reward, 0.0)
    ):
        _err(
            f"all {tally.replayed} replayed gold answer(s) scored zero, no better than a "
            "deliberately wrong answer; the reward function cannot recognize its own reference "
            "answers. check the grader and that its runtime dependencies are installed in this "
            "environment."
        )
        return False
    # the blocking gate abstains far more often than it fires: for sft and opd, for a reference in
    # reasoning markup, for a replay that ran out mid-episode, for an echo run with no gold answer at
    # all, and whenever the junk probe raises. in every one of those a run that scored exactly zero
    # on every episode still printed `overall: PASS` and nothing else -- and a uniformly-zero reward
    # is never a measurement, whatever produced it. GRPO centres rewards within each group, so a
    # constant reward is a zero advantage and a zero gradient: training completes, the loss curve
    # looks unremarkable, and the adapter comes out identical to its warm start. say so here rather
    # than leave the verdict to speak for it.
    #
    # but only when the grader has NOT already been shown to separate. a centered scale that pays
    # gold 0.0 and junk -1.0 has a perfectly good gradient, and the same is true of an env that
    # trains from a per-turn vector while its scalar is a placeholder -- both are the shapes the
    # blocking gate deliberately exempts, and warning about them would send the author of a working
    # environment to debug a correct reward function. a false alarm on the healthy path is what
    # teaches people to ignore this warning on the broken path. separation is positive EVIDENCE, not
    # merely the gate's abstention: a per-turn vector whose values differ, or junk scoring strictly
    # below the gold zero. a probe that failed leaves `junk_reward` None, which proves nothing and so
    # does not excuse the run.
    _warn_on_uniformly_zero_rewards(
        tally.interpretable_rewards,
        algorithm=algorithm,
        replayed_any=tally.replayed_any,
        gold_without_replayable_text=tally.gold_without_replayable_text,
        separates=separates_on_turns or (junk_reward is not None and junk_reward < 0.0),
    )
    return True


def _load_failure(reason: str) -> int:
    _err(f"env test failed: {reason}")
    print("0/0 episodes passed contract checks")
    return _err("overall: FAIL")


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
        from flash.envs.loading.loader import load_freesolo_environment

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
    tally = _Tally()
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
        tally.observe(example, record)
        if record["policy"] == "replay" and reward is not None:
            _warn_on_low_replay_reward(record, reward)
            _warn_on_unfinished_replay(record)
        # outside the replay branch above: an `echo` episode has no gold answer to blame, but its
        # scorer can still crash, and `replayed` is then zero so the grpo gate below is disabled
        # too. reporting nothing there let `flash env test` exit PASS while every score came from a
        # scorer that never ran.
        _report_scorer_error(record)
        # outside the replay branch too, and reported per episode rather than once: `sft_completion`
        # is per-row user code, so one row may interleave its turns correctly while another does
        # not, and an `echo` episode -- whose gold answer holds no assistant turn at all -- can
        # still render a broken role sequence.
        _warn_on_repeated_rendered_roles(
            record["prompt"], record["completion"], multi_turn=bool(env.multi_turn)
        )

    print(f"{passed}/{episode_count} episodes passed contract checks")
    if passed != episode_count:
        return _err("overall: FAIL")
    # default to grpo so omitting the algorithm cannot disable the blocking check.
    algorithm = getattr(args, "algorithm", None) or _REWARD_DRIVEN_ALGORITHM
    grader_recognizes_gold = _check_grader(env, algorithm, tally)
    # run the evaluation checks even when the grader gate already failed, so one `flash env test`
    # reports every broken surface at once instead of hiding the sidecar's errors behind the
    # reward function's. both gates block; neither short-circuits the other.
    if not _check_evaluation_suites(entrypoint, env) or not grader_recognizes_gold:
        return _err("overall: FAIL")
    print(render.ok("overall: PASS") if render.styled() else "overall: PASS")
    return 0

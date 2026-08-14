"""Offline contract checks for local Freesolo environments."""

from __future__ import annotations

import json
import math
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from flash.cli.commands.env.episode import _effective_turn_cap
from flash.cli.commands.env.push import _err, _resolve_local_env_entrypoint
from flash.cli.commands.env.test_evaluations import _check_evaluation_suites
from flash.cli.commands.env.test_warnings import (
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
# characters that give a TOML value structure. text containing none of them is a bare string, so a
# parse failure means the user typed unquoted words; text containing any of them was reaching for
# TOML syntax and a parse failure means they got it wrong.
_TOML_STRUCTURAL_CHARS = frozenset("\"'[]{}=,\n")
# the characters that make a TOML KEY mean something other than its literal spelling: `.` nests,
# quotes make a bare key hold characters it otherwise could not. deliberately narrower than
# _TOML_STRUCTURAL_CHARS, which describes values -- a key is checked before the `=` split, so the
# structural characters of a value are not applicable to it.
_TOML_KEY_STRUCTURAL_CHARS = frozenset(".\"'")
# non-word TOML scalars start with a digit or sign; a parse failure is malformed syntax, not prose.
# include `.` so unsigned `.5` is rejected consistently with signed `+.5`.
_TOML_SCALAR_LEADING_CHARS = frozenset("0123456789+-.")
# the TOML booleans, which are written as bare words rather than starting with a digit or sign and
# are therefore the blind spot of _TOML_SCALAR_LEADING_CHARS. TOML spells them in lowercase only, so
# a case variant is a malformed literal rather than prose and must not forward as a string.
_TOML_BOOLEAN_WORDS = frozenset({"true", "false"})
# reject case variants of TOML's lowercase non-finite floats rather than forwarding them as strings.
# include `infinity`: float coercion turns it back into the unsupported infinite value.
_TOML_NON_FINITE_WORDS = frozenset({"inf", "infinity", "nan"})
# TOML has no null. these are the spellings people reach for anyway, borrowed from json, python, and
# yaml -- all bare words, so they land in the same blind spot: the parse fails, the value carries no
# structural character, and it forwards as its own literal STRING. an env testing `if value is None`
# or `if not value` then reads a truthy string, and no [environment.params] assignment could have
# produced it, since the config has no way to spell an absent value either. omitting the parameter
# is what expresses that, so say so rather than forwarding text nothing asked for.
_TOML_NULL_WORDS = frozenset({"null", "none", "nil"})


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
    before the reward ever sees it (flash/envs/adapter.py). This command has no run config to read
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
        # the gold completion messages as returned, kept so the rendered-role check can inspect the
        # concatenation SFT would train on rather than the per-turn replay this command scores.
        "completion": [],
    }


def _drive_single_turn(env, example: dict, record: dict, *, force_echo: bool = False) -> None:
    prompt = _check_messages(env.prompt_messages(example), "prompt")
    record["prompt"] = prompt
    completion = _gold_completion(env, example)
    record["completion"] = completion
    reference_turns = _reference_turns(completion)
    policy = "echo" if force_echo else _resolve_policy(reference_turns)
    record["policy"] = policy
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


def _drive_multi_turn(env, example: dict, record: dict, *, force_echo: bool = False) -> None:
    state = env.new_rollout_state(example)
    record["prompt"] = _check_messages(state.get("prompt") or state.get("messages"), "prompt")
    completion = _gold_completion(env, example)
    record["completion"] = completion
    reference_turns = _reference_turns(completion)
    policy = "echo" if force_echo else _resolve_policy(reference_turns)
    record["policy"] = policy
    record["thinking_markup"] = _carries_thinking_markup(reference_turns)
    # mirror the worker turn loop (flash/engine/worker/train/rl/child/multiturn.py): drive one model
    # turn, then stop at the hard turn ceiling, on the env's own done signal, or when the
    # env yields no reply. the turn counter rises every turn until it reaches the cap, so a
    # cooperatively-stepping env terminates here exactly as it would in training; no separate
    # non-termination guard is needed.
    #
    # the ceiling is the EFFECTIVE one, not `env.max_turns`: a row that sets `max_episode_turns`
    # below the dataset-wide cap is stopped by its own budget, and `rollout_done` gives that budget
    # precedence (flash/envs/adapter.py). comparing the turn count against `env.max_turns` alone
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
        if not env_msgs:
            break
        # the env's own reply messages feed the chat template for the next turn in the real
        # rollout, so validate their envelope here too: a malformed reply that would break
        # remotely must fail the episode instead of slipping through on a finite reward.
        _check_messages(env_msgs, "env_reply")
        if gold_finished is None and turns >= len(reference_turns):
            # the gold answer has just run out and its last turn is applied: this is the only
            # moment the reference can be judged on its own, before junk padding touches the state.
            gold_finished = _episode_completed(env, state, hard_cap)
        if env.rollout_done(state, max_turns=hard_cap):
            break

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
        # validated exactly like the in-loop replies above: a malformed reply breaks the chat
        # template on the paid run whether it was the last one or not. this call is the ONLY
        # env_reply for an episode stopped at the ceiling, and a per-example `max_episode_turns`
        # makes that the common case rather than a rare one -- leaving it unchecked let an env whose
        # final reply is malformed reach `overall: PASS`.
        #
        # an EMPTY reply is allowed first, exactly as the in-loop path does (`if not env_msgs:
        # break` precedes its own `_check_messages`): an env with nothing further to observe
        # legitimately returns nothing, and `_check_messages` rejects an empty list.
        final_msgs = env.env_reply(state["messages"], state)
        if final_msgs:
            _check_messages(final_msgs, "env_reply")
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
    # says nothing about the reference either way. sampling mid-loop keeps BOTH halves honest: a
    # five-move gold answer against a twelve-turn cap on an environment no move can ever solve --
    # the exact field case this warning exists for -- is still reported, while a short gold answer
    # on a working env, which simply stopped early, is not blamed for the padding that followed.
    #
    # `rollout_done` deliberately is NOT consulted here. the real adapter
    # (flash/envs/adapter.py `rollout_done`) returns True from `turn >= cap` ALONE, so at the ceiling
    # it is True for a healthy env and a dead one alike -- ANDing it in made this warning
    # unsatisfiable in production while the fakes, which override it to a bare `False`, kept the
    # tests green. what distinguishes the two is the env's own completion signal, which is what
    # `_episode_completed` reads.
    unfinished = gold_finished is False if gold_finished is not None else True
    record["hit_turn_cap"] = stopped_at_ceiling and unfinished
    record["reward"], record["scorer_error"], record["scored_text"] = _score_with_error(
        env, "", example, state
    )
    record["state"] = state


def _episode_completed(env, state: dict, hard_cap: int) -> bool:
    """Whether the ENVIRONMENT declared this episode over, as opposed to running out of turns.

    `rollout_done` cannot answer this. The real adapter (flash/envs/adapter.py) returns True from
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


def _separates_on_turn_rewards(env, example: dict, state: dict | None) -> bool:
    """Whether per-turn rewards separate an episode whose scalar does not.

    ``credit_assignment = "per_turn"`` trains from metadata returned through
    ``rollout_rewards_many`` (flash/envs/adapter.py), not ``env.reward``. Distinct turn values prove
    separation; missing, invalid, or unavailable vectors leave the scalar authoritative.
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
    return len({float(turn) for turn in turns}) > 1


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
    replayed: int = 0
    replayed_zero: list[tuple[dict, dict | None]] = field(default_factory=list)
    # whether any episode replayed a gold answer at all, including the ones the blocking gate
    # excludes (reasoning markup, an incomplete replay). an all-echo run scored only deliberate
    # junk, for which zero is the CORRECT answer, so the warning must not read that as a broken
    # grader -- but it is still worth saying that nothing was measured.
    replayed_any: bool = False
    # the first episode that produced a finite reward, whatever its policy. used only as the subject
    # of the advisory warning's separation probe.
    scored_episode: tuple[dict, dict | None] | None = None
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
            self.scored_episode = (example, record["state"])
        # an ECHO episode's reward IS interpretable: zero is the correct score for deliberate junk,
        # and the warning has separate, weaker wording for a run that scored only junk. recorded
        # here because the replay-only bookkeeping below does not apply to it.
        if record["policy"] != "replay":
            self.interpretable_rewards.append(reward)
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
            self.replayed_zero.append((example, record["state"]))


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
    # the per-turn check is cheap -- one metadata call, no episode driven -- so it runs wherever it
    # might inform either consumer.
    separates_on_turns = probe_worth_running and _separates_on_turn_rewards(env, *probe_subject)
    # the junk probe drives a whole extra episode through user code and can bill a paid judge, so it
    # is spent only where its answer changes an outcome: the blocking gate asking (grpo, accountable
    # gold all zero), or the advisory warning about to fire on a run whose gold answers all scored
    # zero. the second condition is `all_accountable_gold_scored_zero`, NOT merely `uniformly_zero`:
    # a run whose accountable gold was never zero has nothing for the probe to compare against, and
    # driving it there bought an unusable number at the cost of an extra billed episode.
    junk_reward = (
        _junk_reward(env, probe_subject[0])
        if probe_worth_running and not separates_on_turns and all_accountable_gold_scored_zero
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
        separates=separates_on_turns or (junk_reward is not None and junk_reward < 0.0),
    )
    return True


def _load_failure(reason: str) -> int:
    _err(f"env test failed: {reason}")
    print("0/0 episodes passed contract checks")
    return _err("overall: FAIL")


def _reject_unsubmittable_param(key: str, value: object) -> None:
    """Reject TOML values that ``[environment.params]`` could not submit.

    JSON excludes TOML date/time objects and non-finite floats, so use ``allow_nan=False``.
    ``ensure_ascii=False`` exposes lone surrogates; encoding the JSON result catches them even when
    nested, because no UTF-8 config can carry that value.
    """
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--param {key} is not JSON-serializable and could not be submitted "
            f"in [environment.params]: {exc}"
        ) from exc
    if not _is_expressible_in_toml(encoded):
        raise ValueError(
            f"--param {key} is not valid UTF-8, so no config file could carry it and the run "
            f"would never receive this value"
        )


def _parse_param_value(key: str, raw: str) -> object:
    """Parse one ``--param`` value the way ``[environment.params]`` would parse it.

    Bare unquoted text is not valid TOML but is what users type most often, so it falls back to a
    string. Everything else must parse, because silently keeping a malformed structured value as a
    literal string would validate the environment against parameters the equivalent config entry
    could never load -- the offline gate would pass on input training rejects.
    """
    value = raw.strip()
    try:
        document = tomllib.loads(f"v = {value}")
    except tomllib.TOMLDecodeError as exc:
        # the fallback is an allowlist, not a blocklist of opening delimiters: `filters=]` opens
        # nothing yet is still malformed TOML, and blocklisting only the openers let it through as
        # the literal string "]". a bare string is text with no TOML structural character in it.
        #
        # ...and no delimiter is needed to be reaching for TOML syntax. `cutoff=2026-13-01` holds
        # none of those characters, so it forwarded as the string "2026-13-01" while the equivalent
        # `[environment.params]` entry fails to load -- the gate passing on a config that cannot be
        # written. same for `1e`, `0x`, `007`, `1_`, `12:99:00`. a leading digit or sign is the
        # tell: every TOML scalar except the bare-word `true`/`false`/`inf`/`nan` spellings starts
        # with one, so such a token is a malformed number or date, not prose.
        if value and not (set(value) & _TOML_STRUCTURAL_CHARS):
            # the booleans are the family of TOML scalars that does NOT start with a digit or sign,
            # so the leading-character test below cannot see them. TOML spells them lowercase only,
            # which makes a python-style `strict=False` parse-fail and fall through here as the
            # STRING "False" -- and a non-empty string is truthy, so an env branching on `if strict`
            # reads it as enabled while the config spelling `false` disables it. the offline gate
            # would pass on the opposite of what the run trains with.
            if value.lower() in _TOML_BOOLEAN_WORDS:
                raise ValueError(
                    f"--param {key} is not a valid TOML value: {exc}. TOML spells "
                    f"{value.lower()} in lowercase; write --param {key}={value.lower()} for the "
                    f"boolean, or --param '{key}=\"{value}\"' to pass it as text"
                ) from exc
            # the non-finite floats are the same blind spot, minus their optional sign -- `-Inf`
            # does start with one of the leading characters, but the message that test raises talks
            # about numbers and dates and would not name what is actually wrong.
            if value.lstrip("+-").lower() in _TOML_NON_FINITE_WORDS:
                raise ValueError(
                    f"--param {key} is not a valid TOML value: {exc}. TOML spells the non-finite "
                    f"floats as lowercase inf and nan, and [environment.params] could not submit "
                    f"one anyway since it is not JSON; pass a finite number, or "
                    f"--param '{key}=\"{value}\"' to pass it as text"
                ) from exc
            # the null spellings, the last bare-word family. unlike the others there is no lowercase
            # form to point at, because TOML cannot express an absent value at all.
            if value.lower() in _TOML_NULL_WORDS:
                raise ValueError(
                    f"--param {key} is not a valid TOML value: {exc}. TOML has no null, so "
                    f"[environment.params] could not carry this either; omit --param {key} to "
                    f"leave it unset, or --param '{key}=\"{value}\"' to pass it as text"
                ) from exc
            if value[0] not in _TOML_SCALAR_LEADING_CHARS:
                # the bare-string path returns before the parsed-value checks below, so the one
                # that applies to text is asked here too. this is the route a surrogate actually
                # takes: it holds no structural character, so it is read as prose.
                _reject_unsubmittable_param(key, value)
                return value
            # quoting is the escape hatch, and it is the same spelling the config needs -- a
            # genuinely textual "3px" has to be written `"3px"` in `[environment.params]` too, so
            # pointing at it keeps the flag and the config in step rather than adding a second
            # spelling that only the flag accepts.
            raise ValueError(
                f"--param {key} is not a valid TOML value: {exc}. it starts like a number or "
                f"date, so [environment.params] would reject it too; quote it "
                f"(--param '{key}=\"{value}\"') to pass it as text"
            ) from exc
        raise ValueError(f"--param {key} is not a valid TOML value: {exc}") from exc
    # a value containing a newline makes `v = <value>` a multi-line document, so tomllib accepts
    # `max_rows=5\nstrict=true` as two assignments and taking only "v" drops the second silently.
    if set(document) != {"v"}:
        extra = ", ".join(sorted(set(document) - {"v"}))
        raise ValueError(
            f"--param {key} contains more than one assignment ({extra}); "
            "pass one KEY=VALUE per --param"
        )
    parsed = document["v"]
    _reject_unsubmittable_param(key, parsed)
    return parsed


def _is_expressible_in_toml(text: str) -> bool:
    """Report whether ``[environment.params]`` can carry ``text`` unchanged.

    TOML quoted keys and strings cover normal text; only values that cannot encode as UTF-8, such as
    lone surrogates, cannot appear in the config. Apply this to both assignment sides.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _literal_param_key(key: str) -> str:
    """Resolve a ``--param`` TOML key to its literal parameter name.

    Parse dotted and quoted keys with tomllib so ``difficulty.level`` nests while
    ``"release.channel"`` stays flat. Reject genuine nesting because params are splatted as kwargs
    in ``flash/envs/base.py``; pass the containing inline table instead.
    """
    name = key
    if set(key) & _TOML_KEY_STRUCTURAL_CHARS:
        try:
            document = tomllib.loads(f"{key} = 0")
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"--param {key} is not a valid TOML key: {exc}. [environment.params] would reject "
                f"this spelling too; quote the name (--param '\"{key}\"=...') to pass it literally"
            ) from exc
        # one assignment yields one top-level entry, whatever the spelling. it is nested exactly
        # when the dots were read as structure, which is the case this flag cannot forward.
        ((name, resolved),) = document.items()
        if isinstance(resolved, dict):
            raise ValueError(
                f"--param {key} uses TOML key syntax that denotes structure, which this flag "
                f"cannot forward faithfully. pass the containing table as one value instead, for "
                f"example --param {name}='{{ level = 3 }}', or quote the name "
                f"(--param '\"{key}\"=...') if the dot is part of it"
            )
    # whether the config can hold the name at all. almost always yes: a QUOTED key carries `bad
    # key`, `a/b`, `café` and the rest, and the schema loader takes it, so those are configs a run
    # really can receive. an earlier guard here rejected anything outside the BARE-key grammar,
    # which blocked validating a working config while claiming the config could not hold the name.
    # what is left is the names a UTF-8 config file cannot physically contain.
    if not _is_expressible_in_toml(name):
        raise ValueError(
            f"--param {key!r} is not valid UTF-8, so no config file could carry it and the run "
            f"would never receive this parameter"
        )
    return name


def _quoted_key_end(item: str, start: int) -> int | None:
    """Index of the quote closing the one at ``start``, or None when it is never closed."""
    quote = item[start]
    index = start + 1
    while index < len(item):
        # only a basic string takes escapes. in a literal string a backslash is just a character,
        # so consuming the next one there would step over the closing quote.
        if item[index] == "\\" and quote == '"':
            index += 2
            continue
        if item[index] == quote:
            return index
        index += 1
    return None


def _split_param_assignment(item: str) -> tuple[str, str, str]:
    """Split ``--param`` at the assignment ``=`` outside quoted key text.

    TOML permits flat keys such as ``"a=b"``. Unterminated quotes fall back to ``partition`` so the
    key validator reports the malformed spelling.
    """
    index = 0
    while index < len(item):
        char = item[index]
        if char in "\"'":
            closing = _quoted_key_end(item, index)
            if closing is None:
                break
            index = closing + 1
            continue
        if char == "=":
            return item[:index], "=", item[index + 1 :]
        index += 1
    return item.partition("=")


def _env_params(args) -> dict:
    """Build the ``load_environment()`` kwargs from ``--split`` / ``--param KEY=VALUE``.

    Mirrors ``[environment.params]`` so the local gate can validate the split a run actually
    trains on. Without this the gate always loaded ``dataset/train.jsonl`` and could pass while
    the configured split was never exercised.
    """
    params: dict = {}
    for item in getattr(args, "param", None) or []:
        key, sep, raw = _split_param_assignment(str(item))
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--param must be KEY=VALUE (got {item!r})")
        params[_literal_param_key(key)] = _parse_param_value(key, raw)
    split = getattr(args, "split", None)
    if split is not None:
        # distinguish "not passed" from "passed empty". `--split "$SPLIT"` with an unset variable
        # is an explicit request for a split, and treating it as absent leaves a `--param
        # split=...` in effect -- so the gate silently validates a different split than the one the
        # command asked for, which is the failure this flag exists to prevent.
        split = str(split).strip()
        if not split:
            raise ValueError("--split requires a non-empty split name")
        params["split"] = split
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

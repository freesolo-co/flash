"""Offline contract checks for local Freesolo environments."""

from __future__ import annotations

import json
import math
import sys
import tomllib
from pathlib import Path

from flash.cli.commands.env.push import _err, _resolve_local_env_entrypoint
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


# the only role whose merge causes the trained-behaviour defect this check exists for. `tool` is
# deliberately excluded: one assistant message carrying two parallel tool calls is answered by two
# `tool` messages in a row, which is the required wire format and what chat templates render as
# separate delimited blocks -- not a collapse. flagging it would train users to edit a correct
# transcript into a broken one, or to ignore the warning entirely.
_MERGEABLE_ROLE = "assistant"


def _repeated_roles(messages: list[dict]) -> list[tuple[int, str]]:
    """The (index, role) of every message whose role repeats the one before it.

    SFT does not replay a gold completion turn by turn. It renders ONE training string from
    ``[*prompt_messages, *completion_messages]`` (flash/engine/profiling/sft_workload.py), so the
    only thing the model learns from is that concatenation -- and nothing in flash validates its
    role sequence. A multi-turn gold answer returned as assistant turns alone renders as one user
    question followed by every answer back to back, which trains the model to dump the whole
    episode into a single reply: the opposite of the behaviour being taught.

    This gate replays those turns individually and scores each one, so it passes either way; the
    defect exists only at the concatenation, which nothing here exercised.
    """
    repeated: list[tuple[int, str]] = []
    previous = ""
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip().lower()
        if role == _MERGEABLE_ROLE and role == previous:
            repeated.append((index, role))
        previous = role
    return repeated


def _rendered_roles(prompt: list[dict], completion: list[dict]) -> str:
    """The rendered role sequence, marking where the completion begins.

    Printed beside the warning because the role ORDER is the defect: a reader needs to see which
    turns merged and on which side of the seam, and no preview of the message text shows that.
    """
    roles = [str(message.get("role", "?")).strip().lower() or "?" for message in prompt]
    roles.append("|")
    roles.extend(str(message.get("role", "?")).strip().lower() or "?" for message in completion)
    return " ".join(roles)


def _warn_on_repeated_rendered_roles(
    prompt: list[dict], completion: list[dict], *, multi_turn: bool
) -> None:
    """Warn when the SFT training string would run two assistant turns together.

    Skipped entirely for a multi-turn environment. There, a gold completion holding consecutive
    assistant turns is the CONTRACT, not a defect: the intervening user turns are produced by
    `env_reply` during the rollout and are deliberately absent from the dataset -- `_drive_multi_turn`
    depends on exactly that shape, replaying one reference turn per model turn. Warning about it
    would fire on every correct multi-turn environment and advise an edit that breaks the replay.

    For a single-turn environment the concatenation IS the trained text, so the same adjacency is a
    real defect: SFT renders one string from `[*prompt_messages, *completion_messages]`
    (flash/engine/profiling/sft_workload.py) and nothing else validates it. A completion returned as
    assistant turns alone then renders as one user question followed by every answer back to back,
    training the model to dump the whole episode into a single reply.

    Reported for every algorithm, not just sft: one published environment serves all three, so a
    single-turn env shaped this way is broken for sft whichever algorithm this invocation named.

    Only the RENDERED sequence is checked, never either list alone -- a prompt may legitimately end
    on an assistant prefill, and the fault is adjacency after concatenation.
    """
    if multi_turn or not prompt or not completion:
        return
    repeated = _repeated_roles([*prompt, *completion])
    if not repeated:
        return
    print(
        render.warn(_repeated_roles_message(repeated, len(prompt)))
        if render.styled()
        else f"warning: {_repeated_roles_message(repeated, len(prompt))}",
        file=sys.stderr,
    )
    print(f"  rendered roles: {_rendered_roles(prompt, completion)}", file=sys.stderr)


def _repeated_roles_message(repeated: list[tuple[int, str]], boundary: int) -> str:
    """Name each collapsing turn in the region that owns it, with that region's own index.

    Three regions, three different edits, so all three are named rather than folded into one: a
    repeat inside the prompt is a `prompt_messages`/`start_episode` fault, one at the seam means the
    prompt ends on a prefill the completion continues, and one inside the completion means the gold
    answer needs the env's follow-up user turns interleaved. Reporting a prompt-internal or seam
    collision as "inside sft_completion" sends the author to a file that is correct.

    Indices are rebased onto the region they belong to. A raw offset into the concatenation is not a
    position the author can look up: with a system prompt prepended, turn 0 of a three-message
    completion is reported as message 3, which does not exist in the file they open.
    """
    inside_prompt = [index for index, _role in repeated if index < boundary]
    at_seam = [index for index, _role in repeated if index == boundary]
    inside_completion = [index for index, _role in repeated if index > boundary]
    parts: list[str] = []
    if inside_prompt:
        where = ", ".join(f"message {index}" for index in inside_prompt)
        parts.append(f"inside prompt_messages ({where})")
    if at_seam:
        parts.append(
            "at the prompt/completion boundary (its first turn continues the prompt's last)"
        )
    if inside_completion:
        # rebased onto sft_completion's own indexing, which is what the author can actually look up.
        where = ", ".join(f"message {index - boundary}" for index in inside_completion)
        parts.append(f"inside sft_completion ({where})")
    remedy = (
        "a gold answer that spans several assistant turns must interleave the user turns it was "
        "answering"
        if inside_completion
        else "each region must alternate roles once concatenated"
    )
    return (
        f"sft would train on consecutive assistant turns {' and '.join(parts)}. sft renders one "
        f"string from prompt_messages + sft_completion, so these turns merge into a single reply -- "
        f"{remedy}"
    )


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


def _scored_text(record: dict) -> str:
    """The exact text the grader scored, as captured at the scoring call.

    Read from the record rather than rebuilt from ``responses``: a multi-turn env whose
    ``step_episode`` returns a ``final_response_text`` has the adapter replace the episode's
    response with that override before scoring, so the replayed turns are not what was graded.
    Falls back to the replayed turns only when nothing was captured (``None``) -- an episode that
    failed before it was scored. An empty capture is a real graded value, not a missing one, and
    an empty answer is exactly the fault a reader needs named.

    Unlike ``_preview`` this preserves whitespace and does not truncate: it is printed with
    ``repr`` beside a zero reward so a trailing newline, a tab, or a wrapper past the preview
    cutoff stays visible. Those are the formatting faults that make a correct grader reject a
    gold answer, so hiding them defeats the diagnostic.
    """
    scored = record.get("scored_text")
    if scored is not None:
        return str(scored)
    return "\n".join(str(item) for item in record.get("responses") or ())


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
    # env yields no reply. the hard cap is fixed at what the trainer passes (env.max_turns)
    # and the turn counter rises every turn until it reaches the cap, so a cooperatively-
    # stepping env terminates here exactly as it would in training; no separate
    # non-termination guard is needed.
    hard_cap = int(env.max_turns)
    turns = 0
    # mirrors the worker's own flag: True while the newest turn has not been through env_reply.
    env_step_pending = False
    # whether the loop ended at the trainer's ceiling rather than on the env's own signal. this is
    # only HALF the verdict: the last model turn has not been applied yet at that point (see the
    # deferred env_reply below), so an env that finishes exactly on its last allowed turn still
    # looks unfinished here. the conclusion is drawn after that turn is applied.
    stopped_at_ceiling = False
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
        env.env_reply(state["messages"], state)
    # decided HERE, not at the break: the last model turn is applied only by the env_reply above, so
    # asking before it would report an env that solves the task on its final allowed turn as one
    # that never finishes -- and send the author to fix termination logic that works.
    #
    # `replay_incomplete` disqualifies the verdict entirely. once the gold answer runs out the
    # driver pads with junk, and junk cannot advance the env, so reaching the ceiling is the
    # padding's doing rather than the reference's. blaming the episode's termination there points
    # at the wrong file: the real fault is a gold answer shorter than the episode.
    record["hit_turn_cap"] = (
        stopped_at_ceiling
        and not record["replay_incomplete"]
        and not env.rollout_done(state, max_turns=hard_cap)
    )
    record["reward"], record["scorer_error"], record["scored_text"] = _score_with_error(
        env, "", example, state
    )
    record["state"] = state


def _scores_gold_no_better_than_junk(env, example: dict, gold_reward: float) -> bool:
    """Whether junk scores at least as well as this gold answer.

    A gold score of zero can still beat negative junk, so compare them directly. Run this extra
    stateful-env pass only after all real episodes have been scored. A probe error is not evidence
    of flat reward because an env may require parseable answers.
    """
    try:
        probe = _new_record()
        if env.multi_turn:
            _drive_multi_turn(env, example, probe, force_echo=True)
        else:
            _drive_single_turn(env, example, probe, force_echo=True)
        junk_reward = probe["reward"]
    except (Exception, SystemExit):
        return False
    if junk_reward is None or not math.isfinite(junk_reward):
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


def _warn_on_uniformly_zero_rewards(
    rewards: list[float], *, algorithm: str, replayed_any: bool
) -> None:
    """Warn when every episode this run scored came back exactly zero.

    Deliberately independent of the blocking gate above, which abstains for a non-grpo algorithm,
    a reference in reasoning markup, a replay that ran out mid-episode, and a junk probe that
    raised. Each abstention is individually right -- none of them is proof the grader works -- but
    together they let a run in which nothing scored anything print `overall: PASS` and no other
    line.

    The two ways a run reaches all-zero are NOT the same finding, so they are not reported with the
    same words:

    * At least one gold answer was replayed. A working grader is supposed to pay its own reference
      more than nothing, so this points at the grader or at the shape of the gold answer it was
      handed.
    * Every episode was `echo`. Echo replays deliberate junk, and a correct grader SHOULD score
      junk zero -- so the zero is not itself suspicious. What is worth saying is that no gold
      answer was ever scored, which means this run carries no evidence either way, and that an
      empty `sft_completion` is also nothing for SFT to train on.

    For GRPO a constant reward is additionally silent rather than merely uninformative: rewards are
    mean-centred within each group (`algorithm.norm_adv_by_std_in_grpo=False`,
    flash/engine/worker/train/rl/verl_config.py), so a constant reward gives zero advantage and zero
    gradient. Training completes, the loss curve looks unremarkable, and the adapter comes out
    identical to its warm start. The end-of-run advantage-spread guard catches that only after the
    GPUs have been paid for (flash/engine/worker/train/rl/checkpoints.py); this catches it before
    they are allocated.

    Only exact zeros trip it. A nonzero constant is a weaker, separate signal -- a reward may
    legitimately be constant across three rows -- and is not diagnosed here.
    """
    if not rewards or any(reward != 0.0 for reward in rewards):
        return
    count = len(rewards)
    if not replayed_any:
        message = (
            f"all {count} scored episode(s) returned reward 0.000000, and every one of them "
            "replayed a deliberately wrong answer because no row supplied a gold one. a zero is "
            "the correct score for that, so this run is not evidence the reward function works -- "
            "and an empty gold completion is also nothing for sft to train on. give the rows a "
            "gold answer (`sft_completion`, or the row's `output`) and re-run"
        )
    else:
        detail = (
            "every rollout would then carry the same reward, which GRPO centres to a zero "
            "advantage and a zero gradient: the run would complete, report an unremarkable loss "
            "curve, and produce an adapter identical to its warm start"
            if algorithm == _REWARD_DRIVEN_ALGORITHM
            else "a reward that is constant across every row measures nothing about the environment"
        )
        message = (
            f"all {count} scored episode(s) returned reward 0.000000, so this run measured "
            f"nothing. {detail}. check the reward function, its runtime dependencies, and that "
            "the gold answer it scored is in the shape it expects"
        )
    print(render.warn(message) if render.styled() else f"warning: {message}", file=sys.stderr)


def _load_failure(reason: str) -> int:
    _err(f"env test failed: {reason}")
    print("0/0 episodes passed contract checks")
    return _err("overall: FAIL")


def _evaluation_example(case) -> dict:
    example = dict(case.metadata or {})
    example["input"] = case.input
    example["output"] = case.expected
    if case.id is not None:
        example["id"] = case.id
    return example


def _normalize_prompt_images(env, example: dict, messages: list[dict]) -> None:
    """Resolve the case's images the way the online command and every training worker do.

    Raises whatever `normalize_prompt_images` raises, so an unreadable, oversized, or malformed
    image fails this gate with the message the caller would have hit at generation time. Text-only
    cases -- the vast majority -- return without importing the multimodal machinery at all.
    """
    from flash.content.multimodal import record_has_images

    if not record_has_images(example, messages):
        return
    from flash.content.multimodal import normalize_prompt_images

    normalize_prompt_images(example, messages, getattr(env, "package_root", None))


def _evaluation_response(env, case) -> tuple[str, str]:
    example = _evaluation_example(case)
    # build the prompt even though the replayed response does not need it. `flash env eval` sends
    # every case through prompt_messages() (flash/cli/commands/env/eval.py `_case_messages`), so a prompt
    # that raises or returns malformed messages for a held-out case is a suite the online command
    # records a prompt-construction error for. checking only the scorer let this offline gate print
    # `overall: PASS` for exactly that sidecar.
    build = getattr(env, "prompt_messages", None)
    if callable(build):
        messages = _check_messages(build(example), "prompt")
        # `prompt_messages()` is only half of the prompt: `flash env eval` then runs
        # `normalize_prompt_images` (`_remote_prompt_messages`), as every training worker does
        # before tokenization. the envelope check above sees only the message list, so a case
        # carrying a top-level `image`/`images` -- a missing or oversized package-relative file --
        # or a malformed image block inside its prompt passed this gate, and the online command then
        # recorded prompt-construction failures for a suite reported `overall: PASS`. run the same
        # normalization here, against the environment's own package root, so both commands reject
        # the same suites.
        _normalize_prompt_images(env, example, messages)
    reference_turns = _reference_turns(_gold_completion(env, example))
    policy = _resolve_policy(reference_turns)
    response = (
        "\n".join(turn for turn in reference_turns if turn)
        if policy == "replay"
        else _ECHO_RESPONSE
    )
    return policy, response


def _check_evaluation_suites(entrypoint: Path, env) -> bool:
    from flash.envs.evaluations import (
        _DEFAULT_EVALUATIONS_PATH,
        EvalSuiteReport,
        has_evaluations,
        load_evaluation_suites,
        normalize_eval_result,
        validate_evaluation_cases,
    )

    if not has_evaluations(entrypoint):
        return True
    source = entrypoint.parent / _DEFAULT_EVALUATIONS_PATH
    try:
        suites = load_evaluation_suites(entrypoint, environment=env)
    except (Exception, SystemExit) as exc:
        reason = str(exc) or exc.__class__.__name__
        _err(f"evaluation checks failed: {reason}")
        return False

    all_valid = True
    for suite in suites:
        results = []
        try:
            cases = validate_evaluation_cases(suite, source=source)
            if not cases:
                all_valid = False
                _err(
                    f"evaluation suite {suite.name} failed contract checks: suite produced no cases"
                )
                continue
            for index, case in enumerate(cases, start=1):
                _policy, response = _evaluation_response(env, case)
                scored = suite.score(case, response)
                result = normalize_eval_result(
                    case,
                    response,
                    scored,
                    case_id=case.id or str(index),
                )
                results.append(result)
            report = EvalSuiteReport(name=suite.name, results=tuple(results))
            # a scorer that reported an error did not grade the case. `flash env eval` counts
            # those as errors and fails the suite, so approving them here would let the offline
            # gate greenlight exactly the sidecar the online command refuses.
            errored = [result for result in results if result.error]
            if errored:
                all_valid = False
                detail = ", ".join(f"{result.case_id}: {result.error}" for result in errored)
                _err(
                    f"evaluation suite {suite.name} failed contract checks: "
                    f"{len(errored)}/{report.total} case(s) reported a scoring error ({detail})"
                )
                continue
            print(
                f"evaluation suite {suite.name}: {report.total}/{report.total} cases "
                f"passed contract checks mean_score={report.mean_score:.6f}"
            )
        except (Exception, SystemExit) as exc:
            all_valid = False
            reason = str(exc) or exc.__class__.__name__
            _err(f"evaluation suite {suite.name} failed contract checks: {reason}")
    return all_valid


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
    # only replay episodes carry a gold answer to score. an echo episode has none, so its reward
    # says nothing about the grader and is counted in neither total.
    replayed = 0
    replayed_zero: list[tuple[dict, dict | None]] = []
    # every finite reward this run scored, whatever the policy or algorithm. the blocking gate below
    # counts only the replay episodes it can hold responsible, and abstains for thinking markup, an
    # incomplete replay, a non-grpo algorithm, or a junk probe that raised -- so a run where nothing
    # scored anything reached `overall: PASS` with no line saying so. this list is what makes the
    # uniformly-zero warning independent of every one of those abstentions.
    scored_rewards: list[float] = []
    # whether any episode replayed a gold answer at all, including the ones the blocking gate
    # excludes (reasoning markup, an incomplete replay). an all-echo run scored only deliberate junk,
    # for which zero is the CORRECT answer, so the warning below must not read that as a broken
    # grader -- but it is still worth saying that nothing was measured.
    replayed_any = False
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
        if reward is not None:
            # recorded for every passing episode, echo included: an echo run whose scores are all
            # zero is equally unmeasured, and is exactly the shape a replay-only counter misses.
            scored_rewards.append(reward)
        if record["policy"] == "replay" and reward is not None:
            replayed_any = True
            # a reference written in reasoning markup cannot be replayed faithfully from here (see
            # `_carries_thinking_markup`), so its score is not evidence about the grader and is kept
            # out of the blocking gate's totals. neither is a multi-turn episode whose gold answer
            # ran out before the rollout did: the graded trajectory is then part reference and part
            # junk, and a zero says nothing about the reference the gate never ran. the advisory
            # warning below still fires for both: a low reward is worth surfacing either way, it
            # just cannot be the reason to fail.
            if not record["thinking_markup"] and not record["replay_incomplete"]:
                replayed += 1
                # an exact zero is what a scorer that recognized nothing returns, so it stays the
                # signature this gate looks for. it is confirmed against a wrong answer once every
                # episode has run, not here: see `_scores_gold_no_better_than_junk`.
                if reward == 0.0:
                    replayed_zero.append((example, record["state"]))
            if reward <= 0.0:
                # two different faults produce this same zero, and naming only the grader sends
                # readers to edit a scorer that is working. the gold completion is the other
                # candidate: `sft_completion` defaults to the row's raw `output`
                # (flash/envs/adapter.py), so a dataset whose `output` is a bare value replays that
                # value verbatim -- which a grader requiring a wrapper (`\boxed{}`, a json object,
                # a tag) is right to score zero.
                message = (
                    f"replay gold answer scored low (reward={reward:.6f}); "
                    "check the reward function or the gold completion it scored"
                )
                print(
                    render.warn(message) if render.styled() else f"warning: {message}",
                    file=sys.stderr,
                )
                # printed with repr rather than through `_preview`: the whole point of showing this
                # text is to expose a formatting defect, and `_preview` collapses whitespace and
                # truncates, so a stray newline or a `\boxed{}` past the cutoff -- exactly the
                # faults this line exists to reveal -- would be invisible.
                # labelled by what it is -- the text the grader received -- not "gold answer": when
                # a multi-turn env overrides the response via `final_response_text`, the scored text
                # is env-authored and calling it the gold answer sends the reader to edit a dataset
                # row that was never scored.
                print(
                    f"  scored text: {_scored_text(record)!r}",
                    file=sys.stderr,
                )
            if record["hit_turn_cap"]:
                # a gold trajectory that never terminates is the signature of an env no rollout can
                # finish -- every move applied twice, a win condition that cannot be reached. the
                # reward alone hides it: a partial-credit grader pays a respectable score for the
                # capped attempt, and gold-vs-junk still clears its bar, so the run reads PASS.
                # what exposes it is the turn count, which is why it is printed as a comparison.
                message = (
                    f"replay gold answer never finished: it used all {record['turns']} turn(s) "
                    "and the environment never reported the episode done. a reference answer that "
                    "cannot complete an episode means no rollout can either; check the episode "
                    "termination condition and whether each turn is applied exactly once"
                )
                print(
                    render.warn(message) if render.styled() else f"warning: {message}",
                    file=sys.stderr,
                )
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
    # LS-005: block only when every gold answer scores zero and deliberate junk scores at least as
    # well; centered rewards may legitimately score gold at zero. partial failures remain warnings.
    # this gate applies only to grpo, which trains from env.reward. sft and opd use other losses.
    # default to grpo so omitting the algorithm cannot disable the blocking check.
    algorithm = getattr(args, "algorithm", None) or _REWARD_DRIVEN_ALGORITHM
    grader_recognizes_gold = not (
        algorithm == _REWARD_DRIVEN_ALGORITHM
        and replayed
        and len(replayed_zero) == replayed
        and not _separates_on_turn_rewards(env, *replayed_zero[0])
        and _scores_gold_no_better_than_junk(env, replayed_zero[0][0], 0.0)
    )
    if grader_recognizes_gold:
        # the blocking gate abstains far more often than it fires: for sft and opd, for a reference
        # in reasoning markup, for a replay that ran out mid-episode, for an echo run with no gold
        # answer at all, and whenever the junk probe raises. in every one of those a run that scored
        # exactly zero on every episode still printed `overall: PASS` and nothing else -- and a
        # uniformly-zero reward is never a measurement, whatever produced it. GRPO centres rewards
        # within each group, so a constant reward is a zero advantage and a zero gradient: training
        # completes, the loss curve looks unremarkable, and the adapter comes out identical to its
        # warm start. say so here rather than leave the verdict to speak for it.
        _warn_on_uniformly_zero_rewards(
            scored_rewards, algorithm=algorithm, replayed_any=replayed_any
        )
    if not grader_recognizes_gold:
        _err(
            f"all {replayed} replayed gold answer(s) scored zero, no better than a deliberately "
            "wrong answer; the reward function cannot recognize its own reference answers. check "
            "the grader and that its runtime dependencies are installed in this environment."
        )
    # run the evaluation checks even when the grader gate already failed, so one `flash env test`
    # reports every broken surface at once instead of hiding the sidecar's errors behind the
    # reward function's. both gates block; neither short-circuits the other.
    if not _check_evaluation_suites(entrypoint, env) or not grader_recognizes_gold:
        return _err("overall: FAIL")
    print(render.ok("overall: PASS") if render.styled() else "overall: PASS")
    return 0

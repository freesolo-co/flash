"""Offline contract checks for local Freesolo environments."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

from flash.catalog import normalize_algorithm
from flash.spec import PER_TURN_CREDIT_ASSIGNMENT

from . import render
from .envpush import _err, _resolve_local_env_entrypoint

_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
# the algorithms whose worker actually calls the environment's scorer. only rl.py reads a reward
# (env.scores_breakdown / env.reward) and only multiturn_rollout.py scores a rollout; the opd and
# opsd workers consume dataset(), prompt_messages(), and sft_completion() and never grade anything.
# so a placeholder scorer is a real defect for grpo and legitimate for every other algorithm, and
# samples_on_policy -- which is about sampling student completions, not grading them -- is the
# wrong question to ask here (codex[bot]).
_REWARD_CONSUMING_ALGORITHMS = frozenset({"grpo"})
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
# drawn on when every fixed candidate collides with the gold answer. single characters, so a control
# built from one of them shares no word with any gold text by construction.
_SYNTHETIC_CONTROL_ALPHABET = "zqxjkvwy0123456789bcdfghlmnprstu"
_SYNTHETIC_CONTROL_WIDTH = 64
_PREVIEW_CHARS = 200
_DEFAULT_EPISODES = 3


@dataclass(frozen=True)
class _Score:
    """One graded completion, as the trainer would see it.

    ``turns`` is the per-turn reward vector a multi-turn env may expose alongside the episode
    scalar; it is None for single-turn scoring and whenever the env offers no usable per-turn
    rewards. Two scores are "the same" for the flat-reward gate only when both parts match, so an
    env that ranks purely through per-turn credit is not mistaken for one that cannot rank.

    Whether the vector is read at all depends on the run's credit-assignment mode, which every
    comparison below takes as ``per_turn``. Only `select_grpo_trainer` reaching `GRPOPerTurnTrainer`
    makes these vectors reach an advantage, so the caller passes ``per_turn=False`` for the default
    mode and the episode scalar decides alone (codex[bot]).
    """

    episode: float
    turns: tuple[float, ...] | None = None

    def is_finite(self) -> bool:
        return math.isfinite(self.episode) and all(
            math.isfinite(value) for value in self.turns or ()
        )

    def _overlap(self, other: _Score) -> tuple[tuple[float, float], ...] | None:
        """Pair the turns both vectors actually reach, or None when neither has one.

        `build_per_turn_advantages` walks to the group's longest vector and, at each index, centres
        only the members present there. A shorter rollout therefore does not void the comparison:
        the turns both reached are still centred against each other. Comparing that overlap is what
        the trainer does, so a wrong control that merely terminated early stays evidence rather than
        being discarded (codex[bot]).
        """
        mine, theirs = self.turns, other.turns
        if mine is None or theirs is None:
            return None
        return tuple(zip(mine, theirs, strict=False))

    def outranks(self, gold: _Score, *, per_turn: bool) -> bool:
        """Report whether this deliberately wrong control beats gold where training reads.

        Asymmetric on purpose. ``_controlled_scores`` admits a control only when it is disjoint from
        EVERY gold turn, so there is no turn at which this completion legitimately scores above the
        replayed gold answer. One credited turn above gold is therefore enough: each turn index is
        centred against its own group mean, so that turn hands the wrong text a positive advantage
        and training reinforces it there, whatever the control does at the other turns. Requiring it
        to be no better anywhere let a crossing pair -- gold (1, 0) against control (0, 1) -- pass as
        "neither dominates" while the trainer rewarded the control at turn 1 (codex[bot]).

        ``per_turn`` is the GROUP's path, not this pair's: the vectors reach an advantage only when
        every member has one, which is the caller's question to settle.
        """
        pairs = self._overlap(gold) if per_turn else None
        if pairs is None:
            return self.episode > gold.episode
        # an empty overlap means a member emitted no credited turn; it earns no advantage anywhere,
        # so it is evidence of nothing rather than a win.
        return any(mine > theirs for mine, theirs in pairs)

    def separates_from(self, other: _Score, *, per_turn: bool) -> bool:
        """Report whether training could tell these two completions apart at all.

        Distinct per-turn vectors that neither dominate -- (1, 0) against (0, 1) -- still produce
        nonzero opposing advantages at both turns, because each turn is centred independently. They
        rank in no direction yet are plainly separable, so the flat gate asks this rather than
        reading "nothing outranks gold" as "the grader cannot rank" (codex[bot]).

        On the per-turn path the episode scalar is not consulted at all: `build_per_turn_advantages`
        REPLACES the episode advantages with centred turn rewards, so identical vectors train
        identically no matter how far apart the episode scores are. Reading the scalar first
        reported separation for a grader that produces exactly zero advantage (codex[bot]).
        """
        pairs = self._overlap(other) if per_turn else None
        if pairs is None:
            return self.episode != other.episode
        return any(mine != theirs for mine, theirs in pairs)


def _fmt_turns(score: _Score) -> str:
    """The per-turn vector as the warning prints it, so a reader sees the numbers that were compared."""
    return "(" + ", ".join(f"{value:.6f}" for value in score.turns or ()) + ")"


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
        # a gold answer carrying reasoning markup is graded by the run as `graded_text` leaves it,
        # which strips the <think> span under a thinking config (flash/engine/worker/rl.py). this
        # command has no run config to tell it whether thinking is on, so it can neither reproduce
        # that text nor rule it out: an exact-answer grader would score both this raw reference and
        # every control zero and report a working env as unable to rank. treat it as a replay the
        # driver cannot reproduce faithfully, which is what the flag already means (codex[bot]).
        return "<think>" not in content and "</think>" not in content
    if isinstance(content, list):
        # text blocks survive extraction verbatim; anything else (an image block) is dropped.
        return all(
            isinstance(block, dict)
            and block.get("type") == "text"
            and "<think>" not in str(block.get("text") or "")
            and "</think>" not in str(block.get("text") or "")
            for block in content
        )
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
        # multi-turn only: the terminal (example, state, turn_count) the gold reward came from, so
        # the controls can be rescored alongside it in one batched call.
        "rollout": None,
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
    record["reward"] = _Score(episode=_grade_single_turn(env, response, example))


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

    # the terminal rollout, kept unscored: the caller batches it with the control rollouts so a
    # listwise grader ranks them together, exactly as the worker submits them (codex[bot]).
    # _grade_gold_if_unscored covers the episodes that batch never runs.
    record["rollout"] = (example, state, len(responses))


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


def _synthetic_control(references: list[str]) -> str | None:
    """Build a wrong answer for gold text that collides with every fixed control, or None.

    Repeats a single character the gold answers never use. That character cannot appear in any gold
    word, so `_control_is_disjoint` holds by construction rather than by luck, and the repetition
    keeps the control long enough to be an implausible answer for an open-ended task. Returns None
    when the references between them use the whole alphabet, so the caller still excludes an episode
    it cannot control rather than inventing evidence.
    """
    used = {character for reference in references for character in reference.casefold()}
    for character in _SYNTHETIC_CONTROL_ALPHABET:
        if character in used:
            continue
        control = character * _SYNTHETIC_CONTROL_WIDTH
        # cheap belt-and-braces: the caller's own disjointness rule is the contract, so ask it
        # rather than trusting the construction to stay equivalent to it.
        if all(_control_is_disjoint(control, reference) for reference in references):
            return control
    return None


def _grade_single_turn(env, completion: str, example: dict) -> float:
    """Score one single-turn completion through the same dispatch the GRPO worker uses.

    ``flash/engine/worker/rl.py`` prefers ``scores_breakdown(...)["total"]`` and only falls back to
    ``reward()``. An environment whose real composite grader lives in ``scores_breakdown`` while
    ``reward`` is inherited or a placeholder would otherwise be judged here on a scorer training
    never calls -- failing a working environment, or passing a broken one (codex[bot]).
    """
    if hasattr(env, "scores_breakdown"):
        return float(env.scores_breakdown(completion, example, None).get("total", 0.0))
    return float(env.reward(completion, example, None))


def _grade_rollouts(env, rollouts: list[tuple[dict, dict, int]]) -> list[_Score]:
    """Score terminal rollouts through the multi-turn worker's own path, in one batched call.

    The multi-turn trainer never calls ``env.reward`` itself: it calls ``score_rollouts``
    (flash/engine/multiturn_rollout.py), which routes through ``rollout_rewards_many`` and returns
    typed rewards carrying optional per-turn values. ``scores_breakdown`` is single-turn only and is
    deliberately not consulted here.

    Every rollout being compared goes through one call because the worker submits its whole request
    list together. A listwise ``rollout_rewards_many`` -- one that ranks candidates against each
    other within the batch -- returns different numbers when handed a single rollout at a time, so
    scoring gold and controls separately can read a working grader as flat, or invert its ranking
    (codex[bot]).

    The per-turn values are what make this more than a scalar. Under
    ``credit_assignment = "per_turn"`` the trainer credits each assistant turn by its own
    group-relative reward (``GRPOPerTurnTrainer``, selected in flash/engine/worker/rl.py), so an env
    may hold the episode score constant while the per-turn vector still separates a gold rollout
    from a wrong one. Reading the episode scalar alone would report that env as unable to rank while
    training learns from it perfectly well (codex[bot]).
    """
    from flash.engine.multiturn_reward_scoring import RolloutScoreRequest, score_rollouts

    requests = [
        RolloutScoreRequest(example=example, state=state, turn_count=turn_count)
        for example, state, turn_count in rollouts
    ]
    return [
        _Score(episode=float(reward.episode), turns=reward.turns)
        for reward in score_rollouts(env, requests)
    ]


def _score_single_turn_control(env, example: dict, control: str) -> _Score:
    """Score one deliberately wrong single-turn answer the way training would score it.

    For single-turn GRPO a grader that raises is not inconclusive: ``reward_fn`` catches exactly
    that and scores the completion 0.0 (flash/engine/worker/rl.py), so training would see a real
    number. Record that same 0.0, otherwise a row that genuinely separates gold from wrong is
    dropped and a later tied row can fail the whole sample alone.

    ``SystemExit`` still propagates. ``reward_fn`` catches only ``Exception``, so a grader that
    exits on an unexpected completion kills training rather than scoring it zero -- and the gold
    replay is already failed for exactly that (codex[bot]).

    Raises ValueError when the grader returns a non-finite score, which is the same contract
    violation the gold answer is already failed for: the policy reaches this scorer with completions
    no more expected than these, and NaN or infinity there yields unusable samples.
    """
    try:
        score = _Score(episode=_grade_single_turn(env, control, example))
    except Exception:
        # mirrors reward_fn's own except branch: the run would score this 0.0 and carry on.
        score = _Score(episode=0.0)
    _require_finite_control(score)
    return score


def _require_finite_control(score: _Score) -> None:
    if not score.is_finite():
        raise ValueError(f"reward is not finite for a non-reference completion: {score.episode}")


def _score_multi_turn_episode(
    env, example: dict, gold: tuple[dict, dict, int], controls: list[str]
) -> tuple[_Score, list[_Score]]:
    """Drive the control rollouts and score them with the gold one in a single batched call.

    A multi-turn reward reads the accumulated rollout state, so a comparable wrong episode has to be
    driven, not assembled -- replay the same loop answering each control at every turn. The scoring
    that follows is deliberately one call over the whole list: the worker submits its entire rollout
    request list to ``score_rollouts`` at once (flash/engine/multiturn_rollout.py), so an env with
    listwise ``rollout_rewards_many`` semantics ranks these candidates against each other exactly as
    training would. Scoring each rollout in its own singleton call can hand back identical numbers
    for a grader that ranks fine in a batch, reporting it as flat, or a different ordering than
    training sees (codex[bot]).

    Unlike single-turn, a raising scorer is not softened to 0.0 here: the rollout path calls
    ``score_rollouts`` with no except branch, so a scorer that raises there aborts the run.
    Swallowing it would pass an environment that cannot survive its first rollout (codex[bot]).
    """
    driven = [_run_rollout(env, example, lambda _index, text=control: text) for control in controls]
    scores = _grade_rollouts(
        env, [gold, *((example, state, len(responses)) for state, responses in driven)]
    )
    control_scores = scores[1:]
    for score in control_scores:
        _require_finite_control(score)
    return scores[0], control_scores


def _usable_controls(references: list[str]) -> list[str] | None:
    """Pick deliberately wrong answers for this example, or None when none is provably wrong.

    Selection only -- no scoring and no env call -- so the caller can drive and score every
    candidate together in whichever batched shape its turn mode requires.

    Every usable control is kept rather than just the first, so an episode counts as separated when
    *any* of them ranks below the gold answer. That keeps a permissive but working grader -- one
    that accepts a wrong English sentence for an open-ended task, say -- from being reported as
    unable to rank, since the degenerate controls still fail it.

    Returns None when no control is provably wrong for this example, so the caller can exclude the
    episode instead of drawing a conclusion the evidence does not support.
    """
    # a text-free gold turn (a native tool call, an image-only block) carries no text a control
    # could collide with, so it cannot make a control unusable. keeping it would send every
    # candidate through _control_is_disjoint's empty-gold guard, return None for the episode, and
    # silently exclude it from the gate -- letting a flat grader pass unexamined (codex[bot]).
    scorable = [reference for reference in references if reference.strip()]
    if not scorable:
        return None
    usable = [
        control
        for control in _CONTROL_CANDIDATES
        # every gold turn has to be wrong under the control, since a multi-turn reward reads the
        # whole transcript: a control matching any single turn is not a wrong episode.
        if all(_control_is_disjoint(control, reference) for reference in scorable)
    ]
    # a gold answer drawing on several alphabets can disqualify the whole fixed set at once -- one
    # word rejects the English candidate, a "z" the repeated-z one, a "0" the repeated-zero one --
    # and returning None there excludes the episode, letting even a constant grader pass unexamined.
    # so synthesize a control from a character the gold answer does not use (codex[bot]).
    if not usable:
        synthetic = _synthetic_control(scorable)
        if synthetic is not None:
            usable = [synthetic]
    return usable or None


def _grade_gold_if_unscored(env, record: dict) -> None:
    """Score the gold rollout on its own when no batched control call produced its reward.

    Multi-turn gold is normally scored inside ``_score_multi_turn_episode``, in the same request
    list as the controls. Episodes that never reach that batch -- an echo policy, a partial replay,
    no provably wrong control, or a control whose scorer raised under the never-grades exemption --
    still need a reward to report, and scoring them here keeps the batched episodes to exactly one
    grader call. Grading gold both alone and in the batch would show a stateful or listwise grader a
    rollout the real run never submits.
    """
    if record["reward"] is None and record["rollout"] is not None:
        record["reward"] = _grade_rollouts(env, [record["rollout"]])[0]


def _controlled_scores(env, example: dict, record: dict) -> list[_Score] | None:
    """Score this episode's deliberately wrong answers, or None when none is provably wrong.

    For multi-turn this rescores the gold rollout in the same batch and writes the batched value
    back to ``record``, so the comparison is always between scores the grader produced from one
    request list. Comparing a singleton gold score against batched control scores would be the same
    listwise mismatch, moved one step (codex[bot]).
    """
    controls = _usable_controls(record["reference_turns"] or record["responses"][:1])
    if controls is None:
        return None
    if not env.multi_turn:
        return [_score_single_turn_control(env, example, control) for control in controls]
    gold, control_scores = _score_multi_turn_episode(env, example, record["rollout"], controls)
    record["reward"] = gold
    return control_scores


def _load_failure(reason: str) -> int:
    _err(f"env test failed: {reason}")
    print("0/0 episodes passed contract checks")
    return _err("overall: FAIL")


def _grades_completions(args) -> bool:
    """Whether the run this environment is for will actually grade a completion.

    The SFT worker builds rows from ``dataset()``, ``prompt_messages()`` and ``sft_completion()``
    and never scores anything (``flash/engine/worker/sft.py``); the OPD and OPSD workers read the
    same hooks and distil against a teacher, so they never grade either. Only GRPO reaches a scorer
    (``flash/engine/worker/rl.py``, ``flash/engine/multiturn_rollout.py``). Any of the others may
    legitimately ship a placeholder scorer that returns one constant, and failing that on reward
    quality would reject a working environment. This command has no config to infer intent from --
    hence the explicit flag. Unset means "unknown", which is not grounds to fail; the ranking check
    still reports its finding as a warning.
    """
    algorithm = getattr(args, "algorithm", None)
    if not algorithm or not str(algorithm).strip():
        return False
    return normalize_algorithm(str(algorithm).strip()) in _REWARD_CONSUMING_ALGORITHMS


def _never_grades(args) -> bool:
    """Whether this run's worker provably never calls the environment's scorer.

    The strict complement of "may grade", and NOT the negation of `_grades_completions`: that folds
    an unset algorithm in with the non-reward ones because neither is grounds to FAIL, whereas here
    an unset algorithm is unknown intent and must keep failing on a scorer it cannot exercise.
    """
    algorithm = getattr(args, "algorithm", None)
    if not algorithm or not str(algorithm).strip():
        return False
    return normalize_algorithm(str(algorithm).strip()) not in _REWARD_CONSUMING_ALGORITHMS


def _reads_per_turn_rewards(args) -> bool:
    """Whether the run this environment is for credits turns individually.

    `train.credit_assignment` defaults to ``per_episode`` (flash/spec.py), and `select_grpo_trainer`
    returns the ordinary scalar trainer for every mode but ``per_turn``
    (flash/engine/worker/rl.py), so on a default run the per-turn vectors are never read and the
    episode scalar is the whole reward. Accepting differing vectors as separation there would pass
    an env whose default run sees one constant episode score and computes zero advantages
    (codex[bot]).

    Unset means the default mode, matching what a config that omits the key would do.
    """
    mode = getattr(args, "credit_assignment", None)
    if not mode or not str(mode).strip():
        return False
    return str(mode).strip().lower() == PER_TURN_CREDIT_ASSIGNMENT


def cmd_env_test(args) -> int:
    """Load a local environment and drive deterministic offline contract checks.

    a fully non-returning environment hook (one that never yields control back) cannot be
    interrupted in-process, so run this under a ci job timeout to bound that class of
    defect; the per-episode turn cap (env.max_turns) bounds any cooperatively-stepping
    multi-turn loop exactly as the trainer does.
    """
    # settle the flags before touching the filesystem: a mistyped --algorithm is a fact about the
    # invocation, so it must be reported as one rather than shadowed by whatever the path check
    # happens to say about an unrelated argument (codex[bot]). normalize_algorithm raises
    # ValueError, which main() renders as an ordinary `error:` line.
    grades = _grades_completions(args)
    per_turn = _reads_per_turn_rewards(args)

    try:
        _, _, entrypoint, _ = _resolve_local_env_entrypoint(Path(args.path))
    except (Exception, SystemExit) as exc:
        reason = str(exc) or exc.__class__.__name__
        return _load_failure(reason.replace("cannot publish", "cannot test"))

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
    control_errors: list[str] = []
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
            # multi-turn gold is graded below, batched with its controls, so there is nothing to
            # check yet there. single-turn gold already has its score, and failing it here keeps a
            # broken grader reported as the reward violation it is rather than as a control error.
            if reward is not None and not reward.is_finite():
                raise ValueError(f"reward is not finite: {reward.episode}")
            if record["policy"] == "replay" and not record["partial_replay"]:
                # the absolute value of a gold reward proves nothing: the contract accepts any
                # finite scalar, so an env may legitimately score its reference 0.0 with worse
                # completions below it. what makes a grader unusable for RL is that it cannot
                # SEPARATE a good completion from a bad one, so score wrong answers and compare.
                try:
                    controls = _controlled_scores(env, example, record)
                except (Exception, SystemExit) as exc:
                    # a control that cannot be graded is a real defect wherever the scorer is
                    # actually called, and an unknown algorithm is not grounds to assume it is not
                    # -- so both keep failing the episode. only an algorithm whose worker provably
                    # never grades (sft/opd/opsd) is exempt: a reference-only scorer is legitimate
                    # there, and this is a fact about the CONTROL rather than about the episode the
                    # driver already replayed successfully (codex[bot]).
                    if not _never_grades(args):
                        raise
                    control_errors.append(str(exc) or exc.__class__.__name__)
                    controls = None
            # multi-turn gold gets its reward from the batch above when one ran; this covers every
            # other path. a gold reward that only goes non-finite in a batch is the same contract
            # violation as one that does alone -- it reaches training identically -- so the check
            # sits outside the control-error exemption, which covers facts about the CONTROL rather
            # than about the replayed episode.
            _grade_gold_if_unscored(env, record)
            reward = record["reward"]
            if reward is None or not reward.is_finite():
                raise ValueError(
                    f"reward is not finite: {reward if reward is None else reward.episode}"
                )
        except (Exception, SystemExit) as exc:
            failure = str(exc) or exc.__class__.__name__

        reward = record["reward"]
        reward_text = "n/a" if reward is None else f"{reward.episode:.6f}"
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
        if controls and reward is not None:
            controlled += 1
            # the group is the gold rollout plus its controls, and `build_per_turn_advantages`
            # falls back to episode advantages for the WHOLE group as soon as one member lacks a
            # vector. so the vectors decide only when every member has one; otherwise the episode
            # scalar is what trains, exactly as in the default credit-assignment mode.
            group_per_turn = per_turn and all(
                score.turns is not None for score in (reward, *controls)
            )
            if any(control.outranks(reward, per_turn=group_per_turn) for control in controls):
                # strictly worse than a deliberately wrong answer. GRPO maximizes this number, so
                # the run would train away from the gold answers -- a broken reward direction is
                # worse than a flat one, and no amount of separation elsewhere redeems it.
                inverted += 1
                # report the numbers the verdict actually read. on the per-turn path the episode
                # scalars are not what trains, and a crossing pair has equal ones -- printing them
                # would read "1.000000 scored higher than 1.000000".
                if group_per_turn:
                    offender = next(
                        control
                        for control in controls
                        if control.outranks(reward, per_turn=group_per_turn)
                    )
                    message = (
                        "a deliberately wrong answer was credited above the replayed gold answer "
                        f"at a turn the two share (gold turns {_fmt_turns(reward)}, wrong answer "
                        f"{_fmt_turns(offender)}); the reward direction looks inverted"
                    )
                else:
                    highest = max(control.episode for control in controls)
                    message = (
                        f"a deliberately wrong answer scored higher ({highest:.6f}) than the "
                        f"replayed gold answer ({reward.episode:.6f}); the reward direction looks "
                        "inverted"
                    )
                print(
                    render.warn(message) if render.styled() else f"warning: {message}",
                    file=sys.stderr,
                )
            elif not any(
                control.separates_from(reward, per_turn=group_per_turn) for control in controls
            ):
                scored_flat += 1
                scored_as = (
                    f"produced the same per-turn rewards {_fmt_turns(reward)}"
                    if group_per_turn
                    else f"scored {reward.episode:.6f}"
                )
                message = (
                    f"replay gold answer and {len(controls)} deliberately wrong answer(s) all "
                    f"{scored_as}; check the reward function"
                )
                print(
                    render.warn(message) if render.styled() else f"warning: {message}",
                    file=sys.stderr,
                )

    print(f"{passed}/{episode_count} episodes passed contract checks")
    if passed != episode_count:
        return _err("overall: FAIL")
    if control_errors:
        # not silent: the reward check ran on less evidence than it looks like it did, and a
        # reference-only grader is worth knowing about before the algorithm ever changes to grpo.
        message = (
            f"{len(control_errors)} episode(s) could not score a deliberately wrong answer "
            f"({control_errors[0]}), so the reward check skipped them. this scorer would fail "
            "under --algorithm grpo, which does grade arbitrary completions."
        )
        print(
            render.warn(message) if render.styled() else f"warning: {message}",
            file=sys.stderr,
        )
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

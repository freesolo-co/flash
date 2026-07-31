"""Offline contract checks for local Freesolo environments."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
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
# enough fallbacks for the unanimity test below to be reachable; matches the fixed candidate count.
_SYNTHETIC_CONTROL_COUNT = 3
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
    # whether each assistant turn actually emitted text, positionally. `build_per_turn_advantages`
    # skips a turn index whose span is zero-width, so a reward coordinate belonging to a text-free
    # turn is one the trainer never reads (see `_overlap`). None means "not known", which only
    # arises where there is no per-turn vector to qualify.
    emitted: tuple[bool, ...] | None = None

    def is_finite(self) -> bool:
        return math.isfinite(self.episode) and all(
            math.isfinite(value) for value in self.turns or ()
        )

    def _overlap(self, other: _Score) -> tuple[tuple[float, float], ...] | None:
        """Pair the turns both vectors actually reach and both credited, or None when neither has one.

        `build_per_turn_advantages` walks to the group's longest vector and, at each index, centres
        only the members present there. A shorter rollout therefore does not void the comparison:
        the turns both reached are still centred against each other. Comparing that overlap is what
        the trainer does, so a wrong control that merely terminated early stays evidence rather than
        being discarded (codex[bot]).

        A turn whose span is zero-width is excluded from that walk entirely
        (`grpo_perturn_trainer.py:67-75` filters members on `spans[i][1] > spans[i][0]`), so its
        reward coordinate is one no advantage is ever computed from. Counting it here read
        separation into a pair the trainer resolves to zero advantage at every turn -- a replay of
        ("", "answer") scoring (1, 0) against a control's (0, 0) looks separated at turn 0, while
        training drops gold from that turn, centres the identical controls alone, and learns
        nothing (codex[bot]).
        """
        mine, theirs = self.turns, other.turns
        if mine is None or theirs is None:
            return None
        pairs = zip(mine, theirs, strict=False)
        credited = zip(self.emitted or (), other.emitted or (), strict=False)
        # both sides must have emitted at an index for the trainer to compare them there. an absent
        # `emitted` means the caller could not say, and the pairs are taken as they were.
        if self.emitted is None or other.emitted is None:
            return tuple(pairs)
        return tuple(
            pair
            for pair, (mine_ok, theirs_ok) in zip(pairs, credited, strict=False)
            if mine_ok and theirs_ok
        )

    def outranks(self, gold: _Score, *, per_turn: bool) -> bool:
        """Report whether this deliberately wrong control beats gold where training reads.

        Asymmetric on purpose. ``_prepare_controls`` admits a control only when it is disjoint from
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


def _credited_turns(score: _Score, group: Sequence[_Score]) -> tuple[float, ...]:
    """This member's rewards at the turn indexes the trainer actually credits in `group`.

    A turn is credited when at least one member emitted text there: `build_per_turn_advantages`
    skips an index whose spans are all zero-width, and centres whoever remains at the others
    (`grpo_perturn_trainer.py:66-76`). A member absent from a credited turn contributes nothing
    there and is simply not represented in its own vector.
    """
    width = max((len(member.turns or ()) for member in group), default=0)
    credited = [
        index
        for index in range(width)
        if any((member.emitted or ())[index : index + 1] == (True,) for member in group)
    ]
    turns = score.turns or ()
    emitted = score.emitted
    return tuple(
        turns[index]
        for index in credited
        if index < len(turns) and (emitted is None or emitted[index])
    )


def _group_separates(gold: _Score, controls: Sequence[_Score], *, per_turn: bool) -> bool:
    """Whether training could tell ANY two members of this group apart.

    The group is the gold rollout plus its controls, and `build_per_turn_advantages` centres each
    credited turn against the members present there -- gold included only where it emitted. So at a
    turn gold sat out, the controls are centred against ONE ANOTHER, and controls scoring 1, 0, and
    -1 receive nonzero advantages from a group this function must therefore not call flat. Asking
    only whether each control separates from gold missed exactly that case, since `_overlap` drops
    the coordinate from every gold pairing (codex[bot]).

    On the episode path there is no such asymmetry, but the question is the same one -- any two
    members scoring differently is a group the trainer can rank -- so it is asked uniformly.
    """
    group = (gold, *controls)
    return any(
        member.separates_from(other, per_turn=per_turn)
        for index, member in enumerate(group)
        for other in group[index + 1 :]
    )


def _scores_zero(gold: _Score, controls: Sequence[_Score], *, per_turn: bool) -> bool:
    """Whether this group's tie sits at zero, where training reads it.

    The all-zero tie is the finding this gate was built for (LS-005): a grader returning nothing for
    its own reference answer is broken or missing a dependency, and no reward shape explains it. A
    tie at any other value has an innocent reading and is only warned about.
    """
    group = (gold, *controls)
    if per_turn:
        return all(value == 0.0 for member in group for value in _credited_turns(member, group))
    return all(member.episode == 0.0 for member in group)


def _fmt_credited_turns(gold: _Score, controls: Sequence[_Score]) -> str:
    """Gold's rewards at the credited turns, which is what the flat finding compared.

    Printing the raw vector instead described turns the comparison had already dropped, so a group
    the trainer genuinely cannot rank was reported with numbers that visibly differ (cursor).
    """
    values = _credited_turns(gold, (gold, *controls))
    return "(" + ", ".join(f"{value:.6f}" for value in values) + ")"


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


def _reference_turns(env, example: dict) -> tuple[list[str], bool, list[dict]]:
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
    # the messages are returned alongside the replay text so the observation check can reuse THIS
    # snapshot. calling sft_completion again for the same episode asks a second time for something
    # an env is free to answer differently -- one that samples a stored trajectory or consumes an
    # iterator returns a different completion, and the driven rollout would then be compared against
    # observations belonging to a trajectory it never replayed, marking an exact replay
    # `partial_replay` and dropping it from the control gate (codex[bot]).
    return [_message_text(m["content"]) for m in assistant], partial, messages


def _resolve_policy(reference_turns: list[str]) -> str:
    return "replay" if "".join(reference_turns).strip() else "echo"


def _observation(message: dict) -> tuple[str, str]:
    """One environment-side turn as the replay check compares it: its role and its text.

    The role is carried, not just the text. A reference recording ``{"role": "tool", ...}`` against
    a rollout emitting the same string as ``{"role": "user", ...}`` is a materially different
    transcript -- it renders differently under the chat template and a role-aware grader scores it
    differently -- so reducing both sides to content alone reported a faithful replay for a
    trajectory the grader never saw (codex[bot]).
    """
    return str(message.get("role", "")).strip().lower(), _message_text(message.get("content"))


def _observation_blocks(
    messages: list[dict], *, skip_messages: int = 0
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """The environment-side turns, grouped by the assistant turn each block follows.

    Grouping is what makes an extra observation visible. Flattened, an env emitting one unexpected
    message between two turns is indistinguishable from one whose next turn simply replied twice;
    kept in blocks, the two differ in the block that holds the extra.

    ``skip_messages`` drops that many leading messages outright, which is how the prompt is excluded
    from the driven side. It is a POSITIONAL prefix, not a filtered count: the rollout state's
    transcript begins as a copy of the prompt (flash/envs/adapter.py:392), so the opening is exactly
    the first ``len(prompt)`` entries whatever roles they hold. Counting only the prompt's
    non-assistant messages instead meant a prompt ENDING in an assistant turn -- a prefill, which
    flash/engine/multiturn_rollout.py copies through unfiltered -- left that turn to open a block,
    shifting every reference block by one and marking a faithful replay `partial_replay`
    (codex[bot]).
    """
    blocks: list[tuple[tuple[str, str], ...]] = []
    current: list[tuple[str, str]] = []
    for message in messages[skip_messages:]:
        if str(message.get("role", "")).strip().lower() == "assistant":
            blocks.append(tuple(current))
            current = []
            continue
        current.append(_observation(message))
    blocks.append(tuple(current))
    # the leading block holds whatever preceded the first assistant turn. on the driven side the
    # prompt is skipped by position and it is empty; a reference that opens with an observation keeps
    # it, and both sides are built the same way, so they stay aligned.
    return tuple(blocks)


def _env_turns_reproduce(reference_messages: list[dict], state: dict, prompt: list[dict]) -> bool:
    """Whether the driven rollout's environment-side turns match the reference trajectory's.

    Only meaningful when the reference records them: a gold completion of assistant turns alone
    says nothing about what the env replied, so there is nothing to contradict and the replay
    stands. When it does record them, they must come back at the same positions with the same
    role and text -- a stochastic or externally-sourced observation otherwise reaches the grader as
    a different episode wearing the reference's assistant strings.

    The comparison is per assistant turn, not over one flattened list. Both sides are cut into the
    blocks of observations that followed each assistant turn, and each block the reference records
    must come back whole: same observations, same order, no extras among them. Reading the driven
    side as a flat prefix accepted an env that interleaved an unexpected system or tool message --
    reference a1/x/a2 against a driven a1/x/<extra>/a2 -- as a faithful replay, though the grader
    received a materially different episode and could score this "gold" transcript like the
    controls (codex[bot]).

    Only what follows the reference's last block is unconstrained: the turn loop appends one more
    env_reply after the final assistant turn, past where the gold transcript stops recording, so it
    is outside what the trajectory claims and is not evidence either way.

    `prompt` is the rollout's opening messages, captured BEFORE the turn loop ran. It cannot be
    recovered from `state` afterwards: an env that keeps its transcript under `messages` with no
    separate `prompt` key has by then appended every driven turn to that same list, so there is
    nothing left to distinguish the opening from the completion.

    `reference_messages` is the SAME snapshot `_reference_turns` took the replayed assistant strings
    from, threaded in rather than re-read. `sft_completion` is not required to be pure -- one that
    samples a stored trajectory or consumes an iterator answers differently each call -- so asking
    it again here compared the driven rollout against observations from a trajectory it never
    replayed (codex[bot]).
    """
    # the same role filter on both sides. dropping system turns from the driven side alone made an
    # env whose env_reply emits one look unreproduced on an exact replay, since the reference keeps
    # it (codex[bot]). the opening prompt is excluded by position below instead.
    reference = _observation_blocks(reference_messages)
    if not any(reference):
        return True
    # align on where the completion begins, not on where the transcript ends. the prompt opens the
    # driven transcript and is not part of the completion, so it is dropped as a positional prefix;
    # what follows is compared positionally. the turn loop appends one more env_reply after the
    # final assistant turn, so anchoring on the tail instead would shift every observation by that
    # trailing reply -- marking a faithful replay unreproduced, and letting a coincidental trailing
    # match hide a divergence earlier in the trajectory (cursor).
    driven = state.get("messages") or []
    completion = _observation_blocks(driven, skip_messages=len(prompt))
    # a reference recording more turns than were driven cannot match either way.
    if len(completion) < len(reference):
        return False
    # every block the reference closed with an assistant turn must come back whole, extras
    # included. its LAST block is the one recording stopped inside, so it is compared as a prefix:
    # the loop appends one more env_reply after the final assistant turn, past anything the gold
    # transcript claims.
    if completion[: len(reference) - 1] != reference[:-1]:
        return False
    tail = reference[-1]
    return completion[len(reference) - 1][: len(tail)] == tail


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
        # multi-turn only: the terminal (example, state, responses) the gold reward comes from, so
        # it can be scored alongside every control in one batched call.
        "rollout": None,
        # multi-turn only: the driven control rollouts, in the same shape, awaiting that same batch.
        # None means no control was provably wrong for this example.
        "control_rollouts": None,
        # the control scores, once that batch has run. None means the episode carries no evidence.
        "control_scores": None,
        # controls that were graded but earn no advantage. not evidence, but still group members,
        # so they decide whether the group takes the per-turn path (see the flat-reward gate).
        "unscorable_controls": (),
    }


def _drive_single_turn(env, example: dict, record: dict) -> None:
    prompt = _check_messages(env.prompt_messages(example), "prompt")
    record["prompt"] = prompt
    # single-turn drives one assistant reply and never checks observations, so the snapshot the
    # multi-turn path threads through has no second reader here.
    reference_turns, record["partial_replay"], _ = _reference_turns(env, example)
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


def _run_rollout(env, example: dict, turn_content) -> tuple[dict, list[str], list[dict]]:
    """Drive one offline multi-turn rollout, returning the final state, turns taken, and prompt.

    `turn_content(index)` supplies the model's text for each turn, so the same loop can replay a
    gold transcript or a deliberately wrong one.

    The opening messages are snapshotted before the loop runs, and for the same reason the worker
    reads them there (`flash/engine/multiturn_rollout.py`): an env may expose its transcript as
    `messages` with no separate `prompt` key, and once the loop has appended to that list the
    opening is no longer recoverable from the state.
    """
    state = env.new_rollout_state(example)
    # `prompt` or `messages`, exactly as the production driver accepts them
    # (flash/engine/multiturn_rollout.py:171-175). requiring `prompt` afterwards failed a state
    # shape supported everywhere else in the codebase, before the reward gate ever ran (codex[bot]).
    prompt = [
        dict(m) for m in _check_messages(state.get("prompt") or state.get("messages"), "prompt")
    ]
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
    return state, responses, prompt


def _drive_multi_turn(env, example: dict, record: dict) -> None:
    reference_turns, record["partial_replay"], reference_messages = _reference_turns(env, example)
    policy = _resolve_policy(reference_turns)
    record["policy"] = policy

    def content(index: int) -> str:
        if policy == "replay" and index < len(reference_turns):
            return reference_turns[index]
        return _ECHO_RESPONSE

    state, responses, prompt = _run_rollout(env, example, content)
    # the snapshot, not a re-read of the state: for an env exposing only `messages` the loop has
    # appended every driven turn to that list by now, so re-reading would record the whole
    # transcript as the prompt and report it back to the user as one.
    record["prompt"] = prompt
    record["responses"].extend(responses)
    record["turns"] = len(responses)
    record["reference_turns"] = reference_turns
    # the env kept the rollout going past the gold transcript, so the tail of `responses` is echo
    # filler rather than the reference. the reward below then grades a transcript no correct policy
    # would produce, which is not the gold reward and must not feed the flat-reward gate -- the same
    # reason a partially representable transcript is excluded.
    if policy == "replay" and len(responses) != len(reference_turns):
        record["partial_replay"] = True
    # the assistant turns are only half the transcript. an env whose observations do not reproduce
    # -- a stochastic one, or one reading outside state -- hands the grader a different episode
    # under the same assistant strings, so a correct grader scores this "gold" rollout like the
    # controls and the gate reports a flat grader for an env that ranks fine (codex[bot]). the
    # reference records what the env replied last time, so compare against it where it does.
    if policy == "replay" and not _env_turns_reproduce(reference_messages, state, prompt):
        record["partial_replay"] = True

    # the terminal rollout, kept unscored: the whole run's rollouts are scored in one batch so a
    # listwise grader ranks them together, exactly as the worker submits them (codex[bot]). see
    # _score_multi_turn_rollouts.
    record["rollout"] = (example, state, responses)


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


def _synthetic_controls(references: list[str]) -> list[str]:
    """Build wrong answers for gold text that collides with every fixed control.

    Each repeats a single character the gold answers never use. That character cannot appear in any
    gold word, so `_control_is_disjoint` holds by construction rather than by luck, and the
    repetition keeps the control long enough to be an implausible answer for an open-ended task.

    Several are built rather than one, from distinct characters, because the inversion verdict reads
    unanimity across controls drawing on mutually exclusive alphabets. A lone fallback could never
    reach that bar, so an inverted grader passed unexamined for any gold text that disqualified the
    whole fixed set -- "answer z 0" being enough to do it (codex[bot]). Returns an empty list when
    the references between them use the whole alphabet, so the caller still excludes an episode it
    cannot control rather than inventing evidence.
    """
    used = {character for reference in references for character in reference.casefold()}
    controls = []
    for character in _SYNTHETIC_CONTROL_ALPHABET:
        if character in used:
            continue
        control = character * _SYNTHETIC_CONTROL_WIDTH
        # cheap belt-and-braces: the caller's own disjointness rule is the contract, so ask it
        # rather than trusting the construction to stay equivalent to it.
        if all(_control_is_disjoint(control, reference) for reference in references):
            controls.append(control)
        if len(controls) == _SYNTHETIC_CONTROL_COUNT:
            break
    return controls


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


def _emitted_turns(responses: list[str]) -> tuple[bool, ...]:
    """Which assistant turns put tokens on the wire, positionally.

    The production rollout appends one span per assistant turn and the per-turn trainer skips any
    whose span is zero-width (`multiturn_rollout.py:192-195`, `grpo_perturn_trainer.py:67-75`). Here
    the turn text is replayed rather than generated, so the analogue of "no tokens" is the empty
    string -- whitespace still tokenizes to something and is credited in a real run.
    """
    return tuple(bool(response) for response in responses)


def _grade_rollouts(env, rollouts: list[tuple[dict, dict, list[str]]]) -> list[_Score]:
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

    Each rollout carries its turn TEXTS rather than just a count, so the score records which turns
    emitted anything -- the coordinates the per-turn trainer actually reads.
    """
    from flash.engine.multiturn_reward_scoring import RolloutScoreRequest, score_rollouts

    requests = [
        RolloutScoreRequest(example=example, state=state, turn_count=len(responses))
        for example, state, responses in rollouts
    ]
    rewards = score_rollouts(env, requests)
    return [
        _Score(episode=float(reward.episode), turns=reward.turns, emitted=_emitted_turns(responses))
        for reward, (_example, _state, responses) in zip(rewards, rollouts, strict=True)
    ]


def _score_single_turn_control(env, example: dict, control: str) -> _Score | None:
    """Score one deliberately wrong single-turn answer the way training would score it.

    For single-turn GRPO a grader that raises is not inconclusive: ``reward_fn`` catches exactly
    that and scores the completion 0.0 (flash/engine/worker/rl.py), so training would see a real
    number. Record that same 0.0, otherwise a row that genuinely separates gold from wrong is
    dropped and a later tied row can fail the whole sample alone.

    ``SystemExit`` still propagates. ``reward_fn`` catches only ``Exception``, so a grader that
    exits on an unexpected completion kills training rather than scoring it zero -- and the gold
    replay is already failed for exactly that (codex[bot]).

    Returns None for a NaN score. NaN is the trainer's supported unscorable marker -- trl excludes
    such a row from the group baseline and zeroes its advantage -- so a grammar-constrained scorer
    marking a synthetic control unscorable is behaving as designed, and the row it produces earns no
    advantage to be evidence of. Infinity is NOT that marker: it is not recognized as unscorable and
    reaches the group as a real number, contaminating every advantage in it, so it still raises
    (codex[bot]).
    """
    try:
        score = _Score(episode=_grade_single_turn(env, control, example))
    except Exception:
        # mirrors reward_fn's own except branch: the run would score this 0.0 and carry on.
        score = _Score(episode=0.0)
    if math.isnan(score.episode):
        return None
    if not score.is_finite():
        raise ValueError(f"reward is not finite for a non-reference completion: {score.episode}")
    return score


def _drive_multi_turn_controls(
    env, example: dict, controls: list[str]
) -> list[tuple[dict, dict, list[str]]]:
    """Drive one wrong episode per control, leaving them unscored for the caller's batch.

    A multi-turn reward reads the accumulated rollout state, so a comparable wrong episode has to be
    driven, not assembled -- replay the same loop answering each control at every turn. Scoring is
    deliberately left to the caller, which submits every episode's gold and controls together (see
    ``_score_multi_turn_rollouts``).
    """
    return [
        (example, state, responses)
        for state, responses, _prompt in (
            _run_rollout(env, example, lambda _index, text=control: text) for control in controls
        )
    ]


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
        usable = _synthetic_controls(scorable)
    return usable or None


def _score_multi_turn_rollouts(env, records: list[dict]) -> None:
    """Score every multi-turn rollout of the whole run -- gold and controls -- in ONE call.

    The worker submits its entire rollout request list to ``score_rollouts`` at once
    (flash/engine/multiturn_rollout.py:687-695), and that list spans the whole generation batch, not
    one example. A ``rollout_rewards_many`` that normalizes or ranks across the list it is handed
    therefore returns different numbers when called once per episode, which is enough to read a
    working grader as flat or as inverted (codex[bot]). So every driven rollout in the run goes into
    a single request list here, and the results are mapped back by position.

    Unlike single-turn, a raising scorer is not softened to 0.0: the rollout path calls
    ``score_rollouts`` with no except branch, so a scorer that raises there aborts the run.
    Swallowing it would pass an environment that cannot survive its first rollout (codex[bot]). A
    raise here belongs to every episode in the batch for the same reason -- production makes exactly
    this one call.

    A control the grader marks unscorable is dropped rather than failed. ``score_rollouts`` turns a
    non-finite episode into NaN deliberately -- it is the trainer's supported marker for a row the
    group baseline excludes and whose advantage is then zeroed -- so an env that marks completions
    outside its grammar unscorable is behaving as designed, not violating the contract. Failing on
    it rejected such an env for the single reason that a fixed control is not valid input for it
    (codex[bot]).
    """
    batch: list[tuple[dict, dict, list[str]]] = []
    # (record, slice of `batch` holding its controls) so each result returns to its own episode.
    layout: list[tuple[dict, slice]] = []
    for record in records:
        if record["rollout"] is None:
            continue
        batch.append(record["rollout"])
        controls = record["control_rollouts"] or []
        start = len(batch)
        batch.extend(controls)
        layout.append((record, slice(start, start + len(controls))))
    if not batch:
        return
    scores = _grade_rollouts(env, batch)
    gold_index = 0
    for record, control_slice in layout:
        record["reward"] = scores[gold_index]
        gold_index = control_slice.stop
        if record["control_rollouts"] is None:
            continue
        # an unscorable control earns no advantage, so it is evidence of neither ranking nor
        # flatness. keeping it would compare gold against a number the trainer never acts on.
        graded = list(scores[control_slice])
        record["control_scores"] = [score for score in graded if score.is_finite()]
        # the dropped ones are still group MEMBERS, and one without a turn vector demotes the whole
        # group to episode scalars in the trainer. so they are carried rather than discarded: they
        # decide which reward path the group actually takes, even though they are evidence of
        # nothing themselves (codex[bot]).
        record["unscorable_controls"] = [score for score in graded if not score.is_finite()]


def _prepare_controls(env, example: dict, record: dict) -> list[_Score] | None:
    """Prepare this episode's deliberately wrong answers, or None when none is provably wrong.

    Single-turn scores them here and returns the scores. Multi-turn only DRIVES them, parking the
    rollouts on the record for the whole run's single batched scoring call
    (``_score_multi_turn_rollouts``) and returning None -- the caller reads ``control_scores`` after
    that batch instead.

    May also return an empty list, when every control was scored but none produced usable evidence.
    The caller treats that the same way it treats None -- the episode is not counted as controlled,
    so the gate never speaks for an episode it could not actually test.
    """
    controls = _usable_controls(record["reference_turns"] or record["responses"][:1])
    if controls is None:
        return None
    if not env.multi_turn:
        # an unscorable control is dropped, exactly as on the multi-turn path: it earns no
        # advantage, so it is evidence of neither ranking nor flatness.
        scored = [_score_single_turn_control(env, example, control) for control in controls]
        return [score for score in scored if score is not None]
    record["control_rollouts"] = _drive_multi_turn_controls(env, example, controls)
    return None


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
    scored_zero = 0
    inverted = 0
    control_errors: list[str] = []
    # driven first, scored second, reported third. every multi-turn rollout of the run -- gold and
    # controls, across all episodes -- is scored between the passes in the one batch shape the
    # worker submits (see _score_multi_turn_rollouts), so nothing here shows the grader a smaller
    # request list than the real run makes (codex[bot]).
    episodes: list[tuple[int, dict, str | None, list[_Score] | None]] = []
    for index, example in enumerate(dataset[:episode_count], start=1):
        record = _new_record()
        failure: str | None = None
        controls: list[_Score] | None = None
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
                    controls = _prepare_controls(env, example, record)
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
        except (Exception, SystemExit) as exc:
            failure = str(exc) or exc.__class__.__name__
        episodes.append((index, record, failure, controls))

    # one scoring call for the whole run, gold rollouts and their controls together: that is the
    # request shape the worker submits, and it spans the generation batch rather than one example
    # (see _score_multi_turn_rollouts). a raise belongs to every episode in it, since production
    # makes exactly this one call and it would abort the run.
    graded = [record for _, record, failure, _ in episodes if not failure]
    batch_failure = None
    try:
        _score_multi_turn_rollouts(env, graded)
    except (Exception, SystemExit) as exc:
        batch_failure = str(exc) or exc.__class__.__name__
        # the controls are this command's own additions, not rollouts any run would submit. under
        # the never-grades exemption a scorer that only chokes on them is a fact about the CONTROL,
        # so drop them and score the gold rollouts training really does submit.
        affected = [record for record in graded if record["control_rollouts"]]
        if _never_grades(args) and affected:
            for record in affected:
                record["control_rollouts"] = None
            try:
                _score_multi_turn_rollouts(env, graded)
            except (Exception, SystemExit):
                pass
            else:
                control_errors.extend(batch_failure for _ in affected)
                batch_failure = None

    for index, record, failure, controls in episodes:
        # multi-turn controls are scored in that batch rather than inline, so they arrive here.
        if controls is None:
            controls = record["control_scores"]
        # graded, earn no advantage, and are excluded from every comparison -- but still members of
        # the group whose reward path is chosen below.
        unscorable = record["unscorable_controls"] or ()
        reward = record["reward"]
        if not failure:
            # a gold reward that only goes non-finite in a batch is the same contract violation as
            # one that does alone -- it reaches training identically -- so the check sits outside
            # the control-error exemption, which covers facts about the CONTROL rather than about
            # the replayed episode.
            if batch_failure and reward is None:
                failure = batch_failure
            elif reward is None or not reward.is_finite():
                failure = f"reward is not finite: {reward if reward is None else reward.episode}"
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
            # `unscorable` are the controls dropped for earning no advantage. they are excluded from
            # every comparison below, but they are still MEMBERS of the group the trainer builds,
            # and `build_per_turn_advantages` demotes the whole group to episode scalars as soon as
            # one member has no turn vector (grpo_perturn_trainer.py:59-63). a non-finite control is
            # represented `episode=NaN, turns=None`, so judging the path on the survivors alone read
            # per_turn where production falls back to the tied scalars and produces no learning
            # signal (codex[bot]).
            group_per_turn = per_turn and all(
                score.turns is not None for score in (reward, *controls, *unscorable)
            )
            # every control outranking gold is worth reporting, but it does NOT establish an
            # inverted grader, so it is reported and never failed on. the controls are disjoint from
            # the gold text LEXICALLY; they are not known-negative answers, and they still share
            # non-lexical properties -- all three run to 64-67 characters, for one. a healthy
            # open-ended grader rewarding response length outscores a short gold reference with
            # every one of them, and maximizing that reward is correct (codex[bot]). only the
            # environment knows which completions are genuinely wrong for its task, and this command
            # has no hook to ask, so the finding stays a warning rather than a verdict.
            outranking = [
                control for control in controls if control.outranks(reward, per_turn=group_per_turn)
            ]
            if len(controls) > 1 and len(outranking) == len(controls):
                inverted += 1
                # report the numbers the finding actually read. on the per-turn path the episode
                # scalars are not what trains, and a crossing pair has equal ones -- printing them
                # would read "1.000000 scored higher than 1.000000".
                if group_per_turn:
                    observed = (
                        "every deliberately wrong answer was credited above the replayed gold "
                        f"answer at a turn they share (gold turns {_fmt_turns(reward)}, wrong "
                        f"answer {_fmt_turns(outranking[0])})"
                    )
                else:
                    highest = max(control.episode for control in outranking)
                    observed = (
                        f"every deliberately wrong answer scored higher (up to {highest:.6f}) than "
                        f"the replayed gold answer ({reward.episode:.6f})"
                    )
                message = (
                    f"{observed}; the reward direction may be inverted, though a grader rewarding "
                    "an open-ended property these answers share -- their length, say -- would look "
                    "the same. check the grader's sign"
                )
                print(
                    render.warn(message) if render.styled() else f"warning: {message}",
                    file=sys.stderr,
                )
            elif not _group_separates(reward, controls, per_turn=group_per_turn):
                scored_flat += 1
                # a tie is only conclusive when it sits at zero. that is the signature this gate
                # exists for -- a grader that cannot recognize its OWN reference answer, from a
                # broken scorer or a missing runtime dependency -- and no property of the controls
                # explains it away. above zero the tie has an innocent reading: the controls are
                # only LEXICALLY disjoint from gold, so a healthy grader rewarding a property they
                # happen to share (a safety scorer awarding 1 to anything without a prohibited
                # phrase, say) ties them with gold while ranking real completions fine (codex[bot]).
                if _scores_zero(reward, controls, per_turn=group_per_turn):
                    scored_zero += 1
                scored_as = (
                    # the credited turns, not the raw vector: a coordinate no member emitted at is
                    # dropped before the comparison, so printing the whole vector claimed rewards
                    # were identical where they visibly differ (cursor).
                    f"produced the same rewards at every turn training reads "
                    f"({_fmt_credited_turns(reward, controls)})"
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
    if inverted:
        # reported, never failed on. a wrong answer outscoring gold is worth a look, but the
        # controls are only LEXICALLY disjoint from the gold text -- they are not known-negative
        # answers, and a grader rewarding an open-ended property they happen to share (length, for
        # one: they run 64-67 characters) outranks a short gold reference while being exactly right
        # to maximize (codex[bot]). failing here would block a working environment, and nothing this
        # command can observe tells the two cases apart.
        message = (
            f"{inverted} replayed episode(s) scored a deliberately wrong answer higher than the "
            "gold answer. check the grader's sign -- though a reward for an open-ended property "
            "those answers share would look the same and be correct."
        )
        print(
            render.warn(message) if render.styled() else f"warning: {message}",
            file=sys.stderr,
        )
    # a grader that returns the same score for its own gold answer and for every deliberately wrong
    # one produces zero advantage on every sampled episode, so the run reaches a gpu unable to learn
    # what the dataset teaches. episodes whose transcript the driver could not reproduce verbatim,
    # and those with no control provably wrong for their gold answer, are excluded above. this
    # samples the first few rows, so the finding is deliberately a claim about the sample:
    # separation anywhere in it is enough to pass.
    #
    # conclusive only when the tie sits at ZERO -- the LS-005 signature this gate was built for, a
    # grader scoring nothing for its own reference answer, which is a broken scorer or a missing
    # runtime dependency however the reward is shaped. a tie at any other value is reported and
    # never failed on: the controls are only LEXICALLY disjoint from gold, so a healthy grader
    # rewarding a property they happen to share ties with them while ranking sampled completions
    # perfectly well. a safety scorer awarding 1 to any response without a prohibited phrase does
    # exactly that, and failing it would block a working environment (codex[bot]).
    #
    # only failed for an algorithm that actually consumes reward(). SFT never calls it, so a
    # placeholder scorer there is not a defect, and without --algorithm the intent is unknown --
    # report the finding without failing rather than block a working environment.
    finding = ""
    if controlled and scored_flat == controlled:
        finding = (
            f"all {controlled} replayed episode(s) scored every deliberately wrong answer exactly "
            "as high as the gold answer; the reward function cannot rank completions. check the "
            "grader and that its runtime dependencies are installed in this environment."
        )
    conclusive = bool(finding) and scored_zero == controlled
    if finding and grades and conclusive:
        _err(finding)
        return _err("overall: FAIL")
    if finding and not conclusive:
        # `not conclusive` is `scored_zero != controlled`, which covers a MIXED sample too -- some
        # episodes tying at zero, some above it. claiming every score was non-zero was then simply
        # false, and it named the wrong episodes to go looking at (cursor). so describe what was
        # actually seen: the reason this is not failed on is that at least one tie was above zero,
        # which is what a grader rewarding a shared property produces.
        tied_above_zero = controlled - scored_zero
        detail = (
            f"{tied_above_zero} of {controlled} tied above zero"
            if scored_zero
            else "every score was the same non-zero value"
        )
        message = (
            f"{finding} {detail}, which a grader rewarding a property these answers share would "
            "also produce, so this is not failed on."
        )
        print(
            render.warn(message) if render.styled() else f"warning: {message}",
            file=sys.stderr,
        )
    elif finding:
        message = f"{finding} pass --algorithm to fail on this instead of warning."
        print(
            render.warn(message) if render.styled() else f"warning: {message}",
            file=sys.stderr,
        )
    print(render.ok("overall: PASS") if render.styled() else "overall: PASS")
    return 0

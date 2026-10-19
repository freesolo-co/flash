"""Advisory warnings for `flash env test`.

Each of these reports a run that passed every contract check while measuring nothing, or measuring
something the verdict does not describe. They are warnings rather than blocking gates: the shapes
they catch are strong evidence of a broken environment but not proof, and `flash env test` already
fails hard on the cases that are certain.

Kept beside `test.py` rather than inside it because the file is at its size limit and these three
checks form one cohesive group: they share the "passed, but nothing was learned" theme and none of
them is referenced by the driving loop except through the entry points re-exported here.
"""

from __future__ import annotations

import sys

from flash.cli.ui import render

# the one algorithm that trains from the environment reward. sft optimizes a supervised loss and opd
# a teacher token loss; neither reads `env.reward`, so a scorer they never call cannot be evidence
# of anything for them.
_REWARD_DRIVEN_ALGORITHM = "grpo"

# the roles whose adjacency collapses into one rendered block. `tool` is deliberately absent: one
# assistant message carrying two parallel tool calls is answered by one `tool` message per call,
# which is the required wire format and what chat templates render as separate delimited blocks --
# not a collapse. flagging it would train users to edit a correct transcript into a broken one, or
# to ignore the warning entirely.
#
# `user` and `system` stay in. two user turns in a row is how an off-by-one trajectory capture shows
# up (the completion re-states the question the prompt already asked), and it duplicates that text
# in the trained string exactly as a doubled assistant turn does.
_MERGEABLE_ROLES = frozenset({"assistant", "user", "system"})


def _emit(message: str) -> None:
    """Print one advisory warning to stderr, styled when the terminal supports it."""
    print(render.warn(message) if render.styled() else f"warning: {message}", file=sys.stderr)


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


def _warn_on_low_replay_reward(record: dict, reward: float) -> None:
    """Warn when a replayed gold answer scored at or below zero, and show what was graded."""
    if reward > 0.0:
        return
    # two different faults produce this same zero, and naming only the grader sends readers to edit
    # a scorer that is working. the gold completion is the other candidate: `sft_completion`
    # defaults to the row's raw `output` (flash/envs/loading/adapter.py), so a dataset whose `output` is a
    # bare value replays that value verbatim -- which a grader requiring a wrapper (`\boxed{}`, a
    # json object, a tag) is right to score zero.
    _emit(
        f"replay gold answer scored low (reward={reward:.6f}); "
        "check the reward function or the gold completion it scored"
    )
    # printed with repr rather than through `_preview`: the whole point of showing this text is to
    # expose a formatting defect, and `_preview` collapses whitespace and truncates, so a stray
    # newline or a `\boxed{}` past the cutoff -- exactly the faults this line exists to reveal --
    # would be invisible. labelled by what it is -- the text the grader received -- not "gold
    # answer": when a multi-turn env overrides the response via `final_response_text`, the scored
    # text is env-authored and calling it the gold answer sends the reader to edit a dataset row
    # that was never scored.
    print(f"  scored text: {_scored_text(record)!r}", file=sys.stderr)


def _warn_on_unfinished_replay(record: dict) -> None:
    """Warn when the gold trajectory used every turn without the environment reporting done.

    A gold trajectory that never terminates is the signature of an env no rollout can finish --
    every move applied twice, a win condition that cannot be reached. The reward alone hides it: a
    partial-credit grader pays a respectable score for the capped attempt, and gold-vs-junk still
    clears its bar, so the run reads PASS. What exposes it is the turn count, which is why it is
    printed as a comparison.

    How far the reference got decides what may be CLAIMED. When it covered the whole episode and the
    env still never reported done, the env is the only remaining explanation. When it ran out first
    the driver padded the tail with junk, and junk cannot drive even a healthy env to termination --
    so a short reference on a working env looks identical from here. Both are worth surfacing and
    neither is worth misdiagnosing, so the incomplete case names both explanations and puts the
    cheaper one first: extending a dataset row costs less than auditing termination logic.
    """
    # checked BEFORE the unfinished gate, not inside it. this is not a reading of an ambiguous
    # situation -- the reference is longer than the cap, so turns the dataset defines were dropped
    # and never replayed. that is true whether or not the capped prefix happened to terminate the
    # episode: an env that declares done on its third turn against a five-turn reference leaves
    # `hit_turn_cap` false and every other check passing, so gating this on the unfinished path
    # made those two dropped turns silent. both numbers are named -- an author cannot act on
    # "raise the cap" without knowing the target.
    if record["gold_exceeds_cap"]:
        # phrased on the DROP, not on "never finished": when the prefix terminated the episode the
        # environment did finish, and saying otherwise would be false.
        #
        # the replayed count is `turns`, NOT `turn_cap`. the cap is what the episode was ALLOWED,
        # and an env that declares done before reaching it replays fewer -- quoting the cap then
        # overstated the replay (3 of 5 when only 2 were sent) and sent the author looking for a
        # third turn that was never scored.
        _emit(
            f"replay gold answer truncated: the gold trajectory is {record['reference_turns']} "
            f"turn(s) but the episode is capped at {record['turn_cap']}, so only the first "
            f"{record['turns']} were replayed and the rest were never scored. this is a "
            "cap/dataset mismatch, not an environment fault: raise the turn cap (`max_turns`, or "
            "the row's `max_episode_turns`) to at least the length of the gold trajectory, or "
            "shorten the row"
        )
        return
    if not record["hit_turn_cap"]:
        return
    opening = (
        f"replay gold answer never finished: it used all {record['turns']} turn(s) and the "
        "environment never reported the episode done"
    )
    if record["replay_incomplete"]:
        _emit(
            f"{opening}, but the reference ran out before the cap and the remaining turns were "
            "filler, which cannot finish an episode either. so this is either a reference that "
            "covers only part of a trajectory or an environment that never terminates; check "
            "whether the dataset row is complete first, then the episode termination condition"
        )
        return
    # even here the environment is not the only explanation. `rollout_done` (flash/envs/loading/adapter.py)
    # returns True at `turn >= cap` REGARDLESS of `state["done"]`, so a fixed-length environment
    # that ends purely by exhausting its budget and never sets `done` is a supported shape that
    # trains fine -- and from here it is indistinguishable from one no rollout can finish, since
    # both replay every turn and both leave `done` unset. what separates them is whether the
    # environment considers the outcome a success, which the reward reports: the dead board that
    # motivated this warning paid partial credit (0.60-0.65), while a fixed-length game pays its
    # normal score. that is a difference of degree, not of kind, so it cannot carry a verdict --
    # the reward is quoted and the reader decides.
    _emit(
        f"{opening} (reward={record['reward']:.6f}). if the episode is meant to end on a win "
        "condition, check the termination condition and whether each turn is applied exactly once "
        "-- a reference that cannot complete an episode means no rollout can either. if it is a "
        "fixed-length episode that ends by using its whole budget, this is expected and the score "
        "above is the thing to trust"
    )


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
        if role in _MERGEABLE_ROLES and role == previous:
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

    Only repeats INSIDE the completion are reported, and the prompt is read solely to know what the
    completion's first turn follows. The other two regions have healthy forms this cannot tell from
    broken ones, and neither is something `sft_completion` authorship can fix:

    - inside `prompt_messages`: a retrieved document in its own user message, or a task and its
      reply-format instruction as separate user turns, are ordinary prompt shapes that every chat
      template renders as separate delimited blocks. Flagging them accused a correct prompt and
      pointed at a completion that restates nothing.
    - at the seam, ONLY when the doubled role is `assistant`: a prompt ending on an assistant
      prefill CONTINUED by the completion is the one adjacency meant to merge -- that is what a
      prefill is -- and the remedy this warning prints would destroy it. A doubled `user` or
      `system` turn at the same boundary has no such form: it is the completion restating a turn the
      prompt already contains, the off-by-one trajectory capture, and it stays reported.

    A false alarm on a healthy environment is what teaches people to ignore this warning on the
    broken one, so the check is kept to the repeats whose evidence is unambiguous.
    """
    if multi_turn or not prompt or not completion:
        return
    # the prompt's last turn is passed as context so a completion whose FIRST turn doubles it is
    # still seen. index 0 is that prompt turn, index 1 is the seam.
    repeated = [
        (index, role)
        for index, role in _repeated_roles([prompt[-1], *completion])
        # a repeat strictly inside the completion always counts; at the seam only a non-assistant
        # role does, since an assistant seam is a legitimate prefill continuation.
        if index > 1 or role != "assistant"
    ]
    if not repeated:
        return
    # rebased from the [prompt[-1], *completion] window onto the concatenation `_repeated_roles_message`
    # expects, where the completion begins at `len(prompt)`.
    rebased = [(index - 1 + len(prompt), role) for index, role in repeated]
    _emit(_repeated_roles_message(rebased, len(prompt)))
    print(f"  rendered roles: {_rendered_roles(prompt, completion)}", file=sys.stderr)


def _repeated_roles_message(repeated: list[tuple[int, str]], boundary: int) -> str:
    """Name each collapsing turn with `sft_completion`'s own index.

    Two shapes reach here, and both are the completion's to fix: a repeat strictly inside it, and a
    non-assistant turn at the seam that restates the prompt's last turn. The caller drops the
    prompt's interior (not `sft_completion`'s to fix) and an assistant seam (a legitimate prefill).

    Indices are rebased onto `sft_completion`, which is what the author can look up: a raw offset
    into the concatenation would report turn 0 of a three-message completion as message 3 when a
    system prompt is prepended, a position that does not exist in the file they open. The seam is
    the completion's message 0.
    """
    at_seam = [(index, role) for index, role in repeated if index == boundary]
    where = ", ".join(f"message {index - boundary} ({role})" for index, role in repeated)
    if at_seam:
        role = at_seam[0][1]
        where = (
            f"{where}; message 0 repeats the prompt's last turn, which is also a {role} turn"
            if len(repeated) > 1
            else f"message 0 ({role}), which repeats the prompt's last turn"
        )
    # the remedy depends on which role doubled. two assistant turns need the env's user turns
    # interleaved; a doubled user turn is the opposite fault -- the completion restating a question
    # the prompt already asked, which is an off-by-one in how the trajectory was captured.
    doubled = {role for _index, role in repeated}
    if doubled == {"assistant"}:
        remedy = (
            "a gold answer that spans several assistant turns must interleave the user turns it "
            "was answering"
        )
    elif "assistant" not in doubled:
        remedy = (
            "a repeated non-assistant turn usually means the trajectory was captured one turn "
            "early, so the completion restates a turn the prompt already contains"
        )
    else:
        remedy = "sft_completion must alternate roles once concatenated"
    return (
        f"sft would train on consecutive same-role turns inside sft_completion ({where}). sft "
        f"renders one string from prompt_messages + sft_completion, so these turns merge into a "
        f"single reply -- {remedy}"
    )


def _warn_on_uniformly_zero_rewards(
    rewards: list[float],
    *,
    algorithm: str,
    replayed_any: bool,
    gold_without_replayable_text: bool = False,
    separates: bool = False,
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
    flash/engine/worker/train/rl/launch/verl_config.py), so a constant reward gives zero advantage and zero
    gradient. Training completes, the loss curve looks unremarkable, and the adapter comes out
    identical to its warm start. The end-of-run advantage-spread guard catches that only after the
    GPUs have been paid for (flash/engine/worker/train/rl/launch/checkpoints.py); this catches it before
    they are allocated.

    Skipped when `separates` says the grader was PROVEN to distinguish answers -- a centered scale
    paying gold 0.0 and junk -1.0, or a per-turn vector that separates while the scalar is only a
    placeholder. Both have a real gradient, so calling them unmeasured would be false, and a false
    alarm on the healthy path is what teaches people to ignore this warning on the broken one.

    Only exact zeros trip it. A nonzero constant is a weaker, separate signal -- a reward may
    legitimately be constant across three rows -- and is not diagnosed here.
    """
    if separates or not rewards or any(reward != 0.0 for reward in rewards):
        return
    count = len(rewards)
    if not replayed_any:
        opening = (
            f"all {count} scored episode(s) returned reward 0.000000, and every one of them "
            "replayed a deliberately wrong answer because no row supplied gold text to replay. "
            "a zero is the correct score for that, so this run is not evidence the reward function "
            "works"
        )
        if gold_without_replayable_text:
            # a gold answer IS present -- it just carries no text this command can send, because
            # the payload is in `tool_calls` or an image block. that target is correct as authored
            # and SFT trains on it fine; telling this author to give the assistant turns text would
            # have them corrupt a working row to satisfy a check. the honest statement is that the
            # offline harness cannot replay it, so the grader has to be exercised elsewhere.
            message = (
                f"{opening}. the rows do carry a gold answer, but its assistant turns hold their "
                "payload outside `content` (a native tool call keeps the arguments in `tool_calls` "
                "and leaves `content` null), so this command had nothing to send and fell back to "
                "junk. that target is fine for SFT as written -- do NOT add assistant text to "
                "satisfy this check. exercise the reward function against a sampled rollout instead"
            )
        else:
            message = (
                f"{opening}. give the rows a gold answer whose assistant turns carry text "
                "(`sft_completion`, or the row's `output`) and re-run"
            )
    else:
        # phrased as a RISK, not a prediction. this command scored at most three offline gold
        # replays, never a group of policy samples, so it cannot know what a real rollout group
        # would pay. saying the adapter is guaranteed to come out unchanged would overstate the
        # evidence, and an operator who trains anyway and sees a normal-looking run learns to
        # distrust the warning.
        if algorithm == _REWARD_DRIVEN_ALGORITHM:
            message = (
                f"all {count} scored episode(s) returned reward 0.000000, so this run measured "
                "nothing. if sampled rollouts score alike, GRPO centres the group to a zero "
                "advantage and a zero gradient: the run completes, reports an unremarkable loss "
                "curve, and produces an adapter identical to its warm start. check the reward "
                "function, its runtime dependencies, and that the gold answer it scored is in the "
                "shape it expects"
            )
        else:
            # sft and opd never read `env.reward` (`_REWARD_DRIVEN_ALGORITHM`), so an all-zero
            # scorer is not a defect for THIS run and "check the reward function" is the wrong
            # instruction: a placeholder `reward()` on an sft-only environment is a deliberate,
            # correct choice, and telling its author to debug it sends them to fix working code.
            # what is still worth saying is that the run carries no evidence about the grader, so
            # nobody should read this PASS as clearance to train grpo on the same environment.
            message = (
                f"all {count} scored episode(s) returned reward 0.000000. {algorithm} does not "
                "train from `env.reward`, so this does not affect this run -- a placeholder scorer "
                "is a legitimate choice for an environment used only this way. it does mean the "
                "run is no evidence the reward function works: before training grpo on this "
                "environment, exercise the grader with `--algorithm grpo`"
            )
    _emit(message)

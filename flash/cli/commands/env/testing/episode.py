"""Drive and score one multi-turn episode for `flash env eval`.

Split out of eval.py, which grades single-turn cases: those send one prompt and score one reply,
while a multi-turn case has to be played out turn by turn before anything can be scored.

The eval helpers this needs are resolved through the module at call time rather than bound as
globals here. A module-level `from .eval import ...` would be circular (eval imports this module),
and it would also freeze the helpers at import time, so a test that patches them on `env.eval`
would no longer reach the episode path.
"""

from __future__ import annotations

import contextlib
import inspect
import sys

from flash.cli.commands.env.testing.evaluations import _evaluation_example
from flash.cli.ui import render
from flash.envs.meta.evaluations import EvalCase, EvalResult, _episode_grading_enabled


def _eval_module():
    """The eval module, looked up per call so patches applied to it are visible here."""
    from flash.cli.commands.env.testing import eval as module

    return module


def _grades_episodes(suite) -> bool:
    """Whether this suite grades a finished transcript rather than one reply.

    Opt-in, and defaulted off. A multi-turn ENVIRONMENT does not imply a transcript-grading
    SUITE: the scaffolded starter pairs a multi-turn env with a suite that deliberately grades
    only the opening action, and its cases carry no `output` for `step_episode` to advance from.
    Keying off the environment scored the wrong turn there and turned a well-formed first action
    into an error, so the suite has to say so itself.
    """
    return _episode_grading_enabled(suite)


def _drive_episode(client, target: str, environment, case: EvalCase, args) -> dict | str:
    """Play one multi-turn episode against the deployed model, or say why it could not run.

    Mirrors the worker turn loop the same way `env test` does (see `_drive_multi_turn` there):
    one model turn, then stop at the hard turn ceiling, on the env's own done signal, or when the
    env yields no reply. A single generation would grade only the first turn of a task whose
    reward reads the whole transcript.
    """
    eval_module = _eval_module()
    example = _evaluation_example(case)
    state = environment.new_rollout_state(example)
    # The per-example budget wins over the dataset-wide ceiling, the same precedence
    # `rollout_done` applies (flash/envs/loading/adapter.py) and both training bridges compute
    # (opd/bridge.py, rl/multi_turn.py). Using only `environment.max_turns` would run a case whose
    # `max_episode_turns` is lower past its own budget.
    hard_cap = _effective_turn_cap(environment, state)
    turns = 0
    # True while the newest turn has not been through env_reply, mirroring the worker's own flag.
    env_step_pending = False
    while True:
        messages = state.get("messages")
        if not messages:
            return "rollout produced no messages to generate from"
        # Normalize on every turn, exactly as the single-turn path and the training workers do.
        # Sending state["messages"] raw would ship package-relative image paths the chat API
        # cannot load, so a multimodal case would silently grade a model that never saw the image.
        prompt = eval_module._remote_prompt_messages(
            environment, example, [dict(m) for m in messages]
        )
        response = eval_module._generate_case(client, target, prompt, args)
        if isinstance(response, eval_module._GenerationFailure):
            return response.error
        environment.record_model_turn(state, response)
        env_step_pending = True
        turns += 1
        if turns >= hard_cap or environment.rollout_done(state, max_turns=hard_cap):
            break
        env_messages = environment.env_reply(state["messages"], state)
        env_step_pending = False
        if not env_messages:
            break
        if environment.rollout_done(state, max_turns=hard_cap):
            break

    # The loop exits before the inter-turn env_reply, leaving the last model turn unapplied. A
    # stateful env would then score a transcript missing the last thing the model did: `env_reply`
    # is what runs `step_episode`, so board state, metadata and `final_response_text` all lag one
    # action behind. Apply it before scoring. Only the inter-turn glue is skipped, since no further
    # model turn is conditioned on the reply.
    #
    # This DOES run at the turn cap, matching training. `rl/multi_turn.py` gates its reply solely on
    # `rollout_done`, which counts `state["turn"]` -- incremented only by `env_reply`, never by
    # `record_model_turn`. So after the capped model turn the counter is still one short, the check
    # passes, and RL steps the env on that turn. `opd/bridge.py` skips only the reply that would
    # glue on a next-turn prompt, which is the same thing skipped here by exiting the loop.
    #
    # An earlier revision suppressed this at the cap. That was wrong twice over: it dropped the
    # last action's side effects from the scored state, and it made the branch unreachable outright
    # -- `env_step_pending` survives only the line-70 break, whose condition is exactly what the
    # extra guard then excluded.
    #
    # An env that already reports done is the one case still skipped: it has ended the episode
    # itself, so `step_episode` has nothing left to apply and both `env test` and the RL worker
    # stop there. Only the CAP half of the old guard was wrong.
    if env_step_pending and not environment.rollout_done(state, hard_cap):
        environment.env_reply(state["messages"], state)
    return state


def _effective_turn_cap(environment, state: dict) -> int:
    """The turn ceiling for THIS episode: the per-example budget if it set one, else the env's.

    `rollout_done` gives `state["max_episode_turns"]` precedence over the dataset-wide cap, and
    both training bridges derive the same effective limit. A driver that counted only
    `environment.max_turns` would keep generating past a shorter per-example budget, grading turns
    training would never have run.
    """
    cap = state.get("max_episode_turns")
    ceiling = int(environment.max_turns)
    if cap is None:
        return ceiling
    # Never above the dataset-wide ceiling, and never below one turn, matching rl/multi_turn.py.
    return max(1, min(ceiling, int(cap)))


def _score_episode_case(
    suite,
    case: EvalCase,
    case_id: str,
    state: dict,
    *,
    scorer,
    state_style: str | None,
    thinking: bool = False,
) -> EvalResult:
    """Grade a finished episode through the suite's own scorer.

    `state["response_text"]` is only the LAST model turn: `record_model_turn` overwrites it every
    turn (flash/envs/loading/adapter.py). Handing a transcript-grading suite that scalar would score one
    turn of an episode it just paid to play out -- the same defect as generating once, moved one
    step later. So the finished state is offered to the suite, exactly as `env test` passes it to
    `reward(completion, example, state)`, which is what reaches the SDK's `score_episodes`.

    `EvalSuite.score(case, response)` is the published two-argument contract, so state is offered
    only to a suite that accepts it. The caller resolves one scorer and signature style for the
    whole suite rather than retrying on TypeError: an error raised INSIDE a state-aware scorer must
    not be retried as a two-argument call and graded on the wrong text.
    """
    response = state.get("response_text")
    if not isinstance(response, str):
        turns = state.get("turns") or []
        response = str(turns[-1]) if turns else ""
    eval_module = _eval_module()
    if state_style is None:
        return eval_module._score_case(
            suite, case, case_id, response, thinking=thinking, scorer=scorer
        )
    return eval_module._score_case(
        suite,
        case,
        case_id,
        response,
        thinking=thinking,
        state=state,
        state_keyword=state_style == "keyword",
        scorer=scorer,
    )


def _state_argument(score) -> str | None:
    """How this scorer takes the episode state: `"keyword"`, `"positional"`, or None for neither.

    Detecting only WHETHER state is accepted is not enough, because the two groups are disjoint:
    a `**kwargs` or keyword-only scorer rejects a third positional argument, and a `*args` scorer
    rejects the `state=` keyword. Passing state the one way the signature cannot take turns a
    suite that opted in to episode grading into `scoring failed` on every case -- a hard zero that
    reads like the model failed rather than like the harness called the scorer wrong.
    """
    if not callable(score):
        return None
    try:
        parameters = list(inspect.signature(score).parameters.values())
    except (TypeError, ValueError):
        # builtins and C callables expose no signature; treat them as the published contract.
        return None
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    named_state = next((parameter for parameter in parameters if parameter.name == "state"), None)
    if named_state is not None:
        if named_state.kind is inspect.Parameter.POSITIONAL_ONLY:
            return "positional" if positional.index(named_state) == 2 else None
        if named_state.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            # by keyword even where positional would also bind: it cannot land on the wrong parameter.
            return "keyword"
    # checked before `*args`, since a scorer with both can take `state=` but one with only
    # `*args` cannot.
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return "keyword"
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        # scorer(case, response, state) reaches *args when the signature has no fixed third slot.
        # zero, one, or two regular positional parameters can all bind that call.
        return "positional" if len(positional) <= 2 else None
    return None


def _warn_if_episode_state_is_hidden(suite, state_style: str | None) -> None:
    """Warn once when full-episode work feeds a scorer that cannot see the transcript."""
    if state_style is not None:
        return
    message = (
        f"suite {getattr(suite, 'name', 'evaluation')!r} sets grades_episodes = True; "
        "each episode will still be played out with one generation per turn, but the scorer "
        "will receive only the episode's final response text, not the transcript. Add a third "
        "`state` argument to "
        "`score(case, response, state)` to grade the transcript."
    )
    print(render.warn(message) if render.styled() else f"warning: {message}", file=sys.stderr)


def _run_episode_cases(
    client, target: str, suite, cases: list[EvalCase], args, environment, thinking=False
) -> tuple[EvalResult, ...]:
    """Run multi-turn cases one episode at a time.

    Each turn's prompt depends on the previous turn's env reply, so a case cannot be batched into
    one generation. Episodes run serially: the environment carries per-episode rollout state, and
    `env test` drives it on one thread for the same reason.
    """
    eval_module = _eval_module()
    case_ids = eval_module._case_ids(cases)
    if not getattr(environment, "multi_turn", False):
        # An episode suite on a single-turn env has no transcript to play out. That is an error,
        # not a 0.0: the pairing is unmeasurable, which is not the same as a failing model.
        eval_module._err(
            f"suite {getattr(suite, 'name', 'evaluation')!r} sets grades_episodes = True, "
            "but this environment is single-turn and has no episode to play"
        )
        return tuple(
            eval_module._generation_error(
                case_id, "suite grades episodes but the environment is single-turn"
            )
            for case_id in case_ids
        )
    scorer = getattr(suite, "score", None)
    state_style = _state_argument(scorer)
    _warn_if_episode_state_is_hidden(suite, state_style)
    results = []
    # the adapter defaults to `thinking = false` (flash/envs/loading/adapter.py), and with it off
    # `_scored_turn_text` returns the turn unstripped, so `state["response_text"]` keeps its
    # `<think>...</think>` wrapper. scoring the separate `response` argument hides that, because
    # `_score_case` strips it; a suite that reads the state instead sees raw reasoning and marks a
    # correct answer wrong. training sets this on the env, so eval has to as well.
    with _reasoning_mode(environment, thinking):
        for case, case_id in zip(cases, case_ids, strict=True):
            try:
                outcome = _drive_episode(client, target, environment, case, args)
            except (Exception, SystemExit) as exc:
                reason = str(exc) or exc.__class__.__name__
                results.append(eval_module._generation_error(case_id, f"episode failed: {reason}"))
                continue
            if isinstance(outcome, str):
                results.append(eval_module._generation_error(case_id, outcome))
                continue
            results.append(
                _score_episode_case(
                    suite,
                    case,
                    case_id,
                    outcome,
                    scorer=scorer,
                    state_style=state_style,
                    thinking=thinking,
                )
            )
    return tuple(results)


@contextlib.contextmanager
def _reasoning_mode(environment, thinking: bool):
    """Put the environment in the run's reasoning mode for the duration of the episodes.

    Restored afterwards rather than set once: the caller owns the adapter and may reuse it for
    another suite, and leaving a flipped flag behind would change how THAT one grades.
    """
    previous = getattr(environment, "thinking", None)
    try:
        environment.thinking = bool(thinking)
    except AttributeError:
        # a stand-in environment that does not expose the flag grades on its own terms
        yield
        return
    try:
        yield
    finally:
        if previous is None:
            with contextlib.suppress(AttributeError):
                del environment.thinking
        else:
            environment.thinking = previous

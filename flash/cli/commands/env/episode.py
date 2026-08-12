"""Drive and score one multi-turn episode for `flash env eval`.

Split out of eval.py, which grades single-turn cases: those send one prompt and score one reply,
while a multi-turn case has to be played out turn by turn before anything can be scored.

The eval helpers this needs are resolved through the module at call time rather than bound as
globals here. A module-level `from .eval import ...` would be circular (eval imports this module),
and it would also freeze the helpers at import time, so a test that patches them on `env.eval`
would no longer reach the episode path.
"""

from __future__ import annotations

from flash.cli.commands.env.test import _evaluation_example
from flash.envs.evaluations import EvalCase, EvalResult


def _eval_module():
    """The eval module, looked up per call so patches applied to it are visible here."""
    from flash.cli.commands.env import eval as module

    return module


def _grades_episodes(suite) -> bool:
    """Whether this suite grades a finished transcript rather than one reply.

    Opt-in, and defaulted off. A multi-turn ENVIRONMENT does not imply a transcript-grading
    SUITE: the scaffolded starter pairs a multi-turn env with a suite that deliberately grades
    only the opening action, and its cases carry no `output` for `step_episode` to advance from.
    Keying off the environment scored the wrong turn there and turned a well-formed first action
    into an error, so the suite has to say so itself.
    """
    return bool(getattr(suite, "grades_episodes", False))


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
    hard_cap = int(environment.max_turns)
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
    # stateful env would then score a transcript missing the last thing the model did, so give it
    # that turn before scoring; only the inter-turn glue is skipped, since no further model turn
    # is conditioned on the reply. This matches `env test` and the worker loops.
    if env_step_pending and not environment.rollout_done(state, hard_cap):
        environment.env_reply(state["messages"], state)
    return state


def _score_episode_case(
    suite, case: EvalCase, case_id: str, state: dict, *, thinking: bool = False
) -> EvalResult:
    """Grade a finished episode through the suite's own scorer.

    `EvalSuite.score` takes text, so the episode is represented by the text the environment itself
    considers scored (`state["response_text"]`, which `env_reply` replaces outright when
    `step_episode` returns a `final_response_text` override). A suite that grades the transcript
    reads it from the environment it was built with; this keeps the published suite contract
    unchanged rather than adding a second scoring entry point.
    """
    response = state.get("response_text")
    if not isinstance(response, str):
        turns = state.get("turns") or []
        response = str(turns[-1]) if turns else ""
    return _eval_module()._score_case(suite, case, case_id, response, thinking=thinking)


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
    results = []
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
        results.append(_score_episode_case(suite, case, case_id, outcome, thinking=thinking))
    return tuple(results)

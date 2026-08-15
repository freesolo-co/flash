"""Offline contract checks for an environment's evaluation sidecar.

Split from `test.py` to keep that file under the 1000-line gate. This group is cohesive and has one
entry point, `check_evaluation_suites`: everything else here exists to build one held-out case's
prompt and replayed response the same way `flash env eval` would.
"""

from __future__ import annotations

from pathlib import Path

from flash.cli.commands.env.push import _err


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
    # imported here rather than at module scope: `test.py` imports THIS module, so a top-level
    # import back into it would be circular.
    from flash.cli.commands.env.test import (
        _ECHO_RESPONSE,
        _check_messages,
        _gold_completion,
        _reference_turns,
        _resolve_policy,
    )

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


def _episode_evaluation_response(env, case) -> tuple[str, dict]:
    """Replay one held-out episode without re-running the environment's reward."""
    # imported here rather than at module scope: `test.py` imports this module.
    from flash.cli.commands.env.test import _drive_multi_turn, _new_record

    record = _new_record()
    _drive_multi_turn(
        env,
        _evaluation_example(case),
        record,
        score=False,
        replay_gold=case.expected is not None,
    )
    state = record["state"]
    response = state.get("response_text")
    if not isinstance(response, str):
        turns = record["responses"]
        response = str(turns[-1]) if turns else ""
    return response, state


def _call_evaluation_scorer(scorer, case, response: str, state: dict, state_style: str | None):
    if state_style is None:
        return scorer(case, response)
    if state_style == "keyword":
        return scorer(case, response, state=state)
    return scorer(case, response, state)


def _check_evaluation_suites(entrypoint: Path, env) -> bool:
    from flash.cli.commands.env.episode import (
        _grades_episodes,
        _state_argument,
        _warn_if_episode_state_is_hidden,
    )
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
            episode_suite = _grades_episodes(suite)
            if episode_suite and not getattr(env, "multi_turn", False):
                raise TypeError(
                    "suite sets grades_episodes = True, but this environment is single-turn "
                    "and has no episode to play"
                )
            scorer = getattr(suite, "score", None)
            state_style = _state_argument(scorer) if episode_suite else None
            if episode_suite:
                _warn_if_episode_state_is_hidden(suite, state_style)
            for index, case in enumerate(cases, start=1):
                state: dict = {}
                if episode_suite:
                    response, state = _episode_evaluation_response(env, case)
                else:
                    _policy, response = _evaluation_response(env, case)
                scored = _call_evaluation_scorer(scorer, case, response, state, state_style)
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

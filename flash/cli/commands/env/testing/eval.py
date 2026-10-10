"""Run a published environment's held-out evaluation suites against a deployed model."""

from __future__ import annotations

import queue
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from flash._internal.channel import CLI_NAME
from flash.cli.commands.env.ops.push import _err
from flash.cli.commands.env.testing.eval_args import (
    _MAX_CONCURRENCY,
    bounded_concurrency,
    finite_float,
    positive_int,
)
from flash.cli.commands.env.testing.evaluations import _evaluation_example
from flash.cli.commands.env.testing.test import _env_params
from flash.cli.ui import render
from flash.envs.meta.evaluations import (
    EvalCase,
    EvalResult,
    EvalSuiteReport,
    _evaluation_path,
    load_evaluation_suites,
    normalize_eval_result,
    validate_evaluation_cases,
)


def _generation_error(case_id: str, message: str) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        passed=False,
        score=0.0,
        response="",
        error=message,
    )


def _case_messages(environment, case: EvalCase) -> list[dict]:
    """The chat messages one case is evaluated on.

    Use `env.prompt_messages(example)` so evaluation includes the environment system prompt used
    by training (`flash/envs/loading/adapter.py`).
    """
    example = _evaluation_example(case)
    build = getattr(environment, "prompt_messages", None)
    if not callable(build):
        return _remote_prompt_messages(
            environment, example, [{"role": "user", "content": case.input}]
        )
    messages = build(example)
    if not isinstance(messages, list) or not messages:
        raise TypeError(
            f"environment prompt_messages() returned {type(messages).__name__}, "
            "expected a non-empty list of chat messages"
        )
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(
                f"environment prompt_messages() returned a {type(message).__name__} message, "
                "expected dicts with role and content"
            )
    return _remote_prompt_messages(environment, example, list(messages))


def _remote_prompt_messages(environment, example: dict, messages: list[dict]) -> list[dict]:
    """The prompt as the serving backend must receive it, matching what training builds.

    Apply `normalize_prompt_images` like every worker so top-level images are retained and local
    paths become remote-safe data URIs. Text-only cases stay untouched.
    """
    from flash.content.multimodal import record_has_images

    if not record_has_images(example, messages):
        return [dict(message) for message in messages]
    from flash.content.multimodal import image_descriptors_to_data_uris, normalize_prompt_images

    package_root = getattr(environment, "package_root", None)
    normalized = normalize_prompt_images(example, messages, package_root)
    uris = iter(image_descriptors_to_data_uris(normalized.descriptors, package_root))
    remote: list[dict] = []
    for message in normalized.messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            copied["content"] = [
                {"type": "image_url", "image_url": {"url": next(uris)}}
                if block.get("type") == "image"
                else block
                for block in content
            ]
        remote.append(copied)
    return remote


def _generate_response(client, target: str, messages: list[dict], args) -> str:
    chunks: list[str] = []
    for chunk in client.chat_stream(
        target,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ):
        if not isinstance(chunk, str):
            raise TypeError(f"model stream returned {type(chunk).__name__}, expected text chunks")
        chunks.append(chunk)
    response = "".join(chunks)
    if not response.strip():
        raise RuntimeError(
            f"no response text from {target}: the request succeeded but the model returned "
            "nothing. the deployment may be unhealthy or still starting; check "
            f"`{CLI_NAME} models deployments` and retry."
        )
    return response


def _generate_case(client, target: str, messages: list[dict], args) -> str | _GenerationFailure:
    """The case's response text, or the reason generation failed. This half runs off-thread.

    SystemExit derives from BaseException, so an environment or client that calls sys.exit()
    would otherwise tear down the whole suite mid-run and lose every case already graded."""
    try:
        return _generate_response(client, target, messages, args)
    except (Exception, SystemExit) as exc:
        return _GenerationFailure(f"generation failed: {str(exc) or exc.__class__.__name__}")


def _scored_response(response: str, *, thinking: bool) -> str:
    """The response as a scorer should see it, matching what training grades.

    Strip reasoning only for `thinking` runs; raw reasoning mis-grades them, while unconditional
    stripping can truncate non-thinking answers that mention `<think>`. An already-structured value
    passes through: re-deriving it from its own answer-only string destroys the raw and thinking
    views only its producer had the generation to build.
    """
    if not thinking:
        return response
    from flash.content.thinking import strip_think, thinking_text
    from flash.envs.loading.adapter import _ScoredResponseText

    if isinstance(response, _ScoredResponseText):
        return response
    answer = strip_think(response)
    return _ScoredResponseText(
        answer if isinstance(answer, str) else response,
        raw=response,
        thinking=thinking_text(response),
    )


def _score_case(
    suite,
    case: EvalCase,
    case_id: str,
    response: str,
    *,
    thinking: bool = False,
    state=None,
    state_keyword: bool = False,
    scorer=None,
) -> EvalResult:
    """Grade one response on the caller's thread.

    Scorers may require main-thread resources such as signal-based timeouts; a lock prevents
    overlap but does not provide thread affinity.

    `state` is the finished episode, passed only for a multi-turn suite that accepts it (see
    `episode._score_episode_case`); single-turn scoring keeps the two-argument contract.
    `state_keyword` says which way that suite's signature can take it -- `**kwargs` and
    keyword-only scorers reject a third positional, `*args` scorers reject the keyword -- so the
    caller decides, having read the signature. Episode scoring also passes the exact callable it
    inspected, so a descriptor cannot change shape between capability detection and invocation.

    The scorer grades the stripped answer; the RESULT records the raw generation. Stripping for the
    scorer must not shorten what is recorded, on the error path least of all -- a failed case is
    when the reasoning that explains it matters most.
    """
    recorded_response = response
    try:
        scored_text = _scored_response(response, thinking=thinking)
        recorded_response = getattr(scored_text, "raw", response)
        score = suite.score if scorer is None else scorer
        if state is None:
            scored = score(case, scored_text)
        elif state_keyword:
            scored = score(case, scored_text, state=state)
        else:
            scored = score(case, scored_text, state)
        return normalize_eval_result(case, recorded_response, scored, case_id=case_id)
    except (Exception, SystemExit) as exc:
        return EvalResult(
            case_id=case_id,
            passed=False,
            score=0.0,
            response=recorded_response,
            error=f"scoring failed: {str(exc) or exc.__class__.__name__}",
        )


def _case_ids(cases: list[EvalCase]) -> list[str]:
    """Stable per-case ids, disambiguating sidecars that reuse one id across cases.

    Retry suffixes until free because forms such as `a#2` are also valid explicit ids.
    """
    ids: list[str] = []
    seen: dict[str, int] = {}
    taken: set[str] = set()
    for index, case in enumerate(cases, start=1):
        case_id = case.id or str(index)
        count = seen.get(case_id, 0)
        resolved = case_id if count == 0 else f"{case_id}#{count + 1}"
        while resolved in taken:
            count += 1
            resolved = f"{case_id}#{count + 1}"
        seen[case_id] = count + 1
        taken.add(resolved)
        ids.append(resolved)
    return ids


def _build_messages(environment, case: EvalCase) -> list[dict] | _GenerationFailure:
    """One case's prompt, or the reason it could not be built.

    Rendered on the caller's thread for the same reason scoring is: prompt_messages runs the
    environment's own `start_episode`, which is user code. A raise must fail its own case, not
    the whole suite."""
    try:
        return _case_messages(environment, case)
    except (Exception, SystemExit) as exc:
        reason = str(exc) or exc.__class__.__name__
        return _GenerationFailure(f"prompt construction failed: {reason}")


def _run_cases(
    client, target: str, suite, cases: list[EvalCase], args, environment=None, thinking=False
) -> tuple[EvalResult, ...]:
    """Generate concurrently, score serially on this thread.

    Scorers can hold thread-bound resources, so a lock around worker scoring is insufficient.
    """
    # A transcript-grading suite must play the episode out: scoring one reply would grade a
    # different task than the run trains on, and would still report a number for it. The suite
    # opts in, because a multi-turn ENVIRONMENT does not imply a transcript-grading SUITE -- the
    # scaffolded starter pairs a multi-turn env with a suite that deliberately grades only the
    # first action, and playing its cases as episodes scores the wrong turn.
    #
    # Imported here, not at module scope: episode.py resolves this module's helpers back out of
    # it, so a top-level import would be circular.
    from flash.cli.commands.env.testing.episode import _grades_episodes, _run_episode_cases

    if _grades_episodes(suite) and environment is not None:
        return _run_episode_cases(
            client, target, suite, cases, args, environment, thinking=thinking
        )
    case_ids = _case_ids(cases)
    prompts = [_build_messages(environment, case) for case in cases]
    if args.concurrency == 1 or len(cases) <= 1:
        responses = [_generate(client, target, prompt, args) for prompt in prompts]
    else:
        responses = _generate_concurrently(client, target, prompts, args)
    return tuple(
        _generation_error(case_id, response.error)
        if isinstance(response, _GenerationFailure)
        else _score_case(suite, case, case_id, response, thinking=thinking)
        for case, case_id, response in zip(cases, case_ids, responses, strict=True)
    )


@dataclass(frozen=True)
class _GenerationFailure:
    """Why one case never produced a response. Carries no case id: the caller pairs by index."""

    error: str


def _generate(
    client, target: str, prompt: list[dict] | _GenerationFailure, args
) -> str | _GenerationFailure:
    """The case's response text, or the reason it never got one.

    A case whose prompt could not be built arrives already failed and is passed straight
    through, so a broken prompt never buys a generation."""
    if isinstance(prompt, _GenerationFailure):
        return prompt
    return _generate_case(client, target, prompt, args)


def _generate_concurrently(
    client, target: str, prompts: list[list[dict] | _GenerationFailure], args
) -> list[str | _GenerationFailure]:
    responses: list[str | _GenerationFailure | None] = [
        prompt if isinstance(prompt, _GenerationFailure) else None for prompt in prompts
    ]
    pending: queue.SimpleQueue = queue.SimpleQueue()
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, _GenerationFailure):
            pending.put((index, prompt))
    outstanding = pending.qsize()
    if outstanding:
        finished: queue.SimpleQueue = queue.SimpleQueue()
        aborted = threading.Event()
        abort_lock = threading.Lock()
        abort_error: BaseException | None = None

        def _run_cases() -> None:
            nonlocal abort_error
            # `aborted` is the cancel_futures half of the old shutdown: a case still queued when
            # the eval is abandoned must never buy a generation.
            while not aborted.is_set():
                try:
                    index, prompt = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    responses[index] = _generate_case(client, target, prompt, args)
                except BaseException as exc:
                    # _generate_case absorbs Exception and SystemExit itself, so anything arriving
                    # here -- KeyboardInterrupt, in practice -- ends the whole eval. carry it to
                    # the caller's thread, which is what `future.result()` used to do: raised in a
                    # worker it would simply kill that thread, and the CLI would grade the run as
                    # if the user had never pressed Ctrl-C.
                    with abort_lock:
                        if abort_error is None:
                            abort_error = exc
                    aborted.set()
                    finished.put(index)
                    return
                finished.put(index)

        # deliberately NOT a ThreadPoolExecutor. Its shutdown(wait=False) only postpones the wait:
        # concurrent.futures registers an interpreter-exit hook that joins every worker thread, so
        # after `main()` reported `aborted` the process still hung until the in-flight chat_stream
        # hit its 30-minute timeout. Daemon threads carry no such hook -- Ctrl-C propagates out of
        # the wait below and the interpreter exits without joining them.
        workers = [
            threading.Thread(target=_run_cases, daemon=True)
            for _ in range(min(args.concurrency, outstanding))
        ]
        for worker in workers:
            worker.start()
        # poll rather than block forever: a worker killed by something its `finally` cannot
        # observe would otherwise strand this wait with no case left to complete it.
        completed = 0
        while completed < outstanding and not aborted.is_set():
            try:
                finished.get(timeout=0.1)
            except queue.Empty:
                if not any(worker.is_alive() for worker in workers):
                    break
                continue
            completed += 1
        if abort_error is not None:
            # the surviving workers are daemons: they are abandoned rather than joined, so the
            # abort reaches the CLI's handler at once instead of after a 30-minute chat_stream.
            raise abort_error.with_traceback(abort_error.__traceback__)
    # a slot is still None only when a case never ran, which is the one path that returns
    # before every request resolved.
    return [
        response if response is not None else _GenerationFailure("generation did not run")
        for response in responses
    ]


def _case_payload(case: EvalCase | None, result: EvalResult) -> dict:
    """One graded case in the shape `POST /api/evals/runs` accepts."""
    expected = None
    if case is not None and case.expected is not None:
        expected = str(case.expected)
    return {
        "case_id": result.case_id,
        "input": case.input if case is not None else None,
        "expected": expected,
        "actual": result.response,
        "score": result.score,
        "success": result.passed,
        "reason": result.reason,
        "error": result.error,
    }


def _spec_project(spec: object) -> str:
    """The project a run's public spec files under, or empty when it names none."""
    if not isinstance(spec, dict):
        return ""
    project = spec.get("project")
    return project.strip() if isinstance(project, str) else ""


def _spec_environment_id(spec: object) -> str:
    """The hub environment a run trains against, as the dashboard stores it.

    Canonicalize managed `github:` refs to their slug like the submit route; return other
    references verbatim rather than guessing.
    """
    if not isinstance(spec, dict):
        return ""
    environment = spec.get("environment")
    if not isinstance(environment, dict):
        return ""
    env_id = environment.get("id")
    if not isinstance(env_id, str) or not env_id.strip():
        return ""
    env_id = env_id.strip()
    from flash.envs.loading.loader import canonical_managed_environment_slug

    try:
        return canonical_managed_environment_slug(env_id) or env_id
    except ValueError:
        # a malformed managed ref is still the identity the run carries; recording it verbatim
        # keeps the report honest about what it graded rather than dropping the provenance.
        return env_id


def _require_accessible_project(project_id: object) -> str:
    """The canonical id of a project this caller can actually upload to.

    Malformed, missing, or inaccessible projects all raise before generation begins.
    """
    from flash.client import ClientError, resolve_project_id
    from flash.client.config import load_credentials
    from flash.core.spec import require_project_id

    try:
        project_id = require_project_id(project_id)
    except (TypeError, ValueError) as exc:
        raise ClientError(str(exc)) from exc

    api_url, api_key = load_credentials()
    if not api_key:
        raise ClientError(
            f"not logged in — run `{CLI_NAME} login` with your freesolo API key (or set FREESOLO_API_KEY)"
        )
    # pass `api_url` so self-hosted planes get shape validation without sending
    # `FREESOLO_INTERNAL_KEY` to api.freesolo.co (SELF_HOSTING.md,
    # `_verifies_against_freesolo`). upload may still fail later, but credential leakage is worse.
    return resolve_project_id(project_id, api_key, api_url)


def _upload_report(
    report: EvalSuiteReport,
    cases: list[EvalCase],
    *,
    project_id: str,
    environment_reference: str,
    target: str,
    started_at: str,
    status: str = "completed",
    error: str | None = None,
) -> int:
    """Record one suite's results against a project, reporting failures without hiding them.

    The environment reference must be a published slug. Upload failure is reported without
    replacing the suite's already-printed verdict.
    """
    from flash.client import ClientError, upload_eval_run
    from flash.client.config import load_credentials

    _, api_key = load_credentials()
    if not api_key:
        return _err(
            f"cannot upload results: not logged in — run `{CLI_NAME} login` with your freesolo "
            "API key (or set FREESOLO_API_KEY)"
        )

    # upload pre-grading failures verbatim; the server excludes them from aggregates.
    # key by resolved ids because duplicate sidecar ids become `id#N`; raw ids can attach the
    # wrong input and expected value.
    by_id = dict(zip(_case_ids(cases), cases, strict=True))
    payload = [_case_payload(by_id.get(result.case_id), result) for result in report.results]
    # a suite whose cases failed to generate or score is not a completed run. only failures BEFORE
    # case execution passed status="failed", so a suite where every case errored uploaded as
    # `completed` with no run-level error while the CLI printed `overall: FAIL` -- the dashboard and
    # the exit code disagreeing about the same run.
    if status == "completed" and report.errors:
        status = "failed"
        error = f"{report.errors}/{report.total} case(s) failed to generate or score"
    try:
        upload_eval_run(
            project_id=project_id,
            suite_name=report.name,
            environment_reference=environment_reference,
            model=target,
            status=status,
            error=error,
            started_at=started_at,
            cases=payload,
            api_key=api_key,
        )
    except ClientError as exc:
        return _err(f"suite {report.name}: upload failed: {exc}")
    detail = f"suite {report.name}: uploaded {len(payload)} case(s)"
    print(render.ok(detail) if render.styled() else detail)
    return 0


def _print_case(result: EvalResult) -> None:
    state = "PASS" if result.passed else "FAIL"
    detail = f"case {result.case_id}: {state} score={result.score:.6f}"
    if result.error:
        detail += f" error={result.error}"
    elif result.reason:
        detail += f" reason={result.reason}"
    if render.styled():
        print(render.ok(detail) if result.passed else render.error(detail))
    else:
        print(detail)


def _print_report(report: EvalSuiteReport) -> None:
    detail = (
        f"suite {report.name}: {report.passed}/{report.total} passed "
        f"pass_rate={report.pass_rate:.2%} mean_score={report.mean_score:.6f}"
    )
    if report.errors:
        detail += f" errors={report.errors} (excluded from pass_rate and mean_score)"
    if render.styled():
        print(render.ok(detail) if report.passed == report.total else render.error(detail))
    else:
        print(detail)


def _resolve_evaluation_target(
    args, client, parsed, api_error, client_error
) -> tuple[str, object, int | None]:
    """require the exact requested checkpoint to have verified ready authority."""

    run_id, _ = parsed
    try:
        run = client.get_run(run_id)
    except (api_error, client_error) as exc:
        if getattr(args, "debug", False):
            raise
        _err(f"env eval failed: could not resolve verified checkpoints for {run_id}: {exc}")
        return args.target, None, _err("overall: FAIL")
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        _err(f"env eval failed: could not resolve verified checkpoints for {run_id}: {exc}")
        return args.target, None, _err("overall: FAIL")
    verified = run.get("verified_checkpoints") if isinstance(run, dict) else None
    if not isinstance(verified, list) or any(not isinstance(item, str) for item in verified):
        _err(f"env eval failed: control plane returned invalid checkpoint authority for {run_id}")
        return args.target, None, _err("overall: FAIL")
    if args.target not in verified:
        _err(f"env eval failed: checkpoint {args.target} has not passed deployment verification")
        return args.target, None, _err("overall: FAIL")
    return args.target, run, None


def _load_evaluation_run_spec(run: object) -> object:
    """load the target run specification from the verified authority record."""

    return run.get("spec") if isinstance(run, dict) else None


def _resolve_evaluation_project(args, project_id, spec, client_error) -> tuple[str, int | None]:
    """Resolve the result upload project phase."""
    if not args.upload or project_id:
        return project_id, None
    # the run's own project, and not a default: an evaluation of these weights files where the
    # weights themselves live. nothing here falls back to a first, sole, or example project --
    # when the run names none and the user did not either, recording would have to invent a home
    # for a permanent result, so it refuses before any paid generation and names both ways out.
    try:
        project_id = _require_accessible_project(_spec_project(spec))
    except client_error as exc:
        if getattr(args, "debug", False):
            raise
        _err(
            f"env eval failed: cannot record results for {args.target}: its project is unknown "
            f"({exc}). pass `--project PROJECT_ID`, or pass --no-upload to score without "
            "recording"
        )
        return project_id, _err("overall: FAIL")
    return project_id, None


def _resolve_evaluation_environment(args, spec, is_managed_environment_slug):
    """Resolve the published environment reference phase."""
    # grade against the published environment the run trained on and file under that identity.
    # a local path is neither stable nor resolvable, so there is no fallback.
    environment_reference = _spec_environment_id(spec)
    if not environment_reference:
        _err(
            f"env eval failed: run {args.target} trains on no published environment. "
            f"publish one with `{CLI_NAME} env push` and train a run against it"
        )
        return environment_reference, _err("overall: FAIL")
    # nonempty is not enough. a run may legitimately train on a generic `github:` ref, which
    # `_spec_environment_id` returns verbatim because it denotes no hub page -- and recording one
    # would file this report under exactly the unlinkable provenance the command exists to stop. the
    # hub is also what makes the package fetchable below, so this is one gate for both: a slug, or
    # no evaluation.
    if not is_managed_environment_slug(environment_reference):
        _err(
            f"env eval failed: run {args.target} trains on {environment_reference}, which is not a "
            f"published environment. publish it with `{CLI_NAME} env push` and train a run against "
            "the resulting namespace/project/name slug"
        )
        return environment_reference, _err("overall: FAIL")
    return environment_reference, None


def _prepare_evaluation_workdir(
    args,
    client,
    environment_reference,
    params,
    workdir,
    api_error,
    client_error,
    default_environment_path,
    pull_environment_package_from_archive,
    load_freesolo_environment,
):
    """Prepare the temporary evaluation work directory phase."""
    try:
        # download once through the control plane, the only credential boundary users have.
        # one archive pins moving environment-hub@main content for both environment and grader.
        # load from its absolute extracted path so a matching local `namespace/project/name` checkout
        # cannot override the published environment.
        package = client.download_env_package(environment_reference)
        entrypoint = (
            pull_environment_package_from_archive(package, Path(workdir) / "package")
            / default_environment_path
        )
    except (api_error, client_error) as exc:
        if getattr(args, "debug", False):
            raise
        _err(
            f"env eval failed: could not download the published environment "
            f"{environment_reference}: {exc}"
        )
        return None, None, None, _err("overall: FAIL")
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        _err(
            f"env eval failed: could not unpack the published environment "
            f"{environment_reference}: {exc}"
        )
        return None, None, None, _err("overall: FAIL")

    try:
        # the same kwargs `env test` builds from --split/--param, so a held-out suite grades the
        # environment the run is actually configured with. loading parameterless rejected an env
        # whose load_environment() requires a setting, and silently built a differently-
        # configured scorer for one that merely defaults.
        environment = load_freesolo_environment(str(entrypoint), **params)
        suites = load_evaluation_suites(entrypoint, environment=environment)
        # where those suites came from, for the case-validation errors below -- the file
        # `load_evaluation_suites` actually read, not a second guess at it.
        sidecar = _evaluation_path(entrypoint)
    except (Exception, SystemExit) as exc:
        # a load failure is a bug in the published environment or its sidecar, not a
        # measurement. --debug asked for the traceback, so let the root handler print it.
        if getattr(args, "debug", False):
            raise
        reason = str(exc) or exc.__class__.__name__
        _err(f"env eval failed: {reason.replace('cannot publish', 'cannot evaluate')}")
        return None, None, None, _err("overall: FAIL")

    if args.suite:
        available = ", ".join(sorted(suite.name for suite in suites))
        suites = [suite for suite in suites if suite.name == args.suite]
        if not suites:
            _err(f"env eval failed: unknown suite {args.suite!r}; available suites: {available}")
            return None, None, None, _err("overall: FAIL")
    return environment, suites, sidecar, None


def _run_evaluation_workdir(
    args,
    client,
    evaluation_target,
    environment_reference,
    project_id,
    thinking,
    workdir,
    environment,
    suites,
    sidecar,
) -> int:
    """Run and report suites from the temporary work directory phase."""
    reports: list[EvalSuiteReport] = []
    for suite in suites:
        # each suite uploads as its own run, so each needs its own start. sharing one timestamp
        # across suites backdates every later run to before the earlier suites' work and
        # inflates its dashboard duration by time it did not spend.
        started_at = datetime.now(UTC).isoformat()
        try:
            cases = validate_evaluation_cases(suite, source=sidecar)
        except (Exception, SystemExit) as exc:
            if getattr(args, "debug", False):
                raise
            reason = str(exc) or exc.__class__.__name__
            _err(f"suite {suite.name} failed to load cases: {reason}")
            report = EvalSuiteReport(
                name=suite.name,
                results=(_generation_error("load", f"case loading failed: {reason}"),),
            )
            _print_report(report)
            reports.append(report)
            if args.upload:
                _upload_report(
                    report,
                    [],
                    project_id=project_id,
                    environment_reference=environment_reference,
                    target=evaluation_target,
                    started_at=started_at,
                    status="failed",
                    error=f"case loading failed: {reason}",
                )
            continue
        if not cases:
            # a suite that graded nothing measured nothing. reporting 0/0 as a pass would
            # turn an empty or over-filtered suite into a green check nobody looks at again.
            _err(f"suite {suite.name} has no cases to run")
            report = EvalSuiteReport(
                name=suite.name,
                results=(_generation_error("load", "suite produced no cases"),),
            )
            _print_report(report)
            reports.append(report)
            if args.upload:
                _upload_report(
                    report,
                    cases,
                    project_id=project_id,
                    environment_reference=environment_reference,
                    target=evaluation_target,
                    started_at=started_at,
                    status="failed",
                    error="suite produced no cases",
                )
            continue
        if args.max_cases is not None:
            cases = cases[: args.max_cases]
        results = _run_cases(
            client, evaluation_target, suite, cases, args, environment, thinking=thinking
        )
        for result in results:
            _print_case(result)
        report = EvalSuiteReport(name=suite.name, results=results)
        _print_report(report)
        reports.append(report)
        # every report uploads, including the ones that never graded a case. skipping them left
        # the dashboard showing the earlier suites as a completed run with the failing suite
        # simply absent -- a green-looking evaluation whose CLI exit code was 1. `_case_payload`
        # tolerates a missing case, so a load failure records its error rather than nothing.
        if args.upload:
            # the errored-case downgrade lives in `_upload_report`, so it covers this call and the
            # two load-failure ones above rather than only the path that happens to run cases.
            _upload_report(
                report,
                cases,
                project_id=project_id,
                environment_reference=environment_reference,
                target=evaluation_target,
                started_at=started_at,
            )

    failed = any(
        report.passed != report.total or any(result.error for result in report.results)
        for report in reports
    )
    if failed:
        return _err("overall: FAIL")
    print(render.ok("overall: PASS") if render.styled() else "overall: PASS")
    return 0


def cmd_env_eval(args) -> int:
    """Score one deployed model target against its own published environment's held-out suites."""
    from flash.client import ApiError, ClientError, client_from_config
    from flash.envs.loading.loader import (
        _DEFAULT_ENVIRONMENT_PATH,
        is_managed_environment_slug,
        load_freesolo_environment,
    )
    from flash.envs.loading.pull import pull_environment_package_from_archive
    from flash.schema import parse_checkpoint_ref

    if args.project and not args.upload:
        return _err("--project cannot be combined with --no-upload")

    parsed = parse_checkpoint_ref(args.target)
    if parsed is None:
        return _err(
            f"invalid evaluation target {args.target!r} "
            "(expected <run_id>/final or <run_id>/step-N)"
        )

    # an explicitly named project is fully settled before anything else happens, including before a
    # client exists: it is already known, so neither its shape nor its accessibility needs the run
    # lookup below, and a typo or an unreachable project costs one second rather than a control
    # plane round trip and then a whole paid suite.
    project_id = ""
    if args.project:
        try:
            project_id = _require_accessible_project(args.project)
        except ClientError as exc:
            if getattr(args, "debug", False):
                raise
            return _err(f"--project must be a valid PROJECT_ID: {exc}")

    try:
        params = _env_params(args)
    except ValueError as exc:
        _err(f"env eval failed: {exc}")
        return _err("overall: FAIL")

    client = client_from_config()
    evaluation_target, run, exit_code = _resolve_evaluation_target(
        args, client, parsed, ApiError, ClientError
    )
    if exit_code is not None:
        return exit_code

    spec = _load_evaluation_run_spec(run)

    # graders must see what training graded, so the run's own `thinking` decides whether the
    # reasoning is stripped first (see `_scored_response`).
    thinking = bool(spec.get("thinking")) if isinstance(spec, dict) else False

    project_id, exit_code = _resolve_evaluation_project(args, project_id, spec, ClientError)
    if exit_code is not None:
        return exit_code

    environment_reference, exit_code = _resolve_evaluation_environment(
        args, spec, is_managed_environment_slug
    )
    if exit_code is not None:
        return exit_code

    with tempfile.TemporaryDirectory(prefix="flash-env-eval-") as workdir:
        environment, suites, sidecar, exit_code = _prepare_evaluation_workdir(
            args,
            client,
            environment_reference,
            params,
            workdir,
            ApiError,
            ClientError,
            _DEFAULT_ENVIRONMENT_PATH,
            pull_environment_package_from_archive,
            load_freesolo_environment,
        )
        if exit_code is not None:
            return exit_code
        return _run_evaluation_workdir(
            args,
            client,
            evaluation_target,
            environment_reference,
            project_id,
            thinking,
            workdir,
            environment,
            suites,
            sidecar,
        )


# re-exported from eval_args so `flash.cli` keeps importing the eval flag validators from here.
__all__ = [
    "_MAX_CONCURRENCY",
    "bounded_concurrency",
    "cmd_env_eval",
    "finite_float",
    "positive_int",
]

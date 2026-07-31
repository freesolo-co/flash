"""Run held-out environment evaluation suites against a deployed model."""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from flash._channel import CLI_NAME
from flash.envs.evaluations import (
    _DEFAULT_EVALUATIONS_PATH,
    EvalCase,
    EvalResult,
    EvalSuiteReport,
    load_evaluation_suites,
    normalize_eval_result,
    validate_evaluation_cases,
)

from . import render
from .env_test import _env_params, _evaluation_example
from .envpush import _err, _resolve_local_env_entrypoint

_MAX_CONCURRENCY = 32


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def finite_float(value: str) -> float:
    """A temperature the chat route will accept.

    `float("nan")` and `float("inf")` parse, so argparse took them and every case then spent a
    request the server rejects for being non-finite (`flash/server/routes/serving.py:1429-1432`),
    turning one bad flag into one doomed paid request per case (codex[bot]).
    """
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {value}")
    return parsed


def bounded_concurrency(value: str) -> int:
    parsed = positive_int(value)
    if parsed > _MAX_CONCURRENCY:
        raise argparse.ArgumentTypeError(f"must be at most {_MAX_CONCURRENCY}")
    return parsed


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

    Training never sends the raw input: every path goes through `env.prompt_messages(example)`,
    which runs the environment's own `start_episode` and injects the training contract as the
    system prompt (flash/envs/adapter.py). Sending only `case.input` would grade the model on a
    prompt no run ever trains on, so a suite could fail purely because the system instructions
    the model was trained under were absent."""
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

    `prompt_messages()` is only half of training's prompt: every worker then runs
    `normalize_prompt_images` (flash/engine/worker/rl.py, sft.py, opd.py), which folds a record's
    top-level `image`/`images` into the first user message and resolves each source to a
    descriptor. Sending the raw messages dropped top-level images entirely, so a multimodal suite
    graded a text-only prompt, and forwarded an environment's package-relative path straight to a
    remote backend that cannot read the evaluator's disk (codex[bot]).

    Descriptors are then rendered as data URIs, the same conversion the image teacher uses to put
    local images on a remote request (flash/engine/worker/opd.py). Text-only cases -- the vast
    majority -- return untouched without importing the multimodal machinery at all."""
    from flash.multimodal import record_has_images

    if not record_has_images(example, messages):
        return [dict(message) for message in messages]
    from flash.multimodal import image_descriptors_to_data_uris, normalize_prompt_images

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

    Training never hands a grader the raw completion: both the single-turn and multi-turn paths
    strip the reasoning through `flash.thinking` first (flash/envs/adapter.py). Evaluating the raw
    string instead mis-grades a thinking deployment against its own environment -- the scaffolded
    multi-turn scorer reads the first token as an int, which is `<think>` for every reasoning run.

    Gated on the run's own `thinking`, never on the text: `strip_think` also cuts at a bare
    `<think>` mention, so applying it to a non-thinking answer that merely names the tag would
    truncate a correct response."""
    if not thinking:
        return response
    from flash.envs.adapter import _ScoredResponseText
    from flash.thinking import strip_think, thinking_text

    answer = strip_think(response)
    return _ScoredResponseText(
        answer if isinstance(answer, str) else response,
        raw=response,
        thinking=thinking_text(response),
    )


def _score_case(
    suite, case: EvalCase, case_id: str, response: str, *, thinking: bool = False
) -> EvalResult:
    """Grade one response on the caller's thread.

    Scoring never runs in a worker. A lock would serialize it but still execute it off the
    main thread, and scorers routinely cannot run there at all: anything installing a
    signal-based timeout raises `signal only works in main thread`, which would be recorded
    as the model failing the case rather than as a broken harness."""
    try:
        scored = suite.score(case, _scored_response(response, thinking=thinking))
        return normalize_eval_result(case, response, scored, case_id=case_id)
    except (Exception, SystemExit) as exc:
        return EvalResult(
            case_id=case_id,
            passed=False,
            score=0.0,
            response=response,
            error=f"scoring failed: {str(exc) or exc.__class__.__name__}",
        )


# the server's own names for a record that is serving traffic versus one still coming up
# (flash/server/routes/serving.py). a busy record's `adapter_revision` is the INCOMING revision.
_READY_DEPLOYMENT_STATES = frozenset({"ready", "deployed"})


def _live_deployment(client, run_id: str) -> dict | None:
    """The revision a bare run id currently serves, whatever step it is on.

    `deployment_for` resolves an EXACT revision, so asking it for a bare run id means "the final
    adapter" (step None) and it rejects a run serving `RUN/step-N`. That would report a deployed
    run as undeployed and refuse to evaluate a model `flash chat RUN` talks to happily. Match on
    the run id alone, exactly as `_rollback_record` does for the same reason
    (flash/cli/commands.py)."""
    for entry in client.deployments() or ():
        listed = entry.get("deployment") or {}
        if run_id not in (listed.get("run_id"), entry.get("run_id")):
            continue
        # a queued replacement is listed with the revision it is rolling OUT to, while the
        # predecessor under `previous_deployment` is the one still answering requests. taking the
        # busy record graded a revision that was not serving yet, and reported it as the run's
        # score -- results attributed to the wrong weights (codex[bot]). `flash chat` reads the
        # predecessor for exactly this reason (`_previous_ready_deployment`).
        if listed.get("state") not in _READY_DEPLOYMENT_STATES:
            previous = listed.get("previous_deployment")
            if not isinstance(previous, dict) or previous.get("state") not in (
                _READY_DEPLOYMENT_STATES
            ):
                # a first rollout has no predecessor to fall back to, so nothing is serving yet.
                # returning the busy record anyway handed back the INCOMING revision and graded
                # weights that were not answering requests (cursor[bot]). the caller reports the
                # run as undeployed, which is what "still coming up" means to an evaluation.
                return None
            listed = previous
        # the listing omits undeployed/dry_run rows, so anything here is servable.
        if not listed.get("run_id") and entry.get("run_id"):
            listed = {**listed, "run_id": entry["run_id"]}
        return listed
    return None


def _case_ids(cases: list[EvalCase]) -> list[str]:
    """Stable per-case ids, disambiguating sidecars that reuse one id across cases.

    Two cases sharing an id would collide in the uploaded payload and in the printed
    report, silently reporting one graded case where two ran.

    The suffix is retried until it is genuinely free, because the disambiguated form is
    itself a legal explicit id: cases `a`, `a`, `a#2` resolved to `a`, `a#2`, `a#2` and
    reintroduced the collision this function exists to remove."""
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

    Only generation is parallel. `suite.score()` always runs here, in case order: a lock
    prevents overlap but not thread affinity, and a scorer holding a resource created while
    the suite was loaded -- a sqlite connection, a tokenizer bound to its creating thread --
    failed every case from a worker even though nothing was concurrent (codex[bot])."""
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
    pool = ThreadPoolExecutor(max_workers=min(args.concurrency, len(prompts)))
    try:
        futures = {
            pool.submit(_generate_case, client, target, prompt, args): index
            for index, prompt in enumerate(prompts)
            if not isinstance(prompt, _GenerationFailure)
        }
        for future in as_completed(futures):
            responses[futures[future]] = future.result()
    finally:
        # not `with`: ThreadPoolExecutor.__exit__ shuts down with wait=True, so Ctrl-C joined
        # every in-flight request before the root handler ever saw the KeyboardInterrupt. a
        # chat_stream call may block for up to 30 minutes, so an aborted eval looked hung for
        # that long (codex[bot]). cancel_futures drops queued cases immediately; requests already
        # in flight are not interruptible from here, so we do not pretend to wait for them.
        pool.shutdown(wait=False, cancel_futures=True)
    # a slot is still None only when the abort above dropped a queued case, which is the one
    # path that returns before every future resolved.
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


def _require_accessible_project(project_id: str) -> str:
    """The canonical id of a project this caller can actually upload to.

    Raises ClientError when the credentials are missing or the project is not reachable from
    this organization, so the refusal happens before a single generation is bought."""
    from flash.client import ApiError, ClientError, get_project
    from flash.client.config import load_credentials

    _, api_key = load_credentials()
    if not api_key:
        raise ClientError(
            "not logged in — run `flash login` with your freesolo API key (or set FREESOLO_API_KEY)"
        )
    try:
        return str(get_project(project_id, api_key)["id"])
    except ApiError as exc:
        if exc.status not in {403, 404}:
            raise
        raise ClientError(
            f"project {project_id!r} is not accessible; run `flash projects list` "
            "and pass a project UUID from the current organization"
        ) from exc


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

    `environment_reference` is the resolved entrypoint, not its directory: a package may hold
    several environment modules, and evaluating `/env/easy.py` then `/env/hard.py` recorded the
    same provenance for two runs graded by different environments.

    Upload failure is reported but does not change the eval's own exit status: the suite
    already ran and its verdict is printed above. Returning FAIL here would relabel a
    passing suite as failing because a network call did not land."""
    from flash.client import ClientError, upload_eval_run
    from flash.client.config import load_credentials

    _, api_key = load_credentials()
    if not api_key:
        return _err(
            "cannot upload results: not logged in — run `flash login` with your freesolo "
            "API key (or set FREESOLO_API_KEY)"
        )

    # a case that failed before it was graded is still uploaded verbatim; the server
    # excludes it from the aggregate so a transport failure never reads as a zero score.
    #
    # key on the same resolved ids the results carry, not on the raw `case.id`: a sidecar
    # that reuses one id across cases resolves the second to `id#2`, so a raw-id map both
    # misses it and hands the first result the *second* case's input and expected value.
    by_id = dict(zip(_case_ids(cases), cases, strict=True))
    payload = [_case_payload(by_id.get(result.case_id), result) for result in report.results]
    # a suite whose cases failed to generate or score is not a completed run. only failures
    # BEFORE case execution passed status="failed", so a suite where every case errored
    # uploaded as `completed` with no run-level error while the CLI printed `overall: FAIL`
    # -- the dashboard and the exit code disagreeing about the same run (codex[bot]).
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


def cmd_env_eval(args) -> int:
    """Score local held-out suites against one deployed model target."""
    from flash.client import ApiError, ClientError, client_from_config
    from flash.envs.loader import load_freesolo_environment
    from flash.schema import parse_adapter_revision, parse_checkpoint_ref

    # checked before the suites run so a bad project id fails in a second rather than after a
    # long paid evaluation whose results would then have nowhere to go. the id is validated
    # here, not just checked for emptiness: upload_eval_run requires a canonical UUID, so a
    # malformed one would otherwise buy every model request and be rejected at the end -- and
    # because upload failure deliberately does not change the verdict, it would still print
    # `overall: PASS` with nothing recorded.
    from flash.spec import require_project_id

    project_id = ""
    if args.upload:
        try:
            project_id = require_project_id(args.project)
        except (TypeError, ValueError) as exc:
            return _err(f"--upload requires a valid --project PROJECT_ID: {exc}")
        # a well-formed UUID is not an accessible project. checking only the shape let a deleted id,
        # or one belonging to another organization, buy the whole evaluation and be rejected at
        # upload -- printing `overall: PASS` with nothing recorded, since upload failure
        # deliberately does not change the verdict. `env setup` already resolves the project this
        # way before scaffolding anything.
        try:
            project_id = _require_accessible_project(project_id)
        except (ApiError, ClientError) as exc:
            if getattr(args, "debug", False):
                raise
            return _err(f"--upload requires a valid --project PROJECT_ID: {exc}")
    if args.project and not args.upload:
        return _err("--project only applies with --upload")

    revision = parse_adapter_revision(args.target)
    parsed = parse_checkpoint_ref(args.target) if revision is None else None
    if revision is None and parsed is None:
        return _err(
            f"invalid evaluation target {args.target!r} "
            "(expected a bare <run_id>, <run_id>/step-N, or full immutable adapter revision)"
        )

    try:
        params = _env_params(args)
    except ValueError as exc:
        _err(f"env eval failed: {exc}")
        return _err("overall: FAIL")

    try:
        _, _, entrypoint, _ = _resolve_local_env_entrypoint(Path(args.path))
        entrypoint = entrypoint.resolve()
        # the same kwargs `env test` builds from --split/--param, so a held-out suite grades the
        # environment the run is actually configured with. loading parameterless rejected an env
        # whose load_environment() requires a setting, and silently built a differently-configured
        # scorer for one that merely defaults (codex[bot]).
        environment = load_freesolo_environment(str(entrypoint), **params)
        suites = load_evaluation_suites(entrypoint, environment=environment)
    except (Exception, SystemExit) as exc:
        # a load failure is a bug in the sidecar or the package layout, not a measurement.
        # --debug asked for the traceback, so let the root handler print it.
        if getattr(args, "debug", False):
            raise
        reason = str(exc) or exc.__class__.__name__
        _err(f"env eval failed: {reason.replace('cannot publish', 'cannot evaluate')}")
        return _err("overall: FAIL")

    if args.suite:
        available = ", ".join(sorted(suite.name for suite in suites))
        suites = [suite for suite in suites if suite.name == args.suite]
        if not suites:
            _err(f"env eval failed: unknown suite {args.suite!r}; available suites: {available}")
            return _err("overall: FAIL")

    client = client_from_config()
    evaluation_target = args.target
    if revision is None and parsed is not None and parsed[1] is None:
        run_id = parsed[0]
        try:
            deployment = _live_deployment(client, run_id)
        except (ApiError, ClientError) as exc:
            if getattr(args, "debug", False):
                raise
            _err(f"env eval failed: could not resolve deployed revision for {run_id}: {exc}")
            return _err("overall: FAIL")
        if deployment is None:
            _err(f"env eval failed: run {run_id} is not deployed")
            return _err("overall: FAIL")
        candidate = deployment.get("adapter_revision")
        resolved = parse_adapter_revision(candidate) if isinstance(candidate, str) else None
        if resolved is None or resolved[0] != run_id:
            _err(
                f"env eval failed: deployment for {run_id} has no valid immutable adapter revision"
            )
            return _err("overall: FAIL")
        evaluation_target = candidate.strip()
        print(f"resolved evaluation target {run_id} to {evaluation_target}")

    # graders must see what training graded, so the run's own `thinking` decides whether the
    # reasoning is stripped first (see `_scored_response`). read once here rather than per case:
    # it is the same answer for every case, and a suite of 200 would otherwise buy 200 lookups.
    thinking = False
    target_run_id = (revision or parsed or (None,))[0]
    if target_run_id:
        try:
            spec = client.get_run(target_run_id).get("spec")
            thinking = bool(spec.get("thinking")) if isinstance(spec, dict) else False
        except Exception as exc:
            # not fatal: a run whose metadata is unreadable can still be evaluated, and the raw
            # response is what every eval graded before this. broad, because an unreachable plane
            # must not turn a working evaluation into a crash. say so rather than grade silently.
            _err(f"warning: could not read thinking mode for {target_run_id}: {exc}")

    reports: list[EvalSuiteReport] = []
    for suite in suites:
        # each suite uploads as its own run, so each needs its own start. sharing one timestamp
        # across suites backdates every later run to before the earlier suites' work and
        # inflates its dashboard duration by time it did not spend.
        started_at = datetime.now(UTC).isoformat()
        try:
            cases = validate_evaluation_cases(
                suite, source=entrypoint.parent / _DEFAULT_EVALUATIONS_PATH
            )
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
                    environment_reference=str(entrypoint),
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
                    environment_reference=str(entrypoint),
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
        # every report uploads, including the ones that never graded a case. skipping them
        # left the dashboard showing the earlier suites as a completed run with the failing
        # suite simply absent -- a green-looking evaluation whose CLI exit code was 1
        # (codex[bot]). `_case_payload` tolerates a missing case, so a load failure records
        # its error rather than nothing.
        if args.upload:
            # the errored-case downgrade lives in `_upload_report`, so it covers this call and the
            # two load-failure ones above rather than only the path that happens to run cases.
            _upload_report(
                report,
                cases,
                project_id=project_id,
                environment_reference=str(entrypoint),
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


__all__ = ["_MAX_CONCURRENCY", "bounded_concurrency", "cmd_env_eval", "positive_int"]

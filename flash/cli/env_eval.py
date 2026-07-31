"""Run held-out environment evaluation suites against a deployed model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from .env_test import _env_params
from .envpush import _err, _resolve_local_env_entrypoint

_MAX_CONCURRENCY = 32


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
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


def _generate_response(client, target: str, case: EvalCase, args) -> str:
    chunks: list[str] = []
    for chunk in client.chat_stream(
        target,
        messages=[{"role": "user", "content": case.input}],
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


def _generate_case(client, target: str, case: EvalCase, args) -> tuple[str, str | None]:
    """Generate one response. Returns (response, error); only this half runs off-thread.

    SystemExit derives from BaseException, so an environment or client that calls sys.exit()
    would otherwise tear down the whole suite mid-run and lose every case already graded."""
    try:
        return _generate_response(client, target, case, args), None
    except (Exception, SystemExit) as exc:
        return "", f"generation failed: {str(exc) or exc.__class__.__name__}"


def _score_case(suite, case: EvalCase, case_id: str, response: str) -> EvalResult:
    """Grade one response on the caller's thread.

    Scoring never runs in a worker. A lock would serialize it but still execute it off the
    main thread, and scorers routinely cannot run there at all: anything installing a
    signal-based timeout raises `signal only works in main thread`, which would be recorded
    as the model failing the case rather than as a broken harness."""
    try:
        scored = suite.score(case, response)
        return normalize_eval_result(case, response, scored, case_id=case_id)
    except (Exception, SystemExit) as exc:
        return EvalResult(
            case_id=case_id,
            passed=False,
            score=0.0,
            response=response,
            error=f"scoring failed: {str(exc) or exc.__class__.__name__}",
        )


def _resolve_case(suite, case: EvalCase, case_id: str, generated: tuple[str, str | None]):
    response, error = generated
    if error is not None:
        return _generation_error(case_id, error)
    return _score_case(suite, case, case_id, response)


def _case_ids(cases: list[EvalCase]) -> list[str]:
    """Stable per-case ids, disambiguating sidecars that reuse one id across cases.

    Two cases sharing an id would collide in the uploaded payload and in the printed
    report, silently reporting one graded case where two ran."""
    ids: list[str] = []
    seen: dict[str, int] = {}
    # a disambiguated id can itself collide with a raw id: cases "x", "x", "x#2" resolve to
    # "x", "x#2", "x#2" if the suffix is not re-checked, putting the collision back. taken
    # holds every id already handed out, so the suffix advances until it is actually free.
    taken: set[str] = set()
    for index, case in enumerate(cases, start=1):
        case_id = case.id or str(index)
        count = seen.get(case_id, 0)
        seen[case_id] = count + 1
        resolved = case_id if count == 0 else f"{case_id}#{count + 1}"
        while resolved in taken:
            count += 1
            seen[case_id] = count + 1
            resolved = f"{case_id}#{count + 1}"
        taken.add(resolved)
        ids.append(resolved)
    return ids


def _run_cases(client, target: str, suite, cases: list[EvalCase], args) -> tuple[EvalResult, ...]:
    case_ids = _case_ids(cases)
    if args.concurrency == 1 or len(cases) <= 1:
        return tuple(
            _resolve_case(suite, case, case_id, _generate_case(client, target, case, args))
            for case, case_id in zip(cases, case_ids, strict=True)
        )

    generated: list[tuple[str, str | None] | None] = [None] * len(cases)
    # cancel_futures drops everything still queued on Ctrl-C. without it the interrupt is
    # only raised after every already-submitted request drains, so a --concurrency 32 run
    # over hundreds of cases keeps buying generations for minutes after the user stopped it.
    pool = ThreadPoolExecutor(max_workers=min(args.concurrency, len(cases)))
    try:
        futures = {
            pool.submit(_generate_case, client, target, case, args): index
            for index, case in enumerate(cases)
        }
        for future in as_completed(futures):
            generated[futures[future]] = future.result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)

    # scoring happens here, on the caller's thread, in case order: suites are user objects
    # that may hold mutable state or install signal handlers, neither of which survives a
    # worker thread.
    return tuple(
        _resolve_case(suite, case, case_id, item)
        for case, case_id, item in zip(cases, case_ids, generated, strict=True)
        if item is not None
    )


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


def _upload_report(
    report: EvalSuiteReport,
    cases: list[EvalCase],
    *,
    project_id: str,
    environment_reference: str,
    target: str,
    started_at: str,
) -> int:
    """Record one suite's results against a project, reporting failures without hiding them.

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
    try:
        upload_eval_run(
            project_id=project_id,
            suite_name=report.name,
            environment_reference=environment_reference,
            model=target,
            status="completed",
            error=None,
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
    from flash.client import client_from_config
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
        # the same params a run submits under [environment.params]. an environment whose
        # load_environment() requires one would fail to load here, and one that merely
        # branches on it would be graded on a different dataset or split than it trains on.
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
    started_at = datetime.now(UTC).isoformat()
    reports: list[EvalSuiteReport] = []
    for suite in suites:
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
            continue
        if args.max_cases is not None:
            cases = cases[: args.max_cases]
        results = _run_cases(client, args.target, suite, cases, args)
        for result in results:
            _print_case(result)
        report = EvalSuiteReport(name=suite.name, results=results)
        _print_report(report)
        reports.append(report)
        if args.upload:
            _upload_report(
                report,
                cases,
                project_id=project_id,
                environment_reference=str(entrypoint.parent),
                target=args.target,
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

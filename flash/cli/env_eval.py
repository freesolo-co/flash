"""Run held-out environment evaluation suites against a deployed model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

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


def _run_case(
    client, target: str, suite, case: EvalCase, case_id: str, args, score_lock: Lock
) -> EvalResult:
    # SystemExit and KeyboardInterrupt derive from BaseException, so a sidecar that calls
    # sys.exit() inside score() would tear down the whole suite mid-run and lose every case
    # already graded. A sidecar that exits is a broken scorer, which is this case's error.
    try:
        response = _generate_response(client, target, case, args)
    except (Exception, SystemExit) as exc:
        return _generation_error(
            case_id, f"generation failed: {str(exc) or exc.__class__.__name__}"
        )
    try:
        # suites are user objects shared across worker threads. a scorer holding mutable state
        # (a tokenizer, a counter, a cached client) would be raced under --concurrency and
        # produce silently wrong scores, which is worse than scoring serially.
        with score_lock:
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


def _case_ids(cases: list[EvalCase]) -> list[str]:
    """Stable per-case ids, disambiguating sidecars that reuse one id across cases.

    Two cases sharing an id would collide in the uploaded payload and in the printed
    report, silently reporting one graded case where two ran."""
    ids: list[str] = []
    seen: dict[str, int] = {}
    for index, case in enumerate(cases, start=1):
        case_id = case.id or str(index)
        count = seen.get(case_id, 0)
        seen[case_id] = count + 1
        ids.append(case_id if count == 0 else f"{case_id}#{count + 1}")
    return ids


def _run_cases(client, target: str, suite, cases: list[EvalCase], args) -> tuple[EvalResult, ...]:
    case_ids = _case_ids(cases)
    score_lock = Lock()
    if args.concurrency == 1 or len(cases) <= 1:
        return tuple(
            _run_case(client, target, suite, case, case_id, args, score_lock)
            for case, case_id in zip(cases, case_ids, strict=True)
        )

    results: list[EvalResult | None] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(cases))) as pool:
        futures = {
            pool.submit(_run_case, client, target, suite, case, case_id, args, score_lock): index
            for index, (case, case_id) in enumerate(zip(cases, case_ids, strict=True))
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return tuple(result for result in results if result is not None)


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
    by_id = {case.id or str(index): case for index, case in enumerate(cases, start=1)}
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

    # checked before the suites run so a missing project id fails in a second rather than
    # after a long paid evaluation whose results would then have nowhere to go.
    if args.upload and not (args.project or "").strip():
        return _err("--upload requires --project PROJECT_ID")
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
        _, _, entrypoint, _ = _resolve_local_env_entrypoint(Path(args.path))
        entrypoint = entrypoint.resolve()
        environment = load_freesolo_environment(str(entrypoint))
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
                project_id=args.project,
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

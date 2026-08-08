"""Run a published environment's held-out evaluation suites against a deployed model."""

from __future__ import annotations

import argparse
import math
import queue
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from flash._channel import CLI_NAME
from flash.envs.evaluations import (
    EvalCase,
    EvalResult,
    EvalSuiteReport,
    _evaluation_path,
    load_evaluation_suites,
    normalize_eval_result,
    validate_evaluation_cases,
)

from . import render
from .env_test import _env_params, _evaluation_example
from .envpush import _err

_MAX_CONCURRENCY = 32


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def finite_float(value: str) -> float:
    """A temperature the chat route will accept.

    Reject non-finite or negative values before they become one paid rejection per case.
    The nonnegative floor matches `flash/schema/__init__.py`.
    """
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {value}")
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(f"must be at least 0.0, got {value}")
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

    Use `env.prompt_messages(example)` so evaluation includes the environment system prompt used
    by training (`flash/envs/adapter.py`).
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

    Strip reasoning only for `thinking` runs; raw reasoning mis-grades them, while unconditional
    stripping can truncate non-thinking answers that mention `<think>`.
    """
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

    Scorers may require main-thread resources such as signal-based timeouts; a lock prevents
    overlap but does not provide thread affinity.
    """
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
# a busy redeploy may still serve its predecessor, which the public listing hides.
# `/v1/deployments` excludes only `undeployed`/`dry_run`, so only `ready` and `busy`
# may answer; other returned states fail closed (flash/cli/commands.py).
_BUSY_DEPLOYMENT_STATES = frozenset({"queued", "smoke_testing", "reconciling"})


def _live_deployment(client, run_id: str) -> dict | None:
    """The run's deployment record, whatever state it is in, or None when it has none.

    Match by run id because `deployment_for` misses step deployments. Busy records name the rollout
    target and hide the revision that may still be serving.
    """
    for entry in client.deployments() or ():
        listed = entry.get("deployment") or {}
        if run_id not in (listed.get("run_id"), entry.get("run_id")):
            continue
        # the listing omits undeployed/dry_run rows, so anything here is servable.
        if not listed.get("run_id") and entry.get("run_id"):
            listed = {**listed, "run_id": entry["run_id"]}
        return listed
    return None


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
        # warm the step-selector capability once before workers start. only successful
        # `/v1/health` checks are cached, so cold workers would multiply a transient or
        # unsupported failure. failures propagate here; only `RUN/step-N` triggers the check.
        client.warm_chat_step_selector(target)
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
    from flash.envs.loader import canonical_managed_environment_slug

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
    from flash.spec import require_project_id

    try:
        project_id = require_project_id(project_id)
    except (TypeError, ValueError) as exc:
        raise ClientError(str(exc)) from exc

    api_url, api_key = load_credentials()
    if not api_key:
        raise ClientError(
            "not logged in — run `flash login` with your freesolo API key (or set FREESOLO_API_KEY)"
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
            "cannot upload results: not logged in — run `flash login` with your freesolo "
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


def cmd_env_eval(args) -> int:
    """Score one deployed model target against its own published environment's held-out suites."""
    from flash.client import ApiError, ClientError, client_from_config
    from flash.envs.loader import (
        _DEFAULT_ENVIRONMENT_PATH,
        is_managed_environment_slug,
        load_freesolo_environment,
    )
    from flash.envs.pull import pull_environment_package_from_archive
    from flash.schema import parse_adapter_revision, parse_checkpoint_ref

    if args.project and not args.upload:
        return _err("--project cannot be combined with --no-upload")

    revision = parse_adapter_revision(args.target)
    parsed = parse_checkpoint_ref(args.target) if revision is None else None
    if revision is None and parsed is None:
        return _err(
            f"invalid evaluation target {args.target!r} "
            "(expected a bare <run_id>, <run_id>/step-N, or full immutable adapter revision)"
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
    evaluation_target = args.target
    if revision is None and parsed is not None:
        run_id, want_step = parsed
        try:
            deployment = _live_deployment(client, run_id)
        except (ApiError, ClientError) as exc:
            if getattr(args, "debug", False):
                raise
            _err(f"env eval failed: could not resolve deployed revision for {run_id}: {exc}")
            return _err("overall: FAIL")
        if deployment is None:
            _err(f"env eval failed: run {args.target} is not deployed")
            return _err("overall: FAIL")
        deployment_state = deployment.get("state")
        servable_states = _READY_DEPLOYMENT_STATES | _BUSY_DEPLOYMENT_STATES
        if want_step is not None:
            # pinned steps resolve through the verified ledger, not the deployment record
            # (`flash/server/routes/serving.py`). allow `failed`, which preserves that ledger;
            # revocation-failed and undeploy paths invalidate it in `flash/runner/deploy.py`.
            servable_states = servable_states | {"failed"}
        if deployment_state not in servable_states:
            # having a record is not having a servable one: the listing keeps terminal states like
            # `failed` and `revocation_failed`, and the chat route has no ready predecessor to fall
            # back to for them, so every case 409s. That spends a whole suite of generation failures
            # to learn what one target error says now. A step that genuinely is not in the ledger
            # still gets one 409 naming the deployed steps, not a suite of them.
            _err(
                f"env eval failed: run {args.target} deployment is {deployment_state or 'unknown'}"
                f"; run `{CLI_NAME} models deploy {run_id}` first"
            )
            return _err("overall: FAIL")
        # a busy record names the rollout target, not the hidden predecessor still serving.
        # leave the user's target unpinned so the chat route can resolve private rollback state or
        # return the correct 409 when nothing serves.
        resolved = None
        if deployment_state in _READY_DEPLOYMENT_STATES:
            candidate = deployment.get("adapter_revision")
            resolved = parse_adapter_revision(candidate) if isinstance(candidate, str) else None
            if resolved is None or resolved[0] != run_id:
                _err(
                    "env eval failed: deployment for "
                    f"{run_id} has no valid immutable adapter revision"
                )
                return _err("overall: FAIL")
        # `RUN/step-N` is not immutable: multiple revisions may share a step. pin only when the
        # live deployment is that step; otherwise the server-only verified ledger may still resolve
        # an older step, and the chat route must decide ambiguity.
        if resolved is not None and (want_step is None or resolved[1] == want_step):
            evaluation_target = candidate.strip()
            print(f"resolved evaluation target {args.target} to {evaluation_target}")
        elif args.upload and want_step is not None:
            # uploaded reports must name immutable weights. the CLI cannot resolve an ambiguous
            # `RUN/step-N`: the verified ledger is server-only and chat does not echo the revision.
            # refuse before generation; bare aliases remain valid because they mean the currently
            # serving revision, including a busy predecessor.
            _err(
                f"env eval failed: cannot upload results for {args.target}: the immutable "
                "revision that will answer it is not knowable here. re-run with the full "
                f"revision from `{CLI_NAME} models deployments`, or pass --no-upload"
            )
            return _err("overall: FAIL")

    # one run lookup supplies its training environment, reasoning mode, and owning project.
    # these are target-run properties, not fallbacks, and reading once avoids one lookup per case.
    spec = None
    target_run_id = (revision or parsed or (None,))[0]
    if target_run_id:
        try:
            spec = client.get_run(target_run_id).get("spec")
        except ClientError as exc:
            # `ApiError` subclasses `ClientError`, so one handler must preserve retryable 5xx/429.
            # both arms abort because the run's environment is required; they differ only in
            # whether retrying can help.
            if getattr(args, "debug", False):
                raise
            answered_definitively = (
                isinstance(exc, ApiError) and exc.status < 500 and exc.status != 429
            )
            if answered_definitively:
                _err(
                    f"env eval failed: could not read the target run {target_run_id}: {exc}. "
                    "its published environment is what supplies the suites to score."
                )
            else:
                _err(
                    f"env eval failed: could not reach the control plane for {target_run_id}: "
                    f"{exc}. retry once it is reachable."
                )
            return _err("overall: FAIL")
        except Exception as exc:
            # anything else is not a transport fault, so it is not retryable either. broad, so an
            # unexpected client shape cannot crash a command the user asked for.
            if getattr(args, "debug", False):
                raise
            _err(f"env eval failed: could not read the target run {target_run_id}: {exc}")
            return _err("overall: FAIL")

    # graders must see what training graded, so the run's own `thinking` decides whether the
    # reasoning is stripped first (see `_scored_response`).
    thinking = bool(spec.get("thinking")) if isinstance(spec, dict) else False

    if args.upload and not project_id:
        # the run's own project, and not a default: an evaluation of these weights files where the
        # weights themselves live. nothing here falls back to a first, sole, or example project --
        # when the run names none and the user did not either, recording would have to invent a home
        # for a permanent result, so it refuses before any paid generation and names both ways out.
        try:
            project_id = _require_accessible_project(_spec_project(spec))
        except ClientError as exc:
            if getattr(args, "debug", False):
                raise
            _err(
                f"env eval failed: cannot record results for {args.target}: its project is unknown "
                f"({exc}). pass `--project PROJECT_ID`, or pass --no-upload to score without "
                "recording"
            )
            return _err("overall: FAIL")

    # grade against the published environment the run trained on and file under that identity.
    # a local path is neither stable nor resolvable, so there is no fallback.
    environment_reference = _spec_environment_id(spec)
    if not environment_reference:
        _err(
            f"env eval failed: run {args.target} trains on no published environment. "
            f"publish one with `{CLI_NAME} env push` and train a run against it"
        )
        return _err("overall: FAIL")
    # nonempty is not enough. a run may legitimately train on a generic `github:` ref, which
    # `_spec_environment_id` returns verbatim because it denotes no hub page -- and recording one
    # would file this report under exactly the unlinkable provenance the command exists to stop. the
    # hub is also what makes the package fetchable below, so this is one gate for both: a slug, or
    # no evaluation.
    if not is_managed_environment_slug(environment_reference):
        _err(
            f"env eval failed: run {args.target} trains on {environment_reference}, which is not a "
            f"published environment. publish it with `{CLI_NAME} env push` and train a run against "
            "the resulting namespace/name slug"
        )
        return _err("overall: FAIL")

    with tempfile.TemporaryDirectory(prefix="flash-env-eval-") as workdir:
        try:
            # download once through the control plane, the only credential boundary users have.
            # one archive pins moving environment-hub@main content for both environment and grader.
            # load from its absolute extracted path so a matching local `namespace/name` checkout
            # cannot override the published environment.
            package = client.download_env_package(environment_reference)
            entrypoint = (
                pull_environment_package_from_archive(package, Path(workdir) / "package")
                / _DEFAULT_ENVIRONMENT_PATH
            )
        except (ApiError, ClientError) as exc:
            if getattr(args, "debug", False):
                raise
            _err(
                f"env eval failed: could not download the published environment "
                f"{environment_reference}: {exc}"
            )
            return _err("overall: FAIL")
        except Exception as exc:
            if getattr(args, "debug", False):
                raise
            _err(
                f"env eval failed: could not unpack the published environment "
                f"{environment_reference}: {exc}"
            )
            return _err("overall: FAIL")

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
            return _err("overall: FAIL")

        if args.suite:
            available = ", ".join(sorted(suite.name for suite in suites))
            suites = [suite for suite in suites if suite.name == args.suite]
            if not suites:
                _err(
                    f"env eval failed: unknown suite {args.suite!r}; available suites: {available}"
                )
                return _err("overall: FAIL")

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


__all__ = ["_MAX_CONCURRENCY", "bounded_concurrency", "cmd_env_eval", "positive_int"]

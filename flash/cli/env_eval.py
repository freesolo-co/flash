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

    `float("nan")` and `float("inf")` parse, so argparse took them and every case then spent a
    request the server rejects for being non-finite (`flash/server/routes/serving.py:1429-1432`),
    turning one bad flag into one doomed paid request per case (codex[bot]).

    A negative value is finite and so passed that check, but the OpenAI sampling contract the
    backend implements requires a nonnegative temperature, so it failed once per case rather than
    once at the flag (codex[bot]). Same floor training already enforces on its own temperature
    (`flash/schema/__init__.py`, `minimum=0.0`).
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
# a busy record may be a redeploy over a live revision, so it can still serve through the
# predecessor the public listing hides. ready plus busy is therefore the whole set that might
# answer; `/v1/deployments` excludes only `undeployed`/`dry_run`, so everything else it returns --
# `failed`, `revocation_failed`, any state a newer plane adds -- reaches this CLI as a record that
# is neither. `flash models deploy --wait` already fails closed on exactly that split
# (flash/cli/commands.py).
_BUSY_DEPLOYMENT_STATES = frozenset({"queued", "smoke_testing", "reconciling"})


def _live_deployment(client, run_id: str) -> dict | None:
    """The run's deployment record, whatever state it is in, or None when it has none.

    `deployment_for` resolves an EXACT revision, so asking it for a bare run id means "the final
    adapter" (step None) and it rejects a run serving `RUN/step-N`. That would report a deployed
    run as undeployed and refuse to evaluate a model `flash chat RUN` talks to happily. Match on
    the run id alone, exactly as `_rollback_record` does for the same reason
    (flash/cli/commands.py).

    The state is left to the caller because only a READY record names the revision that will
    answer: a busy one is listed with the revision it is rolling OUT to. Whether a predecessor is
    still serving underneath it cannot be decided here -- `/v1/deployments` passes every record
    through `public_deployment()`, which drops `previous_deployment` as private rollback state
    (flash/serve/urls.py). Both a redeploy over a live revision and a first rollout that has never
    served therefore arrive as the same bare busy record."""
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
    pending: queue.SimpleQueue = queue.SimpleQueue()
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, _GenerationFailure):
            pending.put((index, prompt))
    outstanding = pending.qsize()
    if outstanding:
        # settle the control plane's step-selector capability HERE, on one thread, before any worker
        # starts. the client caches the answer only after it succeeds, so N workers racing from cold
        # each fire their own /v1/health -- one eval's worth of duplicate requests for a fact about
        # the plane that cannot differ between them (codex[bot]). warming it costs one call and needs
        # no lock in the client, where it would put coordination on every caller for this one race.
        # a `RUN/step-N` target is what makes the check run at all; anything else skips it.
        #
        # A failure propagates rather than being suppressed. Suppressing it did not soften a
        # transient blip, it multiplied it: only a SUCCESSFUL check is cached, so every worker then
        # missed the same cold cache, and one timed-out or rate-limited /v1/health became up to 32
        # more plus one generation error per case -- instead of one target-level failure naming the
        # real cause (chatgpt-codex-connector). An unsupported plane is not transient at all, and
        # answering it with a suite of per-case errors buries the one line that says to use a full
        # revision or upgrade.
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

        # deliberately NOT a ThreadPoolExecutor. Its shutdown(wait=False) only postpones the
        # wait: concurrent.futures registers an interpreter-exit hook that joins every worker
        # thread, so after `main()` reported `aborted` the process still hung until the in-flight
        # chat_stream hit its 30-minute timeout (codex[bot]). Daemon threads carry no such hook --
        # Ctrl-C propagates out of the wait below and the interpreter exits without joining them.
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

    A managed slug (`namespace/name`) is what the environment pages are keyed by, so a `github:`
    ref pointing at the managed hub is canonicalized to the slug it denotes -- the same
    normalization the submit route applies before it records the run's environment
    (`flash/server/routes/runs.py`). Anything else is a reference this side cannot resolve to a
    hub page, so it is returned verbatim rather than guessed at."""
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

    Shape and reachability are one question here because they have one answer at every call
    site: a project that cannot be resolved is not a project to record against, whether it is
    malformed, deleted, or owned by another organization. Raises ClientError in every one of
    those cases, so the refusal happens before a single generation is bought."""
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
    # ``api_url`` is passed deliberately, so a self-hosted plane gets the shape-only check rather
    # than an ownership lookup against api.freesolo.co. The stored key on such a plane IS
    # FREESOLO_INTERNAL_KEY, which controls the plane, and it must not travel to a service the
    # operator does not run (SELF_HOSTING.md; the same boundary `_verifies_against_freesolo` draws
    # for `flash login`). That leaves a known gap -- `upload_eval_run` posts to the hosted API, so a
    # self-hosted `--project` run can still fail at upload after the suite is bought -- but closing
    # it by resolving against the hosted backend would trade a late failure for a leaked
    # plane-root credential, which is the worse outcome.
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

    `environment_reference` is the hub environment the graded run trains against, so the dashboard
    can open it. It is always a published slug, never a path: an evaluation with no published
    environment to name is refused before it runs.

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
            # a pinned step does not go through the deployment record at all: the chat route
            # resolves `RUN/step-N` against the verified ledger and, once it resolves,
            # `has_ready_deploy` is already true, so the terminal-state arms below it never run
            # (`flash/server/routes/serving.py`). `mark_deployment_failed` leaves that ledger
            # intact, so a step verified before a LATER deploy failed still serves, and refusing it
            # here failed an evaluation the server answers 200 (Cursor).
            #
            # exempt `failed` only. `mark_deployment_revocation_failed` and the undeploy paths call
            # `invalidate_verified_adapter_revisions` (`flash/runner/deploy.py`), so under those
            # states the ledger the pin would resolve against is gone and every case 409s -- the
            # same wasted suite this check exists to avoid (Cursor).
            servable_states = servable_states | {"failed"}
        if deployment_state not in servable_states:
            # having a record is not having a servable one: the listing keeps terminal states like
            # `failed` and `revocation_failed`, and the chat route has no ready predecessor to fall
            # back to for them, so every case 409s. That spends a whole suite of generation
            # failures to learn what one target error says now (chatgpt-codex-connector). A step
            # that genuinely is not in the ledger still gets one 409 naming the deployed steps, not
            # a suite of them.
            _err(
                f"env eval failed: run {args.target} deployment is {deployment_state or 'unknown'}"
                f"; run `{CLI_NAME} models deploy {run_id}` first"
            )
            return _err("overall: FAIL")
        # a busy record is listed with the revision it is rolling OUT to, so pinning it would file
        # the scores under weights that were not answering requests (codex[bot]). the predecessor
        # still serving underneath is stripped from the public listing, so this side cannot name
        # it either (see `_live_deployment`) -- and refusing outright would fail an evaluation that
        # `flash chat RUN` serves correctly through that predecessor (codex[bot]). leave the target
        # as the user wrote it and let the chat route resolve it: it reads the private rollback
        # state, and answers with a 409 naming the surface to use when nothing is serving at all.
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
        # `RUN/step-N` is a shorthand, not an identity: the chat route resolves it against the
        # run's whole verified ledger, which can hold several revisions at one step, and picks the
        # deployed one. Forwarding the shorthand therefore graded weights the report could not
        # name, and a later rebuild of the same step would read as the same measurement
        # (codex[bot]). Pin here so generation and the uploaded report carry the immutable value.
        # Only when the live deployment IS the requested step can the CLI name the revision that
        # will answer. Otherwise the shorthand stays: a run keeps its earlier verified revisions
        # after a newer step is deployed, and the chat route serves them -- asked for step-3 while
        # step-20 is live, the server resolves the ledger's single step-3 entry and answers 200.
        # Refusing would fail an evaluation the server runs correctly, so let the server resolve it;
        # it reads the verified ledger this CLI cannot see, and answers a genuinely ambiguous step
        # with a 409 naming the surface to use instead.
        if resolved is not None and (want_step is None or resolved[1] == want_step):
            evaluation_target = candidate.strip()
            print(f"resolved evaluation target {args.target} to {evaluation_target}")
        elif args.upload and want_step is not None:
            # an uploaded report is a permanent record, so it has to name the weights it graded.
            # `RUN/step-N` names a step but not a revision: the server resolves it against the run's
            # whole verified ledger, which can hold several revisions at one step
            # (`_verified_step_index`), and a later rebuild of that step reuses the same shorthand --
            # so two different sets of weights file as one measurement, and afterwards nothing can
            # tell them apart (chatgpt-codex-connector).
            #
            # This side cannot resolve it: the verified ledger is server-only,
            # `/v1/runs/{id}/checkpoints` carries no revision, and the chat route never echoes the
            # revision that answered. So refuse before buying a single generation rather than spend
            # a whole suite on a result nobody can identify. Both ways out are in the message, and
            # evaluating with `--no-upload` still prints the verdict.
            #
            # Only when a step was asked for. A bare alias means "whatever serves this run now",
            # which is what it records, and a busy one deliberately stays unpinned so the run still
            # evaluates through the predecessor `flash chat RUN` uses (codex[bot], test above).
            _err(
                f"env eval failed: cannot upload results for {args.target}: the immutable "
                "revision that will answer it is not knowable here. re-run with the full "
                f"revision from `{CLI_NAME} models deployments`, or pass --no-upload"
            )
            return _err("overall: FAIL")

    # one lookup answers everything this command needs to know ABOUT THE TARGET RUN: which published
    # environment it trains against, whether its responses carry reasoning, and which project owns
    # its results. all three are properties of that run, so none is a default or a fallback -- an
    # evaluation grades the weights against the environment that trained them, and files under the
    # project that owns them.
    #
    # read once here rather than per case: it is the same answer for every case, and a suite of 200
    # would otherwise buy 200 lookups.
    spec = None
    target_run_id = (revision or parsed or (None,))[0]
    if target_run_id:
        try:
            spec = client.get_run(target_run_id).get("spec")
        except ClientError as exc:
            # one handler, not two: `ApiError` subclasses `ClientError`, so an `except ApiError`
            # arm would catch 5xx and 429 before the retryable arm ever saw them (cursor[bot]).
            #
            # both arms end the command, because the spec is no longer an enrichment: it names the
            # environment whose suites this evaluation runs. without it there is nothing to grade
            # against, so neither arm can warn and continue. they stay separate only to say whether
            # retrying is worth anything -- a 4xx is a settled answer about this run, a 5xx or a
            # timeout is the plane failing to answer at all.
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

    # the published environment the graded weights were trained on: both the suites that score them
    # and the identity the report is filed under. a local directory cannot serve either role -- it
    # names no environment anyone else can open, so two developers evaluating one run recorded two
    # different provenances for the same measurement, and neither could be resolved back to a page.
    # there is no fallback: an evaluation with nothing published to name is refused rather than
    # recorded against a path.
    environment_reference = _spec_environment_id(spec)
    if not environment_reference:
        _err(
            f"env eval failed: run {args.target} trains on no published environment. "
            f"publish one with `{CLI_NAME} env push` and train a run against it"
        )
        return _err("overall: FAIL")
    # nonempty is not enough. a run may legitimately train on a generic `github:` ref, which
    # `_spec_environment_id` returns verbatim because it denotes no hub page -- and recording one
    # would file this report under exactly the unlinkable provenance the command exists to stop
    # (codex[bot]). the hub is also what makes the package fetchable below, so this is one gate for
    # both: a slug, or no evaluation.
    if not is_managed_environment_slug(environment_reference):
        _err(
            f"env eval failed: run {args.target} trains on {environment_reference}, which is not a "
            f"published environment. publish it with `{CLI_NAME} env push` and train a run against "
            "the resulting namespace/name slug"
        )
        return _err("overall: FAIL")

    with tempfile.TemporaryDirectory(prefix="flash-env-eval-") as workdir:
        try:
            # ONE download, through the control plane, and everything is graded from what it wrote.
            #
            # through the plane because that is the only credential an ordinary user has: the direct
            # hub path in `load_freesolo_environment` authenticates with an operator-style
            # GITHUB_TOKEN, so evaluating a published environment would have demanded a credential
            # that `env pull` never asks for, and failed without it (codex[bot]).
            #
            # once because a managed slug points at environment-hub@main, which moves, and symbolic
            # refs are deliberately not cached (`_resolve_ref_sha`). resolving per-call let the
            # environment object come from one revision and its grading code from the next
            # (codex[bot]). one archive cannot disagree with itself.
            #
            # and from the extracted path because slug resolution is not uniform: the sidecar lookup
            # prefers a local directory when one named `namespace/name` sits in the cwd, so a
            # matching checkout silently graded the published environment with a working copy's
            # evaluations.py (cursor[bot]). an absolute path into this temp dir has no such branch.
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
            # configured scorer for one that merely defaults (codex[bot]).
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

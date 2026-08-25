"""Deploy, export and undeploy: the commands that move a finished run into serving.

`cmd_deploy --wait` is the reason this is its own module -- polling a revision to servable is a
state machine with its own timing constants, rollback read and permanent-failure rules, and it
dwarfed every other command handler in the package.

Split out of `flash.cli.commands` to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import sys
import threading
import time

from flash._internal.channel import CLI_NAME
from flash.cli.ui import render, tables
from flash.client import ApiError, ClientError, client_from_config

# the states a deployment sits in before the requested revision is actually servable, mirroring
# the set the control plane transitions through. anything else ends the wait: ready, failed, or a
# state this client does not know, which must not spin until the timeout.
_DEPLOYMENT_BUSY_STATES = frozenset({"queued", "smoke_testing", "reconciling"})
# the only states in which the control plane will actually serve the revision, mirroring
# flash/server/routes/serving.py. leaving the busy set is NOT the same as arriving here:
# `revocation_failed` (a concurrent undeploy whose backend cleanup failed) and any state a newer
# plane introduces are both non-busy and non-servable, so `--wait` must fail closed on them
# rather than let `deploy --wait && evaluate` proceed against nothing.
_DEPLOYMENT_READY_STATES = frozenset({"ready", "deployed"})
# an undeploy whose backend cleanup failed (flash/runner/supervise/transitions.py). local serving
# authority is revoked, but the alias may still resolve to the old target -- neither servable nor
# safely assumed idle.
_REVOCATION_FAILED_STATE = "revocation_failed"
_DEPLOY_POLL_SECONDS = 5.0
# `--wait 0` still owes the caller its one read, and a read needs a positive timeout. keep that
# bound short enough that "check once, do not block" stays true against a stalled plane: a longer
# fixed budget just moves the overshoot the per-poll bound exists to prevent.
_DEPLOY_ZERO_WAIT_READ_SECONDS = 1.0
# withheld from each sleep so the read that follows it starts inside the deadline. without this the
# sleep spends the whole remainder and the wait ends on the deadline check having never looked
# again, so a revision that went ready early in a short window reads as queued.
_DEPLOY_FINAL_READ_SECONDS = 1.0
# how much of the final window is slept before its one read. the rest bounds that read, so it stays
# inside the deadline rather than in flight past it. late in the window, not halfway: a midpoint
# read stops watching with half the advertised wait still to run.
_DEPLOY_FINAL_READ_FRACTION = 0.9
# an auth or authorization rejection answers the same way every time; polling through it just
# spends the whole timeout to arrive at the identical error.
_PERMANENT_POLL_STATUSES = frozenset({401, 403})
# the pre-deploy alias read is advisory, so it must not spend the client's default 60s budget
# deciding whether to print a warning: a stalled read would delay every real deploy behind a
# line nobody asked for. short enough to stay unnoticed, long enough for a healthy plane.
# it is spent three ways, because the client's own bounds are per-phase: as a socket timeout it
# bounds one socket operation, and as a body deadline it bounds the body but is only consulted
# once headers are parsed. a peer trickling headers just inside the socket timeout escapes both.
# `_read_within` bounds the total in wall-clock time, which is the one that actually holds.
_ALIAS_WARNING_READ_SECONDS = 5.0


def _await_deployment(client, run_id: str, deployment: dict, timeout: float) -> dict:
    """Poll until the requested revision leaves the busy states, or the timeout expires.

    POST deploy returns as soon as the record is persisted, normally in ``queued`` while the
    previous revision is still the ready one. A caller that starts evaluating on that return
    talks to a reconciling endpoint and mostly gets errors. Polling here makes the returned
    record mean what it appears to mean.
    """
    if str(deployment.get("state") or "") not in _DEPLOYMENT_BUSY_STATES:
        return deployment
    waiting = (
        f"waiting up to {timeout:g}s for {run_id} to become servable; "
        "ctrl-c stops waiting, not the deployment"
    )
    print(render.note(waiting) if render.styled() else f"note: {waiting}", file=sys.stderr)
    deadline = time.monotonic() + timeout
    latest = deployment
    first = True
    # set once the wait enters its final window, so the read that window funds is the last one.
    # see the sleep below: without a stop it would repeat down to clock granularity.
    final_read = False
    while True:
        remaining = deadline - time.monotonic()
        if not first and (remaining <= 0 or final_read):
            break
        if not first:
            # reserve a fixed final-read budget so the loop terminates and observes readiness inside
            # the last window. place that read late, but before the deadline, to avoid both unused wait
            # time and an in-flight overshoot; see
            # `test_deploy_wait_does_not_start_a_read_after_the_deadline_expires`.
            slice_seconds = min(_DEPLOY_POLL_SECONDS, remaining)
            if slice_seconds > _DEPLOY_FINAL_READ_SECONDS:
                slice_seconds -= _DEPLOY_FINAL_READ_SECONDS
            else:
                slice_seconds *= _DEPLOY_FINAL_READ_FRACTION
                final_read = True
            time.sleep(slice_seconds)
            remaining = deadline - time.monotonic()
            # the sleep can consume the whole budget. issuing the read anyway, with the fallback
            # bound below, is how `--wait 0.1` came to block for over a second: check after waking.
            if remaining <= 0:
                break
        # `--wait 0` is documented as "check once, do not block", so the first read happens before
        # the deadline applies. without it zero never calls deployment_for at all and the command
        # judges readiness from the POST body, which is queued on every normal async deploy.
        first = False
        try:
            # bound the read by what is left of the wait. the client's default timeout is 60s, so
            # an unbounded poll inside `--wait 5` blocks far past the deadline the user set. a
            # blanket 1s floor would do the same to shorter waits, so only the expired-budget read
            # -- which is just the zero-wait one-shot -- takes the fixed bound; every other read
            # honours the remainder exactly.
            budget = remaining if remaining > 0 else _DEPLOY_ZERO_WAIT_READ_SECONDS
            current = client.deployment_for(run_id, timeout=budget)
        except ApiError as exc:
            if exc.status in _PERMANENT_POLL_STATUSES:
                # retrying will not fix a rejected key or a run this key cannot see. without this
                # the loop burns the full timeout (30 minutes by default) on a request that
                # answers identically every time, and then reports it as "still queued".
                print(f"warning: cannot check {run_id}: {exc}", file=sys.stderr)
                return latest
        except ClientError:
            # a transient control-plane blip must not fail a deploy that is otherwise progressing;
            # keep polling to the deadline and report whatever we last saw.
            pass
        else:
            if current is None:
                # absence may mean rollback, not deletion: `deployment_for` filters by checkpoint
                # step while `mark_deployment_failed` restores the predecessor. inspect the run's
                # other revision to preserve `last_deploy_error`. recompute the remaining budget and
                # floor it at the zero-wait one-read bound so this classification can complete.
                left = deadline - time.monotonic()
                other = _rollback_record(client, run_id, max(left, _DEPLOY_ZERO_WAIT_READ_SECONDS))
                if other is not None:
                    return other
                print(
                    f"warning: {run_id} is no longer an active deployment; "
                    f"run `{CLI_NAME} models deployments` to check what happened",
                    file=sys.stderr,
                )
                return latest
            latest = current
            if str(current.get("state") or "") not in _DEPLOYMENT_BUSY_STATES:
                return current
    print(
        f"warning: still {str(latest.get('state') or 'unknown')!r} after {timeout:g}s; "
        f"run `{CLI_NAME} models deployments` to keep checking {run_id}",
        file=sys.stderr,
    )
    return latest


def _rollback_record(client, run_id: str, timeout: float) -> dict | None:
    """Return another listed revision after the requested one disappears.

    A failed cross-step redeploy restores the predecessor with ``last_deploy_error``. Preserve its
    real step and attempt stamp so ``_deployment_attempt_failed`` can report rollback accurately.
    """
    from flash.schema import parse_checkpoint_ref

    parsed = parse_checkpoint_ref(run_id)
    if parsed is None:
        return None
    base_run_id, _ = parsed
    # do not exempt final-adapter requests: a failed final redeploy can restore a checkpoint that
    # `deployment_for` excludes by step. `last_deploy_error`, not step direction, identifies rollback.
    try:
        entries = client.deployments(timeout=timeout)
    except (ApiError, ClientError):
        # this runs on the way out of a wait that has already ended; a failed lookup just means we
        # fall back to the original "no longer active" message rather than failing the command.
        return None
    # deployment_for is no help here: it resolves an exact revision, so asking it for the bare run
    # id means "the final adapter" (step None) and it rejects the rolled-back step just as it
    # rejected the requested one. match on the run id alone and let the caller judge identity.
    for entry in entries or ():
        listed = entry.get("deployment") or {}
        if base_run_id not in (listed.get("run_id"), entry.get("run_id")):
            continue
        if not listed.get("last_deploy_error"):
            # without a recorded error there is nothing tying this revision to the requested one's
            # disappearance, and reporting an unrelated deployment as this command's rollback would
            # be the "settle on whichever revision is listed" defect the step filter exists to stop.
            continue
        if not listed.get("run_id") and entry.get("run_id"):
            listed = {**listed, "run_id": entry["run_id"]}
        return listed
    return None


def _served_step_label(step: int | None) -> str:
    """Name a deployment's checkpoint the way the user addressed it: `step-N`, or `final`."""
    return "final" if step is None else f"step-{step}"


def _read_within(budget: float, read):
    """Run an advisory read, giving up on it entirely once ``budget`` seconds have passed.

    The client's two bounds are both per-phase, not total: `timeout` restarts on every socket
    operation and the body deadline is only checked once `urlopen` has finished parsing headers.
    A peer that trickles the status line or headers just inside the socket timeout therefore
    stalls for timeout x however many pauses it takes -- 12s against a 2s budget, measured -- and
    the whole point of the bound is that the deploy the user actually asked for is waiting behind
    it. Only wall-clock time outside the call bounds every phase at once.

    A daemon thread rather than a cancellation: a blocked socket read cannot be interrupted from
    outside, so the read is abandoned rather than stopped. That is sound only because this result
    is discardable -- no warning is the documented outcome for a plane that cannot answer, so an
    overrunning read degrades to silence. Daemon, so a stuck thread cannot keep the interpreter
    alive after the deploy finishes.
    """
    result: list = []
    worker = threading.Thread(
        target=lambda: result.append(_read_or_none(read)),
        daemon=True,
    )
    worker.start()
    worker.join(budget)
    return result[0] if result else None


def _read_or_none(read):
    """The advisory read's failure contract: a plane that cannot answer produces no warning.

    Deliberately broader than `ApiError`/`ClientError`. The client translates urllib's HTTPError
    and URLError, and documents that anything else propagates -- so a connection reset or a
    truncated body mid-read surfaces as the bare `ConnectionResetError` / `IncompleteRead` that
    urllib raised. Those are exactly the "plane cannot answer" case this exists to absorb, and on
    the worker thread an escaping one reaches the thread hook and prints a traceback across a
    deploy that is otherwise proceeding normally: a warning nobody asked for turning into noise
    on a command that worked.
    """
    try:
        return read()
    except Exception:
        # the deploy itself is the authority on whether it can proceed. failing it here -- or
        # printing a traceback beside it -- would turn a warning nobody asked for into a fault in
        # the command it decorates. BaseException still propagates, so KeyboardInterrupt is
        # unaffected.
        return None


def _alias_move_warning(client, base_run_id: str, requested_step: int | None) -> str | None:
    """Warn when deploying moves the shared `<run_id>` alias off a different checkpoint.

    `deploy` registers under the bare run id whatever step it is given, so deploying a second
    checkpoint of the same run does not stand up a second endpoint -- it repoints the one alias,
    and everyone chatting bare `<run_id>` silently changes model. That makes "deploy another
    checkpoint to compare" a destructive operation on the first, which reads as a serving fault
    rather than the deploy that caused it.

    Best-effort by construction: the read is advisory, so a plane that cannot answer must not
    stop the deploy. Returns None when nothing is being displaced -- no deployment, the same
    checkpoint again, or an unreadable current record.
    """
    current = _read_within(
        _ALIAS_WARNING_READ_SECONDS,
        # not `deployment_for`: its step filter hides exactly the record this asks about, a
        # DIFFERENT checkpoint holding the alias.
        lambda: client.deployed_checkpoint(
            base_run_id,
            timeout=_ALIAS_WARNING_READ_SECONDS,
            body_deadline=_ALIAS_WARNING_READ_SECONDS,
        ),
    )
    if current is None:
        return None
    # a `reconciling` record whose activation outcome was never recorded may ALREADY hold the
    # alias. the plane permits replacing exactly that record rather than rejecting it as busy,
    # and resolves the authoritative target through `_activation_predecessor` when it does
    # (flash/server/routes/serving.py). reading it as "not serving" suppressed the warning in
    # the very case the alias is most likely to move out from under someone.
    unknown_activation = current.get("activation_outcome_unknown") is True
    # `revocation_failed` is an undeploy whose BACKEND cleanup failed: local authority is revoked
    # but the alias may still resolve to the old target, and `_validate_deploy_request` rejects
    # only the busy set, so a deploy proceeds from here. reading it as "not serving" suppressed
    # the warning in the case where cleanup already left the alias ambiguous.
    revocation_failed = str(current.get("state") or "") == _REVOCATION_FAILED_STATE
    ambiguous = unknown_activation or revocation_failed
    if str(current.get("state") or "") not in _DEPLOYMENT_READY_STATES and not ambiguous:
        # nothing is being served off the alias yet, so nothing is lost by moving it.
        return None
    cli = CLI_NAME
    # `step-N` is not universally available: `ApiClient.chat` gates a step target behind
    # `_require_chat_step_selector`, which refuses outright on a plane that does not advertise
    # `chat_step_selector_v1`. the capability cannot be read here without spending a /v1/health on
    # an advisory warning, and `_chat_step_selector_available` starts False, so a negative reading
    # cannot tell "unsupported" from "not checked yet". state the condition instead of asserting
    # the command works, and name the selector that works on every plane.
    tail = (
        f"so every client using bare `{base_run_id}` changes model. address a specific "
        f"checkpoint with `{cli} models chat {base_run_id}/step-N` to compare them, or by "
        "immutable adapter revision if this control plane does not support step selectors."
    )
    if revocation_failed:
        # same reason as the unknown-activation arm: this record's `checkpoint_step` describes the
        # deployment whose revocation failed, and whether the backend still answers on the alias
        # is exactly what could not be determined. hedge rather than name a checkpoint.
        return (
            f"{base_run_id} has a deployment whose revocation did not complete, so whether it "
            f"still serves off that shared model id cannot be determined from here; deploying "
            f"{_served_step_label(requested_step)} may move it, "
            f"{tail} run `{cli} models deployments` to see the current state first."
        )
    if unknown_activation:
        # this record's `checkpoint_step` names the INCOMING attempt, not what the alias holds:
        # the plane resolves the live target from `adapter_alias_target` / `previous_deployment`,
        # and both are server-side (`public_deployment` strips the predecessor). naming a
        # checkpoint here would report the wrong one -- worse than naming none -- so say only what
        # this side can actually stand behind.
        return (
            f"{base_run_id} has a deployment whose activation never settled, so the checkpoint "
            f"it currently serves cannot be determined from here; deploying "
            f"{_served_step_label(requested_step)} may move that shared model id, "
            f"{tail} run `{cli} models deployments` to see the current state first."
        )
    raw_step = current.get("checkpoint_step")
    # `int()` does not just reject what it cannot read -- it silently COERCES. `true` becomes 1,
    # `false` becomes 0, and `1.5` truncates to 1, none of them raising. a step this client cannot
    # read as an exact whole number is an unreadable record, not checkpoint 1: reading it as a
    # number either suppresses the warning (deploying step-1 against a `true`) or names a
    # checkpoint that was never live. bool is checked first because it is a subclass of int.
    if isinstance(raw_step, bool):
        return None
    if isinstance(raw_step, float) and not raw_step.is_integer():
        return None
    try:
        served_step = None if raw_step is None else int(raw_step)
    except (TypeError, ValueError, OverflowError):
        # a proxy or older plane can answer 2xx with a step this client cannot read. that is an
        # unreadable current record like any other, NOT a reason to fail the deploy: leaving the
        # conversion unguarded let an advisory read traceback out of the command it decorates.
        # all three arms are reachable from one JSON body: `json.loads` accepts the non-standard
        # `Infinity`/`NaN` literals, and int() answers those with OverflowError and ValueError
        # respectively -- so catching only the obvious two still crashed the deploy.
        return None
    if served_step == requested_step:
        return None
    reach_displaced = ""
    # the record being displaced carries the one identifier that survives the alias move, so name
    # it whenever it is there. `step-N` is not a substitute: it is rejected outright on a plane
    # without `chat_step_selector_v1`, and after this deploy the listing shows only the NEW record
    # -- `public_deployment` strips `previous_deployment` (flash/serve/contract/urls.py) -- so a revision
    # not printed here is not recoverable from anywhere afterwards.
    revision = str(current.get("adapter_revision") or "").strip()
    if served_step is None:
        # the displaced checkpoint is the FINAL adapter, and `_parse_chat_target` accepts only a
        # bare run id, `RUN/step-N`, or a full immutable revision -- there is no `/final`
        # selector. once the alias moves, `step-N` addresses the incoming checkpoint and the bare
        # id is the thing that just changed, so the generic advice cannot reach the final adapter
        # at all. its immutable revision can, and the record being displaced carries one.
        reach_displaced = (
            f" the displaced final adapter has no `/final` selector; reach it by its immutable "
            f"revision `{revision}`."
            if revision
            # without a revision on the record there is nothing to hand the user, and deferring
            # them to a later command would be worse than saying so: `client.deploy` fires as soon
            # as this prints, and the deployments listing then shows only the NEW record --
            # `public_deployment` strips `previous_deployment` (flash/serve/contract/urls.py), so the
            # displaced revision is not in it. promising a selector that the next command cannot
            # produce is the one outcome worse than admitting there is none.
            else (
                " the displaced final adapter has no `/final` selector and this plane did not "
                "report its immutable revision, so it will not be addressable after this deploy; "
                "undeploy and redeploy it to serve it again."
            )
        )
    elif revision:
        # a numbered checkpoint keeps its `step-N` selector, but only on a plane that advertises
        # `chat_step_selector_v1`; where it does not, the qualified advice above leaves the user
        # holding no identifier at all. this is that identifier, and this is the last moment it
        # can be printed.
        reach_displaced = f" the displaced checkpoint's immutable revision is `{revision}`."
    return (
        f"{base_run_id} currently serves {_served_step_label(served_step)}; deploying "
        f"{_served_step_label(requested_step)} moves that shared model id onto the new "
        f"checkpoint, {tail}{reach_displaced}"
    )


def _deployment_attempt_failed(requested: dict, final: dict) -> bool:
    """True when the requested revision is not the one now served.

    ``mark_deployment_failed`` restores the previous ready record and writes only
    ``last_deploy_error``. Compare attempt identity or ``deploy --wait`` can accept the old adapter.
    """
    if str(final.get("state") or "") == "failed":
        return True
    asked = requested.get("requested_at")
    got = final.get("requested_at")
    # a POST that already answered with a settled record ran the deployment synchronously
    # (FLASH_DEPLOY_SYNC, flash/server/routes/serving.py), so it returned the FINISHED row and never
    # exposed the queued attempt. `requested` and `final` are then the same row and their stamps
    # match by construction -- a restored previous revision compares equal to itself and reads as
    # success. the recorded error is the only evidence left, and a deploy that really succeeded
    # writes a fresh record that carries none, so this cannot fire on one.
    if str(requested.get("state") or "") not in _DEPLOYMENT_BUSY_STATES:
        return bool(final.get("last_deploy_error"))
    # a differing stamp means the record on the plane belongs to some other deploy request. that
    # happens without any error at all: a concurrent `deploy` for the same run supersedes this one
    # and reaches ready on ITS checkpoint, and reading only last_deploy_error would call that this
    # command's success. compare the stamps whenever both sides carry one.
    if asked is not None and got is not None:
        return asked != got
    # no attempt stamp to compare: a recorded error is the only signal left.
    return bool(final.get("last_deploy_error"))


def cmd_deploy(args) -> int:
    from flash.schema import parse_checkpoint_ref

    # `flash models deploy <run_id>/step-n` is the same ref `flash runs checkpoint` prints.
    parsed = parse_checkpoint_ref(args.run_id)
    if parsed is None:
        print(
            f"invalid run/checkpoint reference {args.run_id!r} "
            "(expected <run_id> or <run_id>/step-N)",
            file=sys.stderr,
        )
        return 1
    base_run_id, step = parsed
    client = client_from_config()
    # before the POST: after it the alias has already moved, and a warning about a checkpoint that
    # is no longer served reads as history rather than a decision the reader still has. a dry run
    # registers nothing, so it displaces nothing and gets no warning.
    if not args.dry_run:
        alias_warning = _alias_move_warning(client, base_run_id, step)
        if alias_warning:
            print(
                render.warn(alias_warning) if render.styled() else f"warning: {alias_warning}",
                file=sys.stderr,
            )
    dep = client.deploy(args.run_id, dry_run=args.dry_run)
    wait_seconds = getattr(args, "wait", None)
    # a dry run creates no deployment to poll for, so --wait has nothing to observe. test against
    # None, not truthiness: `--wait 0` is an explicit "poll once, do not block" and 0.0 is falsy.
    waited_but_unservable = False
    if wait_seconds is not None and dep.get("state") != "dry_run":
        requested = dep
        dep = _await_deployment(client, args.run_id, dep, wait_seconds)
        # --wait promises the revision is servable on return, so require the plane to SAY it is
        # servable. "not busy" is a weaker claim that also covers a timeout, a vanished listing, an
        # unpollable plane, `revocation_failed`, and any state a newer plane adds; a restored
        # previous revision means the requested one never made it. exiting 0 on any of those lets
        # `deploy --wait && evaluate` run against the wrong adapter, or against none.
        not_ready = str(dep.get("state") or "") not in _DEPLOYMENT_READY_STATES
        waited_but_unservable = not_ready or _deployment_attempt_failed(requested, dep)
    if render.styled():
        print(render.deployed(dep))
    else:
        print(json.dumps(dep, indent=2))
    # a dry run creates no deployment, so the billing / undeploy hint would be misleading.
    if dep.get("state") != "dry_run":
        openai_base = str(dep.get("openai_base_url") or "")
        note = (
            f"serving is billed per token only; use `{CLI_NAME} models undeploy {base_run_id}` "
            "to deregister the adapter."
        )
        print(render.arrow(note) if render.styled() else f"note: {note}", file=sys.stderr)
        if openai_base:
            url_note = (
                f"OpenAI-compatible base URL: {openai_base} — point clients at this /v1 base, "
                "not the bare endpoint (which 404s on /chat/completions)."
            )
            print(
                render.arrow(url_note) if render.styled() else f"note: {url_note}", file=sys.stderr
            )
        state = dep.get("state", "deploying")
        if state == "failed":
            detail = str(dep.get("error") or dep.get("detail") or "unknown error")
            status_note = (
                f"deployment failed: {detail}; run `{CLI_NAME} models deployments` for details "
                f"and retry `{CLI_NAME} models deploy {args.run_id}` after fixing the error."
            )
        elif waited_but_unservable and dep.get("last_deploy_error"):
            # state reads `ready`, but it is the PREVIOUS revision: say so, or the reader trusts
            # the word and never learns the requested checkpoint is not the one being served.
            detail = str(dep.get("last_deploy_error"))
            status_note = (
                f"the requested revision did not become servable ({detail}); the previously "
                f"deployed revision is still serving. retry "
                f"`{CLI_NAME} models deploy {args.run_id}` after fixing the error."
            )
        elif waited_but_unservable:
            # the wait ended without the plane calling this revision servable, and there is no
            # recorded error to explain it: a timeout, or a terminal state that is not ready. the
            # generic "use chat once it is ready" below would read as success next to the exit 1.
            status_note = (
                f"deployment state is {state!r} after waiting; the requested revision is not "
                f"servable yet. run `{CLI_NAME} models deployments` to keep checking it."
            )
        else:
            status_note = (
                f"deployment state is {state!r}; run `{CLI_NAME} models deployments` to check "
                f"progress and use `{CLI_NAME} models chat` once it is ready."
            )
        print(
            render.arrow(status_note) if render.styled() else f"note: {status_note}",
            file=sys.stderr,
        )
    return 1 if dep.get("state") == "failed" or waited_but_unservable else 0


def _hub_repo_missing_errors() -> tuple[type[BaseException], ...]:
    """The hub errors that mean "the destination does not exist yet", not "you may not write".

    Imported lazily and tolerantly for the same reason as flash/serve/deployment/export.py's `_hub_error_types`:
    the CLI must stay importable without huggingface_hub. An empty tuple never matches, so a hub too
    old to export these names fails closed on the strict branch rather than waving an export through.
    """
    for module in ("huggingface_hub.errors", "huggingface_hub.utils"):
        try:
            return (
                __import__(module, fromlist=["RepositoryNotFoundError"]).RepositoryNotFoundError,
            )
        except (ImportError, AttributeError):
            continue
    return ()


def _hf_status_code(exc: BaseException) -> int | None:
    """The HTTP status behind a hub error, or None when the request never got an answer.

    None is the whole point: it separates "the Hub said no" from "the Hub was unreachable", which
    the preflight must treat differently.
    """
    code = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _hub_repo_gated_errors() -> tuple[type[BaseException], ...]:
    """Gated-repo errors, which the hub models as a SUBCLASS of "repository not found".

    So they match the missing-repo tuple and have to be subtracted from it. A gated destination
    exists; it is simply not writable by this token, which is the opposite of creatable.
    """
    for module in ("huggingface_hub.errors", "huggingface_hub.utils"):
        try:
            return (__import__(module, fromlist=["GatedRepoError"]).GatedRepoError,)
        except (ImportError, AttributeError):
            continue
    return ()


def _hf_repo_confirmed_to_exist(api: object, repository: str, token: str) -> bool:
    """Whether the Hub affirmatively said the destination is already there.

    True only on an affirmative answer. A missing `repo_exists`, an error, or any other non-answer
    is False, meaning "not confirmed", which leaves the creation rules in charge. That asymmetry is
    deliberate: bypassing those rules is what lets an export reach a namespace nobody verified, so
    it must be earned by a real answer rather than granted by the absence of one.
    """
    repo_exists = getattr(api, "repo_exists", None)
    if not callable(repo_exists):
        return False
    try:
        return bool(repo_exists(repository, repo_type="model", token=token))
    except Exception:
        return False


# sub-500 statuses that describe the request's transport rather than this token's access: request
# timeout, too-early replay, and rate limiting. 5xx is handled by the range check alongside these.
_HUB_TRANSIENT_STATUSES = frozenset({408, 425, 429})


def _hf_status_is_a_verdict(status: int | None) -> bool:
    """Whether an HTTP status is the Hub answering the permission question about this token.

    401/403/404 are answers: rejected, forbidden, or no such repo for this token. A missing status
    (no response at all), 408, 425, 429, and 5xx are not -- they say the Hub was busy, slow or
    broken, which is a fact about the Hub rather than about the caller's access.

    408 and 425 carry a response, so they reach here rather than the statusless path, but a request
    timeout and a too-early retry are transport outcomes: nothing about them says this token may not
    write. Blocking on one would blame the user's permissions for the Hub being slow.
    """
    if status is None:
        return False
    return status < 500 and status not in _HUB_TRANSIENT_STATUSES


def _without_token(text: str, token: str) -> str:
    """`text` with the token removed, for the warnings that quote a hub exception.

    A local credential rejection quotes what it was handed: httpx raises "Illegal header value
    b'Bearer <token>'" for a token containing a newline, so interpolating the exception prints the
    credential to stderr and into any log or pasted bug report. The exception is still worth showing
    -- it is the only clue to what went wrong -- so redact rather than drop it.

    The escaped renderings have to go too. httpx builds that message from the header BYTES, so a
    token holding a real newline appears as the two characters `\\` and `n`, which a literal replace
    of the token never matches -- the redaction silently does nothing in exactly the case it was
    written for. Each form is removed longest-first so a shorter one cannot bite a piece out of a
    longer one and leave the rest of the credential in place.
    """
    if not token:
        return text
    # repr() of the str and of the utf-8 bytes cover the escapings httpx and friends actually emit;
    # the [1:-1] strips the quotes repr adds so the inner escaped text is what gets matched.
    forms = {token, repr(token)[1:-1], repr(token.encode())[2:-1]}
    for form in sorted(forms, key=len, reverse=True):
        if form:
            text = text.replace(form, "<redacted>")
    return text


def _hf_hub_version() -> str:
    """The installed hub version for messages, without importing the package at module scope.

    Falls back to a phrase, not a version-shaped string: this is only reached when the version could
    not be read, and printing something that looks like a number would name a version nobody has.
    """
    try:
        from huggingface_hub import __version__

        return str(__version__)
    except Exception:
        return "(unknown version)"


def _hf_identity_and_write_access(repository: str, token: str) -> str | None:
    """return the token account after verifying the destination namespace when hub is installed."""
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        if exc.name != "huggingface_hub":
            raise
        return None

    import inspect

    api = HfApi()
    try:
        identity = api.whoami(token=token)
    except Exception as exc:
        # a rejected token is the Hub answering, so refuse. a Hub that was unreachable, rate limited
        # or broken is not an answer: the copy runs on the control plane, not here, so a CLI host
        # without Hub egress would otherwise be unable to export at all -- while the same command
        # skips this check entirely when the package is simply absent. degrade to that behaviour
        # rather than invent a new hard blocker.
        if not _hf_status_is_a_verdict(_hf_status_code(exc)):
            print(
                "warning: could not reach HuggingFace to verify the export namespace "
                f"({_without_token(str(exc), token)}); proceeding without the check",
                file=sys.stderr,
            )
            return None
        raise ClientError(
            "HuggingFace rejected the token before export. Check HF_TOKEN or --api-key and retry."
        ) from exc
    account = str(identity.get("name") or identity.get("username") or "").strip()
    if not account:
        raise ClientError(
            "HuggingFace authenticated the token but returned no account name, so Flash could not "
            "verify the export namespace. Update huggingface-hub or verify the token manually."
        )
    print(f"HuggingFace token resolves to account {account}", file=sys.stderr)

    owner = repository.partition("/")[0].strip()

    # the exact repo first: it is the only authoritative answer, and it is the one the export
    # actually needs. asking the org role first would refuse a `contributor` who can write this
    # very repo, because a role is a coarser fact than the permission being checked.
    auth_check = getattr(api, "auth_check", None)
    exact_probe = callable(auth_check) and "write" in inspect.signature(auth_check).parameters
    if not exact_probe and _hf_repo_confirmed_to_exist(api, repository, token):
        # `auth_check(..., write=True)` only exists from huggingface-hub 1.5, and this package still
        # supports >=1.2, so on 1.2-1.4 there is no exact write probe. the rules below are CREATION
        # rules and a destination that already exists is not being created, so applying them here
        # would reject the org `contributor` this ordering was fixed to admit. degrade to a warning,
        # exactly as this function already does when the Hub is unreachable or the package is absent.
        # only a CONFIRMED existing repo takes this path. an unanswered lookup keeps the creation
        # rules, which is what stops an unverified export into someone else's namespace.
        print(
            f"warning: huggingface-hub {_hf_hub_version()} cannot check write access to an "
            f"existing repo; proceeding without verifying {repository}",
            file=sys.stderr,
        )
        return account
    if exact_probe:
        try:
            auth_check(repository, repo_type="model", token=token, write=True)
            return account
        except Exception as exc:
            # a destination that is not there yet cannot be checked directly, so fall through to the
            # creation rules below. GatedRepoError SUBCLASSES RepositoryNotFoundError, so it must be
            # excluded first: a gated repo exists and this token may not write it, and reading it as
            # "absent" would hand it to the weaker create-permission paths.
            missing = _hub_repo_missing_errors()
            absent = bool(missing) and isinstance(exc, missing)
            if absent and isinstance(exc, _hub_repo_gated_errors()):
                absent = False
            if not absent and not _hf_status_is_a_verdict(_hf_status_code(exc)):
                # no verdict about this token: a timeout or DNS failure (no status at all), a 429,
                # or a 5xx. reporting any of those as "cannot write" blames the user's permissions
                # for a transport fault, and the upload runs on the control plane anyway. warn and
                # proceed, exactly as the whoami path above already does for the same class of
                # failure. 401/403/404 DO answer the question and still fall through to the error.
                print(
                    f"warning: could not verify write access to {repository} "
                    f"({_without_token(str(exc), token)}); proceeding without the check",
                    file=sys.stderr,
                )
                return account
            if not absent:
                raise ClientError(
                    f"HuggingFace token resolves to account {account}, but it cannot write to "
                    f"{repository}. Grant this token write access to that model repo or set HF_TOKEN "
                    "to a token that can write there."
                ) from exc

    # only reached for a destination that does not exist yet, so the question is now whether this
    # token may CREATE it in that namespace. that is what the org role legitimately answers.
    org_role = ""
    for org in identity.get("orgs") or ():
        if not isinstance(org, dict) or str(org.get("name") or "").casefold() != owner.casefold():
            continue
        org_role = str(org.get("role") or org.get("roleInOrg") or "").lower()
        break
    if owner.casefold() != account.casefold() and org_role not in {"write", "admin"}:
        raise ClientError(
            f"HuggingFace token resolves to account {account}, which cannot create "
            f"{repository} in the namespace {owner}. Use --repository {account}/<repo>, choose an "
            "org where this account has write access, or set HF_TOKEN to the intended account."
        )

    access_token = (identity.get("auth") or {}).get("accessToken") or {}
    token_role = str(access_token.get("role") or "").lower()
    if token_role == "write":
        return account
    if token_role != "fineGrained".lower():
        raise ClientError(
            f"HuggingFace token resolves to account {account}, but Flash could not verify write "
            f"scope for the new destination {repository}. Use a write token for that namespace, "
            "or create the repo first so Flash can check its exact write permission."
        )

    permissions = set(access_token.get("fineGrained", {}).get("global") or ())
    for scope in access_token.get("fineGrained", {}).get("scoped") or ():
        if not isinstance(scope, dict):
            continue
        entity = scope.get("entity") or {}
        entity_name = str(entity.get("name") or entity.get("id") or "")
        # only a scope naming the destination counts. crediting every user-typed scope would accept
        # a token whose write grant covers some unrelated repo of its own, which is the silent
        # wrong-namespace export this preflight exists to stop.
        if entity_name.casefold() in {owner.casefold(), repository.casefold()}:
            permissions.update(scope.get("permissions") or ())
    if not ({"repo.write", "repo.content.write"} & permissions):
        raise ClientError(
            f"HuggingFace token resolves to account {account}, but its fine-grained scopes do not "
            f"verify write access for {repository}. Add repository-content write access for that "
            "namespace or use a write token."
        )
    return account


def cmd_export(args) -> int:
    from flash.client.runtime_secrets import resolve_hf_token

    hf_token = resolve_hf_token(args.api_key)
    if not hf_token:
        raise ClientError(
            "no HuggingFace token: pass `--api-key <hf_...>`, or set HF_TOKEN "
            "(export it in your shell or put it in a local .env / .env.local)"
        )
    if args.api_key:
        print(
            "warning: --api-key is visible in process listings; prefer HF_TOKEN or a local "
            ".env / .env.local",
            file=sys.stderr,
        )
    _hf_identity_and_write_access(args.repository, hf_token)
    client = client_from_config()
    progress = (
        f"exporting adapter {args.adapter_id} to {args.repository}; "
        "downloading then re-uploading; this can take a minute..."
    )
    print(render.note(progress) if render.styled() else progress, file=sys.stderr)
    result = client.export(
        args.adapter_id,
        repository=args.repository,
        hf_token=hf_token,
        private=not args.public,
    )
    if render.styled():
        # the control-plane result carries no `private` key, so reflect the privacy we requested
        # (the server applies exactly this) rather than mislabeling a private export as public.
        print(render.exported({**result, "private": not args.public}))
    else:
        print(json.dumps(result, indent=2))
    url = result.get("url", args.repository)
    print(
        render.arrow(f"exported to {url}") if render.styled() else f"exported to {url}",
        file=sys.stderr,
    )
    return 0


def cmd_undeploy(args) -> int:
    result = client_from_config().undeploy(args.run_id)
    if render.styled():
        print(render.undeployed(result))
    else:
        print(json.dumps(result, indent=2))
    return 0


def cmd_deployments(args) -> int:
    rows = client_from_config().deployments()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        if render.styled():
            print(render.empty("deployments", "0 active", "no active deployments"))
        else:
            print("no active deployments")
        return 0
    if render.styled():
        print(tables.deployments_table(rows))
        return 0
    print(
        f"{'RUN ID':<30}  {'STEP':<6}  {'REVISION':<40}  {'STATE':<14}  "
        f"{'VERIFIED AT':<20}  {'OPENAI MODEL':<30}  {'OPENAI BASE URL':<48}  DETAIL"
    )
    for row in rows:
        deployment = row.get("deployment") or {}
        run_id = str(deployment.get("run_id") or row.get("run_id") or "")
        step = deployment.get("checkpoint_step")
        step_text = "final" if step is None else str(step)
        verified_at = deployment.get("verified_at")
        verified_text = (
            "-" if verified_at is None else (render._humanize_ts(verified_at) or str(verified_at))
        )
        revision = str(deployment.get("adapter_revision") or "-")
        state = str(deployment.get("state") or "-")
        openai_model = str(deployment.get("openai_model") or run_id)
        openai_base_url = str(deployment.get("openai_base_url") or "-")
        detail = str(deployment.get("error") or deployment.get("detail") or "")[:160]
        print(
            f"{run_id:<30}  {step_text:<6}  {revision:<40}  {state:<14}  "
            f"{verified_text:<20}  {openai_model:<30}  {openai_base_url:<48}  {detail}"
        )
    return 0


def cmd_chat(args) -> int:
    from flash.schema import parse_adapter_revision, parse_checkpoint_ref

    revision = parse_adapter_revision(args.run_id)
    parsed = parse_checkpoint_ref(args.run_id) if revision is None else None
    if revision is None and parsed is None:
        print(
            f"invalid chat target {args.run_id!r} "
            "(expected a bare <run_id>, <run_id>/step-N, or full immutable adapter revision)",
            file=sys.stderr,
        )
        return 1
    chat_target = args.run_id
    client = client_from_config()
    messages = [{"role": "user", "content": args.message}]
    system = getattr(args, "system", None)
    if system:
        messages.insert(0, {"role": "system", "content": system})
    wrote = False
    pending: list[str] = []
    for chunk in client.chat_stream(
        chat_target,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ):
        # delay the label and blank chunks until real text arrives. otherwise an empty response has
        # non-empty stdout and cannot serve as a health check. release buffered blanks verbatim;
        # flash/cli/commands/env/testing/eval.py grades emptiness the same way.
        if not wrote:
            pending.append(chunk)
            if not chunk.strip():
                continue
            if render.styled():
                print(render.chat_label())
            chunk = "".join(pending)
            wrote = True
        print(chunk, end="", flush=True)
    if not wrote:
        # the request succeeded at the transport level but carried no assistant text, which is what
        # a serving path that stopped applying the run's chat template looks like from here. exiting
        # 0 with an empty stdout makes that indistinguishable from a model that answered nothing, so
        # this surface cannot be used as a health check -- say what happened and fail.
        print(
            f"no response text from {chat_target}: the request succeeded but the model returned "
            "nothing. the deployment may be unhealthy or still starting; check "
            f"`{CLI_NAME} models deployments` and retry.",
            file=sys.stderr,
        )
        return 1
    print()
    return 0

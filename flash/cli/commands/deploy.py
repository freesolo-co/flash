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

from flash.cli.ui import render
from flash.client import ApiError, ClientError


def _commands():
    """The parent package, imported lazily because it re-exports this module.

    `client_from_config` and `CLI_NAME` are patched as attributes of `flash.cli.commands` by the
    cli tests -- the first to install a fake client, the second to prove the dev channel's
    `flash-dev` name reaches the hints these handlers print. Importing either by value here would
    bind the original before the patch lands, so the patch would rebind a name this module never
    reads.
    """
    from flash.cli import commands

    return commands


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
                    f"run `{_commands().CLI_NAME} models deployments` to check what happened",
                    file=sys.stderr,
                )
                return latest
            latest = current
            if str(current.get("state") or "") not in _DEPLOYMENT_BUSY_STATES:
                return current
    print(
        f"warning: still {str(latest.get('state') or 'unknown')!r} after {timeout:g}s; "
        f"run `{_commands().CLI_NAME} models deployments` to keep checking {run_id}",
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
    if str(current.get("state") or "") not in _DEPLOYMENT_READY_STATES and not unknown_activation:
        # nothing is being served off the alias yet, so nothing is lost by moving it.
        return None
    cli = _commands().CLI_NAME
    tail = (
        f"so every client using bare `{base_run_id}` changes model. address a specific "
        f"checkpoint with `{cli} models chat {base_run_id}/step-N` to compare them."
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
    return (
        f"{base_run_id} currently serves {_served_step_label(served_step)}; deploying "
        f"{_served_step_label(requested_step)} moves that shared model id onto the new "
        f"checkpoint, {tail}"
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
    client = _commands().client_from_config()
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
            f"serving is billed per token only; use `{_commands().CLI_NAME} models undeploy {base_run_id}` "
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
                f"deployment failed: {detail}; run `{_commands().CLI_NAME} models deployments` for details "
                f"and retry `{_commands().CLI_NAME} models deploy {args.run_id}` after fixing the error."
            )
        elif waited_but_unservable and dep.get("last_deploy_error"):
            # state reads `ready`, but it is the PREVIOUS revision: say so, or the reader trusts
            # the word and never learns the requested checkpoint is not the one being served.
            detail = str(dep.get("last_deploy_error"))
            status_note = (
                f"the requested revision did not become servable ({detail}); the previously "
                f"deployed revision is still serving. retry "
                f"`{_commands().CLI_NAME} models deploy {args.run_id}` after fixing the error."
            )
        elif waited_but_unservable:
            # the wait ended without the plane calling this revision servable, and there is no
            # recorded error to explain it: a timeout, or a terminal state that is not ready. the
            # generic "use chat once it is ready" below would read as success next to the exit 1.
            status_note = (
                f"deployment state is {state!r} after waiting; the requested revision is not "
                f"servable yet. run `{_commands().CLI_NAME} models deployments` to keep checking it."
            )
        else:
            status_note = (
                f"deployment state is {state!r}; run `{_commands().CLI_NAME} models deployments` to check "
                f"progress and use `{_commands().CLI_NAME} models chat` once it is ready."
            )
        print(
            render.arrow(status_note) if render.styled() else f"note: {status_note}",
            file=sys.stderr,
        )
    return 1 if dep.get("state") == "failed" or waited_but_unservable else 0


def cmd_export(args) -> int:
    from flash.client.runtime_secrets import resolve_hf_token

    hf_token = resolve_hf_token(args.api_key)
    if not hf_token:
        raise ClientError(
            "no HuggingFace token: pass `--api-key <hf_...>`, or set HF_TOKEN "
            "(export it in your shell or put it in a local .env / .env.local)"
        )
    client = _commands().client_from_config()
    progress = (
        f"exporting adapter {args.adapter_id} to {args.repository} — "
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
    result = _commands().client_from_config().undeploy(args.run_id)
    if render.styled():
        print(render.undeployed(result))
    else:
        print(json.dumps(result, indent=2))
    return 0

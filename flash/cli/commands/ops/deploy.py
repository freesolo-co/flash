"""Deploy, export and undeploy: the commands that move a finished run into serving.

`cmd_deploy --wait` is the reason this is its own module -- polling a revision to servable is a
state machine with its own timing constants, rollback read and permanent-failure rules, and it
dwarfed every other command handler in the package.

Split out of `flash.cli.commands` to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import sys
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
    # deployment_for resolves one exact checkpoint and cannot enumerate a run's siblings. match on
    # run grouping metadata only, then let the caller judge the exact checkpoint identity.
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
            "(expected <run_id>/final or <run_id>/step-N)",
            file=sys.stderr,
        )
        return 1
    client = client_from_config()
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
            f"serving is billed per token only; use `{CLI_NAME} models undeploy {args.run_id}` "
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
        f"{'RUN ID':<30}  {'STEP':<6}  {'CHECKPOINT ID':<40}  {'STATE':<14}  "
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
        checkpoint_id = str(deployment.get("checkpoint_id") or "-")
        state = str(deployment.get("state") or "-")
        openai_model = str(deployment.get("openai_model") or run_id)
        openai_base_url = str(deployment.get("openai_base_url") or "-")
        detail = str(deployment.get("error") or deployment.get("detail") or "")[:160]
        print(
            f"{run_id:<30}  {step_text:<6}  {checkpoint_id:<40}  {state:<14}  "
            f"{verified_text:<20}  {openai_model:<30}  {openai_base_url:<48}  {detail}"
        )
    return 0


def cmd_chat(args) -> int:
    from flash.schema import parse_checkpoint_ref

    parsed = parse_checkpoint_ref(args.run_id)
    if parsed is None:
        print(
            f"invalid chat target {args.run_id!r} (expected <run_id>/final or <run_id>/step-N)",
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

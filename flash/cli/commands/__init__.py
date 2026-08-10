"""CLI command handlers for the managed Flash service."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

from flash import __version__
from flash._internal.channel import CLI_NAME
from flash._internal.logging import get_logger
from flash.cli.ui import render
from flash.cli.ui.tty import TtyStatusLine
from flash.client import (
    ApiClient,
    ApiError,
    ClientError,
    client_from_config,
    save_credentials,
    verify_freesolo_key,
)

# `shadowed_login_warning` has no call site left here since the cost quote moved to
# `.train_cost`, but the estimate tests patch it on THIS module and that quote reads it back
# through the package. an autofix that drops it as unused breaks those tests.
from flash.client.config import (  # noqa: F401
    load_credentials,
    load_credentials_with_source,
    shadowed_login_warning,
)
from flash.client.http import has_freesolo_backend
from flash.client.runtime_secrets import runtime_secrets_from_local_env
from flash.client.specs import spec_payload
from flash.core.catalog import public_model_rows
from flash.runner import TERMINAL_STATES
from flash.schema import (
    ConfigError,
    spec_and_train_keys_from_file,
)
from flash.serve.urls import is_freesolo_hosted_url

logger = get_logger("flash.cli")


_USER_ERRORS = (
    ConfigError,
    ClientError,
    FileNotFoundError,
    ValueError,
)

_CLI_DONE_STATES = TERMINAL_STATES | {"deployed"}
_OK_STATES = {"done", "dry_run", "deployed"}
_SPINNER_FRAMES = "|/-\\"
_SPINNER_TICK_SECONDS = 0.1


class _LogFollowSpinner(TtyStatusLine):
    def __init__(self, run_id: str):
        super().__init__()
        self._run_id = run_id
        self._frame = 0

    def render(self, progress: str) -> None:
        if not self._enabled:
            return
        frame = _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)]
        self._frame += 1
        message = f"{frame} following logs for {self._run_id} ({progress})"
        self._write(message)


def _sleep_with_spinner(interval: float, spinner: _LogFollowSpinner, progress: str) -> None:
    if interval <= 0:
        return
    if not spinner.enabled:
        time.sleep(interval)
        return
    ticks = max(1, int(interval / _SPINNER_TICK_SECONDS))
    sleep_for = interval / ticks
    for _ in range(ticks):
        spinner.render(progress)
        time.sleep(sleep_for)


def cmd_version(args) -> int:
    if render.styled():
        print(render.version(__version__))
    else:
        print(f"{CLI_NAME} {__version__}")
    return 0


def unavailable_without_a_freesolo_backend(what: str, *, because: str, instead: str) -> ClientError:
    """The refusal for a command backed only by the hosted Freesolo backend.

    These commands are pure client->api.freesolo.co calls that never touch the plane. Left alone
    they send the operator's plane credential to a service they have no relationship with, which
    answers 401 -- the same failure `flash env setup` hit on the documented quickstart. Naming the
    reason and the way forward is what separates "not built for your deployment" from "your key is
    wrong", which is what a bare 401 says.
    """
    return ClientError(f"{what} is not available on a self-hosted plane: {because}. {instead}.")


def _verifies_against_freesolo(api_url: str, freesolo_url: str | None) -> bool:
    """Whether ``flash login`` should check this key against the hosted Freesolo backend.

    A self-hosted plane verifies ``FREESOLO_INTERNAL_KEY`` itself, so sending that credential to
    the hosted backend would leak it. The client can infer standalone mode only from ``--api-url``;
    an explicit ``--freesolo-url`` still requests external verification. False routes verification
    to ``_verify_key_against_plane`` rather than accepting the key unchecked.
    """
    if freesolo_url:
        return True
    return is_freesolo_hosted_url(api_url)


def _plaintext_transport_warning(api_url: str) -> str:
    """Warn when the api url would carry the key over unencrypted, non-loopback HTTP.

    The key travels as an ``Authorization: Bearer`` header on login and on every command after it,
    and on a standalone plane it is the entire authorization boundary and owns every run -- so an
    observer can submit billed GPU jobs and read or cancel existing ones. Loopback never leaves the
    machine, so it stays quiet: local development is the one place plaintext is fine.

    A warning rather than a refusal: an operator may front the plane with a TLS-terminating proxy
    reached over a private link, and this cannot see that. It must be printed BEFORE the key is
    transmitted, so the caller can still abort.
    """
    from urllib.parse import urlparse

    parsed = urlparse((api_url or "").strip())
    if parsed.scheme != "http":
        return ""
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".localhost"):
        return ""
    return (
        f"{api_url} is plaintext HTTP, so your API key is sent unencrypted and anyone who can "
        "observe the connection can reuse it to submit billed GPU jobs and control your runs. "
        "Use https:// for a remote plane; plaintext is safe only on localhost."
    )


def _plaintext_login_warnings(api_url: str | None, freesolo_url: str | None) -> list[str]:
    """Warn for every url `flash login` may send the key to, not just the plane's.

    Login has two possible destinations and picks between them at request time: the plane itself, or
    the Freesolo identity backend via ``verify_freesolo_key(base_url=...)``. Checking only
    ``api_url`` left the second one silent, so an https plane paired with an ``http://``
    ``--freesolo-url`` (or ``FREESOLO_BASE_URL``) transmitted the bearer key in cleartext with no
    warning at all -- the case where the operator is most likely to believe they are protected.
    """
    seen: set[str] = set()
    warnings: list[str] = []
    for url in (api_url, freesolo_url):
        normalized = (url or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        warning = _plaintext_transport_warning(normalized)
        if warning:
            warnings.append(warning)
    return warnings


def cmd_login(args) -> int:
    api_url = args.api_url or load_credentials()[0]
    identity: dict | None = None
    # before any request: the warning is worthless once the key has already gone over the wire.
    # both destinations are checked here rather than at the branch below, because which one receives
    # the key is decided after this point and the caller needs the warning while it can still abort.
    for transport_warning in _plaintext_login_warnings(
        api_url, getattr(args, "freesolo_url", None)
    ):
        print(
            render.warn(transport_warning) if render.styled() else f"warning: {transport_warning}",
            file=sys.stderr,
        )
    try:
        env_api_key = os.environ.get("FREESOLO_API_KEY")
        api_key = args.api_key or env_api_key
        if not api_key:
            raise ClientError(
                "no API key provided: pass `--api-key <key>` or set FREESOLO_API_KEY. "
                "Create or copy a key at https://freesolo.co/sign-in."
            )
        freesolo_url = getattr(args, "freesolo_url", None)
        if _verifies_against_freesolo(api_url, freesolo_url):
            verify_freesolo_key(api_key, base_url=freesolo_url)
        else:
            # a self-hosted plane is its own key issuer, so it is the only service that CAN
            # verify this key -- and the only one allowed to see it.
            identity = _verify_key_against_plane(api_key, api_url)
    except ClientError as exc:
        if getattr(args, "debug", False):
            raise
        print(render.login_failed(str(exc)), file=sys.stderr)
        return 1
    save_credentials(api_key, api_url=api_url)
    if args.api_key and env_api_key and env_api_key != args.api_key:
        msg = (
            "FREESOLO_API_KEY is set and will override this saved login for future "
            "commands; unset FREESOLO_API_KEY to use the saved key."
        )
        print(render.warn(msg) if render.styled() else f"warning: {msg}", file=sys.stderr)
    # Show who they are right away (the same identity `flash whoami` prints) so they don't
    # have to run a second command. Never echo the key itself. The identity lookup is
    # best-effort: the key is already verified and stored, so a momentary control-plane
    # hiccup must not turn a successful login into a failure.
    print(
        render.login_ok(identity if identity is not None else _identity_or_none(api_key, api_url))
    )
    return 0


_IDENTITY_LOOKUP_TIMEOUT_S = 5.0
# Verification is MANDATORY, so it gets a normal request budget rather than the 5s the optional
# identity card uses: abandoning a slow lookup is harmless when the card is decoration, but here it
# would reject a valid key. Matches the hosted path's 30s (`verify_freesolo_key`) so the same
# decision gets the same budget whoever answers it -- a self-hosted plane that is cold-starting,
# behind a slow proxy, or under database contention is exactly the documented quickstart case.
_LOGIN_VERIFY_TIMEOUT_S = 30.0


def _verify_key_against_plane(api_key: str, api_url: str) -> dict:
    """Verify a key through the self-hosted plane's authenticated identity endpoint.

    Hosted verification would leak the plane-root credential, while skipping verification stores
    invalid keys. ``/v1/me`` both authenticates and returns identity. Require ``kind`` and
    ``key_prefix`` because ``_request`` converts an empty 2xx body to ``{}``; those fields are
    unconditional in ``flash/server/routes/meta.py``.
    """
    identity = ApiClient(api_url, api_key, timeout=_LOGIN_VERIFY_TIMEOUT_S).me()
    if not isinstance(identity, dict) or not identity.get("kind") or not identity.get("key_prefix"):
        raise ClientError(
            f"{api_url} answered the login check but did not return a Flash identity, so the key "
            "could not be verified and was not saved. Check that --api-url points at your Flash "
            'control plane (its /v1/health should report "service": "flash") rather than at a '
            "proxy or another service."
        )
    return identity


def _identity_or_none(api_key: str, api_url: str) -> dict | None:
    # Don't use client_from_config(): ambient FREESOLO_API_KEY would win and show wrong identity.
    try:
        return ApiClient(api_url, api_key, timeout=_IDENTITY_LOOKUP_TIMEOUT_S).me()
    except (ClientError, OSError, ValueError):
        return None


def cmd_whoami(args) -> int:
    _, _, key_source = load_credentials_with_source()
    print(render.whoami(client_from_config().me(), key_source))
    return 0


def cmd_projects_create(args) -> int:
    from flash.client import create_project

    api_url, api_key = load_credentials()
    if not has_freesolo_backend(api_url):
        # no directory to create a row in, and none needed: the plane accepts any well-shaped
        # uuid (server/projects.py under standalone()), so minting one locally IS the create.
        project_id = str(uuid.uuid4())
    else:
        if not api_key:
            raise ClientError("not logged in. Run `flash login` before creating a project")
        project_id = create_project(args.name, getattr(args, "description", None), api_key)["id"]

    # both ids are reported the same way: where it came from is the only difference above.
    if render.styled():
        print(render.project_created(project_id, str(args.name).strip()))
    else:
        print(project_id)
    return 0


def cmd_projects_list(args) -> int:
    from flash.client import list_projects

    api_url, api_key = load_credentials()
    if not has_freesolo_backend(api_url):
        raise unavailable_without_a_freesolo_backend(
            "listing projects",
            because="a self-hosted plane keeps no project directory to enumerate",
            instead=(
                f"`{CLI_NAME} projects create <name>` mints a project id you record yourself; "
                "any id you already use in a config keeps working"
            ),
        )
    if not api_key:
        raise ClientError("not logged in. Run `flash login` before listing projects")
    projects = list_projects(api_key)
    if render.styled():
        print(render.projects_table(projects))
        return 0
    for project in projects:
        project_id = str(project.get("id") or "").strip()
        name = str(project.get("name") or "").strip()
        print(f"{project_id}\t{name}")
    return 0


def cmd_models(args) -> int:
    rows = public_model_rows()
    if render.styled():
        print(render.models_table(rows))
        return 0
    for row in rows:
        print(row["id"])
    return 0


def cmd_gpus(args) -> int:
    """List validated managed GPU classes, VRAM, and estimated $/hr."""
    from flash.providers.base import GPU_INFO
    from flash.providers.runpod.pricing import static_rates as runpod_static_rates

    runpod_rates = runpod_static_rates()
    infos = sorted(
        (info for info in GPU_INFO.values() if info.enum_member), key=lambda g: g.hourly_usd
    )
    tip = (
        "Tip: GPU allocation is automatic by default.\n"
        "The allocator picks the cheapest validated class that fits. Pin a specific class by "
        'adding type = "<CLASS>" to the [gpu] section.'
    )
    if render.styled():
        rows = [(info.name, info.vram_gb, runpod_rates.get(info.name)) for info in infos]
        print(render.gpus_table(rows, tip))
        return 0

    def fmt_rate(v: float | None) -> str:
        return f"{v:>10.2f}" if v else f"{'-':>10}"

    print(f"{'gpu':<16}{'vram':>6}{'$/hr':>11}")
    for info in infos:
        runpod_rate = runpod_rates.get(info.name)
        print(f"{info.name:<16}{info.vram_gb:>5}G{fmt_rate(runpod_rate):>11}")
    print(f"\n{tip}")
    return 0


def cmd_env_list(args) -> int:
    paths: list[str] = []
    if Path("environment.py").is_file():
        paths.append(".")
    local = Path("environments")
    if local.is_dir():
        for p in local.iterdir():
            if p.name.startswith("__"):
                continue
            if p.is_dir():
                stem = p.name.replace("-", "_")
                module = p / f"{stem}.py"
                canonical = p / "environment.py"
                if canonical.is_file() or module.is_file():
                    paths.append(f"environments/{p.name}")
            elif p.suffix == ".py":
                paths.append(f"environments/{p.name}")
    if render.styled():
        print(render.env_list(sorted(paths)))
        return 0
    if paths:
        print(
            "local env sources (publish with `flash env push --project <project-uuid> "
            "--name <name> <path>`):"
        )
        for path in sorted(paths):
            print(f"  {path}")
    else:
        print("no environments yet - scaffold one with `flash env setup`")
    return 0


def cmd_train(args) -> int:
    if getattr(args, "cost", False):
        return _cmd_train_cost(args)
    spec, authored_train_keys = spec_and_train_keys_from_file(
        args.config,
        run_id=None,
        overrides=args.overrides,
        extra_configs=args.extra_configs,
        project_required=True,
    )
    payload = spec_payload(spec, authored_train_keys=authored_train_keys)
    client = client_from_config()
    client_train_schema = _client_train_schema(authored_train_keys)
    runtime_secrets = (
        runtime_secrets_from_local_env(args.config, keys=spec.environment.secrets) or None
    )
    _warn_if_wandb_requested_without_key(spec, runtime_secrets, dry_run=bool(args.dry_run))
    if args.dry_run:
        # dry-run runs submit-time server preflights without allocating a training gpu or charging
        # for training. a rejection surfaces as the server's error with exit status 1. for sft it
        # also requires an exact workload profile, so a miss starts a separate billed profile run
        # (see _raise_if_workload_profile_pending) instead of previewing anything.
        try:
            status = client.create_run(
                payload,
                runtime_secrets=runtime_secrets,
                dry_run=True,
                client_train_schema=client_train_schema,
            )
        except ApiError as exc:
            _raise_if_workload_profile_pending(client, exc)
            detail = _legacy_train_key_rejection_detail(exc, authored_train_keys)
            if detail is None:
                raise
            raise ApiError(exc.status, detail, detail=detail) from exc
        compatibility = status.pop("train_schema_compatibility", None)
        _print_train_schema_compatibility(compatibility)
        # the server fails open on a billing-infra problem, so "cost" is only in the validated list
        # when it was actually checked. absent key = a server that predates the signal, which is
        # equally not a verification -- so treat anything but an explicit True as unverified.
        affordability_verified = status.pop("affordability_verified", None) is True
        cost = "and cost" if affordability_verified else "but NOT cost"
        # sft additionally required a matching workload profile to get this far, and that profile
        # run already imported environment.py and tokenized the dataset. claiming otherwise here
        # would understate what has been checked -- and what has already been billed.
        environment = (
            "your environment.py and the exact dataset were already loaded and tokenized by the "
            "workload profile this quote is built on; model load and gpu/training are first "
            "exercised on the worker after cold-start."
            if spec.algorithm == "sft"
            else "it did NOT import or run your environment.py; dataset loading, "
            "start_episode/episode shapes, reward/scorer, worker imports, model load, and "
            "gpu/training are first exercised on the worker after cold-start."
        )
        print(
            "dry-run validated: config/schema, model+algorithm compatibility, lora rank, "
            f"runtime-secret presence, warm-start source, serving context cap, {cost}. "
            f"{environment}",
            file=sys.stderr,
        )
        if not affordability_verified:
            print(
                "affordability was NOT verified (billing check unavailable); this run can still be "
                "rejected for insufficient balance when you submit it for real.",
                file=sys.stderr,
            )
        if render.styled():
            print(
                render.object_panel(
                    "train", status, "dry run — validated by the server, not submitted"
                )
            )
        else:
            print(json.dumps(status, indent=2))
        return 0
    try:
        status = client.create_run(
            payload,
            runtime_secrets=runtime_secrets,
            client_train_schema=client_train_schema,
        )
    except ApiError as exc:
        # a real submit misses the profile cache the same way a preview does, and the miss starts a
        # separately billed profile run. without this the user sees a bare 409 for a charge they
        # were never told about (see _raise_if_workload_profile_pending).
        _raise_if_workload_profile_pending(client, exc)
        raise
    run_id = status["run_id"]
    logger.info(
        "submitted run %s: model=%s algorithm=%s gpu=%s",
        run_id,
        spec.model,
        spec.algorithm,
        spec.gpu.type or "auto",
    )
    if args.background:
        if render.styled():
            print(render.object_panel("train", status, "submitted (running in background)"))
        else:
            print(json.dumps(status, indent=2))
        return 0
    if render.styled():
        print(render.submitted(run_id), file=sys.stderr)
    else:
        print(
            f"run {run_id} submitted; following logs (Ctrl-C detaches and the run keeps "
            f"billing, `{CLI_NAME} runs log {run_id} --follow` resumes, "
            f"`{CLI_NAME} runs cancel {run_id}` stops it)",
            file=sys.stderr,
        )
    return _follow_run(client, run_id)


def _log_follow_progress(status: dict | None, fallback_state: str) -> tuple[str, str]:
    """Return (authoritative state, compact progress) for the log-follow spinner."""
    status = status or {}
    state = str(status.get("state") or fallback_state or "unknown")
    parts = [state]
    heartbeat = status.get("last_heartbeat") if isinstance(status, dict) else None
    # retries rewind steps while state remains running, so surface the 0-based attempt identity.
    # prefer live `remote.attempt`; the heartbeat may belong to the superseded worker. fall back only
    # when `remote` is absent, not explicitly null: null marks the allocation window after teardown,
    # when the new attempt is unknown and the attached heartbeat is stale.
    from flash.providers._lifecycle.poll import _attempt_int

    remote = status.get("remote")
    remote_cleared = "remote" in status and remote is None
    attempt = _attempt_int(remote.get("attempt")) if isinstance(remote, dict) else None
    if attempt is None and not remote_cleared and isinstance(heartbeat, dict):
        attempt = _attempt_int(heartbeat.get("attempt"))
    if isinstance(heartbeat, dict):
        heartbeat_age_seconds = render._heartbeat_age_seconds(heartbeat.get("ts"))
        # stage and step come from the heartbeat, attempt from `remote` below. during the relaunch
        # window those are two different attempts, so printing them side by side reads as progress
        # the replacement worker has not made: `stage=sft_step step=455 attempt=1` attributes the
        # superseded worker's 455 steps to an attempt that just started from zero -- the exact
        # rewind the attempt counter was added to explain. mark the heartbeat-sourced fields as the
        # previous attempt's instead of dropping them: the run really did reach that step, and
        # suppressing it entirely would read as no progress at all. a cleared `remote` is the same
        # relaunch window, and the ping is just as superseded there: the worker that produced it has
        # already been torn down. `heartbeat_is_current_attempt` answers True because it cannot
        # prove otherwise from the identity alone, so the qualifier has to come from the clear
        # itself or `step=455` reads as the replacement's progress.
        stale_heartbeat = remote_cleared or not render.heartbeat_is_current_attempt(
            status, heartbeat
        )
        stage = heartbeat.get("stage")
        if stage:
            parts.append(f"stage={stage}")
            if state == "running":
                warmup = render.warmup_message(
                    stage,
                    heartbeat_age_seconds,
                    not stale_heartbeat,
                )
                if warmup:
                    parts.append(warmup)
        step = heartbeat.get("step")
        if step is not None:
            parts.append(f"step={step}")
        # live heartbeat age so a long quiet phase reads as "alive, throttled" not "frozen".
        # minute granularity: the non-TTY follow path prints a line whenever this string changes,
        # so a seconds-precision age would emit one line per poll.
        if heartbeat_age_seconds is not None:
            mins = int(heartbeat_age_seconds // 60)
            parts.append(f"hb={mins}m" if mins else "hb=<1m")
        if stale_heartbeat and (stage or step is not None or heartbeat_age_seconds is not None):
            # one marker for the whole heartbeat-sourced group rather than a suffix on each field:
            # the staleness is a property of the ping, not of any one value it carried.
            #
            # it TRAILS the group, and `attempt=` is appended after it, so the split is positional:
            # everything before the marker came from the superseded ping, everything after is live.
            # covering `hb=` matters as much as covering the step -- that age is the old worker's
            # ping too, so a fresh `hb=<1m` printed outside the marker read as the replacement
            # worker being alive when nothing had been heard from it at all, the opposite of what
            # the age is there to say.
            parts.append("(prev attempt)")
    if attempt:
        parts.append(f"attempt={attempt}")
    # what this run has committed to spend so far. while it is live that is the submit-time quote,
    # since the settled charge is not written until the terminal transition -- following a run for an
    # hour and never seeing a cost is how a user loses track of what it is costing. it sits next to
    # realized_cost below so the quote and the settled charge read as one pair.
    #
    # a settled zero is a real answer and prints: `runs list` and `runs status` both show $0.0000
    # there, and dropping it only here made the same terminal run read as costed in one surface and
    # uncosted in another. what stays suppressed is the pre-settlement zero -- state carries no cost
    # yet, and `cost=$0.0000` on a queued run states a charge nobody has computed.
    amount, is_estimate = render.run_cost(status)
    settled = str(status.get("state") or "") in render.SETTLED_COST_STATES
    if amount or is_estimate or settled:
        parts.append(f"cost={'~' if is_estimate else ''}${amount:.4f}")
    realized = status.get("realized_cost_usd")
    if realized is not None:
        if isinstance(realized, (int, float)):
            parts.append(f"realized_cost=${realized:.4f}")
        else:
            parts.append(f"realized_cost={realized}")
    return state, " ".join(parts)


_FOLLOW_METRIC_FIELDS = (
    ("reward", "reward"),
    ("reward_std", "reward_std"),
    ("grad_norm", "grad_norm"),
    ("kl", "kl"),
    ("entropy", "entropy"),
    ("frac_reward_zero_std", "frac_zero_std"),
    ("mean_completion_tokens", "comp_len"),
    ("truncation_rate", "trunc"),
    ("max_completion_tokens", "max_comp_tokens"),
)


def _log_follow_metric_rows(status: dict | None, seen_steps: set) -> list[str]:
    """Return unseen heartbeat-backed RL metric rows, deduplicated by attempt and optimizer step."""
    heartbeat = (status or {}).get("last_heartbeat")
    if not isinstance(heartbeat, dict):
        return []
    # during a retry, status.remote.attempt can already point at the replacement worker while
    # last_heartbeat still belongs to the prior attempt; don't render that stale attempt's rows
    if not render.heartbeat_is_current_attempt(status, heartbeat):
        return []
    metrics_last = heartbeat.get("metrics_last")
    if not isinstance(metrics_last, list):
        return []
    rows = []
    attempt = heartbeat.get("attempt")
    for metrics in metrics_last:
        if not isinstance(metrics, dict):
            continue
        step = metrics.get("step")
        if step is None:
            continue
        try:
            step_key = int(step)
        except (TypeError, ValueError):
            step_key = str(step)
        metric_key = (attempt, step_key)
        if metric_key in seen_steps:
            continue
        seen_steps.add(metric_key)
        parts = [f"step={step_key}"]
        for key, label in _FOLLOW_METRIC_FIELDS:
            value = metrics.get(key)
            if value is None:
                continue
            if isinstance(value, float):
                value = f"{value:.6g}"
            parts.append(f"{label}={value}")
        rows.append(" ".join(parts))
    return rows


def _poll_logs(client: ApiClient, run_id: str, interval: float) -> tuple[str, bool]:
    """Stream offset-paged logs until the run reaches a terminal state.

    Returns (terminal state, whether any log bytes were printed)."""
    offset = 0
    printed_any = False
    last_progress: str | None = None
    seen_metric_steps: set = set()
    spinner = _LogFollowSpinner(run_id)
    try:
        while True:
            page = client.get_logs(run_id, offset=offset)
            if page["logs"]:
                spinner.clear()
                print(page["logs"], end="", flush=True)
                printed_any = True
            offset = page["offset"]
            # The log file can lag worker heartbeat/status updates, so lifecycle/progress must come
            # from the run status endpoint. The log page's embedded state is only a fallback for
            # older servers or test doubles.
            status = client.get_run(run_id)
            state, progress = _log_follow_progress(status, str(page.get("state") or ""))
            metric_rows = _log_follow_metric_rows(status, seen_metric_steps)
            if metric_rows:
                spinner.clear()
                for row in metric_rows:
                    print(row, file=sys.stderr, flush=True)
            if state in _CLI_DONE_STATES:
                spinner.clear()
                return state, printed_any
            if not spinner.enabled and progress != last_progress:
                print(f"status: {progress}", file=sys.stderr, flush=True)
                last_progress = progress
            _sleep_with_spinner(interval, spinner, progress)
    finally:
        spinner.clear()


def _render_status(status: dict) -> str:
    """One rendering of a run status: themed panel on a TTY, indented JSON on the machine path."""
    return render.run_status(status) if render.styled() else json.dumps(status, indent=2)


def _print_detached_note(run_id: str) -> None:
    """Say what ctrl-c actually did: detached the stream, left the paid run running.

    The generic handler prints "aborted", which reads as "the run stopped". It did not, so the
    next thing a user does is re-run `flash train` and pay for a duplicate. Name the run and both
    commands that act on it.
    """
    detached = f"detached from {run_id}; the run is still going and still billing"
    resume = f"resume with `{CLI_NAME} runs log {run_id} --follow`"
    stop = f"stop it with `{CLI_NAME} runs cancel {run_id}`"
    if render.styled():
        print(render.warn(detached), file=sys.stderr)
        print(render.arrow(resume), file=sys.stderr)
        print(render.arrow(stop), file=sys.stderr)
    else:
        print(f"warning: {detached}", file=sys.stderr)
        print(f"note: {resume}", file=sys.stderr)
        print(f"note: {stop}", file=sys.stderr)


def _follow_run(client: ApiClient, run_id: str) -> int:
    """Poll logs until the run reaches a terminal state, then print the final status."""
    try:
        state, _ = _poll_logs(client, run_id, interval=2.0)
    except KeyboardInterrupt:
        _print_detached_note(run_id)
        return 130
    print(_render_status(client.get_run(run_id)))
    return 0 if state in _OK_STATES else 1


def _follow_status(client: ApiClient, run_id: str, interval: float = 2.0) -> int:
    """Poll run status until terminal, without replaying worker logs."""
    last_rendered: str | None = None
    try:
        while True:
            status = client.get_run(run_id)
            rendered = _render_status(status)
            if rendered != last_rendered:
                print(rendered)
                last_rendered = rendered
            state = str(status.get("state") or "")
            if state in _CLI_DONE_STATES:
                return 0 if state in _OK_STATES else 1
            time.sleep(interval)
    except KeyboardInterrupt:
        _print_detached_note(run_id)
        return 130


def _print_worker_output(client: ApiClient, run_id: str, *, printed_any: bool = False) -> bool:
    for name, text in (client.get_worker_output(run_id) or {}).items():
        if not text:
            continue
        sep = "\n" if printed_any else ""
        if render.styled():
            print(f"{sep}{render.log_section(name)}")
        else:
            print(f"{sep}----- {name} -----")
        print(text, end="" if text.endswith("\n") else "\n")
        printed_any = True
    return printed_any


def cmd_log(args) -> int:
    client = client_from_config()
    if getattr(args, "follow", False):
        try:
            state, printed_any = _poll_logs(client, args.run_id, interval=2.0)
        except KeyboardInterrupt:
            _print_detached_note(args.run_id)
            return 130
        _print_worker_output(client, args.run_id, printed_any=printed_any)
        return 0 if state in _OK_STATES else 1
    text = str(client.get_logs(args.run_id, offset=0).get("logs") or "")
    if text:
        print(text, end="" if text.endswith("\n") else "\n")
    _print_worker_output(client, args.run_id, printed_any=bool(text))
    return 0


def cmd_status(args) -> int:
    client = client_from_config()
    if getattr(args, "follow", False):
        return _follow_status(client, args.run_id)
    print(_render_status(client.get_run(args.run_id)))
    return 0


def cmd_runs(args) -> int:
    runs = client_from_config().list_runs()
    if not runs:
        if render.styled():
            print(render.empty("runs", "0 runs", "no runs yet — submit one with `flash train`"))
        else:
            print("no runs yet")
        return 0
    if render.styled():
        print(render.runs_table(runs))
        return 0
    print(f"{'RUN_ID':<32}  {'STATE':<11}  {'ALGO':<5}  {'COST($)':>8}  {'GPU':<22}  MODEL")
    for r in sorted(runs, key=lambda r: r.get("updated_at", 0), reverse=True):
        spec = r.get("spec") or {}
        model = spec.get("model", "")
        algorithm = str(spec.get("algorithm") or "-").upper()
        where = render.gpu_label(spec, r.get("remote") or {})
        amount, is_estimate = render.run_cost(r)
        cost = f"{'~' if is_estimate else ''}{amount:.4f}"
        print(
            f"{r['run_id']:<32}  {r['state']:<11}  {algorithm:<5}  {cost:>8}  {where:<22}  {model}"
        )
    return 0


def cmd_cancel(args) -> int:
    client = client_from_config()
    status = client.cancel_run(args.run_id)
    payload = {"run_id": args.run_id, "state": status["state"]}
    # A cancelled run is not necessarily worthless: every completed save interval already streamed
    # a deployable checkpoint, even though the run shows adapter_ref=null / cost=0. Surface the
    # surviving steps here so the run isn't discarded unseen. Best-effort: cancel never fails on it.
    checkpoints: list[dict] = []
    if payload["state"] == "cancelled":
        try:
            checkpoints = client.checkpoints(args.run_id)
        except Exception:
            checkpoints = []
    if render.styled():
        print(render.cancelled(payload))
    else:
        print(json.dumps(payload, indent=2))
    if checkpoints:
        # Best-effort hint (the cancel already succeeded), so never crash on a malformed checkpoint
        # shape: coerce steps defensively — a dict missing 'step' or carrying a non-int must not raise
        # a traceback here. Only surface the `step-N` deploy example when we recovered a real step.
        steps = []
        for c in checkpoints:
            try:
                steps.append(int(c["step"]))
            except (KeyError, TypeError, ValueError):
                continue
        # stderr in the plain path so the machine-readable stdout JSON stays untouched.
        out = sys.stdout if render.styled() else sys.stderr
        base = (
            f"{len(checkpoints)} deployable checkpoint(s) survive this cancel — list with "
            f"`flash runs checkpoint {args.run_id}`"
        )
        msg = (
            f"{base}, deploy one with `flash models deploy {args.run_id}/step-{max(steps)}`."
            if steps
            else f"{base}."
        )
        print(render.note(msg) if render.styled() else msg, file=out)
    return 0


def cmd_checkpoints(args) -> int:
    checkpoints = client_from_config().checkpoints(args.run_id)
    if not checkpoints:
        message = (
            f"no deployable checkpoints for {args.run_id} yet "
            "(RL/opd stream one per save interval; SFT-only runs have none)."
        )
        if render.styled():
            print(render.empty("checkpoints", "0 deployable", message))
        else:
            print(message, file=sys.stderr)
        return 0
    if render.styled():
        print(render.checkpoints_table(args.run_id, checkpoints))
        return 0
    from flash.schema import format_checkpoint_ref

    for c in checkpoints:
        # single-space, unpadded columns so a plain `grep "step N"` / awk split works; the ref is
        # the canonical short form, paste-able into train.init_from_adapter.
        print(f"step {c['step']} {format_checkpoint_ref(args.run_id, c['step'])}")
    print(
        f"\ndeploy one with `flash models deploy {args.run_id}/step-<STEP>`.",
        file=sys.stderr,
    )
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
        print(render.deployments_table(rows))
        return 0
    print(
        f"{'RUN ID':<30}  {'STEP':<6}  {'REVISION':<40}  {'STATE':<14}  "
        f"{'VERIFIED AT':<20}  {'OPENAI MODEL':<30}  DETAIL"
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
        detail = str(deployment.get("error") or deployment.get("detail") or "")[:160]
        print(
            f"{run_id:<30}  {step_text:<6}  {revision:<40}  {state:<14}  "
            f"{verified_text:<20}  {openai_model:<30}  {detail}"
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
        # flash/cli/commands/env/eval.py grades emptiness the same way.
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


# re-exported at the bottom rather than imported at the top: `deploy` resolves names back through
# this package, so a top import would be circular. the parser and the cli tests both address these
# handlers as attributes of `flash.cli.commands`, which is why they stay on it after the move.
from flash.cli.commands.deploy import (  # noqa: E402,F401
    _DEPLOY_FINAL_READ_FRACTION,
    _DEPLOY_FINAL_READ_SECONDS,
    _DEPLOY_POLL_SECONDS,
    _DEPLOY_ZERO_WAIT_READ_SECONDS,
    _DEPLOYMENT_BUSY_STATES,
    _DEPLOYMENT_READY_STATES,
    _PERMANENT_POLL_STATUSES,
    _await_deployment,
    _deployment_attempt_failed,
    _rollback_record,
    cmd_deploy,
    cmd_export,
    cmd_undeploy,
)

# re-exported at the bottom rather than imported at the top: `train_cost` resolves names back
# through this package, so a top import would be circular. `cmd_train` below calls these, and the
# estimate tests import `_warn_if_wandb_requested_without_key` from this package by name.
from flash.cli.commands.train_cost import (  # noqa: E402,F401
    _client_train_schema,
    _cmd_train_cost,
    _cmd_train_cost_offline,
    _cmd_train_cost_sft,
    _exact_sft_cost_rows,
    _legacy_train_key_rejection_detail,
    _print_exact_sft_cost,
    _print_train_schema_compatibility,
    _profile_charge,
    _raise_if_workload_profile_pending,
    _warn_if_wandb_requested_without_key,
)

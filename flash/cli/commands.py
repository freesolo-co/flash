"""CLI command handlers for the managed Flash service."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from flash import __version__
from flash._channel import CLI_NAME
from flash._logging import get_logger
from flash.catalog import public_model_rows
from flash.client import (
    ApiClient,
    ApiError,
    ClientError,
    client_from_config,
    save_credentials,
    verify_freesolo_key,
)
from flash.client.config import load_credentials
from flash.client.runtime_secrets import runtime_secrets_from_local_env
from flash.client.specs import spec_payload
from flash.cost.spec import runconfig_from_spec
from flash.runner import TERMINAL_STATES
from flash.schema import (
    ConfigError,
    spec_and_train_keys_from_file,
    spec_from_file,
    train_schema_metadata,
)

from . import render
from ._tty import TtyStatusLine

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
_LEGACY_TRAIN_UNKNOWN_KEYS_RE = re.compile(
    r"\A\[train\] unknown key\(s\): "
    r"(?P<keys>[A-Za-z_][A-Za-z0-9_]*(?:, [A-Za-z_][A-Za-z0-9_]*)*) "
    r"\(allowed: [A-Za-z_][A-Za-z0-9_]*(?:, [A-Za-z_][A-Za-z0-9_]*)*\)\Z"
)


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


def cmd_login(args) -> int:
    try:
        env_api_key = os.environ.get("FREESOLO_API_KEY")
        api_key = args.api_key or env_api_key
        if not api_key:
            raise ClientError(
                "no API key provided: pass `--api-key <key>` or set FREESOLO_API_KEY. "
                "Create or copy a key at https://freesolo.co/sign-in."
            )
        verify_freesolo_key(api_key, base_url=getattr(args, "freesolo_url", None))
    except ClientError as exc:
        if getattr(args, "debug", False):
            raise
        print(render.login_failed(str(exc)), file=sys.stderr)
        return 1
    api_url = args.api_url or load_credentials()[0]
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
    print(render.login_ok(_identity_or_none(api_key, api_url)))
    return 0


_IDENTITY_LOOKUP_TIMEOUT_S = 5.0


def _identity_or_none(api_key: str, api_url: str) -> dict | None:
    # Don't use client_from_config(): ambient FREESOLO_API_KEY would win and show wrong identity.
    try:
        return ApiClient(api_url, api_key, timeout=_IDENTITY_LOOKUP_TIMEOUT_S).me()
    except (ClientError, OSError, ValueError):
        return None


def cmd_whoami(args) -> int:
    print(render.whoami(client_from_config().me()))
    return 0


def cmd_projects_create(args) -> int:
    from flash.client import create_project

    _api_url, api_key = load_credentials()
    if not api_key:
        raise ClientError("not logged in. Run `flash login` before creating a project")
    result = create_project(args.name, getattr(args, "description", None), api_key)
    project_id = result["id"]
    if render.styled():
        print(render.project_created(project_id, str(args.name).strip()))
    else:
        print(project_id)
    return 0


def cmd_projects_list(args) -> int:
    from flash.client import list_projects

    _api_url, api_key = load_credentials()
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


def _cmd_train_cost(args) -> int:
    """`flash train --cost`: print the pre-flight USD cost for the config and exit (no submit).

    Catalog-only and deterministic. SFT cost never imports the environment; it requires a positive
    [train].max_examples row count instead of guessing or locally counting a dataset."""
    from flash.cost import estimate_cost
    from flash.lora_rank import preflight_train_context_within_serving

    spec = spec_from_file(
        args.config,
        run_id=None,
        overrides=args.overrides,
        extra_configs=args.extra_configs,
        project_required=True,
    )
    preflight_train_context_within_serving(spec)
    if spec.train.init_from_adapter:
        # --cost is offline/catalog-only and cannot read the source adapter, so the rank stays at the
        # local default. Warm starts train and are priced at the SOURCE adapter's authoritative rank
        # (resolved server-side at submit/dry-run), which can be higher — so this estimate may
        # under-quote. stderr keeps stdout clean for machine-readable callers.
        print(
            "warning: warm-start (train.init_from_adapter) cost uses the default LoRA rank; the "
            "source adapter's rank is authoritative and resolved at submit, so a higher-rank source "
            "may cost more than this estimate. Run `flash train --dry-run` for a source-rank quote.",
            file=sys.stderr,
        )
    estimate = estimate_cost(runconfig_from_spec(spec))
    if render.styled():
        print(render.cost_panel(estimate))
    else:
        print(estimate.breakdown())
    return 0


def _legacy_train_key_rejection_detail(
    exc: ApiError, authored_train_keys: frozenset[str]
) -> str | None:
    if exc.status != 400:
        return None
    match = _LEGACY_TRAIN_UNKNOWN_KEYS_RE.fullmatch(str(exc))
    if match is None:
        return None
    metadata = train_schema_metadata()
    unsupported = sorted(set(match.group("keys").split(", ")) & authored_train_keys & set(metadata))
    if not unsupported:
        return None
    declared = ", ".join(
        f"{key} (minimum released Flash version {metadata[key]})" for key in unsupported
    )
    return (
        f"{exc}. Unsupported authored [train] key(s): {declared}; "
        "client/server [train] schemas disagree"
    )


def _print_train_schema_compatibility(result: object) -> None:
    if not isinstance(result, dict):
        message = "client/server [train] schema compatibility is unverifiable (legacy server)"
    elif result.get("status") == "agreement":
        message = "client/server [train] schemas agree exactly"
    else:
        differences = []
        for label, key in (
            ("client-only keys", "client_only"),
            ("server-only keys", "server_only"),
        ):
            values = result.get(key)
            if isinstance(values, list) and values:
                differences.append(f"{label}: {', '.join(str(value) for value in values)}")
        metadata = result.get("introduced_in_differences")
        if isinstance(metadata, list) and metadata:
            rendered = ", ".join(
                f"{item['key']} (client {item['client']}, server {item['server']})"
                for item in metadata
                if isinstance(item, dict)
                and all(isinstance(item.get(key), str) for key in ("key", "client", "server"))
            )
            if rendered:
                differences.append(f"introduced_in differences: {rendered}")
        suffix = f"; {'; '.join(differences)}" if differences else ""
        message = f"client/server [train] schemas disagree{suffix}"
    text = f"train schema: {message}"
    print(render.note(text) if render.styled() else text, file=sys.stderr)


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
    client_train_schema = {
        "version": __version__,
        "fields": train_schema_metadata(),
        "authored_keys": sorted(authored_train_keys),
    }
    runtime_secrets = (
        runtime_secrets_from_local_env(args.config, keys=spec.environment.secrets) or None
    )
    if args.dry_run:
        # dry-run runs submit-time server preflights without importing user code, allocating a gpu,
        # or charging anything. a rejection surfaces as the server's error with exit status 1.
        try:
            status = client.create_run(
                payload,
                runtime_secrets=runtime_secrets,
                dry_run=True,
                client_train_schema=client_train_schema,
            )
        except ApiError as exc:
            detail = _legacy_train_key_rejection_detail(exc, authored_train_keys)
            if detail is None:
                raise
            raise ApiError(exc.status, detail) from exc
        compatibility = status.pop("train_schema_compatibility", None)
        _print_train_schema_compatibility(compatibility)
        print(
            "dry-run validated: config/schema, model+algorithm compatibility, lora rank, "
            "runtime-secret presence, warm-start source, serving context cap, and cost. it did NOT "
            "import or run your environment.py; dataset loading, start_episode/episode shapes, "
            "reward/scorer, worker imports, model load, and gpu/training are first exercised on the "
            "worker after cold-start.",
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
    status = client.create_run(
        payload,
        runtime_secrets=runtime_secrets,
        client_train_schema=client_train_schema,
    )
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
            f"run {run_id} submitted; following logs "
            f"(Ctrl-C detaches, `flash runs log {run_id} --follow` resumes)",
            file=sys.stderr,
        )
    return _follow_run(client, run_id)


def _log_follow_progress(status: dict | None, fallback_state: str) -> tuple[str, str]:
    """Return (authoritative state, compact progress) for the log-follow spinner."""
    status = status or {}
    state = str(status.get("state") or fallback_state or "unknown")
    parts = [state]
    heartbeat = status.get("last_heartbeat") if isinstance(status, dict) else None
    if isinstance(heartbeat, dict):
        heartbeat_age_seconds = render._heartbeat_age_seconds(heartbeat.get("ts"))
        stage = heartbeat.get("stage")
        if stage:
            parts.append(f"stage={stage}")
            if state == "running":
                warmup = render.warmup_message(
                    stage,
                    heartbeat_age_seconds,
                    render.heartbeat_is_current_attempt(status, heartbeat),
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


def _follow_run(client: ApiClient, run_id: str) -> int:
    """Poll logs until the run reaches a terminal state, then print the final status."""
    state, _ = _poll_logs(client, run_id, interval=2.0)
    print(_render_status(client.get_run(run_id)))
    return 0 if state in _OK_STATES else 1


def _follow_status(client: ApiClient, run_id: str, interval: float = 2.0) -> int:
    """Poll run status until terminal, without replaying worker logs."""
    last_rendered: str | None = None
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
        state, printed_any = _poll_logs(client, args.run_id, interval=2.0)
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
        print(
            f"{r['run_id']:<32}  {r['state']:<11}  {algorithm:<5}  "
            f"{r.get('cost_usd', 0.0):>8.4f}  {where:<22}  {model}"
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
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 and not first:
            break
        if not first:
            # hold back a slice of the budget for the read this sleep precedes, so a revision that
            # becomes ready early in a window no longer than one poll interval is still observed
            # rather than reported as queued. reserving a fixed slice (rather than sleeping a
            # fraction of the remainder) is what makes this terminate: a fraction leaves a positive
            # remainder forever, while this drives the remainder to the slice and then to zero.
            slice_seconds = min(_DEPLOY_POLL_SECONDS, remaining)
            if slice_seconds > _DEPLOY_FINAL_READ_SECONDS:
                slice_seconds -= _DEPLOY_FINAL_READ_SECONDS
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
                # the listing drops a run once its deployment is gone, so vanishing mid-wait is
                # terminal, not slow; continuing here would just burn the whole timeout.
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


def _deployment_attempt_failed(requested: dict, final: dict) -> bool:
    """True when the revision we asked for is not the one now being served.

    A failed redeploy does not leave a `failed` record. `mark_deployment_failed` restores the
    previous deployment verbatim and records the failure only in `last_deploy_error`, so the run
    ends up `ready` on the OLD adapter. Treating that as success is how
    `deploy --wait && evaluate` silently evaluates the previous checkpoint. Compare the attempt
    identity instead of trusting the state word.
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
    base_run_id, _step = parsed
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


def cmd_export(args) -> int:
    from flash.client.runtime_secrets import resolve_hf_token

    hf_token = resolve_hf_token(args.api_key)
    if not hf_token:
        raise ClientError(
            "no HuggingFace token: pass `--api-key <hf_...>`, or set HF_TOKEN "
            "(export it in your shell or put it in a local .env / .env.local)"
        )
    client = client_from_config()
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
    if render.styled():
        print(render.chat_label())
    wrote = False
    for chunk in client.chat_stream(
        chat_target,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ):
        print(chunk, end="", flush=True)
        wrote = True
    if wrote:
        print()
    return 0

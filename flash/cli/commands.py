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
        "Tip: GPU class selection is fully automatic — the submit-time allocator always picks the\n"
        "cheapest validated managed class that fits the model, so you don't pin a GPU type."
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
        print("local env sources (publish with `flash env push --name <name> <path>`):")
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
        # dry-run is a faithful server-side preview: it sends the same declared secrets and runs the
        # same config, warm-start, serving, and cost preflights as a real submit, but allocates no gpu
        # and charges nothing. a rejection surfaces as the server's error with exit status 1.
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
        spec.gpu.type,
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
            f"(Ctrl-C detaches, `flash log {run_id} --follow` resumes)",
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
        stage = heartbeat.get("stage")
        if stage:
            parts.append(f"stage={stage}")
        step = heartbeat.get("step")
        if step is not None:
            parts.append(f"step={step}")
        # live heartbeat age so a long quiet phase reads as "alive, throttled" not "frozen".
        # minute granularity: the non-TTY follow path prints a line whenever this string changes,
        # so a seconds-precision age would emit one line per poll.
        ts = heartbeat.get("ts")
        if isinstance(ts, (int, float)) and ts > 0:
            mins = int(max(0.0, time.time() - ts) // 60)
            parts.append(f"hb={mins}m" if mins else "hb=<1m")
    realized = status.get("realized_cost_usd")
    if realized is not None:
        if isinstance(realized, (int, float)):
            parts.append(f"realized_cost=${realized:.4f}")
        else:
            parts.append(f"realized_cost={realized}")
    return state, " ".join(parts)


def _poll_logs(client: ApiClient, run_id: str, interval: float) -> tuple[str, bool]:
    """Stream offset-paged logs until the run reaches a terminal state.

    Returns (terminal state, whether any log bytes were printed)."""
    offset = 0
    printed_any = False
    last_progress: str | None = None
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
            f"`flash checkpoints {args.run_id}`"
        )
        msg = (
            f"{base}, deploy one with `flash deploy {args.run_id}/step-{max(steps)}`."
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
        f"\ndeploy one with `flash deploy {args.run_id}/step-<STEP>`.",
        file=sys.stderr,
    )
    return 0


def cmd_deploy(args) -> int:
    from flash.schema import parse_checkpoint_ref

    # `flash deploy <run_id>/step-N` is the same checkpoint ref `flash checkpoints` prints.
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
    if render.styled():
        print(render.deployed(dep))
    else:
        print(json.dumps(dep, indent=2))
    # a dry run creates no deployment, so the billing / undeploy hint would be misleading.
    if dep.get("state") != "dry_run":
        openai_base = str(dep.get("openai_base_url") or "")
        note = (
            f"serving is billed per token only; use `flash undeploy {base_run_id}` "
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
                f"deployment failed: {detail}; run `flash deployments` for details and "
                f"retry `flash deploy {args.run_id}` after fixing the error."
            )
        else:
            status_note = (
                f"deployment state is {state!r}; run `flash deployments` to check progress "
                "and use `flash chat` once it is ready."
            )
        print(
            render.arrow(status_note) if render.styled() else f"note: {status_note}",
            file=sys.stderr,
        )
    return 1 if dep.get("state") == "failed" else 0


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
        f"{'VERIFIED AT':<18}  {'OPENAI MODEL':<30}  DETAIL"
    )
    for row in rows:
        deployment = row.get("deployment") or {}
        run_id = str(deployment.get("run_id") or row.get("run_id") or "")
        step = deployment.get("checkpoint_step")
        step_text = "final" if step is None else str(step)
        verified_at = deployment.get("verified_at")
        verified_text = "-" if verified_at is None else str(verified_at)
        revision = str(deployment.get("adapter_revision") or "-")
        state = str(deployment.get("state") or "-")
        openai_model = str(deployment.get("openai_model") or run_id)
        detail = str(deployment.get("error") or deployment.get("detail") or "")[:160]
        print(
            f"{run_id:<30}  {step_text:<6}  {revision:<40}  {state:<14}  "
            f"{verified_text:<18}  {openai_model:<30}  {detail}"
        )
    return 0


def cmd_chat(args) -> int:
    from flash.schema import parse_adapter_revision, parse_checkpoint_ref

    revision = parse_adapter_revision(args.run_id)
    parsed = parse_checkpoint_ref(args.run_id) if revision is None else None
    if revision is None and parsed is None:
        print(
            f"invalid chat target {args.run_id!r} "
            "(expected a bare <run_id> or full immutable adapter revision)",
            file=sys.stderr,
        )
        return 1
    if revision is None and parsed[1] is not None:
        print(
            "RUN_ID/step-N is not a valid chat target because it would route through the mutable "
            "run alias; use the full immutable adapter revision returned by `flash deployments`",
            file=sys.stderr,
        )
        return 1
    chat_target = args.run_id if revision is not None else parsed[0]
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

"""`flash serve`: stand up a serving backend you own, on your own Modal account.

`flash models deploy` / `chat` / `undeploy` talk to a serving backend over HTTP. On the hosted
plane that backend is Freesolo's; a self-hosted plane has none, so those commands dead-end. These
commands generate one from the catalog's validated serving config, deploy it to the user's Modal
account, and print the environment variable that connects the two.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from flash._internal.channel import CLI_NAME
from flash.cli.ui import render
from flash.core.catalog import MODELS, ModelInfo, get_model
from flash.serve.backend.generate import (
    DEFAULT_SCALEDOWN_WINDOW,
    app_name_for,
    gpu_named,
    write_app,
)
from flash.serve.backend.gpus import (
    MODAL_GPUS,
    Fit,
    cheapest_fitting,
    default_gpu,
    estimate_fit,
    recommend,
)
from flash.serve.contract import (
    REQUIRED_SERVING_CAPABILITIES,
    ServingHealthError,
    parse_serving_health,
)
from flash.serve.probe import healthz_with_retry, probe_serving_key, redacted_error, request_json

DEFAULT_APP_FILE = "flash_serving_app.py"
SECRET_NAME = "flash-serving"
# The console script that runs the control plane (pyproject `[project.scripts]`). Deploy, chat and
# undeploy are its routes, so it -- not this CLI -- is what has to see the serving variables.
SERVER_NAME = "flash-server"
# Modal prints the deployed web endpoint on stdout; this is the prefix we look for.
_URL_MARKER = "https://"


def _err(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _require_model(model_id: str) -> ModelInfo:
    try:
        return get_model(model_id)
    except (KeyError, ValueError) as exc:
        known = ", ".join(sorted(MODELS))
        raise ValueError(f"unknown model {model_id!r}. supported: {known}") from exc


def _gpus_tip(info: ModelInfo, rows: list[Fit], model_id: str) -> str:
    """What the table cannot show in a column, and the honest caveats.

    The estimates are arithmetic over the catalog's model geometry, not measurements, and saying so
    is the difference between a useful guide and a number someone budgets against.
    """
    marked = next((fit for fit in rows if fit.is_catalog_default), None)
    lines = []
    if marked is not None:
        lines.append(
            f"* {marked.gpu.name} is what Freesolo runs this model on in production, validated on "
            "real hardware. Prefer it unless you have a reason not to."
        )
    non_native = [fit.gpu.name for fit in rows if not fit.fp8_native and fit.fits]
    if non_native:
        lines.append(
            f"{', '.join(non_native)} lack native fp8 tensor cores (compute capability < 8.9), so "
            "they serve fp8 through a slower weight-only path."
        )
    lines.append(
        "FITS/SPARE are ESTIMATES computed from model geometry (weights + KV cache + the LoRA "
        "buffers vLLM pre-allocates), measured against the fraction of the card vLLM claims. "
        "SPEED is a relative band from memory bandwidth, not a measured tokens/sec. $/HR is "
        "approximate and Modal bills per second while a container runs, so an idle app costs "
        "nothing."
    )
    lines.append(f"Generate an app for one: {CLI_NAME} serve setup --model {model_id}")
    return "\n".join(lines)


def cmd_serve_gpus(args) -> int:
    """Show which Modal GPUs can serve a model, with fit, speed, and price."""
    info = _require_model(args.model)
    rows = recommend(info, context_len=getattr(args, "context_len", 0) or 0)
    tip = _gpus_tip(info, rows, info.id)
    if render.styled():
        print(render.serving_gpus_table(rows, tip))
        return 0
    print(f"{'gpu':<12}{'vram':>6}{'fits':>8}{'spare':>8}{'speed':>10}{'$/hr':>8}")
    for fit in rows:
        spare = f"{fit.free_gb:.0f}G" if fit.fits else "-"
        print(
            f"{fit.gpu.name:<12}{fit.gpu.vram_gb:>5}G{fit.headroom:>8}{spare:>8}"
            f"{fit.speed:>10}{fit.gpu.usd_hr:>8.2f}"
        )
    print(f"\n{tip}")
    return 0


def _modal_cli() -> str | None:
    return shutil.which("modal")


def _modal_is_authenticated() -> bool:
    """Whether the local modal CLI has credentials.

    Checked before generating anything: finding out at `modal deploy` time means the user has
    already been told the app was written and is about to be deployed.
    """
    try:
        done = subprocess.run(
            ["modal", "profile", "current"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _setup_instructions() -> str:
    return (
        f"Modal is not set up yet. Install it and authenticate:\n"
        f"  pip install 'freesolo-flash[serve-modal]'\n"
        f"  modal setup\n"
        f"Then create the secret. The app pulls weights with HF_TOKEN, and authenticates "
        f"callers with FLASH_SERVING_KEY -- a Modal URL is public, so without a key anyone "
        f"who finds it can load adapters and spend your GPU budget.\n"
        f"Keep the key: flash has to send the same value back, so generate it into a variable "
        f"rather than inline, or every request gets a 401.\n"
        f"  export FREESOLO_INTERNAL_KEY=$(python -c "
        f"'import secrets; print(secrets.token_urlsafe(32))')\n"
        f"  modal secret create {SECRET_NAME} HF_TOKEN=hf_... "
        f'FLASH_SERVING_KEY="$FREESOLO_INTERNAL_KEY"\n'
        f"Then re-run: {CLI_NAME} serve setup --model <model>"
    )


def _confirm(prompt: str) -> bool:
    try:
        answer = input(render.warn(prompt) if render.styled() else prompt)
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def cmd_serve_setup(args) -> int:
    """Generate a Modal serving app for a model and, with consent, deploy it."""
    info = _require_model(args.model)

    gpu = default_gpu(info)
    if getattr(args, "gpu", None):
        gpu = gpu_named(args.gpu)
        if gpu is None:
            offered = ", ".join(card.name for card in MODAL_GPUS)
            return _err(f"unknown Modal GPU {args.gpu!r}. choose one of: {offered}")
        # An explicit card still has to hold the model. Without this the same estimate that
        # `serve gpus` shows as "no" is skipped on the one path that spends money: setup writes
        # the catalog's fixed config, deploys, pulls the weights, and the engine OOMs on a cold
        # start the user paid for. Refused with the number, not just a verdict.
        fit = estimate_fit(info, gpu)
        if not fit.fits:
            return _err(
                f"{gpu.name} cannot serve {info.id}: needs about {fit.total_gb:.0f} GB but "
                f"{gpu.name} offers about {fit.budget_gb:.0f} GB usable. "
                f"run `{CLI_NAME} serve gpus --model {info.id}` to see what fits."
            )
    if gpu is None:
        # no validated card for this model; fall back to the cheapest that fits and say so.
        fit = cheapest_fitting(recommend(info))
        if fit is None:
            return _err(f"no Modal GPU is large enough to serve {info.id}")
        gpu = fit.gpu
        print(f"note: {info.id} has no production-validated card; using the cheapest that fits.")

    destination = Path(getattr(args, "output", None) or DEFAULT_APP_FILE).resolve()
    dry_run = bool(getattr(args, "dry_run", False))
    # Before writing anything. The instructions below end with "re-run this command", and a file
    # written on the way out is exactly what makes that re-run fail with FileExistsError. `--dry-run`
    # deploys nothing, so it does not need Modal at all.
    if not dry_run and (_modal_cli() is None or not _modal_is_authenticated()):
        print(_setup_instructions(), file=sys.stderr)
        return 1

    try:
        write_app(
            info,
            destination,
            gpu=gpu,
            # `is None`, not `or`: 0 is a meaningful value (stop the container as soon as it goes
            # idle) and `or` would silently rewrite it to the 300s default.
            scaledown_window=(
                DEFAULT_SCALEDOWN_WINDOW
                if getattr(args, "scaledown_window", None) is None
                else args.scaledown_window
            ),
            secret_name=SECRET_NAME,
            overwrite=bool(getattr(args, "force", False)),
        )
    except FileExistsError as exc:
        return _err(str(exc))
    except ValueError as exc:
        return _err(str(exc))
    # Every other filesystem refusal: --output naming an existing directory, an unwritable parent,
    # a read-only volume, no space. These reach the top-level handler otherwise, which re-raises on
    # the non-styled path, so an ordinary permissions problem prints a traceback instead of the
    # `error:` line every other failure in this command produces. Listed after FileExistsError,
    # which is an OSError subclass and keeps its own more specific message.
    except OSError as exc:
        return _err(f"could not write {destination}: {exc}")
    print(f"wrote {destination}")
    print(f"  model  {info.id}")
    print(f"  gpu    {gpu.name}  (~${gpu.usd_hr:.2f}/hr while serving, $0 idle)")

    if dry_run:
        # Quoted, because this line is meant to be COPIED into a shell. `--output` accepts any
        # writable path, and one containing a space splits into two arguments when pasted -- so the
        # instruction for deploying the file we just wrote fails to deploy it.
        print(
            f"\ndry run: not deploying. deploy it yourself with:\n"
            f"  modal deploy {shlex.quote(str(destination))}"
        )
        return 0

    if not getattr(args, "yes", False):
        prompt = (
            f"deploy {destination.name} to Modal now? "
            f"this starts a {gpu.name} container on first use [y/N] "
        )
        if not _confirm(prompt):
            print(
                f"not deployed. run `modal deploy {shlex.quote(str(destination))}` when ready.",
                file=sys.stderr,
            )
            return 1

    return _deploy(destination)


def _deploy(app_file: Path) -> int:
    """Run `modal deploy` and surface the URL flash needs."""
    try:
        done = subprocess.run(
            ["modal", "deploy", str(app_file)],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _err(f"could not run modal deploy: {exc}")
    output = f"{done.stdout}\n{done.stderr}"
    print(done.stdout, end="")
    if done.returncode != 0:
        print(done.stderr, end="", file=sys.stderr)
        return _err("modal deploy failed")
    url = _deployed_url(output)
    if not url:
        print(
            "\ndeployed, but the web endpoint URL was not found in modal's output. "
            "find it with `modal app list` and set FREESOLO_SERVING_URL yourself, in the "
            f"environment of the {SERVER_NAME} process (see below).",
            file=sys.stderr,
        )
        return 0
    print(f"\ndeployed at {url}")
    print(_control_plane_instructions(url))
    _warn_if_unauthenticated(url, app_file)
    return 0


def _control_plane_instructions(url: str) -> str:
    """Where the serving variables have to be set.

    `models deploy`/`chat`/`undeploy` are CONTROL PLANE operations: the CLI calls the server, and
    the server's routes are what read FREESOLO_SERVING_URL and contact the backend. Exporting it
    in the shell that ran `serve setup` reaches the CLI and not the server, so following that
    instruction leaves every deploy failing on an unset serving URL -- with a setup transcript
    that looked complete.
    """
    return (
        f"Set these in the environment of your {SERVER_NAME} process -- not just this shell. "
        f"{SERVER_NAME} reads its process environment, so an already-running server needs a "
        f"restart to pick them up:\n"
        f"  export FREESOLO_SERVING_URL={url}\n"
        f"  export FREESOLO_INTERNAL_KEY=<the FLASH_SERVING_KEY you put in the modal secret>\n"
        f"  {SERVER_NAME} --host 0.0.0.0 --port 8080   # restart it\n"
        f"then: {CLI_NAME} models deploy <run-id> && {CLI_NAME} models chat <run-id> -m 'hi'"
    )


def _warn_if_unauthenticated(url: str, app_file: Path | None = None) -> None:
    """Say so if the deployed app accepts unauthenticated writes.

    The app only enforces a key when FLASH_SERVING_KEY is in its secret, and the URL is public.
    Ask the app itself rather than guessing from local env: the secret lives in Modal, so the
    deployed container is the only thing that knows whether a key is set. Silence on an
    unreadable or older /healthz is deliberate -- warning without evidence trains users to
    ignore the warning.

    ``app_file`` is the file that was actually deployed. The redeploy step names it rather than
    the default: under ``--output`` a hardcoded name would deploy an unrelated file or fail, and
    the public keyless endpoint would stay up after the user did exactly what they were told.
    """
    payload = healthz_with_retry(url)
    if payload is None:
        return
    try:
        health = parse_serving_health(payload)
    except ServingHealthError:
        return
    if health.requires_key is not False:
        return
    print(
        "\nwarning: this app has no FLASH_SERVING_KEY, so anyone with the URL can register "
        "adapters and spend your GPU budget. set one and redeploy:\n"
        "  export FREESOLO_INTERNAL_KEY=$(python -c "
        "'import secrets; print(secrets.token_urlsafe(32))')\n"
        f"  modal secret create {SECRET_NAME} HF_TOKEN=hf_... "
        'FLASH_SERVING_KEY="$FREESOLO_INTERNAL_KEY"\n'
        f"  modal deploy {shlex.quote(str(app_file if app_file is not None else DEFAULT_APP_FILE))}",
        file=sys.stderr,
    )


def _deployed_url(output: str) -> str:
    """The web endpoint URL from modal's deploy output.

    Modal prints several URLs (the dashboard app page among them); the serving endpoint is the
    one on modal.run, so match on that rather than taking the first link.
    """
    for token in output.replace(",", " ").split():
        candidate = token.strip().rstrip(".)>\"'")
        if candidate.startswith(_URL_MARKER) and ".modal.run" in candidate:
            return candidate
    return ""


def cmd_serve_status(args) -> int:
    """Check the configured serving backend and report what it supports."""
    from flash.serve.errors import ServingError
    from flash.serve.urls import displayable_url, internal_key_header, serving_base_url

    try:
        base = serving_base_url()
    except ServingError as exc:
        return _err(str(exc))
    shown = displayable_url(base)
    headers = internal_key_header()
    try:
        result = request_json(base, headers)
    except Exception as exc:
        return _err(
            f"serving backend at {shown} did not answer /healthz: {redacted_error(exc, base)}"
        )
    if result.status_code != 200:
        detail = result.error or f"HTTP {result.status_code}"
        return _err(
            f"serving backend at {shown} did not answer /healthz: {redacted_error(detail, base)}"
        )
    try:
        health = parse_serving_health(result.payload)
    except ServingHealthError as exc:
        if exc.code == "non_object":
            return _err(f"serving backend at {shown} returned a non-object /healthz payload")
        return _err(f"serving backend at {shown} did not return capabilities as a list of strings")

    print(f"serving:      {shown}")
    print(f"base models:  {', '.join(health.base_models) or '-'}")
    print(f"capabilities: {', '.join(health.capabilities) or '-'}")
    missing = sorted(REQUIRED_SERVING_CAPABILITIES - set(health.capabilities))
    if missing:
        print(f"\nmissing required capabilities: {', '.join(missing)}", file=sys.stderr)
        return 1
    if health.ok is False:
        print(
            "\nthe backend reports itself unhealthy (ok: false). its capabilities are right, so "
            "this is a runtime problem -- check the app's logs before deploying.",
            file=sys.stderr,
        )
        return 1
    if health.requires_key is not False:
        return _verify_serving_key(base, shown, headers)
    print(f"\nready. deploy a run with: {CLI_NAME} models deploy <run-id>")
    return 0


def _verify_serving_key(base: str, shown: str, headers: dict[str, str]) -> int:
    """Exercise the configured key and explicitly dispatch the probe's HTTP status."""
    try:
        result = probe_serving_key(base, headers)
    except Exception as exc:
        print(
            f"\ncould not verify the serving key against {shown}: {redacted_error(exc, base)}",
            file=sys.stderr,
        )
        return 1

    code = result.status_code
    if code in (401, 403):
        print(
            f"\nthe backend at {shown} rejected the serving key ({code}). set "
            f"FREESOLO_INTERNAL_KEY to the value of the app's FLASH_SERVING_KEY secret; "
            f"deploys will fail until it matches.",
            file=sys.stderr,
        )
        return 1
    if code >= 500:
        print(
            f"\nthe backend at {shown} returned {code} when the serving key was checked, so "
            f"the key could not be verified. check the app's logs before deploying.",
            file=sys.stderr,
        )
        return 1
    if code == 404:
        print(f"\nready. deploy a run with: {CLI_NAME} models deploy <run-id>")
        return 0
    if 400 <= code < 500:
        print(
            f"\nthe backend at {shown} answered {code} for a read-back of an unknown adapter "
            f"id, where the contract requires 404. `{CLI_NAME} models deploy` polls this route "
            f"and fails on any non-404 4xx, so deploys will not work against this backend.",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nthe backend at {shown} answered {code} with a record for an adapter id that was "
        f"never registered, where the contract requires 404. `{CLI_NAME} models deploy` "
        f"cross-checks the record it reads back, so a backend that fabricates records fails "
        f"every deploy.",
        file=sys.stderr,
    )
    return 1


def cmd_serve_teardown(args) -> int:
    """Stop the Modal app so it holds no resources."""
    info = _require_model(args.model)
    app = app_name_for(info.id)
    if _modal_cli() is None:
        return _err(f"modal CLI not found. stop the app yourself with: modal app stop {app}")
    if not getattr(args, "yes", False) and not _confirm(
        f"stop Modal app {app}? deployed adapters stop serving [y/N] "
    ):
        print("aborted; app left running", file=sys.stderr)
        return 1
    try:
        done = subprocess.run(
            ["modal", "app", "stop", app], capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _err(f"could not run modal app stop: {exc}")
    print(done.stdout, end="")
    if done.returncode != 0:
        print(done.stderr, end="", file=sys.stderr)
        return _err(f"could not stop {app}")
    print(f"stopped {app}")
    return 0

"""Generate, inspect, and stop a self-hosted Modal serving backend."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from flash._internal.channel import CLI_NAME
from flash.cli.ui import render
from flash.core.catalog import MODELS, ModelInfo, get_model
from flash.serve.backend.generate import DEFAULT_SCALEDOWN_WINDOW, app_name_for, write_app
from flash.serve.contract import (
    REQUIRED_SERVING_CAPABILITIES,
    ServingHealthError,
    parse_serving_health,
)
from flash.serve.probe import probe_serving_key, redacted_error, request_json

DEFAULT_APP_FILE = "flash_serving_app.py"
SECRET_NAME = "flash-serving"
SERVER_NAME = "flash-server"
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


def _modal_cli() -> str | None:
    return shutil.which("modal")


def _modal_is_authenticated() -> bool:
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
    info = _require_model(args.model)
    serving = getattr(info, "serving", None)
    if serving is None or not isinstance(serving.gpu, str) or not serving.gpu.strip():
        return _err(f"{info.id} has no validated serving GPU in the flash catalog")
    gpu = serving.gpu
    destination = Path(getattr(args, "output", None) or DEFAULT_APP_FILE).resolve()
    dry_run = bool(getattr(args, "dry_run", False))
    if not dry_run and (_modal_cli() is None or not _modal_is_authenticated()):
        print(_setup_instructions(), file=sys.stderr)
        return 1
    try:
        write_app(
            info,
            destination,
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
    except OSError as exc:
        return _err(f"could not write {destination}: {exc}")
    print(f"wrote {destination}")
    print(f"  model  {info.id}")
    print(f"  gpu    {gpu}")
    if dry_run:
        print(
            f"\ndry run: not deploying. deploy it yourself with:\n"
            f"  modal deploy {shlex.quote(str(destination))}"
        )
        return 0
    if not getattr(args, "yes", False):
        prompt = (
            f"deploy {destination.name} to Modal now? "
            f"this starts a {gpu} container on first use [y/N] "
        )
        if not _confirm(prompt):
            print(
                f"not deployed. run `modal deploy {shlex.quote(str(destination))}` when ready.",
                file=sys.stderr,
            )
            return 1
    return _deploy(destination)


def _deploy(app_file: Path) -> int:
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
    return 0


def _control_plane_instructions(url: str) -> str:
    return (
        f"Set these in the environment of your {SERVER_NAME} process -- not just this shell. "
        f"{SERVER_NAME} reads its process environment, so an already-running server needs a "
        f"restart to pick them up:\n"
        f"  export FREESOLO_SERVING_URL={url}\n"
        f"  export FREESOLO_INTERNAL_KEY=<the FLASH_SERVING_KEY you put in the modal secret>\n"
        f"  {SERVER_NAME} --host 0.0.0.0 --port 8080   # restart it\n"
        f"then: {CLI_NAME} models deploy <run-id> && {CLI_NAME} models chat <run-id> -m 'hi'"
    )


def _deployed_url(output: str) -> str:
    for token in output.replace(",", " ").split():
        candidate = token.strip().rstrip(".)>\"'")
        if candidate.startswith(_URL_MARKER) and ".modal.run" in candidate:
            return candidate
    return ""


def cmd_serve_status(args) -> int:
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

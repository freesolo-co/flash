"""`flash serve`: stand up a serving backend you own, on your own Modal account.

`flash models deploy` / `chat` / `undeploy` talk to a serving backend over HTTP. On the hosted
plane that backend is Freesolo's; a self-hosted plane has none, so those commands dead-end. These
commands generate one from the catalog's validated serving config, deploy it to the user's Modal
account, and print the environment variable that connects the two.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from flash._internal.channel import CLI_NAME
from flash.cli.ui import render
from flash.core.catalog import MODELS, get_model
from flash.serve.backend.generate import (
    DEFAULT_SCALEDOWN_WINDOW,
    app_name_for,
    gpu_named,
    write_app,
)
from flash.serve.backend.gpus import MODAL_GPUS, cheapest_fitting, default_gpu, recommend

DEFAULT_APP_FILE = "flash_serving_app.py"
SECRET_NAME = "flash-serving"
# Modal prints the deployed web endpoint on stdout; this is the prefix we look for.
_URL_MARKER = "https://"


def _err(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _model_or_error(model_id: str):
    try:
        return get_model(model_id), None
    except (KeyError, ValueError):
        known = ", ".join(sorted(MODELS))
        return None, _err(f"unknown model {model_id!r}. supported: {known}")


def _fit_rows(info, context_len: int = 0) -> list[dict]:
    return [
        {
            "gpu": fit.gpu.name,
            "vram_gb": fit.gpu.vram_gb,
            "headroom": fit.headroom,
            "free_gb": fit.free_gb,
            "speed": fit.speed,
            "usd_hr": fit.gpu.usd_hr,
            "default": fit.is_catalog_default,
            "fp8_native": fit.fp8_native,
        }
        for fit in recommend(info, context_len=context_len)
    ]


def _gpus_tip(info, rows: list[dict], model_id: str) -> str:
    """What the table cannot show in a column, and the honest caveats.

    The estimates are arithmetic over the catalog's model geometry, not measurements, and saying so
    is the difference between a useful guide and a number someone budgets against.
    """
    marked = next((row for row in rows if row["default"]), None)
    lines = []
    if marked is not None:
        lines.append(
            f"* {marked['gpu']} is what Freesolo runs this model on in production, validated on "
            "real hardware. Prefer it unless you have a reason not to."
        )
    non_native = [row["gpu"] for row in rows if not row["fp8_native"] and row["headroom"] != "no"]
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
    info, error = _model_or_error(args.model)
    if info is None:
        return error
    rows = _fit_rows(info, context_len=getattr(args, "context_len", 0) or 0)
    tip = _gpus_tip(info, rows, info.id)
    if render.styled():
        print(render.serving_gpus_table(rows, tip))
        return 0
    print(f"{'gpu':<12}{'vram':>6}{'fits':>8}{'spare':>8}{'speed':>10}{'$/hr':>8}")
    for row in rows:
        spare = f"{row['free_gb']:.0f}G" if row["headroom"] != "no" else "-"
        print(
            f"{row['gpu']:<12}{row['vram_gb']:>5}G{row['headroom']:>8}{spare:>8}"
            f"{row['speed']:>10}{row['usd_hr']:>8.2f}"
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
        f"who finds it can load adapters and spend your GPU budget:\n"
        f"  modal secret create {SECRET_NAME} HF_TOKEN=hf_... "
        f"FLASH_SERVING_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')\n"
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
    info, error = _model_or_error(args.model)
    if info is None:
        return error

    gpu = default_gpu(info)
    if getattr(args, "gpu", None):
        gpu = gpu_named(args.gpu)
        if gpu is None:
            offered = ", ".join(card.name for card in MODAL_GPUS)
            return _err(f"unknown Modal GPU {args.gpu!r}. choose one of: {offered}")
    if gpu is None:
        # no validated card for this model; fall back to the cheapest that fits and say so.
        fit = cheapest_fitting(recommend(info))
        if fit is None:
            return _err(f"no Modal GPU is large enough to serve {info.id}")
        gpu = fit.gpu
        print(f"note: {info.id} has no production-validated card; using the cheapest that fits.")

    destination = Path(getattr(args, "output", None) or DEFAULT_APP_FILE).resolve()
    try:
        write_app(
            info,
            destination,
            gpu=gpu,
            scaledown_window=getattr(args, "scaledown_window", None) or DEFAULT_SCALEDOWN_WINDOW,
            secret_name=SECRET_NAME,
            overwrite=bool(getattr(args, "force", False)),
        )
    except FileExistsError as exc:
        return _err(str(exc))
    except ValueError as exc:
        return _err(str(exc))
    print(f"wrote {destination}")
    print(f"  model  {info.id}")
    print(f"  gpu    {gpu.name}  (~${gpu.usd_hr:.2f}/hr while serving, $0 idle)")

    if getattr(args, "dry_run", False):
        print(f"\ndry run: not deploying. deploy it yourself with:\n  modal deploy {destination}")
        return 0

    if _modal_cli() is None or not _modal_is_authenticated():
        print(f"\n{_setup_instructions()}", file=sys.stderr)
        return 1

    if not getattr(args, "yes", False):
        prompt = (
            f"deploy {destination.name} to Modal now? "
            f"this starts a {gpu.name} container on first use [y/N] "
        )
        if not _confirm(prompt):
            print(f"not deployed. run `modal deploy {destination}` when ready.", file=sys.stderr)
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
            "find it with `modal app list` and export FREESOLO_SERVING_URL yourself.",
            file=sys.stderr,
        )
        return 0
    print(f"\ndeployed. point flash at it:\n  export FREESOLO_SERVING_URL={url}")
    print(f"then: {CLI_NAME} models deploy <run-id> && {CLI_NAME} models chat <run-id> -m 'hi'")
    _warn_if_unauthenticated(url)
    return 0


def _healthz(url: str) -> dict | None:
    """The app's own /healthz, or None if it cannot be read. Never raises: this is advisory."""
    try:
        import urllib.request

        with urllib.request.urlopen(f"{url}/healthz", timeout=15) as response:
            payload = json.load(response)
    # broad on purpose: a warning must never fail a deploy that already succeeded
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _warn_if_unauthenticated(url: str) -> None:
    """Say so if the deployed app accepts unauthenticated writes.

    The app only enforces a key when FLASH_SERVING_KEY is in its secret, and the URL is public.
    Ask the app itself rather than guessing from local env: the secret lives in Modal, so the
    deployed container is the only thing that knows whether a key is set. Silence on an
    unreadable or older /healthz is deliberate -- warning without evidence trains users to
    ignore the warning.
    """
    payload = _healthz(url)
    if payload is None or payload.get("requires_key") is not False:
        return
    print(
        "\nwarning: this app has no FLASH_SERVING_KEY, so anyone with the URL can register "
        "adapters and spend your GPU budget. set one and redeploy:\n"
        f"  modal secret create {SECRET_NAME} HF_TOKEN=hf_... "
        "FLASH_SERVING_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')\n"
        "  then export FREESOLO_INTERNAL_KEY to the same value",
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
    from flash.serve.deploy import ServingError, serving_base_url

    try:
        base = serving_base_url()
    except ServingError as exc:
        return _err(str(exc))
    try:
        import httpx
    except ImportError:
        return _err("httpx is required. install with: pip install 'freesolo-flash[serve-modal]'")
    try:
        response = httpx.get(f"{base}/healthz", timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # any transport or decode failure is the same answer to the user
        return _err(f"serving backend at {base} did not answer /healthz: {exc}")

    capabilities = payload.get("capabilities") or []
    print(f"serving:      {base}")
    print(f"base models:  {', '.join(payload.get('base_models') or []) or '-'}")
    print(f"capabilities: {', '.join(capabilities) or '-'}")
    required = {"immutable_adapter_revisions", "alias_compare_and_swap"}
    missing = sorted(required - set(capabilities))
    if missing:
        # deploy would fail on this, so say it here rather than at deploy time.
        print(f"\nmissing required capabilities: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"\nready. deploy a run with: {CLI_NAME} models deploy <run-id>")
    return 0


def cmd_serve_teardown(args) -> int:
    """Stop the Modal app so it holds no resources."""
    info, error = _model_or_error(args.model)
    if info is None:
        return error
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

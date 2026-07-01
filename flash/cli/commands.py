"""CLI command handlers for the managed Flash service."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from flash import __version__
from flash._channel import CLI_NAME
from flash._logging import get_logger
from flash.catalog import public_model_rows
from flash.client import (
    ApiClient,
    ClientError,
    client_from_config,
    save_credentials,
    verify_freesolo_key,
)
from flash.client.config import load_credentials
from flash.client.runtime_secrets import runtime_secrets_from_local_env
from flash.client.specs import spec_payload
from flash.cost.spec import runconfig_from_spec
from flash.runner import TERMINAL_STATES, new_run_id
from flash.schema import ConfigError, spec_from_file

from . import render
from .training_doc import TRAINING_MD

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


class _LogFollowSpinner:
    def __init__(self, run_id: str):
        self._run_id = run_id
        self._frame = 0
        self._last_len = 0
        self._active = False
        self._enabled = sys.stderr.isatty()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def render(self, state: str) -> None:
        if not self._enabled:
            return
        frame = _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)]
        self._frame += 1
        message = f"{frame} following logs for {self._run_id} ({state})"
        padding = " " * max(0, self._last_len - len(message))
        sys.stderr.write(f"\r{message}{padding}")
        sys.stderr.flush()
        self._last_len = len(message)
        self._active = True

    def clear(self) -> None:
        if not (self._enabled and self._active):
            return
        sys.stderr.write(f"\r{' ' * self._last_len}\r")
        sys.stderr.flush()
        self._active = False


def _sleep_with_spinner(interval: float, spinner: _LogFollowSpinner, state: str) -> None:
    if interval <= 0:
        return
    if not spinner.enabled:
        time.sleep(interval)
        return
    ticks = max(1, int(interval / _SPINNER_TICK_SECONDS))
    sleep_for = interval / ticks
    for _ in range(ticks):
        spinner.render(state)
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
    _ = save_credentials(api_key, api_url=api_url)
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


_STARTER_ENV_PY = '''\
"""Starter Freesolo environment.

Edit datasets/train.jsonl and the reward code, then upload with
`flash env push --name my-env .`.

A managed run should use the returned [environment] id from
`flash env push --name my-env .`.

This starter keeps a tiny smoke-test dataset in datasets/train.jsonl. Replace it
with your real training rows before a real run.
"""

from __future__ import annotations

import json
from pathlib import Path

from freesolo.datasets.types import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult


DEFAULT_DATASET_PATH = Path(__file__).parent / "datasets" / "train.jsonl"


def load_jsonl(path: str | Path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def exact_match_reward(example: TaskExample, response_text: str) -> RewardResult:
    expected = str(example.output or "").strip()
    score = 1.0 if expected and expected in response_text else 0.0
    return RewardResult(score=score, threshold=1.0)


class StarterEnv(EnvironmentSingleTurn):
    dataset = load_jsonl(DEFAULT_DATASET_PATH)

    def build_prompt_messages(self, example: TaskExample, prompt_text: str):
        return [{"role": "user", "content": example.input}]

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        return exact_match_reward(example, response_text)


def load_environment(dataset_path: str | None = None, **kwargs) -> StarterEnv:
    env = StarterEnv()
    if dataset_path:
        env.dataset = load_jsonl(dataset_path)
    return env
'''

_STARTER_DATASET_JSONL = """\
{"input":"What is 2 + 2?","output":"4"}
{"input":"What is 3 + 5?","output":"8"}
"""


def cmd_env_setup(args) -> int:
    Path("configs").mkdir(exist_ok=True)
    Path("datasets").mkdir(exist_ok=True)
    dataset = Path("datasets/train.jsonl")
    if not dataset.exists():
        dataset.write_text(_STARTER_DATASET_JSONL)
    starter_env = Path("environment.py")
    if not starter_env.exists():
        starter_env.write_text(_STARTER_ENV_PY)
    env_comment = (
        "# Environment: upload this project folder with\n"
        "# `flash env push --name my-env .`, then paste the returned id below.\n"
        "# If the environment reads secrets with os.environ, list only the env var names here.\n"
        "# Values are read from your shell or .env at submit time and are not stored in the spec.\n"
        "[environment]\n"
        'id = ""\n\n'
        '# secrets = ["SERPAPI_API_KEY"]\n\n'
    )
    rl = Path("configs/rl.toml")
    if not rl.exists():
        rl.write_text(
            'model = "Qwen/Qwen3.5-4B"\n'
            'algorithm = "grpo"\n\n'
            f"{env_comment}"
            "[train]\n"
            "steps = 150\n"
            "lora_rank = 32\n"
            "# GPU and HF artifacts are managed automatically by the platform: the GPU is\n"
            "# the cheapest fitting managed class, and artifacts live in a private environment-scoped repo.\n"
        )
    sft = Path("configs/sft.toml")
    if not sft.exists():
        sft.write_text(
            'model = "Qwen/Qwen3.5-4B"\n'
            'algorithm = "sft"\n\n'
            f"{env_comment}"
            "[train]\n"
            "epochs = 1\n"
            "lora_rank = 32\n"
            "# GPU and HF artifacts are managed automatically by the platform: the GPU is\n"
            "# the cheapest fitting managed class, and artifacts live in a private environment-scoped repo.\n"
        )
    training = Path("TRAINING.md")
    if not training.exists():
        # Explicit UTF-8: TRAINING_MD has non-ASCII chars that raise UnicodeEncodeError under a non-UTF-8 locale.
        training.write_text(TRAINING_MD, encoding="utf-8")
    scaffolded = [
        "environment.py",
        "datasets/train.jsonl",
        "configs/rl.toml",
        "configs/sft.toml",
        "TRAINING.md",
    ]
    if render.styled():
        print(render.env_setup(scaffolded))
        return 0
    print(f"ensured {', '.join(scaffolded)}")
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

    Catalog-only and deterministic; SFT uses the actual local training-token count when the env
    and tokenizer are importable. An uncapped SFT run must be able to count the env's train split,
    otherwise it errors instead of guessing a dataset size."""
    from flash.cost import estimate_cost

    spec = spec_from_file(
        args.config,
        run_id=None,
        overrides=args.overrides,
        extra_configs=args.extra_configs,
    )
    estimate = estimate_cost(runconfig_from_spec(spec))
    if render.styled():
        print(render.cost_panel(estimate))
    else:
        print(estimate.breakdown())
    return 0


def cmd_train(args) -> int:
    if getattr(args, "cost", False):
        return _cmd_train_cost(args)
    spec = spec_from_file(
        args.config,
        run_id=new_run_id() if args.dry_run else None,
        overrides=args.overrides,
        extra_configs=args.extra_configs,
    )
    if args.dry_run:
        payload = {"run_id": spec.run_id, "state": "dry_run", "spec": spec.to_dict()}
        if render.styled():
            print(
                render.object_panel("train", payload, "dry run — validated locally, not submitted")
            )
        else:
            print(json.dumps(payload, indent=2))
        return 0
    client = client_from_config()
    status = client.create_run(
        spec_payload(spec),
        runtime_secrets=runtime_secrets_from_local_env(args.config, keys=spec.environment.secrets),
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


def _poll_logs(client: ApiClient, run_id: str, interval: float) -> str:
    """Stream offset-paged logs until the run reaches a terminal state; return that state."""
    offset = 0
    spinner = _LogFollowSpinner(run_id)
    try:
        while True:
            page = client.get_logs(run_id, offset=offset)
            if page["logs"]:
                spinner.clear()
                print(page["logs"], end="", flush=True)
            offset = page["offset"]
            if page["state"] in _CLI_DONE_STATES:
                spinner.clear()
                return page["state"]
            _sleep_with_spinner(interval, spinner, page["state"])
    finally:
        spinner.clear()


def _follow_run(client: ApiClient, run_id: str) -> int:
    """Poll logs until the run reaches a terminal state, then print the final status."""
    state = _poll_logs(client, run_id, interval=2.0)
    status = client.get_run(run_id)
    if render.styled():
        print(render.run_status(status))
    else:
        print(json.dumps(status, indent=2))
    return 0 if state in _OK_STATES else 1


def _follow_status(client: ApiClient, run_id: str, interval: float = 2.0) -> int:
    """Poll run status until terminal, without replaying worker logs."""
    last_rendered: str | None = None
    while True:
        status = client.get_run(run_id)
        rendered = render.run_status(status) if render.styled() else json.dumps(status, indent=2)
        if rendered != last_rendered:
            print(rendered)
            last_rendered = rendered
        state = str(status.get("state") or "")
        if state in _CLI_DONE_STATES:
            return 0 if state in _OK_STATES else 1
        time.sleep(interval)


def _print_log_text(text: str) -> bool:
    if not text:
        return False
    print(text, end="")
    if not text.endswith("\n"):
        print()
    return True


def _print_log_snapshot(client: ApiClient, run_id: str) -> bool:
    """Print a finite snapshot from the log endpoint without following new bytes."""
    page = client.get_logs(run_id, offset=0)
    return _print_log_text(str(page.get("logs") or ""))


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
        state = _poll_logs(client, args.run_id, interval=2.0)
        _print_worker_output(client, args.run_id, printed_any=True)
        return 0 if state in _OK_STATES else 1
    printed_any = _print_log_snapshot(client, args.run_id)
    _print_worker_output(client, args.run_id, printed_any=printed_any)
    return 0


def cmd_status(args) -> int:
    client = client_from_config()
    if getattr(args, "follow", False):
        return _follow_status(client, args.run_id)
    status = client.get_run(args.run_id)
    if render.styled():
        print(render.run_status(status))
    else:
        print(json.dumps(status, indent=2))
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
        where = render._run_gpu(spec, r.get("remote") or {})
        print(
            f"{r['run_id']:<32}  {r['state']:<11}  {algorithm:<5}  "
            f"{r.get('cost_usd', 0.0):>8.4f}  {where:<22}  {model}"
        )
    return 0


def cmd_cancel(args) -> int:
    status = client_from_config().cancel_run(args.run_id)
    payload = {"run_id": args.run_id, "state": status["state"]}
    if render.styled():
        print(render.cancelled(payload))
    else:
        print(json.dumps(payload, indent=2))
    return 0


def cmd_checkpoints(args) -> int:
    checkpoints = client_from_config().checkpoints(args.run_id)
    if not checkpoints:
        message = (
            f"no deployable checkpoints for {args.run_id} yet "
            "(RL streams one per save interval; SFT-only runs have none)."
        )
        if render.styled():
            print(render.empty("checkpoints", "0 deployable", message))
        else:
            print(message, file=sys.stderr)
        return 0
    if render.styled():
        print(render.checkpoints_table(args.run_id, checkpoints))
        return 0
    for c in checkpoints:
        print(f"step {c['step']:>6}  {c['repo_id']}:{c['subfolder']}")
    print(
        f"\ndeploy one with `flash deploy {args.run_id} --step <STEP>`.",
        file=sys.stderr,
    )
    return 0


def cmd_deploy(args) -> int:
    dep = client_from_config().deploy(
        args.run_id,
        dry_run=args.dry_run,
        step=getattr(args, "step", None),
    )
    if render.styled():
        print(render.deployed(dep))
    else:
        print(json.dumps(dep, indent=2))
    # a dry run creates no deployment, so the billing / undeploy hint would be misleading.
    if dep.get("state") != "dry_run":
        note = (
            f"serving is billed per token only; use `flash undeploy {args.run_id}` "
            "to deregister the adapter."
        )
        print(render.arrow(note) if render.styled() else f"note: {note}", file=sys.stderr)
    return 0


def cmd_export(args) -> int:
    from flash.client.runtime_secrets import resolve_hf_token

    hf_token = resolve_hf_token(args.api_key)
    if not hf_token:
        raise ClientError(
            "no HuggingFace token: pass `--api-key <hf_...>`, or set HF_TOKEN "
            "(export it in your shell or put it in a local .env / .env.local)"
        )
    client = client_from_config()
    where = f" (step {args.step})" if args.step is not None else ""
    progress = (
        f"exporting adapter {args.adapter_id}{where} to {args.repository} — "
        "downloading then re-uploading; this can take a minute..."
    )
    print(render.note(progress) if render.styled() else progress, file=sys.stderr)
    result = client.export(
        args.adapter_id,
        repository=args.repository,
        hf_token=hf_token,
        step=args.step,
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
    if not rows:
        if render.styled():
            print(render.empty("deployments", "0 active", "no active deployments"))
        else:
            print("no active deployments")
        return 0
    if render.styled():
        print(render.deployments_table(rows))
        return 0
    print(f"{'RUN_ID':<32}  {'GPU':<9}  ENDPOINT")
    for r in rows:
        d = r.get("deployment") or {}
        print(f"{r['run_id']:<32}  {d.get('gpu', '?'):<9}  {d.get('endpoint_name', '')}")
    return 0


def cmd_chat(args) -> int:
    client = client_from_config()
    messages = [{"role": "user", "content": args.message}]
    if render.styled():
        print(render.chat_label())
    stream = getattr(client, "chat_stream", None)
    if stream is not None:
        wrote = False
        for chunk in stream(
            args.run_id,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ):
            print(chunk, end="", flush=True)
            wrote = True
        if wrote:
            print()
        return 0

    resp = client.chat(
        args.run_id,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(resp["choices"][0]["message"]["content"])
    return 0
